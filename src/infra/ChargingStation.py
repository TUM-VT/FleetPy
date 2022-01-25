# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
from __future__ import annotations
import logging
import typing as tp
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict, List, Optional, Tuple
from operator import attrgetter, itemgetter

# additional module imports (> requirements)
# ------------------------------------------

# src imports
# -----------
from pyproj import Transformer

from src.fleetctrl.FleetControlBase import FleetControlBase
from src.misc.globals import G_PUBLIC_CHARGING_FILE, G_DIR_OUTPUT, G_CHARGING_STATION_SEARCH_RADIUS
from src.routing.NetworkBase import NetworkBase
from src.simulation.Vehicles import SimulationVehicle
from src.simulation.StationaryProcess import ChargingProcess

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
LOG = logging.getLogger(__name__)

MAX_CHARGING_SEARCH = 100
UTM_CRS = "+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
STATION_ID_COL = "station_id"
STATION_SOCKET_COL = "socket nr"
STATION_SOCKET_POWER_COL = "max_socket_power"
STATION_NETWORK_NODE_COL = "nearest_node"
STATION_NODE_DISTANCE_COL = "nearest_node_distance"


class ChargingSocket:
    """ This class represents a single charging socket """

    def __init__(self, socket_id, max_socket_power):
        self.id = socket_id
        self.max_socket_power = max_socket_power
        self.attached_vehicle: Optional[SimulationVehicle] = None
        self.initial_soc = None
        self.connect_time = None

    def __repr__(self):
        vehicle = None if self.attached_vehicle is None else self.attached_vehicle.vid
        return f"socket id: {self.id}, attached vehicle {vehicle}"

    @property
    def transferred_power(self):
        power = 0
        if self.attached_vehicle is not None:
            power = (self.attached_vehicle.soc - self.initial_soc) * self.attached_vehicle.battery_size
        return power

    def attach(self, sim_time, vehicle: SimulationVehicle):
        """ Attaches vehicle to the socket

        :param sim_time:    current simulation time
        :param vehicle:     simulation vehicle
        :return:            True if the socket was empty and the vehicle attaches successfully
        """

        is_attached = False
        if self.attached_vehicle is None:
            self.attached_vehicle = vehicle
            self.initial_soc = vehicle.soc
            self.connect_time = sim_time
            is_attached = True
        return is_attached

    def detach(self):
        assert self.attached_vehicle is not None, f"no vehicle to detach from socket {self.id}"
        self.attached_vehicle = None
        self.initial_soc = None
        self.connect_time = None

    def charge_vehicle(self, delta_time):
        """ Linear charging of the attached vehicle

        :param delta_time: the time increment in seconds
        """
        delta_power = self.max_socket_power * delta_time / 3600
        delta_soc = delta_power / self.attached_vehicle.battery_size
        self.attached_vehicle.soc = min(1.0, self.attached_vehicle.soc + delta_soc)

    def calculate_remaining_time_of_attached_vehicle(self, end_soc=1.0) -> float:
        """ Returns the remaining time in seconds required to charge the attached vehicle

        :param end_soc: the final soc for which the remaining charging duration is required. Default is full soc (1.0)
        """

        return self.calculate_charge_duration(self.attached_vehicle, self.attached_vehicle.soc, end_soc)

    def calculate_charge_duration(self, veh_object: SimulationVehicle, start_soc, end_soc) -> float:
        """ Calculates the charging duration in seconds required to charge the provided vehicle

        :param veh_object:      the vehicle object for which charging duration is required
        :param start_soc:       the starting SOC of the vehicle.
        :param end_soc:         The final soc upto which the vehicle should be charged.
        """
        start_soc = veh_object.soc if start_soc is None else start_soc
        remaining_battery = (end_soc - start_soc) * self.attached_vehicle.battery_size
        return remaining_battery / self.max_socket_power * 3600


class ChargingStation:
    """ This class represents a public charging station with multiple sockets """
    station_history = defaultdict(list)
    station_history_file_path = None

    def __init__(self, station_id, node, socket_ids, max_socket_powers: List[float]):
        self.id = station_id
        self.pos = (node, None, None)
        self._sockets: Dict[int, ChargingSocket] = {id: ChargingSocket(id, max_power)
                                                       for id, max_power in zip(socket_ids, max_socket_powers)}
        self._sorted_sockets = sorted(self._sockets.values(), key=attrgetter("max_socket_power"), reverse=True)
        self._vid_socket_dict: Dict[int, ChargingSocket] = {}

        # Dictionary for the charging schedule with booking ids as keys
        self._booked_processes: Dict[str, ChargingProcess] = {}
        self._socket_bookings: Dict[int, List[ChargingProcess]] = {ID: [] for ID in self._sockets}
        self._current_processes: Dict[int, Optional[ChargingProcess]] = {ID: None for ID in self._sockets}

    def __add_to_scheduled(self, booking: ChargingProcess):
        self._booked_processes[booking.id] = booking
        self._socket_bookings[booking.socket_id].append(booking)

    def __remove_from_scheduled(self, booking: ChargingProcess):
        del self._booked_processes[booking.id]
        self._socket_bookings[booking.socket_id].remove(booking)

    def calculate_charge_durations(self, veh_object: SimulationVehicle, start_soc=None,
                                   end_soc=1.0) -> Dict[int, float]:
        """ Calculates the charging duration in seconds required to charge the given vehicle

        :param veh_object:      the vehicle object for which charging duration is required
        :param start_soc:       the starting SOC of the vehicle. By default the start_soc is derived from current soc
                                of the given veh_object
        :param end_soc:         The final soc upto which the vehicle should be charged. The default is full soc (1.0)
        :returns:   Dict with socket ids as keys and charging duration as values
        """

        start_soc = veh_object.soc if start_soc is None else start_soc
        return {socket.id: socket.calculate_charge_duration(veh_object, start_soc, end_soc)
                for socket in self._sorted_sockets}

    def get_running_processes(self):
        """ Returns the currently running and scheduled charging processes

        :returns: - Dict of currently running charging processes with socket_id as key
                  - Dict of scheduled charging processes with socket_id as key
        """
        return self._current_processes.copy(), self._socket_bookings.copy()

    def get_current_schedules(self, sim_time) -> Dict[int, List[Tuple[float, float, str]]]:
        """ Returns the time slots already booked/occupied for each of the charging sockets

        :param sim_time: current simulation time
        :returns:   - Dict with socket_id as keys and list of (start time, end time, booking id) as values
        """

        schedules = {socket_id: [] for socket_id in self._sockets}
        for socket_id, current_booking in self._current_processes.items():
            if current_booking is not None:
                end_time = sim_time + current_booking.remaining_duration_to_finish(sim_time)
                schedules[socket_id].append((current_booking.start_time, end_time, current_booking.id))
        for socket_id, bookings in self._socket_bookings.items():
            for scheduled_booking in bookings:
                schedules[socket_id].append((scheduled_booking.start_time, scheduled_booking.end_time,
                                             scheduled_booking.id))
            schedules[socket_id] = sorted(schedules[socket_id], key=itemgetter(0))
        return schedules

    def start_charging_process(self, sim_time, booking: ChargingProcess) -> bool:
        """ Starts the provided charging process by connecting the vehicle to the socket

        :param sim_time:    Current simulation time
        :param booking:     The charging process instance to be started
        """

        socket = self._sockets[booking.socket_id]
        found_empty_socket = socket.attach(sim_time, booking.veh)
        assert found_empty_socket is True, f"unable to connect to the socket {socket} at station {self.id}"
        self._vid_socket_dict[booking.veh.vid] = socket
        self._current_processes[socket.id] = booking
        self.__remove_from_scheduled(booking)
        return found_empty_socket

    def end_charging_process(self, sim_time, booking: ChargingProcess):
        socket = self._sockets[booking.socket_id]
        assert socket.attached_vehicle.vid == booking.veh.vid
        self.__append_to_history(sim_time, booking.veh, "detach", socket)
        socket.detach()
        del self._vid_socket_dict[booking.veh.vid]
        self._current_processes[socket.id] = None

    def cancel_booking(self, sim_time, booking: ChargingProcess):
        if booking in self._current_processes.values():
            LOG.warning(f"canceling an already running charging process {booking.id} at station {self.id}")
            self.end_charging_process(sim_time, booking)
        else:
            self.__remove_from_scheduled(booking)

    def update_charging_state(self, booking: ChargingProcess, delta_time):
        """ Charges the vehicle according to delta_time """

        socket = self._sockets[booking.socket_id]
        assert socket.attached_vehicle.vid == booking.veh.vid, "the vehicle attached to the socket and the vehicle " \
                                                               "in the booking are not same"
        socket.charge_vehicle(delta_time)

    def remaining_charging_time(self, vid):
        assert vid in self._vid_socket_dict, f"vehicle {vid} not attached to any socket at station {self.id}, " \
                                             f"remaining time cannot be calculated"
        return self._vid_socket_dict[vid].calculate_remaining_time_of_attached_vehicle()

    def make_booking(self, sim_time, socket_id, vehicle: SimulationVehicle, start_time=None,
                     end_time=None) -> ChargingProcess:
        """ Makes a booking for the vehicle

        :param sim_time:    current simulation time
        :param socket_id:   id of the socket to be booked
        :param vehicle:     vehicle object
        :param start_time:  start time of the booking (default value is equal to sim_time)
        :param end_time:    end time of the charging. The default value (None) will charge the vehicle upto full soc
        """

        start_time = sim_time if start_time is None else start_time
        booking_id = f"{socket_id}_{vehicle.vid}_{start_time}"
        booking = ChargingProcess(booking_id, vehicle, self, start_time, end_time)
        self.__add_to_scheduled(booking)
        return booking

    def modify_booking(self, sim_time, booking: ChargingProcess):
        raise NotImplemented

    def __append_to_history(self, sim_time, vehicle, event_name, socket=None):
        if self.station_history_file_path is not None:
            self.station_history["time"].append(sim_time)
            self.station_history["event"].append(event_name)
            self.station_history["station_id"].append(self.id)
            self.station_history["vid"].append(vehicle.vid)
            self.station_history["veh_type"].append(vehicle.veh_type)
            self.station_history["current_soc"].append(round(vehicle.soc, 3))
            self.station_history["socket_id"].append("" if socket is None else socket.id)
            self.station_history["socket_power"].append("" if socket is None else socket.max_socket_power)
            self.station_history["initial_soc"].append("" if socket is None else round(socket.initial_soc, 3))
            self.station_history["transferred_power"].append("" if socket is None else socket.transferred_power)
            self.station_history["connection_duration"].append("" if socket is None else sim_time - socket.connect_time)
            if len(self.station_history) > 1000:
                self.write_history_to_file()

    def add_final_states_to_history(self, sim_time, vehicle):
        socket = self._vid_socket_dict[vehicle.vid]
        self.__append_to_history(sim_time, vehicle, "final_state", socket)

    @staticmethod
    def write_history_to_file():
        file = Path(ChargingStation.station_history_file_path)
        if len(ChargingStation.station_history) > 0:
            df = DataFrame(ChargingStation.station_history)
            df.to_csv(file, index=False, mode="a", header=not file.exists())
            ChargingStation.station_history = defaultdict(list)

    @staticmethod
    def set_history_file_path(path):
        ChargingStation.station_history_file_path = Path(path)


# TODO # keep track of parking (inactive) vehicles | query parking lots
class Depot(ChargingStation):
    """This class represents a charging station with parking lots for inactive vehicles."""
    pass


class ChargingInfrastructureOperator:

    # TODO # init with routing engine, charging operator attributes
    # TODO # load data in metric system
    # TODO # functionality for parking lots here | functionality for activating/deactivating in fleetctrl/fleetsizing
    # TODO # functionality for depot charging in fleetctrl/charging
    def __init__(self, fleetctrl, operator_attributes):
        """This class represents the operator for the charging infrastructure.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param solver: solver for optimization problems
        """

        self.fleetctrl: FleetControlBase = fleetctrl
        self.routing_engine: NetworkBase = fleetctrl.routing_engine
        self.operator_attributes = operator_attributes
        self.charging_stations: tp.List[ChargingStation] = self._loading_charging_stations()
        self.station_by_id: tp.Dict[int, ChargingStation] = {station.id: station for station in self.charging_stations}

        # Calculate and save the utm coordinates of the charging stations for calculating euclidean distances
        stations_lonlat = self.routing_engine.return_positions_lon_lat([x.pos for x in self.charging_stations])
        proj_transformer = Transformer.from_proj('epsg:4326', UTM_CRS)
        x_utm, y_utm = proj_transformer.transform([x[1] for x in stations_lonlat], [x[0] for x in stations_lonlat])
        self.__charging_stations_utm = np.array(list(zip(x_utm, y_utm)))


    def _loading_charging_stations(self):
        """ Loads the charging stations from the provided csv file"""

        dir_names = self.fleetctrl.dir_names
        stations_file = self.operator_attributes.get(G_PUBLIC_CHARGING_FILE, None)
        assert stations_file is not None, f"Public charging stations file must exist in folder demand -> (network name) " \
                                          f"-> public and provided via config parameter {G_PUBLIC_CHARGING_FILE}"
        stations_file = Path(Path(dir_names["demand"]).parent.parent, "public", stations_file)
        stations_df = pd.read_csv(stations_file)
        stations_dict = dict(stations_df.groupby(STATION_ID_COL).apply(lambda x: x.to_dict(orient='list')))
        file = Path(dir_names[G_DIR_OUTPUT]).joinpath("4-charging_stats.csv")
        if file.exists():
            file.unlink()
        ChargingStation.set_history_file_path(file)
        stations = [ChargingStation(info[STATION_ID_COL][0],
                                    info[STATION_NETWORK_NODE_COL][0],
                                    info[STATION_SOCKET_COL],
                                    info[STATION_SOCKET_POWER_COL]) for info in stations_dict.values()]
        return stations

    def modify_booking(self, sim_time, booking: ChargingProcess):
        booking.station.modify_booking(sim_time, booking)

    def cancel_booking(self, sim_time, booking: ChargingProcess):
        booking.station.cancel_booking(sim_time, booking)

    def book_station(self, sim_time, vehicle: SimulationVehicle, station_id, socket_id, start_time=None,
                     end_time=None) -> ChargingProcess:
        """ Books a socket at the charging station """

        station = self.station_by_id[station_id]
        return station.make_booking(sim_time, socket_id, vehicle, start_time, end_time)

    def get_charging_slots(self, sim_time, vehicle: SimulationVehicle, planned_start_time, planned_veh_pos,
                           planned_veh_soc, desired_veh_soc, max_number_charging_stations=1, max_sockets_per_station=1) \
            -> tp.List[tp.Tuple[int, int, int, int, float]]:
        """ Returns specific charging possibilities for a vehicle at a future time, place with estimated SOC and desired SOC.

        :param sim_time: current simulation time
        :param vehicle: the vehicle for which the charging slot is required
        :param planned_start_time: earliest time at which charging should be considered
        :param planned_veh_pos: time from which vehicle will drive to charging station
        :param planned_veh_soc: estimated vehicle SOC at that position
        :param desired_veh_soc: desired final SOC after charging
        :param max_number_charging_stations: maximum number of charging stations to consider
        :param max_sockets_per_station: maximum number of sockets per charging station to consider
        :return: list of specific offers of ChargingOperator consisting of
                    (charging station id, charging socket id, booking start time, booking end time, expected booking end soc)
        """
        considered_stations = self._get_considered_stations(planned_veh_pos)
        list_offers = []
        for station in considered_stations:
            list_station_offers = []
            charge_durations = station.calculate_charge_durations(vehicle, planned_veh_soc, desired_veh_soc)
            estimated_arrival_time = planned_start_time + self.routing_engine.return_travel_costs_1to1(planned_veh_pos, station.pos)
            for socket_id, socket_schedule in station.get_current_schedules(sim_time).items():
                socket_charge_duration = charge_durations[socket_id]
                possible_start_time = estimated_arrival_time
                found_free_slot = False
                for planned_booking in socket_schedule:
                    # check whether charging process can be finished before next booking
                    possible_end_time = possible_start_time + socket_charge_duration
                    pb_start_time, pb_et, pb_id = planned_booking
                    if possible_end_time <= pb_start_time:
                        list_station_offers.append((station.id, socket_id, possible_start_time, possible_end_time, desired_veh_soc))
                        found_free_slot = True
                        break
                    possible_start_time = pb_et
                if not found_free_slot:
                    possible_end_time = possible_start_time + socket_charge_duration
                    list_station_offers.append((station.id, socket_id, possible_start_time, possible_end_time, desired_veh_soc))
                # TODO # check methodology to stop
                if len(list_station_offers) == max_sockets_per_station:
                    if possible_start_time == estimated_arrival_time:
                        break
                    else:
                        # only keep offer with earliest start
                        list_station_offers = sorted(list_offers, key=lambda x: x[2])[:1]
            list_offers.extend(list_station_offers)
            # TODO # check methodology to stop
            if len(list_offers) >= max_number_charging_stations:
                break
        return list_offers

    def _get_considered_stations(self, position):
        """ Returns the list of stations nearest stations (using euclidean distance) within search radius of the
        position.

        :param position: position around which the station is sought
        :returns:   List of charging station in order of proximity to the provided position
        """

        # TODO # think about using routing 1-to-X functionality to search charging stations -> information could be used to calculate estimated_arrival_time
        veh_lon, veh_lat = self.routing_engine.return_positions_lon_lat([position])[0]
        proj_transformer = Transformer.from_proj('epsg:4326', UTM_CRS)
        veh_x_utm = np.array(proj_transformer.transform(veh_lat, veh_lon))
        euclidean = np.linalg.norm(veh_x_utm - self.__charging_stations_utm, axis=1)
        closest_inx = np.argpartition(euclidean, MAX_CHARGING_SEARCH)[:MAX_CHARGING_SEARCH]
        distances = euclidean[closest_inx]
        station_inx = closest_inx[distances <= self.operator_attributes[G_CHARGING_STATION_SEARCH_RADIUS]]
        return [self.charging_stations[x] for x in station_inx]





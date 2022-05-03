# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
from __future__ import annotations
import logging
import typing as tp
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from operator import attrgetter, itemgetter

# additional module imports (> requirements)
# ------------------------------------------

# src imports
# -----------
from pyproj import Transformer

from src.misc.globals import *
from src.simulation.StationaryProcess import ChargingProcess
from src.fleetctrl.planning.VehiclePlan import ChargingPlanStop, VehiclePlan, RoutingTargetPlanStop
from src.misc.config import decode_config_str
if TYPE_CHECKING:
    from src.routing.NetworkBase import NetworkBase
    from src.simulation.Vehicles import SimulationVehicle
    from src.fleetctrl.FleetControlBase import FleetControlBase

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
LOG = logging.getLogger(__name__)

MAX_CHARGING_SEARCH = 100   # TODO # in globals?
LARGE_INT = 100000000

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

    def calculate_charge_duration(self, veh_object: SimulationVehicle, start_soc, end_soc) -> float:
        """ Calculates the charging duration in seconds required to charge the provided vehicle

        :param veh_object:      the vehicle object for which charging duration is required
        :param start_soc:       the starting SOC of the vehicle.
        :param end_soc:         The final soc upto which the vehicle should be charged.
        """
        start_soc = veh_object.soc if start_soc is None else start_soc
        remaining_battery = (end_soc - start_soc) * veh_object.battery_size
        return remaining_battery / self.max_socket_power * 3600


class ChargingStation:
    """ This class represents a public charging station with multiple sockets """
    station_history = defaultdict(list)
    station_history_file_path = None

    def __init__(self, station_id, ch_op_id, node, socket_ids, max_socket_powers: List[float]):
        self.id = station_id
        self.ch_op_id = ch_op_id
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
        LOG.debug(f"start charging process: {booking} at time {sim_time}")
        LOG.debug(f"with schedule {self.get_current_schedules(sim_time)}")
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

    def remaining_charging_time(self, sim_time, vid, end_soc=1.0) -> float:
        """ Returns the remaining time in seconds required to charge the attached vehicle
        it is either charged until end_soc or until the next process is scheduled
        :param sim_time: current simulation time
        :param vid: vehicle id
        :param end_soc: the final soc for which the remaining charging duration is required. Default is full soc (1.0)
        """
        assert vid in self._vid_socket_dict, f"vehicle {vid} not attached to any socket at station {self.id}, " \
                                             f"remaining time cannot be calculated"
        attached_socked = self._vid_socket_dict[vid]
        full_charge_duration = attached_socked.calculate_charge_duration(attached_socked.attached_vehicle, attached_socked.attached_vehicle.soc, end_soc)
        LOG.debug(" -> full charge duration at time {}: {}".format(sim_time, full_charge_duration))
        # test for next bookings
        if len(self._socket_bookings.get(attached_socked.id, [])) > 0:
            next_start_time = min(self._socket_bookings[attached_socked.id], key=lambda x:x.start_time).start_time
            assert next_start_time - sim_time > 0, f"uncancelled bookings in charging station {self.id} at time {sim_time} with schedule { [str(c) for c in self._socket_bookings[attached_socked.id] ]}!"
            if next_start_time - sim_time < full_charge_duration:
                return next_start_time - sim_time
        return full_charge_duration

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
        booking_id = f"{self.id}_{socket_id}_{vehicle.vid}_{int(start_time)}_{int(end_time)}"
        booking = ChargingProcess(booking_id, vehicle, self, start_time, end_time)
        self.__add_to_scheduled(booking)
        return booking

    def modify_booking(self, sim_time, booking: ChargingProcess):
        raise NotImplemented
    
    def get_charging_slots(self, sim_time, vehicle, planned_arrival_time, planned_start_soc, desired_end_soc, max_offers_per_station=1):
        """ Returns specific charging possibilities for a vehicle at this charging station
        a future time, place with estimated SOC and desired SOC.

        :param sim_time: current simulation time
        :param vehicle: the vehicle for which the charging slot is required
        :param planned_arrival_time: earliest time at which charging should be considered
        :param planned_start_soc: estimated vehicle SOC at that position
        :param desired_end_soc: desired final SOC after charging
        :param max_offers_per_station: maximum number of offers per charging station to consider
        :return: list of specific offers of ChargingOperator consisting of
                    (charging station id, charging socket id, booking start time, booking end time, expected booking end soc, max_charging_power)
        """

        list_station_offers = []
        charge_durations = self.calculate_charge_durations(vehicle, planned_start_soc, desired_end_soc)
        estimated_arrival_time = planned_arrival_time
        for socket_id, socket_schedule in self.get_current_schedules(sim_time).items():
            socket_charge_duration = charge_durations[socket_id]
            possible_start_time = estimated_arrival_time
            found_free_slot = False
            max_power = self._sockets[socket_id].max_socket_power
            for planned_booking in socket_schedule:
                # check whether charging process can be finished before next booking
                possible_end_time = possible_start_time + socket_charge_duration
                pb_start_time, pb_et, pb_id = planned_booking
                if possible_end_time <= pb_start_time:
                    list_station_offers.append((self.id, socket_id, possible_start_time, possible_end_time, desired_end_soc, max_power))
                    found_free_slot = True
                    break
                if pb_et > possible_start_time:
                    possible_start_time = pb_et
            if not found_free_slot:
                possible_end_time = possible_start_time + socket_charge_duration
                list_station_offers.append((self.id, socket_id, possible_start_time, possible_end_time, desired_end_soc, max_power))
                if possible_start_time == planned_arrival_time and len(list_station_offers) >= max_offers_per_station:
                    LOG.debug("early brake in offer search")
                    break
            LOG.debug(f"possible slots for station {self.id} at socket {socket_id}:")
            LOG.debug(f"    -> {list_station_offers}")
        # TODO # check methodology to stop
        if len(list_station_offers) > max_offers_per_station:
            # only keep offer with earliest start
            list_station_offers = sorted(list_station_offers, key=lambda x: x[2])[:max_offers_per_station]
        return list_station_offers
    
    def add_external_booking(self, start_time, end_time, sim_time, veh_struct):
        """ this methods adds an external booking to the charging station and therefor occupies a socket for a given time
        :param start_time: start time of booking
        :param end_time: end time of booking
        :param sim_time : current simulation time
        :return: True, if booking is possible; False if all sockets are already occupied"""
        found_socked = None
        for socket_id, socket_schedule in self.get_current_schedules(sim_time).items():
            if len(socket_schedule) == 0:
                found_socked = socket_id
                break
            else:
                last_end_time = start_time
                for pb_start_time, pb_end_time, _ in sorted(socket_schedule,key=lambda x:x[0]):
                    if pb_start_time >= end_time and last_end_time <= start_time:
                        found_socked = socket_id
                        last_end_time = pb_end_time
                        break
                    if pb_start_time >= end_time:
                        last_end_time = pb_end_time
                        break
                    last_end_time = pb_end_time
                if found_socked is not None:
                    break
                if last_end_time <= start_time:
                    found_socked = socket_id
                    break
        if found_socked is None:
            return False
        else:
            self.make_booking(sim_time, found_socked, veh_struct, start_time=start_time, end_time=end_time)
            return True

    def __append_to_history(self, sim_time, vehicle, event_name, socket=None):
        if self.station_history_file_path is not None:
            self.station_history["time"].append(sim_time)
            self.station_history["event"].append(event_name)
            self.station_history["station_id"].append(self.id)
            self.station_history["ch_op_id"].append(self.ch_op_id)
            self.station_history["vid"].append(vehicle.vid)
            self.station_history["op_id"].append(vehicle.op_id)
            self.station_history["veh_type"].append(vehicle.veh_type)
            self.station_history["current_soc"].append(round(vehicle.soc, 3))
            self.station_history["socket_id"].append("" if socket is None else socket.id)
            self.station_history["socket_power"].append("" if socket is None else socket.max_socket_power)
            self.station_history["initial_soc"].append("" if socket is None else round(socket.initial_soc, 3))
            self.station_history["transferred_power"].append("" if socket is None else socket.transferred_power)
            self.station_history["connection_duration"].append("" if socket is None else sim_time - socket.connect_time)
            if len(self.station_history) > 0:
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
    def __init__(self, station_id, ch_op_id, node, socket_ids, max_socket_powers: List[float], number_parking_spots):
        super().__init__(station_id, ch_op_id, node, socket_ids, max_socket_powers)
        self.number_parking_spots = number_parking_spots
        self.deactivated_vehicles: tp.List[SimulationVehicle] = []
        
    @property
    def free_parking_spots(self):
        return self.number_parking_spots - len(self.deactivated_vehicles)
    
    @property
    def parking_vehicles(self):
        return len(self.deactivated_vehicles)
        
    def schedule_inactive(self, veh_obj):
        """ adds the vehicle to park at the depot
        :param veh_obj: vehicle obj"""
        LOG.debug(f"park vid {veh_obj.vid} in depot {self.id} with parking vids {[x.vid for x in self.deactivated_vehicles]}")
        self.deactivated_vehicles.append(veh_obj)
        
    def schedule_active(self, veh_obj):
        """ removes the vehicle from the depot
        :param veh_obj: vehicle obj"""
        LOG.debug(f"activate vid {veh_obj.vid} in depot {self.id} with parking vids {[x.vid for x in self.deactivated_vehicles]}")
        self.deactivated_vehicles.remove(veh_obj)
        
    def pick_vehicle_to_be_active(self) -> SimulationVehicle:
        """ selects the vehicle with highest soc from the list of deactivated vehicles (does not activate the vehicle yet!)
        :return: simulation vehicle obj"""
        return max([veh for veh in self.deactivated_vehicles if veh.pos == self.pos], key = lambda x:x.soc)
    
    def refill_charging(self, fleetctrl: FleetControlBase, simulation_time, keep_free_for_short_term=0):
        """This method fills empty charging slots in a depot with the lowest SOC parking (status 5) vehicles.
        The vehicles receive a locked ChargingPlanStop , which will be followed by another inactive planstop.
        These will directly be assigned to the vehicle. The charging process is directly booked with a socked.

        :param fleetctrl: FleetControl class
        :param simulation_time: current simulation time
        :param keep_free_for_short_term: optional parameter in order to keep short-term charging capacity (Not implemented yet)
        :return: None
        """
        if keep_free_for_short_term != 0:
            raise NotImplementedError("keep free for short term is not implemented yet!")
        # check for vehicles that require charging
        list_consider_charging: List[SimulationVehicle] = []
        for veh_obj in self.deactivated_vehicles:
            if veh_obj.soc == 1.0 or veh_obj.status != VRL_STATES.OUT_OF_SERVICE:
                continue
            # check whether veh_obj already has vcl
            consider_charging = True
            for vrl in veh_obj.assigned_route:
                if vrl.status == VRL_STATES.CHARGING:
                    consider_charging = False
                    break
            if consider_charging:
                list_consider_charging.append(veh_obj)
        if not list_consider_charging:
            return

        for veh_obj in sorted(list_consider_charging, key = lambda x:x.soc):
            charging_options = self.get_charging_slots(simulation_time, veh_obj, simulation_time, veh_obj.soc, 1.0)
            if len(charging_options) > 0:
                selected_charging_option = min(charging_options, key=lambda x:x[3])
                ch_process = self.make_booking(simulation_time, selected_charging_option[1], veh_obj, start_time=selected_charging_option[2], end_time=selected_charging_option[3])
                start_time, end_time = ch_process.get_scheduled_start_end_times()
                charging_task_id = (self.ch_op_id, ch_process.id)
                ch_ps = ChargingPlanStop(self.pos, charging_task_id=charging_task_id, earliest_start_time=start_time, duration=end_time-start_time,
                                         charging_power=selected_charging_option[5], locked=True)
                
                assert fleetctrl.veh_plans[veh_obj.vid].list_plan_stops[-1].get_state() == G_PLANSTOP_STATES.INACTIVE
                if start_time == simulation_time:
                    LOG.debug(" -> start now")
                    # finish current status 5 task
                    veh_obj.end_current_leg(simulation_time)
                    # modify veh-plan: insert charging before list position -1
                    fleetctrl.veh_plans[veh_obj.vid].add_plan_stop(ch_ps, veh_obj, simulation_time,
                                                                    fleetctrl.routing_engine,
                                                                    return_copy=False, position=-1)
                    fleetctrl.lock_current_vehicle_plan(veh_obj.vid)
                    # assign vehicle plan
                    fleetctrl.assign_vehicle_plan(veh_obj, fleetctrl.veh_plans[veh_obj.vid], simulation_time, assigned_charging_task=(charging_task_id, ch_process))
                else:
                    LOG.debug(" -> start later")
                    # modify veh-plan:
                    # finish current inactive task
                    _, inactive_vrl = veh_obj.end_current_leg(simulation_time)
                    fleetctrl.receive_status_update(veh_obj.vid, simulation_time, [inactive_vrl])
                    # add new inactivate task with corresponding duration
                    inactive_ps_1 = RoutingTargetPlanStop(self.pos, locked=True, duration=start_time - simulation_time, planstop_state=G_PLANSTOP_STATES.INACTIVE)
                    # add inactivate task after charging
                    inactive_ps_2 = RoutingTargetPlanStop(self.pos, locked=True, duration=LARGE_INT, planstop_state=G_PLANSTOP_STATES.INACTIVE)
                    # new veh plan
                    new_veh_plan = VehiclePlan(veh_obj, simulation_time, fleetctrl.routing_engine, [inactive_ps_1, ch_ps, inactive_ps_2])
                    
                    fleetctrl.lock_current_vehicle_plan(veh_obj.vid)
                    # assign vehicle plan
                    fleetctrl.assign_vehicle_plan(veh_obj, new_veh_plan, simulation_time, assigned_charging_task=(charging_task_id, ch_process))

class PublicChargingInfrastructureOperator:

    def __init__(self, ch_op_id: int, public_charging_station_file: str, ch_operator_attributes: dict,
                 scenario_parameters: dict, dir_names: dict, routing_engine: NetworkBase, initial_charging_events_f: str = None):
        """This class represents the operator for the charging infrastructure.

        :param ch_op_id: id of charging operator
        :param public_charging_station_file: path to file where charging stations are loaded from
        :param ch_operator_attributes: dictionary that can contain additionally required parameters (parameter specific for charging operator)
        :param scenario_parameters: dictionary that contain global scenario parameters
        :param dir_names: dictionary that specifies the folder structure of the simulation
        :param routing_engine: reference to network class
        :param initial_charging_events_f: in this file charging events are specified that are booked at the beginning of the simulation
        """

        self.ch_op_id = ch_op_id
        self.routing_engine = routing_engine
        self.ch_operator_attributes = ch_operator_attributes
        self.charging_stations: tp.List[ChargingStation] = self._loading_charging_stations(public_charging_station_file, dir_names)
        self.station_by_id: tp.Dict[int, ChargingStation] = {station.id: station for station in self.charging_stations}
        self.pos_to_list_station_id: tp.Dict[tuple, tp.List[int]] = {}
        for station_id, station in self.station_by_id.items():
            try:
                self.pos_to_list_station_id[station.pos].append(station_id)
            except KeyError:
                self.pos_to_list_station_id[station.pos] = [station_id]
                
        self.max_search_radius = scenario_parameters.get(G_CH_OP_MAX_STATION_SEARCH_RADIUS)
        self.max_considered_stations = scenario_parameters.get(G_CH_OP_MAX_CHARGING_SEARCH, 100)
        
        sim_start_time = scenario_parameters[G_SIM_START_TIME]
        sim_end_time = scenario_parameters[G_SIM_END_TIME]
        
        self.sim_time_step = scenario_parameters[G_SIM_TIME_STEP]
        
        if initial_charging_events_f is not None:
            class VehicleStruct():
                def __init__(self) -> None:
                    self.vid = -1
            
            charging_events = pd.read_csv(initial_charging_events_f)
            for station_id, start_time, end_time in zip(charging_events["charging_station_id"].values, charging_events["start_time"].values, charging_events["end_time"].values):
                if end_time < sim_start_time or start_time > sim_end_time:
                    continue
                self.station_by_id[station_id].add_external_booking(start_time, end_time, sim_start_time, VehicleStruct())
                

    def _loading_charging_stations(self, public_charging_station_file, dir_names) -> List[ChargingStation]:
        """ Loads the charging stations from the provided csv file"""
        stations_df = pd.read_csv(public_charging_station_file)
        file = Path(dir_names[G_DIR_OUTPUT]).joinpath("4_charging_stats.csv")
        if file.exists():
            file.unlink()
        ChargingStation.set_history_file_path(file)
        stations = []
        for _, row in stations_df.iterrows():
            station_id = row[G_INFRA_CS_ID]
            node_index = row[G_NODE_ID]
            cunit_dict = decode_config_str(row[G_INFRA_CU_DEF])
            if cunit_dict is None:
                cunit_dict = {}
            socked_ids = [i for i in range(sum(cunit_dict.values()))]
            socked_powers = []
            for power, number in cunit_dict.items():
                socked_powers += [power for _ in range(number)]
            stations.append(ChargingStation(station_id, self.ch_op_id, node_index, socked_ids, socked_powers))
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
                           planned_veh_soc, desired_veh_soc, max_number_charging_stations=1, max_offers_per_station=1) \
            -> tp.List[tp.Tuple[int, int, int, int, float, float, float]]:
        """ Returns specific charging possibilities for a vehicle at a future time, place with estimated SOC and desired SOC.

        :param sim_time: current simulation time
        :param vehicle: the vehicle for which the charging slot is required
        :param planned_start_time: earliest time at which charging should be considered
        :param planned_veh_pos: time from which vehicle will drive to charging station
        :param planned_veh_soc: estimated vehicle SOC at that position
        :param desired_veh_soc: desired final SOC after charging
        :param max_number_charging_stations: maximum number of charging stations to consider
        :param max_offers_per_station: maximum number of offers per charging station to consider
        :return: list of specific offers of ChargingOperator consisting of
                    (charging station id, charging socket id, booking start time, booking end time, expected booking end soc, max charging power)
        """
        considered_station_list = self._get_considered_stations(planned_veh_pos)
        list_offers = []
        c = 0
        for station_id, tt, dis in considered_station_list:
            station = self.station_by_id[station_id]
            estimated_arrival_time = planned_start_time + tt
            list_station_offers = station.get_charging_slots(sim_time, vehicle, estimated_arrival_time, planned_veh_soc, desired_veh_soc, max_offers_per_station=max_offers_per_station)
            LOG.debug(f"possible charge offers from station {station_id} for veh {vehicle.vid} with tt {tt} : {planned_start_time} -> {estimated_arrival_time}")
            LOG.debug(f"{list_station_offers}")
            list_offers.extend(list_station_offers)
            # TODO # check methodology to stop
            if len(list_station_offers) > 0:
                c += 1
                if c == max_number_charging_stations:
                    break
        return list_offers

    def _get_considered_stations(self, position: tuple) -> tp.List[tuple]:
        """ Returns the list of stations nearest stations (using euclidean distance) within search radius of the
        position.

        :param position: position around which the station is sought
        :returns:   List of tuple charging station id, travel time from position, travel distanc from position
                        in order of proximity to the provided position
        """
        r_list = self.routing_engine.return_travel_costs_1toX(position, self.pos_to_list_station_id.keys(),
                                                        max_routes=self.max_considered_stations, max_cost_value=self.max_search_radius)
        r = []
        c = 0
        for d_pos, _, tt, dis in r_list:
            for station_id in self.pos_to_list_station_id[d_pos]:
                r.append( (station_id, tt, dis) )
                c += 1
                if c == self.max_considered_stations:
                    return r
        return r
    
    def _remove_unrealized_bookings(self, sim_time):
        """ this method removes all planned bookings that are not ended by the update of a simulation vehicle and are there considered as not realized
        :param sim_time: simulation time"""
        for s_id, charging_station in self.station_by_id.items():
            schedule_dict = charging_station.get_current_schedules(sim_time)
            running_processes = charging_station.get_running_processes()[0]
            for socket_id, schedule in schedule_dict.items():
                if len(schedule) > 0:
                    start_time, end_time, booking_id = min(schedule, key=lambda x:x[1])
                    end_booking_flag = False
                    if end_time <= sim_time:
                        end_booking_flag = True
                    if end_time - start_time > self.sim_time_step and end_time <= sim_time + self.sim_time_step:
                        end_booking_flag = True
                    if end_booking_flag:
                        if running_processes.get(socket_id) is None or running_processes.get(socket_id).id != booking_id:
                            LOG.debug("end unrealized booking at time {} at station {} socket {}: {}".format(sim_time, s_id, socket_id, booking_id))
                            try:
                                ch_process = charging_station._booked_processes[booking_id]
                                charging_station.cancel_booking(sim_time, ch_process)
                            except KeyError:
                                LOG.warning("couldnt cancel charging booking {}".format(booking_id))
                        
    
    def time_trigger(self, sim_time):
        """ this method is triggered in each simulation time step
        :param sim_time: simulation time"""
        t = time.time()
        self._remove_unrealized_bookings(sim_time)
        LOG.debug("charging infra time trigger took {}".format(time.time() - t))
    

class OperatorChargingAndDepotInfrastructure(PublicChargingInfrastructureOperator):
    """ this class has similar functionality like a ChargingInfrastructureOperator but is unique for each MoD operator (only the corresponding operator
    has access to the charging stations """
    # TODO # functionality for parking lots here | functionality for activating/deactivating in fleetctrl/fleetsizing
    # TODO # functionality for depot charging in fleetctrl/charging
    def __init__(self, op_id: int, depot_file: str, operator_attributes: dict,
                 scenario_parameters: dict, dir_names: dict, routing_engine: NetworkBase):
        """This class represents the operator for the charging infrastructure.

        :param op_id: id of mod operator this depot class belongs to
        :param depot_file: path to file where charging stations an depots are loaded from
        :param operator_attributes: dictionary that can contain additionally required parameters (parameter specific for the mod operator)
        :param scenario_parameters: dictionary that contain global scenario parameters
        :param dir_names: dictionary that specifies the folder structure of the simulation
        :param routing_engine: reference to network class
        """
        super().__init__(f"op_{op_id}", depot_file, operator_attributes, scenario_parameters, dir_names, routing_engine)
        self.depot_by_id: tp.Dict[int, Depot] = {depot_id : depot for depot_id, depot in self.station_by_id.items() if depot.number_parking_spots > 0}
        
    def _loading_charging_stations(self, depot_file, dir_names) -> List[Depot]:
        """ Loads the charging stations from the provided csv file"""
        stations_df = pd.read_csv(depot_file)
        file = Path(dir_names[G_DIR_OUTPUT]).joinpath("4_charging_stats.csv")
        if file.exists():
            file.unlink()
        ChargingStation.set_history_file_path(file)
        stations = []
        for _, row in stations_df.iterrows():
            station_id = row[G_INFRA_CS_ID]
            node_index = row[G_NODE_ID]
            cunit_dict = decode_config_str(row[G_INFRA_CU_DEF])
            number_parking_spots = row[G_INFRA_MAX_PARK]
            if cunit_dict is None:
                cunit_dict = {}
            socked_ids = [i for i in range(sum(cunit_dict.values()))]
            socked_powers = []
            for power, number in cunit_dict.items():
                socked_powers += [power for _ in range(number)]
            stations.append(Depot(station_id, self.ch_op_id, node_index, socked_ids, socked_powers, number_parking_spots))
        return stations
    
    def find_nearest_free_depot(self, pos, check_free=True) -> Depot:
        """This method can be used to send a vehicle to the next depot.

        :param pos: final vehicle position
        :param check_free: if set to False, the check for free parking is ignored
        :return: Depot
        """
        free_depot_positions = {}
        for depot in self.depot_by_id.values():
            if depot.free_parking_spots > 0 or not check_free:
                free_depot_positions[depot.pos] = depot
        re_list = self.routing_engine.return_travel_costs_1toX(pos, free_depot_positions.keys(), max_routes=1)
        if re_list:
            destination_pos = re_list[0][0]
            depot = free_depot_positions[destination_pos]
        else:
            depot = None
        return depot
    
    def time_trigger(self, sim_time):
        super().time_trigger(sim_time)

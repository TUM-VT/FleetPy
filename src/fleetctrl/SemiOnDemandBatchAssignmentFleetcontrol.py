# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
from src.fleetctrl.RidePoolingBatchOptimizationFleetControlBase import RidePoolingBatchOptimizationFleetControlBase
from src.misc.globals import *
from typing import Dict, List, TYPE_CHECKING

# additional module imports (> requirements)
# ------------------------------------------
import numpy as np
import pandas as pd
import shapely
import time
import pyproj
import geopandas as gpd

# src imports
# -----------
from src.simulation.Offers import TravellerOffer
from src.fleetctrl.planning.VehiclePlan import VehiclePlan, PlanStop
from src.fleetctrl.planning.PlanRequest import PlanRequest

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
LOG = logging.getLogger(__name__)
LARGE_INT = 100000

from src.routing.NetworkBase import NetworkBase

if TYPE_CHECKING:
    from src.simulation.Vehicles import SimulationVehicle


# -------------------------------------------------------------------------------------------------------------------- #
# functions
# ---------
def create_stations(columns):
    return PTStation(columns["station_id"], columns["network_node_index"])


INPUT_PARAMETERS_RidePoolingBatchAssignmentFleetcontrol = {
    "doc": """Semi-on-Demand Hybrid Route Batch assignment fleet control (by Max Ng in Dec 2023)
        reference RidePoolingBatchAssignmentFleetcontrol and LinebasedFleetControl
        ride pooling optimisation is called after every optimisation_time_step and offers are created in the time_trigger function
        if "user_max_wait_time_2" is given:
            if the user couldnt be assigned in the first try, it will be considered again in the next opt-step with this new max_waiting_time constraint
        if "user_offer_time_window" is given:
            after accepting an offer the pick-up time is constraint around the expected pick-up time with an interval of the size of this parameter""",
    "inherit": "RidePoolingBatchOptimizationFleetControlBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


class PTStation:
    def __init__(self, station_id, street_network_node_id):
        self.station_id = station_id
        self.street_network_node_id = street_network_node_id


class PtLine:
    def __init__(self, line_id, pt_fleetcontrol_module, routing_engine: NetworkBase, sim_vehicle_dict, vid_to_schedule,
                 sim_start_time, fixed_length, flex_detour, alignment_file, loop_route=False):
        """
        :param line_id: Line ID
        :type line_id: int
        :param pt_fleetcontrol_module: reference to the fleet control module
        :type pt_fleetcontrol_module: SemiOnDemandBatchAssignmentFleetcontrol
        :param routing_engine: routing engine
        :type routing_engine: NetworkBase
        :param sim_vehicle_dict: dict veh_id -> simulaiton vehicle obj (for this line)
        :type sim_vehicle_dict: dict
        :param vid_to_schedule: dict veh_id -> schedule_df
        :type vid_to_schedule: dict
        :param sim_start_time: simulation start time
        :type sim_start_time: float
        :param fixed_length: length of routes that are fixed route (unit in km)
        :type fixed_length: float
        :param flex_detour: detour allowed for flexible portion (factor)
        :type flex_detour: float
        :param alignment_file: alignment file name
        :type alignment_file: str
        :param loop_route: indicates if the route is a loop route (i.e., end at the same station as the start)
        Caution: loop_route cannot lie along the same line (projection would be wrong)
        :type loop_route: bool
        """
        # TODO: future extension to allow definition of multiple fixed route portions
        self.line_id = line_id
        self.pt_fleetcontrol_module: SemiOnDemandBatchAssignmentFleetcontrol = pt_fleetcontrol_module
        self.routing_engine = routing_engine
        self.fixed_length = fixed_length
        self.loop_route = loop_route

        self.sim_vehicles: Dict[int, SimulationVehicle] = sim_vehicle_dict  # line_vehicle_id -> SimulationVehicle
        self.veh_plans: Dict[int, VehiclePlan] = {}  # line_vehicle_id -> vehicle plan

        self.node_index_to_station_id = {}  # node_index -> station id
        self.station_id_km_run = {}  # station id -> km run

        # self.flex_detour = 1.4  # 40% detour allowed for flexible portion
        self.flex_detour = flex_detour

        # transit line alignment
        # alignment has to be single direction line
        # load alignment name from parameters G_PT_ALIGNMENT_F
        self.line_alignment: shapely.LineString = self.load_pt_line_alignment(
            os.path.join(pt_fleetcontrol_module.dir_names[G_DIR_PT], alignment_file.format(line_id=self.line_id)))

        # convert crs of line
        self.dest_crs = "EPSG:32632"  # WGS 84 / UTM zone 32N, for measurement in m
        self.network_crs = self.routing_engine.crs
        self.meter_project = pyproj.Transformer.from_crs(pyproj.CRS(self.network_crs), pyproj.CRS(self.dest_crs),
                                              always_xy=True).transform
        self.point_project = pyproj.Transformer.from_crs(pyproj.CRS(self.network_crs), pyproj.CRS(self.dest_crs),
                                                         always_xy=True).transform

        self.line_alignment_meter = shapely.ops.transform(self.meter_project, self.line_alignment)
        self.route_length = self.line_alignment_meter.length / 1000  # convert to km

        self.fixed_length: float = fixed_length



        for vid, schedule_df in vid_to_schedule.items():
            list_plan_stops = []
            first_stop_in_stim_time_found = False
            return_run: bool = False  # whether a stop in the flexible route portion has been found
            first_flex_dept_time = 0  # last fixed departure time
            first_return_fixed_dept_time = None  # first fixed departure time in the return route
            last_veh_time = sim_start_time # last vehicle time
            last_veh_trip = -1 # last vehicle trip
            for _, scheduled_stop in schedule_df.iterrows():
                # skip stops before sim start time
                if scheduled_stop["departure"] < sim_start_time:
                    continue

                station_id = scheduled_stop["station_id"]
                node_index = self.pt_fleetcontrol_module.station_dict[station_id].street_network_node_id

                terminus_id = 3580 # TODO: put in terminus station_id

                # if trip_id is different from last trip_id, add a planned stop every min before this stop
                if scheduled_stop["trip_id"] != last_veh_trip and station_id == terminus_id:
                    return_run = False  # reset return_run
                    first_return_fixed_dept_time = None # reset first_return_fixed_dept_time

                    last_veh_trip = scheduled_stop["trip_id"]
                    list_plan_stops.append(PlanStop(
                                self.routing_engine.return_node_position(node_index),
                                latest_start_time=last_veh_time + 60,
                                earliest_end_time=scheduled_stop["departure"] - 1,
                                # duration=(scheduled_stop["departure"] - 1) - (last_veh_time + 60) - 1,  # 1s, set the boarding/alighting duration to be nominal,
                                locked = True,
                                # will not be overwritten by the insertion
                                planstop_state=G_PLANSTOP_STATES.RESERVATION,
                            ))
                    # t = last_veh_time + 60
                    # while t+60 < scheduled_stop["departure"]:
                    #     list_plan_stops.append(PlanStop(
                    #         self.routing_engine.return_node_position(node_index),
                    #         earliest_end_time=t,
                    #         latest_start_time=t-1,
                    #         # duration=1,  # 1s, set the boarding/alighting duration to be nominal,
                    #         locked = True,
                    #         # will not be overwritten by the insertion
                    #     ))
                    #     t += 58
                    #     list_plan_stops.append(PlanStop(
                    #         self.routing_engine.return_node_position(node_index),
                    #         earliest_end_time=t,
                    #         latest_start_time=t-1,
                    #         # duration=1,  # 1s, set the boarding/alighting duration to be nominal,
                    #         locked = False,
                    #         # will not be overwritten by the insertion
                    #     ))
                    #     t += 2
                    LOG.debug(
                        f"forced initial stop from {last_veh_time} until {scheduled_stop['departure']} | trip id {scheduled_stop['trip_id']}")



                earliest_departure_dict = {}
                if not np.isnan(scheduled_stop["departure"]):
                    if not return_run:  # fixed route outbound part
                        earliest_departure_dict[-1] = scheduled_stop["departure"]
                        first_flex_dept_time = scheduled_stop["departure"]

                    elif first_return_fixed_dept_time is None:  # first stop after flexible route
                        # earliest_departure_dict[-1] = first_flex_dept_time + (scheduled_stop[
                        #                                    "departure"] - first_flex_dept_time) * self.flex_detour
                        earliest_departure_dict[-1] = (scheduled_stop["departure"]
                                                       + (scheduled_stop["departure"] - first_flex_dept_time) * self.flex_detour)
                        first_return_fixed_dept_time = earliest_departure_dict[-1]
                    else:  # return inbound part: adjust departure time for detour; flexible route will be discarded
                        earliest_departure_dict[-1] = (scheduled_stop["departure"] - first_flex_dept_time
                                                       + first_return_fixed_dept_time)

                ps = PlanStop(self.routing_engine.return_node_position(node_index),
                              latest_start_time=earliest_departure_dict[-1],
                              earliest_end_time=earliest_departure_dict[-1]+1,
                              # duration=1,  # 1s, set the boarding/alighting duration to be nominal,
                              # locked=False,
                              # will not be overwritten by the insertion
                              )
                self.station_id_km_run[station_id] = self.return_pos_km_run(ps.get_pos())
                last_veh_time = earliest_departure_dict[-1]

                # if the station is not in the fixed portion, skip it (except for the first stop in the whole schedule)
                if self.station_id_km_run[station_id] > self.fixed_length and first_stop_in_stim_time_found:
                # if self.station_id_km_run[station_id] > self.fixed_length:
                    return_run = True
                    # if this is not the terminus stop, skip it
                    if station_id != terminus_id:
                        continue

                list_plan_stops.append(ps)
                self.node_index_to_station_id[node_index] = station_id
                LOG.debug(
                    f"sim start time {sim_start_time} | earliest departure {earliest_departure_dict} | {scheduled_stop['departure']}")
                if earliest_departure_dict.get(-1,
                                               -1) < sim_start_time and not first_stop_in_stim_time_found:  # remove schedules before start time
                    list_plan_stops = []
                    if scheduled_stop["departure"] > 75000:
                        LOG.debug(f"cleared list plan stops for vid {vid} with sim start time {sim_start_time}")
                else:
                    first_stop_in_stim_time_found = True

                init_state = {
                    G_V_INIT_NODE: list_plan_stops[0].get_pos()[0],  # set the initial node to be the first stop
                    G_V_INIT_SOC: 1,
                    G_V_INIT_TIME: sim_start_time
                }

                # TODO: delete the following debug code
                self.sim_vehicles[vid].set_initial_state(self.pt_fleetcontrol_module, routing_engine, init_state,
                                                         sim_start_time, veh_init_blocking=False)
                interplan = VehiclePlan(self.sim_vehicles[vid], sim_start_time, routing_engine, list_plan_stops)
                interplan.update_tt_and_check_plan(self.sim_vehicles[vid], sim_start_time, routing_engine,keep_feasible=True)
                LOG.debug(f"interplan: {interplan}")
                LOG.debug(f"interplan feas: {interplan.is_feasible()}")
                if not interplan.is_feasible():
                    exit()

            # init vehicle position at first stop
            init_state = {
                G_V_INIT_NODE: list_plan_stops[0].get_pos()[0],  # set the initial node to be the first stop
                G_V_INIT_SOC: 1,
                G_V_INIT_TIME: sim_start_time
            }
            LOG.debug(f"line vid {vid} with list plan stops: {[str(x) for x in list_plan_stops[:10]]}")
            self.sim_vehicles[vid].set_initial_state(self.pt_fleetcontrol_module, routing_engine, init_state,
                                                     sim_start_time, veh_init_blocking=False)
            self.veh_plans[vid] = VehiclePlan(self.sim_vehicles[vid], sim_start_time, routing_engine, list_plan_stops)
            self.veh_plans[vid].update_plan(self.sim_vehicles[vid], sim_start_time, routing_engine,
                                            keep_time_infeasible=True)
            self.pt_fleetcontrol_module.assign_vehicle_plan(self.sim_vehicles[vid], self.veh_plans[vid], sim_start_time)
        # sort station_id_km_run by km run
        self.station_id_km_run = {k: v for k, v in sorted(self.station_id_km_run.items(), key=lambda item: item[1])}



    def load_pt_line_alignment(self, line_alignment_f):
        """ this method loads the alignment of the PT line
        :param line_alignment_f: line alignment file (geojson)
        :type line_alignment_f: str
        """
        # Load the GeoJSON file
        gdf = gpd.read_file(line_alignment_f)

        # Get the LineString
        return gdf['geometry'].iloc[0]

    def check_request_flexible(self, rq, origin_dest):
        """ check if request is flexible portion of line
        Use GIS to project the request point to the route and check if it is within a certain distance of the route
        :param rq: request object containing all request information
        :type rq: RequestDesign
        :param origin_dest: "origin" or "destination"
        :type origin_dest: str
        :return: True if request is in flexible portion of line, False otherwise
        """
        if origin_dest == "origin":
            pos_to_check = rq.get_origin_pos()
        elif origin_dest == "destination":
            pos_to_check = rq.get_destination_pos()
        else:
            raise NotImplementedError("origin_dest must be either 'origin' or 'destination'")

        # pos_to_check = [node, NONE, NONE]
        coord_to_check = self.routing_engine.return_position_coordinates(pos_to_check)

        point_to_check = shapely.Point(coord_to_check)

        return self.check_point_flexible(point_to_check)

    def check_point_flexible(self, point: shapely.Point):
        """
        Use GIS to project the request point to the route and check if it is within a certain distance of the route
        :param point: request point
        :type point: shapely.Point
        :return: True if request is in flexible portion of line, False otherwise
        """
        # get the distance from the line start
        line_km_run = self.return_km_run(point)

        # check if this is a loop route
        # if self.loop_route:
        if True: # loop route not implemented, all routes set as single direction
            # if the distance is less than the fixed length, return True
            if line_km_run > self.fixed_length:
                return True
            else:
                return False
        else:
            if line_km_run > self.fixed_length and line_km_run < self.line_alignment.length - self.fixed_length:
                return True
            else:
                return False

    def project_point_to_line(self, line: shapely.LineString, point: shapely.Point) -> float:
        """
        Use GIS to project the request point to the route and return the distance from the line start in km
        :param line: line
        :type line: shapely.LineString
        :param point: request point
        :type point: shapely.Point
        :return: distance from the line start in km
        :rtype: float
        """
        # line converted in init to the right projection already, but not the points


        # convert crs of line and point
        # crs_line = shapely.ops.transform(project, line)
        crs_point = shapely.ops.transform(self.point_project, point)
        # crs_point = point

        return line.project(crs_point) / 1000  # convert to km

    def project_point_to_point(self, point1: shapely.Point, point2: shapely.Point) -> float:
        """
        Use GIS to project the request point to the CRS and return the distance between the two points in km
        :param point1: point 1
        :type point: shapely.Point
        :param point2: point 2
        :type point: shapely.Point
        :return: distance between the two points in km
        :rtype: float
        """
        # convert crs of line and point
        # crs_line = shapely.ops.transform(project, line)
        crs_point1 = shapely.ops.transform(self.point_project, point1)
        crs_point2 = shapely.ops.transform(self.point_project, point2)
        # crs_point = point

        return crs_point1.distance(crs_point2) / 1000  # convert to km

    def return_pos_km_run(self, pos) -> float:
        """ this method returns the km run of the position
        :param pos: position
        :type pos: tuple
        :return: km run
        :rtype: float
        """
        # Get projection of the position on the line and then check adjacent two stations

        coord_to_check = self.routing_engine.return_position_coordinates(pos)
        point_to_check = shapely.Point(coord_to_check)

        return self.return_km_run(point_to_check) # projection done in return_km_run -> project_point_to_line

    def return_km_run(self, point: shapely.Point) -> float:
        """
        Use GIS to project the request point to the route and return the distance from the line start
        :param point: request point
        :type point: shapely.Point
        :return: distance from the line start in km
        :rtype: float
        """
        return self.project_point_to_line(self.line_alignment_meter, point)

    # def return_distance_to_station(self, pos: tuple, station_id):
    #     """ this method returns the distance of a position to the station of station_id
    #     :param pos: position
    #     :type pos: tuple
    #     :param station_id: station id
    #     :type station_id: int
    #     :return: distance to closest station
    #     :rtype: float
    #     """
    #     # convert position to point
    #     coord_to_check = self.routing_engine.return_position_coordinates(pos)
    #     point_to_check = shapely.Point(coord_to_check)
    #     return self.return_distance_to_station(point_to_check, station_id)

    def return_distance_to_station(self, point, station_id):
        """ this method returns the distance of a Point to the station of station_id
        :param point: request point or position
        :type point: shapely.Point or tuple
        :param station_id: station id
        :type station_id: int
        :return: distance to station
        :rtype: float
        """
        # if point is not a shapely.Point, convert it to one (assumed a tuple)
        if not isinstance(point, shapely.Point):
            coord_to_check = self.routing_engine.return_position_coordinates(point)
            point_to_check = shapely.Point(coord_to_check)
            return self.return_distance_to_station(point_to_check, station_id)

        station_pos = self.routing_engine.return_node_position(
            self.pt_fleetcontrol_module.station_dict[station_id].street_network_node_id
        )
        station_coord = self.routing_engine.return_position_coordinates(station_pos)
        station_point = shapely.Point(station_coord)
        # return station_point.distance(point)
        return self.project_point_to_point(point, station_point)

    def find_closest_station(self, pos):
        """ this method returns the closest station to the given position
        :param pos: position
        :type pos: tuple
        :return: station id, distance to closest station
        """
        # Get projection of the position on the line and then check adjacent two stations
        line_km_run = self.return_pos_km_run(pos)
        # find the closest two adjacent stations by line_km_run with self.station_id_km_run
        closest_station_id = None
        closest_station_km_run = None
        last_station_id = None
        for station_id, station_km_run in self.station_id_km_run.items():
            last_station_id = closest_station_id
            closest_station_id = station_id
            closest_station_km_run = station_km_run
            if station_km_run > line_km_run:
                break

        # check if the closest station is the next or last station by Euclidian distance
        if closest_station_km_run is None:
            raise NotImplementedError("closest_station_km_run is None")
        if (last_station_id is None or
                self.return_distance_to_station(pos, closest_station_id) <= self.return_distance_to_station(pos,
                                                                                                      last_station_id)):
            return closest_station_id, self.return_distance_to_station(pos, closest_station_id)
        else:
            return last_station_id, self.return_distance_to_station(pos, last_station_id)

    def receive_status_update(self, vid, simulation_time, list_finished_VRL, force_update=True):
        """This method can be used to update plans and trigger processes whenever a simulation vehicle finished some
         VehicleRouteLegs.

        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        :param list_finished_VRL: list of VehicleRouteLeg objects
        :type list_finished_VRL: list
        :param force_update: indicates if also current vehicle plan feasibilities have to be checked
        :type force_update: bool
        """
        veh_obj = self.sim_vehicles[vid]
        self.veh_plans[vid].update_plan(veh_obj, simulation_time, self.routing_engine, list_finished_VRL)

    def query_travel_time_infos(self, o_pos, d_pos, earliest_start_time, nr_pax):
        """ this method will return waiting time and travel time between o_pos and d_pos starting from earliest start time
        if both positions are in the line with earliest arrival time
        :param o_pos: origin position
        :param d_pos: destination postion
        :param earliest_start_time: earliest starting time
        :param nr_pax: number of passengers
        :return: tuple of (waiting time, travel time, arrival time) if both nodes in line, None else
        """
        if self.node_index_to_station_id.get(o_pos[0]) is None or self.node_index_to_station_id.get(d_pos[0]) is None:
            return None
        best_arrival_time = float("inf")
        best_waiting = float("inf")
        best_travel_time = float("inf")
        for vid, veh_plan in self.veh_plans.items():
            first_stop_in_time_found = False
            pu_time = None
            do_time = None
            cur_pax = self.sim_vehicles[vid].get_nr_pax_without_currently_boarding()
            for i, ps in enumerate(veh_plan.list_plan_stops):
                cur_pax += ps.get_change_nr_pax()
                if ps.is_locked():
                    continue
                if not first_stop_in_time_found and ps.get_duration_and_earliest_departure()[1] >= earliest_start_time:
                    first_stop_in_time_found = True
                if first_stop_in_time_found:
                    if ps.get_pos() == o_pos:
                        if i < len(veh_plan.list_plan_stops) - 1:
                            if veh_plan.list_plan_stops[i + 1].get_pos() == ps.get_pos():
                                continue
                        if cur_pax + nr_pax > self.sim_vehicles[vid].max_pax:
                            continue
                        pu_time = ps.get_duration_and_earliest_departure()[1]
                        p_cur_pax = cur_pax + nr_pax
                        for j in range(i + 1, len(veh_plan.list_plan_stops)):
                            ps = veh_plan.list_plan_stops[j]
                            if ps.get_pos() == d_pos and pu_time is not None:
                                do_time = ps.get_duration_and_earliest_departure()[1]
                                break
                            p_cur_pax += ps.get_change_nr_pax()
                            if p_cur_pax > self.sim_vehicles[vid].max_pax:
                                pu_time = None
                                break
                        if pu_time is not None and do_time is not None:
                            break
            if do_time is not None:
                if do_time < best_arrival_time:
                    best_arrival_time = do_time
                    best_waiting = pu_time - earliest_start_time
                    best_travel_time = do_time - pu_time
                elif do_time == best_arrival_time and do_time - pu_time < best_travel_time:
                    best_arrival_time = do_time
                    best_waiting = pu_time - earliest_start_time
                    best_travel_time = do_time - pu_time
        LOG.info("query travel time {} : at : {} wt: {} tt: {}".format(self.line_id, best_arrival_time, best_waiting,
                                                                       best_travel_time))
        return best_waiting, best_travel_time, best_arrival_time

    def assign_user(self, rid, simulation_time):
        # not used
        """
        this function is called when a request is accepted and already assigned to this pt line -> update vehplans
        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """

        rq = self.pt_fleetcontrol_module.rq_dict[rid]
        nr_pax = rq.nr_pax
        o_pos, earliest_start_time, _ = rq.get_o_stop_info()
        d_pos, _, _ = rq.get_d_stop_info()
        if self.node_index_to_station_id.get(o_pos[0]) is None or self.node_index_to_station_id.get(d_pos[0]) is None:
            raise NotImplementedError("line {} cant served rid {} | {}".format(self.line_id, rid, rq))
        best_arrival_time = float("inf")
        best_travel_time = float("inf")
        # best_waiting = float("inf")
        best_vid = None
        best_o_ps_index = None
        best_d_ps_index = None

        # Find the best vehicle to assign the request to
        for vid, veh_plan in self.veh_plans.items():
            first_stop_in_time_found = False
            pu_time = None
            do_time = None
            o_ps_index = None
            d_ps_index = None
            cur_pax = self.sim_vehicles[vid].get_nr_pax_without_currently_boarding()
            # Find the best origin stops to assign the request to
            for i, ps in enumerate(veh_plan.list_plan_stops):
                cur_pax += ps.get_change_nr_pax()
                if ps.is_locked():
                    continue
                if not first_stop_in_time_found and ps.get_duration_and_earliest_departure()[1] >= earliest_start_time:
                    first_stop_in_time_found = True
                if first_stop_in_time_found:
                    if ps.get_pos() == o_pos:
                        if i < len(veh_plan.list_plan_stops) - 1:
                            if veh_plan.list_plan_stops[i + 1].get_pos() == ps.get_pos():
                                continue
                        if cur_pax + nr_pax > self.sim_vehicles[vid].max_pax:
                            continue
                        pu_time = ps.get_duration_and_earliest_departure()[1]
                        o_ps_index = i
                        p_cur_pax = cur_pax + nr_pax
                        # Find the best destination stop to assign the request to
                        for j in range(i + 1, len(veh_plan.list_plan_stops)):
                            ps = veh_plan.list_plan_stops[j]
                            if ps.get_pos() == d_pos and pu_time is not None:
                                do_time = ps.get_duration_and_earliest_departure()[1]
                                d_ps_index = j
                                break
                            p_cur_pax += ps.get_change_nr_pax()
                            if p_cur_pax > self.sim_vehicles[vid].max_pax:
                                pu_time = None
                                o_ps_index = None
                                break
                        if pu_time is not None and do_time is not None:
                            break
            if do_time is not None:
                if do_time < best_arrival_time:
                    best_arrival_time = do_time
                    # best_waiting = pu_time - earliest_start_time
                    best_travel_time = do_time - pu_time
                    best_o_ps_index = o_ps_index
                    best_d_ps_index = d_ps_index
                    best_vid = vid
                elif do_time == best_arrival_time and do_time - pu_time < best_travel_time:
                    best_arrival_time = do_time
                    # best_waiting = pu_time - earliest_start_time
                    best_travel_time = do_time - pu_time
                    best_o_ps_index = o_ps_index
                    best_d_ps_index = d_ps_index
                    best_vid = vid

        # Assign the request to the best vehicle
        list_plan_stops = self.veh_plans[best_vid].list_plan_stops

        o_ps: PlanStop = list_plan_stops[best_o_ps_index]
        boarding_list = o_ps.get_list_boarding_rids() + [rid]
        new_boarding_dict = {1: boarding_list, -1: o_ps.get_list_alighting_rids()}
        new_o_ps = PlanStop(o_ps.get_pos(), boarding_dict=new_boarding_dict,
                            earliest_end_time=o_ps.get_duration_and_earliest_departure()[1],
                            change_nr_pax=o_ps.get_change_nr_pax() + rq.nr_pax)
        list_plan_stops[best_o_ps_index] = new_o_ps

        d_ps: PlanStop = list_plan_stops[best_d_ps_index]
        deboarding_list = d_ps.get_list_alighting_rids() + [rid]
        new_boarding_dict = {1: d_ps.get_list_boarding_rids(), -1: deboarding_list}
        new_d_ps = PlanStop(d_ps.get_pos(), boarding_dict=new_boarding_dict,
                            earliest_end_time=d_ps.get_duration_and_earliest_departure()[1],
                            change_nr_pax=d_ps.get_change_nr_pax() - rq.nr_pax)
        list_plan_stops[best_d_ps_index] = new_d_ps

        new_veh_plan = VehiclePlan(self.sim_vehicles[best_vid], simulation_time, self.routing_engine, list_plan_stops)
        self.pt_fleetcontrol_module.assign_vehicle_plan(self.sim_vehicles[best_vid], new_veh_plan, simulation_time)
        self.pt_fleetcontrol_module.rid_to_assigned_vid[rid] = best_vid

    def remove_rid_from_line(self, rid, assigned_vid, simulation_time):
        """ this function is called when a request is canceled and allready assigned to pt line -> remove rid from vehplans
        """
        raise NotImplementedError


class SemiOnDemandBatchAssignmentFleetcontrol(RidePoolingBatchOptimizationFleetControlBase):
    def __init__(self, op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                 dir_names, op_charge_depot_infra=None, list_pub_charging_infra=[]):
        """Combined fleet control for semi-on-demand flexible & fixed route
        Reference LinebasedFleetControl for more information on the solely fixed-route implementation.

        ride pooling optimisation is called after every optimisation_time_step and offers are created in the time_trigger function
        if "user_max_wait_time_2" is given:
            if the user couldnt be assigned in the first try, it will be considered again in the next opt-step with this new max_waiting_time constraint
        if "user_offer_time_window" is given:
            after accepting an offer the pick-up time is constraint around the expected pick-up time with an interval of the size of this parameter


        :param op_id: operator id
        :type op_id: int
        :param operator_attributes: dictionary with keys from globals and respective values
        :type operator_attributes: dict
        :param list_vehicles: simulation vehicles; their assigned plans should be instances of the VehicleRouteLeg class
        :type list_vehicles: list
        :param routing_engine: routing engine
        :type routing_engine: Network
        :param scenario_parameters: access to all scenario parameters (if necessary)
        :type scenario_parameters: dict
        :param op_charge_depot_infra: reference to a OperatorChargingAndDepotInfrastructure class (optional) (unique for each operator)
        :type op_charge_depot_infra: OperatorChargingAndDepotInfrastructure
        :param list_pub_charging_infra: list of PublicChargingInfrastructureOperator classes (optional) (accesible for all agents)
        :type list_pub_charging_infra: list of PublicChargingInfrastructureOperator
        """

        # TODO: now always assume a loop route; to generalize

        operator_attributes[G_RA_RP_BATCH_OPT] = "InsertionHeuristic"  # hard code over-ride Alonso-Mora
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                         dir_names=dir_names, op_charge_depot_infra=op_charge_depot_infra,
                         list_pub_charging_infra=list_pub_charging_infra)
        self.max_wait_time_2 = operator_attributes.get(G_OP_MAX_WT_2, None)
        # if np.isnan(self.max_wait_time_2):
        #     self.max_wait_time_2 = None
        self.offer_pickup_time_interval = operator_attributes.get(G_OP_OFF_TW, None)
        # if np.isnan(self.offer_pickup_time_interval):
        #     self.offer_pickup_time_interval = None
        self.unassigned_requests_1 = {}
        self.unassigned_requests_2 = {}

        # fixed route parameters
        self.pt_data_dir = os.path.join(dir_names[G_DIR_PT])
        self.fixed_length = scenario_parameters.get(G_PT_FIXED_LENGTH, None)
        self.flex_detour = scenario_parameters.get(G_PT_FLEX_DETOUR, None)
        self.alignment_file = scenario_parameters.get(G_PT_ALIGNMENT_F, 0)

        self.base_fare = scenario_parameters.get(G_PT_FARE_B, 0)
        self.walking_speed = scenario_parameters.get(G_WALKING_SPEED, 0)
        station_node_f = os.path.join(self.pt_data_dir, scenario_parameters.get(G_PT_STATION_F, 0))
        station_node_df = pd.read_csv(station_node_f)

        self.begin_approach_buffer_time = 0
        # self.station_dict = {}  # station_id -> PTStation
        tmp_station_dict = station_node_df.apply(create_stations, axis=1).to_dict()
        self.station_dict: Dict[int, PTStation] = {}
        for _, pt_station in tmp_station_dict.items():
            self.station_dict[pt_station.station_id] = pt_station
        # creation of additional access options to pt-station objects
        self.st_nw_stations: Dict[int, List[int]] = {}  # street nw node id -> list station ids
        tmp_st_nw_stations = {}
        for station_obj in self.station_dict.values():
            if station_obj.street_network_node_id in tmp_st_nw_stations.keys():
                tmp_st_nw_stations[station_obj.street_network_node_id].append(station_obj)
            else:
                tmp_st_nw_stations[station_obj.street_network_node_id] = [station_obj]
        # sort such that rail stations are first
        for k, v in tmp_st_nw_stations.items():
            self.st_nw_stations[k] = v

        # line specification output
        # pd.DataFrame(pt_line_specifications_list).to_csv(
        #     os.path.join(dir_names[G_DIR_OUTPUT], f"3-{self.op_id}_pt_vehicles.csv"), index=False)

        # PT lines
        self.PT_lines: Dict[int, PtLine] = {}  # line -> PtLine obj
        self.pt_vehicle_to_line = {}  # pt_veh_id -> line
        self.walking_dist_origin = {} # rid -> walking time to origin
        self.walking_dist_destination = {} # rid -> walking time to destination
        # self.sim_time = -1

        # # check whether the stations are in flexible portion
        # self.station_id_flexible: Dict[int, bool] = {}  # station id -> bool
        # # load coordinates of stations through self.station_dict
        # for station_id, station_obj in self.station_dict.items():
        #     self.station_id_flexible[station_id] = self.PT_lines[0].check_point_flexible(
        #         self.routing_engine.return_node_position(station_obj.street_network_node_id))

        # init line information
        # schedules are read and the vehicles that have to be created in the fleetsimulation class are collected
        schedules = pd.read_csv(os.path.join(self.pt_data_dir, scenario_parameters[G_PT_SCHEDULE_F]))
        pt_vehicle_id = 0
        pt_line_specifications_list = []
        self.vehicles_to_initialize = {}  # pt_vehicle_id -> veh_type
        self.schedule_to_initialize = {}  # line -> pt_vehicle_id -> schedule_df
        for key, vehicle_line_schedule in schedules.groupby(["LINE", "line_vehicle_id", "vehicle_type"]):
            line, line_vehicle_id, vehicle_type = key
            pt_line_specifications_list.append(
                {"line": line, "line_vehicle_id": line_vehicle_id, "vehicle_type": vehicle_type,
                 "sim_vehicle_id": pt_vehicle_id})
            self.vehicles_to_initialize[pt_vehicle_id] = vehicle_type
            if self.schedule_to_initialize.get(line) is None:
                self.schedule_to_initialize[line] = {}
            self.schedule_to_initialize[line][pt_vehicle_id] = vehicle_line_schedule
            pt_vehicle_id += 1

        # line specification output
        pd.DataFrame(pt_line_specifications_list).to_csv(os.path.join(dir_names[G_DIR_OUTPUT], f"3-{self.op_id}_pt_vehicles.csv"), index=False)

        #
        self.rq_dict = {}
        self.routing_engine = routing_engine
        self.zones = zone_system
        self.dyn_output_dict = {}
        # self.rid_to_assigned_vid = {}
        #
        self._vid_to_assigned_charging_process = {}
        self.veh_plans = {}
        self.dyn_fleet_sizing = None
        self.repo = None
        LOG.info(f"SoD finish first init {len(self.PT_lines)}")

    def return_vehicles_to_initialize(self) -> Dict[int, str]:
        """
        return vehicles that have to be initialized in the fleetsimulation class
        :return dict pt_vehicle_id -> veh_type
        """
        return self.vehicles_to_initialize

    def continue_init(self, sim_vehicle_objs, sim_start_time):
        """
        this method continues initialization after simulation vehicles have been created in the fleetsimulation class
        :param sim_vehicle_objs: ordered list of sim_vehicle_objs
        :param sim_start_time: simulation start time
        """
        self.sim_time = sim_start_time
        veh_obj_dict = {veh.vid: veh for veh in sim_vehicle_objs}
        for line, vid_to_schedule_dict in self.schedule_to_initialize.items():
            for vid in vid_to_schedule_dict.keys():
                self.pt_vehicle_to_line[vid] = line
            schedule_vehicles = {vid: veh_obj_dict[vid] for vid in vid_to_schedule_dict.keys()}
            self.PT_lines[line] = PtLine(line, self, self.routing_engine, schedule_vehicles, vid_to_schedule_dict,
                                         sim_start_time, self.fixed_length, self.flex_detour, self.alignment_file,)
        LOG.info(f"SoD finish continue_init {len(self.PT_lines)}")

    def assign_vehicle_plan(self, veh_obj, vehicle_plan, sim_time, force_assign=False, assigned_charging_task=None,
                            add_arg=None):
        """ this method should be used to assign a new vehicle plan to a vehicle

        WHEN OVERWRITING THIS FUNCTION MAKE SURE TO CALL AT LEAST THE LINES BELOW (i.e. super())

        :param veh_obj: vehicle obj to assign vehicle plan to
        :type veh_obj: SimulationVehicle
        :param vehicle_plan: vehicle plan that should be assigned
        :type vehicle_plan: VehiclePlan
        :param sim_time: current simulation time in seconds
        :type sim_time: int
        :param force_assign: this parameter can be used to enforce the assignment, when a plan is (partially) locked
        :type force_assign: bool
        :param add_arg: possible additional argument if needed
        :type add_arg: not defined here
        """
        super().assign_vehicle_plan(veh_obj, vehicle_plan, sim_time, force_assign=force_assign,
                                    assigned_charging_task=assigned_charging_task, add_arg=add_arg)
        # new_vrl = vehicle_plan.build_VRL(veh_obj, self.rq_dict, charging_management=self.charging_management)
        # LOG.debug("init plan")
        # for ps in vehicle_plan.list_plan_stops:
        #     LOG.info(str(ps))
        # LOG.debug("init vrl")
        # for x in new_vrl:
        #     LOG.info(str(x))
        # veh_obj.assign_vehicle_plan(new_vrl, sim_time, force_ignore_lock=force_assign)

        if self.PT_lines.get(self.pt_vehicle_to_line[veh_obj.vid]) is not None:
            self.PT_lines[self.pt_vehicle_to_line[veh_obj.vid]].veh_plans[veh_obj.vid] = vehicle_plan
        # else:
        #     LOG.warning("couldnt find {} or {} | only feasible in init".format(veh_obj.vid, self.pt_vehicle_to_line.get(
        #         veh_obj.vid)))
        # self.veh_plans[veh_obj.vid] = vehicle_plan

        # should be redundant with super()
        # for rid in get_assigned_rids_from_vehplan(vehicle_plan):
        #     pax_info = vehicle_plan.get_pax_info(rid)
        #     self.rq_dict[rid].set_assigned(pax_info[0], pax_info[1])
        #     self.rid_to_assigned_vid[rid] = veh_obj.vid

    def receive_status_update(self, vid, simulation_time, list_finished_VRL, force_update=True):
        """This method can be used to update plans and trigger processes whenever a simulation vehicle finished some
         VehicleRouteLegs.

        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        :param list_finished_VRL: list of VehicleRouteLeg objects
        :type list_finished_VRL: list
        :param force_update: indicates if also current vehicle plan feasibilities have to be checked
        :type force_update: bool
        """
        self.sim_time = simulation_time

        if self.PT_lines:
            self.PT_lines[self.pt_vehicle_to_line[vid]].receive_status_update(vid, simulation_time, list_finished_VRL,
                                                                              force_update=force_update)
        veh_obj = self.sim_vehicles[vid]
        upd_utility_val = self.compute_VehiclePlan_utility(simulation_time, veh_obj, self.veh_plans[vid])
        self.veh_plans[vid].set_utility(upd_utility_val)

    def return_ptline_of_user(self, rq):
        """ this method returns the PT line that the user belongs to
        :param rq: request object containing all request information
        :type rq: RequestDesign
        :return: PT line
        :rtype: PtLine
        """

        # TODO: return the PT line that the user belongs based on some logics
        # currently, just assign to the first line
        first_index = next(iter(self.PT_lines))
        return self.PT_lines[first_index]

    def user_request(self, rq, sim_time):
        """ This method is triggered for a new incoming request. It generally generates a PlanRequest from the rq and
        adds it to the database.
        :param rq: request object containing all request information
        :type rq: RequestDesign
        :param sim_time: current simulation time
        :type sim_time: int
        """

        # get user's PT line
        pt_line = self.return_ptline_of_user(rq)

        # check if the origin and destination are in the flexible portion of the line
        pick_up_pos = rq.get_origin_pos()
        drop_off_pos = rq.get_destination_pos()
        # assign to the closest stop, compute the walking distance to the assigned stop
        to_check = {"origin": pick_up_pos, "destination": drop_off_pos}
        flex_or_not = {"origin": False, "destination": False}
        walking_dist_dict = {"origin": 0.0, "destination": 0.0}
        walking_time = {"origin": 0.0, "destination": 0.0}
        for origin_dest, pos in to_check.items():
            # if in the fixed portion, update the pick-up / drop-off location to the assigned stop
            flex_or_not[origin_dest] = pt_line.check_request_flexible(rq, origin_dest)
            if not pt_line.check_request_flexible(rq, origin_dest):
                # check the closest stop to the request location (pos)
                closest_station_id, walking_dist = pt_line.find_closest_station(pos)
                to_check[origin_dest] = pt_line.routing_engine.return_node_position(
                    self.station_dict[closest_station_id].street_network_node_id
                )
                walking_dist_dict[origin_dest] = walking_dist
                walking_time[origin_dest] = walking_dist / self.walking_speed

        # update the pick-up / drop-off location to the assigned stop
        pick_up_pos = to_check["origin"]
        drop_off_pos = to_check["destination"]

        # check if walking time already exists, if not, add it
        if self.walking_dist_origin.get(rq.rid):
            LOG.debug(f"walking time origin overridden {rq.rid}")
        self.walking_dist_origin[rq.rid] = walking_dist_dict["origin"]
        if self.walking_dist_destination.get(rq.rid):
            LOG.debug(f"walking time destination overridden {rq.rid}")
        self.walking_dist_destination[rq.rid] = walking_dist_dict["destination"]

        # if pt_line.check_request_flexible(rq, "origin"):
        #     pick_up_pos = [pt_line.routing_engine.return_node_position(
        #         self.station_dict[pt_line.node_index_to_station_id[pick_up_pos[0]]].street_network_node_id
        #     ), None, None]
        # if pt_line.check_request_flexible(rq, "destination"):
        #     drop_off_pos = [pt_line.routing_engine.return_node_position(
        #         self.station_dict[pt_line.node_index_to_station_id[drop_off_pos[0]]].street_network_node_id
        #     ), None, None]

        # local implementation of super().user_request(rq, sim_time)
        LOG.debug(f"Incoming request {rq.__dict__} at time {sim_time}")
        if self.rq_dict.get(rq.get_rid_struct()):
            return
        t0 = time.perf_counter()

        self.sim_time = sim_time
        prq = PlanRequest(rq, self.routing_engine, min_wait_time=self.min_wait_time,
                          max_wait_time=self.max_wait_time,
                          max_detour_time_factor=self.max_dtf, max_constant_detour_time=self.max_cdt,
                          add_constant_detour_time=self.add_cdt, min_detour_time_window=self.min_dtw,
                          boarding_time=self.const_bt,
                          # use the updated pick-up / drop-off location
                          pickup_pos=pick_up_pos, dropoff_pos=drop_off_pos,
                          walking_time_start=walking_time["origin"], walking_time_end=walking_time["destination"],
                          )
        rid_struct = rq.get_rid_struct()

        if prq.o_pos == prq.d_pos:
            LOG.debug(f"automatic decline for rid {rid_struct}!")
            self._create_rejection(prq, sim_time)
            return

        self.new_requests[rid_struct] = 1
        self.rq_dict[rid_struct] = prq

        o_pos, t_pu_earliest, t_pu_latest = prq.get_o_stop_info()
        if t_pu_earliest - sim_time > self.opt_horizon:
            self.reservation_module.add_reservation_request(prq, sim_time)
            LOG.debug(f"reservation rid {rid_struct}")
            prq.set_reservation_flag(True)
            self.RPBO_Module.add_new_request(rid_struct, prq, consider_for_global_optimisation=False)
        else:
            self.RPBO_Module.add_new_request(rid_struct, prq)

        # record cpu time
        dt = round(time.perf_counter() - t0, 5)
        old_dt = self._get_current_dynamic_fleetcontrol_value(sim_time, G_FCTRL_CT_RQU)
        if old_dt is None:
            new_dt = dt
        else:
            new_dt = old_dt + dt
        output_dict = {G_FCTRL_CT_RQU: new_dt}
        self._add_to_dynamic_fleetcontrol_output(sim_time, output_dict)

        if not self.rq_dict[rq.get_rid_struct()].get_reservation_flag():
            self.unassigned_requests_1[rq.get_rid_struct()] = 1
        return {}

        # no need to create user offer (like in the PT line calling self._create_user_offer())
        # it will be called during batch processing

    def user_confirms_booking(self, rid, simulation_time):
        # no need add LinebasedFleetControl implementation
        # separate fixed and flexible portion (did NOT do, suppose stops updated in Request)
        # if fixed, change the locations of the pick-up and drop-off to the assigned stops
        """This method is used to confirm a customer booking.
        in a first step the pick-up time constraints are updated based on the offer mad, if "user_offer_time_window" is given
        in the second step higher level database processes are triggered to fix request

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        pu_offer_tuple = self._get_offered_time_interval(rid)
        if pu_offer_tuple is not None:
            new_earliest_pu, new_latest_pu = pu_offer_tuple
            self.change_prq_time_constraints(simulation_time, rid, new_latest_pu, new_ept=new_earliest_pu)

        super().user_confirms_booking(rid, simulation_time)

    def user_cancels_request(self, rid, simulation_time):
        # no need add LinebasedFleetControl implementation
        # no need separate fixed and flexible portion
        """This method is used to confirm a customer cancellation.
        in the first step the assigned vehicle plan for the rid is updated (rid is removed from plan/v2rb)
        in the second step higher level data base processes are triggered to delete request

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        assigned_vid = self.rid_to_assigned_vid.get(rid, None)
        if assigned_vid is not None:
            veh_obj = self.sim_vehicles[assigned_vid]
            assigned_plan = self.RPBO_Module.get_current_assignment(assigned_vid)
            new_best_plan = self.RPBO_Module.get_vehicle_plan_without_rid(veh_obj, assigned_plan, rid, simulation_time)
            if new_best_plan is not None:
                self.assign_vehicle_plan(veh_obj, new_best_plan, simulation_time, force_assign=True)
            else:
                assigned_plan = VehiclePlan(veh_obj, self.sim_time, self.routing_engine, [])
                self.assign_vehicle_plan(veh_obj, assigned_plan, simulation_time, force_assign=True)
        super().user_cancels_request(rid, simulation_time)

    def _call_time_trigger_request_batch(self, simulation_time):
        """ this function first triggers the upper level batch optimisation
        based on the optimisation solution offers to newly assigned requests are created in the second step with following logic:
        declined requests will recieve an empty dict
        unassigned requests with a new assignment try in the next opt-step dont get an answer
        new assigned request will recieve a non empty offer-dict

        a retry is only made, if "user_max_wait_time_2" is given

        every request as to answer to an (even empty) offer to be deleted from the system!

        :param simulation_time: current time in simulation
        :return: dictionary rid -> offer for each unassigned request, that will recieve an answer. (offer: dictionary with plan specific entries; empty if no offer can be made)
        :rtype: dict
        """
        # TODO: remove super() and alter the veh list to only flexible route for flexible demand; only fixed vehicles for fixed demand
        super()._call_time_trigger_request_batch(simulation_time)
        # embed()
        rid_to_offers = {}
        if self.sim_time % self.optimisation_time_step == 0:
            new_unassigned_requests_2 = {}
            # rids to be assigned in first try
            for rid in self.unassigned_requests_1.keys():
                assigned_vid = self.rid_to_assigned_vid.get(rid, None)
                prq = self.rq_dict[rid]
                if assigned_vid is None:
                    if self.max_wait_time_2 is not None and self.max_wait_time_2 > 0:  # retry with new waiting time constraint (no offer returned)
                        new_unassigned_requests_2[rid] = 1
                        self.RPBO_Module.delRequest(rid)
                        _, earliest_pu, _ = prq.get_o_stop_info()
                        new_latest_pu = earliest_pu + self.max_wait_time_2
                        self.change_prq_time_constraints(simulation_time, rid, new_latest_pu)
                        self.RPBO_Module.add_new_request(rid, prq)
                    else:  # no retry, rid declined
                        self._create_user_offer(prq, simulation_time)
                else:
                    assigned_plan = self.veh_plans[assigned_vid]
                    self._create_user_offer(prq, simulation_time, assigned_vehicle_plan=assigned_plan)
            for rid in self.unassigned_requests_2.keys():  # check second try rids
                assigned_vid = self.rid_to_assigned_vid.get(rid, None)
                if assigned_vid is None:  # decline
                    self._create_user_offer(self.rq_dict[rid], simulation_time)
                else:
                    prq = self.rq_dict[rid]
                    assigned_plan = self.veh_plans[assigned_vid]
                    self._create_user_offer(prq, simulation_time, assigned_vehicle_plan=assigned_plan)
            self.unassigned_requests_1 = {}
            self.unassigned_requests_2 = new_unassigned_requests_2  # retry rids
            LOG.debug("end of opt:")
            LOG.debug("unassigned_requests_2 {}".format(self.unassigned_requests_2))
            LOG.debug("offers: {}".format(rid_to_offers))

    def vehicle_information_needed_for_optimisation(self, sim_time):
        """ needed for coupling with aimsun: tests if new vehicle information should be fetched
        better way to to that?
        """
        self.sim_time = sim_time
        if sim_time % self.optimisation_time_step == 0:
            return True
        else:
            return False

    def _create_user_offer(self, rq, simulation_time, assigned_vehicle_plan=None, offer_dict_without_plan={}):
        # follow implementation here and do not follow LinebasedFleetControl
        if assigned_vehicle_plan is not None:
            pu_time, do_time = assigned_vehicle_plan.pax_info.get(rq.get_rid_struct())
            add_offer = {}
            pu_offer_tuple = self._get_offered_time_interval(rq.get_rid_struct())
            if pu_offer_tuple is not None:
                new_earliest_pu, new_latest_pu = pu_offer_tuple
                add_offer[G_OFFER_PU_INT_START] = new_earliest_pu
                add_offer[G_OFFER_PU_INT_END] = new_latest_pu

            # additional info for output here, e.g., fixed/flexible, access time
            add_offer[G_OFFER_WALKING_DISTANCE_ORIGIN] = self.walking_dist_origin[rq.get_rid_struct()]
            add_offer[G_OFFER_WALKING_DISTANCE_DESTINATION] = self.walking_dist_destination[rq.get_rid_struct()]

            offer = TravellerOffer(rq.get_rid(), self.op_id, pu_time - rq.get_rq_time(), do_time - pu_time,
                                   int(rq.init_direct_td * self.dist_fare + self.base_fare),
                                   additional_parameters=add_offer)
            rq.set_service_offered(offer)
        else:
            offer = self._create_rejection(rq, simulation_time)
        return offer

    def _get_offered_time_interval(self, rid):
        """ this function creates an offer for the pickup-intervall for a request in case the parameter G_OP_OFF_TW is given depending on the planned pick-up time of the assigned tour
        :param rid: request id
        :return: None, if G_OP_OFF_TW is not given, tuple (new_earlest_pickup_time, new_latest_pick_up_time) for new pickup time constraints
        """
        if self.offer_pickup_time_interval is not None:  # set new pickup time constraints based on expected pu-time and offer time interval
            prq = self.rq_dict[rid]
            _, earliest_pu, latest_pu = prq.get_o_stop_info()
            vid = self.rid_to_assigned_vid[rid]
            assigned_plan = self.veh_plans[vid]
            pu_time, _ = assigned_plan.pax_info.get(rid)
            if pu_time - self.offer_pickup_time_interval / 2.0 < earliest_pu:
                new_earliest_pu = earliest_pu
                new_latest_pu = earliest_pu + self.offer_pickup_time_interval
            elif pu_time + self.offer_pickup_time_interval / 2.0 > latest_pu:
                new_latest_pu = latest_pu
                new_earliest_pu = latest_pu - self.offer_pickup_time_interval
            else:
                new_earliest_pu = pu_time - self.offer_pickup_time_interval / 2.0
                new_latest_pu = pu_time + self.offer_pickup_time_interval / 2.0
            return new_earliest_pu, new_latest_pu
        else:
            return None

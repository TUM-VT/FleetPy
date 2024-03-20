# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
from src.fleetctrl.RidePoolingBatchOptimizationFleetControlBase import RidePoolingBatchOptimizationFleetControlBase
from src.fleetctrl.SemiOnDemandBatchAssignmentFleetcontrol import PTStation, PtLine, SemiOnDemandBatchAssignmentFleetcontrol
from src.misc.globals import *
from typing import Dict, List, TYPE_CHECKING, Any

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
# from src.fleetctrl.SoDZonalControlRL import SoDZonalControlRL

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
LOG = logging.getLogger(__name__)
LARGE_INT = 100000
# TOL = 0.1

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

class PtLineZonal(PtLine):
    def __init__(self, line_id, pt_fleetcontrol_module, schedule_vehicles, vid_to_schedule_dict, sim_start_time, sim_end_time):
        """
        :param line_id: Line ID
        :type line_id: int
        :param pt_fleetcontrol_module: reference to the fleet control module
        :type pt_fleetcontrol_module: SemiOnDemandBatchAssignmentFleetcontrol
        :param sim_start_time: simulation start time
        :type sim_start_time: float
        :param sim_end_time: simulation end time
        :type sim_end_time: float
        :param loop_route: indicates if the route is a loop route (i.e., end at the same station as the start)
        Caution: loop_route cannot lie along the same line (projection would be wrong)
        :type loop_route: bool
        """

        # call super().__init__() to load original SoD vehicles
        # vehicles are all locked at the last scheduled location
        super().__init__(line_id, pt_fleetcontrol_module, schedule_vehicles, vid_to_schedule_dict, sim_start_time, sim_end_time)

        self.run_schedule = self.pt_fleetcontrol_module.run_schedule
        self.sim_end_time = self.pt_fleetcontrol_module.sim_end_time

        # zonal setting parameters
        self.n_zones = self.pt_fleetcontrol_module.n_zones
        if self.fixed_length >= self.route_length:
            self.n_zones = 1

        self.n_reg_veh = self.pt_fleetcontrol_module.n_reg_veh
        self.pt_zone_min_detour_time = self.pt_fleetcontrol_module.pt_zone_min_detour_time

        zone_len = (self.route_length - self.fixed_length) / self.n_zones
        self.zone_x_min = [self.fixed_length + zone_len * i for i in range(self.n_zones)]
        self.zone_x_max = [self.fixed_length + zone_len * (i + 1) for i in range(self.n_zones)]

        self.veh_zone_assignment = {}  # dict vid -> zone assignment; -1 for regular vehicles
        # if self.n_zones > 1:
        for vid in self.sim_vehicles.keys():
            if vid < self.n_reg_veh:
                self.veh_zone_assignment[vid] = -1
            else:
                self.veh_zone_assignment[vid] = vid % self.n_zones

    def return_x_zone(self, x) -> int:
        """
        Return the zone of the x
        :param x: x-coord
        :type x: float
        :return: zone
        :rtype: int
        """
        # fully flexible route case: check if x is around the terminus
        if (self.fixed_length < min(self.station_id_km_run.items(), key=lambda x: x[1])[1]
                and x < min(self.station_id_km_run.items(), key=lambda x: x[1])[1] + G_PT_X_TOL):
            return -1

        if x < self.fixed_length:
            return -1
        for i in range(self.n_zones):
            if self.zone_x_min[i] <= x < self.zone_x_max[i]:
                return i
        return i

    def return_pos_zone(self, pos) -> int:
        """
        Return the zone of the position
        :param pos: position
        :type pos: float
        :return: zone
        :rtype: int
        """
        x_pos = self.return_pos_km_run(pos)
        return self.return_x_zone(x_pos)

    def find_closest_station_to_x(self, x, higher_than_x):
        """
        Find the closest station to x
        :param x: position
        :type x: float
        :param higher_than_x: indicates if the station should be higher than x
        :type higher_than_x: bool
        :return: closest station_id to x
        :rtype: int
        """
        last_station_id = self.run_schedule["station_id"].iloc[0]
        last_km = -1
        for station in self.run_schedule["station_id"]:
            station_km_run = self.station_id_km_run[station]
            if station_km_run > x:
                if higher_than_x:
                    return station
                else:
                    return last_station_id
                break
            if last_km > station_km_run: # return part, break
                break
            last_km = station_km_run
            last_station_id = station
        # for station_id, station_km_run in self.station_id_km_run.items():
        #     if station_km_run > x:
        #         if higher_than_x:
        #             return station_id
        #         else:
        #             return last_station_id
        #         break
        #     last_station_id = station_id
        if not higher_than_x:
            return last_station_id
        else:
            return None

    def get_time_between_x(self, x1, x2):
        """
        Get the scheduled time to travel back and fro between x1 and x2
        (by referencing the schedule for the closest stops in the bound)
        :param x1: lower bound of x
        :type x1: float
        :param x2: upper bound of x
        :type x2: float
        :return: time between x1 and x2
        :rtype: float
        """
        x1_station = self.find_closest_station_to_x(x1, True)
        x2_station = self.find_closest_station_to_x(x2, False)
        x1_time = self.run_schedule.loc[self.run_schedule["station_id"] == x1_station, "departure"].values[0]
        x2_time = self.run_schedule.loc[self.run_schedule["station_id"] == x2_station, "departure"].values[0]
        return x2_time - x1_time

    def get_time_from_terminus_to_x(self, x):
        """
        Get the scheduled time to travel from terminus to x
        :param x: position
        :type x: float
        :return: time from terminus to x
        :rtype: float
        """
        x_station = self.find_closest_station_to_x(x, True)
        x_time = self.run_schedule.loc[self.run_schedule["station_id"] == x_station, "departure"].values[0]
        return x_time

    def get_travel_time_between_nodes(self, n1, n2):
        """
        Get the travel time between two nodes
        :param n1: node 1
        :type n1: int
        :param n2: node 2
        :type n2: int
        :return: travel time
        :rtype: float
        """
        if n1==n2:
            return 0
        return self.routing_engine.return_travel_costs_1to1((n1,None,None), (n2,None,None))[1] # travel time

    def set_zonal_veh_plan_schedule(self, vid, sim_time, start_time, x_max, x_min):
        """
        Set the schedule for the zonal vehicle plan in the next cycle from start_time.
        :param vid: vehicle id
        :type vid: int
        :param sim_time: simulation time
        :type sim_time: float
        :param start_time: start time of the next cycle
        :type start_time: float
        :param x_max: max km run for the vehicle
        :type x_max: float
        :param x_min: min km run for the vehicle
        :type x_min: float
        """

        assert x_min >= self.fixed_length  # check if x_min is in the flexible route
        assert x_max >= x_min  # check if x_max is greater than x_min

        # TODO: currently assume all zonal routes serve fixed route first
        # 1. remove previous schedule block
        if len(self.pt_fleetcontrol_module.veh_plans[vid].list_plan_stops) != 1:  # check if the plan has only one stop
            LOG.error(f"More stops than 1: {sim_time}: vid {vid} with list plan : {self.pt_fleetcontrol_module.veh_plans[vid]}")
        if len(self.pt_fleetcontrol_module.veh_plans[vid].list_plan_stops) > 0:
            last_planstop = self.pt_fleetcontrol_module.veh_plans[vid].list_plan_stops[-1]
        # last_veh_time = last_planstop._latest_start_time
        #     if not last_planstop.locked:  # check if the last plan stop is locked
            if last_planstop.direct_earliest_end_time != self.sim_end_time:
                LOG.error(f"{sim_time}: vid {vid} with list plan not ending with lock: {self.pt_fleetcontrol_module.veh_plans[vid]}")
            else:
                self.pt_fleetcontrol_module.veh_plans[vid].delete_plan_stop(last_planstop, self.sim_vehicles[vid], sim_time, self.routing_engine)

        # 2. calculate scheduled time from stops (between x_min and x_max)
        fixed_route_stop = self.find_closest_station_to_x(self.fixed_length, False)
        fixed_route_time = self.run_schedule.loc[self.run_schedule["station_id"] == fixed_route_stop, "departure"].values[0]
        fixed_route_node = self.pt_fleetcontrol_module.station_dict[fixed_route_stop].street_network_node_id
        fixed_route_return_time = self.run_schedule["departure"].max() - fixed_route_time

        # find non-stop network travel time from fixed_route_stop to x_min
        # TODO: future extension that the x_min_stop is not necessarily an existing stop
        x_min_stop = self.find_closest_station_to_x(x_min, False)
        x_min_node = self.pt_fleetcontrol_module.station_dict[x_min_stop].street_network_node_id
        fixed_route_to_x_min_time = self.get_travel_time_between_nodes(fixed_route_node, x_min_node)

        x_max_stop = self.find_closest_station_to_x(x_max, False)
        x_max_node = self.pt_fleetcontrol_module.station_dict[x_max_stop].street_network_node_id
        x_min_to_x_max_time = self.get_travel_time_between_nodes(x_min_node, x_max_node)

        # 3. set the schedule
        new_schedule = self.run_schedule.copy()

        # copy the schedule in the fixed route
        new_schedule = new_schedule.loc[new_schedule["departure"] <= fixed_route_time]
        # add x_min_stop with departure time = fixed_route_time + fixed_route_to_x_min_time
        # new_schedule = new_schedule.append(
        #     {"station_id": x_min_stop, "departure": fixed_route_time + fixed_route_to_x_min_time}
        #     , ignore_index=True
        # )
        new_flex_route_start_time = fixed_route_time + fixed_route_to_x_min_time
        new_flex_route_end_time = (new_flex_route_start_time +
                                   max(x_min_to_x_max_time * 2 * self.flex_detour, self.pt_zone_min_detour_time))
        new_fixed_route_return_time = new_flex_route_end_time + fixed_route_to_x_min_time
        # add x_min_stop with departure time = fixed_route_time + fixed_route_to_x_min_time + x_min_to_x_max_time * detour_factor
        # new_schedule = new_schedule.append(
        #     {
        #         "station_id": x_min_stop
        #         , "departure": new_fixed_route_return_time
        #      }
        #     , ignore_index=True
        # )
        # add the schedule in the fixed route (return) with adjusted departure time
        return_schedule = self.run_schedule.copy()
        return_schedule = return_schedule.loc[return_schedule["departure"] >= fixed_route_return_time]

        return_schedule["departure"] += new_fixed_route_return_time - fixed_route_return_time
        # new_schedule = new_schedule.append(return_schedule, ignore_index=True)
        new_schedule = pd.concat([new_schedule, return_schedule], ignore_index=True)

        new_schedule["departure"] += start_time

        # 4. apply to vehicle plan
        list_plan_stops = []
        # list_plan_stops = self.pt_fleetcontrol_module.veh_plans[vid].list_plan_stops


        # add a schedule block before the new cycle
        station_id = self.terminus_id
        node_index = self.pt_fleetcontrol_module.station_dict[station_id].street_network_node_id
        if sim_time + 60 < start_time:
            list_plan_stops.append(PlanStop(
                self.routing_engine.return_node_position(node_index),
                latest_start_time=sim_time,
                earliest_end_time=start_time - 1,
                # duration=(scheduled_stop["departure"] - 1) - (last_veh_time + 60) - 1,  # 1s, set the boarding/alighting duration to be nominal,
                locked=True,
                # will not be overwritten by the insertion
                planstop_state=G_PLANSTOP_STATES.RESERVATION,
            ))

        # add each stop
        for _, scheduled_stop in new_schedule.iterrows():
            station_id = scheduled_stop["station_id"]
            node_index = self.pt_fleetcontrol_module.station_dict[station_id].street_network_node_id

            ps = PlanStop(self.routing_engine.return_node_position(node_index),
                          latest_start_time=scheduled_stop["departure"],
                          earliest_end_time=scheduled_stop["departure"] + 1,
                          # duration=1,  # 1s, set the boarding/alighting duration to be nominal,
                          # locked=False,
                          # will not be overwritten by the insertion
                          )
            list_plan_stops.append(ps)

        # 5. add the schedule block to the schedule after this cycle
        last_veh_time = new_schedule["departure"].max() + 60
        station_id = self.terminus_id
        node_index = self.pt_fleetcontrol_module.station_dict[station_id].street_network_node_id
        list_plan_stops.append(PlanStop(
            self.routing_engine.return_node_position(node_index),
            latest_start_time=last_veh_time,
            earliest_end_time=self.sim_end_time,
            # duration=(scheduled_stop["departure"] - 1) - (last_veh_time + 60) - 1,  # 1s, set the boarding/alighting duration to be nominal,
            locked=True,
            # locked_end=True,
            # will not be overwritten by the insertion
            planstop_state=G_PLANSTOP_STATES.RESERVATION,
        ))

        interplan = VehiclePlan(self.sim_vehicles[vid], sim_time, self.routing_engine, list_plan_stops)
        interplan.update_tt_and_check_plan(self.sim_vehicles[vid], sim_time, self.routing_engine, keep_feasible=True)
        LOG.debug(f"interplan: {interplan}")
        LOG.debug(f"interplan feas: {interplan.is_feasible()}")
        if not interplan.is_feasible():
            LOG.error(f"vid {vid} with list plan stops: {[str(x) for x in list_plan_stops[:10]]}")
            LOG.error(f"interplan: {interplan}")
            LOG.error(f"interplan feas: {interplan.is_feasible()}")
            exit()

        LOG.debug(f"Time {sim_time}: Assign new bus schedule vid {vid} with list plan stops: {[str(x) for x in list_plan_stops[:10]]}")
        self.pt_fleetcontrol_module.veh_plans[vid] = VehiclePlan(self.sim_vehicles[vid], sim_time, self.routing_engine, list_plan_stops)
        self.pt_fleetcontrol_module.veh_plans[vid].update_plan(self.sim_vehicles[vid], sim_time, self.routing_engine,
                                        keep_time_infeasible=True)
        self.pt_fleetcontrol_module.assign_vehicle_plan(
            self.sim_vehicles[vid], self.pt_fleetcontrol_module.veh_plans[vid], sim_time, force_assign=True)

        self.veh_flex_time[vid].append([start_time + new_flex_route_start_time, start_time + new_flex_route_end_time])

    def hold_veh_in_terminus(self, vid, start_time, sim_time):
        """ Hold the vehicle in the terminus for a certain period of time
        :param vid: vehicle id
        :type vid: int
        :param start_time: start time
        :type start_time: float
        :param end_time: end time
        :type end_time: float
        :param sim_time: simulation time
        :type sim_time: float
        """
        raise NotImplementedError

        # 1. remove previous schedule block
        # if len(self.pt_fleetcontrol_module.veh_plans[vid].list_plan_stops) != 1:  # check if the plan has only one stop
        #     LOG.error(f"{sim_time}: vid {vid} with list plan stops: {self.pt_fleetcontrol_module.veh_plans[vid].list_plan_stops}")
        # if len(self.pt_fleetcontrol_module.veh_plans[vid].list_plan_stops) > 0:
        #     last_planstop = self.pt_fleetcontrol_module.veh_plans[vid].list_plan_stops[-1]
        # # last_veh_time = last_planstop._latest_start_time
        #     assert last_planstop.locked  # check if the last plan stop is locked
        #
        #     self.pt_fleetcontrol_module.veh_plans[vid].delete_plan_stop(last_planstop, self.sim_vehicles[vid], sim_time, self.routing_engine)
        #
        # list_plan_stops = []
        # station_id = self.terminus_id
        # node_index = self.pt_fleetcontrol_module.station_dict[station_id].street_network_node_id
        # list_plan_stops.append(PlanStop(
        #     self.routing_engine.return_node_position(node_index),
        #     latest_start_time=start_time,
        #     earliest_end_time=self.sim_end_time-1,
        #     # duration=(scheduled_stop["departure"] - 1) - (last_veh_time + 60) - 1,  # 1s, set the boarding/alighting duration to be nominal,
        #     locked=True,
        #     # locked_end=True,
        #     # will not be overwritten by the insertion
        #     planstop_state=G_PLANSTOP_STATES.RESERVATION,
        # ))
        # self.pt_fleetcontrol_module.veh_plans[vid] = VehiclePlan(self.sim_vehicles[vid], sim_time, self.routing_engine,
        #                                                          list_plan_stops)
        # self.pt_fleetcontrol_module.veh_plans[vid].update_plan(self.sim_vehicles[vid], sim_time, self.routing_engine,
        #                                                        keep_time_infeasible=True)
        # self.pt_fleetcontrol_module.assign_vehicle_plan(self.sim_vehicles[vid],
        #                                                 self.pt_fleetcontrol_module.veh_plans[vid], sim_time)

    def return_rid_zone(self, rid) -> int:
        """
        Return the zone of the request based on origin;
        if both origin and destination are not in the same zone (> -1), return -1
        if both origin and destination are in the fixed zone, return None
        :param rid: request id
        :type rid: int
        :return: zone
        :rtype: int
        """
        rq = self.pt_fleetcontrol_module.rq_dict[rid]
        pu_zone = self.return_pos_zone(rq.o_pos)
        do_zone = self.return_pos_zone(rq.d_pos)
        if pu_zone == -1 and do_zone == -1:
            return None
        if pu_zone != do_zone and pu_zone != -1 and do_zone != -1:
            return -1
        return max(pu_zone, do_zone)


class SoDZonalBatchAssignmentFleetcontrol(SemiOnDemandBatchAssignmentFleetcontrol):
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

        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                         dir_names, op_charge_depot_infra, list_pub_charging_infra)

        # load schedule for zonal vehicles
        schedules = pd.read_csv(os.path.join(self.pt_data_dir, scenario_parameters[G_PT_SCHEDULE_F]))
        self.run_schedule = schedules.loc[schedules["trip_id"] == 0]
        # check if the starting departure is 0, otherwise raise error
        if self.run_schedule["departure"].iloc[0] != 0:
            raise ValueError("Starting departure time of the schedule should be 0")

        self.list_veh_in_terminus = {} # list of vehicles in terminus, for zonal control, =1 if in terminus; =-1 if processed

        # to update in receive_status_update, to check in time trigger

        # load demand distribution
        demand_dist_f = os.path.join(self.pt_data_dir, scenario_parameters.get(G_PT_DEMAND_DIST, 0))
        self.demand_dist_df = pd.read_csv(demand_dist_f)

        # Zonal control parameters
        self.n_zones = scenario_parameters.get(G_PT_N_ZONES, 1)
        self.zone_x = [] # to set in continue_init()
        self.n_reg_veh = scenario_parameters.get(G_PT_REG_VEH, 0)
        self.pt_zone_min_detour_time = scenario_parameters.get(G_PT_ZONE_MIN_DETOUR_TIME,0)

        self.regular_headway = scenario_parameters.get(G_PT_REG_HEADWAY, 0)
        self.zone_headway = scenario_parameters.get(G_PT_ZONAL_HEADWAY, 0)

        # zonal control state dataset
        self.zonal_control_state = {}
        self.last_zonal_dept = {}

        # zonal control reward
        self.rejected_rid_times = {} # dict of dict for rejected rid time for each vehicle
        self.recent_rid_times = {} # dict of dict for recent rid for each vehicle, storing alighting times, waiting & riding times
        self.recent_cum_veh_dist = {}
        self.last_cum_veh_dist = 0.0
        self.dict_to_board_rid = {} # list of rid awaiting to board
        self.dict_to_alight_rid = {} # list of rid awaiting to alight
        self.reward_time_window = scenario_parameters.get(G_PT_RL_REWARD_TIME_WINDOW, 0)
        self.reward_sat_demand = scenario_parameters.get(G_PT_RL_REWARD_SAT_DEMAND, 0)
        self.reward_wait_time = scenario_parameters.get(G_PT_RL_REWARD_WAIT_TIME, 0)
        self.reward_ride_time = scenario_parameters.get(G_PT_RL_REWARD_RIDE_TIME, 0)
        self.reward_veh_dist = scenario_parameters.get(G_PT_RL_REWARD_VEH_DIST, 0)
        self.cum_reward = 0.0

        # zonal control RL model
        self.zone_state = 3
        model_state = 4 # forecast, # of SAVs, requests to serve not in zones, # of SAVs available to deploy
        self.state_dim = self.n_zones * self.zone_state + model_state

        # zone_action = 3 # prob to assign a vehicle, left boundary, right boundary
        # model_action = 1 # prob to not assign a vehicle
        # self.action_dim = self.n_zones * zone_action + model_action
        self.action_dim = self.n_zones + 1

        # self.RL_model = SoDZonalControlRL(self.state_dim, self.action_dim)
        LOG.debug(f"RL model initialized with state dim {self.state_dim} and action dim {self.action_dim}")

        # load previous training data if available
        # RL_data_path = os.path.join(self.pt_data_dir, "model_checkpoint.pth")
        # if os.path.exists(RL_data_path):
        #     self.RL_model.load_model(RL_data_path)
        #     LOG.info(f"RL model loaded from {RL_data_path}")
        self.time_step = scenario_parameters.get(G_SIM_TIME_STEP, 1)

        self.RL_train_no_iter = scenario_parameters.get(G_PT_RL_TRAIN_ITER, 0)
        # numpy arrays to store for training
        # self.RL_state = np.zeros([self.state_dim, self.RL_train_no_iter], dtype=np.float32)
        # self.RL_action = np.zeros([self.RL_train_no_iter], dtype=np.float32)
        # self.RL_reward = np.zeros([self.RL_train_no_iter], dtype=np.float32)
        # self.RL_time = np.zeros([self.RL_train_no_iter], dtype=np.float32)
        # self.RL_log_prob = np.zeros([self.RL_train_no_iter], dtype=np.float32)
        # self.RL_state_values = np.zeros([self.RL_train_no_iter], dtype=np.float32)
        # self.RL_train_iter = 0 # current iterations after last training
        # self.RL_last_reward = 0.0

    def continue_init(self, sim_vehicle_objs, sim_start_time, sim_end_time):
        """
        this method continues initialization after simulation vehicles have been created in the fleetsimulation class
        :param sim_vehicle_objs: ordered list of sim_vehicle_objs
        :param sim_start_time: simulation start time
        """
        self.sim_time = sim_start_time
        self.sim_end_time = sim_end_time
        veh_obj_dict = {veh.vid: veh for veh in sim_vehicle_objs}
        for line, vid_to_schedule_dict in self.schedule_to_initialize.items():
            for vid in vid_to_schedule_dict.keys():
                self.pt_vehicle_to_line[vid] = line
            schedule_vehicles = {vid: veh_obj_dict[vid] for vid in vid_to_schedule_dict.keys()}
            self.PT_lines[line] = PtLineZonal(
                line, self, schedule_vehicles, vid_to_schedule_dict, sim_start_time, sim_end_time
            )
        LOG.info(f"SoD finish continue_init {len(self.PT_lines)}")



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
        super().receive_status_update(vid, simulation_time, list_finished_VRL, force_update)

        # check which vehicles are in terminus
        line_id = next(iter(self.PT_lines))
        terminus_id = self.PT_lines[line_id].terminus_id
        terminus_node = self.station_dict[terminus_id].street_network_node_id


        # check if the vehicle is at the terminus and has no passengers
        if self.sim_vehicles[vid].pos[0] == terminus_node and not self.sim_vehicles[vid].pax:
            # check if the vehicle is not listed in the terminus or if it has already ended the last schedule
            if (self.list_veh_in_terminus.get(vid) is None
                    and (not self.sim_vehicles[vid].assigned_route
                    or self.sim_vehicles[vid].assigned_route[0].earliest_end_time == self.sim_end_time)):
                self.list_veh_in_terminus[vid] = 1
        else:
            if self.list_veh_in_terminus.get(vid) is not None:
                del self.list_veh_in_terminus[vid]

        # TODO: check whether this matches the schedule; if not, raise an error

    def _create_user_offer(self, rq: PlanRequest, simulation_time: int,
                           assigned_vehicle_plan: VehiclePlan=None, offer_dict_without_plan: Dict={}):
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
            add_offer[G_OFFER_WALKING_DISTANCE_ORIGIN] = self.walking_dist_origin[rq.get_rid_struct()] * 1000 # in m
            add_offer[G_OFFER_WALKING_DISTANCE_DESTINATION] = self.walking_dist_destination[rq.get_rid_struct()] * 1000 # in m

            PT_lines = self.return_ptline()
            add_offer[G_OFFER_ZONAL_ORIGIN_ZONE] = PT_lines.return_pos_zone(rq.o_pos)
            add_offer[G_OFFER_ZONAL_DESTINATION_ZONE] = PT_lines.return_pos_zone(rq.d_pos)

            offer = TravellerOffer(rq.get_rid(), self.op_id, pu_time - rq.get_rq_time(), do_time - pu_time,
                                   int(rq.init_direct_td * self.dist_fare + self.base_fare),
                                   additional_parameters=add_offer)
            rq.set_service_offered(offer)

            self.dict_to_board_rid[rq.get_rid()] = 1
            self.dict_to_alight_rid[rq.get_rid()] = 1
        else:
            offer = self._create_rejection(rq, simulation_time)
            self.rejected_rid_times[rq.get_rid()] = simulation_time
        return offer

    def user_cancels_request(self, rid: Any, simulation_time: int):
        super().user_cancels_request(rid, simulation_time)
        self.rejected_rid_times[rid] = simulation_time

    def time_trigger(self, simulation_time : int, RL_action=None):
        """This method is used to perform time-triggered processes. These are split into the following:
        1) the optimization of the current assignments of requests
        2) other tasks (repositioning, charging, fleetsizing, pricing)

        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        # check whether reservation requests should be considered as immediate requests
        rids_to_reveal = self.reservation_module.reveal_requests_for_online_optimization(simulation_time)
        for rid in rids_to_reveal:
            LOG.debug(f"activate {rid} with for global optimisation at time {simulation_time}!")
            self._prq_from_reservation_to_immediate(rid, simulation_time)
        if RL_action is None:
            self._call_time_trigger_request_batch(simulation_time)
        else:
            return self._call_time_trigger_request_batch(simulation_time, RL_action=RL_action)
        self._call_time_trigger_additional_tasks(simulation_time)

    def _call_time_trigger_request_batch(self, simulation_time, RL_action=None):
        """ this function first triggers the upper level batch optimisation
        based on the optimisation solution offers to newly assigned requests are created in the second step with following logic:
        declined requests will receive an empty dict
        unassigned requests with a new assignment try in the next opt-step dont get an answer
        new assigned request will receive a non empty offer-dict

        a retry is only made, if "user_max_wait_time_2" is given

        every request as to answer to an (even empty) offer to be deleted from the system!

        :param simulation_time: current time in simulation
        :return: dictionary rid -> offer for each unassigned request, that will recieve an answer. (offer: dictionary with plan specific entries; empty if no offer can be made)
        :rtype: dict
        """
        # TODO: remove super() and alter the veh list to only flexible route for flexible demand; only fixed vehicles for fixed demand
        RidePoolingBatchOptimizationFleetControlBase._call_time_trigger_request_batch(self, simulation_time)

        # embed()
        # Zonal: make use of max_wait_time_2 to put decision on hold (in case new zonal vehicle plans assigned)
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
                        self.RPBO_Module.delete_request(rid)
                        _, earliest_pu, _ = prq.get_o_stop_info()
                        new_latest_pu = earliest_pu + self.max_wait_time_2
                        self.change_prq_time_constraints(simulation_time, rid, new_latest_pu)
                        self.RPBO_Module.add_new_request(rid, prq)
                    else:  # no retry, rid declined
                        self._create_user_offer(prq, simulation_time)
                        # self.rejected_rid_times[rid] = simulation_time
                else:
                    assigned_plan = self.veh_plans[assigned_vid]
                    self._create_user_offer(prq, simulation_time, assigned_vehicle_plan=assigned_plan)
            for rid in self.unassigned_requests_2.keys():  # check second try rids
                assigned_vid = self.rid_to_assigned_vid.get(rid, None)
                if rid not in self.rq_dict:
                    LOG.debug(f"rid {rid} no longer in rq_dict")
                    continue
                if assigned_vid is None:  # decline
                    prq = self.rq_dict[rid]
                    _, earliest_pu, _ = prq.get_o_stop_info()
                    new_latest_pu = earliest_pu + self.max_wait_time_2

                    if self.sim_time > new_latest_pu: # if max_wait_time_2 is exceeded, decline
                        self._create_user_offer(self.rq_dict[rid], simulation_time)
                        # self.rejected_rid_times[rid] = simulation_time
                    else: # otherwise, add it back to the list (to check in next iteration)
                        new_unassigned_requests_2[rid] = 1
                        self.RPBO_Module.add_new_request(rid, prq)
                else:
                    prq = self.rq_dict[rid]
                    assigned_plan = self.veh_plans[assigned_vid]
                    self._create_user_offer(prq, simulation_time, assigned_vehicle_plan=assigned_plan)
            self.unassigned_requests_1 = {}
            self.unassigned_requests_2 = new_unassigned_requests_2  # retry rids
            LOG.debug("end of opt:")
            LOG.debug("unassigned_requests_2 {}".format(self.unassigned_requests_2))
            LOG.debug("offers: {}".format(rid_to_offers))

        # Prepare dataset for zonal control
        # ---------------------------------
        PT_line = self.return_ptline()


        # convert action to zonal control action (vehicle assign prob, left boundary, right boundary) and model action (prob to not assign a vehicle)
        # zonal_veh_assign_prob = []
        # zonal_boundary_left = []
        # zonal_boundary_right = []
        # # no_veh_assign_prob = 0
        # for z in range(PT_line.n_zones):
        #     zonal_veh_assign_prob.append(action[z * 3])
        #     zonal_boundary_left.append(action[z * 3 + 1])
        #     zonal_boundary_right.append(action[z * 3 + 2])
        # zonal_veh_assign_prob = np.array(zonal_veh_assign_prob)
        # no_veh_assign_prob = action[-1]

        # Zonal control action
        # --------------------------

        # randomly send out vehicles first
        regular_headway = self.regular_headway
        # zone_headway = self.zone_headway


        LOG.debug(f"Time {simulation_time} Vehicles in terminus: {self.list_veh_in_terminus}")
        # send out regular vehicles
        z = -1
        if simulation_time >= self.last_zonal_dept.get(z, 0) + regular_headway:
            for vid in self.list_veh_in_terminus.keys():
                if PT_line.veh_zone_assignment[vid] == z and self.list_veh_in_terminus[vid] == 1:
                    LOG.debug(f"Time {simulation_time}: Zonal control: send out regular vehicle {vid}")
                    PT_line.set_zonal_veh_plan_schedule(vid, simulation_time, simulation_time + 300, PT_line.route_length, PT_line.fixed_length)
                    self.last_zonal_dept[z] = simulation_time
                    self.list_veh_in_terminus[vid] = -1
                    break

        # only send out 1 zonal vehicle per time period
        zonal_veh_deployed = None
        # send out zonal vehicles according to action
        for vid in self.list_veh_in_terminus.keys():
            if self.list_veh_in_terminus[vid] == 1 and PT_line.veh_zone_assignment[vid] != -1: # in terminus and not processed, and not regular vehicles
                # Pass state to DL model, get action
                # action, log_prob, state_value = self.RL_model.select_action(
                #     state_np)  # Implement this method in SoDZonalControlRL

                # yield state_np, reward, done, info

                # LOG.debug(f"Time {simulation_time}: Zonal control: state {state_np}, "
                #           f"reward {self.cum_reward - self.RL_last_reward}, action {action}, "
                #           f"log_prob {log_prob}, state_value {state_value}")

                # save state, reward, action, log_prob, mask to numpy arrays
                # self.RL_state[:, self.RL_train_iter] = state_np
                # self.RL_action[self.RL_train_iter] = action
                # self.RL_reward[self.RL_train_iter] = self.cum_reward - self.RL_last_reward
                # self.RL_last_reward = self.cum_reward
                # self.RL_time[self.RL_train_iter] = simulation_time / self.time_step
                # self.RL_log_prob[self.RL_train_iter] = log_prob
                # self.RL_state_values[self.RL_train_iter] = state_value
                #
                # self.RL_train_iter += 1

                # send to zone with max zonal_veh_assign_prob, unless < no_veh_assign_prob
                # max_prob = zonal_veh_assign_prob.max()
                # z = zonal_veh_assign_prob.argmax()
                z = RL_action
                # z = action

                if z < self.n_zones: # if the action is for zonal control
                # if max_prob > no_veh_assign_prob:
                    # TODO: to set a more sensible boundary variable
                    # zone_length = PT_line.zone_x_max[z] - PT_line.zone_x_min[z]
                    zonal_veh_deployed = z

                    new_zone_x_max = PT_line.zone_x_max[z]
                    new_zone_x_min = PT_line.zone_x_min[z]

                    # new_zone_x_max = PT_line.zone_x_max[z] + zonal_boundary_right[z] * zone_length
                    # new_zone_x_min = PT_line.zone_x_min[z] + zonal_boundary_left[z] * zone_length
                    #
                    # # swap the boundaries if they are not in the correct order
                    # if new_zone_x_max < new_zone_x_min:
                    #     new_zone_x_max, new_zone_x_min = new_zone_x_min, new_zone_x_max
                    #
                    # # bound the zone change
                    # new_zone_x_max = max(min(new_zone_x_max, PT_line.route_length), PT_line.fixed_length)
                    # new_zone_x_min = min(max(new_zone_x_min, PT_line.fixed_length), PT_line.route_length)

                    LOG.debug(f"Time {simulation_time}: Zonal control: send out zonal vehicle {vid} to zone {z} "
                              f"with boundaries {new_zone_x_min} and {new_zone_x_max}")

                    PT_line.set_zonal_veh_plan_schedule(vid, simulation_time, simulation_time + 300,
                                                        new_zone_x_max, new_zone_x_min)
                    self.list_veh_in_terminus[vid] = -1  # processed
                    # state_np[z * 2 + 1] += 1 # increase nos of SAVs assigned in the zone in the state variable

                break # only send out 1 vehicle per time period
                    # zonal_veh_assign_prob[z] = no_veh_assign_prob - 1  # to avoid sending the same vehicle to the same zone

                # retrain the model after a number of iterations
                # if self.RL_train_iter >= self.RL_train_no_iter:
                #     self.RL_train_iter = 0
                    # RL_masks = np.ones(self.RL_train_no_iter)

                    # reward is the short-term results - GAE will incorporate results from multiple time steps
                    # LOG.debug(f"Time {simulation_time}: Zonal control: training RL model with state {self.RL_state}, "
                    #           f"reward {self.RL_reward}, action {self.RL_action}, log_prob {self.RL_log_prob}, "
                    #           f"state_value {self.RL_state_values}")
                    # self.RL_model.ppo_update(self.RL_state.T, self.RL_action, self.RL_reward, self.RL_log_prob,
                    #                          self.RL_time)


        # TODO: prepare for each zone, unsatisfied demand & nos of vehicles assigned
        # self.zonal_control_state["simulation_time"] = simulation_time
        # self.zonal_control_state["unassigned_requests_no"] = len(self.unassigned_requests_2)
        self.zonal_control_state["demand_forecast"] = self.get_closest_demand_forecast_time(simulation_time)

        # for each unassigned_requests_2, count the number of requests in each zone if zonal express is needed
        self.zonal_control_state["unsatisfied_demand"] = [0] * PT_line.n_zones
        for z in range(PT_line.n_zones):
            self.zonal_control_state["unsatisfied_demand"][z] = 0
        for rid in self.unassigned_requests_2.keys():
            rq_zone = PT_line.return_rid_zone(rid)
            if rq_zone and rq_zone >= 0:
                self.zonal_control_state["unsatisfied_demand"][rq_zone] += 1

        # for each assigned vehicle, count the number of vehicles in each zone
        # self.zonal_control_state["nos_of_SAVs_assigned"] = [0] * PT_line.n_zones
        # for z in range(PT_line.n_zones):
        #     self.zonal_control_state["nos_of_SAVs_assigned"][z] = 0
        # for vid in PT_line.sim_vehicles.keys():
        #     if vid not in self.list_veh_in_terminus.keys():
        #         veh_zone = PT_line.veh_zone_assignment[vid]
        #         if veh_zone >= 0:
        #             self.zonal_control_state["nos_of_SAVs_assigned"][veh_zone] += 1

        # for each zone, evaluate the number of requests yet to pick up / drop off
        self.zonal_control_state["requests_to_serve"] = [0] * (PT_line.n_zones + 1)
        for rid in self.dict_to_board_rid.keys():
            rq_zone = PT_line.return_rid_zone(rid)
            if rq_zone and rq_zone >= 0:
                self.zonal_control_state["requests_to_serve"][rq_zone] += 1
            else:
                self.zonal_control_state["requests_to_serve"][-1] += 1
        for rid in self.dict_to_alight_rid.keys():
            rq_zone = PT_line.return_rid_zone(rid)
            if rq_zone and rq_zone >= 0:
                self.zonal_control_state["requests_to_serve"][rq_zone] += 1
            else:
                self.zonal_control_state["requests_to_serve"][-1] += 1

        # for each assigned vehicle, evaluate the amount of free time remaining in each zone
        self.zonal_control_state["SAV_free_time_left"] = [0] * PT_line.n_zones
        self.zonal_control_state["SAV_active"] = 0
        for vid in PT_line.sim_vehicles.keys():
            veh_zone = PT_line.veh_zone_assignment[vid]
            if veh_zone >= 0 and len(PT_line.veh_flex_time[vid]) > 0:
                veh_flex_start_time, veh_flex_end_time = PT_line.veh_flex_time[vid][-1]

                if len(PT_line.veh_flex_time[vid]) > 1:
                    assert PT_line.veh_flex_time[vid][-2][1] < simulation_time # the previous flex time should be less than the current time

                if veh_flex_end_time > simulation_time: # if vehicle assigned new schedule
                    self.zonal_control_state["SAV_free_time_left"][veh_zone] += veh_flex_end_time - max(simulation_time, veh_flex_start_time)

            # count if SAVs are active
            terminus_id = PT_line.terminus_id
            terminus_node = self.station_dict[terminus_id].street_network_node_id
            if vid not in self.list_veh_in_terminus.keys() or self.list_veh_in_terminus[vid] != 1:
                if (self.sim_vehicles[vid].pos[0] != terminus_node
                    and self.sim_vehicles[vid].assigned_route
                    and self.sim_vehicles[vid].assigned_route[0].earliest_end_time != self.sim_end_time):
                    self.zonal_control_state["SAV_active"] += 1

        n_SAV_available = 0
        for vid in self.list_veh_in_terminus.keys():
            if self.list_veh_in_terminus[vid] == 1 and PT_line.veh_zone_assignment[vid] != -1:
                n_SAV_available += 1


        # group state values into a numpy array
        state_np = np.zeros(self.state_dim)
        for z in range(PT_line.n_zones):
            state_np[z * self.zone_state] = self.zonal_control_state["unsatisfied_demand"][z]
            state_np[z * self.zone_state + 1] = self.zonal_control_state["SAV_free_time_left"][z]
            state_np[z * self.zone_state + 2] = self.zonal_control_state["requests_to_serve"][z]
        state_np[-1] = self.zonal_control_state["demand_forecast"]
        state_np[-2] = self.zonal_control_state["requests_to_serve"][-1]
        state_np[-3] = self.zonal_control_state["SAV_active"]
        state_np[-4] = n_SAV_available


        # Prepare reward for zonal control
        # --------------------------------
        # check self.rejected_rid_times for number of rejected rids, and remove old rids
        rejected_rid_number = len(self.rejected_rid_times)
        rid_to_del = []
        for rid in self.rejected_rid_times.keys():
            if simulation_time - self.rejected_rid_times[rid] >= self.reward_time_window:
                rid_to_del.append(rid)
        for rid in rid_to_del:
            del self.rejected_rid_times[rid]

        # check self.recent_rid_times for recent rid times
        # rider_number = len(self.recent_rid_times)
        total_wait_time = 0
        total_ride_time = 0
        rid_to_del = []
        for rid in self.recent_rid_times.keys():
            # remove old rid from recent_rid_times if exceeded time window
            if simulation_time - self.recent_rid_times[rid][G_PT_ZC_RID_SIM_TIME] >= self.reward_time_window:
                rid_to_del.append(rid)
        for rid in rid_to_del:
            del self.recent_rid_times[rid]

        rider_number = len(self.recent_rid_times)
        for rid in self.recent_rid_times.keys():
            total_wait_time += self.recent_rid_times[rid][G_PT_ZC_RID_WAIT_TIME]
            total_ride_time += self.recent_rid_times[rid][G_PT_ZC_RID_RIDE_TIME]


        # rej_prop = rejected_rid_number / (rider_number + rejected_rid_number + 1)  # +1 to avoid division by 0
        # if rider_number > 0:
        #     avg_wait_time = total_wait_time / rider_number
        #     avg_ride_time = total_ride_time / rider_number
        # else:
        #     avg_wait_time = self.regular_headway / 2
        #     avg_ride_time = 20 * 60  # 20 minutes



        # collect vehicle distances
        total_cum_veh_dist = 0.0
        for v in self.sim_vehicles:
            total_cum_veh_dist += v.cumulative_distance

        self.recent_cum_veh_dist[simulation_time] = total_cum_veh_dist - self.last_cum_veh_dist
        self.last_cum_veh_dist = total_cum_veh_dist

        veh_dist_inc = np.average(list(self.recent_cum_veh_dist.values()))
        # remove old veh_dist_inc from recent_cum_veh_dist if exceeded time window
        t_to_del = []
        for t in self.recent_cum_veh_dist.keys():
            if simulation_time - t >= self.reward_time_window:
                t_to_del.append(t)
        for t in t_to_del:
            del self.recent_cum_veh_dist[t]

        # combine values to reward
        reward = rider_number * self.reward_sat_demand \
            - total_wait_time * self.reward_wait_time \
            - total_ride_time * self.reward_ride_time \
            - veh_dist_inc * self.reward_veh_dist * self.reward_time_window / self.time_step
        # reward = - rejected_rid_number
        self.cum_reward += reward
        LOG.debug(f"Time {simulation_time}: Zonal control: reward {reward} with rider_number {rider_number}, "
                  f"total_wait_time {total_wait_time}, total_ride_time {total_ride_time}, veh_dist_inc {veh_dist_inc}")
        # reward = rej_prop * self.reward_sat_demand - \
        #             avg_wait_time * self.reward_wait_time - \
        #             avg_ride_time * self.reward_ride_time - \
        #             veh_dist_inc * self.reward_veh_dist
        # reward = -reward  # to convert from cost


        # for z in range(PT_line.n_zones):
        #     if simulation_time >= self.last_zonal_dept.get(z, 0) + zone_headway:
        #         LOG.debug(f"Time {simulation_time}: Zonal control: choosing vehicles for zone {z} from {self.list_veh_in_terminus}")
        #         zone_veh_done = False
        #         for vid in self.list_veh_in_terminus.keys():
        #             if PT_line.veh_zone_assignment[vid] == z and self.list_veh_in_terminus[vid] == 1: # in terminus and not processed
        #                 if zone_veh_done:
        #                     # PT_line.hold_veh_in_terminus(vid, simulation_time, simulation_time)
        #                     pass
        #                 else:
        #                     PT_line.set_zonal_veh_plan_schedule(vid, simulation_time, simulation_time + 300, PT_line.zone_x_max[z], PT_line.zone_x_min[z])
        #                     self.last_zonal_dept[z] = simulation_time
        #                     self.list_veh_in_terminus[vid] = -1  # processed
        #                     zone_veh_done = True

        done = self.sim_time + self.time_step >= self.sim_end_time
        if RL_action is not None:
            return state_np, reward, done, False, {}, zonal_veh_deployed

        # if simulation_time close to end of simulation by time step, save the RL model
        # if simulation_time + self.time_step >= self.sim_end_time:
        #     # training at the end of simulation
        #     if self.RL_train_iter > 10:
        #         self.RL_state = self.RL_state[:, :self.RL_train_iter]
        #         self.RL_action = self.RL_action[:self.RL_train_iter]
        #         self.RL_reward = self.RL_reward[:self.RL_train_iter]
        #         self.RL_time = self.RL_time[:self.RL_train_iter]
        #         self.RL_log_prob = self.RL_log_prob[:self.RL_train_iter]
        #         self.RL_state_values = self.RL_state_values[:self.RL_train_iter]
        #         # reward is the short-term results - GAE will incorporate results from multiple time steps
        #         LOG.debug(f"Time {simulation_time}: Zonal control: training RL model with state {self.RL_state}, "
        #                   f"reward {self.RL_reward}, action {self.RL_action}, log_prob {self.RL_log_prob}, "
        #                   f"state_value {self.RL_state_values}")
        #         self.RL_model.ppo_update(self.RL_state.T, self.RL_action, self.RL_reward, self.RL_log_prob,
        #                                  self.RL_time)
        #
        #     self.RL_model.save_model(os.path.join(self.pt_data_dir, "model_checkpoint.pth"))
        #     LOG.info(f"RL model saved to {os.path.join(self.pt_data_dir, 'model_checkpoint.pth')}")
        #
        #     LOG.info(f"RL model: cumulative reward {self.cum_reward}")
        #     # save the metric (cumulative reward) to a file (add to the last row)
        #     with open(os.path.join(self.pt_data_dir, "reward_metric.csv"), "a") as f:
        #         f.write(f"{self.cum_reward}\n")
        #
        #     # save average policy losses and value losses
        #     policy_losses, value_losses, discounted_rewards = self.RL_model.return_losses()
        #     avg_policy_loss = np.mean(policy_losses)
        #     avg_value_loss = np.mean(value_losses)
        #     avg_discounted_reward = np.mean(discounted_rewards)
        #     with open(os.path.join(self.pt_data_dir, "policy_loss_metric.csv"), "a") as f:
        #         f.write(f"{avg_policy_loss}\n")
        #     with open(os.path.join(self.pt_data_dir, "value_loss_metric.csv"), "a") as f:
        #         f.write(f"{avg_value_loss}\n")
        #     with open(os.path.join(self.pt_data_dir, "discounted_reward_metric.csv"), "a") as f:
        #         f.write(f"{avg_discounted_reward}\n")


    def get_closest_demand_forecast_time(self, simulation_time) -> float:
        """
        This function returns the closest demand forecast (from self.demand_dist_df) to the current simulation time
        :param simulation_time: current simulation time
        :type simulation_time: float
        :return: closest demand forecast
        :rtype: float
        """

        # self.demand_dist_df is a dataframe with columns "seconds" and "count", every 900s
        return self.demand_dist_df["count"].iloc[np.argmin(np.abs(self.demand_dist_df["seconds"] - simulation_time))]

    def return_ptline(self) -> PtLineZonal:
        """
        Dummy method to return the first PT line (to change when there are multiple lines)
        :return: PT line
        :rtype: PtLine
        """
        first_index = next(iter(self.PT_lines))
        return self.PT_lines[first_index]

    def acknowledge_boarding(self, rid : int, vid : int, simulation_time : int):
        """This method can trigger some database processes whenever a passenger is starting to board a vehicle.

        :param rid: request id
        :type rid: int
        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        super().acknowledge_boarding(rid, vid, simulation_time)

        del self.dict_to_board_rid[rid]

    def acknowledge_alighting(self, rid : int, vid : int, simulation_time : int):
        """This method can trigger some database processes whenever a passenger is finishing to alight a vehicle.

        :param rid: request id
        :type rid: int
        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        self.sim_time = simulation_time
        LOG.debug(f"acknowledge alighting {rid} from {vid} at {simulation_time}")
        self.RPBO_Module.set_database_in_case_of_alighting(rid, vid)

        # update rid times for zonal control reward
        self.recent_rid_times[rid] = {}
        self.recent_rid_times[rid][G_PT_ZC_RID_SIM_TIME] = simulation_time
        self.recent_rid_times[rid][G_PT_ZC_RID_WAIT_TIME] = self.rq_dict[rid].pu_time - self.rq_dict[rid].get_rq_time()
        self.recent_rid_times[rid][G_PT_ZC_RID_RIDE_TIME] = simulation_time - self.rq_dict[rid].pu_time

        del self.dict_to_alight_rid[rid]

        del self.rq_dict[rid]
        try:
            del self.rid_to_assigned_vid[rid]
        except KeyError:
            pass
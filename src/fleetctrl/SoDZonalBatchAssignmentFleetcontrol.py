# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
from src.fleetctrl.RidePoolingBatchOptimizationFleetControlBase import RidePoolingBatchOptimizationFleetControlBase
from src.fleetctrl.SemiOnDemandBatchAssignmentFleetcontrol import PTStation, PtLine, \
    SemiOnDemandBatchAssignmentFleetcontrol
from src.misc.globals import *
from typing import Dict, Any

# additional module imports (> requirements)
# ------------------------------------------
import numpy as np
import pandas as pd

# src imports
# -----------
from src.simulation.Offers import TravellerOffer
from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from src.fleetctrl.planning.PlanRequest import PlanRequest

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
LOG = logging.getLogger(__name__)
LARGE_INT = 100000


# -------------------------------------------------------------------------------------------------------------------- #
# functions
# ---------
def create_stations(columns):
    return PTStation(columns["station_id"], columns["network_node_index"])


INPUT_PARAMETERS_RidePoolingBatchAssignmentFleetcontrol = {
    "doc": """Semi-on-Demand Hybrid Route Batch assignment fleet control with reinforcement-learning-based zonal control 
    (by Max Ng in Apr 2024)
        reference RidePoolingBatchAssignmentFleetcontrol and LinebasedFleetControl
        ride pooling optimisation is called after every optimisation_time_step and offers are created 
        in the time_trigger function
        if "user_max_wait_time_2" is given:
            if the user couldnt be assigned in the first try, it will be considered again in the next opt-step 
            with this new max_waiting_time constraint
        if "user_offer_time_window" is given:
            after accepting an offer the pick-up time is constraint around the expected pick-up time 
            with an interval of the size of this parameter""",
    "inherit": "RidePoolingBatchOptimizationFleetControlBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


class PtLineZonal(PtLine):
    def __init__(self, line_id, pt_fleetcontrol_module, schedule_vehicles, vid_to_schedule_dict, sim_start_time,
                 sim_end_time):
        """
        :param line_id: Line ID
        :type line_id: int
        :param pt_fleetcontrol_module: reference to the fleet control module
        :type pt_fleetcontrol_module: SoDZonalBatchAssignmentFleetcontrol
        :param schedule_vehicles: vehicles to be scheduled
        :type schedule_vehicles: dict
        :param vid_to_schedule_dict: dictionary of vid to schedule
        :type vid_to_schedule_dict: dict
        :param sim_start_time: simulation start time
        :type sim_start_time: int
        :param sim_end_time: simulation end time
        :type sim_end_time: int
        """
        # call super().__init__() to load original SoD vehicles
        # vehicles are all locked at the last scheduled location
        super().__init__(line_id, pt_fleetcontrol_module, schedule_vehicles, vid_to_schedule_dict, sim_start_time,
                         sim_end_time)
        self.n_zones = None
        self.pt_zone_min_detour_time = None
        self.pt_zone_max_detour_time = None
        self.zone_x_min = None
        self.zone_x_max = None
        self.zone_boundary_adj = None
        self.zone_boundary_adj_step = 0.1
        self.veh_zone_assignment = None
        self.rid_zone_assignment = None

    def set_pt_zone(self, n_zones):
        # zonal setting parameters
        self.n_zones = n_zones

        self.pt_zone_min_detour_time = self.pt_fleetcontrol_module.pt_zone_min_detour_time
        self.pt_zone_max_detour_time = self.pt_fleetcontrol_module.pt_zone_max_detour_time

        zone_len = (self.route_length - self.fixed_length) / self.n_zones
        self.zone_x_min = [self.fixed_length + zone_len * i for i in range(self.n_zones)]
        self.zone_x_max = [self.fixed_length + zone_len * (i + 1) for i in range(self.n_zones)]
        # add in regular route
        self.zone_x_min += [self.fixed_length]
        self.zone_x_max += [self.route_length]

        self.zone_boundary_adj = [0] * (self.n_zones - 1)

        self.veh_zone_assignment = {}  # dict vid -> zone assignment; -1 for regular vehicles
        self.rid_zone_assignment = {}  # dict rid -> zone assignment; -1 for regular requests
        # if self.n_zones > 1:
        # for vid in self.sim_vehicles.keys():
        #     if vid < self.n_reg_veh:
        #         self.veh_zone_assignment[vid] = -1
        #     else:
        #         self.veh_zone_assignment[vid] = vid % self.n_zones

    def return_x_zone(self, x) -> int:
        """
        Return the zone of the x
        :param x: x-coord
        :type x: float
        :return: zone
        :rtype: int
        """
        # fully flexible route case: check if x is around the terminus
        if (self.fixed_length < min(self.station_id_km_run.items(), key=lambda y: y[1])[1]
                and x < min(self.station_id_km_run.items(), key=lambda y: y[1])[1] + G_PT_X_TOL):
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
        :type pos: tuple
        :return: zone
        :rtype: int
        """
        x_pos = self.return_pos_km_run(pos)
        return self.return_x_zone(x_pos)

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

    def return_rid_zone(self, rid) -> int | None:
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
        pu_zone = self.return_pos_zone(rq.get_o_stop_info()[0])
        do_zone = self.return_pos_zone(rq.get_d_stop_info()[0])
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

        ride pooling optimisation is called after every optimisation_time_step and offers are created
        in the time_trigger function
        if "user_max_wait_time_2" is given:
            if the user couldnt be assigned in the first try, it will be considered again in the next opt-step
            with this new max_waiting_time constraint
        if "user_offer_time_window" is given:
            after accepting an offer the pick-up time is constraint around the expected pick-up time with an interval
            of the size of this parameter

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
        :param op_charge_depot_infra: reference to a OperatorChargingAndDepotInfrastructure class (optional)
        (unique for each operator)
        :type op_charge_depot_infra: OperatorChargingAndDepotInfrastructure
        :param list_pub_charging_infra: list of PublicChargingInfrastructureOperator classes (optional)
        (accesible for all agents)
        :type list_pub_charging_infra: list of PublicChargingInfrastructureOperator
        """
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                         dir_names, op_charge_depot_infra, list_pub_charging_infra)

        self.pt_zone_min_detour_time = None
        self.pt_zone_max_detour_time = None

        # load demand distribution
        demand_dist_f = os.path.join(self.pt_data_dir, scenario_parameters.get(G_PT_DEMAND_DIST, 0))
        self.demand_dist_df = pd.read_csv(demand_dist_f)

        self.n_zones = None
        self.n_zones_max = None
        self.zone_x = None
        self.zonal_control_state = None
        self.rejected_rid_times = None
        self.recent_rid_times = None
        self.recent_cum_veh_dist = None
        self.last_cum_veh_dist = None
        self.dict_to_board_rid = None
        self.dict_to_alight_rid = None
        self.reward_time_window = None
        self.reward_sat_demand = None
        self.reward_wait_time = None
        self.reward_ride_time = None
        self.reward_veh_dist = None
        self.cum_reward = None
        self.zone_state = None
        self.state_dim = None
        self.action_dim = None
        self.veh_assign_zone = None
        self.assigned_zone_time_dict = None
        self.action_time_dict = None
        self.zone_SAV_assigned_no_time_dict = None
        self.state_time_df = None
        self.time_step = None
        self.RL_train_no_iter = None
        self.served_pax = None
        self.last_served_pax = None

        self.output_np = None

    def continue_init(self, sim_vehicle_objs: dict, sim_start_time: int, sim_end_time: int):
        """
        this method continues initialization after simulation vehicles have been created in the fleetsimulation class
        :param sim_vehicle_objs: ordered list of sim_vehicle_objs
        :param sim_start_time: simulation start time
        :param sim_end_time: simulation end time
        """
        self.sim_time = sim_start_time
        self.sim_end_time = sim_end_time

        self.regular_headway = self.scenario_parameters.get(G_PT_REG_HEADWAY, 0)
        self.n_reg_veh = self.scenario_parameters.get(G_PT_ZONE_N_REG_VEH, 0)

        veh_obj_dict = {veh.vid: veh for veh in sim_vehicle_objs}
        for line, vid_to_schedule_dict in self.schedule_to_initialize.items():
            for vid in vid_to_schedule_dict.keys():
                self.pt_vehicle_to_line[vid] = line
            schedule_vehicles = {vid: veh_obj_dict[vid] for vid in vid_to_schedule_dict.keys()}
            self.PT_lines[line] = PtLineZonal(
                line, self, schedule_vehicles, vid_to_schedule_dict, sim_start_time, sim_end_time
            )
        LOG.info(f"SoDZonal finish continue_init {len(self.PT_lines)}")

        # Zonal control parameters
        pt_line = self.return_ptline()

        scenario_parameters = self.scenario_parameters
        self.pt_zone_min_detour_time = scenario_parameters.get(G_PT_ZONE_MIN_DETOUR_TIME, 0)
        self.pt_zone_max_detour_time = scenario_parameters.get(G_PT_ZONE_MAX_DETOUR_TIME, 0)
        self.n_zones = scenario_parameters.get(G_PT_N_ZONES, 1)

        self.n_zones_max = self.n_zones
        if self.fixed_length >= pt_line.route_length:
            self.n_zones_max = 1

        pt_line.set_pt_zone(self.n_zones)

        self.zone_x = []  # to set in continue_init()
        # self.n_reg_veh = scenario_parameters.get(G_PT_REG_VEH, 0)

        # self.regular_headway = scenario_parameters.get(G_PT_REG_HEADWAY, 0)
        # self.zone_headway = scenario_parameters.get(G_PT_ZONAL_HEADWAY, 0)

        # zonal control state dataset
        self.zonal_control_state = {}

        # zonal control reward
        self.rejected_rid_times = {}  # dict of dict for rejected rid time for each vehicle
        self.recent_rid_times = {}
        # dict of dict for recent rid for each vehicle, storing alighting times, waiting & riding times
        self.recent_cum_veh_dist = {}
        self.last_cum_veh_dist = 0.0
        self.dict_to_board_rid = {}  # list of rid awaiting to board
        self.dict_to_alight_rid = {}  # list of rid awaiting to alight
        self.reward_time_window = scenario_parameters.get(G_PT_RL_REWARD_TIME_WINDOW, 0)
        self.reward_sat_demand = scenario_parameters.get(G_PT_RL_REWARD_SAT_DEMAND, 0)
        self.reward_wait_time = scenario_parameters.get(G_PT_RL_REWARD_WAIT_TIME, 0)
        self.reward_ride_time = scenario_parameters.get(G_PT_RL_REWARD_RIDE_TIME, 0)
        self.reward_veh_dist = scenario_parameters.get(G_PT_RL_REWARD_VEH_DIST, 0)
        self.cum_reward = 0.0

        # zonal control RL model
        # TODO: read from scenario_parameters instead of hardcoding
        self.zone_state = 5
        model_state = 4  # forecast, # of SAVs, requests to serve not in zones, # of SAVs available to deploy
        # model_state = 8 - 1  # forecast, # of SAVs, requests to serve not in zones, # of SAVs available to deploy
        self.state_dim = (self.n_zones + 1) * self.zone_state + model_state

        # zone_action = 3 # prob to assign a vehicle, left boundary, right boundary
        # model_action = 1 # prob to not assign a vehicle
        # self.action_dim = self.n_zones * zone_action + model_action
        self.action_dim = self.n_zones + 1 + 1  # zones, do nothing, regular route

        self.veh_assign_zone = {}  # dict of dict for assigned zone for each vehicle
        self.assigned_zone_time_dict = {}  # dict of dict for assigned zone at each time
        self.action_time_dict = {}  # dict of dict for action at each time
        self.zone_SAV_assigned_no_time_dict = {}  # dict of dict of numbers of assigned SAV at each time
        self.state_time_df = None

        # self.RL_model = SoDZonalControlRL(self.state_dim, self.action_dim)
        LOG.debug(f"RL model initialized with state dim {self.state_dim} and action dim {self.action_dim}")

        # load previous training data if available
        # RL_data_path = os.path.join(self.pt_data_dir, "model_checkpoint.pth")
        # if os.path.exists(RL_data_path):
        #     self.RL_model.load_model(RL_data_path)
        #     LOG.info(f"RL model loaded from {RL_data_path}")
        self.time_step = scenario_parameters.get(G_SIM_TIME_STEP, 1)

        self.RL_train_no_iter = scenario_parameters.get(G_PT_RL_TRAIN_ITER, 0)

        self.last_zonal_dept = np.array([sim_start_time] * (self.n_zones + 1))
        # numpy arrays to store for training
        # self.RL_state = np.zeros([self.state_dim, self.RL_train_no_iter], dtype=np.float32)
        # self.rl_action = np.zeros([self.RL_train_no_iter], dtype=np.float32)
        # self.RL_reward = np.zeros([self.RL_train_no_iter], dtype=np.float32)
        # self.RL_time = np.zeros([self.RL_train_no_iter], dtype=np.float32)
        # self.RL_log_prob = np.zeros([self.RL_train_no_iter], dtype=np.float32)
        # self.RL_state_values = np.zeros([self.RL_train_no_iter], dtype=np.float32)
        # self.RL_train_iter = 0 # current iterations after last training
        # self.RL_last_reward = 0.0
        self.served_pax = 0
        self.last_served_pax = 0

    def _create_user_offer(self, rq: PlanRequest, simulation_time: int,
                           assigned_vehicle_plan: VehiclePlan = None, offer_dict_without_plan: Dict = {}):
        """
        creating the offer for a requests, tracking the request in the dict_to_board_rid and dict_to_alight_rid

        :param rq: plan request
        :type rq: PlanRequest obj
        :param simulation_time: current simulation time
        :type simulation_time: int
        :param assigned_vehicle_plan: vehicle plan of initial solution to serve this request
        :type assigned_vehicle_plan: VehiclePlan None
        :param offer_dict_without_plan: can be used to create an offer that is not derived from a vehicle plan
                    entries will be used to create/extend offer
        :type offer_dict_without_plan: dict or None
        :return: offer for request
        :rtype: TravellerOffer
        """
        if assigned_vehicle_plan is not None:
            pu_time, do_time = assigned_vehicle_plan.pax_info.get(rq.get_rid_struct())
            add_offer = {}
            pu_offer_tuple = self._get_offered_time_interval(rq.get_rid_struct())
            if pu_offer_tuple is not None:
                new_earliest_pu, new_latest_pu = pu_offer_tuple
                add_offer[G_OFFER_PU_INT_START] = new_earliest_pu
                add_offer[G_OFFER_PU_INT_END] = new_latest_pu

            # additional info for output here, e.g., fixed/flexible, access time
            add_offer[G_OFFER_WALKING_DISTANCE_ORIGIN] = self.walking_dist_origin[rq.get_rid_struct()] * 1000  # in m
            add_offer[G_OFFER_WALKING_DISTANCE_DESTINATION] = self.walking_dist_destination[
                                                                  rq.get_rid_struct()] * 1000  # in m

            pt_lines = self.return_ptline()
            add_offer[G_OFFER_ZONAL_ORIGIN_ZONE] = pt_lines.return_pos_zone(rq.get_o_stop_info()[0])
            add_offer[G_OFFER_ZONAL_DESTINATION_ZONE] = pt_lines.return_pos_zone(rq.get_d_stop_info()[0])
            pt_lines.rid_zone_assignment[rq.get_rid()] = pt_lines.return_rid_zone(rq.get_rid())

            offer = TravellerOffer(rq.get_rid(), self.op_id, pu_time - rq.get_rq_time(), do_time - pu_time,
                                   int(rq.init_direct_td * self.dist_fare + self.base_fare),
                                   additional_parameters=add_offer)
            rq.set_service_offered(offer)

            self.dict_to_board_rid[rq.get_rid()] = 1
            self.dict_to_alight_rid[rq.get_rid()] = 1
            self.served_pax += 1
        else:
            if rq.o_pos != rq.d_pos:  # check if not auto rejection
                self.rejected_rid_times[rq.get_rid()] = simulation_time
            offer = self._create_rejection(rq, simulation_time)
        return offer

    def user_cancels_request(self, rid: Any, simulation_time: int):
        if rid in self.rq_dict.keys():  # check if not auto rejection
            self.rejected_rid_times[rid] = simulation_time
            pt_lines = self.return_ptline()
            pt_lines.rid_zone_assignment[rid] = pt_lines.return_rid_zone(rid)
        super().user_cancels_request(rid, simulation_time)
        if rid in self.dict_to_board_rid:
            del self.dict_to_board_rid[rid]
        if rid in self.dict_to_alight_rid:
            del self.dict_to_alight_rid[rid]

    def time_trigger(self, simulation_time: int, rl_action=None):
        """This method is used to perform time-triggered processes. These are split into the following:
        1) the optimization of the current assignments of requests
        2) other tasks (repositioning, charging, fleetsizing, pricing)

        :param simulation_time: current simulation time
        :type simulation_time: float
        :param rl_action: action from the reinforcement learning model
        :type rl_action: np.ndarray
        """
        # check whether reservation requests should be considered as immediate requests
        rids_to_reveal = self.reservation_module.reveal_requests_for_online_optimization(simulation_time)
        for rid in rids_to_reveal:
            LOG.debug(f"activate {rid} with for global optimisation at time {simulation_time}!")
            self._prq_from_reservation_to_immediate(rid, simulation_time)
        if rl_action is None:
            self._call_time_trigger_request_batch(simulation_time)
        else:
            rl_var = self._call_time_trigger_request_batch(simulation_time, rl_action=rl_action)
        self._call_time_trigger_additional_tasks(simulation_time)

        if rl_action is not None:
            return rl_var

    def _call_time_trigger_request_batch(self, simulation_time, rl_action=None):
        """ this function first triggers the upper level batch optimisation
        based on the optimisation solution offers to newly assigned requests are created in the second step
        with following logic:
        declined requests will receive an empty dict
        unassigned requests with a new assignment try in the next opt-step dont get an answer
        new assigned request will receive a non-empty offer-dict

        a retry is only made, if "user_max_wait_time_2" is given

        every request as to answer to an (even empty) offer to be deleted from the system!

        :param simulation_time: current time in simulation
        :return: dictionary rid -> offer for each unassigned request, that will recieve an answer.
        (offer: dictionary with plan specific entries; empty if no offer can be made)
        :rtype: dict
        """
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
                    if self.max_wait_time_2 is not None and self.max_wait_time_2 > 0:
                        # retry with new waiting time constraint (no offer returned)
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

                    if self.sim_time > new_latest_pu:  # if max_wait_time_2 is exceeded, decline
                        self._create_user_offer(self.rq_dict[rid], simulation_time)
                        # self.rejected_rid_times[rid] = simulation_time
                    else:  # otherwise, add it back to the list (to check in next iteration)
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
        pt_line = self.return_ptline()

        # convert action to zonal control action (vehicle assign prob, left boundary, right boundary)
        # and model action (prob to not assign a vehicle)
        # zonal_veh_assign_prob = []
        # zonal_boundary_left = []
        # zonal_boundary_right = []
        # # no_veh_assign_prob = 0
        # for z in range(pt_line.n_zones):
        #     zonal_veh_assign_prob.append(action[z * 3])
        #     zonal_boundary_left.append(action[z * 3 + 1])
        #     zonal_boundary_right.append(action[z * 3 + 2])
        # zonal_veh_assign_prob = np.array(zonal_veh_assign_prob)
        # no_veh_assign_prob = action[-1]

        # Zonal control action
        # --------------------------
        # randomly send out vehicles first
        # regular_headway = self.regular_headway
        # zone_headway = self.zone_headway

        LOG.debug(f"Time {simulation_time} Vehicles in terminus: {self.list_veh_in_terminus}")
        # send out regular vehicles
        # z = -1
        # if simulation_time >= self.last_zonal_dept[z] + regular_headway:
        #     for vid in self.list_veh_in_terminus.keys():
        #         if pt_line.veh_zone_assignment[vid] == z and self.list_veh_in_terminus[vid] == 1:
        #             LOG.debug(f"Time {simulation_time}: Zonal control: send out regular vehicle {vid}")
        #             pt_line.set_zonal_veh_plan_schedule(vid, simulation_time, simulation_time + 300,
        #             pt_line.route_length, pt_line.fixed_length)
        #             self.last_zonal_dept[z] = simulation_time
        #             self.list_veh_in_terminus[vid] = -1
        #             break

        # only send out 1 zonal vehicle per time period
        zonal_veh_deployed = None
        # send out zonal vehicles according to action
        LOG.debug(f"Time {simulation_time}: Zonal control: action {rl_action}")
        z = rl_action[0]  # zone to send the vehicle
        if z > self.n_zones:
            z = -1

        if z != self.n_zones and z >= self.n_zones_max:  # if z is out of range, set to the regular zone
            z = -1

        self.action_time_dict[simulation_time] = z

        zone_boundary_change = rl_action[1]
        # possible: also change boundary for state calculation

        # adjust zone boundaries based on rl_action
        if zone_boundary_change < (self.n_zones - 1) * 2:  # if not no change:
            zone_to_change = zone_boundary_change // 2

            # pt_line.zone_x_max[zone_to_change] \
            #     += pt_line.zone_boundary_adj_step * (zone_boundary_change % 2 * 2 - 1)
            # pt_line.zone_x_max[zone_to_change] = min(pt_line.zone_x_max[zone_to_change], pt_line.route_length)
            # pt_line.zone_x_max[zone_to_change] = max(pt_line.zone_x_max[zone_to_change], pt_line.fixed_length)
            #
            # pt_line.zone_x_min[zone_to_change+1] = pt_line.zone_x_max[zone_to_change]

            pt_line.zone_boundary_adj[zone_to_change] += (
                    pt_line.zone_boundary_adj_step * (zone_boundary_change % 2 * 2 - 1))
            # pt_line.zone_boundary_adj[zone_to_change] = max(pt_line.zone_boundary_adj[zone_to_change], 0)
            # pt_line.zone_boundary_adj[zone_to_change] = min(pt_line.zone_boundary_adj[zone_to_change],
            #                                                 pt_line.route_length - pt_line.fixed_length)

        LOG.debug(f"Time {simulation_time}: Zonal control: boundaries {pt_line.zone_x_max}")

        # sort self.list_veh_in_terminus by keys
        self.list_veh_in_terminus = dict(sorted(self.list_veh_in_terminus.items()))

        # n_reg_veh = 4  # TODO: change to parameter
        for vid in self.list_veh_in_terminus.keys():
            # if self.list_veh_in_terminus[vid] == 1 and pt_line.veh_zone_assignment[vid] != -1:
            # in terminus and not processed, and not regular vehicles
            # if vid < n_reg_veh and z != -1:  # reserved vehicles for regular routes
            if vid < self.n_reg_veh and rl_action[0] != -1:  # reserved vehicles for regular routes
                continue

            if self.list_veh_in_terminus[vid] == 1:  # in terminus and not processed
                if z != self.n_zones:  # if the action is for zonal control
                    # Pass state to DL model, get action
                    # action, log_prob, state_value = self.RL_model.select_action(
                    #     state_np)  # Implement this method in SoDZonalControlRL

                    # yield state_np, reward, done, info

                    # LOG.debug(f"Time {simulation_time}: Zonal control: state {state_np}, "
                    #           f"reward {self.cum_reward - self.RL_last_reward}, action {action}, "
                    #           f"log_prob {log_prob}, state_value {state_value}")

                    # save state, reward, action, log_prob, mask to numpy arrays
                    # self.RL_state[:, self.RL_train_iter] = state_np
                    # self.rl_action[self.RL_train_iter] = action
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

                    # z = action

                    # if max_prob > no_veh_assign_prob:
                    # zone_length = pt_line.zone_x_max[z] - pt_line.zone_x_min[z]
                    # zonal_veh_deployed = z
                    #
                    new_zone_x_max = pt_line.zone_x_max[z]
                    new_zone_x_min = pt_line.zone_x_min[z]

                    # if z<self.n_zones-1:
                    #     new_zone_x_max = min(new_zone_x_max + pt_line.zone_boundary_adj[z], pt_line.route_length)
                    #     new_zone_x_max = max(new_zone_x_max, pt_line.fixed_length)
                    # if z>0:
                    #     new_zone_x_min = max(new_zone_x_min + pt_line.zone_boundary_adj[z-1], pt_line.fixed_length)
                    #     new_zone_x_min = min(new_zone_x_min, pt_line.route_length)

                    if new_zone_x_max < new_zone_x_min:
                        LOG.debug(f"Time {simulation_time}: Zonal control: zone {z} has negative length; skipped")
                        break

                    LOG.debug(f"Time {simulation_time}: Zonal control: send out zonal vehicle {vid} to zone {z} "
                              f"with boundaries {pt_line.zone_x_min[z]} and {pt_line.zone_x_max[z]}")

                    pt_line.set_veh_plan_schedule(vid, simulation_time, simulation_time + 300,
                                                  new_zone_x_min, new_zone_x_max,
                                                  self.pt_zone_min_detour_time, self.pt_zone_max_detour_time,
                                                  )

                    self.list_veh_in_terminus[vid] = -1  # processed
                    self.last_zonal_dept[z] = simulation_time
                    pt_line.veh_zone_assignment[vid] = z
                    self.assigned_zone_time_dict[simulation_time] = z
                    self.veh_assign_zone[vid] = z
                    # state_np[z * 2 + 1] += 1 # increase nos of SAVs assigned in the zone in the state variable

                    break  # only send out 1 vehicle per time period
                    # zonal_veh_assign_prob[z] = no_veh_assign_prob - 1
                    # to avoid sending the same vehicle to the same zone
                else:
                    self.assigned_zone_time_dict[simulation_time] = self.n_zones

        # RL_penalty = 0
        # if rl_action[0] < self.n_zones and simulation_time not in self.assigned_zone_time_dict.keys():
        # invalid action
        #     RL_penalty = 5

        # retrain the model after a number of iterations
        # if self.RL_train_iter >= self.RL_train_no_iter:
        #     self.RL_train_iter = 0
        # RL_masks = np.ones(self.RL_train_no_iter)

        # reward is the short-term results - GAE will incorporate results from multiple time steps
        # LOG.debug(f"Time {simulation_time}: Zonal control: training RL model with state {self.RL_state}, "
        #           f"reward {self.RL_reward}, action {self.rl_action}, log_prob {self.RL_log_prob}, "
        #           f"state_value {self.RL_state_values}")
        # self.RL_model.ppo_update(self.RL_state.T, self.rl_action, self.RL_reward, self.RL_log_prob,
        #                          self.RL_time)

        # Prepare reward for zonal control
        # --------------------------------
        # check self.rejected_rid_times for number of rejected rids, and remove old rids
        rejected_rid_number = len(self.rejected_rid_times)  # rejected_rid is added after this time trigger
        self.zonal_control_state["rejected_rid_number"] = [0] * (pt_line.n_zones + 1)
        rid_to_del = []
        last_rejected_rid_number = 0
        for rid in self.rejected_rid_times.keys():
            if simulation_time - self.rejected_rid_times[rid] >= self.reward_time_window:
                rid_to_del.append(rid)
            if simulation_time - self.rejected_rid_times[rid] <= self.time_step:
                z = pt_line.rid_zone_assignment[rid]
                if z is not None:
                    self.zonal_control_state["rejected_rid_number"][z] += 1
                else:
                    self.zonal_control_state["rejected_rid_number"][-1] += 1
            if simulation_time - self.rejected_rid_times[rid] <= self.reward_time_window:
                last_rejected_rid_number += 1
        for rid in rid_to_del:
            del self.rejected_rid_times[rid]

        # check self.recent_rid_times for recent rid times
        rider_number = len(self.recent_rid_times)
        total_wait_time = 0
        last_wait_time = 0
        total_ride_time = 0
        last_ride_time = 0
        rid_to_del = []
        for rid in self.recent_rid_times.keys():
            # remove old rid from recent_rid_times if exceeded time window
            if simulation_time - self.recent_rid_times[rid][G_PT_ZC_RID_SIM_TIME] >= self.reward_time_window:
                rid_to_del.append(rid)
            if simulation_time - self.recent_rid_times[rid][G_PT_ZC_RID_SIM_TIME] <= self.time_step:
                last_wait_time += self.recent_rid_times[rid][G_PT_ZC_RID_WAIT_TIME]
                last_ride_time += self.recent_rid_times[rid][G_PT_ZC_RID_RIDE_TIME]
        for rid in rid_to_del:
            del self.recent_rid_times[rid]

        # rider_number = len(self.recent_rid_times)
        for rid in self.recent_rid_times.keys():
            total_wait_time += self.recent_rid_times[rid][G_PT_ZC_RID_WAIT_TIME]
            total_ride_time += self.recent_rid_times[rid][G_PT_ZC_RID_RIDE_TIME]

        # rej_prop = rejected_rid_number / (rider_number + rejected_rid_number + 1)  # +1 to avoid division by 0
        if rider_number > 0:
            avg_wait_time = total_wait_time / rider_number
            avg_ride_time = total_ride_time / rider_number
        else:
            avg_wait_time = 0
            avg_ride_time = 0

        # collect vehicle distances
        total_cum_veh_dist = 0.0
        for v in self.sim_vehicles:
            total_cum_veh_dist += v.cumulative_distance

        self.recent_cum_veh_dist[simulation_time] = total_cum_veh_dist - self.last_cum_veh_dist
        self.last_cum_veh_dist = total_cum_veh_dist

        veh_dist_inc = np.average(list(self.recent_cum_veh_dist.values()))
        last_veh_dist_inc = 0
        # remove old veh_dist_inc from recent_cum_veh_dist if exceeded time window
        t_to_del = []
        for t in self.recent_cum_veh_dist.keys():
            if simulation_time - t >= self.reward_time_window:
                t_to_del.append(t)
            if simulation_time - t <= self.time_step:
                last_veh_dist_inc += self.recent_cum_veh_dist[t]
        for t in t_to_del:
            del self.recent_cum_veh_dist[t]

        # combine values to reward
        # reward = - last_rejected_rid_number * self.reward_sat_demand \
        #     - last_wait_time * self.reward_wait_time \
        #     - last_ride_time * self.reward_ride_time \
        #     - last_veh_dist_inc * self.reward_veh_dist
        # reward = - last_rejected_rid_number
        self.zonal_control_state["last_rejected_rid_number"] = last_rejected_rid_number
        # self.cum_reward += reward
        # LOG.debug(f"Time {simulation_time}: Zonal control: reward {reward} with last_rejected_rid_number {last_rejected_rid_number}, "
        #           f"last_wait_time {last_wait_time}, last_ride_time {last_ride_time}, last_veh_dist_inc {last_veh_dist_inc}")
        # reward = rej_prop * self.reward_sat_demand - \
        #             avg_wait_time * self.reward_wait_time - \
        #             avg_ride_time * self.reward_ride_time - \
        #             veh_dist_inc * self.reward_veh_dist
        # reward = -reward  # to convert from cost

        # TODO: prepare for each zone, unsatisfied demand & nos of vehicles assigned
        # self.zonal_control_state["simulation_time"] = simulation_time
        # self.zonal_control_state["unassigned_requests_no"] = len(self.unassigned_requests_2)
        self.zonal_control_state["demand_forecast"] = self.get_closest_demand_forecast_time(simulation_time)

        self.zonal_control_state["zone_boundary"] = [0] * (pt_line.n_zones - 1)
        for z in range(pt_line.n_zones - 1):
            # self.zonal_control_state["boundary_adj"][z] = pt_line.zone_boundary_adj[z]
            self.zonal_control_state["zone_boundary"][z] = pt_line.zone_x_max[z]

        # for each unassigned_requests_2, count the number of requests in each zone if zonal express is needed
        self.zonal_control_state["unsatisfied_demand"] = [0] * (pt_line.n_zones + 1)
        for z in range(pt_line.n_zones):
            self.zonal_control_state["unsatisfied_demand"][z] = 0
        for rid in self.unassigned_requests_2.keys():
            rq_zone = pt_line.return_rid_zone(rid)
            if rq_zone is not None:
                if rq_zone >= 0:
                    self.zonal_control_state["unsatisfied_demand"][rq_zone] += 1
                else:
                    self.zonal_control_state["unsatisfied_demand"][-1] += 1

        # for each assigned vehicle, count the number of vehicles in each zone
        # self.zonal_control_state["nos_of_SAVs_assigned"] = [0] * pt_line.n_zones
        # for z in range(pt_line.n_zones):
        #     self.zonal_control_state["nos_of_SAVs_assigned"][z] = 0
        # for vid in pt_line.sim_vehicles.keys():
        #     if vid not in self.list_veh_in_terminus.keys():
        #         veh_zone = pt_line.veh_zone_assignment[vid]
        #         if veh_zone >= 0:
        #             self.zonal_control_state["nos_of_SAVs_assigned"][veh_zone] += 1

        # for each zone, evaluate the number of requests yet to pick up / drop off
        self.zonal_control_state["requests_to_serve"] = [0] * (pt_line.n_zones + 1)
        for rid in self.dict_to_board_rid.keys():
            rq_zone = pt_line.return_rid_zone(rid)
            if rq_zone is not None:
                if rq_zone >= 0:
                    self.zonal_control_state["requests_to_serve"][rq_zone] += 1
                else:
                    self.zonal_control_state["requests_to_serve"][-1] += 1
        for rid in self.dict_to_alight_rid.keys():
            rq_zone = pt_line.return_rid_zone(rid)
            if rq_zone is not None:
                if rq_zone >= 0:
                    self.zonal_control_state["requests_to_serve"][rq_zone] += 1
                else:
                    self.zonal_control_state["requests_to_serve"][-1] += 1

        # for each assigned vehicle, evaluate the amount of free time remaining in each zone
        self.zonal_control_state["SAV_free_time_left"] = [0] * (pt_line.n_zones + 1)
        self.zonal_control_state["SAV_active"] = 0
        for vid in pt_line.sim_vehicles.keys():
            if vid not in pt_line.veh_zone_assignment.keys():
                continue
            veh_zone = pt_line.veh_zone_assignment[vid]
            # if veh_zone >= 0 and len(pt_line.veh_flex_time[vid]) > 0:
            if len(pt_line.veh_flex_time[vid]) > 0:
                veh_flex_start_time, veh_flex_end_time = pt_line.veh_flex_time[vid][-1]

                if len(pt_line.veh_flex_time[vid]) > 1:
                    if pt_line.veh_flex_time[vid][-2][1] >= simulation_time:
                        LOG.error(
                            f"Time {simulation_time}: Zonal control: vehicle {vid} has overlapping flex time, "
                            f"{pt_line.veh_flex_time[vid][-2][1]}")
                    assert pt_line.veh_flex_time[vid][-2][
                               1] < simulation_time  # the previous flex time should be less than the current time

                if veh_flex_end_time > simulation_time:  # if vehicle assigned new schedule
                    self.zonal_control_state["SAV_free_time_left"][veh_zone] \
                        += veh_flex_end_time - max(simulation_time, veh_flex_start_time)

            # count if SAVs are active
            terminus_id = pt_line.terminus_id
            terminus_node = self.station_dict[terminus_id].street_network_node_id
            if vid not in self.list_veh_in_terminus.keys() or self.list_veh_in_terminus[vid] != 1:
                if (self.sim_vehicles[vid].pos[0] != terminus_node
                        and self.sim_vehicles[vid].assigned_route
                        and self.sim_vehicles[vid].assigned_route[0].earliest_end_time != self.sim_end_time):
                    self.zonal_control_state["SAV_active"] += 1

        self.zonal_control_state["n_SAV_available"] = 0
        for vid in self.list_veh_in_terminus.keys():
            # if self.list_veh_in_terminus[vid] == 1 and pt_line.veh_zone_assignment[vid] != -1:
            if vid < self.n_reg_veh:  # skip regular vehicles
                continue
            if self.list_veh_in_terminus[vid] == 1:
                self.zonal_control_state["n_SAV_available"] += 1

        # add last departure time for each zone
        self.zonal_control_state["last_zonal_dept"] = [0] * (pt_line.n_zones + 1)
        for z in range(-1, pt_line.n_zones):
            self.zonal_control_state["last_zonal_dept"][z] = simulation_time - self.last_zonal_dept[z]

        # add average wait time and ride time
        self.zonal_control_state["avg_wait_time"] = avg_wait_time
        self.zonal_control_state["avg_ride_time"] = avg_ride_time
        # self.zonal_control_state["rejected_rid_number"] = rejected_rid_number
        self.zonal_control_state["veh_dist_inc"] = veh_dist_inc

        self.zonal_control_state["time_from_regular"] = simulation_time % self.regular_headway

        if self.state_time_df is None:
            self.state_time_df = pd.DataFrame(columns=list(self.zonal_control_state.keys()))
        self.state_time_df.loc[simulation_time] = self.zonal_control_state

        LOG.debug(f"Time {simulation_time}: Zonal control: state {self.zonal_control_state}")

        self.zonal_control_state_backup = self.zonal_control_state.copy()
        self.zonal_control_state = self.normalize_state(self.zonal_control_state)
        # group state values into a numpy array
        state_np = np.zeros(self.state_dim)
        for z in range(pt_line.n_zones + 1):
            state_np[z * self.zone_state] = self.zonal_control_state["unsatisfied_demand"][z]
            state_np[z * self.zone_state + 1] = self.zonal_control_state["SAV_free_time_left"][z]
            state_np[z * self.zone_state + 2] = self.zonal_control_state["requests_to_serve"][z]
            state_np[z * self.zone_state + 3] = self.zonal_control_state["last_zonal_dept"][z]
            state_np[z * self.zone_state + 4] = self.zonal_control_state["rejected_rid_number"][z]
            # if z >= 0 and z < pt_line.n_zones - 1:
            #     state_np[z * self.zone_state + 5] = self.zonal_control_state["zone_boundary"][z]
        state_np[-1] = self.zonal_control_state["demand_forecast"]
        # state_np[-2] = self.zonal_control_state["requests_to_serve"][-1]
        state_np[-2] = self.zonal_control_state["SAV_active"]
        state_np[-3] = self.zonal_control_state["n_SAV_available"]
        state_np[-4] = self.zonal_control_state["time_from_regular"]
        # state_np[-5] = self.zonal_control_state["avg_ride_time"]
        # state_np[-6] = self.zonal_control_state["avg_wait_time"]
        # state_np[-7] = self.zonal_control_state["veh_dist_inc"]

        # store nos of SAVs assigned to each zone
        sav_assigned_no_dict = {}
        for vid in pt_line.sim_vehicles.keys():
            # if self.list_veh_in_terminus[vid] == 1 and pt_line.veh_zone_assignment[vid] != -1:
            # in terminus and not processed, and not regular vehicles
            if vid not in self.list_veh_in_terminus.keys() or self.list_veh_in_terminus[vid] != 1:
                if vid in self.veh_assign_zone.keys():
                    z = int(self.veh_assign_zone[vid])
                    # if z is np array, extract first value

                    if z in sav_assigned_no_dict.keys():
                        sav_assigned_no_dict[z] += 1
                    else:
                        sav_assigned_no_dict[z] = 1

        self.zone_SAV_assigned_no_time_dict[simulation_time] = sav_assigned_no_dict

        # combine values to reward
        # reward = - last_rejected_rid_number * self.reward_sat_demand \
        #     - last_wait_time * self.reward_wait_time \
        #     - last_ride_time * self.reward_ride_time \
        #     - last_veh_dist_inc * self.reward_veh_dist
        # reward = - last_rejected_rid_number - RL_penalty
        reward = - last_rejected_rid_number

        # served_request_now = self.served_pax - self.last_served_pax
        # self.last_served_pax = self.served_pax
        # reward = served_request_now - RL_penalty

        self.cum_reward += reward
        LOG.debug(
            f"Time {simulation_time}: Zonal control: reward {reward} "
            f"with last_rejected_rid_number {last_rejected_rid_number}, "
            f"last_wait_time {last_wait_time}, last_ride_time {last_ride_time}, last_veh_dist_inc {last_veh_dist_inc}")

        # for z in range(pt_line.n_zones):
        #     if simulation_time >= self.last_zonal_dept.get(z, 0) + zone_headway:
        #         LOG.debug(f"Time {simulation_time}: Zonal control: choosing vehicles for zone {z} from {self.list_veh_in_terminus}")
        #         zone_veh_done = False
        #         for vid in self.list_veh_in_terminus.keys():
        #             if pt_line.veh_zone_assignment[vid] == z and self.list_veh_in_terminus[vid] == 1: # in terminus and not processed
        #                 if zone_veh_done:
        #                     # pt_line.hold_veh_in_terminus(vid, simulation_time, simulation_time)
        #                     pass
        #                 else:
        #                     pt_line.set_zonal_veh_plan_schedule(vid, simulation_time, simulation_time + 300, pt_line.zone_x_max[z], pt_line.zone_x_min[z])
        #                     self.last_zonal_dept[z] = simulation_time
        #                     self.list_veh_in_terminus[vid] = -1  # processed
        #                     zone_veh_done = True

        done = self.sim_time + self.time_step >= self.sim_end_time
        if rl_action is not None:
            return state_np, reward, done, False, {}, zonal_veh_deployed, self.zonal_control_state_backup[
                "n_SAV_available"]

        # if simulation_time close to end of simulation by time step, save the RL model
        # if simulation_time + self.time_step >= self.sim_end_time:
        #     # training at the end of simulation
        #     if self.RL_train_iter > 10:
        #         self.RL_state = self.RL_state[:, :self.RL_train_iter]
        #         self.rl_action = self.rl_action[:self.RL_train_iter]
        #         self.RL_reward = self.RL_reward[:self.RL_train_iter]
        #         self.RL_time = self.RL_time[:self.RL_train_iter]
        #         self.RL_log_prob = self.RL_log_prob[:self.RL_train_iter]
        #         self.RL_state_values = self.RL_state_values[:self.RL_train_iter]
        #         # reward is the short-term results - GAE will incorporate results from multiple time steps
        #         LOG.debug(f"Time {simulation_time}: Zonal control: training RL model with state {self.RL_state}, "
        #                   f"reward {self.RL_reward}, action {self.rl_action}, log_prob {self.RL_log_prob}, "
        #                   f"state_value {self.RL_state_values}")
        #         self.RL_model.ppo_update(self.RL_state.T, self.rl_action, self.RL_reward, self.RL_log_prob,
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

    def normalize_state(self, zonal_control_state):
        zonal_control_state["demand_forecast"] = (zonal_control_state["demand_forecast"] - 200) / 200
        zonal_control_state["zone_boundary"] = [x / 2 for x in zonal_control_state["zone_boundary"]]
        zonal_control_state["unsatisfied_demand"] = [(x - 5) / 5 for x in zonal_control_state["unsatisfied_demand"]]
        zonal_control_state["SAV_free_time_left"] = [(x - 2000) / 2000 for x in
                                                     zonal_control_state["SAV_free_time_left"]]
        zonal_control_state["requests_to_serve"] = [(x - 20) / 20 for x in zonal_control_state["requests_to_serve"]]
        zonal_control_state["SAV_active"] = (zonal_control_state["SAV_active"] - 5) / 5
        zonal_control_state["n_SAV_available"] = (zonal_control_state["n_SAV_available"] - 5) / 5
        zonal_control_state["last_zonal_dept"] = [(x - 1800) / 1800 for x in zonal_control_state["last_zonal_dept"]]
        zonal_control_state["avg_wait_time"] = (zonal_control_state["avg_wait_time"] - 600) / 600
        zonal_control_state["avg_ride_time"] = (zonal_control_state["avg_ride_time"] - 600) / 600
        zonal_control_state["rejected_rid_number"] = [(x - 5) / 5 for x in zonal_control_state["rejected_rid_number"]]
        zonal_control_state["veh_dist_inc"] = (zonal_control_state["veh_dist_inc"] - 5000) / 5000
        zonal_control_state["time_from_regular"] = (zonal_control_state["time_from_regular"] - 300) / 300
        return zonal_control_state

    def get_closest_demand_forecast_time(self, simulation_time) -> float:
        """
        This function returns the closest demand forecast (from self.demand_dist_df) to the current simulation time
        :param simulation_time: current simulation time
        :type simulation_time: float
        :return: the closest demand forecast
        :rtype: float
        """
        # self.demand_dist_df is a dataframe with columns "seconds" and "count", every 900s
        return self.demand_dist_df["count"].iloc[np.argmin(np.abs(self.demand_dist_df["seconds"] - simulation_time))]

    def return_ptline(self) -> PtLineZonal:
        """
        Dummy method to return the first PT line (to change when there are multiple lines)
        :return: PT line
        :rtype: PtLineZonal
        """
        first_index = next(iter(self.PT_lines))
        return self.PT_lines[first_index]

    def acknowledge_boarding(self, rid: int, vid: int, simulation_time: int):
        """
        This method can trigger some database processes whenever a passenger is starting to board a vehicle.
        :param rid: request id
        :type rid: int
        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        super().acknowledge_boarding(rid, vid, simulation_time)

        del self.dict_to_board_rid[rid]

    def acknowledge_alighting(self, rid: int, vid: int, simulation_time: int):
        """
        This method can trigger some database processes whenever a passenger is finishing to alight a vehicle.
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
        if rid in self.rid_to_assigned_vid.keys():
            del self.rid_to_assigned_vid[rid]

    def output_assigned_zone_time(self, output_file: str):
        """
        convert self.assigned_zone_time_dict and self.action_time_dict to numpy array and output as csv
        """
        self.output_np = np.zeros((len(self.action_time_dict), 3))
        i = 0
        for t in self.action_time_dict.keys():
            self.output_np[i, 0] = t
            self.output_np[i, 1] = self.action_time_dict[t]
            if t in self.assigned_zone_time_dict.keys():
                self.output_np[i, 2] = self.assigned_zone_time_dict[t]
            else:
                self.output_np[i, 2] = self.n_zones
            i += 1

        np.savetxt(output_file, self.output_np, delimiter=",")
        LOG.info(f"RL results saved to {output_file}")

    def output_no_sav_zone_assigned_time(self, output_file: str):
        """
        convert self.zone_SAV_assigned_no_time_dict to numpy array and output as csv
        """
        self.output_np = np.zeros((len(self.action_time_dict), 1 + self.n_zones + 1))
        i = 0
        for t in self.action_time_dict.keys():
            self.output_np[i, 0] = t
            if t in self.zone_SAV_assigned_no_time_dict.keys():
                for z in range(self.n_zones):
                    if z in self.zone_SAV_assigned_no_time_dict[t].keys():
                        self.output_np[i, z + 1] = self.zone_SAV_assigned_no_time_dict[t][z]
                if -1 in self.zone_SAV_assigned_no_time_dict[t].keys():
                    self.output_np[i, self.n_zones + 1] = self.zone_SAV_assigned_no_time_dict[t][-1]
            i += 1

        np.savetxt(output_file, self.output_np, delimiter=",")

    def output_state_df(self, output_file: str):
        """
        output state_time_df to a csv file
        """
        self.state_time_df.to_csv(output_file)

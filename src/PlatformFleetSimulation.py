# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import typing as tp

import logging
import importlib
import os
import json

import numpy as np
import pandas as pd

# additional module imports (> requirements)
# ------------------------------------------
# from IPython import embed

# src imports
# -----------

from src.FleetSimulationBase import FleetSimulationBase
from src.simulation.FreelancerSimulationVehicle import FreelancerSimulationVehicle
from src.misc.init_modules import load_fleet_control_module, load_routing_engine
from src.fleetctrl.FreelancerFleetControl import FreelancerFleetControl

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

if tp.TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.routing.NetworkBase import NetworkBase

INPUT_PARAMETERS_PlatformFleetSimulation = {
    "doc" : "this simulation class is used to simulate mod platforms; should later be included in FleetSimulationBase",
    "inherit" : "FleetSimulationBase",
    "input_parameters_mandatory": [G_PLAT_DRIVER_FILE],   # TODO requires G_AR_MAX_DEC_T == 0 (specify somehow?)
    "input_parameters_optional": [],
    "mandatory_modules": [], 
    "optional_modules": []
}


# -------------------------------------------------------------------------------------------------------------------- #
# functions
# ---------


# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class PlatformFleetSimulation(FleetSimulationBase):
    """
    this simulation class is used to simulate mod platforms; should later be included in FleetSimulationBase
    """

    def check_sim_env_spec_inputs(self, scenario_parameters):
        if scenario_parameters[G_AR_MAX_DEC_T] != 0:
            raise EnvironmentError(f"Scenario parameter {G_AR_MAX_DEC_T} has to be set to 0 for simulations in the "
                                   f"{self.__class__.__name__} environment!")

    def add_init(self, scenario_parameters):
        """
        Simulation specific additional init.
        :param scenario_parameters: row of pandas data-frame; entries are saved as x["key"]
        """
        for op_id, op in enumerate(self.operators):
            if op_id < len(self.list_op_dicts):
                operator_attributes = self.list_op_dicts[op_id]
                op.add_init(operator_attributes, self.scenario_parameters)

    def step(self, sim_time):
        """This method determines the simulation flow in a time step.
            # 1) update fleets and network
            # 2) get new travelers, add to undecided request
            # 3) sequential processes for each undecided request: request -> offer -> user-decision
            # 4) periodically for waiting requests: run decision process -> possibly leave system (cancellation)
            # 5) periodically operator: call ride pooling optimization, repositioning, charging management
            # 6) trigger charging infra 

        :param sim_time: new simulation time
        :return: None
        """
        # 1)
        self.update_sim_state_fleets(sim_time - self.time_step, sim_time)
        new_travel_times = self.routing_engine.update_network(sim_time)
        if new_travel_times:
            for op_id in range(self.n_op):
                self.operators[op_id].inform_network_travel_time_update(sim_time)
        # 2)
        list_undecided_travelers = list(self.demand.get_undecided_travelers(sim_time))
        last_time = sim_time - self.time_step
        if last_time < self.start_time:
            last_time = None
        list_new_traveler_rid_obj = self.demand.get_new_travelers(sim_time, since=last_time)
        # 3)
        for rid, rq_obj in list_undecided_travelers + list_new_traveler_rid_obj:
            for op_id in range(self.n_op):
                LOG.debug(f"Request {rid}: Checking AMoD option of operator {op_id} ...")
                # TODO # adapt fleet control
                self.operators[op_id].user_request(rq_obj, sim_time)
                amod_offer = self.operators[op_id].get_current_offer(rid)
                LOG.debug(f"amod offer {amod_offer}")
                if amod_offer is not None:
                    rq_obj.receive_offer(op_id, amod_offer, sim_time)
            self._rid_chooses_offer(rid, rq_obj, sim_time)
        # 4)
        self._check_waiting_request_cancellations(sim_time)
        # 5)
        for op in self.operators:
            op.time_trigger(sim_time)
        # 6)
        for ch_op_dict in self.charging_operator_dict.values():
            for ch_op in ch_op_dict.values():
                ch_op.time_trigger(sim_time)
        # record at the end of each time step
        self.record_stats()
        
    def evaluate(self):
        output_dir = self.dir_names[G_DIR_OUTPUT]
        from src.evaluation.eval_platform import eval_platform_scenario, add_user_trip_occupancies_to_stats
        eval_platform_scenario(output_dir)
        add_user_trip_occupancies_to_stats(output_dir)

    def add_evaluate(self):
        """Runs standard and simulation environment specific evaluations over simulation results."""
        # output_dir = self.dir_names[G_DIR_OUTPUT]
        # from src.evaluation.temporal import run_complete_temporal_evaluation
        # run_complete_temporal_evaluation(output_dir, method="snapshot")
        pass
    
    def _load_fleetctr_vehicles(self):
        """ Loads the fleet controller and vehicles """

        # simulation vehicles and fleet control modules
        LOG.info("Initialization of MoD fleets...")
        route_output_flag = self.scenario_parameters.get(G_SIM_ROUTE_OUT_FLAG, True)
        replay_flag = self.scenario_parameters.get(G_SIM_REPLAY_FLAG, False)
        veh_type_list = []
        # assume FreelancerFleetControl is not specified in config (will always be last operator)
        # all vehicles are initialized with all fleetcontrols
        freelancer_driver_file = self.scenario_parameters[G_PLAT_DRIVER_FILE]
        driver_df = pd.read_csv(os.path.join(self.dir_names[G_DIR_FCTRL], "freelancer_drivers", self.scenario_parameters[G_NETWORK_NAME], freelancer_driver_file))
        
        freelancer_op_id = self.n_op
        self.op_output = [[] for _ in range(self.n_op + 1)] # shared list among vehicles
        list_vehicles = []
        vid_to_start_pos = {}
        vid = 0
        for _, driver_row in driver_df.iterrows():
            driver_id = driver_row["driver_id"]
            veh_type = driver_row["veh_type"]
            possible_operators = driver_row["possible_operators"]
            possible_operators = [int(op_id) for op_id in possible_operators.split(";") if op_id != ""]
            start_node = driver_row.get("start_node")
            start_pos = None
            if start_node is not None:
                start_pos = (int(start_node), None, None)
                vid_to_start_pos[(freelancer_op_id, vid)] = start_pos
                LOG.warning("Init vehicle node currently not implemented!")
            operating_times = driver_row.get("operating_times")
            if operating_times is not None:
                operating_times = [int(t) for t in operating_times.split(";")]
            else:
                operating_times = []
            
            veh = FreelancerSimulationVehicle(freelancer_op_id, vid, self.dir_names[G_DIR_VEH], veh_type, self.routing_engine, 
                                              self.demand.rq_db, self.op_output[freelancer_op_id], route_output_flag, replay_flag, freelancer_op_id,
                                              possible_op_ids=possible_operators, operating_intervals=operating_times, driver_id=int(driver_id))
            list_vehicles.append(veh)
            self.sim_vehicles[(freelancer_op_id, vid)] = veh
            veh_type_list.append( [";".join([str(x) for x in possible_operators]), vid, veh_type])
            
            vid += 1
        # TODO here hard code op input?
        freelancer_attributes = {G_OP_VR_CTRL_F: {"func_key" : "total_travel_times"}, G_RA_REOPT_TS: 900}
        fl_repo_m = self.scenario_parameters.get(G_FL_REPO_M)
        if fl_repo_m:
            freelancer_attributes[G_OP_REPO_M] = fl_repo_m
            freelancer_attributes[G_OP_REPO_TS] = 900
            freelancer_attributes[G_OP_REPO_TH_DEF] = f"0;{self.scenario_parameters[G_SIM_END_TIME]}"
        #
        freelancer_op = FreelancerFleetControl(freelancer_op_id, freelancer_attributes, list_vehicles, self.routing_engine, self.zones,
                                    self.scenario_parameters, self.dir_names, self.charging_operator_dict["op"].get(freelancer_op_id, None), list(self.charging_operator_dict["pub"].values()))
        
        for op_id in range(self.n_op):
            operator_attributes = self.list_op_dicts[op_id]
            operator_module_name = operator_attributes[G_OP_MODULE]

            OpClass: FleetControlBase = load_fleet_control_module(operator_module_name)
            self.operators.append(OpClass(op_id, operator_attributes, list_vehicles, self.routing_engine, self.zones,
                                        self.scenario_parameters, self.dir_names, self.charging_operator_dict["op"].get(op_id, None), list(self.charging_operator_dict["pub"].values())))
        
        self.operators.append(freelancer_op)
                
        veh_type_f = os.path.join(self.dir_names[G_DIR_OUTPUT], "2_vehicle_types.csv")
        veh_type_df = pd.DataFrame(veh_type_list, columns=[G_V_OP_ID, G_V_VID, G_V_TYPE])
        veh_type_df.to_csv(veh_type_f, index=False)
        self.vehicle_update_order: tp.Dict[tp.Tuple[int, int], int] = {vid : 1 for vid in self.sim_vehicles.keys()}
        
    def update_sim_state_fleets(self, last_time, next_time, force_update_plan=False):
        """
        This method updates the simulation vehicles, records, ends and starts tasks and returns some data that
        will be used for additional state updates (fleet control information, demand, network, ...)
        
        needed updates: op_id are inferred from vehicle object
        
        :param last_time: simulation time before the state update
        :param next_time: simulation time of the state update
        :param force_update_plan: flag that can force vehicle plan to be updated
        """
        LOG.debug(f"updating MoD state from {last_time} to {next_time}")
        #for opid_vid_tuple, veh_obj in self.sim_vehicles.items():
        for vid_list_id, veh_obj in sorted(self.sim_vehicles.items(), key=lambda x:self.vehicle_update_order[x[0]]):
            active_op_id = veh_obj.op_id
            vid = veh_obj.vid
            boarding_requests, alighting_requests, passed_VRL, dict_start_alighting =\
                veh_obj.update_veh_state(last_time, next_time)
            if veh_obj.status == VRL_STATES.CHARGING:
                self.vehicle_update_order[vid_list_id] = 0
            else:
                self.vehicle_update_order[vid_list_id] = 1
            for rid, boarding_time_and_pos in boarding_requests.items():
                boarding_time, boarding_pos = boarding_time_and_pos
                LOG.debug(f"rid {rid} boarding at {boarding_time} at pos {boarding_pos}")
                self.demand.record_boarding(rid, vid, active_op_id, boarding_time, pu_pos=boarding_pos)
                self.operators[active_op_id].acknowledge_boarding(rid, vid, boarding_time)
            for rid, alighting_start_time_and_pos in dict_start_alighting.items():
                # record user stats at beginning of alighting process
                alighting_start_time, alighting_pos = alighting_start_time_and_pos
                LOG.debug(f"rid {rid} deboarding at {alighting_start_time} at pos {alighting_pos}")
                self.demand.record_alighting_start(rid, vid, active_op_id, alighting_start_time, do_pos=alighting_pos)
            for rid, alighting_end_time in alighting_requests.items():
                # # record user stats at end of alighting process
                self.demand.user_ends_alighting(rid, vid, active_op_id, alighting_end_time)
                self.operators[active_op_id].acknowledge_alighting(rid, vid, alighting_end_time)
            # send update to operator
            if len(boarding_requests) > 0 or len(dict_start_alighting) > 0:
                self.operators[active_op_id].receive_status_update(vid, next_time, passed_VRL, True)
            else:
                self.operators[active_op_id].receive_status_update(vid, next_time, passed_VRL, force_update_plan)
                
    def load_initial_state(self):
        """This method initializes the simulation vehicles. It can consider an initial state file. Moreover, an
        active_vehicle files would be considered as the FleetControl already set the positions of vehicles in the depot
        and therefore the "if veh_obj.pos is None:" condition does not trigger.
        The VehiclePlans of the respective FleetControls are also adapted for blocked vehicles.

        :return: None
        """
        init_f_flag = False
        init_state_f = None
        if self.scenario_parameters.get(G_INIT_STATE_SCENARIO):
            init_state_f = os.path.join(self.dir_names[G_DIR_MAIN], "studies",
                                        self.scenario_parameters[G_STUDY_NAME], "results",
                                        str(self.scenario_parameters.get(G_INIT_STATE_SCENARIO, "None")),
                                        "final_state.csv")
            init_f_flag = True
            if not os.path.isfile(init_state_f):
                raise FileNotFoundError(f"init state variable {G_INIT_STATE_SCENARIO} given but file {init_state_f} not found!")
        set_unassigned_vid = set([(veh_obj.op_id, veh_obj.vid) for veh_obj in self.sim_vehicles.values()
                                  if veh_obj.pos is None])
        if init_f_flag:
            # set according to initial state if available
            init_state_df = pd.read_csv(init_state_f)
            init_state_df.set_index([G_V_OP_ID, G_V_VID], inplace=True)
            for sim_vid, veh_obj in self.sim_vehicles.items():
                if veh_obj.pos is None:
                    op_fleetctrl = self.operators[veh_obj.op_id]
                    init_state_info = init_state_df.loc[sim_vid]
                    if init_state_info is not None:
                        veh_obj.set_initial_state(op_fleetctrl, self.routing_engine, init_state_info,
                                                  self.scenario_parameters[G_SIM_START_TIME], self.init_blocking)
                        set_unassigned_vid.remove(sim_vid)
        if len(set_unassigned_vid) > 0:
            op_init_distributions = {}
            boarding_nodes = self.routing_engine.get_must_stop_nodes()
            if not boarding_nodes:
                boarding_nodes = list(range(self.routing_engine.get_number_network_nodes()))
            op_init_distributions[len(self.operators)-1] = {bn : 1.0/len(boarding_nodes) for bn in boarding_nodes}
            #LOG.debug("init distributons: {}".format(op_init_distributions))
            for sim_vid in set_unassigned_vid:
                veh_obj = self.sim_vehicles[sim_vid]
                if veh_obj.pos is None:
                    op_fleetctrl = self.operators[veh_obj.op_id]
                    init_dist = op_init_distributions[veh_obj.op_id]
                    r = np.random.random()
                    s = 0.0
                    init_node = None
                    for n, prob in init_dist.items():
                        s += prob
                        if s >= r:
                            init_node = n
                            break
                    if init_node is None:
                        LOG.error(f"No init node found for random val {r} and init dist {init_dist}")
                    # randomly position all vehicles
                    init_state_info = {}
                    init_state_info[G_V_INIT_NODE] = init_node# np.random.choice(init_node)
                    init_state_info[G_V_INIT_TIME] = self.scenario_parameters[G_SIM_START_TIME]
                    init_state_info[G_V_INIT_SOC] = 0.5 * (1 + np.random.random())
                    veh_obj.set_initial_state(op_fleetctrl, self.routing_engine, init_state_info,
                                                self.scenario_parameters[G_SIM_START_TIME], self.init_blocking)
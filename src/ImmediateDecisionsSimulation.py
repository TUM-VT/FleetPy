# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import importlib
import os
import json
import pandas as pd

# additional module imports (> requirements)
# ------------------------------------------
# from IPython import embed

# src imports
# -----------

from src.FleetSimulationBase import FleetSimulationBase

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_ImmediateDecisionsSimulation = {
    "doc" : "in this simulation each request immediatly decides for or against an offer",
    "inherit" : "FleetSimulationBase",
    "input_parameters_mandatory": [],   # TODO requires G_AR_MAX_DEC_T == 0 (specify somehow?)
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
class ImmediateDecisionsSimulation(FleetSimulationBase):
    """
    Init main simulation module. Check the documentation for a flowchart of this particular simulation environment.
    Main attributes:
    - agent list per time step query public transport and fleet operator for offers and immediate decide
    - fleet operator offers ride pooling service
    - division of study area
        + first/last mile service in different parts of the study area
        + different parking costs/toll/subsidy in different parts of the study area
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
        super().add_init(scenario_parameters)
        self.list_vehicle_state_dict = []
        self.dict_dirs = get_directory_dict(scenario_parameters)
        self.dict_step = {}

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
            self._get_fleet_status_request(rid, rq_obj, sim_time)
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
        self._save_fleet_status(sim_time)

    def add_evaluate(self):
        """Runs standard and simulation environment specific evaluations over simulation results."""
        output_dir = self.dir_names[G_DIR_OUTPUT]
        # from src.evaluation.temporal import run_complete_temporal_evaluation
        # run_complete_temporal_evaluation(output_dir, method="snapshot")
        # from src.evaluation.standard import current_state_eval
        # current_state_eval(output_dir, self.dict_dirs)

    def _get_fleet_status(self, sim_time):
        """"Record the current state of the fleet. Anything regarding requests is not considered."""
        str_pos = 'Pos'
        str_l_dest = 'Last Destination'
        str_num_stops = 'Number of Stops'
        str_pax = 'Nr. Pax'
        str_cl_remaining_time = 'Remaining Time CL'
        str_last_time_op = 'Last Time OP'
        str_last_pos_op = 'Last Pos OP'

        sorted_sim_vehicle_keys = sorted(self.sim_vehicles.keys())

        for sim_vid in sorted_sim_vehicle_keys:
            self.list_vehicle_states.append(self.sim_vehicles[sim_vid].return_current_vehicle_state(str_pos, str_l_dest,
                                                                                                    str_num_stops,
                                                                                                    str_pax,
                                                                                                    str_cl_remaining_time))
            veh_obj = self.sim_vehicles[sim_vid]
            for op_id in range(self.n_op):
                last_time, last_pos, _ = self.operators[op_id].veh_plans[sim_vid[1]].return_after_locked_availability(veh_obj, sim_time)
                self.list_vehicle_states[-1][str_last_time_op] = last_time
                self.list_vehicle_states[-1][str_last_pos_op] = last_pos


    def _get_fleet_status_request(self, rid, rq_obj, sim_time):
        """Record the current fleet status regarding a specific request."""
        str_pos = 'Pos'
        str_l_dest = 'Last Destination'
        str_num_stops = 'Number of Stops'
        str_pax = 'Nr. Pax'
        str_cl_remaining_time = 'Remaining Time CL'
        str_vehicle_status = 'Current Vehicle Status'
        str_dist_start_start = 'Distance Current Position - Origin'
        str_dist_end_start =  'Distance End Position - Origin'
        str_dist_end_end = 'Distance End Position - Destination'
        str_last_time_op = 'Last Time OP'
        str_last_pos_op = 'Last Pos OP'
        str_time_until_free = 'Time Until Free'

        list_str_features = [G_V_OP_ID, G_V_VID, G_RQ_ID, str_pos, str_l_dest, str_num_stops, str_pax,
                             str_cl_remaining_time, str_vehicle_status, str_dist_start_start, str_dist_end_start,
                             str_dist_end_end, str_last_time_op, str_last_pos_op]

        sorted_sim_vehicle_keys = sorted(self.sim_vehicles.keys())

        dict_request = {feature: [] for feature in list_str_features}
        for sim_vid in sorted_sim_vehicle_keys:
            dict_request = self.sim_vehicles[sim_vid].return_current_vehicle_state(str_pos, str_l_dest, str_num_stops,
                                                                                   str_pax, str_cl_remaining_time,
                                                                                   dict_request)

            dict_request = self.sim_vehicles[sim_vid].return_current_vehicle_state_request(str_dist_start_start,
                                                                                           str_dist_end_start,
                                                                                           str_dist_end_end,
                                                                                           str_vehicle_status,
                                                                                           rid, rq_obj, dict_request)
            veh_obj = self.sim_vehicles[sim_vid]
            for op_id in range(self.n_op):
                last_time, last_pos, _ = self.operators[op_id].veh_plans[sim_vid[1]].return_after_locked_availability(veh_obj, sim_time)
                dict_request[str_last_time_op].append(last_time)
                dict_request[str_time_until_free].append(last_time-sim_time)
                dict_request[str_last_pos_op].append(int(last_pos[0]))

        self.dict_step[str(rid)] = dict_request


    def _save_fleet_status(self, sim_time):
        """Save previously recorded fleet states to external csv file"""
        path_current_state = os.path.join(self.dir_names[G_DIR_OUTPUT], "current_state")
        path_current_state_file = os.path.join(path_current_state, f'data_{sim_time}.json')

        if not os.path.isdir(path_current_state):
            os.makedirs(path_current_state)

        with open(path_current_state_file, "w") as f:
            f.write(json.dumps(self.dict_step), indent=4)

        self.dict_step = {}

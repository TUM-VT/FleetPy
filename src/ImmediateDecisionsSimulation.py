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
            self._get_fleet_status(rid, rq_obj, sim_time)
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
        self._save_fleet_status()

    def add_evaluate(self):
        """Runs standard and simulation environment specific evaluations over simulation results."""
        # output_dir = self.dir_names[G_DIR_OUTPUT]
        # from src.evaluation.temporal import run_complete_temporal_evaluation
        # run_complete_temporal_evaluation(output_dir, method="snapshot")
        pass

    def _get_fleet_status(self, rid, rq_obj, sim_time):
        """Record the current fleet status after """
        str_pos = 'Pos'
        str_l_dest = 'Last Destination'
        str_num_stops = 'Number of Stops'
        str_pax = 'Nr. Pax'
        str_rid = 'Request ID'
        str_cl_remaining_time = 'Remaining Time CL'
        str_vehicle_status = 'Current Vehicle Status'
        str_dist_start_start = 'Distance Current Position - Origin'
        str_dist_end_start =  'Distance End Position - Origin'
        str_dist_end_end = 'Distance End Position - Destination'

        sorted_sim_vehicle_keys = sorted(self.sim_vehicles.keys())

        list_vehicle_states = [self.sim_vehicles[sim_vid].return_current_vehicle_state(str_pos, str_l_dest,
                                                                                       str_num_stops, str_pax,
                                                                                       str_cl_remaining_time,
                                                                                       str_vehicle_status)
                               for sim_vid in sorted_sim_vehicle_keys]

        list_vehicle_request_states = [self.sim_vehicles[sim_vid].return_current_vehicle_state_request(str_dist_start_start,
                                                                                                       str_dist_end_start,
                                                                                                       str_dist_end_end,
                                                                                                       rid, rq_obj)
                                       for sim_vid in sorted_sim_vehicle_keys]

        list_complete_vehicle_state = [dict(i, **j) for i, j in
                                       zip(list_vehicle_states, list_vehicle_request_states)]
        dict_vehicles_states = {
            i[G_V_VID]: {str_rid: rid, str_pos: i[str_pos], str_l_dest: i[str_l_dest], str_pax: i[str_pax],
                         str_num_stops: i[str_num_stops],
                         str_dist_start_start: i[str_dist_start_start],
                         str_dist_end_start: i[str_dist_end_start],
                         str_dist_end_end: i[str_dist_end_end], str_cl_remaining_time:
                             i[str_cl_remaining_time], str_vehicle_status: i[str_vehicle_status]}
            for i in list_complete_vehicle_state}
        dict_vehicles_states['Sim Time'] = sim_time
        self.list_vehicle_state_dict.append(dict_vehicles_states)

    def _save_fleet_status(self):
        """Save previously recorded fleet states to external csv file"""
        path_current_state = os.path.join(self.dir_names[G_DIR_OUTPUT], "current_state.csv")

        df_current_DB = pd.DataFrame(self.list_vehicle_state_dict)
        self.list_vehicle_state_dict = []

        if os.path.isfile(path_current_state):
            write_mode, write_header = "a", False
        else:
            write_mode, write_header = "w", True
        df_current_DB.set_index('Sim Time', inplace=True)
        df_current_DB.to_csv(path_current_state, mode=write_mode, header=write_header)

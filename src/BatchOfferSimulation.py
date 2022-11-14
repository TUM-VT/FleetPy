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

# src imports
# -----------
from src.FleetSimulationBase import FleetSimulationBase

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------------------------- #
# functions
# ---------


# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
INPUT_PARAMETERS_BatchOfferSimulation = {
    "doc" :     """
    this fleet simulation class is used for the ride pooling bmw study
    customers request trips from a single ride-pooling operator continously in time.
    offers are only created after the optimisation step of the operator and fetched from the time_trigger function
    """,
    "inherit" : "FleetSimulationBase",
    "input_parameters_mandatory": [
    ],
    "input_parameters_optional": [
    ],
    "mandatory_modules": [
    ], 
    "optional_modules": []
}

class BatchOfferSimulation(FleetSimulationBase):
    """
    this fleet simulation class is used for the ride pooling bmw study
    customers request trips from a single ride-pooling operator continously in time.
    offers are only created after the optimisation step of the operator and fetched from the time_trigger function
    """
    def add_init(self, scenario_parameters):
        """
        Simulation specific additional init.
        :param scenario_parameters: row of pandas data-frame; entries are saved as x["key"]
        """
        super().add_init(scenario_parameters)

    def step(self, sim_time):
        """This method determines the simulation flow in a time step.
            # 1) update fleets and network
            # 2) get new travelers, add to undecided request
            # 3) make request (without immediate response) to operators
            # 4) periodically for waiting requests: run decision process -> possibly leave system (cancellation)
            # 5) call time trigger -> offer to all undecided assigned requests
            # 6) sequential processes for each undecided request: user-decision
            # 7) trigger charging ops

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
        last_time = sim_time - self.time_step
        if last_time < self.start_time:
            last_time = None
        list_new_traveler_rid_obj = self.demand.get_new_travelers(sim_time, since=last_time)

        self.create_database_2(sim_time, list_new_traveler_rid_obj)
        # 3)
        for rid, rq_obj in list_new_traveler_rid_obj:
            for op_id in range(self.n_op):
                LOG.debug(f"Request {rid}: To operator {op_id} ...")
                self.operators[op_id].user_request(rq_obj, sim_time)

        # 4)
        self._check_waiting_request_cancellations(sim_time)

        # 5)
        for op_id, op_obj in enumerate(self.operators):
            # here offers are created in batch assignment
            op_obj.time_trigger(sim_time)

        # 6)
        for rid, rq_obj in self.demand.get_undecided_travelers(sim_time):
            for op_id in range(self.n_op):
                amod_offer = self.operators[op_id].get_current_offer(rid)
                LOG.debug(f"amod offer {amod_offer}")
                if amod_offer is not None:
                    rq_obj.receive_offer(op_id, amod_offer, sim_time)
            self._rid_chooses_offer(rid, rq_obj, sim_time)
            
        # 7)
        for ch_op_dict in self.charging_operator_dict.values():
            for ch_op in ch_op_dict.values():
                ch_op.time_trigger(sim_time)

        # self.create_database(sim_time, list_new_traveler_rid_obj)

        self.record_stats()

    def create_database(self, sim_time, list_new_traveler_rid_obj):
        '''
        This method creates an additional output file containing information about user requests and fleet vehicle
        state.

        :return: None
        '''
        # ideally define these as globals such that they do not need to be redefined during every run
        str_pos = 'Pos'
        str_l_dest = 'Last Destination'
        str_num_stops = 'Number of Stops'
        str_pax = 'Nr. Pax'
        str_dist_start_start = 'Distance Start Vehicle Start Request'
        str_dist_end_start = 'Distance End Vehicle Start Request'
        str_dist_end_end = 'Distance End Vehicle End Request'
        str_rid = 'Request ID'
        str_cl_remaining_time = 'Remaining Time CL'
        str_n_assigned_route = 'Number of Assigned Routes'
        path_current_state = os.path.join(self.dir_names[G_DIR_OUTPUT], "current_state.csv")

        sorted_sim_vehicle_keys = sorted(self.sim_vehicles.keys())
        list_vehicle_state_requests = []
        if list_new_traveler_rid_obj:
            for rid, rq_obj in list_new_traveler_rid_obj:
                list_vehicle_states = [self.sim_vehicles[sim_vid].return_current_state(str_pos, str_l_dest,
                                                                                       str_num_stops, str_pax,
                                                                                       str_dist_start_start,
                                                                                       str_dist_end_start,
                                                                                       str_dist_end_end,
                                                                                       str_cl_remaining_time,
                                                                                       str_n_assigned_route,
                                                                                       rid,
                                                                                       rq_obj)
                                       for sim_vid in sorted_sim_vehicle_keys]
                dict_vehicles_states = {i[G_V_VID]: {str_pos: i[str_pos], str_l_dest: i[str_l_dest], str_pax: i[str_pax],
                                                     str_num_stops: i[str_num_stops],
                                                     str_dist_start_start: i[str_dist_start_start],
                                                     str_dist_end_start: i[str_dist_end_start],
                                                     str_dist_end_end: i[str_dist_end_end], str_cl_remaining_time:
                                                     i[str_cl_remaining_time], str_n_assigned_route:
                                                     i[str_n_assigned_route], str_rid: rid}
                                        for i in list_vehicle_states}
                dict_vehicles_states['Sim Time'] = sim_time
                list_vehicle_state_requests.append(dict_vehicles_states)
        else:
            list_vehicle_states = [self.sim_vehicles[sim_vid].return_current_state(str_pos, str_l_dest, str_num_stops,
                                                                                   str_pax, list_start_end_position,
                                                                                   str_dist_start_start,
                                                                                   str_dist_end_start, str_dist_end_end,
                                                                                   str_cl_remaining_time,
                                                                                   str_n_assigned_route)
                                   for sim_vid in sorted_sim_vehicle_keys]
            dict_vehicles_states = {i[G_V_VID]: {str_pos: i[str_pos], str_l_dest: i[str_l_dest], str_pax: i[str_pax],
                                                 str_num_stops: i[str_num_stops],
                                                 str_dist_start_start: i[str_dist_start_start],
                                                 str_dist_end_start: i[str_dist_end_start],
                                                 str_dist_end_end: i[str_dist_end_end], str_cl_remaining_time:
                                                 i[str_cl_remaining_time], str_n_assigned_route:
                                                 i[str_n_assigned_route]}
                                    for i in list_vehicle_states}
            dict_vehicles_states['Sim Time'] = sim_time
            list_vehicle_state_requests.append(dict_vehicles_states)
        df_current_DB = pd.DataFrame(list_vehicle_state_requests)

        if os.path.isfile(path_current_state):
            write_mode, write_header = "a", False
        else:
            write_mode, write_header = "w", True
        df_current_DB.set_index('Sim Time', inplace=True)
        df_current_DB.to_csv(path_current_state, mode=write_mode, header=write_header)

    def create_database_2(self, sim_time, list_new_traveler_rid_obj):
        '''
        This method creates an additional output file containing information about user requests and fleet vehicle
        state.

        :return: None
        '''
        # ideally define these as globals such that they do not need to be redefined during every run
        str_pos = 'Pos'
        str_l_dest = 'Last Destination'
        str_num_stops = 'Number of Stops'
        str_pax = 'Nr. Pax'
        str_dist_start_start = 'Distance Start Vehicle Start Request'
        str_dist_end_start = 'Distance End Vehicle Start Request'
        str_dist_end_end = 'Distance End Vehicle End Request'
        str_rid = 'Request ID'
        str_cl_remaining_time = 'Remaining Time CL'
        str_n_assigned_route = 'Number of Assigned Routes'
        str_vehicle_status = 'Current Vehicle Status'
        path_current_state = os.path.join(self.dir_names[G_DIR_OUTPUT], "current_state.csv")

        sorted_sim_vehicle_keys = sorted(self.sim_vehicles.keys())
        list_vehicle_states = [self.sim_vehicles[sim_vid].return_current_vehicle_state(str_pos, str_l_dest,
                                                                                       str_num_stops, str_pax,
                                                                                       str_cl_remaining_time,
                                                                                       str_vehicle_status)
                               for sim_vid in sorted_sim_vehicle_keys]
        list_vehicle_state_dict = []
        if list_new_traveler_rid_obj:
            for rid, rq_obj in list_new_traveler_rid_obj:
                list_vehicle_request_states = [self.sim_vehicles[sim_vid].return_current_vehicle_state_request(
                    str_dist_start_start, str_dist_end_start, str_dist_end_end, rid, rq_obj)
                                       for sim_vid in sorted_sim_vehicle_keys]
                list_complete_vehicle_state = [dict(i, **j) for i,j in zip(list_vehicle_states, list_vehicle_request_states)]
                dict_vehicles_states = {
                    i[G_V_VID]: {str_rid: rid, str_pos: i[str_pos], str_l_dest: i[str_l_dest], str_pax: i[str_pax],
                                 str_num_stops: i[str_num_stops],
                                 str_dist_start_start: i[str_dist_start_start],
                                 str_dist_end_start: i[str_dist_end_start],
                                 str_dist_end_end: i[str_dist_end_end], str_cl_remaining_time:
                                     i[str_cl_remaining_time], str_vehicle_status: i [str_vehicle_status]}
                    for i in list_complete_vehicle_state}
                dict_vehicles_states['Sim Time'] = sim_time
                list_vehicle_state_dict.append(dict_vehicles_states)
            df_current_DB = pd.DataFrame(list_vehicle_state_dict)

            if os.path.isfile(path_current_state):
                write_mode, write_header = "a", False
            else:
                write_mode, write_header = "w", True
            df_current_DB.set_index('Sim Time', inplace=True)
            df_current_DB.to_csv(path_current_state, mode=write_mode, header=write_header)
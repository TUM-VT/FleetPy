# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import importlib
import os
import json
import pandas as pd
from abc import abstractmethod, ABCMeta
import pathlib
import sys


# additional module imports (> requirements)
# ------------------------------------------
# from IPython import embed

# src imports
# -----------
from src.FleetSimulationBase import FleetSimulationBase
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.simulation.Vehicles import ExternallyMovingSimulationVehicle
from src.misc.init_modules import load_fleet_control_module
# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)

log_level = "DEBUG"
# -------------------------------------------------------------------------------------------------------------------- #
# functions
# ---------


# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----

INPUT_PARAMETERS_SUMOcontrolledSim = {
    "doc" : """this simulation environment couples fleetpy with SUMO. vehicle plans are sent to sumo while current vehicle states are sent back to fleetpy.
            vehicles are removed from sumo while standing.
            the simulation flow models the immediateDescisionSimulation""",
    "inherit" : "FleetSimulationBase",
    "input_parameters_mandatory": [],   # TODO requires G_AR_MAX_DEC_T == 0 (specify somehow?)
    "input_parameters_optional": [G_SUMO_STAT_INT],
    "mandatory_modules": [], 
    "optional_modules": []
}

class SUMOcontrolledSim(FleetSimulationBase):
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
        print(f"Operator Routing Mode: {scenario_parameters[G_OP_ROUTING_MODE]}")
        self.routing_engine.set_routing_mode(scenario_parameters[G_OP_ROUTING_MODE])

    def step(self, sim_time):
        """This method determines the simulation flow in a time step.
            # 1) update fleets and network
            # 2) get new travelers, add to undecided request
            # 3) sequential processes for each undecided request: request -> offer -> user-decision
            # 4) periodically for waiting requests: run decision process -> possibly leave system (cancellation)
            # 5) periodically operator: call ride pooling optimization, repositioning, charging management

        :param sim_time: new simulation time
        :return: None
        """
        LOG.debug(f"________SUMOcontrolledSimulation time: {sim_time}_______")
        # 1) update fleets and network
        leg_status_dict = self.update_sim_state_fleets(sim_time - self.time_step, sim_time)
        new_travel_times = self.routing_engine.update_network(sim_time)
     

        if new_travel_times:
            for op_id in range(self.n_op):
                self.operators[op_id].inform_network_travel_time_update(sim_time)
        # 2) get new travelers, add to undecided request
        list_undecided_travelers = list(self.demand.get_undecided_travelers(sim_time))
        last_time = sim_time - self.time_step
        if last_time < self.start_time:
            last_time = None
        list_new_traveler_rid_obj = self.demand.get_new_travelers(sim_time, since=last_time)
        # 3) sequential processes for each undecided request: request -> offer -> user-decision
        for rid, rq_obj in list_undecided_travelers + list_new_traveler_rid_obj:
            LOG.debug(f'rid is {rid} and rq_obj is {rq_obj}')
            LOG.debug(f'range(self.n_op) {range(self.n_op)}')
            for op_id in range(self.n_op):
                LOG.debug(f'op_id {op_id}')
                LOG.debug(f"Request {rid}: Checking AMoD option of operator {op_id} ...")
                # TODO # adapt fleet control
                #self.user_request(rq_obj, sim_time)
                self.operators[op_id].user_request(rq_obj, sim_time)    
                amod_offer = self.operators[op_id].get_current_offer(rid)
                LOG.debug(f'amod offer {amod_offer} ')
                if amod_offer is not None:
                    rq_obj.receive_offer(op_id, amod_offer, sim_time)
                    #LOG.debug(f'rq_obj.receive_offer(op_id, amod_offer, sim_time){rq_obj.receive_offer(op_id, amod_offer, sim_time)}')
            self._rid_chooses_offer(rid, rq_obj, sim_time)
        # 4) periodically for waiting requests: run decision process -> possibly leave system (cancellation)
        self._check_waiting_request_cancellations(sim_time)
        # 5) periodically operator: call ride pooling optimization, repositioning, charging management
        for op in self.operators:
            op.time_trigger(sim_time)
        # record at the end of each time step
        self.record_stats()
        return leg_status_dict

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
        for op_id in range(self.n_op):
            operator_attributes = self.list_op_dicts[op_id]
            operator_module_name = operator_attributes[G_OP_MODULE]
            self.op_output[op_id] = []  # shared list among vehicles
            if not operator_module_name == "LinebasedFleetControl":
                fleet_composition_dict = operator_attributes[G_OP_FLEET]
                list_vehicles = []
                vid = 0
                for veh_type, nr_veh in fleet_composition_dict.items():
                    for _ in range(nr_veh):
                        veh_type_list.append([op_id, vid, veh_type])
                        tmp_veh_obj = ExternallyMovingSimulationVehicle(op_id, vid, self.dir_names[G_DIR_VEH], veh_type,
                                                        self.routing_engine, self.demand.rq_db,
                                                        self.op_output[op_id], route_output_flag,
                                                        replay_flag)
                        list_vehicles.append(tmp_veh_obj)
                        self.sim_vehicles[(op_id, vid)] = tmp_veh_obj
                        vid += 1
                OpClass: FleetControlBase = load_fleet_control_module(operator_module_name)
                self.operators.append(OpClass(op_id, operator_attributes, list_vehicles, self.routing_engine, self.zones,
                                            self.scenario_parameters, self.dir_names, self.charging_operator_dict["op"].get(op_id, None), list(self.charging_operator_dict["pub"].values())))
            else:
                from dev.fleetctrl.LinebasedFleetControl import LinebasedFleetControl
                OpClass = LinebasedFleetControl(op_id, self.gtfs_data_dir, self.routing_engine, self.zones, self.scenario_parameters, self.dir_names, self.charging_operator_dict["op"].get(op_id, None), list(self.charging_operator_dict["pub"].values()))
                init_vids = OpClass.return_vehicles_to_initialize()
                list_vehicles = []
                for vid, veh_type in init_vids.items():
                    veh_type_list.append([op_id, vid, veh_type])
                    tmp_veh_obj = ExternallyMovingSimulationVehicle(op_id, vid, self.dir_names[G_DIR_VEH], veh_type,
                                                        self.routing_engine, self.demand.rq_db,
                                                        self.op_output[op_id], route_output_flag,
                                                        replay_flag)
                    list_vehicles.append(tmp_veh_obj)
                    self.sim_vehicles[(op_id, vid)] = tmp_veh_obj
                OpClass.continue_init(list_vehicles, self.start_time)
                self.operators.append(OpClass)
        veh_type_f = os.path.join(self.dir_names[G_DIR_OUTPUT], "2_vehicle_types.csv")
        veh_type_df = pd.DataFrame(veh_type_list, columns=[G_V_OP_ID, G_V_VID, G_V_TYPE])
        veh_type_df.to_csv(veh_type_f, index=False)
        self.vehicle_update_order = {vid : 1 for vid in self.sim_vehicles.keys()}

    def update_sim_state_fleets(self, last_time, next_time, force_update_plan=False):
        """
        This method updates the simulation vehicles, records, ends and starts tasks and returns some data that
        will be used for additional state updates (fleet control information, demand, network, ...)
        :param last_time: simulation time before the state update
        :param next_time: simulation time of the state update
        :param force_update_plan: flag that can force vehicle plan to be updated
        """
        leg_status_dict = {}
        LOG.debug(f"updating MoD state from {last_time} to {next_time}")
        for opid_vid_tuple, veh_obj in self.sim_vehicles.items():
            op_id, vid = opid_vid_tuple
            boarding_requests, alighting_requests, passed_VRL, dict_start_alighting =\
                veh_obj.update_veh_state(last_time, next_time)
            for rid, boarding_time_and_pos in boarding_requests.items():
                boarding_time, boarding_pos = boarding_time_and_pos
                LOG.debug(f"rid {rid} boarding at {boarding_time} at pos {boarding_pos}")
                self.demand.record_boarding(rid, vid, op_id, boarding_time, pu_pos=boarding_pos)
                self.operators[op_id].acknowledge_boarding(rid, vid, boarding_time)
            for rid, alighting_start_time_and_pos in dict_start_alighting.items():
                # record user stats at beginning of alighting process
                alighting_start_time, alighting_pos = alighting_start_time_and_pos
                LOG.debug(f"rid {rid} deboarding at {alighting_start_time} at pos {alighting_pos}")
                self.demand.record_alighting_start(rid, vid, op_id, alighting_start_time, do_pos=alighting_pos)
            for rid, alighting_end_time in alighting_requests.items():
                # # record user stats at end of alighting process
                self.demand.user_ends_alighting(rid, vid, op_id, alighting_end_time)
                self.operators[op_id].acknowledge_alighting(rid, vid, alighting_end_time)
                LOG.debug(f"alighting end time {alighting_end_time}")
            # send update to operator
            if len(boarding_requests) > 0 or len(dict_start_alighting) > 0:
                self.operators[op_id].receive_status_update(vid, next_time, passed_VRL, True)# Hier passiert nichts
                LOG.debug(f"boarding_requests {boarding_requests} next_time {next_time}")
            else:
                self.operators[op_id].receive_status_update(vid, next_time, passed_VRL, force_update_plan) #Hier passiert auch nichts
                LOG.debug(f"force_update_plan {force_update_plan}")#Always False
            
            #leg_status = veh_obj[1]
            LOG.debug(f"vehicle object {veh_obj} leg status: {veh_obj.status} position = {veh_obj.pos}")
            leg_status_dict[opid_vid_tuple] = (veh_obj.status, veh_obj.pos)
        LOG.debug(f"leg_status_dict{leg_status_dict}")
        return leg_status_dict


    def update_vehicle_positions(self, vehicle_to_position_dict, simulation_time):
        ''' This method is called to update the vehicle positions in the fleetsimulation
        :param vehicle_to_position_dict: dictionary (operator_id, fleetsim vehicle id) -> fleetsim network position (tuple (o_node, d_node, frac_position) or (o_node, None, None))
            for a single operator operator_id = 0
        :param simulation_time: current simulation time
        return: None'''
        for op_veh_id, position in vehicle_to_position_dict.items():
            LOG.debug(f'This is the position {position}')
            if position:
                try:
                    self.sim_vehicles[op_veh_id].update_vehicle_position(position, simulation_time)
                except KeyError:
                    LOG.warning("update_vehicle_positions failed")

    def get_vehicle_and_op_ids(self):
        opid_vid_tuple_list = []
        for opid_vid_tuple, veh_obj in self.sim_vehicles.items():
            #op_id, vid = opid_vid_tuple
            opid_vid_tuple_list.append(opid_vid_tuple)
        return(opid_vid_tuple_list)


    def get_new_vehicle_routes(self, sim_time):
        ''' This method retrieves routes from vehicles that have to be updated or started
        :return: dictionary (op_id, vehicle_id) -> list node indices in fleet sim network 
                0th entry corresponds to the start node of the current edge; 
                vehicle routes that dont need to updated are not part of the dictionary
                if a vehicle has to stop moving, an empty list is returned '''
        return_dict = {}
        for opid_vid_tuple, veh_obj in self.sim_vehicles.items():
            #print(veh_obj.vid,veh_obj.pos,veh_obj.status, veh_obj.cl_remaining_time,[rq.get_rid_struct() for rq in veh_obj.pax])
            #print([str(leg_obj) for leg_obj in veh_obj.assigned_route])
            new_route = veh_obj.get_new_route()      
            if new_route is not None:
                return_dict[opid_vid_tuple] = new_route
        LOG.debug(f'This is the return_dict from get_new_vehicle_routes{return_dict}')
        return return_dict

    def get_unserved_request_information(self):
        """ this method returns information of requests that have not been served in the last simulation time step
        :return: dictionary rq_id -> (origin_node, destination_node)"""
        LOG.warning("gut unserved request information not implemented yet") # TODO
        return {}

    def update_network_travel_times(self, new_travel_time_dict, sim_time):
        '''This method takes new edge travel time information from sumo and updates the network in the fleet simulation
        :param new_travel_time_dict: dict edge_id (o_node, d_node) -> edge_traveltime [s]
        :param sim_time: current simulation time
        :return None'''
        LOG.warning(f'new_travel_time_dict{new_travel_time_dict}')
        self.routing_engine.external_update_edge_travel_times(new_travel_time_dict)
        LOG.warning(f'self.operators {self.operators}')
        #for op in self.operators.values():
        #    op.inform_network_travel_time_update(sim_time)  # TODO # this wont work for multiprocessing!

    def vehicles_reached_destination(self, simulation_time, vids_reached_destination):
        """ this function is triggered if fleet vehicles in SUMO reached its destination;
        updates vehicle states, triggers start of boarding processes, adds information to stats
        :param simulation_time: int time of simulation from SUMO
        :param vids_reached_destination: list of (opid, vid)"""
        LOG.debug("vehicles reached destination")
        LOG.debug(f"simulation_time {simulation_time}")
        LOG.debug(f"vids_reached_destinations {vids_reached_destination}")

        for opid_vid in vids_reached_destination:
            LOG.debug(f'opid_vid {opid_vid}')
            veh_obj = self.sim_vehicles[opid_vid]
            LOG.debug(f"veh {veh_obj} reached destination")
            veh_obj.reached_destination(simulation_time)
        return  
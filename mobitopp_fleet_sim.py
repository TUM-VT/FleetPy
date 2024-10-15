# -------------------------------------------------------------------------------------------------------------------- #
# standard imports
import os
import sys
import logging
import traceback
import time

# -------------------------------------------------------------------------------------------------------------------- #
# further imports
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# add project directory to sys path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# -------------------------------------------------------------------------------------------------------------------- #
# fleet-sim package imports
from src.com.mobitopp import MobiToppSocket
from src.FleetSimulationBase import FleetSimulationBase
from src.fleetctrl.planning.PlanRequest import PlanRequest

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
from src.misc.globals import *
import src.misc.config as config
LOG = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------------------------- #
# help functions
def read_yaml_inputs(yaml_file):
    """This function transforms the yaml_file input into a dictionary and changes keys if necessary"""
    return_dict = config.ConstantConfig(yaml_file)
    return return_dict

def routing_min_distance_cost_function(travel_time, travel_distance, current_node_index):
    """computes the customized section cost for routing (input for routing functions)

    :param travel_time: travel_time of a section
    :type travel time: float
    :param travel_distance: travel_distance of a section
    :type travel_distance: float
    :param current_node_index: index of current_node_obj in dijkstra computation to be settled
    :type current_node_index: int
    :return: travel_cost_value of section
    :rtype: float
    """
    return travel_distance

# -------------------------------------------------------------------------------------------------------------------- #
# main class
class MobiToppFleetServer(FleetSimulationBase):
    """
    This class is the main entry point for the fleet simulation in transmove project.
    It runs batch optimization of the ride-pooling operator at specified times and returns the arrivals along with the fleet state to fleetJava in mobitopp.
    """
    def __init__(self, scenario_parameters):
        super().__init__(scenario_parameters)
        init_status = 0
        # initialization of communication socket
        self.com = MobiToppSocket(self, init_status)
        self.last_fs_time = 0
        self.fs_time = self.scenario_parameters[G_SIM_START_TIME]
        # only one operator: TransmoveFleetControl
        self.mt_op = self.operators[0]
        self.current_offer_id = 1

        self.bt = self.mt_op.bt

        self.incomingRid = {}

    @staticmethod
    def get_directory_dict(scenario_parameters):
        """
        This function provides the correct paths to certain data according to the specified data directory structure.
        :param scenario_parameters: simulation input (pandas series)
        :return: dictionary with paths to the respective data directories
        """
        print("get_directory_dict")
        # TODO # include zones and forecasts later on
        # study_name = scenario_parameters[G_STUDY_NAME]
        scenario_name = scenario_parameters[G_SCENARIO_NAME]
        network_name = scenario_parameters[G_NETWORK_NAME]
        demand_name = scenario_parameters.get(G_DEMAND_NAME, None)
        zone_name = scenario_parameters.get(G_ZONE_SYSTEM_NAME, None)
        infra_name = scenario_parameters.get(G_INFRA_NAME, None)
        fc_type = scenario_parameters.get(G_FC_TYPE, None)
        fc_t_res = scenario_parameters.get(G_FC_TR, None)
        # gtfs_name = scenario_parameters.get(G_GTFS_NAME, None)
        #
        dirs = {}
        dirs[G_DIR_MAIN] = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir,
                                                        os.path.pardir))
        dirs[G_DIR_DATA] = os.path.join(dirs[G_DIR_MAIN], "data", "fleetsim")
        #dirs[G_DIR_OUTPUT] = os.path.join(dirs[G_DIR_MAIN], "output", "results", "simulation", scenario_name)
        dirs[G_DIR_OUTPUT] = os.path.join(dirs[G_DIR_MAIN], "output", scenario_name, "simulation-fleetsim")
        dirs[G_DIR_NETWORK] = os.path.join(dirs[G_DIR_DATA], "networks", network_name)
        dirs[G_DIR_VEH] = os.path.join(dirs[G_DIR_DATA], "vehicles")
        dirs[G_DIR_FCTRL] = os.path.join(dirs[G_DIR_DATA], "fleetctrl")
        if infra_name is not None:
            dirs[G_DIR_INFRA] = os.path.join(dirs[G_DIR_DATA], "infra", infra_name, network_name)
        # dirs[G_DIR_DEMAND] = os.path.join(dirs[G_DIR_DATA], "demand", demand_name, "matched", network_name)
        if zone_name is not None:
            dirs[G_DIR_ZONES] = os.path.join(dirs[G_DIR_DATA], "zones", zone_name, network_name)
            if fc_type is not None and fc_t_res is not None:
                dirs[G_DIR_FC] = os.path.join(dirs[G_DIR_DATA], "demand", demand_name, "aggregated", zone_name, str(fc_t_res))
        # if gtfs_name is not None:
        #     dirs[G_DIR_PT] = os.path.join(dirs[G_DIR_DATA], "pubtrans", gtfs_name)
        return dirs

    def add_init(self, scenario_parameters):
        """
        Simulation specific additional init.
        :param scenario_parameters: row of pandas data-frame; entries are saved as x["key"]
        """
        super().add_init(scenario_parameters)

    def check_sim_env_spec_inputs(self, scenario_parameters):
        """
        This function checks the simulation environment specific inputs.
        :param scenario_parameters: simulation input (pandas series)
        :return: scenario_parameters
        """
        return scenario_parameters
    
    def add_evaluate(self):
        """
        This function calls the evaluation function.
        """
        output_dir = self.dir_names[G_DIR_OUTPUT]
        from src.evaluation.MOIA_temporal import temporal_evaluation
        temporal_evaluation(output_dir)

    def run(self):
        """
        This function controls the simulation.
        """
        # simulation is controlled by mobitopp triggers
        self.com.keep_socket_alive()
        # save final state, record remaining travelers and vehicle tasks
        self.save_final_state()
        self.record_remaining_assignments()
        # call fleet evaluation
        self.evaluate()

    def register_request(self, agent_id, o_node, d_node, simulation_time, earliest_pickup_time, number_passengers):
        """
        This method registers a new request in the fleet simulation. The request is forwarded to the fleet control for batch optimization.
        For transmove, SlaveDemand is used.
        """
        rq_info_dict = {G_RQ_ID: agent_id, G_RQ_ORIGIN: o_node, G_RQ_DESTINATION: d_node, G_RQ_TIME: simulation_time,
                        G_RQ_EPT: earliest_pickup_time, G_RQ_PAX: number_passengers}
        rq_series = pd.Series(rq_info_dict)
        rq_series.name = agent_id
        # add request to demand database: self.rq_db
        rq_obj = self.demand.add_request(rq_series, self.current_offer_id, self.routing_engine, self.fs_time)
        # add the rq to the database of the operator
        # use RidePoolingBatchAssignmentFleetcontrol and add the rq to self.incoming_requests
        self.mt_op.user_request(rq_obj, self.fs_time)
        self.current_offer_id += 1
        self.incomingRid[agent_id] = 1

    def optimize_fleet(self):
        """
        This method calls all operator functions to control the vehicle fleet. This includes an improved customer-
        assignment, moving illegally parked vehicles, charging vehicles and a vehicle repositioning strategy.
        This process is blocking.
        """
        # customer-vehicle assignment, move illegally parking vehicles, charging, vehicle repositioning
        self.mt_op.time_trigger(self.fs_time)

        # recorder offer and fare in rq_obj
        for rid in self.incomingRid:
            rq_obj = self.demand.rq_db[rid]
            prq: PlanRequest = self.mt_op.rq_dict.get(rid, None)
            if prq is not None:
                offer = prq.get_current_offer()
                if offer is not None:
                    rq_obj.receive_offer(0, offer, self.fs_time)
                    rq_obj.fare = rq_obj.offer[0].get(G_OFFER_FARE, 0)
        self.incomingRid = {}

    def get_current_fleet_state(self, simulation_time):
        """This method returns the current fleet state.
        """
        return self.mt_op.get_current_fleet_state(simulation_time)

    def update_network(self, simulation_time):
        """
        This method updates all edge travel times in the network. Furthermore, all current V2RBs are updated, where
        all previously accepted solutions remain feasible.
        :param simulation_time:
        """
        new_travel_times = self.routing_engine.update_network(simulation_time) 
        if new_travel_times:
            self.update_vehicle_routes(simulation_time)
            self.mt_op.inform_network_travel_time_update(simulation_time)

    def update_state(self, simulation_time):
        """
        This method increases the time of the fleet simulation (it receives the time of the next time step as
        argument), performs all vehicle processes, including boarding processes (and the resulting RV and RR deletions),
        and returns a list of customers that arrive at their destination in this time step.
        :param simulation_time:
        :return: list of agent-ids of arriving customers
        """
        # from FleetSimulation.update_sim_state_fleets()
        list_arrivals = []
        self.last_fs_time = self.fs_time
        self.fs_time = simulation_time
        LOG.debug(f"updating MoD state from {self.last_fs_time} to {self.fs_time}")
        for opid_vid_tuple, veh_obj in sorted(self.sim_vehicles.items(), key=lambda x:self.vehicle_update_order[x[0]]):
            op_id, vid = opid_vid_tuple
            boarding_requests, alighting_requests, passed_VRL, dict_start_alighting =\
                veh_obj.update_veh_state(self.last_fs_time, self.fs_time)
            if veh_obj.status == VRL_STATES.CHARGING:
                self.vehicle_update_order[opid_vid_tuple] = 0
            else:
                self.vehicle_update_order[opid_vid_tuple] = 1
            for rid, boarding_time_and_pos in boarding_requests.items():
                boarding_time, boarding_pos = boarding_time_and_pos
                LOG.debug(f"rid {rid} boarding vehicle {vid} of operator {op_id} at {boarding_time} at pos {boarding_pos}")
                rq_obj = self.demand.rq_db[rid]
                walking_distance = self._return_walking_distance(rq_obj.o_pos, boarding_pos)
                t_access = walking_distance * self.scenario_parameters[G_WALKING_SPEED]
                self.demand.record_boarding(rid, vid, op_id, boarding_time, pu_pos=boarding_pos, t_access=t_access)
                self.mt_op.acknowledge_boarding(rid, vid, boarding_time)
            for rid, alighting_start_time_and_pos in dict_start_alighting.items():
                # record user stats at beginning of alighting process
                alighting_start_time, alighting_pos = alighting_start_time_and_pos
                LOG.debug(f"rid {rid} alighting vehicle {vid} of operator {op_id} at {alighting_start_time} at pos {alighting_pos}")
                rq_obj = self.demand.rq_db[rid]
                walking_distance = self._return_walking_distance(rq_obj.d_pos, alighting_pos)
                t_egress = walking_distance * self.scenario_parameters[G_WALKING_SPEED]
                self.demand.record_alighting_start(rid, vid, op_id, alighting_start_time, do_pos=alighting_pos, t_egress=t_egress)
            
            # collect arrivals and hypothetical arrivals information for mobitopp
            list_arrivals = self._collect_arrivals(alighting_requests)
            
            for rid, alighting_end_time in alighting_requests.items():
                # record user stats at end of alighting process
                # delete the request from self.demand.rq_db
                self.demand.user_ends_alighting(rid, vid, op_id, alighting_end_time)
                # delete the request from self.mt_op.rq_dict
                self.mt_op.acknowledge_alighting(rid, vid, alighting_end_time)
            # send update to operator
            self.mt_op.receive_status_update(vid, self.fs_time, passed_VRL, force_update=True) 
        self.record_stats()
        return list_arrivals
    
    def _collect_arrivals(self, alighting_requests):
        """This method collects the arrivals from the alighting_requests.
        """
        # collect the arrivals from the alighting_requests
        list_arrivals = [self.demand.rq_db[rid] for rid in alighting_requests.keys()]
        # collect the hypothetical arrivals
        self._collect_hypothetical_alighting(list_arrivals)  
        return list_arrivals
    
    def _collect_hypothetical_alighting(self, list_arrivals: list):
        """This method checks if there are any hypothetical alighting requests that need to be returned to mobitopp.
        Requests are only returned if their alighting time is in the time frame between self.last_fs_time and self.fs_time.
        """
        # There is already a check process in socket, 
        # maybe we can just return all hypothetical requests and save them in the self.delayed_arrivals.
        for rid, offer in self.mt_op.hypothetical_alighting_requests.items():
            rq_obj = self.demand.rq_db[rid]
            # update rq_obj with the offer
            rq_obj.service_opid = offer.__getitem__("service_opid")
            rq_obj.service_vid = offer.__getitem__("service_vid")
            rq_obj.t_access = 0
            rq_obj.t_egress = 0
            rq_obj.pu_time = offer.__getitem__("pu_time")
            rq_obj.do_time = offer.__getitem__("do_time")
            list_arrivals.append(rq_obj)
            self.demand.record_user(rid)
        # empty the hypothetical alighting requests dictionary
        self.mt_op.hypothetical_alighting_requests = {}
        return list_arrivals
    
    def evaluate(self):
        """Runs standard and simulation environment specific evaluations over simulation results."""
        output_dir = self.dir_names[G_DIR_OUTPUT]
        # standard evaluation
        from src.evaluation.standard import standard_evaluation
        standard_evaluation(output_dir)
        # TODO: update the MOIA evaluation, do we still need it?
        from src.evaluation.MOIA_temporal import temporal_evaluation
        temporal_evaluation(output_dir)   

    def process_charging(self):
        """
        This method processes the charging of vehicles.
        """
        for ch_op_dict in self.charging_operator_dict.values():
            for ch_op in ch_op_dict.values():
                ch_op.time_trigger(self.fs_time)

    def _return_walking_distance(self, origin_pos, target_pos):
        """ returns the walking distance from an origin network position to an target network position
        :param origin_pos: network position of origin
        :param target_pos: network position of target
        :return: walking distance in m
        """
        walking_distance = None
        # TODO：why do we need to calculate the walking distance twice?
        _, _, dis1 = self.routing_engine.return_travel_costs_1to1(origin_pos, 
                                                                  target_pos, 
                                                                  customized_section_cost_function = routing_min_distance_cost_function)
        _, _, dis2 = self.routing_engine.return_travel_costs_1to1(target_pos, 
                                                                  origin_pos, 
                                                                  customized_section_cost_function = routing_min_distance_cost_function)
        if dis1 < dis2:
            walking_distance = dis1
        else:
            walking_distance = dis2
        return walking_distance


# -------------------------------------------------------------------------------------------------------------------- #
# script call
if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise IOError("Fleet simulation requires exactly one input parameter (*yaml configuration file)!")
        yaml_file = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, sys.argv[1]))
        print(yaml_file)
        if not os.path.isfile(yaml_file) or not yaml_file.endswith("yaml"):
            prt_str = f"Configuration file {yaml_file} not found or with invalid file ending!"
            raise IOError(prt_str)
        # read configuration and bring to standard input format
        scenario_parameter_dict = read_yaml_inputs(yaml_file)
        # initialize fleet simulation
        fs = MobiToppFleetServer(scenario_parameter_dict)
        # run simulation
        fs.run()
        print("Simulation completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        input("Press Enter to close the window...")
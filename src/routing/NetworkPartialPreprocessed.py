"""
Authors: Roman Engelhardt, Florian Dandl
TUM, 2020
In order to guarantee transferability of models, Network models should follow the following conventions.
Classes should be called
Node
Edge
Network
in order to guarantee correct import in other modules.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import os
import logging

# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
import numpy as np

# src imports
# -----------
from src.routing.NetworkBasic import NetworkBasic, Node, Edge
from src.routing.routing_imports.Router import Router

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_NetworkPartialPreprocessed = {
    "doc" : """
        This network uses numpy traveltime tables which are processed beforehand and stored in the data folder
        Instead of preprocessing the whole network, only the travel time of the shortest routes between first x nodes are preprocessed.
        Note ,that nodes should be sorted first based on their expected usage as routing targets
        see: src/preprocessing/networks/create_partially_preprocessed_travel_time_tables.py for creating the numpy tables
            src/preprocessing/networks/network_manipulation.py for sorting the network nodes

        This network stores additionally computed traveltimes in a dictionary and reuses this data.
        
        As fallback, Dijkstra's algorithm implemented in Python is used.
        """,
    "inherit" : "NetworkBasic",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class NetworkPartialPreprocessed(NetworkBasic):
    def __init__(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        """
        The network will be initialized.

        this network uses numpy traveltime tables which are processed beforehand and stored in the data folder
        in this table the travel times and travel distances of the time shortest routes between all nodes with the first x indices are preprocessed
        note that nodes should be sorted first based on their expected usage as routing targets
        see: src/preprocessing/networks/create_partially_preprocessed_travel_time_tables.py for creating the numpy tables
            src/preprocessing/networks/network_manipulation.py for sorting the network nodes

        This network stores additionally computed traveltimes in a dictionary and reuses this data

        :param network_name_dir: name of the network_directory to be loaded
        :param type: determining whether the base or a pre-processed network will be used
        :param scenario_time: applying travel times for a certain scenario at a given time in the scenario
        :param network_dynamics_file_name: file-name of the network dynamics file
        :type network_dynamics_file_name: str
        """
        self.tt_table = None
        self.dis_table = None
        self.max_preprocessed_index = -1

        super().__init__(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)

        self.travel_time_infos = {} #(o,d) -> (tt, dis)
        if self.tt_table is None:
            self._load_travel_info_tables(network_name_dir, scenario_time=None)

    def _load_travel_info_tables(self, network_name_dir, scenario_time = None):
        if scenario_time is None:
            f = "base"
        else:
            f = str(self.travel_time_file_folders[scenario_time])
        self.tt_table = np.load(os.path.join(network_name_dir, f, "tt_matrix.npy"))
        self.dis_table = np.load(os.path.join(network_name_dir, f, "dis_matrix.npy"))
        self.max_preprocessed_index = self.tt_table.shape[0]
        LOG.info(" ... travel time tables loaded until index {}".format(self.max_preprocessed_index))

    def update_network(self, simulation_time, update_state = True):
        """This method can be called during simulations to update travel times (dynamic networks).

        :param simulation_time: time of simulation
        :type simulation_time: float
        :return: new_tt_flag True, if new travel times found; False if not
        :rtype: bool
        """
        LOG.debug(f"update network {simulation_time} | preproc index {self.max_preprocessed_index}")
        self.sim_time = simulation_time
        if update_state:
            if self.travel_time_file_folders.get(simulation_time, None) is not None:
                LOG.info("update travel times in network {}".format(simulation_time))
                self.load_tt_file(simulation_time)
                self.travel_time_infos = {}
                return True
        return False

    def load_tt_file(self, scenario_time):
        """
        loads new travel time files for scenario_time
        """
        super().load_tt_file(scenario_time)
        self._load_travel_info_tables(self.network_name_dir, scenario_time=scenario_time)

    def return_travel_costs_1to1(self, origin_position, destination_position, customized_section_cost_function = None):
        """
        This method will return the travel costs of the fastest route between two nodes.
        :param origin_position: (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :param destination_position: (destination_edge_origin_node_index, destination_edge_destination_node_index, relative_position)
        :param customized_section_cost_function: function to compute the travel cost of an section: args: (travel_time, travel_distance, current_dijkstra_node) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :return: (cost_function_value, travel time, travel_distance) between the two nodes
        """
        trivial_test = self.test_and_get_trivial_route_tt_and_dis(origin_position, destination_position)
        if trivial_test is not None:
            return trivial_test[1]
        origin_node = origin_position[0]
        origin_overhead = (0.0, 0.0, 0.0)
        if origin_position[1] is not None:
            origin_node = origin_position[1]
            origin_overhead = self.get_section_overhead(origin_position, from_start=False)
        destination_node = destination_position[0]
        destination_overhead = (0.0, 0.0, 0.0)
        if destination_position[1] is not None:
            destination_overhead = self.get_section_overhead(destination_position, from_start=True)
        s = None
        #LOG.warning("get1to1: {} -> {} table: {} dict {}".format(origin_node, destination_node, self.max_preprocessed_index, self.travel_time_infos.get( (origin_node, destination_node) , None)))
        if customized_section_cost_function is None:
            if origin_node < self.max_preprocessed_index and destination_node < self.max_preprocessed_index:
                s = (self.tt_table[origin_node][destination_node], self.tt_table[origin_node][destination_node], self.dis_table[origin_node][destination_node])
            else:
                s = self.travel_time_infos.get( (origin_node, destination_node) , None)
        if s is None:
            R = Router(self, origin_node, destination_nodes=[destination_node], mode='bidirectional', customized_section_cost_function=customized_section_cost_function)
            s = R.compute(return_route=False)[0][1]
            self.travel_time_infos[(origin_node, destination_node)] = s
        return (s[0] + origin_overhead[0] + destination_overhead[0], s[1] + origin_overhead[1] + destination_overhead[1], s[2] + origin_overhead[2] + destination_overhead[2])

    def add_travel_infos_to_database(self, travel_info_dict):
        """ this function can be used to include externally computed (e.g. multiprocessing) route travel times
        into the database if present
        :param travel_info_dict: dictionary with keys (origin_position, target_positions) -> values (cost_function_value, travel_time, travel_distance)
        """
        for od_pos, s in travel_info_dict.items():
            # since only travel infos between nodes are stored the inverse function from get_travel_costs_1to1 has to be called here
            origin_position, destination_position = od_pos
            origin_node = origin_position[0]
            origin_overhead = (0.0, 0.0, 0.0)
            if origin_position[1] is not None:
                origin_node = origin_position[1]
                origin_overhead = self.get_section_overhead(origin_position, from_start=False)
            destination_node = destination_position[0]
            if self.max_preprocessed_index >= origin_node and self.max_preprocessed_index >= destination_node:
                continue
            if self.travel_time_infos.get( (origin_node, destination_node) ) is not None:
                continue
            destination_overhead = (0.0, 0.0, 0.0)
            if destination_position[1] is not None:
                destination_overhead = self.get_section_overhead(destination_position, from_start=True)
            s_adopted = (s[0] - origin_overhead[0] - destination_overhead[0], s[1] - origin_overhead[1] - destination_overhead[1], s[2] - origin_overhead[2] - destination_overhead[2])
            self.travel_time_infos[(origin_node, destination_node)] = s_adopted

    def _reset_internal_attributes_after_travel_time_update(self):
        self.travel_time_infos = {}
        self.tt_table = None
        self.dis_table = None
        self.max_preprocessed_index = -1

    def _add_to_database(self, o_node, d_node, cfv, tt, dis):
        """ this function is call when new routing results have been computed
        depending on the class the function can be overwritten to store certain results in the database
        """
        #LOG.warning("add to db: {} -> {} tt {}".format(o_node, d_node, tt ))
        if self.max_preprocessed_index < o_node or self.max_preprocessed_index < d_node:
            if self.travel_time_infos.get( (o_node, d_node) ) is None:
                #LOG.warning("done")
                self.travel_time_infos[ (o_node, d_node) ] = (cfv, tt, dis)
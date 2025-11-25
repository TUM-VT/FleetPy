# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import os
import logging
import time

# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
import numpy as np

# src imports
# -----------
from src.routing.NetworkBasicWithStoreCpp import NetworkBasicWithStoreCpp

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

MAX_MATRIX_SIZE = 2000  # maximum number of nodes for which a full matrix is stored

INPUT_PARAMETERS_NetworkBasicWithStoreOnlineMatrixCpp = {
    "doc" : """
        This routing class does all routing computations based on dijkstras algorithm.
        Compared to NetworkBasicWithStore, this module has the same methods but is implemented in C++ and included via Cython.
        Compared to NetworkBasicCpp.py, this class stores already computed travel infos in a dictionary and returns the values from this dictionary if queried again.
        To install the coupling to C++, you need to run `src\routing\cpp_router\setup.py`
        """,
    "inherit" : "NetworkBasicWithStoreCpp",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class NetworkBasicWithStoreOnlineMatrixCpp(NetworkBasicWithStoreCpp):
    def __init__(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        """
        The network will be initialized.
        This network stores routing results from return_travel_costs_1to1 in a database to retrieve them in case they are queried again
        additionally, if calling the function return_travel_costs_1to1, internally a dijkstra to all boarding nodes is called in case origin and destination is a boarding node
            these results are not returned, but directly stored in the database in case they are needed again

        :param network_name_dir: name of the network_directory to be loaded
        :param type: determining whether the base or a pre-processed network will be used
        :param scenario_time: applying travel times for a certain scenario at a given time in the scenario
        :param network_dynamics_file_name: file-name of the network dynamics file
        :type network_dynamics_file_name: str
        """
        super().__init__(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)
        
        self._routing_query_counter = {} # node_index -> number of queries
        
        self._node_index_to_matrix_index = {} # node_index -> matrix index
        self.tt_table = None
        self.dis_table = None
        
    def return_travel_costs_1to1(self, origin_position, destination_position, customized_section_cost_function = None):
        """
        This method will return the travel costs of the fastest route between two nodes.
        :param origin_position: (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :param destination_position: (destination_edge_origin_node_index, destination_edge_destination_node_index, relative_position)
        :param customized_section_cost_function: function to compute the travel cost of an section: args: (travel_time, travel_distance, current_dijkstra_node) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :return: (cost_function_value, travel time, travel_distance) between the two nodes
        """
        if customized_section_cost_function is not None:
            return super().return_travel_costs_1to1(origin_position, destination_position, customized_section_cost_function = customized_section_cost_function)
        # Update query counter
        self._routing_query_counter[origin_position[0]] = self._routing_query_counter.get(origin_position[0], 0) +1
        self._routing_query_counter[origin_position[0]] += 1
        self._routing_query_counter[destination_position[0]] = self._routing_query_counter.get(destination_position[0], 0) +1
        self._routing_query_counter[destination_position[0]] += 1
        
        o_matrix_index = self._node_index_to_matrix_index.get(origin_position[0], None)
        d_matrix_index = self._node_index_to_matrix_index.get(destination_position[0], None)
        if o_matrix_index is not None and d_matrix_index is not None:
            tt = self.tt_table[o_matrix_index, d_matrix_index]
            dis = self.dis_table[o_matrix_index, d_matrix_index]
            return tt, tt, dis

        return super().return_travel_costs_1to1(origin_position, destination_position, customized_section_cost_function = customized_section_cost_function)
    
    def load_tt_file(self, scenario_time, ext_path=None):
        r = super().load_tt_file(scenario_time, ext_path)
        
        self.tt_table = None
        self.dis_table = None
        self._node_index_to_matrix_index = {}
        
        matrix_size = min(len(self._routing_query_counter), MAX_MATRIX_SIZE)
        if matrix_size < 2:
            return r
        
        sorted_nodes = sorted(self._routing_query_counter.items(), key=lambda item: item[1], reverse=True)
        selected_nodes = [item[0] for item in sorted_nodes[:matrix_size]]
        
        self.tt_table = np.full((matrix_size, matrix_size), np.inf)
        self.dis_table = np.full((matrix_size, matrix_size), np.inf)
        
        LOG.info("start computing new travel time tables for top {} most queried nodes...".format(matrix_size))
        t = time.time()
        node_positions = [self.return_node_position(n) for n in selected_nodes]
        for i, s in enumerate(node_positions):
            self._node_index_to_matrix_index[s] = i
            r = self.return_travel_costs_1toX(s, node_positions)
            for e_pos, _, tt, dis in r:
                o_index = s[0]
                d_index = e_pos[0]
                self.tt_table[o_index][d_index] = tt
                self.dis_table[o_index][d_index] = dis
            if i % 500 == 0:
                LOG.info(" .... {}/{} done!".format(i, len(node_positions)))
        LOG.info(" ... done after {}s".format(time.time() - t))
        return r
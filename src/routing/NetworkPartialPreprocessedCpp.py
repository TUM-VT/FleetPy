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
from pathlib import Path

# src imports
# -----------
from src.routing.NetworkBasicCpp import NetworkBasicCpp
from src.routing.cpp_router.PyNetwork import PyNetwork

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_NetworkPartialPreprocessedCpp = {
    "doc" : """
        This network uses numpy traveltime tables which are processed beforehand and stored in the data folder
        Instead of preprocessing the whole network, only the travel time of the shortest routes between first x nodes are preprocessed.
        Note ,that nodes should be sorted first based on their expected usage as routing targets
        see: src/preprocessing/networks/create_partially_preprocessed_travel_time_tables.py for creating the numpy tables
            src/preprocessing/networks/network_manipulation.py for sorting the network nodes

        This network stores additionally computed traveltimes in a dictionary and reuses this data.
        
        As fallback, Dijkstra's algorithm implemented in Python is used.
        
        Compared to NetworkPartialPreprocessed, this module has the same methods but is implemented in C++ and included via Cython.
        To install the coupling to C++, you need to run `src\routing\cpp_router\setup.py`
        """,
    "inherit" : "NetworkBasicCpp",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class NetworkPartialPreprocessedCpp(NetworkBasicCpp):
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
        self._table_keys = set()

        super().__init__(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)

        self.travel_time_infos = {} #(o,d) -> (tt, dis)
        if self.tt_table is None:
            self._load_travel_info_tables(network_name_dir, scenario_time=None)

    def _load_travel_info_tables(self, network_name_dir, scenario_time = None):
        if scenario_time is None:
            f = "base"
        else:
            f = str(self.travel_time_file_infos[scenario_time])
        try:
            dist_df = Path(network_name_dir).joinpath(f, "travel_distance_df.csv")
            dist_table = Path(network_name_dir).joinpath(f, "dis_matrix.npy")
            if dist_df.exists():
                df = pd.read_csv(os.path.join(network_name_dir, f, "travel_distance_df.csv"), index_col=0)
                df.columns = df.columns.astype(int)
                self.dis_table = df.to_dict(orient="index")
                time_df = Path(network_name_dir).joinpath(f, "travel_time_df.csv")
                df = pd.read_csv(time_df, index_col=0)
                df.columns = df.columns.astype(int)
                self.tt_table = df.to_dict(orient="index")
                LOG.info(f"Loaded travel distance and time from csv tables {dist_df} and {time_df}")
                print(f"Loaded travel distance and time from csv tables {dist_df} and {time_df}")
            elif dist_table.exists():
                array = np.load(str(dist_table))
                self.dis_table = {i: {j: array[i, j] for j in range(array.shape[1])} for i in range(array.shape[0])}
                time_table = Path(network_name_dir).joinpath(f, "tt_matrix.npy")
                array = np.load(str(time_table))
                self.tt_table = {i: {j: array[i, j] for j in range(array.shape[1])} for i in range(array.shape[0])}
                LOG.info(f"Loaded travel distance and time from numpy tables {dist_table} and {time_table}")
                print(f"Loaded travel distance and time from numpy tables {dist_table} and {time_table}")
        except FileNotFoundError:
            LOG.warning(" ... no preprocessing files found!")
        if self.tt_table is not None and len(self.tt_table) > 0:
            for key, dest_dict in self.tt_table.items():
                for dest_key in dest_dict.keys():
                    self._table_keys.add((key, dest_key))

    def update_network(self, simulation_time, update_state = True):
        """This method can be called during simulations to update travel times (dynamic networks).

        :param simulation_time: time of simulation
        :type simulation_time: float
        :return: new_tt_flag True, if new travel times found; False if not
        :rtype: bool
        """
        self.sim_time = simulation_time
        if update_state:
            if self.travel_time_file_infos.get(simulation_time, None) is not None:
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
        if self._tt_infos_from_folder:
            self._load_travel_info_tables(self.network_name_dir, scenario_time=scenario_time)
            
    def _return_node_to_node_travel_costs_1to1(self, origin_node, destination_node):
        s = None
        if self.tt_table is not None and origin_node in self.tt_table and destination_node in self.tt_table[origin_node]:
            tt, dis = self.tt_table[origin_node][destination_node], self.dis_table[origin_node][destination_node]
            if self._current_tt_factor is not None:
                tt = tt * self._current_tt_factor
                s = (tt, tt, dis)
            else:
                s = (tt, tt, dis)
        else:
            s = self.travel_time_infos.get( (origin_node, destination_node) , None)
        if s is None:
            s = self.cpp_router.computeTravelCosts1To1(origin_node, destination_node)
            if self._current_tt_factor is not None:
                s = (s[0] * self._current_tt_factor, s[1])
            if s[0] < -0.001:
                print("no route found? {} -> {} {}".format(origin_node, destination_node, s))
                s = (float("inf"), float("inf"))
            s = (s[0], s[0], s[1])
            self._add_to_database(origin_node, destination_node, s[0], s[1], s[2])
        return s
        

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
        
        if origin_position[1] is None and destination_position[1] is None:
            return self._return_node_to_node_travel_costs_1to1(origin_position[0], destination_position[0])
        
        trivial_test = self.test_and_get_trivial_route_tt_and_dis(origin_position, destination_position)
        if trivial_test is not None:
            return trivial_test[1]
        origin_node = int(origin_position[0])
        origin_overhead = (0.0, 0.0, 0.0)
        if origin_position[1] is not None:
            origin_node = origin_position[1]
            origin_overhead = self.get_section_overhead(origin_position, from_start=False)
        destination_node = int(destination_position[0])
        destination_overhead = (0.0, 0.0, 0.0)
        if destination_position[1] is not None:
            destination_overhead = self.get_section_overhead(destination_position, from_start=True)
        s = None
        if customized_section_cost_function is None:
            if self.tt_table is not None and origin_node in self.tt_table and destination_node in self.tt_table[origin_node]:
                tt, dis = self.tt_table[origin_node][destination_node], self.dis_table[origin_node][destination_node]
                if self._current_tt_factor is not None:
                    tt = tt * self._current_tt_factor
                    s = (tt, tt, dis)
                else:
                    s = (tt, tt, dis)
            else:
                s = self.travel_time_infos.get( (origin_node, destination_node) , None)
        if s is None:
            s = self.cpp_router.computeTravelCosts1To1(origin_node, destination_node)
            if self._current_tt_factor is not None:
                s = (s[0] * self._current_tt_factor, s[1])
            if s[0] < -0.001:
                print("no route found? {} -> {} {}".format(origin_node, destination_node, s))
                s = (float("inf"), float("inf"))
            s = (s[0], s[0], s[1])
            if customized_section_cost_function is None:
                self._add_to_database(origin_node, destination_node, s[0], s[1], s[2])
        return (s[0] + origin_overhead[0] + destination_overhead[0], s[0] + origin_overhead[1] + destination_overhead[1], s[2] + origin_overhead[2] + destination_overhead[2])

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
            if self.travel_time_infos.get( (origin_node, destination_node) ) is not None:
                continue
            destination_overhead = (0.0, 0.0, 0.0)
            if destination_position[1] is not None:
                destination_overhead = self.get_section_overhead(destination_position, from_start=True)
            s_adopted = (s[0] - origin_overhead[0] - destination_overhead[0], s[1] - origin_overhead[1] - destination_overhead[1], s[2] - origin_overhead[2] - destination_overhead[2])
            self.travel_time_infos[(origin_node, destination_node)] = s_adopted

    def _reset_internal_attributes_after_travel_time_update(self):
        self.travel_time_infos = {}
        if self._tt_infos_from_folder:
            self.tt_table = None
            self.dis_table = None

    def _add_to_database(self, o_node, d_node, cfv, tt, dis):
        """ this function is call when new routing results have been computed
        depending on the class the function can be overwritten to store certain results in the database
        """
        if self.travel_time_infos.get( (o_node, d_node) ) is None:
            self.travel_time_infos[ (o_node, d_node) ] = (cfv, tt, dis)
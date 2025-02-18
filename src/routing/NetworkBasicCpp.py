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
from src.routing.NetworkBasic import NetworkBasic
from src.routing.cpp_router.PyNetwork import PyNetwork

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_NetworkBasicCpp = {
    "doc" : """
        This routing class does all routing computations based on dijkstras algorithm.
        Compared to NetworkBasic, this module has the same methods but is implemented in C++ and included via Cython.
        To install the coupling to C++, you need to run `src\routing\cpp_router\setup.py`
        """,
    "inherit" : "NetworkBasic",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class NetworkBasicCpp(NetworkBasic):
    def __init__(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        """
        The network will be initialized.
        This network only uses basic routing algorithms (dijkstra and bidirectional dijkstra)
        :param network_name_dir: name of the network_directory to be loaded
        :param type: determining whether the base or a pre-processed network will be used
        :param scenario_time: applying travel times for a certain scenario at a given time in the scenario
        :param network_dynamics_file_name: file-name of the network dynamics file
        :type network_dynamics_file_name: str
        """
        self.cpp_router = None
        super().__init__(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)

    def loadNetwork(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        LOG.info("load c++ router!")
        nodes_f = os.path.join(network_name_dir, "base", "nodes.csv")
        edges_f = os.path.join(network_name_dir, "base", "edges.csv")
        self.cpp_router = PyNetwork(nodes_f.encode(), edges_f.encode())
        super().loadNetwork(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)

    def load_tt_file(self, scenario_time):
        """
        loads new travel time files for scenario_time
        """
        super().load_tt_file(scenario_time)
        f = self.travel_time_file_folders[scenario_time]
        tt_file = os.path.join(f, "edges_td_att.csv")
        self.cpp_router.updateEdgeTravelTimes(tt_file.encode())

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
        s = self.cpp_router.computeTravelCosts1To1(origin_node, destination_node)
        if s[0] < -0.001:
            s = (float("inf"), float("inf"))
        res = (s[0] + origin_overhead[0] + destination_overhead[0], s[0] + origin_overhead[1] + destination_overhead[1], s[1] + origin_overhead[2] + destination_overhead[2])
        if customized_section_cost_function is None:
            self._add_to_database(origin_node, destination_node, s[0], s[0], s[1])
        return res

    def return_travel_costs_Xto1(self, list_origin_positions, destination_position, max_routes=None, max_cost_value=None, customized_section_cost_function = None):
        """
        This method will return a list of tuples of origin node and travel time of the X fastest routes between
        a list of possible origin nodes and a certain destination node, whereas the route starts at certain origins can
        be offset. Combinations that dont fullfill all constraints will not be returned.
        :param list_origin_positions: list of origin_positions (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :param destination_position: destination position : (destination_edge_origin_node_index, destination_edge_destination_node_index, relative_position)
        :param max_routes: maximal number of fastest route triples that should be returned
        :param max_cost_value: latest cost function value of a route at destination to be considered as solution (max time if customized_section_cost_function == None)
        :param customized_section_cost_function: function to compute the travel cost of an section: args: (travel_time, travel_distance, current_dijkstra_node) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :return: list of (origin_position, cost_function_value, travel time, travel_distance) tuples
        """
        if customized_section_cost_function is not None:
            return super().return_travel_costs_Xto1(list_origin_positions, destination_position, max_routes=max_routes, max_cost_value=max_cost_value, customized_section_cost_function = customized_section_cost_function)
        origin_nodes = {}
        return_list = []
        for pos in list_origin_positions:
            trivial_test = self.test_and_get_trivial_route_tt_and_dis(pos, destination_position)
            if trivial_test is not None:
                if max_cost_value is not None and trivial_test[1][0] > max_cost_value:
                    continue
                return_list.append( (pos, trivial_test[1][0], trivial_test[1][1], trivial_test[1][2]))
                continue
            start_node = pos[0]
            if pos[1] is not None:
                start_node = pos[1]
            try:
                origin_nodes[start_node].append(pos)
            except:
                origin_nodes[start_node] = [pos]
        destination_node = destination_position[0]
        destination_overhead = (0.0, 0.0, 0.0)
        if destination_position[1] is not None:
            destination_overhead = self.get_section_overhead(destination_position, from_start=True)
        if len(origin_nodes.keys()) > 0:
            s = self.cpp_router.computeTravelCostsXto1(destination_node, origin_nodes.keys(), max_time_range = max_cost_value, max_targets = max_routes)
            for org_node, tt, dis in s:
                if tt < -0.0001:
                    continue
                self._add_to_database(org_node, destination_node, tt, tt, dis)
                tt += destination_overhead[1]
                dis += destination_overhead[2]
                for origin_position in origin_nodes[org_node]:
                    origin_overhead = (0.0, 0.0, 0.0)
                    if origin_position[1] is not None:
                        origin_overhead = self.get_section_overhead(origin_position, from_start=False)
                    tt += origin_overhead[1]
                    dis += origin_overhead[2]
                    if max_cost_value is not None and tt > max_cost_value:
                        #pass
                        continue
                    return_list.append( (origin_position, tt, tt, dis) )
        if max_routes is not None and len(return_list) > max_routes:
            return sorted(return_list, key = lambda x:x[1])[:max_routes]
        return return_list

    def return_travel_costs_1toX(self, origin_position, list_destination_positions, max_routes=None, max_cost_value=None, customized_section_cost_function = None):
        """
        This method will return a list of tuples of destination node and travel time of the X fastest routes between
        a list of possible origin nodes and a certain destination node, whereas the route starts at certain origins can
        be offset. Combinations that dont fullfill all constraints will not be returned.
        :param origin_position: origin_position: (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :param list_destination_positions: list of destination positions : (destination_edge_origin_node_index, destination_edge_destination_node_index, relative_position)
        :param max_routes: maximal number of fastest route triples that should be returned
        :param max_cost_value: latest cost function value of a route at destination to be considered as solution (max time if customized_section_cost_function == None)
        :param customized_section_cost_function: function to compute the travel cost of an section: args: (travel_time, travel_distance, current_dijkstra_node) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :return: list of (destination_position, cost_function_value, travel time, travel_distance) tuples
        """
        if customized_section_cost_function is not None:
            return super().return_travel_costs_1toX(origin_position, list_destination_positions, max_routes=max_routes, max_cost_value=max_cost_value, customized_section_cost_function = customized_section_cost_function)
        destination_nodes = {}
        return_list = []
        for pos in list_destination_positions:
            trivial_test = self.test_and_get_trivial_route_tt_and_dis(origin_position, pos)
            if trivial_test is not None:
                if max_cost_value is not None and trivial_test[1][0] > max_cost_value:
                    continue
                return_list.append( (pos, trivial_test[1][0], trivial_test[1][1], trivial_test[1][2]))
                continue
            start_node = pos[0]
            try:
                destination_nodes[start_node].append(pos)
            except:
                destination_nodes[start_node] = [pos]
        origin_node = origin_position[0]
        origin_overhead = (0.0, 0.0, 0.0)
        if origin_position[1] is not None:
            origin_node = origin_position[1]
            origin_overhead = self.get_section_overhead(origin_position, from_start=False)
        if len(destination_nodes.keys()) > 0:
            s = self.cpp_router.computeTravelCosts1toX(origin_node, destination_nodes.keys(), max_time_range = max_cost_value, max_targets = max_routes)
            for dest_node, tt, dis in s:
                if tt < -0.0001:
                    continue
                self._add_to_database(origin_node, dest_node, tt, tt, dis)
                tt += origin_overhead[1]
                dis += origin_overhead[2]
                for destination_position in destination_nodes[dest_node]:
                    destination_overhead = (0.0, 0.0, 0.0)
                    if destination_position[1] is not None:
                        destination_overhead = self.get_section_overhead(destination_position, from_start=True)
                    tt += destination_overhead[1]
                    dis += destination_overhead[2]
                    if max_cost_value is not None and tt > max_cost_value:
                        continue
                    return_list.append( (destination_position, tt, tt, dis) )
        if max_routes is not None and len(return_list) > max_routes:
            return sorted(return_list, key = lambda x:x[1])[:max_routes]
        return return_list

    def return_best_route_1to1(self, origin_position, destination_position, customized_section_cost_function = None):
        """
        This method will return the best route [list of node_indices] between two nodes,
        while origin_position[0] and destination_postion[1](or destination_position[0] if destination_postion[1]==None) is included.
        :param origin_position: (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :param destination_position: (destination_edge_origin_node_index, destination_edge_destination_node_index, relative_position)
        :param customized_section_cost_function: function to compute the travel cost of an section: args: (travel_time, travel_distance, current_dijkstra_node) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :return : route (list of node_indices) of best route
        """
        if customized_section_cost_function is not None:
            return super().return_best_route_1to1(origin_position, destination_position, customized_section_cost_function=customized_section_cost_function)
        trivial_test = self.test_and_get_trivial_route_tt_and_dis(origin_position, destination_position)
        if trivial_test is not None:
            return trivial_test[0]
        origin_node = origin_position[0]
        destination_node = destination_position[0]
        if origin_position[1] is not None:
            origin_node = origin_position[1]
        node_list = self.cpp_router.computeRoute1To1(origin_node, destination_node)
        if origin_node != origin_position[0]:
            node_list = [origin_position[0]] + node_list
        if destination_position[1] is not None:
            node_list.append(destination_position[1])
        return node_list
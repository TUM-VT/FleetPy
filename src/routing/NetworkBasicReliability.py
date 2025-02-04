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
from src.routing.NetworkBasic import Edge as BasicEdge
from src.routing.NetworkBasic import Node as BasicNode

#from src.routing.cpp_router.PyNetwork import PyNetwork
from src.routing.routing_imports.RouterReliability import RouterReliability
from src.routing.routing_imports.Router import Router

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)



INPUT_PARAMETERS_NetworkBasicReliability = {
    "doc" : "this routing class does all routing computations based on dijkstras algorithm and considers Traveltime Mean and Variance",
    "inherit" : "NetworkBase",
    "input_parameters_mandatory": [G_NETWORK_NAME],
    "input_parameters_optional": [G_NW_DYNAMIC_F],
    "mandatory_modules": [],
    "optional_modules": []
}
def read_node_line(columns):
    return Node(int(columns["node_index"]), int(columns["is_stop_only"]), float(columns["pos_x"]), float(columns["pos_y"]))


    
class Node(BasicNode):
    def __init__(self, node_index, is_stop_only, pos_x, pos_y, node_order=None):
        self.node_index = node_index
        self.is_stop_only = is_stop_only
        self.pos_x = pos_x
        self.pos_y = pos_y
        # 
        self.edges_to = {}  #node_obj -> edge
        self.edges_from = {}    #node_obj -> edge
        #
        self.travel_infos_from = {} #node_index -> (tt, dis)
        self.travel_infos_from_std = {} #node_index -> (std)
        self.travel_infos_to = {}   #node_index -> (tt, dis)
        self.travel_infos_to_std = {}   #node_index -> (std)
        #
        # attributes set during path calculations
        self.is_target_node = False     # is set and reset in computeFromNodes
        #attributes for forwards dijkstra
        self.prev = None
        self.settled = 1
        self.cost_index = -1
        self.cost = None
        # attributes for backwards dijkstra (for bidirectional dijkstra)
        self.next = None
        self.settled_back = 1
        self.cost_index_back = -1
        self.cost_back = None
    
    def add_next_edge_to(self, other_node, edge):
        #print("add next edge to: {} -> {}".format(self.node_index, other_node.node_index))
        self.edges_to[other_node] = edge
        self.travel_infos_to[other_node.node_index] = (edge.get_tt(),edge.get_distance(), edge.get_tt_std())

    def add_prev_edge_from(self, other_node, edge):
        self.edges_from[other_node] = edge
        self.travel_infos_from[other_node.node_index] = (edge.get_tt(),edge.get_distance(), edge.get_tt_std())

    def get_travel_infos_to(self, other_node_index):
        return self.travel_infos_to[other_node_index]

    def get_travel_infos_from(self, other_node_index):
        return self.travel_infos_from[other_node_index]
    

class Edge(BasicEdge):
    def __init__(self, edge_index, distance, travel_time,travel_time_std):
        self.edge_index = edge_index
        self.distance = distance
        self.travel_time = travel_time
        self.travel_time_std = travel_time_std
    
    def set_tt_std(self, travel_time_std):
        self.travel_time_std = travel_time_std
    
    def get_tt_std(self):
        """
        :return: (current) travel time standard deviation on edge
        """
        return self.travel_time_std
    
    def get_tt_mean_std(self):
        """
        :return: (travel time mean, travel time standard deviation) tuple
        """
        return (self.travel_time, self.travel_time_std)

class NetworkBasicReliability(NetworkBasic):
    def __init__(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        self.nodes = []     #list of all nodes in network (index == node.node_index)
        self.network_name_dir = network_name_dir
        self._tt_infos_from_folder = True
        self._current_tt_factor = None
        self.travel_time_file_infos = self._load_tt_folder_path(network_dynamics_file_name=network_dynamics_file_name)
        self.loadNetwork(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)
        self.current_dijkstra_number = 1    #used in dijkstra-class
        self.sim_time = 0   # TODO #
        self.zones = None   # TODO #
        with open(os.sep.join([self.network_name_dir, "base","crs.info"]), "r") as f:
            self.crs = f.read()   
        self.user_vot = None 
        self.user_vor = None
      
    def loadNetwork(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        nodes_f = os.path.join(network_name_dir, "base", "nodes.csv")
        print(f"Loading nodes from {nodes_f} ...")
        nodes_df = pd.read_csv(nodes_f)
        self.nodes = nodes_df.apply(read_node_line, axis=1)
        edges_f = os.path.join(network_name_dir, "base", "edges.csv")
        print(f"Loading edges from {edges_f} ...")
        edges_df = pd.read_csv(edges_f)
        for _, row in edges_df.iterrows():
            o_node = self.nodes[row[G_EDGE_FROM]]
            d_node = self.nodes[row[G_EDGE_TO]]
            if G_EDGE_TT_STD in row.keys():
                travel_time_std = row[G_EDGE_TT_STD]
            else:
                travel_time_std = 0
            tmp_edge = Edge(edge_index=(o_node, d_node), distance=row[G_EDGE_DIST], travel_time= row[G_EDGE_TT],travel_time_std=travel_time_std)
            o_node.add_next_edge_to(d_node, tmp_edge)
            d_node.add_prev_edge_from(o_node, tmp_edge)
        print("... {} nodes loaded!".format(len(self.nodes)))
        if scenario_time is not None:
            latest_tt = None
            if len(self.travel_time_file_folders.keys()) > 0:
                tts = sorted(list(self.travel_time_file_folders.keys()))
                for tt in tts:
                    if tt > scenario_time:
                        break
                    latest_tt = tt
                self.load_tt_file(latest_tt)
    
    def load_tt_file(self, scenario_time):
        """
        loads new travel time files for scenario_time
        """
        self._reset_internal_attributes_after_travel_time_update()
        f = self.travel_time_file_infos[scenario_time]
        if self._tt_infos_from_folder:
            tt_file = os.path.join(f, "edges_td_att.csv")
            tmp_df = pd.read_csv(tt_file)
            for from_node, to_node, edge_tt, edge_tt_std in zip(tmp_df[G_EDGE_FROM], tmp_df[G_EDGE_TO], tmp_df['edge_tt'], tmp_df[G_EDGE_TT_STD]):
                self._set_edge_tt(from_node, to_node, edge_tt,edge_tt_std)

    def _set_edge_tt(self, o_node_index, d_node_index, new_travel_time,new_travel_time_std):
        o_node = self.nodes[o_node_index]
        d_node = self.nodes[d_node_index]
        edge_obj = o_node.edges_to[d_node]
        edge_obj.set_tt(new_travel_time)
        edge_obj.set_tt_std(new_travel_time_std)
        new_tt, dis = edge_obj.get_tt_distance()
        o_node.travel_infos_to[d_node_index] = (new_tt, dis,new_travel_time_std)
        d_node.travel_infos_from[o_node_index] = (new_tt, dis,new_travel_time_std)
    

    def get_section_infos(self, start_node_index, end_node_index):
        """
        :param start_node_index_index: index of start_node of section
        :param end_node_index: index of end_node of section
        :return: (travel time, distance); if no section between nodes (None, None)
        """
        if self._current_tt_factor is None:
            tt, dis, tt_std = self.nodes[start_node_index].get_travel_infos_to(end_node_index)
            return (tt, dis, tt_std)
        else:
            tt, dis, tt_std = self.nodes[start_node_index].get_travel_infos_to(end_node_index)
            return (tt * self._current_tt_factor, dis,tt_std)
        
    def get_section_infos_std(self, start_node_index, end_node_index):
        """
        :param start_node_index_index: index of start_node of section
        :param end_node_index: index of end_node of section
        :return: (travel time, distance, travel time std); if no section between nodes (None, None)
        """
        if self._current_tt_factor is None:
            return self.nodes[start_node_index].get_travel_infos_to(end_node_index)
        else:
            tt, dis, tt_std = self.nodes[start_node_index].get_travel_infos_to(end_node_index)
            return (tt * self._current_tt_factor, dis, tt_std)
    

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
        trivial_test = self.test_and_get_trivial_route_tt_and_dis(origin_position, destination_position)
        if trivial_test is not None:
            return trivial_test[0]
        origin_node = origin_position[0]
        destination_node = destination_position[0]
        if origin_position[1] is not None:
            origin_node = origin_position[1]
        if customized_section_cost_function == None:
            customized_section_cost_function=self.customized_section_cost_function
        R = RouterReliability(self, origin_node, destination_nodes=[destination_node], mode='bidirectional', customized_section_cost_function=customized_section_cost_function)
        node_list = R.compute(return_route=True)[0][0]
        if origin_node != origin_position[0]:
            node_list = [origin_position[0]] + node_list
        if destination_position[1] is not None:
            node_list.append(destination_position[1])
        return node_list
    
    def return_travel_costs_1to1_reliability(self, origin_position, destination_position, customized_section_cost_function = None):
        """
        This method will return the travel costs of the fastest route between two nodes considering reliability.
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
        if customized_section_cost_function == None:
            customized_section_cost_function=self.customized_section_cost_function
        if self._current_tt_factor is None:
            R = RouterReliability(self, origin_node, destination_nodes=[destination_node], mode='bidirectional', customized_section_cost_function=customized_section_cost_function)                    
            s = R.compute(return_route=False)[0][1]
        else:
            R = Router(self, origin_node, destination_nodes=[destination_node], mode='bidirectional', customized_section_cost_function=customized_section_cost_function)
            s = R.compute(return_route=False)[0][1]
            s = (s[0] * self._current_tt_factor, s[1] * self._current_tt_factor, s[2])
        res = (s[0] + origin_overhead[0] + destination_overhead[0], s[1] + origin_overhead[1] + destination_overhead[1], s[2] + origin_overhead[2] + destination_overhead[2])
        if customized_section_cost_function is None:
            self._add_to_database(origin_node, destination_node, s[0], s[1], s[2])
        return res

    def customized_section_cost_function(self,tt_mean,tt_std,node_index=None):
        vor=self.user_vor
        vot=self.user_vot
        cfv = vot * tt_mean + vor * tt_std
        return cfv
    
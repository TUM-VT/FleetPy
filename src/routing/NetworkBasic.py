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
from pyproj import Transformer

# src imports
# -----------
from src.routing.NetworkBase import NetworkBase
from src.routing.routing_imports.Router import Router

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)

# import os
# import pandas as pd
# import imports.Router as Router

INPUT_PARAMETERS_NetworkBasic = {
    "doc" : "this routing class does all routing computations based on dijkstras algorithm",
    "inherit" : "NetworkBase",
    "input_parameters_mandatory": [G_NETWORK_NAME],
    "input_parameters_optional": [G_NW_DYNAMIC_F],
    "mandatory_modules": [],
    "optional_modules": []
}


def read_node_line(columns):
    return Node(int(columns["node_index"]), int(columns["is_stop_only"]), float(columns["pos_x"]), float(columns["pos_y"]))

class Node():
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
        self.travel_infos_to = {}   #node_index -> (tt, dis)
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

    def __str__(self):
        return str(self.node_index)

    def must_stop(self):
        return self.is_stop_only

    def get_position(self):
        return (self.pos_x, self.pos_y)

    def get_next_node_edge_pairs(self, ch_flag = False):
        """
        :return: list of (node, edge) tuples [references to objects] in forward direction
        """
        return self.edges_to.items()

    def get_prev_node_edge_pairs(self, ch_flag = False):
        """
        :return: list of (node, edge) tuples [references to objects] in backward direction
        """
        return self.edges_from.items()

    def add_next_edge_to(self, other_node, edge):
        #print("add next edge to: {} -> {}".format(self.node_index, other_node.node_index))
        self.edges_to[other_node] = edge
        self.travel_infos_to[other_node.node_index] = edge.get_tt_distance()

    def add_prev_edge_from(self, other_node, edge):
        self.edges_from[other_node] = edge
        self.travel_infos_from[other_node.node_index] = edge.get_tt_distance()

    def get_travel_infos_to(self, other_node_index):
        return self.travel_infos_to[other_node_index]

    def get_travel_infos_from(self, other_node_index):
        return self.travel_infos_from[other_node_index]



class Edge():
    def __init__(self, edge_index, distance, travel_time):
        self.edge_index = edge_index
        self.distance = distance
        self.travel_time = travel_time
        #

    def __str__(self):
        return "-".join(self.edge_index)

    def set_tt(self, travel_time):
        self.travel_time = travel_time

    def get_tt(self):
        """
        :return: (current) travel time on edge
        """
        return self.travel_time

    def get_distance(self):
        """
        :return: distance of edge
        """
        return self.distance

    def get_tt_distance(self):
        """
        :return: (travel time, distance) tuple
        """
        return (self.travel_time, self.distance)


# Position: (start_node_id, end_node_id, relative_pos)
#   -> (node_id, None, None) in case vehicle is on a node
#   -> relative_pos in [0.0, 1.0]
# A Route is defined as list of node-indices (int)
# while all given start-and end-position nodes are included


class NetworkBasic(NetworkBase):
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
        self.nodes = []     #list of all nodes in network (index == node.node_index)
        self.network_name_dir = network_name_dir
        self.travel_time_file_folders = self._load_tt_folder_path(network_dynamics_file_name=network_dynamics_file_name)
        self.loadNetwork(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)
        self.current_dijkstra_number = 1    #used in dijkstra-class
        self.sim_time = 0   # TODO #
        self.zones = None   # TODO #
        with open(os.sep.join([self.network_name_dir, "base","crs.info"]), "r") as f:
            self.crs = f.read()

    def loadNetwork(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        nodes_f = os.path.join(network_name_dir, "base", "nodes.csv")
        print(f"Loading nodes from {nodes_f} ...")
        nodes_df = pd.read_csv(nodes_f)
        self.nodes = nodes_df.apply(read_node_line, axis=1)
        #
        edges_f = os.path.join(network_name_dir, "base", "edges.csv")
        print(f"Loading edges from {edges_f} ...")
        edges_df = pd.read_csv(edges_f)
        for _, row in edges_df.iterrows():
            o_node = self.nodes[row[G_EDGE_FROM]]
            d_node = self.nodes[row[G_EDGE_TO]]
            tmp_edge = Edge((o_node, d_node), row[G_EDGE_DIST], row[G_EDGE_TT])
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

    def _load_tt_folder_path(self, network_dynamics_file_name=None):
        """ this method searches in the network-folder for travel_times folder. the name of the folder is defined by the simulation time from which these travel times are valid
        stores the corresponding time to trigger loading of new travel times ones the simulation time is reached.
        """
        tt_folders = {}
        if network_dynamics_file_name is None:
            LOG.info("... no network dynamics file given -> read folder structure")
            for f in os.listdir(self.network_name_dir):
                time = None
                try:
                    time = int(f)
                except:
                    continue
                tt_folders[time] = os.path.join(self.network_name_dir, f)
        else:
            LOG.info("... load network dynamics file: {}".format(os.path.join(self.network_name_dir, network_dynamics_file_name)))
            nw_dynamics_df = pd.read_csv(os.path.join(self.network_name_dir, network_dynamics_file_name))
            nw_dynamics_df.set_index("simulation_time", inplace=True)
            for sim_time, tt_folder_name in nw_dynamics_df["travel_time_folder"].items():
                tt_folders[int(sim_time)] = os.path.join(self.network_name_dir, str(tt_folder_name))
        return tt_folders

    def update_network(self, simulation_time, update_state = True):
        """This method can be called during simulations to update travel times (dynamic networks).

        :param simulation_time: time of simulation
        :type simulation_time: float
        :return: new_tt_flag True, if new travel times found; False if not
        :rtype: bool
        """
        LOG.debug(f"update network {simulation_time}")
        self.sim_time = simulation_time
        if update_state:
            if self.travel_time_file_folders.get(simulation_time, None) is not None:
                self.load_tt_file(simulation_time)
                return True
        return False
    
    def reset_network(self, simulation_time: float):
        """ this method is used in case a module changed the travel times to future states for forecasts
        it resets the network to the travel times a stimulation_time
        :param simulation_time: current simulation time"""
        sorted_tts = sorted(self.travel_time_file_folders.keys())
        if len(sorted_tts) > 2:
            for i in range(len(sorted_tts) - 1):
                if sorted_tts[i] <= simulation_time and sorted_tts[i+1] > simulation_time:
                    self.update_network(sorted_tts[i])
                    return
            if sorted_tts[-1] <= simulation_time:
                self.update_network(sorted_tts[-1])
                return

    def load_tt_file(self, scenario_time):
        """
        loads new travel time files for scenario_time
        """
        self._reset_internal_attributes_after_travel_time_update()
        f = self.travel_time_file_folders[scenario_time]
        tt_file = os.path.join(f, "edges_td_att.csv")
        tmp_df = pd.read_csv(tt_file)
        tmp_df.set_index(["from_node","to_node"], inplace=True)
        for edge_index_tuple, new_tt in tmp_df["edge_tt"].iteritems():
            self._set_edge_tt(edge_index_tuple[0], edge_index_tuple[1], new_tt)

    def _set_edge_tt(self, o_node_index, d_node_index, new_travel_time):
        o_node = self.nodes[o_node_index]
        d_node = self.nodes[d_node_index]
        edge_obj = o_node.edges_to[d_node]
        edge_obj.set_tt(new_travel_time)
        new_tt, dis = edge_obj.get_tt_distance()
        o_node.travel_infos_to[d_node_index] = (new_tt, dis)
        d_node.travel_infos_from[o_node_index] = (new_tt, dis)

    def get_node_list(self):
        """
        :return: list of node objects.
        """
        return self.nodes

    def get_number_network_nodes(self):
        return len(self.nodes)

    def get_must_stop_nodes(self):
        """ returns a list of node-indices with all nodes with a stop_only attribute """
        return [n.node_index for n in self.nodes if n.must_stop()]

    def return_position_from_str(self, position_str):
        a, b, c = position_str.split(";")
        if b == "-1":
            return (int(a), None, None)
        else:
            return (int(a), int(b), float(c))

    def return_node_coordinates(self, node_index):
        return self.nodes[node_index].get_position()

    def return_position_coordinates(self, position_tuple):
        """Returns the spatial coordinates of a position.

        :param position_tuple: (o_node, d_node, rel_pos) | (o_node, None, None)
        :return: (x,y) for metric systems
        """
        if position_tuple[1] is None:
            return self.return_node_coordinates(position_tuple[0])
        else:
            c0 = np.array(self.return_node_coordinates(position_tuple[0]))
            c1 = np.array(self.return_node_coordinates(position_tuple[1]))
            c_rel = position_tuple[2] * c1 + (1 - position_tuple[2]) * c0
            return c_rel[0], c_rel[1]

    def return_network_bounding_box(self):
        min_x = min([node.pos_x for node in self.nodes])
        max_x = max([node.pos_x for node in self.nodes])
        min_y = min([node.pos_y for node in self.nodes])
        max_y = max([node.pos_y for node in self.nodes])
        proj_transformer = Transformer.from_proj(self.crs, 'epsg:4326')
        lats, lons = proj_transformer.transform([min_x, max_x], [min_y, max_y])
        return list(zip(lons, lats))

    def return_positions_lon_lat(self, position_tuple_list: list) -> list:
        pos_list = [self.return_position_coordinates(pos) for pos in position_tuple_list]
        x, y = list(zip(*pos_list))
        proj_transformer = Transformer.from_proj(self.crs, 'epsg:4326')
        lats, lons = proj_transformer.transform(x, y)
        return list(zip(lons, lats))

    def get_section_infos(self, start_node_index, end_node_index):
        """
        :param start_node_index_index: index of start_node of section
        :param end_node_index: index of end_node of section
        :return: (travel time, distance); if no section between nodes (None, None)
        """
        return self.nodes[start_node_index].get_travel_infos_to(end_node_index)

    def return_route_infos(self, route, rel_start_edge_position, start_time):
        """
        This method returns the information travel information along a route. The start position is given by a relative
        value on the first edge [0,1], where 0 means that the vehicle is at the first node.
        :param route: list of nodes
        :param rel_start_edge_position: float [0,1] determining the start position
        :param start_time: can be used as an offset in case the route is planned for a future time
        :return: (arrival time, distance to travel)
        """
        arrival_time = start_time
        distance = 0
        _, start_tt, start_dis = self.get_section_overhead( (route[0], route[1], rel_start_edge_position), from_start=False)
        arrival_time += start_tt
        distance += start_dis
        if len(route) > 2:
            for i in range(2, len(route)):
                tt, dis = self.get_section_infos(route[i-1], route[i])
                arrival_time += tt
                distance += dis
        return (arrival_time, distance)

    def assign_route_to_network(self, route, start_time):
        """This method can be used for dynamic network models in which the travel times will be derived from the
        number of vehicles/routes assigned to the network.

        :param route: list of nodes
        :param start_time: can be used as an offset in case the route is planned for a future time
        :return:
        TODO
        """
        pass

    def get_section_overhead(self, position, from_start=True, customized_section_cost_function=None):
        """This method computes the section overhead for a certain position.

        :param position: (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :param from_start: computes already traveled travel_time and distance,
                           if False: computes rest travel time (relative_position -> 1.0-relative_position)
        :param customized_section_cost_function: customized routing objective function
        :return: (cost_function_value, travel time, travel_distance)
        """
        if position[1] is None:
            return 0.0, 0.0, 0.0
        all_travel_time, all_travel_distance = self.nodes[position[0]].get_travel_infos_to(position[1])
        overhead_fraction = position[2]
        if not from_start:
            overhead_fraction = 1.0 - overhead_fraction
        all_travel_cost = all_travel_time
        if customized_section_cost_function is not None:
            all_travel_cost = customized_section_cost_function(all_travel_time, all_travel_distance, self.nodes[position[1]])
        return all_travel_cost * overhead_fraction, all_travel_time * overhead_fraction, all_travel_distance * overhead_fraction

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
        R = Router(self, origin_node, destination_nodes=[destination_node], mode='bidirectional', customized_section_cost_function=customized_section_cost_function)
        s = R.compute(return_route=False)[0][1]
        res = (s[0] + origin_overhead[0] + destination_overhead[0], s[1] + origin_overhead[1] + destination_overhead[1], s[2] + origin_overhead[2] + destination_overhead[2])
        if customized_section_cost_function is None:
            self._add_to_database(origin_node, destination_node, s[0], s[1], s[2])
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
            R = Router(self, destination_node, destination_nodes=origin_nodes.keys(), time_radius = max_cost_value, max_settled_targets = max_routes, forward_flag = False, customized_section_cost_function=customized_section_cost_function)
            s = R.compute(return_route=False)
            for entry in s:
                cfv, tt, dis = entry[1]
                if cfv < 0 or cfv == float("inf"):
                    continue
                org_node = entry[0][0]
                if customized_section_cost_function is None:
                    self._add_to_database(org_node, destination_node, cfv, tt, dis)
                cfv += destination_overhead[0]
                tt += destination_overhead[1]
                dis += destination_overhead[2]
                for origin_position in origin_nodes[org_node]:
                    origin_overhead = (0.0, 0.0, 0.0)
                    if origin_position[1] is not None:
                        origin_overhead = self.get_section_overhead(origin_position, from_start=False)
                    cfv += origin_overhead[0]
                    tt += origin_overhead[1]
                    dis += origin_overhead[2]
                    if max_cost_value is not None and cfv > max_cost_value:
                        #pass
                        continue
                    return_list.append( (origin_position, cfv, tt, dis) )
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
            R = Router(self, origin_node, destination_nodes=destination_nodes.keys(), time_radius = max_cost_value, max_settled_targets = max_routes, forward_flag = True, customized_section_cost_function=customized_section_cost_function)
            s = R.compute(return_route=False)
            for entry in s:
                cfv, tt, dis = entry[1]
                if tt < 0 or cfv == float("inf"):
                    continue
                dest_node = entry[0][-1]
                if customized_section_cost_function is None:
                    self._add_to_database(origin_node, dest_node, cfv, tt, dis)
                cfv += origin_overhead[0]
                tt += origin_overhead[1]
                dis += origin_overhead[2]
                for destination_position in destination_nodes[dest_node]:
                    destination_overhead = (0.0, 0.0, 0.0)
                    if destination_position[1] is not None:
                        destination_overhead = self.get_section_overhead(destination_position, from_start=True)
                    cfv += destination_overhead[0]
                    tt += destination_overhead[1]
                    dis += destination_overhead[2]
                    if max_cost_value is not None and cfv > max_cost_value:
                        continue
                    return_list.append( (destination_position, cfv, tt, dis) )
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
        trivial_test = self.test_and_get_trivial_route_tt_and_dis(origin_position, destination_position)
        if trivial_test is not None:
            return trivial_test[0]
        origin_node = origin_position[0]
        destination_node = destination_position[0]
        if origin_position[1] is not None:
            origin_node = origin_position[1]
        R = Router(self, origin_node, destination_nodes=[destination_node], mode='bidirectional', customized_section_cost_function=customized_section_cost_function)
        node_list = R.compute(return_route=True)[0][0]
        if origin_node != origin_position[0]:
            node_list = [origin_position[0]] + node_list
        if destination_position[1] is not None:
            node_list.append(destination_position[1])
        return node_list

    def return_best_route_Xto1(self, list_origin_positions, destination_position, max_cost_value=None, customized_section_cost_function = None):
        """This method will return the best route between a list of possible origin nodes and a certain destination
        node. A best route is defined by [list of node_indices] between two nodes,
        while origin_position[0] and destination_position[1](or destination_position[0]
        if destination_position[1]==None) is included. Combinations that do not fulfill all constraints
        will not be returned.

        :param list_origin_positions: list of origin_positions
                (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :type list_origin_positions: list
        :param destination_position: (origin_node_index, destination_node_index, relative_position) of destination_edge
        :type destination_position: list
        :param max_cost_value: latest cost function value of a route at destination to be considered as solution
                (max time if customized_section_cost_function == None)
        :type max_cost_value: float/None
        :param customized_section_cost_function: function to compute the travel cost of an section
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: list of node-indices of the fastest route (empty, if no route is found, that fullfills the constraints)
        :rtype: list
        """
        origin_nodes = {}
        return_route = None
        best_cfv = float("inf")
        for pos in list_origin_positions:
            trivial_test = self.test_and_get_trivial_route_tt_and_dis(pos, destination_position)
            if trivial_test is not None:
                if max_cost_value is not None and trivial_test[1][0] > max_cost_value:
                    continue
                if trivial_test[1][0] < best_cfv:
                    return_route = trivial_test[0]
                    best_cfv = trivial_test[1][0]
                continue
            start_node = pos[0]
            if pos[1] is not None:
                start_node = pos[1]
            try:
                origin_nodes[start_node].append(pos)
            except:
                origin_nodes[start_node] = [pos]
        if len(origin_nodes.keys()) > 0:
            destination_node = destination_position[0]
            destination_overhead = (0.0, 0.0, 0.0)
            if destination_position[1] is not None:
                destination_overhead = self.get_section_overhead(destination_position, from_start=True)
            R = Router(self, destination_node, destination_nodes=origin_nodes.keys(), time_radius = max_cost_value, forward_flag = False, customized_section_cost_function=customized_section_cost_function)
            s = R.compute(return_route=True)
            for entry in s:
                cfv, tt, dis = entry[1]
                if tt < 0:
                    continue
                cfv += destination_overhead[0]
                tt += destination_overhead[1]
                dis += destination_overhead[2]
                org_node = entry[0][0]
                for origin_position in origin_nodes[org_node]:
                    origin_overhead = (0.0, 0.0, 0.0)
                    if origin_position[1] is not None:
                        origin_overhead = self.get_section_overhead(origin_position, from_start=False)
                    cfv += origin_overhead[0]
                    tt += origin_overhead[1]
                    dis += origin_overhead[2]
                    if max_cost_value is not None and cfv > max_cost_value:
                        continue
                    if cfv > best_cfv:
                        continue
                    node_list = entry[0][:]
                    if origin_position[1] is not None:
                        if destination_position[1] is not None:
                            node_list = [origin_position[0]] + node_list + [destination_position[1]]
                        else:
                            node_list = [origin_position[0]] + node_list
                    else:
                        if destination_position[1] is not None:
                            node_list = node_list + [destination_position[1]]
                    best_cfv = cfv
                    if len(node_list) < 2:
                        return_route = []
                    else:
                        return_route = node_list 
        return return_route

    def return_best_route_1toX(self, origin_position, list_destination_positions, max_cost_value=None, customized_section_cost_function = None):
        """This method will return the best route between a list of possible destination nodes and a certain origin
        node. A best route is defined by [list of node_indices] between two nodes,
        while origin_position[0] and destination_position[1](or destination_position[0]
        if destination_position[1]==None) is included. Combinations that do not fulfill all constraints
        will not be returned.
        Specific to this framework: max_cost_value = None translates to max_cost_value = DEFAULT_MAX_X_SEARCH

        :param origin_position: (origin_node_index, destination_node_index, relative_position) of origin edge
        :type origin_position: list
        :param list_destination_positions: list of destination positions
                (origin_node_index, destination_node_index, relative_position) of destination_edge
        :type list_destination_positions: list
        :param max_cost_value: latest cost function value of a route at destination to be considered as solution
                (max time if customized_section_cost_function == None)
        :type max_cost_value: float/None
        :param customized_section_cost_function: function to compute the travel cost of an section
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: list of node-indices of the fastest route (empty, if no route is found, that fullfills the constraints)
        :rtype: list
        """
        destination_nodes = {}
        return_route = []
        best_cfv = float("inf")
        for pos in list_destination_positions:
            trivial_test = self.test_and_get_trivial_route_tt_and_dis(origin_position, pos)
            if trivial_test is not None:
                if max_cost_value is not None and trivial_test[1][0] > max_cost_value:
                    continue
                if trivial_test[1][0] < best_cfv:
                    return_route = trivial_test[0]
                    best_cfv = trivial_test[1][0]
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
            R = Router(self, origin_node, destination_nodes=destination_nodes.keys(), time_radius = max_cost_value, forward_flag = True)
            s = R.compute(return_route=True)
            for entry in s:
                cfv, tt, dis = entry[1]
                if tt < 0:
                    continue
                cfv += origin_overhead[0]
                tt += origin_overhead[1]
                dis += origin_overhead[2]
                dest_node = entry[0][-1]
                for destination_position in destination_nodes[dest_node]:
                    destination_overhead = (0.0, 0.0, 0.0)
                    if destination_position[1] is not None:
                        destination_overhead = self.get_section_overhead(destination_position, from_start=True)
                    cfv += destination_overhead[0]
                    tt += destination_overhead[1]
                    dis += destination_overhead[2]
                    if max_cost_value is not None and cfv > max_cost_value:
                        continue
                    if cfv > best_cfv:
                        continue
                    node_list = entry[0][:]
                    if origin_position[1] is not None:
                        if destination_position[1] is not None:
                            node_list = [origin_position[0]] + node_list + [destination_position[1]]
                        else:
                            node_list = [origin_position[0]] + node_list
                    else:
                        if destination_position[1] is not None:
                            node_list = node_list + [destination_position[1]]
                    best_cfv = cfv
                    if len(node_list) < 2:
                        return_route = []
                    else:
                        return_route = node_list 

        return return_route

    def test_and_get_trivial_route_tt_and_dis(self, origin_position, destination_position):
        """ this functions test for trivial routing solutions between origin_position and destination_position
        if no trivial solution is found
        :return None
        else
        :return (route, (travel_time, travel_distance))
        """
        if origin_position[0] == destination_position[0]:
            if origin_position[1] is None:
                if destination_position[1] is None:
                    return ([], (0.0, 0.0, 0.0) )
                else:
                    return ([destination_position[0], destination_position[1]], self.get_section_overhead(destination_position) )
            else:
                if destination_position[1] is None:
                    return None
                else:
                    if destination_position[1] == origin_position[1]:
                        if origin_position[2] > destination_position[2]:
                            return None
                        else:
                            effective_position = (origin_position[0], origin_position[1], destination_position[2] - origin_position[2])
                            cfv, tt, dis = self.get_section_overhead(effective_position, from_start = True)
                            return ([destination_position[0], destination_position[1]], (cfv, tt, dis)) 
                    else:
                        return None
        elif origin_position[1] is not None and origin_position[1] == destination_position[0]:
            rest = self.get_section_overhead(origin_position, from_start = False)
            rest_dest = self.get_section_overhead(destination_position, from_start = True)
            route = [origin_position[0], origin_position[1]]
            #print(f"nw basic argh {rest} {rest_dest}")
            if destination_position[1] is not None:
                route.append( destination_position[1] )
            return (route, (rest[0] + rest_dest[0], rest[1] + rest_dest[1], rest[2] + rest_dest[2]))
        return None

    def return_travel_cost_matrix(self, list_positions, customized_section_cost_function = None):
        """This method will return the cost_function_value between all positions specified in list_positions

        :param list_positions: list of positions to be computed
        :type list_positions: list
        :param customized_section_cost_function: function to compute the travel cost of an section
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: dictionary: (o_pos,d_pos) -> (cfv, tt, dist)
        :rtype: dict
        """
        return_dict = {}
        for o_pos in list_positions:
            res = self.return_travel_costs_1toX(o_pos, list_positions, customized_section_cost_function=customized_section_cost_function)
            for d_pos, cfv, tt, dist in res:
                return_dict[(o_pos, d_pos)] = (cfv, tt, dist)
        return return_dict

    def move_along_route(self, route, last_position, time_step, sim_vid_id=None, new_sim_time=None,
                         record_node_times=False): # TODO # correct first entry of route!!!!
        """This method computes the new position of a (vehicle) on a given route (node_index_list) from it's
        last_position (position_tuple). The first entry of route has to be the same as the first entry of last_position!
        Specific to this framework: count moving vehicles to street network density! make sure to do this before
        updating the network!

        :param route: list of node_indices of the current route
        :type route: list
        :param last_position: position_tuple of starting point
        :type last_position: list
        :param time_step: time [s] passed since last observed at last_position
        :type time_step: float
        :param sim_vid_id: id of simulation vehicle; required for simulation environments with external traffic simulator
        :type sim_vid_id: int
        :param new_sim_time: new time to coordinate simulation times
        :type new_sim_time: float
        :param record_node_times: if this flag is set False, the output list_passed_node_times will always return []
        :type record_node_times: bool
        :return: returns a tuple with
                i) new_position_tuple
                ii) driven distance
                iii) arrival_in_time_step [s]: -1 if vehicle did not reach end of route | time since beginning of time
                        step after which the vehicle reached the end of the route
                iv) list_passed_nodes: if during the time step a number of nodes were passed, these are
                v) list_passed_node_times: list of checkpoint times at the respective passed nodes
        """
        if new_sim_time is not None:
            end_time = new_sim_time + time_step
            last_time = new_sim_time
        else:
            end_time = self.sim_time + time_step
            last_time = self.sim_time
        c_pos = last_position
        if c_pos[2] is None:
            if len(route) == 0:
                return c_pos, 0, last_time, [], []
            c_pos = (c_pos[0], route[0], 0.0)
        list_passed_nodes = []
        list_passed_node_times = []
        arrival_in_time_step = -1
        driven_distance = 0
        #
        c_cluster = None
        last_dyn_step = None
        for i in range(len(route)):
            # check remaining time on current edge
            if c_pos[2] is None:
                c_pos = (c_pos[0], route[i], 0)
            rel_factor = (1 - c_pos[2])
            tt, td = self.nodes[c_pos[0]].get_travel_infos_to(c_pos[1])
            if tt > 86400:
                LOG.warning(f"move_along_route: very large travel time on edge ({c_pos[0]} -> {c_pos[1]} for vid {sim_vid_id} at time {new_sim_time}) (blocked after tt update?) -> vehicle jumps this edge")
                tt = 0
            c_edge_tt = tt
            c_edge_td = td
            next_node_time = last_time + rel_factor * c_edge_tt
            if next_node_time > end_time:
                # move vehicle to final position of current edge
                end_rel_factor = (end_time - last_time) / tt + c_pos[2]
                #print(end_rel_factor, end_time, last_time, c_edge_tt, c_pos[2])
                driven_distance += (end_rel_factor - c_pos[2]) * c_edge_td
                c_pos = (c_pos[0], c_pos[1], end_rel_factor)
                arrival_in_time_step = -1
                break
            else:
                # move vehicle to next node/edge and record data
                driven_distance += rel_factor * c_edge_td
                next_node = route[i]
                list_passed_nodes.append(next_node)
                if record_node_times:
                    list_passed_node_times.append(next_node_time)
                last_time = next_node_time
                c_pos = (next_node, None, None)
                arrival_in_time_step = last_time
        return c_pos, driven_distance, arrival_in_time_step, list_passed_nodes, list_passed_node_times

    def add_travel_infos_to_database(self, travel_info_dict):
        """ this function can be used to include externally computed (e.g. multiprocessing) route travel times
        into the database if present

        it adds all infos from travel_info_dict to its database self.travel_time_infos
        its database is from node to node, therefore overheads have to be removed from routing results

        :param travel_info_dict: dictionary with keys (origin_position, target_positions) -> values (cost_function_value, travel_time, travel_distance)
        """
        pass

    def _reset_internal_attributes_after_travel_time_update(self):
        pass

    def _add_to_database(self, o_node, d_node, cfv, tt, dis):
        """ this function is call when new routing results have been computed
        depending on the class the function can be overwritten to store certain results in the database
        """
        pass

    
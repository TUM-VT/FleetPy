# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import os
import logging
import pathlib

# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
import numpy as np
from pyproj import Transformer

# src imports
# -----------
from src.routing.NetworkBase import NetworkBase

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)
EPS = 0.001
DEFAULT_MAX_X_SEARCH = 6000
LARGE_INT = 10000000


# -------------------------------------------------------------------------------------------------------------------- #
# functions
# ---------
def read_node_line(columns):
    return Node(int(columns[G_NODE_ID]), int(columns[G_NODE_STOP_ONLY]),
                float(columns[G_NODE_X]), float(columns[G_NODE_Y]))


# -------------------------------------------------------------------------------------------------------------------- #
# help classes
# ------------
class Node:
    def __init__(self, node_index, is_stop_only, pos_x, pos_y):
        # node_id -> node_obj
        self.to_nodes = {}
        self.from_nodes = {}
        #
        self.surround_next = {}  # max search -> set of d_node indices [dict_keys]
        self.surround_prev = {}  # max search -> set of o_node indices [dict_keys]
        self.node_index = node_index
        self.is_stop_only = is_stop_only
        self.pos_x = pos_x
        self.pos_y = pos_y

        self.zone_id = None

    def __str__(self):
        return str(self.node_index)

    def add_next_edge_to(self, other_node):
        self.to_nodes[other_node.node_index] = other_node

    def add_prev_edge_from(self, other_node):
        self.from_nodes[other_node.node_index] = other_node

    def must_stop(self):
        return self.is_stop_only

    def set_surround_prev(self, time_range, list_surround_prev): 
        if time_range > 600:
            return
        self.surround_prev[time_range] = list_surround_prev

    def set_surround_next(self, time_range, list_surround_next):
        if time_range > 600:
            return
        self.surround_next[time_range] = list_surround_next

    def set_zone_id(self, zone_id):
        self.zone_id = zone_id

    def get_zone_id(self):
        return self.zone_id

# -------------------------------------------------------------------------------------------------------------------- #
# module class
# ------------
INPUT_PARAMETERS_NetworkTTMatrix = {
    "doc" : """
        Routing based on a preprocessed TT-Matrix. For changing travel-times, tt-scaling factors can be read from a file.
        see: src/preprocessing/networks/create_travel_time_tables.py for creating the numpy tables.
        """,
    "inherit" : "NetworkBase",
    "input_parameters_mandatory": [G_NETWORK_NAME],
    "input_parameters_optional": [G_NW_DYNAMIC_F],
    "mandatory_modules": [],
    "optional_modules": []
}

class NetworkTTMatrix(NetworkBase):
    """Routing based on TT Matrix, tt-scaling factor is read from file."""
    def __init__(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        """The network will be initialized.

        :param network_name_dir: name of the network_directory to be loaded
        :type network_name_dir: str
        :param type: determining whether the base or a pre-processed network will be used
        :type type: str
        :param scenario_time: applying travel times from a certain date (given by string "scenario_time_tt-infos.csv")
        :type scenario_time: str
        :param network_dynamics_file_name: file-name of the network dynamics file
        :type network_dynamics_file_name: str
        """
        super().__init__(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)
        self.zones = None
        self.tt_factor = 1.0
        self.network_name_dir = network_name_dir
        with open(os.path.join(self.network_name_dir, "base", "crs.info"), "r") as f:
            self.crs = f.read()
        # load network structure: nodes
        nodes_f = os.path.join(network_name_dir, "base", "nodes.csv")
        print(f"\t ... loading nodes from {nodes_f} ...")
        nodes_df = pd.read_csv(nodes_f)
        self.nodes = nodes_df.apply(read_node_line, axis=1).to_list()
        self.number_nodes = len(self.nodes)
        # load network structure: edges
        edges_f = os.path.join(network_name_dir, "base", "edges.csv")
        print(f"\t ... loading edges from {edges_f} ...")
        edges_df = pd.read_csv(edges_f)
        for _, edge_info_row in edges_df.iterrows():
            o_node_index = int(edge_info_row[G_EDGE_FROM])
            o_node_obj = self.nodes[o_node_index]
            d_node_index = int(edge_info_row[G_EDGE_TO])
            d_node_obj = self.nodes[d_node_index]
            if o_node_index != d_node_index:
                o_node_obj.add_next_edge_to(d_node_obj)
                d_node_obj.add_prev_edge_from(o_node_obj)
        # load TT and TD matrices
        print(f"\t ... loading network travel time/distance tables ...")
        tt_table_f = os.path.join(self.network_name_dir, "ff", "tables", "nn_fastest_tt.npy")
        self.tt_numpy = np.load(tt_table_f)
        self.tt = self.tt_numpy.tolist()
        distance_table_f = os.path.join(self.network_name_dir, "ff", "tables", "nn_fastest_distance.npy")
        self.td = np.load(distance_table_f).tolist()
        # load travel times
        self.current_tt_factor_index = 0
        self.sorted_tt_factor_times = []
        self.update_tt_factors = {}
        self._precalculated_tt_paths = {}
        self._current_tt_path = None
        self.load_tt_file(None)
        self.sim_time = 0
        self._load_dynamic_network(network_dynamics_file_name)

    def _load_dynamic_network(self, file_or_folder):
        """ Prepares the network for the dynamic travel times using time-dependent scaling factor or travel time
        matrices. If both scaling factor file and the folder (with precalculated travel time matrices) exist, the
        preference is given to the scaling factor file. """

        if file_or_folder is not None:
            path = pathlib.Path(self.network_name_dir, file_or_folder)
            # First check if its scaling factor file
            if path.is_file():
                path = pathlib.Path(self.network_name_dir, file_or_folder)
                nw_dynamics_df = pd.read_csv(str(path))
                nw_dynamics_df.set_index("simulation_time", inplace=True)
                if "travel_time_factor" in nw_dynamics_df.columns:
                    for sim_time, tt_factor in nw_dynamics_df["travel_time_factor"].items():
                        self.update_tt_factors[int(sim_time)] = tt_factor
                    self.sorted_tt_factor_times = sorted(self.update_tt_factors.keys())
                    LOG.info(f"Loaded travel time scaling factors from {str(path)}")
                elif "travel_time_folder" in nw_dynamics_df.columns:
                    for sim_time, tt_folder_path in nw_dynamics_df["travel_time_folder"].items():
                        self._precalculated_tt_paths[int(sim_time)] = pathlib.Path(self.network_name_dir, tt_folder_path)
                    self.sorted_tt_factor_times = sorted(self._precalculated_tt_paths.keys())
                    LOG.info(f"Loaded travel time folders from {str(path)}")
                # elif "travel_time_folder" in nw_dynamics_df.columns:
                #     for sim_time, tt_folder in nw_dynamics_df["travel_time_folder"].items():
                #         self._precalculated_tt_paths[sim_time] = path.parent.joinpath(tt_folder)
                #     self.sorted_tt_factor_times = sorted(self._precalculated_tt_paths.keys())
                else:
                    raise IOError(f"The file {str(path)} does not contain travel_time_factor or travel_time_folder "
                                  f"column")
            elif path.is_dir():
                if len(list(path.iterdir())) == 0:
                    raise IOError(f"Did not find any folder for the precalculated travel time matrices for dynamic "
                                  f"travel times within {str(path)}")
                for folder in path.iterdir():
                    self._precalculated_tt_paths[int(folder.name)] = folder
                self.sorted_tt_factor_times = sorted(self._precalculated_tt_paths.keys())
            else:
                raise IOError(f"Did not find network dynamics scaling factor file or precalculated travel time "
                              f"matrices folder in {file_or_folder}")
        else:
            LOG.info("No travel time scaling factor file or precalculated dynamic travel time folder given.")

    def load_tt_file(self, scenario_time):
        pass

    def add_init_data(self, tt_factor_f):
        pass

    def update_network(self, simulation_time, update_state=True):
        """This method can be called during simulations to check whether a new travel time file should be loaded
        (deterministic networks)

        :param simulation_time: time of simulation
        :type simulation_time: float
        :return: True, if new travel times found; False if not
        :rtype: bool
        """
        self.sim_time = simulation_time
        tt_updated = False
        if self.sorted_tt_factor_times:
            for inx, next_time in enumerate(self.sorted_tt_factor_times[self.current_tt_factor_index:],
                                            self.current_tt_factor_index):
                if next_time <= simulation_time:
                    self.current_tt_factor_index = inx
                    if self.update_tt_factors:
                        new_tt_factor = self.update_tt_factors[next_time]
                        if self.tt_factor != new_tt_factor:
                            self.tt_factor = new_tt_factor
                            tt_updated = True
                    elif self._precalculated_tt_paths:
                        path = self._precalculated_tt_paths[next_time]
                        if self._current_tt_path != path:
                            self.tt_numpy = np.load(path.joinpath("tt_matrix.npy"))
                            self.tt = self.tt_numpy.tolist()
                            self._current_tt_path = path
                            tt_updated = True
                    if tt_updated is True:
                        LOG.info("update network at {}".format(simulation_time))
                        break
        return tt_updated
    
    def reset_network(self, simulation_time: float):
        """ this method is used in case a module changed the travel times to future states for forecasts
        it resets the network to the travel times a stimulation_time
        :param simulation_time: current simulation time"""
        LOG.debug("reset network at {}".format(simulation_time))
        if self.sorted_tt_factor_times:
            self.current_tt_factor_index = 0
            if len(self.sorted_tt_factor_times) > 2:
                for i in range(len(self.sorted_tt_factor_times) - 1):
                    if self.sorted_tt_factor_times[i] <= simulation_time and self.sorted_tt_factor_times[i+1] > simulation_time:
                        self.update_network(self.sorted_tt_factor_times[i], update_state=True)
                        return
                if self.sorted_tt_factor_times[-1] <= simulation_time:
                    self.update_network(self.sorted_tt_factor_times[-1], update_state=True)
                    return

    def get_number_network_nodes(self):
        """This method returns a list of all street network node indices.

        :return: number of network nodes
        :rtype: int
        """
        return self.number_nodes

    def get_must_stop_nodes(self):
        """ returns a list of node-indices with all nodes with a stop_only attribute """
        return [n.node_index for n in self.nodes if n.must_stop()]

    def return_node_coordinates(self, node_index):
        """Returns the spatial coordinates of a node.

        :param node_index: id of node
        :type node_index: int
        :return: (x,y) for metric systems
        :rtype: list
        """
        node_obj = self.nodes[node_index]
        return node_obj.pos_x, node_obj.pos_y

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

    def get_section_infos(self, start_node_index, end_node_index):
        """Returns travel time and distance information of a section.

        :param start_node_index: index of start_node of section
        :param end_node_index: index of end_node of section
        :return: (travel time, distance); if no section between nodes (None, None)
        :rtype: list
        """
        tt = self.tt[start_node_index][end_node_index]
        scaled_tt = tt * self.tt_factor
        if tt not in [None, np.nan, np.inf]:
            return scaled_tt, self.td[start_node_index][end_node_index]
        else:
            return None, None

    def return_route_infos(self, route, rel_start_edge_position, start_time=0):
        """This method returns the information travel information along a route. The start position is given by a
        relative value on the first edge [0,1], where 0 means that the vehicle is at the first node.

        :param route: list of node ids
        :type route: list
        :param rel_start_edge_position: float [0,1] determining the start position
        :type rel_start_edge_position: float
        :param start_time: can be used as an offset in case the route is planned for a future time
        :type start_time: float
        :return: (arrival time, distance to travel)
        :rtype: list
        """
        if not route or len(route) < 2:
            return 0,0
        rel_pos = (route[0], route[1], rel_start_edge_position)
        ff_time, sum_distance = self._get_section_overhead(rel_pos, False)
        sum_time = ff_time * self.tt_factor
        i = 2
        while i < len(route):
            scaled_route_infos = self.get_section_infos(route[i-1], route[i])
            sum_time += scaled_route_infos[0]
            sum_distance += scaled_route_infos[1]
            i += 1
        sum_time += start_time
        return sum_time, sum_distance

    def assign_route_to_network(self, route, start_time, end_time=None, number_vehicles=1):
        """This method can be used for dynamic network models in which the travel times will be derived from the
        number of vehicles/routes assigned to the network.

        :param route: list of nodes
        :type route: list
        :param start_time: start of travel, can be used as an offset in case the route is planned for a future time
        :type start_time: float
        :param end_time: optional parameter; can be used to assign a vehicle to the cluster of the first node of the
                    route for a certain time
        :type end_time: float
        :param number_vehicles: optional parameter; can be used to assign multiple vehicles at once
        :type number_vehicles: int
        """
        pass

    def move_along_route(self, route, last_position, time_step, sim_vid_id=None, new_sim_time=None,
                         record_node_times=False):
        if new_sim_time is not None:
            end_time = new_sim_time + time_step
            last_time = new_sim_time
        else:
            end_time = self.sim_time + time_step
            last_time = self.sim_time
        c_pos = last_position
        if c_pos[2] is None:
            c_pos = (c_pos[0], route[0], 0.0)
        list_passed_nodes = []
        list_passed_node_times = []
        arrival_in_time_step = -1
        driven_distance = 0
        #
        for i in range(len(route)):
            # check remaining time on current edge
            if c_pos[2] is None:
                c_pos = (c_pos[0], route[i], 0)
            rel_factor = (1 - c_pos[2])
            # c_edge_tt = rel_factor * self.tt_factor * self.tt[c_pos[0]][c_pos[1]]
            # next_node_time = last_time + c_edge_tt
            c_edge_tt = self.tt_factor * self.tt[c_pos[0]][c_pos[1]]
            next_node_time = last_time + rel_factor * c_edge_tt
            end_time = np.round(end_time, 2)
            next_node_time = np.round(next_node_time, 2)  # TODO # raw value leads to 1.00000002 being recognized as > 1
            # TODO # this occasionally led to travel times 1 second shorter than direct travel times
            if next_node_time > end_time:
                # move vehicle to final position of current edge
                end_rel_factor = c_pos[2] + (end_time - last_time) / c_edge_tt
                driven_distance += (end_rel_factor - c_pos[2]) * self.td[c_pos[0]][c_pos[1]]
                c_pos = (c_pos[0], c_pos[1], end_rel_factor)
                arrival_in_time_step = -1
                break
            else:
                # move vehicle to next node/edge and record data
                driven_distance += rel_factor * self.td[c_pos[0]][c_pos[1]]
                next_node = route[i]
                list_passed_nodes.append(next_node)
                if record_node_times:
                    list_passed_node_times.append(next_node_time)
                last_time = next_node_time
                c_pos = (next_node, None, None)
                arrival_in_time_step = last_time
        return c_pos, driven_distance, arrival_in_time_step, list_passed_nodes, list_passed_node_times

    def return_travel_costs_1to1(self, origin_position, destination_position, customized_section_cost_function=None):
        """This method will return the travel costs of the fastest route between two nodes.

        :param origin_position: (origin_node_index, destination_node_index, relative_position) of current_edge
        :type origin_position: list
        :param destination_position: (origin_node_index, destination_node_index, relative_position) of destination_edge
        :type destination_position: list
        :param customized_section_cost_function: function to compute the travel cost of an section:
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: (cost_function_value, travel time, travel_distance) between the two nodes
        :rtype: list
        """
        origin_node, destination_node, add_tt, add_dist = self._get_od_nodes_and_section_overheads(origin_position,
                                                                                                   destination_position)
        # matrix lookup
        tt = self.tt[origin_node.node_index][destination_node.node_index]
        dist = self.td[origin_node.node_index][destination_node.node_index]
        # scaling
        scaled_tt = (add_tt + tt) * self.tt_factor
        return scaled_tt, scaled_tt, dist + add_dist

    def return_best_route_1to1(self, origin_position, destination_position, customized_section_cost_function=None):
        """This method will return the best route [list of node indices] between two nodes, where origin_position[0] and
        destination_position[1] or (destination_position[0] if destination_postion[1]==None) are included

        :param origin_position: (origin_node_index, destination_node_index, relative_position) of current_edge
        :type origin_position: list
        :param destination_position: (origin_node_index, destination_node_index, relative_position) of destination_edge
        :type destination_position: list
        :param customized_section_cost_function: function to compute the travel cost of an section
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: list of node-indices of the fastest route
        """
        origin_node, destination_node, add_tt, add_dist = self._get_od_nodes_and_section_overheads(origin_position,
                                                                                                   destination_position)
        if origin_node == destination_node:
            if destination_position[1] is None:
                return [origin_node.node_index]
            else:
                return [origin_node.node_index, destination_position[1]]
        node_list, _ = self._lookup_dijkstra_1to1(origin_node, destination_node)
        if origin_node.node_index != origin_position[0]:
            node_list = [origin_position[0]] + node_list
        if destination_position[1] is not None:
            node_list.append(destination_position[1])
        return node_list

    def return_travel_costs_Xto1(self, list_origin_positions, destination_position, max_routes=None,
                                 max_cost_value=None, customized_section_cost_function=None):
        """This method will return a list of tuples of origin positions and cost values of the X fastest routes between
        a list of possible origin nodes and a certain destination node. Combinations that do not fulfill all constraints
        will not be returned.
        Specific to this framework: max_cost_value = None translates to max_cost_value = DEFAULT_MAX_X_SEARCH

        :param list_origin_positions: list of origin_positions
                (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :type list_origin_positions: list
        :param destination_position: (origin_node_index, destination_node_index, relative_position) of destination_edge
        :type destination_position: list
        :param max_routes: maximal number of fastest route triples that should be returned
        :type max_routes: int/None
        :param max_cost_value: latest cost function value of a route at destination to be considered as solution
                (max time if customized_section_cost_function == None)
        :type max_cost_value: float/None
        :param customized_section_cost_function: function to compute the travel cost of an section
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: list of (origin_position, cost_function_value, travel time, travel_distance) tuples
        :rtype: list
        """
        # LOG.debug("return_tc_Xto1 inputs {} {} {} {}".format(destination_position, max_cost_value, max_routes,
        #                                                      customized_section_cost_function))
        # get list of (origin nodes, add_tt, add_td) -> assign positions to origin_node
        if len(list_origin_positions) == 0:
            return []
        # assumption: d-pos is on node
        d_node = self.nodes[destination_position[0]]
        # set default max_cost_value if not available
        if max_cost_value is None:
            max_time = DEFAULT_MAX_X_SEARCH
        else:
            max_time = max_cost_value
        # check if prev valid for destination node
        set_d_surround_prev = d_node.surround_prev.get(max_time)
        if set_d_surround_prev is None:
            set_d_surround_prev = self._compute_d_surround_prev(d_node, max_time)
        # LOG.debug("return_tc_Xto1 max_time? {} set_d_surround_prev? {}".format(max_time, set_d_surround_prev))
        return_list = []
        for o_pos in list_origin_positions:
            if o_pos[0] in set_d_surround_prev or o_pos[1] in set_d_surround_prev:
                cfv, scaled_route_tt, td = self.return_travel_costs_1to1(o_pos, destination_position,
                                                                         customized_section_cost_function)
                # still check as surrounding is computed for free flow condition
                if not max_cost_value or scaled_route_tt <= max_cost_value:
                    return_list.append((o_pos, cfv, scaled_route_tt, td))
        # sort if only limited amount of routes should be returned
        if max_routes:
            return_list.sort(key=lambda x: x[1])
            return_list = return_list[:max_routes]
        # if len(return_list) == 0:
        #     LOG.warning("no route found for return_travel_costs_Xto1 to target {} number origins {} time range
        #     {}".format(destination_position, len(list_origin_positions), max_cost_value))
        return return_list

    def return_best_route_Xto1(self, list_origin_positions, destination_position, max_cost_value=None,
                               customized_section_cost_function = None):
        """This method will return the best route between a list of possible origin nodes and a certain destination
        node. A best route is defined by [list of node_indices] between two nodes,
        while origin_position[0] and destination_position[1](or destination_position[0]
        if destination_position[1]==None) is included. Combinations that do not fulfill all constraints
        will not be returned.
        Specific to this framework: max_cost_value = None translates to max_cost_value = DEFAULT_MAX_X_SEARCH

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
        :return: list of node-indices of the fastest routes
        :rtype: list
        """
        # call return_travel_costs_Xto1() to determine best origin position (this method checks whether data is
        # available or has to be computed) and then build 1-to-1 route from prev dict)
        list_route_infos = self.return_travel_costs_Xto1(list_origin_positions, destination_position, max_routes=1,
                                                         max_cost_value=max_cost_value)
        if not list_route_infos:
            return []
        origin_position, _, _, _ = list_route_infos[0]
        return self.return_best_route_1to1(origin_position, destination_position)

    def return_travel_costs_1toX(self, origin_position, list_destination_positions, max_routes=None,
                                 max_cost_value=None, customized_section_cost_function = None):
        """This method will return a list of tuples of destination node and travel time of the X fastest routes between
        a list of possible destination nodes and a certain origin node. Combinations that do not fulfill all constraints
        will not be returned.

        :param origin_position: (origin_node_index, destination_node_index, relative_position) of origin edge
        :type origin_position: list
        :param list_destination_positions: list of destination positions
                (origin_node_index, destination_node_index, relative_position) of destination_edge
        :type list_destination_positions: list
        :param max_routes: maximal number of fastest route triples that should be returned
        :type max_routes: int/None
        :param max_cost_value: latest cost function value of a route at destination to be considered as solution
                (max time if customized_section_cost_function == None)
        :param customized_section_cost_function: function to compute the travel cost of an section
                which takes the args: (travel_time, travel_distance, current_dijkstra_node_index) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :type customized_section_cost_function: func
        :return: list of (destination_position, cost_function_value, travel time, travel_distance) tuples
        :rtype: list
        """
        # get list of (destination nodes, add_tt, add_td) -> assign positions to origin_node
        if len(list_destination_positions) == 0:
            return []
        if origin_position[1] is not None:
            o_node = self.nodes[origin_position[1]]
        else:
            o_node = self.nodes[origin_position[0]]
        # set default max_cost_value if not available
        if max_cost_value is None:
            max_time = DEFAULT_MAX_X_SEARCH
        else:
            max_time = max_cost_value
        # check if prev valid for destination node
        set_o_surround_next = o_node.surround_next.get(max_time)
        if set_o_surround_next is None:
            set_o_surround_next = self._compute_o_surround_prev(o_node, max_time, save=False)
        # could be that the current edge is not in surrounding
        set_o_surround_next.add(origin_position[0])
        return_list = []
        for d_pos in list_destination_positions:
            if d_pos[0] in set_o_surround_next or d_pos[1] in set_o_surround_next:
                cfv, scaled_route_tt, td = self.return_travel_costs_1to1(origin_position, d_pos,
                                                                         customized_section_cost_function)
                # still check as surrounding is computed for free flow condition
                if not max_cost_value or scaled_route_tt <= max_cost_value:
                    return_list.append((d_pos, cfv, scaled_route_tt, td))
        # sort if only limited amount of routes should be returned
        if max_routes:
            return_list.sort(key=lambda x: x[1])
            return_list = return_list[:max_routes]
        return return_list

    def return_best_route_1toX(self, origin_position, list_destination_positions, max_cost_value=None,
                               customized_section_cost_function = None):
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
        :return: list of node-indices of the fastest routes
        :rtype: list
        """
        # call return_travel_costs_Xto1() to determine best origin position (this method checks whether data is
        # available or has to be computed) and then build 1-to-1 route from prev dict)
        list_route_infos = self.return_travel_costs_1toX(origin_position, list_destination_positions, max_routes=1,
                                                         max_cost_value=max_cost_value)
        if not list_route_infos:
            return []
        destination_position, _, _, _ = list_route_infos[0]
        return self.return_best_route_1to1(origin_position, destination_position)

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
        result_dict = {}
        for o_pos in list_positions:
            for d_pos in list_positions:
                result_dict[(o_pos, d_pos)] = self.return_travel_costs_1to1(o_pos, d_pos,
                                                                            customized_section_cost_function)
            return result_dict

    # internal methods
    # ----------------
    def _get_section_overhead(self, position, traveled_from_start = True):
        """This method returns unscaled overhead values.

        :param position: (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :param traveled_from_start: computes already traveled travel_time and distance,
                if False: computes rest travel time (relative_position -> 1.0-relative_position)
        :return: (rest travel time, rest travel distance)
        """
        if position[1] is None:
            return (0.0, 0.0)
        o_node_index = position[0]
        d_node_index = position[1]
        all_travel_time = self.tt[o_node_index][d_node_index]
        all_travel_distance = self.td[o_node_index][d_node_index]
        overhead_fraction = position[2]
        if not traveled_from_start:
            overhead_fraction = 1.0 - overhead_fraction
        return all_travel_time * overhead_fraction, all_travel_distance * overhead_fraction

    def _get_od_nodes_and_section_overheads(self, origin_position, destination_position):
        """This method returns o_node, d_node and unscaled overhead values.

        :param origin_position: (current_edge_o_node_index, current_edge_d_node_index, relative_position)
        :param destination_position: (destination_edge_o_node_index, destination_edge_d_node_index, relative_position)
        :return: (origin_node, destination_node, add_tt, add_dist)
        """
        trivial = False
        if origin_position[0] == destination_position[0] and origin_position[1] == destination_position[1]:
            if origin_position[2] is not None and destination_position[2] is not None:
                if origin_position[2] < destination_position[2]:
                    overhead = destination_position[2] - origin_position[2]
                    ov_tt, ov_dist = self._get_section_overhead((origin_position[0], origin_position[1], overhead))
                    return self.nodes[origin_position[0]], self.nodes[origin_position[0]], ov_tt, ov_dist
            elif origin_position[2] is None and destination_position[2] is None:
                return self.nodes[origin_position[0]], self.nodes[origin_position[0]], 0.0, 0.0
            elif origin_position[2] is None:
                ov_tt, ov_dist = self._get_section_overhead(destination_position)
                return self.nodes[origin_position[0]], self.nodes[origin_position[0]], ov_tt, ov_dist
            # last case is not trivial
        if not trivial:
            add_0_tt, add_0_dist = self._get_section_overhead(origin_position, False)
            add_1_tt, add_1_dist = self._get_section_overhead(destination_position)
            if origin_position[1] is not None:
                o_node = self.nodes[origin_position[1]]
            else:
                o_node = self.nodes[origin_position[0]]
            return o_node, self.nodes[destination_position[0]], add_0_tt + add_1_tt, add_0_dist + add_1_dist

    def _lookup_dijkstra_1to1(self, origin_node, destination_node):
        """This internal method computes the lookup Dijkstra between two nodes.

        :param origin_node: node object of origin
        :type origin_node: Node
        :param destination_node: node object of destination
        :type destination_node: Node
        :return: node_index_list, scaled_route_tt, distance
        :rtype: list
        """
        current_node = origin_node
        node_list = [current_node.node_index]
        route_tt = 0.0
        scaled_route_tt = 0.0
        total_tt = self.tt[origin_node.node_index][destination_node.node_index]
        if total_tt in [None, np.nan, np.inf]:
            prt_str = f"There is no route from {origin_node} to {destination_node}"
            raise AssertionError(prt_str)
        while current_node != destination_node:
            found_next_node = False
            for next_node_id, next_node_obj in current_node.to_nodes.items():
                # since complete travel time matrix is known, nodes on the route have to satisfy
                # A->B + B->C = A->C
                if next_node_obj.is_stop_only and next_node_obj != destination_node:
                    continue
                next_tt = self.tt[current_node.node_index][next_node_id]
                from_next_tt = self.tt[next_node_id][destination_node.node_index]
                if route_tt + next_tt + from_next_tt - total_tt < EPS:
                    found_next_node = True
                    node_list.append(next_node_id)
                    route_tt += next_tt
                    scaled_route_tt += (next_tt * self.tt_factor)
                    current_node = next_node_obj
                    break
            if not found_next_node:
                prt_str = f"Could not find next node after current node {current_node} in search" \
                          f" of route from {origin_node} to {destination_node}; current node list: {node_list}"
                raise AssertionError(prt_str)
        return node_list, scaled_route_tt

    def _compute_d_surround_prev(self, destination_node, max_time_value):
        """This method computes the surrounding node-indices within a certain max_time value range of a node, saves it
        for the respective node and returns it.

        :param destination_node: node
        :param max_time_value: search range
        :return: set of origin_node_indices that are close enough to reach destination node within max_time_value
        """
        d_node_index = destination_node.node_index
        d_surround = set(np.argwhere(self.tt_numpy[:, d_node_index] <= max_time_value).flatten())
        destination_node.surround_prev[max_time_value] = d_surround
        return d_surround

    def _compute_o_surround_prev(self, origin_node, max_time_value, save=False):
        """This method computes the surrounding node-indices within a certain max_time value range of a node, saves it
        for the respective node and returns it.

        :param origin_node: node
        :param max_time_value: search range
        :param save: save surrounding to origin node
        :return: set of destination_node_indices that are close enough to be reached from origin within max_time_value
        """
        o_node_index = origin_node.node_index
        o_surround = set(np.argwhere(self.tt_numpy[o_node_index, :] <= max_time_value).flatten())
        if save:
            origin_node.surround_next[max_time_value] = o_surround
        return o_surround

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

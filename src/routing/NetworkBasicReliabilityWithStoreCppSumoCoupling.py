"""
Authors: Roman Engelhardt, Florian Dandl, Joel Brodersen
TUM, 2025
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
from datetime import datetime
import pathlib
import random
import math

# src imports
# -----------
from src.routing.NetworkBasicWithStoreCppSumoCoupling import NetworkBasicWithStoreCppSumoCoupling
from src.routing.cpp_router.PyNetwork import PyNetwork
from src.routing.NetworkBasic import Edge as BasicEdge
from src.routing.NetworkBasic import Node as BasicNode
# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
LOG = logging.getLogger(__name__)
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
        self.travel_infos_from = {} #node_index -> (tt, dis,var)
        self.travel_infos_to = {}   #node_index -> (tt, dis,var)
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
        self.travel_infos_to[other_node.node_index] = (edge.get_tt(),edge.get_distance(), edge.get_tt_var())

    def add_prev_edge_from(self, other_node, edge):
        self.edges_from[other_node] = edge
        self.travel_infos_from[other_node.node_index] = (edge.get_tt(),edge.get_distance(), edge.get_tt_var())

    def get_travel_infos_to(self, other_node_index):
        return self.travel_infos_to[other_node_index]

    def get_travel_infos_from(self, other_node_index):
        return self.travel_infos_from[other_node_index]
    

class Edge(BasicEdge):
    def __init__(self, edge_index, distance, travel_time,travel_time_var):
        self.edge_index = edge_index
        self.distance = distance
        self.travel_time = travel_time
        self.travel_time_var = travel_time_var
        self.cost_function_value = None
    
    def set_tt_var(self, travel_time_var):
        self.travel_time_var = travel_time_var
    
    def get_tt_var(self):
        """
        :return: (current) travel time standard deviation on edge
        """
        return self.travel_time_var
    
    def set_cfv(self,cfv):
        self.cost_function_value = cfv

    def get_tt_mean_var(self):
        """
        :return: (travel time mean, travel time standard deviation) tuple
        """
        return (self.travel_time, self.travel_time_var)
    
    def get_init_cfv(self,cost_function):
        self.cost_function_value = cost_function(tt_mean=self.travel_time,tt_var=self.travel_time_var,dis=self.distance)

class NetworkBasicReliabilityWithStoreCppSumoCoupling(NetworkBasicWithStoreCppSumoCoupling):
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

        #super().__init__(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)
        #self.cpp_router = None
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
        self._tt_infos_from_folder = True
        self._current_tt_factor = None
        self.travel_time_file_infos = self._load_tt_folder_path(network_dynamics_file_name=network_dynamics_file_name)
        #self.loadNetwork(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)
        self.current_dijkstra_number = 1    #used in dijkstra-class
        self.sim_time = 0   # TODO #
        self.zones = None   # TODO #
        with open(os.sep.join([self.network_name_dir, "base","crs.info"]), "r") as f:
            self.crs = f.read()   
        self.travel_time_infos = {} #(o,d) -> (tt, dis)       
        self.travel_time_infos_reliability = {} #(o,d) -> (tt, var)
        self.user_vot = None
        self.user_vor = None
        self.routing_mode = "edge_tt"


        for attr, value in vars(self).items():
            print(f"{attr}: {value}")
        breakpoint()
    #def loadNetwork(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
       # LOG.info("load c++ router!")
       # print("load c++ router!")
        #nodes_f = os.path.join(network_name_dir, "base", "nodes.csv")
        #edges_f = os.path.join(network_name_dir, "base", "edges.csv")
       # self.cpp_router = PyNetwork(nodes_f.encode(), edges_f.encode())
       # super().loadNetwork(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)
        #for attr, value in vars(self).items():
            #print(f"{attr}: {value}")

    def set_routing_mode(self,routing_mode):
        self.routing_mode = routing_mode
    
    def set_user_vor(self,user_vor):
        self.user_vor = user_vor

    def set_user_vot(self,user_vot):
        self.user_vot = user_vot
    
    def _save_cpp_tt_file(self,tt_file_df,scenario_time):
        """
        cleans folder for csv handover to cpp form prievious simulation runs and saves tt_file_df as a new csv file into it.
        """
        py_path  = pathlib.Path(__file__)
        save_path_dir = py_path.parent/"cpp_router"/"tt_updates"
        if os.path.exists(save_path_dir):
            for filename in os.listdir(save_path_dir):
                file_path = os.path.join(save_path_dir, filename)
                try:
                    # If it's a file, delete it
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        random_number = random.randint(0, 9999)
        date = datetime.now().strftime("%Y%m%d%H%M%S%f")
        save_file_path_cpp = save_path_dir/f"{scenario_time}_tt_update_{date}_{random_number}.csv"
        tt_file_df.to_csv(save_file_path_cpp)
        return str(save_file_path_cpp)
    
    def load_tt_file(self, scenario_time): ##TODO adjust for 
        """
        loads new travel time files for scenario_time
        """
        self._reset_internal_attributes_after_travel_time_update()
        if self._tt_infos_from_folder:
            f = self.travel_time_file_infos[scenario_time]
            tt_file = os.path.join(f, "edges_td_att.csv")
            tt_file_df = pd.read_csv(tt_file)  
            tt_file_df['edge_cfv'] = tt_file_df.apply(lambda row: self.customized_section_cost_function(tt_mean=row['edge_tt'], tt_var=row['edge_var']), axis=1)
            columns_to_keep = ['from_node', 'to_node', 'edge_tt', 'distance', 'edge_std', 'edge_cfv','edge_var']
            tt_file_df = tt_file_df.drop(columns=[col for col in tt_file_df.columns if col not in columns_to_keep])
            save_file_path_cpp = self._save_cpp_tt_file(tt_file_df=tt_file_df,scenario_time=scenario_time)           
            self.cpp_router.updateEdgeTravelTimes(save_file_path_cpp.encode())
            for from_node, to_node, edge_tt, edge_var,edge_cfv in zip(tt_file_df[G_EDGE_FROM], tt_file_df[G_EDGE_TO], tt_file_df['edge_tt'], tt_file_df["edge_var"],tt_file_df['edge_cfv']):
                self._set_edge_tt(o_node_index=from_node, d_node_index=to_node, new_travel_time=edge_tt,new_travel_time_var=edge_var,new_cost_function_value=edge_cfv)

    def loadNetwork(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        nodes_f = os.path.join(network_name_dir, "base", "nodes.csv")
        print(f"Loading nodes from {nodes_f} ...")
        nodes_df = pd.read_csv(nodes_f)
        self.nodes = nodes_df.apply(read_node_line, axis=1)
        edges_f = os.path.join(network_name_dir, "base", "edges.csv")
        print(f"Loading edges from {edges_f} ...")
        edges_df = pd.read_csv(edges_f)
        self.cpp_router = PyNetwork(nodes_f.encode(), edges_f.encode())
        for _, row in edges_df.iterrows():
            o_node = self.nodes[row[G_EDGE_FROM]]
            d_node = self.nodes[row[G_EDGE_TO]]
            if G_EDGE_TT_STD in row.keys():
                travel_time_std = row[G_EDGE_TT_STD]
            else:
                travel_time_std = 0
            tmp_edge = Edge(edge_index=(o_node, d_node), distance=row[G_EDGE_DIST], travel_time= row[G_EDGE_TT],travel_time_var=travel_time_std^2)
            tmp_edge.get_init_cfv(self.customized_section_cost_function)
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
    


    def _set_edge_tt(self, o_node_index, d_node_index, new_travel_time,new_travel_time_var,new_cost_function_value=None):
        o_node = self.nodes[o_node_index]
        d_node = self.nodes[d_node_index]
        edge_obj = o_node.edges_to[d_node]
        edge_obj.set_tt(new_travel_time)
        edge_obj.set_tt_var(new_travel_time_var)
        edge_obj.set_cfv(new_cost_function_value)
        new_tt, dis = edge_obj.get_tt_distance()
        o_node.travel_infos_to[d_node_index] = (new_tt, dis,new_travel_time_var)
        d_node.travel_infos_from[o_node_index] = (new_tt, dis,new_travel_time_var)


    def external_update_edge_travel_times(self, new_travel_time_dict):
        '''This method takes new edge travel time information from sumo and updates the network in the fleet simulation
        :param new_travel_time_dict: dict edge_id (o_node, d_node) -> edge_traveltime [s]
        :return None'''
        self._reset_internal_attributes_after_travel_time_update()
        for edge_index_tuple, (new_tt,new_var) in new_travel_time_dict.items():
            o_node = self.nodes[edge_index_tuple[0]]
            d_node = self.nodes[edge_index_tuple[1]]
            edge_obj = o_node.edges_to[d_node]
            self._set_edge_tt(edge_index_tuple[0], edge_index_tuple[1], new_tt, new_var) 
        print(f"Updated {len(new_travel_time_dict)} edges in FP Routing Engine with simulated values.")

    def load_tt_file_SUMO(self, resultsPath,sim_time):
        """
        loads new travel time files for scenario_time
        """
        if self._tt_infos_from_folder:
            tt_file = os.path.join(resultsPath, "EdgeTravelTimes", f"SUMO_travel_times_{sim_time}.csv")
            self.cpp_router.updateEdgeTravelTimes(tt_file.encode())
            print(f"{tt_file} loaded into cpp Router at step {sim_time}")


    
    def return_travel_costs_1to1(self, origin_position, destination_position, customized_section_cost_function = None,mode="edge_tt"):
        """
        This method will return the travel costs of the fastest route between two nodes considering reliability.
        :param origin_position: (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :param destination_position: (destination_edge_origin_node_index, destination_edge_destination_node_index, relative_position)
        :param customized_section_cost_function: function to compute the travel cost of an section: args: (travel_time, travel_distance, current_dijkstra_node) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :return: (cost_function_value, travel time, travel_time_var) between the two nodes
        """
        trivial_test = self.test_and_get_trivial_route_tt_and_dis(origin_position, destination_position)
        if trivial_test is not None:
            return trivial_test[1]
        origin_node = origin_position[0]
        origin_overhead = (0.0, 0.0, 0.0, 0.0)
        if origin_position[1] is not None:
            origin_node = origin_position[1]
            origin_overhead = self.get_section_overhead(origin_position, from_start=False)
        destination_node = destination_position[0]
        destination_overhead = (0.0, 0.0, 0.0, 0.0)
        if destination_position[1] is not None:
            destination_overhead = self.get_section_overhead(destination_position, from_start=True)
 
        if self.travel_time_infos.get( (origin_node, destination_node) ) is not None:
            s = self.travel_time_infos.get((origin_node, destination_node))
        else:
            s = self.cpp_router.computeTravelCosts1To1(origin_node, destination_node,mode)
            if s[0] == np.inf: # TODO # seems to be a bug. Do you know what's the problem here?
                LOG.warning(f"in return_travel_costs_1to1, travel_time_infos from nodes {origin_node} to {destination_node}"
                         f"yields s={s}")
                s = self.cpp_router.computeTravelCosts1To1(origin_node, destination_node)

                LOG.warning(f"after recalculation of the routes, s={s}")

  
            self._add_to_database(origin_node, destination_node, s[0], s[1], s[2],s[3])

        return (s[0] + origin_overhead[0] + destination_overhead[0], s[1] + origin_overhead[1] + destination_overhead[1], s[2] + origin_overhead[2] + destination_overhead[2], s[3] + origin_overhead[3] + destination_overhead[3])

    
        

    
    def get_section_overhead(self, position, from_start=True, customized_section_cost_function=None):
        """This method computes the section overhead for a certain position.

        :param position: (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :param from_start: computes already traveled travel_time and distance,
                           if False: computes rest travel time (relative_position -> 1.0-relative_position)
        :param customized_section_cost_function: customized routing objective function
        :return: (cost_function_value, travel time, travel_distance)
        """
        if position[1] is None:
            return 0.0, 0.0, 0.0, 0.0
        all_travel_time, all_travel_distance,all_travel_time_var = self.get_section_infos(position[0], position[1])
        overhead_fraction = position[2]
        if not from_start:
            overhead_fraction = 1.0 - overhead_fraction
        all_travel_cost = self.customized_section_cost_function(tt_mean=all_travel_time,tt_var=all_travel_time_var, dis=all_travel_distance)
        return all_travel_time * overhead_fraction, all_travel_distance * overhead_fraction,all_travel_time_var*overhead_fraction,all_travel_cost*overhead_fraction
    
    def get_section_infos(self, start_node_index, end_node_index):
        """
        :param start_node_index_index: index of start_node of section
        :param end_node_index: index of end_node of section
        :return: (travel time, distance, var); if no section between nodes (None, None)
        """
        tt, dis, tt_var = self.nodes[start_node_index].get_travel_infos_to(end_node_index)

        return tt, dis, tt_var
  

    def customized_section_cost_function(self,tt_mean=None,tt_var=None,dis=None):
            vor=self.user_vor
            vot=self.user_vot
            cfv = vot * tt_mean + vor * math.sqrt(tt_var)
            return cfv
    
    def _add_to_database(self, o_node, d_node, tt, dis,var,cfv):
        """ this function is call when new routing results have been computed
        depending on the class the function can be overwritten to store certain results in the database
        """
        if self.travel_time_infos.get((o_node, d_node)) is None:
            self.travel_time_infos[(o_node, d_node)] = (tt,dis,var,cfv)

    def return_travel_costs_Xto1(self, list_origin_positions, destination_position, max_routes=None, max_cost_value=None, customized_section_cost_function = None,mode="edge_tt"):
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
                return_list.append( (pos, trivial_test[1][0], trivial_test[1][1], trivial_test[1][2], trivial_test[1][3]))
                continue
            start_node = pos[0]
            if pos[1] is not None:
                start_node = pos[1]
            try:
                origin_nodes[start_node].append(pos)
            except:
                origin_nodes[start_node] = [pos]

        destination_node = destination_position[0]
        destination_overhead = (0.0, 0.0, 0.0, 0.0)
        if destination_position[1] is not None:
            destination_overhead = self.get_section_overhead(destination_position, from_start=True)
        
        if len(origin_nodes.keys()) > 0:
            s = self.cpp_router.computeTravelCostsXto1(destination_node, origin_nodes.keys(), max_time_range = max_cost_value, max_targets = max_routes,mode=mode)
            # s has structure [(origin_node,tt,dis,var,cfv)]
            for result in s:
                org_node,tt, dis,var,cfv = result
                if tt < -0.0001:
                    continue
                self._add_to_database(org_node, destination_node, tt, dis, var,cfv)
                tt += destination_overhead[1]
                dis += destination_overhead[2]
                for origin_position in origin_nodes[org_node]:
                    origin_overhead = (0.0, 0.0, 0.0, 0.0)
                    if origin_position[1] is not None:
                        origin_overhead = self.get_section_overhead(origin_position, from_start=False)
                        
                    tt += origin_overhead[0]
                    dis += origin_overhead[1]
                    var += origin_overhead[2]
                    cfv += origin_overhead[3]
                    if max_cost_value is not None and tt > max_cost_value:
                        #pass
                        continue
                    return_list.append( (origin_position, tt, dis,var,cfv) )
        if max_routes is not None and len(return_list) > max_routes:
            return sorted(return_list, key = lambda x:x[1])[:max_routes]
        return return_list
    
    def return_best_route_1to1(self, origin_position, destination_position, customized_section_cost_function = None, mode="edge_tt"):
        """
        This method will return the best route [list of node_indices] between two nodes,
        while origin_position[0] and destination_postion[1](or destination_position[0] if destination_postion[1]==None) is included.
        :param origin_position: (current_edge_origin_node_index, current_edge_destination_node_index, relative_position)
        :param destination_position: (destination_edge_origin_node_index, destination_edge_destination_node_index, relative_position)
        :param customized_section_cost_function: function to compute the travel cost of an section: args: (travel_time, travel_distance, current_dijkstra_node) -> cost_value
                if None: travel_time is considered as the cost_function of a section
        :return : route (list of node_indices) of best route
        """
       # if customized_section_cost_function is not None:
            #return super().return_best_route_1to1(origin_position, destination_position, customized_section_cost_function=customized_section_cost_function)
        trivial_test = self.test_and_get_trivial_route_tt_and_dis(origin_position, destination_position)
        if trivial_test is not None:
            return trivial_test[0]
        origin_node = origin_position[0]
        destination_node = destination_position[0]
        if origin_position[1] is not None:
            origin_node = origin_position[1]
        node_list = self.cpp_router.computeRoute1To1(origin_node, destination_node, mode=mode)
        if origin_node != origin_position[0]:
            node_list = [origin_position[0]] + node_list
        if destination_position[1] is not None:
            node_list.append(destination_position[1])
        #print(f"best route: {origin_position} --> {destination_position}; Mode =  {mode}: {node_list}")
        return node_list
    
    def test_and_get_trivial_route_tt_and_dis(self, origin_position, destination_position):
        """ this functions test for trivial routing solutions between origin_position and destination_position
        if no trivial solution is found
        :return None
        else
        :return (route, (travel_time, travel_distance))
        """
        #LOG.debug(f"Trivial Test: {origin_position} --> {destination_position}")
        if origin_position[0] == destination_position[0]:
            if origin_position[1] is None:
                if destination_position[1] is None:
                    return ([], (0.0, 0.0, 0.0, 0.0) )
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
                            tt, dis, var,cfv = self.get_section_overhead(effective_position, from_start = True)
                            return ([destination_position[0], destination_position[1]], (tt, dis, var,cfv)) 
                    else:
                        return None
        elif origin_position[1] is not None and origin_position[1] == destination_position[0]:
            rest = self.get_section_overhead(origin_position, from_start = False)
            rest_dest = self.get_section_overhead(destination_position, from_start = True)
            route = [origin_position[0], origin_position[1]]
            #print(f"nw basic argh {rest} {rest_dest}")
            if destination_position[1] is not None:
                route.append( destination_position[1] )
            return (route, (rest[0] + rest_dest[0], rest[1] + rest_dest[1], rest[2] + rest_dest[2],rest[3] + rest_dest[3]))
        return None

    def load_tt_file_SUMO(self, resultsPath,sim_time):
        """
        loads new travel time files for scenario_time
        """
        if self._tt_infos_from_folder:
            tt_file = os.path.join(resultsPath, "EdgeTravelTimes", f"SUMO_travel_times_{sim_time}.csv")
            self.cpp_router.updateEdgeTravelTimes(tt_file.encode())
    
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
        travel_time_var = 0
        cost_function_value = 0
        start_tt, start_dis, start_var,start_cfv = self.get_section_overhead( (route[0], route[1], rel_start_edge_position), from_start=False)
        arrival_time += start_tt
        distance += start_dis
        travel_time_var += start_var
        cost_function_value += start_cfv
        if len(route) > 2:
            for i in range(2, len(route)):
                tt, dis,var = self.get_section_infos(route[i-1], route[i])
                arrival_time += tt
                distance += dis
                travel_time_var += var
                cost_function_value += self.customized_section_cost_function(tt_mean=tt,tt_var=var,dis=dis)
        return (arrival_time, distance,travel_time_var,cost_function_value)
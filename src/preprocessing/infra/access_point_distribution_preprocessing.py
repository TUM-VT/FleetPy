import sys
import os
import numpy as np
import pandas as pd
import pickle
import rtree

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(MAIN_DIR)
sys.path.append(MAIN_DIR)

from src.routing.NetworkBasic import NetworkBasic as Network

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

def match_stop_nodes_to_stop_nodes_in_walking_range(nw, max_walking_range, stop_node_definition = "all"):
    """ computes the walking distance from all boarding/stop-nodes to all boarding/stop-nodes in walking range
    :param nw: network obj
    :param max_walking_range: maximum walking distance [m]
    :param stop_node_definition: defines which nodes are considered boarding/stop-nodes: "all": all nodes are considered; "must_stop": only nodes with attribute n.is_stop_only==True considered; list of node_indices -> only those considered
    :return: dict origin_stop_node_index -> stop_node_index in walking distance -> walking distance
    """
    stop_node_to_stop_node_in_walking_range = {}
    if type(stop_node_definition) == list:
        stop_node_pos = [nw.return_node_position(x) for x in stop_node_definition]
    elif stop_node_definition == "all":
        stop_node_pos = [nw.return_node_position(x.index) for x in nw.get_node_list()]
    elif stop_node_definition == "must_stop":
        stop_node_pos = [nw.return_node_position(x.index) for x in nw.get_node_list() if x.must_stop()]
    else:
        raise EnvironmentError("Unknown stop_node_definition specification!")
    print("start computing walking distances between {} possible stop nodes in {}m walking range ...".format(len(stop_node_pos), max_walking_range))
    for i, stop_node in enumerate(stop_node_pos):
        if i%100 == 0:
            print("   ...{}/{}".format(i, len(stop_node_pos)))
        stop_node_to_stop_node_in_walking_range[stop_node] = {}
        res_hin = nw.return_travel_costs_1toX(stop_node, stop_node_pos,
                                max_cost_value=max_walking_range, customized_section_cost_function = routing_min_distance_cost_function)
        res_back = nw.return_travel_costs_Xto1(stop_node_pos, stop_node,
                                max_cost_value=max_walking_range, customized_section_cost_function = routing_min_distance_cost_function)
        for boarding_node_pos, dis, _, _ in res_hin + res_back:
            if stop_node_to_stop_node_in_walking_range[stop_node].get(boarding_node_pos, 99999999.9) > dis:
                stop_node_to_stop_node_in_walking_range[stop_node][boarding_node_pos] = dis
    return stop_node_to_stop_node_in_walking_range

def load_active_bps(nw, nw_name, infra_name):
    """ loads boarding point from a matched infra specification to a matched network
    :param nw: network obj
    :param nw_name: str name of the network
    :param infra_name: str name of the infra directory
    :return: dict node_position -> 1 for all nodes stored in the boarding_point.csv file
    """
    active_bps = {}
    bp_df = pd.read_csv(os.path.join(MAIN_DIR, "data", "infra", infra_name, nw_name, "boarding_points.csv"))
    for bp_node in bp_df["node_index"].values:
        active_bps[nw.return_node_position(bp_node)] = 1
    return active_bps

def create_bp_subset_x_per_max_walkrange_random(seed, active_bps, stop_node_to_stop_node_in_walking_range, number_boarding_points_in_walking_range, print_steps = True):
    """ reduces randomly boarding points by randomly selecting an active boarding location and removing it, if the number of boarding location of all boarding locations in walking range is still larger than number_of_boarding_points_in walking range
    :param seed: random seed (int)
    :param active_bps: dict boarding_point_position -> 1 (output from load_active_bps())
    :param stop_node_to_stop_node_in_walking_range: output from match_stop_nodes_to_stop_nodes_in_walking_range()
    :param number_boarding_points_in_walking_range: (int > 0) maximum number of boarding locations for each boarding location in walking range
    :return: stop_node_position -> stop_node_position in walking range -> distance to walk for all stop_nodes after removal
    """

    def get_removable_bps(active_bps_to_stop, stop_to_active_bps):
        removable_bps = []
        for bp, stop_dict in active_bps_to_stop.items():
            not_enough_bps_flag = False 
            for stop in stop_dict.keys():
                if len(stop_to_active_bps[stop].keys()) <= number_boarding_points_in_walking_range:
                    not_enough_bps_flag = True
                    break
            if not not_enough_bps_flag:
                removable_bps.append(bp)
        return removable_bps

    np.random.seed(21061992+seed)
    stop_to_active_bps = {}
    active_bps_to_stop = {}
    for stop, target_dict in stop_node_to_stop_node_in_walking_range.items():
        stop_to_active_bps[stop] = {}
        for target, dis in target_dict.items():
            if active_bps.get(target):
                stop_to_active_bps[stop][target] = dis
                try:
                    active_bps_to_stop[target][stop] = dis
                except:
                    active_bps_to_stop[target] = {stop : dis}

    print("start removing boarding points!")
    removable_bps = get_removable_bps(active_bps_to_stop, stop_to_active_bps)
    c = 0
    while len(removable_bps) > 0:
        if c % 100 == 0 and print_steps:
            print("   ... still removable: {}".format(len(removable_bps)))
        c += 1
        d = np.random.randint(0, len(removable_bps))
        remove_bp = removable_bps[d]
        for stop in list(active_bps_to_stop[remove_bp].keys()):
            del stop_to_active_bps[stop][remove_bp]
        del active_bps_to_stop[remove_bp]
        removable_bps = get_removable_bps(active_bps_to_stop, stop_to_active_bps)

    target_to_active_bps = {}
    for bp, stop_dict in active_bps_to_stop.items():
        for stop, walking_dis in stop_dict.items():
            try:
                target_to_active_bps[stop][bp] = walking_dis
            except:
                target_to_active_bps[stop] = {bp : walking_dis}

    min_number_stops = float("inf")
    max_number_stops = 0
    avg_number_stops = 0
    avg_dis_to_closest_bp = 0
    for stop, bp_dict in target_to_active_bps.items():
        n_stops = len(bp_dict.keys())
        if n_stops < min_number_stops:
            min_number_stops = n_stops
        if n_stops > max_number_stops:
            max_number_stops = n_stops
        avg_number_stops += n_stops
        closest_dis = float("inf")
        for walking_dis in bp_dict.values():
            if walking_dis < closest_dis:
                closest_dis = walking_dis
        avg_dis_to_closest_bp += closest_dis
    print("evaluation of new boarding points:")
    print("minimum number of boarding points in walking range: {}".format(min_number_stops))
    print("maximum number of boarding points in walking range: {}".format(max_number_stops))
    print("average number of boarding points in walking range: {}".format(avg_number_stops/len(target_to_active_bps.keys())))
    print("average distance to closest boarding point: {}m".format(avg_dis_to_closest_bp/len(target_to_active_bps.keys())))

    return active_bps_to_stop


def store_bp_infra(new_infra_name, nw_name, active_bp_list):
    base_path = os.path.join(MAIN_DIR, "data", "infra")
    p = os.path.join(base_path, new_infra_name)
    if not os.path.isdir(p):
        os.mkdir(p)
    p = os.path.join(p, nw_name)
    if not os.path.isdir(p):
        os.mkdir(p)
    p = os.path.join(p, "boarding_points.csv")
    dict_list = [{"node_index" : x[0]} for x in active_bp_list]
    bp_df = pd.DataFrame(dict_list)
    bp_df.to_csv(p, index=False)

def create_maximum_init_bps(nw, nw_name, max_adjacent_vel = None, zone_system_name = None, stop_only_nodes = False):
    """ this function creates a maximum set of boarding points of all nodes not getting filtered by the additional parameters
    :param max_adjecent_vel: if given only consideres nodes which are connected to edges with freeflow velocity below this value (m/s)
    :param zone_system_name: if given only consideres nodes belong to the given zone_system (assumes matching has been done before)
    :param stop_only_nodes: if true only nodes with stop_only attribute are considere
    :return: dictionary node_position -> 1 for nodes that are considered boarding points
    """
    bp_nodes = {}
    if zone_system_name is None:
        bp_nodes = {nw.return_node_position(n) : 1 for n in range(len(nw.nodes))}
    else:
        node_zone_df = pd.read_csv(os.path.join(MAIN_DIR, "data", "zones", zone_system_name, nw_name, "node_zone_info.csv"), index_col=0)
        bp_nodes = {nw.return_node_position(n) : 1 for n, z in node_zone_df["zone_id"].items()}
    if stop_only_nodes:
        for node_pos in list(bp_nodes.keys()):
            if not nw.nodes[node_pos[0]].must_stop():
                del bp_nodes[node_pos]
    if max_adjacent_vel is not None:
        if not stop_only_nodes:
            for node_pos in list(bp_nodes.keys()):
                node_index = node_pos[0]
                node = nw.nodes[node_index]
                infeasible = False
                for next_node, edge in node.get_next_node_edge_pairs():
                    if edge.travel_time == 0:
                        continue
                    v = edge.distance/edge.travel_time
                    if v > max_adjacent_vel:
                        infeasible = True
                        break
                if not infeasible:
                    for next_node, edge in node.get_prev_node_edge_pairs():
                        if edge.travel_time == 0:
                            continue
                        v = edge.distance/edge.travel_time
                        if v > max_adjacent_vel:
                            infeasible = True
                            break 
                if infeasible:
                    del bp_nodes[node_pos]    
        else:
            for node_pos in list(bp_nodes.keys()):
                node_index = node_pos[0]
                node = nw.nodes[node_index]
                infeasible = False
                for next_node, edge1 in node.get_next_node_edge_pairs():
                    if infeasible:
                        break
                    for next_node, edge in next_node.get_next_node_edge_pairs():
                        if edge.travel_time == 0:
                            continue
                        v = edge.distance/edge.travel_time
                        if v > max_adjacent_vel:
                            infeasible = True
                            break
                if not infeasible:
                    for next_node, edge1 in node.get_prev_node_edge_pairs():
                        if infeasible:
                            break
                        for next_node, edge in node.get_prev_node_edge_pairs():
                            if edge.travel_time == 0:
                                continue
                            v = edge.distance/edge.travel_time
                            if v > max_adjacent_vel:
                                infeasible = True
                                break 
                if infeasible:
                    del bp_nodes[node_pos]           
    print(" ... init {} boarding points".format(len(bp_nodes)))

    return bp_nodes

def geometric_bp_filter(nw, bp_nodes, number_bp_nodes_end):
    """ this function geometrically and randomly removes boarding nodes until only number_bp_nodes_end are left
    method: pick random node and remove closest other node
    """
    def init_rtree(current_bps):
        r_tree = rtree.index.Index()
        for bp_pos in bp_nodes.keys():
            bp_index = bp_pos[0]
            coords = nw.return_node_coordinates(bp_index)
            r_tree.insert(bp_index, (coords[0], coords[1], coords[0], coords[1]) )
        return r_tree
    r_tree = init_rtree(bp_nodes)
    i = 0
    deleted = {}
    n_deleted = 0
    bp_nodes_list = list(bp_nodes.keys())
    org_bps = len(bp_nodes_list)
    while True:
        r = np.random.randint(len(bp_nodes_list))
        bp_index = bp_nodes_list[r][0]
        if deleted.get(bp_index):
            continue
        coords = nw.return_node_coordinates(bp_index)
        nearest_nodes =  list(r_tree.nearest((coords[0], coords[1], coords[0], coords[1]), num_results=2))
        for n in nearest_nodes:
            if n != bp_index and not deleted.get(n):
                deleted[n] = 1
                n_deleted += 1
        i += 1
        if i % 1000 == 0:
            print(".... -> deleted {}".format(n_deleted))
            for index in deleted.keys():
                del bp_nodes[nw.return_node_position(index)]
            deleted = {}
            r_tree = init_rtree(bp_nodes)
            bp_nodes_list = list(bp_nodes.keys())
        if org_bps - n_deleted <= number_bp_nodes_end:
            break
    for index in deleted.keys():
        del bp_nodes[nw.return_node_position(index)]
    print(" ... {} bp nodes left".format(len(bp_nodes)))
    return bp_nodes
    

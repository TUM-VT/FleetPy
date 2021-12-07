import os
import sys
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool

tum_fleet_sim_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(tum_fleet_sim_path)
from src.routing.NetworkBasic import NetworkBasic as Network 
from src.infra.BoardingPointInfrastructure import BoardingPointInfrastructure

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

def find_access_node_from_boarding_point(nw, boarding_postion, access_point_positions, walking_range):
    possible_access_points = {}
    res_hin = nw.return_travel_costs_1toX(boarding_postion, access_point_positions,
                            max_cost_value=walking_range, customized_section_cost_function = routing_min_distance_cost_function)
    res_back = nw.return_travel_costs_Xto1(access_point_positions, boarding_postion,
                            max_cost_value=walking_range, customized_section_cost_function = routing_min_distance_cost_function)
    for access_node_pos, dis, _, _ in res_hin + res_back:
        if possible_access_points.get(access_node_pos, 99999999.9) > dis:
            possible_access_points[access_node_pos] = dis
    return_list = []
    for access_node_pos, dis in sorted(possible_access_points.items(), key = lambda x : x[1]):
        return_list.append( (access_node_pos, dis) )

    if len(return_list) == 0:
        res_hin = nw.return_travel_costs_1toX(boarding_postion, access_point_positions, max_routes=1,
                                customized_section_cost_function = routing_min_distance_cost_function)
        res_back = nw.return_travel_costs_Xto1(access_point_positions, boarding_postion, max_routes=1,
                                customized_section_cost_function = routing_min_distance_cost_function)
        shortest = min(res_hin + res_back, key = lambda x:x[1])
        return_list.append( (shortest[0], shortest[1]) )
    return return_list

def preprocess_boarding_points_in_walking_range(nw_name, infra_name, walking_range, must_stop_only = False, number_cores = 1):
    """ this function the boarding nodes and walking distances from all stop nodes within a given walking range
    by computing distance minimal forward and backwards dijkstras
    and stores it in the corresponding folder to be readable by BoardingPointInfraStructure.py
    :param nw_name: name of the network folder where the bp infrastructure is matched to
    :param infra_name: name of the corresponding boardingpoint infrastructure
    :param walking_range: max walking distance in [m]; note that the preprocessed file is not useable if the walking range is exceeded in the simulation
    """
    nw_path = os.path.join(tum_fleet_sim_path, 'data', 'networks', nw_name)
    bp_path = os.path.join(tum_fleet_sim_path, 'data', 'infra', infra_name, nw_name)

    if not os.path.isdir(nw_path):
        print("network {} not found!".format(nw_name))
        exit()
    if not os.path.isdir(bp_path):
        print("boarding points {} for nw {} not found!".format(infra_name, nw_name))
        exit()

    routing_engine = Network(nw_path)
    bp_infra = BoardingPointInfrastructure(bp_path, routing_engine, load_preprocessed = False)

    line0 = "max_walking_range:{}".format(walking_range)
    lines = [line0]
    if not must_stop_only:
        all_nodes = [routing_engine.return_node_position(i.node_index) for i in routing_engine.get_node_list()]
    else:
        all_nodes = [routing_engine.return_node_position(i.node_index) for i in routing_engine.get_node_list() if i.is_stop_only]
    ap_to_bp_dis = {}
    bp_pos_list = list(bp_infra.boarding_point_node_positions.keys())

    if number_cores == 1:
        for i, bp_node in enumerate(bp_pos_list):
            if i%100 == 0 and i>0:
                print(" ... {}/{} done".format(i, len(bp_pos_list)))
            aps = find_access_node_from_boarding_point(routing_engine, bp_node, all_nodes, walking_range)
            for ap, dis in aps:
                try:
                    ap_to_bp_dis[ap].append( (bp_node, dis) )
                except:
                    ap_to_bp_dis[ap] = [(bp_node, dis)]
    else:
        all_len = len(bp_pos_list)
        per_cor = int(np.math.ceil(all_len/number_cores))
        print("all nodes {} | per core {} | n cores {}".format(all_len, per_cor, number_cores))
        part_lists = []
        for i in range(number_cores):
            part_lists.append(bp_pos_list[(i*per_cor) : min([(i+1)*per_cor, len(bp_pos_list)])])
        p = Pool(number_cores)
        x = [p.apply_async(compute_part_boarding_points_in_walking_range, args=(nw_name, infra_name, walking_range, part_node_pos, all_nodes)) for part_node_pos in part_lists]
        ap_to_bp_dis = {}
        for z in x:
            ap_to_bp_dis_part = z.get()
            for ap, bp_node_dis_list in ap_to_bp_dis_part.items():
                try:
                    ap_to_bp_dis[ap] += bp_node_dis_list
                except:
                    ap_to_bp_dis[ap] = bp_node_dis_list
        p.close()
    # for i, stop_node in enumerate(all_nodes):
    #     if i%100 == 0 and i>0:
    #         print(" ... {}/{} done".format(i, len(all_nodes)))
    #     #print(stop_node)
    #     boarding_nodes = bp_infra.return_boarding_points_in_walking_range(routing_engine.return_node_position(stop_node), max_walking_range=walking_range)
    for stop_node, boarding_nodes in ap_to_bp_dis.items():
        line_values = [str(stop_node[0])]
        for boarding_node_pos, dist in sorted(boarding_nodes, key = lambda x:x[1]):
            line_values.append("{};{}".format(boarding_node_pos[0], dist))
        lines.append(",".join(line_values))

    with open(os.path.join(bp_path, "closest_bps_preprocessed.csv"), "w") as f:
        f.write("\n".join(lines))

def compute_part_boarding_points_in_walking_range(nw_name, infra_name, walking_range, part_bp_pos_list, all_nodes):
    nw_path = os.path.join(tum_fleet_sim_path, 'data', 'networks', nw_name)
    bp_path = os.path.join(tum_fleet_sim_path, 'data', 'infra', infra_name, nw_name)

    if not os.path.isdir(nw_path):
        print("network {} not found!".format(nw_name))
        exit()
    if not os.path.isdir(bp_path):
        print("boarding points {} for nw {} not found!".format(infra_name, nw_name))
        exit()

    routing_engine = Network(nw_path)
    bp_infra = BoardingPointInfrastructure(bp_path, routing_engine, load_preprocessed = False)

    ap_to_bp_dis = {}
    for i, bp_node in enumerate(part_bp_pos_list):
        if i%100 == 0 and i>0:
            print(" ... {}/{} done".format(i, len(part_bp_pos_list)))
        aps = find_access_node_from_boarding_point(routing_engine, bp_node, all_nodes, walking_range)
        for ap, dis in aps:
            try:
                ap_to_bp_dis[ap].append( (bp_node, dis) )
            except:
                ap_to_bp_dis[ap] = [(bp_node, dis)]
    return ap_to_bp_dis

def preprocess_closest_bp_for_each_node(nw_name, infra_name, must_stop_only = False, number_cores = 1):
    nw_path = os.path.join(tum_fleet_sim_path, 'data', 'networks', nw_name)
    bp_path = os.path.join(tum_fleet_sim_path, 'data', 'infra', infra_name, nw_name)

    if not os.path.isdir(nw_path):
        print("network {} not found!".format(nw_name))
        exit()
    if not os.path.isdir(bp_path):
        print("boarding points {} for nw {} not found!".format(infra_name, nw_name))
        exit()

    routing_engine = Network(nw_path)
    bp_infra = BoardingPointInfrastructure(bp_path, routing_engine, load_preprocessed = True)

    if not must_stop_only:
        all_nodes = [routing_engine.return_node_position(i.node_index) for i in routing_engine.get_node_list()]
    else:
        all_nodes = [routing_engine.return_node_position(i.node_index) for i in routing_engine.get_node_list() if i.is_stop_only]

    node_to_bp_dis = {} # node_index -> (nearest_bp, dis)

    to_remove = []
    for node_pos in all_nodes:
        r = False
        for wd, bps in bp_infra.computed_possible_boarding_points.get(node_pos, {}).items():
            if len(bps) > 0:
                closest = min(bps, key = lambda x:x[1])
                node_to_bp_dis[node_pos[0]] = (closest[0][0], closest[1])
                r = True
                break
        if r:
            to_remove.append(node_pos)
    print("not needed to be preprocess: {}".format(len(to_remove)))
    for x in to_remove:
        all_nodes.remove(x)

    if number_cores == 1:
        c = 0
        for node_pos in all_nodes:
            closest_bps = bp_infra.return_boarding_points_in_walking_range(node_pos, 0, max_boarding_points=1, return_nearest_else=True)
            if len(closest_bps) > 0:
                closest = min(closest_bps, key = lambda x:x[1])
                node_to_bp_dis[node_pos[0]] = (closest[0][0], closest[1])
            else:
                node_to_bp_dis[node_pos[0]] = (-1, -1)
            c += 1
            if c % 500 == 0:
                print("...{}/{} done".format(c, len(all_nodes)))

        df_list = [{"node_index" : n, "closest_bp_index" : c[0], "walking_distance" : c[1]} for n, c in node_to_bp_dis.items()]
    else:
        df_list = [{"node_index" : n, "closest_bp_index" : c[0], "walking_distance" : c[1]} for n, c in node_to_bp_dis.items()]
        all_len = len(all_nodes)
        per_cor = int(np.math.ceil(all_len/number_cores))
        print("all nodes {} | per core {} | n cores {}".format(all_len, per_cor, number_cores))
        part_lists = []
        for i in range(number_cores):
            part_lists.append(all_nodes[(i*per_cor) : min([(i+1)*per_cor, len(all_nodes)])])
        p = Pool(number_cores)
        x = [p.apply_async(return_part_closest_bp_for_each_node, args=(nw_name, infra_name, part_node_pos)) for part_node_pos in part_lists]
        for z in x:
            df_list += z.get()
        p.close()

    
    df = pd.DataFrame(df_list)
    df.to_csv(os.path.join(bp_path, "nearest_bp.csv"), index=False)

def return_part_closest_bp_for_each_node(nw_name, infra_name, node_pos_list):

    print("preprocess {} nodes".format(len(node_pos_list)))
    nw_path = os.path.join(tum_fleet_sim_path, 'data', 'networks', nw_name)
    bp_path = os.path.join(tum_fleet_sim_path, 'data', 'infra', infra_name, nw_name)

    if not os.path.isdir(nw_path):
        print("network {} not found!".format(nw_name))
        exit()
    if not os.path.isdir(bp_path):
        print("boarding points {} for nw {} not found!".format(infra_name, nw_name))
        exit()

    routing_engine = Network(nw_path)
    bp_infra = BoardingPointInfrastructure(bp_path, routing_engine, load_preprocessed = True)

    node_to_bp_dis = {} # node_index -> (nearest_bp, dis)

    c = 0
    for node_pos in node_pos_list:
        closest_bps = bp_infra.return_boarding_points_in_walking_range(node_pos, 0, max_boarding_points=1, return_nearest_else=True)
        if len(closest_bps) > 0:
            closest = min(closest_bps, key = lambda x:x[1])
            node_to_bp_dis[node_pos[0]] = (closest[0][0], closest[1])
        else:
            node_to_bp_dis[node_pos[0]] = (-1, -1)
        c += 1
        if c % 500 == 0:
            print("...{}/{} done".format(c, len(node_pos_list)))

    df_list = [{"node_index" : n, "closest_bp_index" : c[0], "walking_distance" : c[1]} for n, c in node_to_bp_dis.items()]

    return df_list


if __name__ == "__main__":
    """
    this script preprocesses walking distances from all nodes to the nearest boarding points
    in walking range. creates file "closest_bps_preprocessed.csv" which is loaded in BoardingPointInfraStructure.py

    usage of script:
    either import module in other script and call preprocess_boarding_points_in_walking_range
    or call with arguments:
    first argument: name of network folder to apply on
    second argument: name of infra_folder to apply on (boarding_points.csv has to be given!)
    third argument: maximum walking distance in m
    forth argument (optional): indicates if only must_stop_nodes are preprocessed otherwise all network nodes (1 for True, 0 for False). True is standard
    """

    if len(sys.argv) > 2:
        nw_name = sys.argv[1]
        infra_name = sys.argv[2]
        max_walking_range = float(sys.argv[3])
        if len(sys.argv) > 3:
            if sys.argv[4] == 1:
                must_stop_only = True
            else:
                must_stop_only = False
        else:
            must_stop_only = False
        preprocess_boarding_points_in_walking_range(nw_name, infra_name, max_walking_range, must_stop_only=must_stop_only)
        preprocess_closest_bp_for_each_node(nw_name, infra_name, max_walking_range, must_stop_only=must_stop_only)
    else:
        print("wrong usage of script! read string in __main__-function!")
        raise EnvironmentError
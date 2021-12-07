import os 
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool
fleet_sim_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
try:
    from src.routing.NetworkBasic import NetworkBasic as Network
except:
    #fleet_sim_path = r'C:\Users\ge37ser\Documents\Coding\TUM_VT_FleetSimulation\tum-vt-fleet-simulation'    #to be adopted
    os.sys.path.append(fleet_sim_path)
    from src.routing.NetworkBasic import NetworkBasic as Network

""" this script is used to preprocess travel time tables for the routing_engine
        NetworkPartialPreprocessed.py
in this routing engine the fastest routes between the node index [0, x[ are preprocessed and their
travel times and travel distances are stored in a numpy-matrix which can be read by the routing engine during
a simulation.
Note, that these first x note should optimally be network nodes between routing tasks are querried frequently compared to other node combinations.
Usually, this nodes correspond to depots, boardingpoints, nodes with the "stop_only" attribute.
Methods from the file network_manipulation.py can be used to sort node_indices for these priority nodes
"""

def get_travel_times_available(network_path):
    """ finds all travel time folders for a given network path.
    each folder will be preprocessed 
    :param network_path: full path to netork folder
    :return: list of times with different network travel times + None (free flow)
    """
    routing_engine = Network(network_path)
    travel_times = list(routing_engine.travel_time_file_folders.keys())
    return [None] + travel_times

def load_special_infra_nodes(infra_name, network_name):
    """ this function loads depot and boarding node_indices from a given matched infra directory.
    If they should be used as preprocessed nodes, they have to span the indices [0, x[ !
    :param infra_name: str name of the used infrastructure dir
    :param network_name: str name of the used network (they have to be matched!)
    :return: sorted list of node indices
    """
    infra_path = os.path.join(fleet_sim_path, "data", "infra", infra_name, network_name)
    if not os.path.isdir(infra_path):
        print("infra data not found: {}".format(infra_path))
        exit()
    special_nodes = {}
    if os.path.isfile(os.path.join(infra_path, "boarding_points.csv")):
        print("... boarding points file found")
        bp_df = pd.read_csv(os.path.join(infra_path, "boarding_points.csv"))
        for _, entries in bp_df.iterrows():
            special_nodes[int(entries["node_index"])] = 1
    if os.path.isfile(os.path.join(infra_path, "depots.csv")):
        print("... depot file found")
        depot_df = pd.read_csv(os.path.join(infra_path, "depots.csv"))
        for _, entries in depot_df.iterrows():
            special_nodes[int(entries["node_index"])] = 1
    return list(sorted(list(special_nodes.keys())))
    

def createXto1StopNodeTravelInfoTables(network_path, time_range, sim_time = None, special_nodes = None):
    """ TODO wrong: this function creates the travel time tables "tt_matrix" and "dis_matrix" and stores them
    in the network-folder to be read by the routing_engine NetworkPartialPreprocessed.py
    :param network_path: full path to the corresponding network
    :param sim_time: time of travel times files that should be preprocessed (None corresponds to "base")
    :param special_nodes: None: preprocessing between all nodes with the "is_stop_only"-attribute
                            list of node indices: preprocessing between all given nodes
                            Note: in both cases they have to be sorted!
    """
    print("preprocessing {} for time {}".format(network_path, sim_time))
    routing_engine = Network(network_path, scenario_time=sim_time)
    if special_nodes is None:
        stop_nodes = routing_engine.get_must_stop_nodes()
    else:
        stop_nodes = special_nodes
    #stop_nodes = [i for i in range(10)]
    for i, node_index in enumerate(stop_nodes):
        if i != node_index:
            raise EnvironmentError("stop nodes are not sorted! all stop nodes have to be at node indices [0, N[! counter {} stop_node_index {}".format(i, node_index))
    
    max_length = 0
    routing_table = {}

    stop_node_positions = [routing_engine.return_node_position(n) for n in stop_nodes]
    all_pos = [routing_engine.return_node_position(n) for n in range(len(routing_engine.nodes))]
    length = len(stop_nodes)
    print("start preprocessing")
    N_rows = 0
    for i, stop_node_pos in enumerate(stop_node_positions):
        if i%100 == 0:
            print(" ... {}/{} done! max length {}".format(i, length, max_length))
            print(".... -> number rows {}".format(N_rows))
        res = routing_engine.return_travel_costs_Xto1(all_pos, stop_node_pos, max_cost_value=time_range)
        res = sorted(res, key = lambda x:x[2])
        l = 0
        routing_table[stop_node_pos[0]] = {}
        for o_pos, _, tt, dis in res:
            if tt > time_range:
                break
            try:
                routing_table[stop_node_pos[0]][o_pos[0]] = (tt, dis)
            except:
                routing_table[stop_node_pos[0]] = {o_pos[0] : (tt, dis)}
            l += 1
            N_rows += 1

    matrix = np.zeros( (N_rows, 4), dtype=float)
    c = 0
    for stop_node, stop_node_dict in routing_table.items():
        for o_node, tt_dis in stop_node_dict.items():
            matrix[c][0] = float(stop_node)
            matrix[c][1] = float(o_node)
            matrix[c][2] = tt_dis[0]
            matrix[c][3] = tt_dis[1]

    if sim_time is None:
        folder_out = os.path.join(network_path, "base")
    else:
        folder_out = routing_engine.travel_time_file_folders[sim_time]
    np.save(os.path.join(folder_out, "Xto1_matrix_{}".format(int(time_range))), matrix)

def preprocess(network_path, time_range, number_cores = 1, special_nodes = None):
    """ computes the preprocessed travel time tables for all different travel time files
    creates the travel time tables "tt_matrix" and "dis_matrix" and stores them
    in the network-folder
    :param network_path: path to the corresponding network file
    :param number_cores: number of travel_time files that are preprocessed in parallel
    :param special_nodes: None: preprocessing between all nodes with the "is_stop_only"-attribute
                            list of node indices: preprocessing between all given nodes
                            Note: in both cases they have to be sorted!
    """
    travel_times_to_compute = get_travel_times_available(network_path)
    if number_cores == 1:
        for time in travel_times_to_compute:
            createXto1StopNodeTravelInfoTables(network_path, time_range, sim_time=time, special_nodes=special_nodes)
    else:
        p = Pool(number_cores)
        x = [p.apply_async(createXto1StopNodeTravelInfoTables, args=(network_path, time_range, t, special_nodes)) for t in travel_times_to_compute]
        y = [z.get() for z in x]
        p.close()


if __name__ == "__main__":
    #nw_path = r'C:\Users\ge37ser\Documents\Coding\TUM_VT_FleetSimulation\tum-vt-fleet-simulation\data\networks\Aimsun_Munich_2020_04_15_Majid_reduced_ER_BP_all'
    nw_name = "MOIA_HH_22122020_reduced"
    infra_name = "MOIA_base"
    nw_path = os.path.join(fleet_sim_path, "data", "networks", nw_name)
    special_nodes = load_special_infra_nodes(infra_name, nw_name)
    # special_nodes = special_nodes[:6]
    # print(special_nodes)
    preprocess(nw_path, 720, number_cores=1, special_nodes=special_nodes)
import os 
import sys
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool
fleet_sim_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
os.sys.path.append(fleet_sim_path)
try:
    from src.routing.NetworkBasicCpp import NetworkBasicCpp as Network
except:
    print("cpp router not found")
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

def get_travel_times_available(network_path, network_dynamics_file):
    """ finds all travel time folders for a given network path.
    each folder will be preprocessed 
    :param network_path: full path to netork folder
    :return: list of times with different network travel times + None (free flow)
    """
    routing_engine = Network(network_path, network_dynamics_file)
    travel_times = []
    folders_found = {}
    for sim_time, folder in routing_engine.travel_time_file_folders.items():
        if folders_found.get(folder):
            continue
        folders_found[folder] = 1
        travel_times.append(sim_time)
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
    

def createStopNodeTravelInfoTables(network_path, network_dynamics_file = None, sim_time = None, special_nodes = None):
    """ this function creates the travel time tables "tt_matrix" and "dis_matrix" and stores them
    in the network-folder to be read by the routing_engine NetworkPartialPreprocessed.py
    :param network_path: full path to the corresponding network
    :param sim_time: time of travel times files that should be preprocessed (None corresponds to "base")
    :param special_nodes: None: preprocessing between all nodes with the "is_stop_only"-attribute
                            list of node indices: preprocessing between all given nodes
                            Note: in both cases they have to be sorted!
    """
    print("preprocessing {} for time {}".format(network_path, sim_time))
    routing_engine = Network(network_path, network_dynamics_file_name=network_dynamics_file, scenario_time=sim_time)
    if special_nodes is None:
        stop_nodes = routing_engine.get_must_stop_nodes()
    else:
        stop_nodes = special_nodes
    #stop_nodes = [i for i in range(10)]
    for i, node_index in enumerate(stop_nodes):
        if i != node_index:
            raise EnvironmentError("stop nodes are not sorted! all stop nodes have to be at node indices [0, N[! counter {} stop_node_index {}".format(i, node_index))
    tt_matrix = np.ones( (len(stop_nodes), len(stop_nodes) ) ) * np.inf
    dis_matrix = np.ones( (len(stop_nodes), len(stop_nodes) ) ) * np.inf
    print("start computing travel times")
    t = time.time()
    node_positions = [routing_engine.return_node_position(n) for n in stop_nodes]
    for i, s in enumerate(node_positions):
        r = routing_engine.return_travel_costs_1toX(s, node_positions)
        for e_pos, _, tt, dis in r:
            o_index = s[0]
            d_index = e_pos[0]
            tt_matrix[o_index][d_index] = tt
            dis_matrix[o_index][d_index] = dis
        if i % 100 == 0:
            print(" .... {}/{} done!".format(i, len(node_positions)))
    print(" ... done after {}s".format(time.time() - t))
    if sim_time is None:
        folder_out = os.path.join(network_path, "base")
    else:
        folder_out = routing_engine.travel_time_file_folders[sim_time]
    np.save(os.path.join(folder_out, "tt_matrix"), tt_matrix)
    np.save(os.path.join(folder_out, "dis_matrix"), dis_matrix)

def preprocess(network_path, network_dynamics_file = None, number_cores = 1, special_nodes = None):
    """ computes the preprocessed travel time tables for all different travel time files
    creates the travel time tables "tt_matrix" and "dis_matrix" and stores them
    in the network-folder
    :param network_path: path to the corresponding network file
    :param number_cores: number of travel_time files that are preprocessed in parallel
    :param network_dynamics_file: network dynamicsfile
    :param special_nodes: None: preprocessing between all nodes with the "is_stop_only"-attribute
                            list of node indices: preprocessing between all given nodes
                            Note: in both cases they have to be sorted!
    """
    travel_times_to_compute = get_travel_times_available(network_path, network_dynamics_file)
    print(travel_times_to_compute)
    if number_cores == 1:
        for time in travel_times_to_compute:
            createStopNodeTravelInfoTables(network_path, network_dynamics_file, sim_time=time, special_nodes=special_nodes)
    else:
        p = Pool(number_cores)
        x = [p.apply_async(createStopNodeTravelInfoTables, args=(network_path, network_dynamics_file, t, special_nodes)) for t in travel_times_to_compute]
        y = [z.get() for z in x]
        p.close()

def preprocess_with_given_infra(nw_name, infra_name, number_cores = 1, network_dynamics_file = None):
    """ computes the preprocessed travel time tables for all different travel time files
    creates the travel time tables "tt_matrix" and "dis_matrix" and stores them
    in the network-folder
    :param network_name: name to the corresponding network file
    :param infra_name: look for all boarding and depot nodes and preprocess travel times within these nodes
                            Note that these node indices have to be ordererd from 0 to N_prep
    :param number_cores: number of travel_time files that are preprocessed in parallel
    :param network_dynamics_file: network dynamicsfile
    """
    nw_path = os.path.join(fleet_sim_path, "data", "networks", nw_name)
    special_nodes = load_special_infra_nodes(infra_name, nw_name)
    preprocess(nw_path, network_dynamics_file=network_dynamics_file, number_cores=number_cores, special_nodes=special_nodes)

def preprocess_until_max_index(nw_name, max_index, number_cores = 1, network_dynamics_file = None):
    """ computes the preprocessed travel time tables for all different travel time files
    creates the travel time tables "tt_matrix" and "dis_matrix" and stores them
    in the network-folder
    :param network_name: name to the corresponding network file
    :param max_index: preprocesses the network between all nodes with indices from 0 to max_index (excluded)
    :param number_cores: number of travel_time files that are preprocessed in parallel
    :param network_dynamics_file: network dynamicsfile
    """
    nw_path = os.path.join(fleet_sim_path, "data", "networks", nw_name)
    special_nodes = [i for i in range(max_index)]
    preprocess(nw_path, network_dynamics_file=network_dynamics_file, number_cores=number_cores, special_nodes=special_nodes)


if __name__ == "__main__":
    """ this script computes numpy travel time tables "tt_matrix.npy" and "dis_matrix.npy" which are used in
    the routing engine "NetworkPartialPreprocessed.py" and NetworkPartialPreprocessedCpp.py".
    These travel time tables stores travel time and distance information of the fastest route between node i and node j.
    in the tables i,j are indices of the tables.
    Therefore, preprocessing is only feasible within a set of the first N_prep nodes.
    U should therefore make sure, that the network-nodes are ordered in a way, that the nodes with the smallest indices
    correspond to network nodes which travel times are queried regularly by the routing engine within the simulation

    usage:
    either import the module and call corresponding methods in script 
    or:

    script arguments:
    0: name of the network to preprocess
    1: either integer number of the maximum node-index to preprocess (test for int!)
        or: name of a infra-structure directory specifying the nodes to preprocess (infra-nodes have to be sorted from 0 to N_max!)
    2 (optional): int number of cores to use for preprocessing
    3 (optional): name of network dynamics file
    """
    number_cores = 1
    network_dynamics_file = None
    nw_name = sys.argv[1]
    try:
        number_cores = int(sys.argv[3])
    except:
        pass
    try:
        network_dynamics_file = sys.argv[4]
    except:
        pass
    try:
        N_max = int(sys.argv[2])
    except:
        N_max = None
    if N_max is not None:
        preprocess_until_max_index(nw_name, N_max, number_cores=number_cores, network_dynamics_file=network_dynamics_file)
    else:
        infra_name = sys.argv[2]
        preprocess_with_given_infra(nw_name, infra_name, number_cores=number_cores, network_dynamics_file=network_dynamics_file)

import sys
import os
import time
import numpy as np
import pandas as pd
from numba import njit, prange

LARGE = np.Inf


def addEdgeToAdjacencyMatrix(column, A, B):
    A[int(column["from_node"]), int(column["to_node"])] = column["use_tt"]
    B[int(column["from_node"]), int(column["to_node"])] = column["distance"]


def createAdjacencyMatrix(nw_dir, scenario_time=None):
    print("\t ... creating adjacency matrix A ...")
    # find out about size of matrix
    node_f = os.path.join(nw_dir, "base", "nodes.csv")
    node_df = pd.read_csv(node_f)
    number_nodes = len(node_df)
    # no routing through stop nodes possible!
    stop_only_node_df =  node_df[node_df["is_stop_only"] == True]
    set_stop_nodes = set(stop_only_node_df["node_index"])
    # travel time: A | distance: B
    A = LARGE * np.ones((number_nodes, number_nodes))
    B = LARGE * np.ones((number_nodes, number_nodes))
    # create adjacency matrix from edges
    edge_f = os.path.join(nw_dir, "base", "edges.csv")
    edge_df = pd.read_csv(edge_f)
    if scenario_time is None:
        edge_df["use_tt"] = edge_df["travel_time"].copy()
    else:
        edge_df.set_index(["from_node", "to_node"], inplace=True)
        tmp_edge_f = os.path.join(nw_dir, scenario_time, "edges_td_att.csv")
        tmp_edge_df = pd.read_csv(tmp_edge_f, index_col=[0,1])
        tmp_edge_df.rename({"edge_tt":"use_tt"}, axis=1, inplace=True)
        edge_df = pd.merge(edge_df, tmp_edge_df, left_index=True, right_index=True)
        edge_df = edge_df.reset_index()
    edge_df.apply(addEdgeToAdjacencyMatrix, axis=1, args=(A,B))
    for i in range(number_nodes):
        A[i,i] = 0.0
    for i in range(number_nodes):
        B[i,i] = 0.0
    return A, B, number_nodes, set_stop_nodes


# computation time test: hardly difference between parallel=False and parallel=True
@njit(parallel=True)
def fw_lvl23(k, A, B, number_nodes):
    for i in prange(number_nodes):
        for j in range(number_nodes):
            tmp = A[i,k]+A[k,j]
            if tmp < A[i,j]:
                A[i,j] = tmp
                B[i,j] = B[i,k]+B[k,j]


def create_travel_time_table(nw_dir, scenario_time=None, save_npy=True, save_csv=False):
    (A, B, number_nodes, set_stop_nodes) = createAdjacencyMatrix(nw_dir, scenario_time)
    print("Running Floyd Warshall (travel time and distance of fastest route) in mp numba mode ...")
    t0 = time.perf_counter()
    for k in range(number_nodes):
        # do not route through stop nodes!
        if k in set_stop_nodes:
            print(f"Skipping intermediary routing via stop node {k}")
            continue
        if k%250 == 0:
            print(f"\t ... loop {k}/{number_nodes}")
        fw_lvl23(k, A, B, number_nodes)
    cpu_time = round(time.perf_counter() - t0, 3)
    print(f"\t ... finished in {cpu_time} seconds")
    #
    if scenario_time is None:
        output_dir = os.path.join(nw_dir, "ff", "tables")
    else:
        output_dir = os.path.join(nw_dir, scenario_time, "tables")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    #
    print(f"\t ... saving files to {output_dir} ...")
    if save_npy:
        out_f_npy = os.path.join(output_dir, f"nn_fastest_tt.npy")
        np.save(out_f_npy, A)
    if save_csv:
        out_f_csv = os.path.join(output_dir, f"nn_fastest_tt.csv")
        indices = [x for x in range(number_nodes)]
        df = pd.DataFrame(A, columns=indices)
        df.to_csv(out_f_csv)
    #
    if save_npy:
        dist_f_npy = os.path.join(output_dir, f"nn_fastest_distance.npy")
        np.save(dist_f_npy, B)
    if save_csv:
        dist_f_csv = os.path.join(output_dir, f"nn_fastest_distance.csv")
        indices = [x for x in range(number_nodes)]
        df = pd.DataFrame(B, columns=indices)
        df.to_csv(dist_f_csv)
    #
    return cpu_time


if __name__ == "__main__":
    network_name_dir = sys.argv[1]
    try:
        scenario_time = sys.argv[2]
    except:
        scenario_time = None
    create_travel_time_table(network_name_dir, scenario_time)

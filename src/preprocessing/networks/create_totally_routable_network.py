import os
import random
import shutil
import pandas as pd
import numpy as np

ASSUMED_TD_SO = 500
ASSUMED_TT_SO = 60
ASSUMED_TD_NSO = 1000
ASSUMED_TT_NSO = 120
SQ_MAX_DIST = 1000**2
TEST_VAL = 10000


def connect_with_tt_matrix(tt_m, n_df, e_df, so_df, check_outgoing, check_incoming, found_for_n):
    test_node = None
    while not test_node:
        tmp = random.choice(n_df.index)
        if tmp not in check_outgoing.keys() and tmp not in check_incoming.keys():
            test_node = tmp

    # method 1: connect with each other
    add_io_edge_counter = 0
    for from_n in check_outgoing.keys():
        if from_n in found_for_n:
            continue
        f_pos_x = n_df.loc[from_n, "pos_x"]
        f_pos_y = n_df.loc[from_n, "pos_y"]
        for to_n in check_incoming.keys():
            if to_n in found_for_n:
                continue
            t_pos_x = n_df.loc[to_n, "pos_x"]
            t_pos_y = n_df.loc[to_n, "pos_y"]
            if (t_pos_x - f_pos_x)**2 + (t_pos_y - f_pos_y)**2 <= SQ_MAX_DIST:
                if tt_m[to_n,test_node] < TEST_VAL:
                    e_df = e_df.append({"from_node": from_n, "to_node": to_n, "distance": ASSUMED_TD_NSO,
                                        "travel_time": ASSUMED_TT_NSO, "source_edge_id": ""}, ignore_index=True)
                    add_io_edge_counter += 1
                    found_for_n.add(to_n)
                    found_for_n.add(from_n)
                    break
    # method 2: connect with other, meaningful nearby node
    add_other_edge_counter = 0
    for from_n in check_outgoing.keys():
        if from_n in found_for_n:
            continue
        f_pos_x = n_df.loc[from_n, "pos_x"]
        f_pos_y = n_df.loc[from_n, "pos_y"]
        tmp_df = n_df.copy()
        tmp_df["sq_dist"] = tmp_df.apply(lambda x: (x["pos_x"] - f_pos_x)**2 + (x["pos_y"] - f_pos_y)**2, axis=1)
        tmp_df.sort_values("sq_dist", inplace=True)
        for to_n, row in tmp_df.iterrows():
            if from_n == to_n or to_n in so_df.index:
                continue
            if tt_m[to_n, test_node] < TEST_VAL:
                e_df = e_df.append({"from_node": from_n, "to_node": to_n, "distance": ASSUMED_TD_NSO,
                                    "travel_time": ASSUMED_TT_NSO, "source_edge_id": ""}, ignore_index=True)
                add_other_edge_counter += 1
                found_for_n.add(to_n)
                found_for_n.add(from_n)
                break
    for to_n in check_incoming.keys():
        if to_n in found_for_n:
            continue
        t_pos_x = n_df.loc[to_n, "pos_x"]
        t_pos_y = n_df.loc[to_n, "pos_y"]
        tmp_df = n_df.copy()
        tmp_df["sq_dist"] = tmp_df.apply(lambda x: (x["pos_x"] - t_pos_x) ** 2 + (x["pos_y"] - t_pos_y) ** 2, axis=1)
        tmp_df.sort_values("sq_dist", inplace=True)
        for from_n, row in tmp_df.iterrows():
            if from_n == to_n  or from_n in so_df.index:
                continue
            if tt_m[test_node, from_n] < TEST_VAL:
                e_df = e_df.append({"from_node": from_n, "to_node": to_n, "distance": ASSUMED_TD_NSO,
                                    "travel_time": ASSUMED_TT_NSO, "source_edge_id": ""}, ignore_index=True)
                add_other_edge_counter += 1
                found_for_n.add(to_n)
                found_for_n.add(from_n)
                break
    return e_df, add_io_edge_counter, add_other_edge_counter


def add_edges_to_make_nw_completely_routable_tt_matrix(nw_main_dir):
    """This script adds edges from nodes without next edges to the next node with outgoing edges.
    No new nodes are created, hence the network can be used as is, but all nodes should be reachable.
    The old edges.csv is saved as orig_edges.csv unless this file already exists.

    BEWARE: This script assumes the availability of a tt_matrix file.

    :param nw_main_dir: network main directory
    :return: None
    """
    # read files
    print("reading data ...")
    tt_m_f = os.path.join(nw_main_dir, "ff", "tables", "nn_fastest_tt.npy")
    if os.path.isfile(tt_m_f):
        tt_m = np.load(tt_m_f)
    else:
        tt_m = None
    n_f = os.path.join(nw_main_dir, "base", "nodes.csv")
    n_df = pd.read_csv(n_f, index_col=0)
    e_f = os.path.join(nw_main_dir, "base", "edges.csv")
    e_df = pd.read_csv(e_f)

    # make copy
    e_f_copy = os.path.join(nw_main_dir, "base", "orig_edges.csv")
    if not os.path.isfile(e_f_copy):
        shutil.copy(e_f, e_f_copy)

    # identify stop only nodes; these are not fully routable
    so_df = n_df[n_df["is_stop_only"]].copy()

    # count outgoing and incoming edges (ignoring stop only nodes)
    print("checking connections ...")
    connect = {}  # so_id -> -1/+1 -> list nodes
    print("\t...outgoing")
    check_outgoing = {}  # node_id -> -1 / so_id
    unique_from = e_df["from_node"].unique()
    for out_n in n_df.index:
        if out_n not in unique_from:
            check_outgoing[out_n] = -1
    for out_n, out_n_df in e_df.groupby("from_node"):
        not_so_df = out_n_df[~out_n_df["to_node"].isin(so_df.index)]
        if not_so_df.shape[0] == 0:
            if out_n.shape[0] > 0:
                so_n = out_n_df["to_node"].tolist()[0]
                check_outgoing[out_n] = so_n
                try:
                    connect[so_n][-1].append(out_n)
                except KeyError:
                    connect[so_n] = {-1: out_n}
            else:
                check_outgoing[out_n] = -1
    print("\t...incoming")
    check_incoming = {}  # node_id -> -1 / so_id
    unique_to = e_df["to_node"].unique()
    for in_n in n_df.index:
        if in_n not in unique_to:
            check_incoming[in_n] = -1
    for in_n, in_n_df in e_df.groupby("to_node"):
        not_so_df = in_n_df[~in_n_df["from_node"].isin(so_df.index)]
        if not_so_df.shape[0] == 0:
            if in_n_df.shape[0] > 0:
                so_n = in_n_df["from_node"].tolist()[0]
                check_incoming[in_n] = so_n
                try:
                    connect[so_n][1].append(in_n)
                except KeyError:
                    connect[so_n] = {1: in_n}
            else:
                check_incoming[in_n] = -1
    print(f"\t\tnodes to be connected: {len(check_outgoing)} without outgoing edges and {len(check_incoming)} without"
          f" incoming edges")

    test_node = None
    while not test_node:
        tmp = random.choice(n_df.index)
        if tmp not in check_outgoing.keys() and tmp not in check_incoming.keys():
            test_node = tmp
    found_for_n = set()
    # first connect nodes that are connected via a stop_only node
    print("connecting nodes that are connected via stop-only nodes ...")
    add_so_edge_counter = 0
    for so_n, connect_dict in connect.items():
        if -1 in connect_dict.keys() and +1 in connect_dict.keys():
            def_from_n = connect_dict[-1][0]
            for to_n in connect_dict[1]:
                if tt_m[to_n, test_node] < TEST_VAL:
                    e_df = e_df.append({"from_node": def_from_n, "to_node": to_n, "distance": ASSUMED_TD_SO,
                                        "travel_time": ASSUMED_TT_SO, "source_edge_id": ""}, ignore_index=True)
                    add_so_edge_counter += 1
                    found_for_n.add(to_n)
            found_for_n.add(def_from_n)
            def_to_n = connect_dict[1][0]
            # from 0 -> 0 already was made
            for from_n in connect_dict[-1][1:]:
                if tt_m[test_node, from_n] < TEST_VAL:
                    e_df = e_df.append({"from_node": from_n, "to_node": def_to_n, "distance": ASSUMED_TD_SO,
                                        "travel_time": ASSUMED_TT_SO, "source_edge_id": ""}, ignore_index=True)
                    add_so_edge_counter += 1
                    found_for_n.add(from_n)
    print(f"\tadded {add_so_edge_counter} edges that are connected via a stop-only node")

    # take care of rest
    print(f"taking care of remaining nodes to be connected: {len(check_outgoing)} without outgoing edges"
          f" and {len(check_incoming)} without incoming edges")
    if tt_m is not None:
        e_df, added_io_edges, add_other_edges = connect_with_tt_matrix(tt_m, n_df, e_df, so_df, check_outgoing,
                                                                       check_incoming, found_for_n)
    else:
        raise NotImplementedError("Implementation currently requires existence of TT matrix!")
    print(f"\tadded {added_io_edges} new edges between boundary nodes and {add_other_edges} to typical nodes")

    # save
    total_added_edges = add_so_edge_counter + added_io_edges + add_other_edges
    if total_added_edges > 0:
        e_df.to_csv(e_f, index=False)
        print(f"\tsaved new edge file {e_f}! Remember to run network preprocessing again!")
    else:
        print("\t no new edges were added!")


if __name__ == "__main__":
    nw_main_dir = r"C:\Users\ne53qez\Data_and_Simulation\tum-vt-fleet-simulation\data\networks\MUNbene_withBPs_300_1_LHMArea_OVstations_reduced_Flo"
    add_edges_to_make_nw_completely_routable_tt_matrix(nw_main_dir)

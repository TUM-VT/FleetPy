import os
import sys
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(MAIN_DIR)
from src.misc.globals import *


def get_node_demand(demand_network_dir, glob_str):
    """This method creates a dictionary that counts all demands for a node (departure & arrival).

    :return: node_demand: node_index -> demand_count
    :rtype: dict
    """
    print("\n\n... collecting demand per node")
    node_demand_df = None
    if glob_str is None:
        list_demand_f = glob.glob(f"{demand_network_dir}/*.csv")
    else:
        print(f"... using glob-str {glob_str}")
        list_demand_f = glob.glob(f"{demand_network_dir}/{glob_str}")
    number_f = len(list_demand_f)
    print(f"... found {number_f} files")
    counter = 0
    prt_str = ""
    for f in list_demand_f:
        counter += 1
        if counter % 10 == 0:
            print(f"\t ... processing file {counter}/{number_f}")
        tmp_df = pd.read_csv(f)
        # print(tmp_df.head())
        o_series = tmp_df[G_RQ_ORIGIN].value_counts()
        d_series = tmp_df[G_RQ_DESTINATION].value_counts()
        sum_df = pd.concat([o_series, d_series], axis=1)
        sum_df.fillna(0, inplace=True)
        sum_df["add_node_demand"] = sum_df[G_RQ_ORIGIN] + sum_df[G_RQ_DESTINATION]
        if node_demand_df is None:
            node_demand_df = sum_df[["add_node_demand"]]
            node_demand_df.rename({"add_node_demand": "node_demand"}, axis=1, inplace=True)
        else:
            node_demand_df["add_node_demand"] = sum_df["add_node_demand"]
            node_demand_df.fillna(0, inplace=True)
            node_demand_df["node_demand"] = node_demand_df["node_demand"] + node_demand_df["add_node_demand"]
            node_demand_df.drop("add_node_demand", axis=1, inplace=True)
        # print(node_demand_df)
        prt_str = f"File number:{counter}/{number_f} | Number of nodes:{len(node_demand_df)} |" \
                  f" Total demand: {node_demand_df['node_demand'].sum()}\nSummary:\n" \
                  f"{node_demand_df['node_demand'].describe()}"
        # print(prt_str)
    print(prt_str)
    return node_demand_df["node_demand"].to_dict()


def set_zone(row, min_edge_length, min_x, min_y, match_zones_gdf):
    x_bin = np.floor((row[G_NODE_X] - min_x) / min_edge_length)
    y_bin = np.floor((row[G_NODE_Y] - min_y) / min_edge_length)
    zone_str = f"{int(x_bin)};{int(y_bin)}"
    if zone_str in match_zones_gdf.index:
        return match_zones_gdf.loc[zone_str, "zone_id"]
    else:
        return -1


def create_quadratic_zones(network_base_dir_name, min_edge_length_str, number_aggregation_levels_str, zone_system_name,
                           consider_is_stop_only_str=False, demand_network_dir=None, glob_str=None):
    """This function creates quadratic zones over the network area defined by the node positions. It can build several
    aggregation levels based on the minimum edge length. The area will only contain zones that contain nodes.
    This script assumes that the positions are given in a metric system!

    :param network_base_dir_name: base directory with nodes.csv (assuming a metric system for pos_x and pos_y)
    :param min_edge_length_str: minimum edge length in meters
    :param number_aggregation_levels_str: number of aggregation levels (to more aggregated zones)
    :param zone_system_name: the zone system will be saved with this name
    :param consider_is_stop_only_str: only consider is_stop_only nodes for zone creation
    :param demand_network_dir: optional; path to matched network dir of demand; only consider zones with demand
    :param glob_str: can be used to filter demand glob file search
    :return: None
    """
    if consider_is_stop_only_str == "True":
        consider_is_stop_only = True
    else:
        consider_is_stop_only = False
    min_edge_length = int(min_edge_length_str)
    number_aggregation_levels = int(number_aggregation_levels_str)
    print(f"Building square zones for network {network_base_dir_name} with min_edge_length={min_edge_length} and"
          f"{number_aggregation_levels} aggregation levels ...")
    nodes_f = os.path.join(network_base_dir_name, "nodes.csv")
    nodes_df = pd.read_csv(nodes_f, index_col=0)
    all_nodes_df = nodes_df.copy()
    node_demand_dict = {}
    if consider_is_stop_only:
        print("\t considering only zones that have is_stop_only nodes")
        nodes_df = nodes_df[nodes_df["is_stop_only"]]
    if demand_network_dir is not None:
        print(f"\t considering only zones that have demand in {demand_network_dir}")
        node_demand_dict = get_node_demand(demand_network_dir, glob_str)
        nodes_df = nodes_df[nodes_df.index.isin(node_demand_dict.keys())]
    min_x = np.floor(nodes_df[G_NODE_X].min() / min_edge_length) * min_edge_length
    max_x = np.ceil(nodes_df[G_NODE_X].max() / min_edge_length) * min_edge_length
    print(f"x bins: {(max_x - min_x) / min_edge_length}")
    min_y = np.floor(nodes_df[G_NODE_Y].min() / min_edge_length) * min_edge_length
    max_y = np.ceil(nodes_df[G_NODE_Y].max() / min_edge_length) * min_edge_length
    print(f"y bins: {(max_y - min_y) / min_edge_length}")
    #
    print("\t ... getting x-y bins for the different aggregation levels")
    for lvl_agg in range(number_aggregation_levels):
        print(f"\t\t ... level {lvl_agg}")
        nodes_df[f"bin_x {lvl_agg}"] = nodes_df.apply(lambda x: np.floor((x[G_NODE_X] - min_x) /
                                                                         (min_edge_length*(2**lvl_agg))), axis=1)
        nodes_df[f"bin_y {lvl_agg}"] = nodes_df.apply(lambda x: np.floor((x[G_NODE_Y] - min_y) /
                                                                         (min_edge_length*(2**lvl_agg))), axis=1)
        print(f'\t\t\t ... x-bins: {nodes_df[f"bin_x {lvl_agg}"].unique()}')
        print(f'\t\t\t ... y-bins: {nodes_df[f"bin_y {lvl_agg}"].unique()}')
    #
    print("\t ... creating node <-> zone_id relation for nodes that have to be considered")
    for lvl_agg in range(number_aggregation_levels):
        print(f"\t\t ... level {lvl_agg}")
        if lvl_agg == 0:
            col_id = "zone_id"
            col_name = "zone_name"
        else:
            col_id = f"zone_agg {lvl_agg}"
            col_name = f"zone_agg_name {lvl_agg}"
        zone_counter = 0
        bin_id_dict = {}
        id_name_dict = {}
        for zone_xy, zone_xy_df in nodes_df.groupby([f"bin_x {lvl_agg}", f"bin_y {lvl_agg}"]):
            bin_id_dict[zone_xy] = zone_counter
            id_name_dict[zone_counter] = ";".join([str(int(x)) for x in zone_xy])
            zone_counter += 1
        nodes_df[col_id] = nodes_df.apply(lambda x: bin_id_dict[(x[f"bin_x {lvl_agg}"], x[f"bin_y {lvl_agg}"])], axis=1)
        nodes_df[col_name] = nodes_df.apply(lambda x: id_name_dict[x[col_id]], axis=1)
        print(f'\t\t\t ... number zones {zone_counter}')
        #
    #
    print("\t ... creating lowest level aggregation zone geometry")
    list_zones = []
    for labels, zero_lvl_zone_df in nodes_df.groupby(["zone_id", "zone_name"]):
        zone_entries = {G_ZONE_ZID: labels[0], G_ZONE_NAME: labels[1]}
        zone_xy = [int(x) for x in labels[1].split(";")]
        x_bounds = [min_x + (zone_xy[0] + xi)*min_edge_length for xi in range(2)]
        y_bounds = [min_y + (zone_xy[1] + yj)*min_edge_length for yj in range(2)]
        # clockwise definition of square
        zone_entries["geometry"] = Polygon([(x_bounds[0], y_bounds[0]), (x_bounds[0], y_bounds[1]),
                                            (x_bounds[1], y_bounds[1]), (x_bounds[1], y_bounds[0]),
                                            (x_bounds[0], y_bounds[0])])
        for lvl_agg in range(1,number_aggregation_levels):
            col_id = f"zone_agg {lvl_agg}"
            col_name = f"zone_agg_name {lvl_agg}"
            # higher aggregation zones will have the same value for all nodes in lower level aggregation zone
            zone_entries[col_id] = zero_lvl_zone_df[col_id].to_list()[0]
            zone_entries[col_name] = zero_lvl_zone_df[col_name].to_list()[0]
        list_zones.append(zone_entries)
    #
    print("\t ... checking coordinate reference system and creating GeoDataFrame")
    crs_f = os.path.join(network_base_dir_name, "crs.info")
    nodes_geojson = os.path.join(network_base_dir_name, "nodes_all_infos.geojson")
    if os.path.isfile(crs_f):
        with open(crs_f) as fhin:
            set_crs = {"init": fhin.read().strip()}
    elif os.path.isfile(nodes_geojson):
        tmp_gdf = gpd.read_file(nodes_geojson)
        set_crs = tmp_gdf.crs
    else:
        set_crs = None
    zones_gdf = gpd.GeoDataFrame(list_zones, crs=set_crs)
    #
    print("\t ... finding relation for all nodes and setting outside nodes to -1")
    m_zones_gdf = zones_gdf.set_index("zone_name")
    m_zones_gdf["centroid"] = m_zones_gdf["geometry"].centroid
    all_nodes_df["zone_id"] = all_nodes_df.apply(set_zone, args=(min_edge_length, min_x, min_y, m_zones_gdf), axis=1)
    all_nodes_df["is_centroid"] = 0
    #
    print("\t ... finding centroid nodes")
    for zone_id, tmp_df in all_nodes_df.groupby("zone_id"):
        if zone_id == -1:
            continue
        if consider_is_stop_only:
            tmp_df = tmp_df[tmp_df["is_stop_only"]]
        if demand_network_dir is not None:
            tmp_df = tmp_df[tmp_df.index.isin(node_demand_dict.keys())]
        centroid = zones_gdf.loc[zone_id, "geometry"].centroid
        tmp_df["distance"] = tmp_df.apply(lambda x: Point(x["pos_x"], x["pos_y"]).distance(centroid), axis=1)
        best_node_index = tmp_df["distance"].idxmin()
        all_nodes_df.loc[best_node_index, "is_centroid"] = 1
    #
    zone_output_dir = os.path.join(MAIN_DIR, "data", "zones", zone_system_name)
    zone_nw_output_dir = os.path.join(zone_output_dir, os.path.basename(os.path.dirname(network_base_dir_name)))
    if not os.path.isdir(zone_nw_output_dir):
        os.makedirs(zone_nw_output_dir)
    #
    node_to_zone_f = os.path.join(zone_nw_output_dir, "node_zone_info.csv")
    # drop_cols = [col for col in nodes_df.columns if (col.startswith("bin") or col.startswith("pos"))]
    # nodes_to_zones = nodes_df.drop(drop_cols, axis=1)
    # nodes_to_zones.to_csv(node_to_zone_f)
    all_nodes_df.drop(["is_stop_only", "pos_x", "pos_y"], axis=1, inplace=True)
    all_nodes_df.to_csv(node_to_zone_f)
    print(f"\t ... wrote {node_to_zone_f}")
    #
    print(zones_gdf)
    print(zones_gdf.describe())
    print(zones_gdf.columns)
    zones_gdf.to_csv("tmp.csv")
    zones_f = os.path.join(zone_output_dir, "polygon_definition.geojson")
    zones_gdf.to_file(zones_f[:-7] + "shp")
    zones_gdf.to_file(zones_f, driver="GeoJSON")
    # zones_gdf.to_file(zones_f, driver="GeoJSON", encoding="utf-8")
    print(f"\t ... wrote {zones_f}")


if __name__ == "__main__":
    if len(sys.argv) not in [5,6,7,8]:
        raise IOError("Incorrect number of input arguments!")
    create_quadratic_zones(*sys.argv[1:])

import os
from re import match
import sys
import numpy as np
import pandas as pd 
import geopandas as gpd 
import rtree
import matplotlib.pyplot as plt

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(MAIN_DIR)
sys.path.append(MAIN_DIR)

def match_nodes_to_centroid_zone_system(nw_name, zone_name, centroid_node_ids = []):
    """ this function matches all nodes of a network to the closest centroid in case the zone-system is defined
    by centroids points instead of polygons (e.g. aimsun)
    :param nw_name: folder name of the corresponding network
    :param zone_name: folder name of der corresponding centroid zone system 
    :param centroid_node_ids: network nodes that will be marked as centroids (possible reloc targets)"""
    zoning_path = os.path.join(MAIN_DIR, "data", "zones", zone_name)
    zoning_file = gpd.read_file(os.path.join(zoning_path, "centroid_definition.geojson"))

    nw_path = os.path.join(MAIN_DIR, "data", "networks", nw_name)
    nw_node_file = gpd.read_file(os.path.join(nw_path, "base", "nodes_all_infos.geojson"))

    r_tree = rtree.index.Index()
    for _, entry in zoning_file.iterrows():
        centroid_point = entry["geometry"]
        coords = (centroid_point.x, centroid_point.y)
        index = entry["zone_id"]
        r_tree.insert(index, (coords[0], coords[1], coords[0], coords[1]) )

    node_zone_id_list = {}
    cen_nodes = {}
    for i, entry in nw_node_file.iterrows():
        node_point = entry["geometry"]
        coords = (node_point.x, node_point.y)
        nearest_centroid =  list(r_tree.nearest((coords[0], coords[1], coords[0], coords[1]), 1))[0]
        node_zone_id_list[i] = nearest_centroid
        if entry["node_index"] in centroid_node_ids:
            cen_nodes[i] = True
        else:
            cen_nodes[i] = False

    nw_node_file["zone_id"] = pd.Series(node_zone_id_list)
    nw_node_file["is_centroid"] = pd.Series(cen_nodes)

    node_zone_info_df = pd.DataFrame([nw_node_file["node_index"], nw_node_file["zone_id"], nw_node_file["is_centroid"]]).transpose()
    out_folder = os.path.join(zoning_path, nw_name)
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    node_zone_info_df.to_csv(os.path.join(out_folder, "node_zone_info.csv"))

    ax = zoning_file.plot(color = "k", marker = "x")
    nw_node_file[nw_node_file["is_centroid"] == True].plot(ax = ax, column = "zone_id")
    plt.show()

def get_possible_centroid_nodes_from_partial_preprocessing(nw_name):
    """ this function returns a list of partially preprocessed nodes to used them as zone systems
    (for fast routing) """
    nw_path = os.path.join(MAIN_DIR, "data", "networks", nw_name)
    ppf = os.path.join(nw_path, "base", "tt_matrix.npy")
    if os.path.isfile(ppf):
        tt_matrx = np.load(ppf)
        return [i for i in range(tt_matrx.shape[0])]
    else:
        raise FileExistsError("file {} not found! not preprocessed?".format(ppf))

if __name__ == "__main__":
    zone_name = "Aimsun_Munich_Centroids"
    nw_name = "Aimsun_Munich_2020_04_15_Majid_reduced_ER_BP_all"
    cen_nodes = get_possible_centroid_nodes_from_partial_preprocessing(nw_name)
    match_nodes_to_centroid_zone_system(nw_name, zone_name, centroid_node_ids=cen_nodes)

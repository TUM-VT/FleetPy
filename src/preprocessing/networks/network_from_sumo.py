import csv
import xml.etree.ElementTree as ET
import operator
import pandas as pd
import numpy as np
import os
import sys
import geopandas as gpd
from shapely.geometry import Point, LineString
from tqdm import tqdm
import argparse


"""
This script converts network.net.xml files from SUMO into nodes.csv/edges.csv and nodes_all_infos.geojson/edges_all_infos.geojson for FleetPy.

Args:
    xmlfile (str): Path to SUMO network XML file (network.net.xml)
    -n, --network (str): Name of the network (Network is saved to FleetPy\data\networks\{Name})
    -a, --allowed-modes (str, optional): SUMO modes that FleetPy vehicles correspond to (e.g., "passenger,bus,taxi"). 
                                         FleetPy vehicles will only use lanes that allow at least one of these modes.
                                         Defaults to "all" (all SUMO modes allowed).

Usage:
    python network_from_sumo.py <xmlfile> -n <network_name> [-a <allowed_modes>]
    
Example:
    python network_from_sumo.py network.net.xml -n my_network -a "passenger,taxi"
"""

## For SUMO version 1.19 (05/2024):
ALL_MODES_SUMO = {"passenger","private","taxi","bus","coach","delivery","truck","trailer","emergency","motorcycle","moped",
                  "bicycle","pedestrian","tram","rail_electric","rail_fast","rail_urban","rail","evehicle","army","ship",
                  "authority","vip","hov","custom1","custom2"}


def check_lane_usability(lane_modes_allow=str,lane_modes_disallow=str,allowed_modes_FP=str,ALL_MODES_SUMO=set):
    """
    # Checks for a lane with defined modes that are allowed and not allowed if a Fleetpy-vehicle can use it
    # Returns TRUE if it is usable and FALSE if not
    """
    if lane_modes_allow == None and lane_modes_disallow == None:
        lane_modes_allow = ALL_MODES_SUMO
    elif lane_modes_allow == None and lane_modes_disallow != None:
        lane_modes_allow = ALL_MODES_SUMO - set(lane_modes_disallow)
    elif lane_modes_allow != None:
        lane_modes_allow = set(lane_modes_allow.split(" "))
    
    if len(lane_modes_allow.intersection(set(allowed_modes_FP))) > 0: 
        lane_usable_by_FP = True
    else:
        lane_usable_by_FP = False
    return lane_usable_by_FP

def create_nodes_and_edges_from_xml(xmlfile, allowed_modes="all"):
    """
    Approach is to parse the xml file of the sumo network for the relevant information
    """
    tree = ET.parse(xmlfile)
    root_node = tree.getroot()
    
    nodes = {}  # node_index -> node info
    node_source_id_to_node_indices = {} # node_source_id -> node_indices
    edges = {}  # edge_source_id -> edge info
    
    if allowed_modes == "all":
        allowed_modes_FP = ALL_MODES_SUMO
    else:
        allowed_modes_FP = set(allowed_modes.split(","))
    # Create start and end node for each sumo link
    node_index = 0
    for edge in tqdm(root_node.findall('edge'),desc="Create start and end node for each sumo link"):
        function = edge.get('function')
        ## First only non-internal edges
        if function != 'internal':
            usable_lane_count = 0
            for lane in edge.findall('lane'):
                lane_modes_allow = lane.get("allow")
                lane_modes_disallow = lane.get("disallow")    
                lane_usable_by_FP = check_lane_usability(lane_modes_allow,lane_modes_disallow,allowed_modes_FP,ALL_MODES_SUMO)
                if lane_usable_by_FP == True:
                    usable_lane_count += 1
                    min_travel_time = round(float(lane.get("length"))/float(lane.get("speed")),3) ## Travel time derived fom Max Speed of Lane
                    distance = round(float(lane.get("length")),2)
            
            if usable_lane_count > 0:
                new_edgeID = edge.get('id')
                # Start Node
                new_from_node = edge.get('from')
                nodes[f"S_{new_edgeID}"] = {"node_index" : node_index, "source_node_id" : new_from_node}
                try:
                    node_source_id_to_node_indices[new_from_node].append(f"S_{new_edgeID}")
                except KeyError:
                    node_source_id_to_node_indices[new_from_node] = [f"S_{new_edgeID}"]
                node_index += 1
                
                # End Node
                new_to_node = edge.get('to')
                nodes[f"E_{new_edgeID}"] = {"node_index" : node_index, "source_node_id" : new_to_node}
                try:
                    node_source_id_to_node_indices[new_to_node].append(f"E_{new_edgeID}")
                except KeyError:
                    node_source_id_to_node_indices[new_to_node] = [f"E_{new_edgeID}"]
                node_index += 1

                # Add FP-Information to SUMO-Network:
                for param in edge.findall("param"):
                    if param.get("key") == "fp_from_node" or param.get("key") == "fp_to_node":
                        edge.remove(param)
                                
                ET.SubElement(edge, "param", {"key": "fp_from_node", "value": str(node_index-2) })
                ET.SubElement(edge, "param", {"key": "fp_to_node", "value": str(node_index-1) })

                edges[new_edgeID] = {"from_node" : node_index-2, "to_node" : node_index-1, "source_edge_id" : new_edgeID, "number_of_usable_lanes":usable_lane_count,"travel_time":min_travel_time,"distance":distance}
    
                
    #Add Connecting edges (within intersections)
    for connection in tqdm(root_node.findall('connection'),desc="Add Connecting edges (within intersections)"):
        fromEdge = connection.get('from')
        toEdge = connection.get('to')
        if fromEdge not in edges.keys() or toEdge not in edges.keys() or ":" in fromEdge or ":" in toEdge:
            continue
        internalEdge_id = connection.get("via")
        if "_" in internalEdge_id:
            last_underscore_index = internalEdge_id.rfind('_') 
            if last_underscore_index != -1:
                internalEdge_id= internalEdge_id[:last_underscore_index]

        for edge_element in root_node.findall("edge"): #find internal edge element for id
            if edge_element.get("id")==internalEdge_id:
                break
        
        ## Get average speed and length of internal lanes
        internal_lanes_speeds = []
        internal_lanes_lengths = []
        internal_lanes_count = 0

        for internal_lane in edge_element.findall("lane"):

            internal_lane_usable_by_FP = check_lane_usability(internal_lane.get("allow"),internal_lane.get("disallow"),allowed_modes_FP,ALL_MODES_SUMO)
            # Check if internal edge can be used by FP-vehicles
            if internal_lane_usable_by_FP == False:
                continue
            else:
                internal_lanes_speeds.append(float(internal_lane.get("speed")))
                internal_lanes_lengths.append(float(internal_lane.get("length")))
                internal_lanes_count += 1
        
        if len(internal_lanes_speeds)>0 and len(internal_lanes_lengths)>0:
            internal_lane_speed_avg = np.mean(internal_lanes_speeds)
            internal_lanes_length_avg = np.mean(internal_lanes_lengths)
            min_travel_time = round(internal_lanes_length_avg/internal_lane_speed_avg,3)
            internal_lanes_length_avg = round(internal_lanes_length_avg,1)
        else:
            continue   
        edge_connection = f"i_{fromEdge}_{toEdge}"
        from_node_index = nodes[f"E_{fromEdge}"]["node_index"]
        to_node_index = nodes[f"S_{toEdge}"]["node_index"]
        edges[internalEdge_id] = {"from_node" : from_node_index, "to_node" : to_node_index, "distance" : internal_lanes_length_avg, "travel_time" : min_travel_time, "source_edge_id" : internalEdge_id, "number_of_usable_lanes":internal_lanes_count,"connecting_edges":edge_connection}
  

    #add node information (Coordinates)
    for node in tqdm(nodes,desc="Add node information (Coordinates)"):
        source_node_id = nodes[node]["source_node_id"]
        for junction in root_node.findall('junction'):
            if junction.get("id") == source_node_id:
                pos_x = junction.get('x')
                pos_y = junction.get('y')
                break

        nodes[node]["pos_x"] = float(pos_x)
        nodes[node]["pos_y"] = float(pos_y)
        nodes[node]["is_stop_only"] = False
          
    return nodes, edges, root_node
    
def store_network(nodes, edges, network_name,root_node,xmlfile):
    node_list = sorted(list(nodes.values()), key=lambda x:x["node_index"])
    edges_list = list(edges.values())
    node_df = pd.DataFrame(node_list)
    edges_df = pd.DataFrame(edges_list)

    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "networks")
    p = os.path.join(p, network_name)
    if not os.path.isdir(p):
        os.mkdir(p)
    p = os.path.join(p, "base")
    if not os.path.isdir(p):
        os.mkdir(p)
    node_df.to_csv(os.path.join(p, "nodes.csv"), index=False)
    print(f"File saved to {os.path.join(p, 'nodes.csv')}")
    edges_df.to_csv(os.path.join(p, "edges.csv"), index=False)
    print(f"File saved to {os.path.join(p, 'edges.csv')}")

    node_gdf_dict = {}
    for _, node in nodes.items():
        node["geometry"] = Point(node["pos_x"], node["pos_y"])
        node_gdf_dict[node["node_index"]] = node
    edge_gdf_dict = {}
    for edge in edges.values():
        edge["geometry"] = LineString([(node_gdf_dict[edge["from_node"]]["pos_x"], node_gdf_dict[edge["from_node"]]["pos_y"]), (node_gdf_dict[edge["to_node"]]["pos_x"], node_gdf_dict[edge["to_node"]]["pos_y"])])
        edge_gdf_dict[edge["source_edge_id"]] = edge
    node_gdf = gpd.GeoDataFrame(list(node_gdf_dict.values()))
    edge_gdf = gpd.GeoDataFrame(list(edge_gdf_dict.values()))
    node_gdf.to_file(os.path.join(p, "nodes_all_infos.geojson"), index=False, driver="GeoJSON")
    edge_gdf.to_file(os.path.join(p, "edges_all_infos.geojson"), index=False, driver="GeoJSON")
    
    with open(os.path.join(p, "crs.info"), "w") as f:
        f.write("unknowncrs")

    new_SUMO_net_tree = ET.ElementTree(root_node)
    new_SUMO_net_tree.write(os.path.join(p, "new_network.xml"))
    print(f"New XML-File saved to {os.path.join(p, 'new_network.xml')}")
    
def create_network(xmlfile, nw_name, allowed_modes):
    nodes, edges, root_node = create_nodes_and_edges_from_xml(xmlfile, allowed_modes)
    store_network(nodes, edges, nw_name,root_node,xmlfile)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert SUMO network to FleetPy format')
    parser.add_argument('xmlfile', help='Path to SUMO network XML file')
    parser.add_argument('-n', '--network', dest='nw_name', required=True, help='Name of the network')
    parser.add_argument('-a', '--allowed-modes', dest='allowed_modes', default='all', help='SUMO modes allowed (e.g., "passenger,bus,taxi")')
    
    args = parser.parse_args()
    
    create_network(args.xmlfile, args.nw_name, args.allowed_modes)

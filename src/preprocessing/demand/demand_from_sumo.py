import xml.etree.ElementTree as ET
import pandas as pd
import sys
import pathlib
import os
from tqdm import tqdm
import gzip

"""
This script transforms demand files from SUMO (.xml) into demand files for Fleetpy (.csv). 
As SUMO Trips are defined Edge to Edge and FleetPy Trips are defined Node to Node the starting Node from each SUMO Edge is taken as a starting Node.
Requires Network already translated from SUMO to Fleetpy.

Input Variables:
- xmlfile (str):     Path to SUMO-XML File/SUMO-XML.gz File
- demand_name (str): Name of the demand set 
- demand_type (str): Type of the Demand (trips/routes) 
- nw_name (str):     Name of the transformed Network in the FleetPy Repository

Output:
Saves .csv file into FleetPy\data\demand\{dmd_name}
   
"""

PY_PATH = pathlib.Path(__file__)
FLEETPY_PATH = PY_PATH.parent.parent.parent.parent


def transform_demand_SUMO_to_fp(xmlfile,demand_type,nw_name):
    if xmlfile.endswith('.gz'):
        # If the file is gzipped, open it with gzip
        with gzip.open(xmlfile, 'rb') as f:
            tree = ET.parse(f)
    elif xmlfile.endswith('.xml'):
        # If it's a normal XML file, open it normally
        tree = ET.parse(xmlfile)
    else:
        print(f"Wrong Filename: {xmlfile}. XML/GZ File required")
    
    demand_root = tree.getroot()
    nw_Path = FLEETPY_PATH /  "data" / "networks" / nw_name / "base"

    edges_df = pd.read_csv(nw_Path.joinpath("edges.csv"))
    request_ids = []
    rq_times = []
    start_nodes = []
    end_nodes = []
    orig_ids =[]

    if demand_type == "trips":
        for trip in tqdm(demand_root.findall("trip"), desc="Iterating over all Routes"):
            start_edge = trip.get("from")
            start_nodes.append(get_fp_node_from_SUMO_edge(start_edge,edges_df))
            end_edge = trip.get("to")
            end_nodes.append(get_fp_node_from_SUMO_edge(end_edge,edges_df))
            request_ids.append(int(trip.get('id')))
            rq_times.append(float(trip.get("depart")))
            orig_ids.append(trip.get('id'))
        
    elif demand_type == "routes":
        count = 0
        for vehicle in tqdm(demand_root.findall("vehicle"),desc="Iterating over all Routes"):
                route = vehicle.find("route")
                edges_list = route.get("edges").split(" ")
                start_nodes.append(get_fp_node_from_SUMO_edge(edges_list[0],edges_df))
                end_nodes.append(get_fp_node_from_SUMO_edge(edges_list[-1],edges_df))
                orig_ids.append(vehicle.get('id'))
                rq_times.append(float(vehicle.get("depart")))
                request_ids.append(count)
                count += 1

    fp_df = pd.DataFrame({"request_id":request_ids,"orig_id":orig_ids,"rq_time":rq_times,"start":start_nodes,"end":end_nodes})
    fp_df = fp_df.reset_index(drop=True)
    
    return fp_df

def save_demand_files(fp_df,demand_type,nw_name):
    if not os.path.exists(FLEETPY_PATH /"data" / "demand" / demand_name):
        os.makedirs(FLEETPY_PATH / "data" / "demand" / demand_name)
    if not os.path.exists(FLEETPY_PATH / "data" / "demand" / demand_name/ "matched" / nw_name):
        os.makedirs(FLEETPY_PATH / "data" / "demand" / demand_name/ "matched" / nw_name)
    
    demand_path = FLEETPY_PATH  / "data" / "demand" / demand_name / "matched" / nw_name/ f"{demand_name}.csv"
    fp_df.to_csv(demand_path)
    print(f"Demand File saved to: {demand_path}")

def get_fp_node_from_SUMO_edge(SUMO_EDGE,edges_df):
    filtered_df = edges_df[edges_df['source_edge_id'] == SUMO_EDGE]
    start_node = filtered_df["from_node"].values[0]
    return start_node


if __name__ == "__main__":
    if len(sys.argv) == 5:
        xmlfile = sys.argv[1]
        demand_name = sys.argv[2]
        demand_type = sys.argv[3]
        nw_name = sys.argv[4]
        fp_df = transform_demand_SUMO_to_fp(xmlfile,demand_type,nw_name)
        save_demand_files(fp_df,demand_type,nw_name)

    else:
        print("wrong call!")
        print("arguments of this script should be the path to the SUMO demand XML-file and the name of the created demand-set")
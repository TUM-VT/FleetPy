import xml.etree.ElementTree as ET
import os
import pandas as pd
import gzip

def create_fleetpy_network_from_matsim(matsim_network_path, fleetpy_data_path, network_name):
    """
    Create FleetPy network based on MATSim network.
    :param matsim_network_path: Path to the MATSim network file.
    :return: FleetPy network DataFrame.
    """
    # Load the MATSim network XML file
    print("matsim_network_path:", matsim_network_path)
    if matsim_network_path.endswith(".gz"):
        with gzip.open(matsim_network_path, 'rb') as f:
            tree = ET.parse(f)
    else:
        tree = ET.parse(matsim_network_path)
    root = tree.getroot()
    
    # Extract nodes
    nodes = []
    source_node_id_to_index = {}
    #node_index,is_stop_only,pos_x,pos_y
    for node in root.find("nodes").findall("node"):
        nodes.append({
            "node_index": len(nodes),
            "source_node_id": node.get("id"),
            "pos_x": float(node.get("x")),
            "pos_y": float(node.get("y")),
            "is_stop_only" : False
        })
        source_node_id_to_index[node.get("id")] = len(nodes) - 1
        
    # Convert nodes to a DataFrame
    nodes_df = pd.DataFrame(nodes)
        
    # Extract links
    links = []
    matsim_edge_to_fp_edge, fp_edge_to_matsim_edge = {}, {}
    #from_node,to_node,distance,travel_time,source_edge_id
    for link in root.find("links").findall("link"):
        links.append({
            "source_edge_id": link.get("id"),
            "from_node": source_node_id_to_index[link.get("from")],
            "to_node": source_node_id_to_index[link.get("to")],
            "distance": float(link.get("length")),
            "travel_time": float(link.get("freespeed"))/float(link.get("length")),
            "capacity": float(link.get("capacity")),
            "freespeed": float(link.get("freespeed")),
            "permlanes": float(link.get("permlanes"))
        })
        matsim_edge_to_fp_edge[int(links[-1]["source_edge_id"])] = (int(links[-1]["from_node"]), int(links[-1]["to_node"]))
        try:
            fp_edge_to_matsim_edge[int(links[-1]["from_node"])][int(links[-1]["to_node"])] = int(links[-1]["source_edge_id"])
        except KeyError:
            fp_edge_to_matsim_edge[int(links[-1]["from_node"])] = {}
            fp_edge_to_matsim_edge[int(links[-1]["from_node"])][int(links[-1]["to_node"])] = int(links[-1]["source_edge_id"])

    # Convert links to a DataFrame
    links_df = pd.DataFrame(links)
    
    output_path = os.path.join(fleetpy_data_path, "networks", network_name, "base")
    os.makedirs(output_path, exist_ok=True)
    nodes_df.to_csv(os.path.join(output_path, "nodes.csv"), index=False)
    links_df.to_csv(os.path.join(output_path, "edges.csv"), index=False)
    crs_str = "MATSIM CRS"
    with open(os.path.join(output_path, "crs.info"), "w") as f:
        f.write(crs_str)
    print("FleetPy network created from MATSim network: {}".format(output_path))
    
    return matsim_edge_to_fp_edge, fp_edge_to_matsim_edge

class MATSimStop:
    def __init__(self, id, x, y): # todo
        self.id = id
        self.x = x
        self.y = y
        
    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y
        }

if __name__ == "__main__":
    matsim_network_path = r"C:\Users\ge37ser\Documents\Projekte\VW_AD\Auswertung_Umfrage\vw-ad-java\scenarios\equil\network.xml"
    Fleetpy_data_path = r"C:\Users\ge37ser\Documents\Coding\FleetPy\data"
    network_name = "matsim_test"
    create_fleetpy_network_from_matsim(matsim_network_path, Fleetpy_data_path, network_name)
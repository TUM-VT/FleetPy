import os
import sys
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.preprocessing.networks.create_travel_time_tables import create_travel_time_table


NW_BASE_DIR = "base"
NODES_FILE = "nodes.csv"
EDGES_FILE = "edges.csv"
HUBS_FILE = "hubs.csv"
CRS_FILE = "crs.info"

NW_DIR = os.path.join(REPO_ROOT, "data", "networks")
INFRA_DIR = os.path.join(REPO_ROOT, "data", "infra")


def get_row_cols(length, width, cell_size):
    """
    Calculate the number of rows and columns for the grid network based on the specified parameters.

    Parameters:
    - length: Length of the corridor (in km)
    - width: Width of the corridor (in km)
    - cell_size: Size of each grid cell (in meters)
    """
    cols = int((length * 1000) / cell_size) + 1
    rows = int((width * 1000) / cell_size) + 1
    return rows, cols


def generate_nodes(rows, cols, cell_size):
    """
    Generate nodes for a grid network based on the specified parameters.

    Parameters:
    - rows: Number of rows in the grid
    - cols: Number of columns in the grid
    - cell_size: Size of each grid cell (in meters)
    """
    nodes = []
    for r in range(rows):
        for c in range(cols):
            node_index = r * cols + c
            nodes.append({
                "node_index": node_index,
                "is_stop_only": False,
                "pos_x": c * cell_size,
                "pos_y": r * cell_size
            })

    return nodes


def generate_edges(rows, cols, cell_size, speed):
    """
    Generate edges for a grid network based on the specified parameters.

    Parameters:
    - rows: Number of rows in the grid
    - cols: Number of columns in the grid
    - cell_size: Size of each grid cell (in meters)
    - speed: Speed of travel (in km/h)
    """
    edges = []

    eid = 0
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c
            for dr, dc in [(1, 0), (0, 1)]:  # down and right neighbors
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < rows and 0 <= c2 < cols:
                    nid2 = r2 * cols + c2
                    dist = float(cell_size)
                    speed_ms = speed * 1000 / 3600  # Convert km/h to m/s
                    tt = dist / speed_ms  # TODO check
                    edges += [
                        {"from_node": nid,  "to_node": nid2, "distance": dist,
                            "travel_time": tt, "source_edge_id": eid},
                        {"from_node": nid2, "to_node": nid,  "distance": dist,
                            "travel_time": tt, "source_edge_id": eid + 1},
                    ]
                    eid += 2

    return edges


def generate_hubs(rows, cols, n_hubs):
    """
    Generate hub-node assignments for a grid network.
    Hubs are placed on corridor edges in the middle row.

    Parameters:
    - rows: Number of rows in the grid
    - cols: Number of columns in the grid
    - n_hubs: Number of hubs in the network
    """
    if n_hubs < 1:
        return []
    if n_hubs > 2:
        raise ValueError("Currently, only 1 or 2 hubs are supported.")

    mid_row = rows // 2
    mid_row_first_cell = mid_row * cols

    print(
        f"Generating {n_hubs} hubs for a grid with {rows} rows and {cols} columns.")
    print(
        f"Middle row index: {mid_row}, first cell index in middle row: {mid_row_first_cell}")

    # Place a single hub at the rightmost midrow node; for 2 hubs, add the leftmost midrow node.
    hub_nodes = [{"node_index": mid_row_first_cell + (cols - 1), "hub_id": 0}]
    if n_hubs == 2:
        hub_nodes.append({"node_index": mid_row_first_cell, "hub_id": 1})

    return hub_nodes


def write_network_to_file(nodes, edges, hubs, output_dir):
    """
    Write the generated network to a file.

    Parameters:
    - nodes: List of nodes in the network
    - edges: List of edges in the network
    - hubs: List of hubs in the network
    - output_dir: Directory to save the generated network files
    """
    base_dir = os.path.join(output_dir, NW_BASE_DIR)
    os.makedirs(base_dir, exist_ok=True)

    pd.DataFrame(nodes).to_csv(os.path.join(
        base_dir, NODES_FILE), index=False)
    pd.DataFrame(edges).to_csv(os.path.join(
        base_dir, EDGES_FILE), index=False)
    pd.DataFrame(hubs).to_csv(os.path.join(base_dir, HUBS_FILE), index=False)
    with open(os.path.join(base_dir, CRS_FILE), "w") as f:
        f.write("EPSG:32632")


def generate_grid_network(length, width, n_hubs, cell_size, speed, name):
    """Generate one grid network and optional boarding-point infrastructures."""
    rows, cols = get_row_cols(length, width, cell_size)
    nodes = generate_nodes(rows, cols, cell_size)
    edges = generate_edges(rows, cols, cell_size, speed)
    hubs = generate_hubs(rows, cols, n_hubs)

    output_dir = os.path.join(NW_DIR, name)
    write_network_to_file(nodes, edges, hubs, output_dir)
    create_travel_time_table(output_dir)


def generate_networks(nw_ranges):
    """Generate all configured networks and return a list of dicts with name and corridor dimensions.

    Each entry: {"name": str, "length_km": float, "width_km": float}
    """
    lengths = nw_ranges["lengths"]
    widths = nw_ranges["widths"]
    num_hubs = nw_ranges["num_hubs"]
    cell_size = nw_ranges["cell_size"]
    speed = nw_ranges["default_speed"]

    networks = []
    for length in lengths:
        for width in widths:
            for n_hubs in num_hubs:
                nw_name = f"grid_l{length}_w{width}_hubs{n_hubs}_cell{cell_size}"
                generate_grid_network(length, width, n_hubs, cell_size, speed, nw_name)
                networks.append({"name": nw_name, "length_km": length, "width_km": width})

    return networks

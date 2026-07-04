import numpy as np
import os
import pandas as pd

from src.misc.globals import *
from utils.distributions import get_location_distribution, get_time_distribution

REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."))
DEMAND_DIR = os.path.join(REPO_ROOT, "data", "demand", "agimo_wp1", "matched")
INFRA_DIR = os.path.join(REPO_ROOT, "data", "infra")

FROM_HUB = "from_hub"
TO_HUB = "to_hub"
_NODE_IDS_BY_NETWORK = {}
_NODE_COORDS_BY_NETWORK = {}
SECONDS_PER_HOUR = 3600
DEFAULT_BOARDING_TIME = 30  # seconds


def generate_demand_scenario(nw_name, rq_name, areal_density_pax_km2h, corridor_length_km, corridor_width_km,
                             dir_pct, seed, spatial_dist, temporal_dist,
                             user_group_shares, user_group_params, end_time,
                             overwrite=True):
    """
    Generate a demand scenario CSV with per-request user-group constraint columns.

    Parameters:
    - nw_name: Name of the network for which to generate demand scenario
    - rq_name: Name of the demand scenario file
    - areal_density_pax_km2h: Demand density in pax/km²/h; total arrival rate Λ is derived as
      areal_density_pax_km2h × corridor_length_km × corridor_width_km
    - corridor_length_km: Corridor length in km
    - corridor_width_km: Corridor width in km
    - dir_pct: Fraction of trips directed toward a hub
    - seed: Random seed for reproducibility
    - spatial_dist: Spatial distribution of the demand
    - temporal_dist: Temporal distribution of the demand
    - user_group_shares: Dict mapping group name → share (must sum to 1.0)
    - user_group_params: Dict mapping group name → constraint/VoT parameter dict
    - end_time: Simulation end time (s)
    - overwrite: If False, skip writing if the file already exists
    """
    output_dir = os.path.join(DEMAND_DIR, nw_name)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, rq_name + ".csv")
    if not overwrite and os.path.exists(output_path):
        return

    rng = np.random.default_rng(seed)
    total_lambda_pax_h = areal_density_pax_km2h * corridor_length_km * corridor_width_km
    num_requests = int(round(total_lambda_pax_h * end_time / SECONDS_PER_HOUR))

    node_ids = get_node_ids_for_network(nw_name)
    node_coords = get_node_coordinates_for_network(nw_name)
    hubs = get_hubs_for_network(nw_name)
    location_distribution = get_location_distribution(spatial_dist, node_ids)
    time_distribution = get_time_distribution(temporal_dist, end_time)

    groups = list(user_group_shares.keys())
    probs = [user_group_shares[g] for g in groups]

    requests = []
    for _ in range(num_requests):
        rq_time = time_distribution.sample(rng)
        direction = FROM_HUB if rng.random() < dir_pct else TO_HUB

        if direction == FROM_HUB:
            end_loc = location_distribution.sample(rng)
            start_loc = nearest_hub(end_loc, hubs, node_coords)
        else:
            start_loc = location_distribution.sample(rng)
            end_loc = nearest_hub(start_loc, hubs, node_coords)

        group = rng.choice(groups, p=probs)
        gp = user_group_params[group]

        rq = {
            "rq_time": rq_time,
            "start": start_loc,
            "end": end_loc,
            "direction": direction,
            "user_group": group,
            G_AR_MAX_WT: gp[G_AR_MAX_WT],
            G_RQ_MRD: gp[G_RQ_MRD],
            G_WALKING_SPEED: gp[G_WALKING_SPEED],
            G_MAX_WALKING_DIST: gp[G_MAX_WALKING_DIST],
            G_MC_VOT: gp[G_MC_VOT],
            G_VOW_FACTOR: gp[G_VOW_FACTOR],
            G_V_WAIT_FACTOR: gp[G_V_WAIT_FACTOR],
            G_V_REL_FACTOR: gp.get(G_V_REL_FACTOR, 0.0)
        }
        requests.append(rq)

    write_demand_to_csv(requests, output_dir, rq_name, overwrite)


def nearest_hub(location, hubs, node_coords):
    """
    Determine the nearest hub for a given location.

    Parameters:
    - location: The location for which to find the nearest hub
    - hubs: List of hub locations
    - node_coords: Mapping from node ID to (x, y) coordinates

    Returns:
    - The nearest hub location
    """
    if not hubs:
        return None

    loc_x, loc_y = node_coords[int(location)]
    best_hub = hubs[0]
    hub_x, hub_y = node_coords[int(best_hub)]
    best_dist = (hub_x - loc_x) * (hub_x - loc_x) + \
        (hub_y - loc_y) * (hub_y - loc_y)

    for hub in hubs[1:]:
        hub_x, hub_y = node_coords[int(hub)]
        dist = (hub_x - loc_x) * (hub_x - loc_x) + \
            (hub_y - loc_y) * (hub_y - loc_y)
        if dist < best_dist:
            best_dist = dist
            best_hub = hub

    return best_hub


def write_demand_to_csv(requests, output_dir, rq_name, overwrite=False):
    """
    Write the generated demand requests to a CSV file.

    Parameters:
    - requests: List of demand requests
    - output_dir: Directory to save the CSV file
    - rq_name: Name of the request file
    - overwrite: If False, skip writing if the file already exists
    """
    output_path = os.path.join(output_dir, rq_name + ".csv")
    if not overwrite and os.path.exists(output_path):
        return
    df = pd.DataFrame(requests)
    df = df.sort_values("rq_time").reset_index(drop=True)
    df["request_id"] = df.index
    df.to_csv(output_path, index=False)


def get_node_ids_for_network(nw_name):
    """
    Retrieve the node IDs for a given network.

    Parameters:
    - nw_name: Name of the network

    Returns:
    - List of node IDs
    """
    node_ids = _NODE_IDS_BY_NETWORK.get(nw_name)
    if node_ids is None:
        nodes_f = os.path.join(
            REPO_ROOT, "data", "networks", nw_name, "base", "nodes.csv")
        nodes = np.loadtxt(nodes_f, delimiter=",",
                           skiprows=1, usecols=(0, 2, 3), dtype=float)
        nodes = np.atleast_2d(nodes)
        node_ids = nodes[:, 0].astype(int).tolist()
        _NODE_COORDS_BY_NETWORK[nw_name] = {
            int(node_id): (pos_x, pos_y)
            for node_id, pos_x, pos_y in nodes
        }
        _NODE_IDS_BY_NETWORK[nw_name] = node_ids
    return node_ids


def get_node_coordinates_for_network(nw_name):
    node_coords = _NODE_COORDS_BY_NETWORK.get(nw_name)
    if node_coords is None:
        get_node_ids_for_network(nw_name)
        node_coords = _NODE_COORDS_BY_NETWORK[nw_name]
    return node_coords


def get_hubs_for_network(nw_name):
    """
    Retrieve the hub IDs for a given network.

    Parameters:
    - nw_name: Name of the network

    Returns:
    - List of hub IDs
    """
    hubs_f = os.path.join(
        REPO_ROOT, "data", "networks", nw_name, "base", "hubs.csv")
    if not os.path.exists(hubs_f):
        return []
    hubs_df = pd.read_csv(hubs_f)
    return hubs_df["node_index"].tolist()

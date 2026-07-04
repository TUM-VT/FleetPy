import math
import sys

import numpy as np
import os
import pandas as pd
import pyproj

REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."))
DEMAND_DIR = os.path.join(REPO_ROOT, "data", "demand", "agimo_wp1", "matched")
INFRA_DIR = os.path.join(REPO_ROOT, "data", "infra")

# CRS all node coordinates are projected into for cartesian distance calculations
MEASUREMENT_CRS = "EPSG:32632"

_NODE_IDS_BY_NETWORK = {}
_NODE_COORDS_BY_NETWORK = {}
SECONDS_PER_HOUR = 3600
DEFAULT_BOARDING_TIME = 30  # seconds

sys.path.insert(0, REPO_ROOT)

from src.misc.globals import *
from utils.distributions import get_location_distribution, get_time_distribution
from utils.network_utils import BOARDING_INFRA_NAME


def generate_demand_scenario(nw_name, rq_name, areal_density_pax_km2h, corridor_length_km, corridor_width_km,
                             dir_pct, seed, spatial_dist, temporal_dist,
                             user_group_shares, user_group_params, end_time,
                             overwrite=True, boarding_points=None, boarding_match_radius=None):
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
    - boarding_points: Optional list of boarding-point node IDs. If given, the non-hub end of
      each request is additionally matched to a boarding point (column G_RQ_BOARDING_NODE), for
      the stop-based on-demand service. start/end are unaffected.
    - boarding_match_radius: Match radius (m) used for boarding-point matching, applied uniformly
      to all requests regardless of user group. Required if boarding_points is given.
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
        direction = G_DIR_FROM_HUB if rng.random() < dir_pct else G_DIR_TO_HUB

        if direction == G_DIR_FROM_HUB:
            non_hub_loc = end_loc = location_distribution.sample(rng)
            start_loc = hub_loc = nearest_hub(end_loc, hubs, node_coords)
        else:
            non_hub_loc = start_loc = location_distribution.sample(rng)
            end_loc = hub_loc = nearest_hub(start_loc, hubs, node_coords)

        group = rng.choice(groups, p=probs)
        gp = user_group_params[group]

        rq = {
            "rq_time": rq_time,
            "start": start_loc,
            "end": end_loc,
            G_RQ_DIRECTION: direction,
            "user_group": group,
            G_AR_MAX_WT: gp[G_AR_MAX_WT],
            G_RQ_MRD: gp[G_RQ_MRD],
            G_WALKING_SPEED: gp[G_WALKING_SPEED],
            G_MAX_WALKING_DIST: gp[G_MAX_WALKING_DIST],
            G_MC_VOT: gp[G_MC_VOT],
            G_VOW_FACTOR: gp[G_VOW_FACTOR],
            G_V_WAIT_FACTOR: gp[G_V_WAIT_FACTOR],
            G_MC_NO_OFFER_PENALTY: gp.get(G_MC_NO_OFFER_PENALTY, 0.0)
        }
        if boarding_points:
            rq[G_RQ_BOARDING_NODE] = match_boarding_point(
                non_hub_loc, boarding_points, node_coords, hub_loc, boarding_match_radius)
        requests.append(rq)

    write_demand_to_csv(requests, output_dir, rq_name, overwrite)


def _nearest_node(location, candidates, node_coords, tie_break_ref=None):
    """
    Find the nearest candidate node to a location by cartesian distance (node_coords are assumed
    already projected into a common metric CRS, see get_node_coordinates_for_network).

    Parameters:
    - location: The location for which to find the nearest candidate
    - candidates: List of candidate node IDs
    - node_coords: Mapping from node ID to (x, y) coordinates
    - tie_break_ref: Optional node ID. If more than one candidate is within floating-point
      tolerance of the minimum distance, the candidate closest to tie_break_ref is returned
      instead of just the first one encountered.

    Returns:
    - The nearest candidate node ID, or None if candidates is empty
    """
    if not candidates:
        return None

    loc_x, loc_y = node_coords[int(location)]
    dists = []
    for candidate in candidates:
        cand_x, cand_y = node_coords[int(candidate)]
        dist = (cand_x - loc_x) * (cand_x - loc_x) + (cand_y - loc_y) * (cand_y - loc_y)
        dists.append((candidate, dist))

    min_dist = min(dist for _, dist in dists)
    tied = [candidate for candidate, dist in dists if math.isclose(dist, min_dist, rel_tol=1e-9, abs_tol=1e-6)]

    if len(tied) == 1 or tie_break_ref is None:
        return tied[0]

    ref_x, ref_y = node_coords[int(tie_break_ref)]
    return min(tied, key=lambda candidate: (node_coords[int(candidate)][0] - ref_x) ** 2 +
                                            (node_coords[int(candidate)][1] - ref_y) ** 2)


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
    return _nearest_node(location, hubs, node_coords)


def match_boarding_point(location, boarding_points, node_coords, hub_loc, match_radius):
    """
    Determine the boarding point to match a (non-hub) location to: among all boarding points
    within match_radius of location, pick the one CLOSEST to the request's hub -- directly
    minimizing the vehicle-side hub<->boarding-point drive leg, subject to staying within the
    walking budget. match_radius is applied uniformly to every request regardless of user group
    (boarding-point matching is a property of the spatial demand process, not of behavioral
    user-group differentiation). Falls back to the closest boarding point to location if none
    are within range.

    Parameters:
    - location: The (non-hub) location for which to find a boarding point
    - boarding_points: List of boarding-point node IDs
    - node_coords: Mapping from node ID to (x, y) coordinates
    - hub_loc: The hub node ID this request is matched to
    - match_radius: Uniform match radius (m), independent of any user group's own
      max_walking_dist threshold

    Returns:
    - The matched boarding-point node ID, or None if boarding_points is empty
    """
    if not boarding_points:
        return None

    loc_x, loc_y = node_coords[int(location)]
    max_dist_sq = match_radius * match_radius
    in_range = [candidate for candidate in boarding_points
                if (node_coords[int(candidate)][0] - loc_x) ** 2 +
                   (node_coords[int(candidate)][1] - loc_y) ** 2 <= max_dist_sq]

    if not in_range:
        # nothing within range: fall back to the closest boarding point available. The
        # simulation-time walking-distance threshold check (StopBasedUserGroupRequest,
        # inherited from UserGroupRequest.choose_offer) is what ultimately decides whether such
        # an over-range match gets declined.
        return _nearest_node(location, boarding_points, node_coords)

    return _nearest_node(hub_loc, in_range, node_coords)


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


def get_network_crs(nw_name):
    """
    Retrieve the CRS a network's node coordinates are stored in (same file the routing engine
    itself reads, see NetworkBasic.py/NetworkTTMatrix.py).

    Parameters:
    - nw_name: Name of the network

    Returns:
    - CRS string (e.g. "EPSG:32632")
    """
    crs_f = os.path.join(REPO_ROOT, "data", "networks", nw_name, "base", "crs.info")
    with open(crs_f, "r") as f:
        return f.read()


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

        network_crs = get_network_crs(nw_name)
        project = pyproj.Transformer.from_crs(
            pyproj.CRS(network_crs), pyproj.CRS(MEASUREMENT_CRS), always_xy=True).transform
        proj_x, proj_y = project(nodes[:, 1], nodes[:, 2])
        _NODE_COORDS_BY_NETWORK[nw_name] = {
            int(node_id): (pos_x, pos_y)
            for node_id, pos_x, pos_y in zip(nodes[:, 0].astype(int), proj_x, proj_y)
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


def get_boarding_points_for_network(nw_name, infra_name=BOARDING_INFRA_NAME):
    """
    Retrieve the boarding-point node IDs for a given network.

    Parameters:
    - nw_name: Name of the network
    - infra_name: Name of the boarding-point infrastructure (see network_utils.BOARDING_INFRA_NAME)

    Returns:
    - List of boarding-point node IDs, or [] if none have been generated for this network
    """
    bp_f = os.path.join(INFRA_DIR, infra_name, nw_name, "boarding_points.csv")
    if not os.path.exists(bp_f):
        return []
    bp_df = pd.read_csv(bp_f)
    return bp_df["node_index"].tolist()

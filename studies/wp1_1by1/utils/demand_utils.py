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

# Minimum hub<->non-hub trip distance (m) for generated demand -- a generic "anyone would just
# walk this" cutoff, independent of any user group's own max_walking_dist tolerance. Trips shorter
# than this are resampled (see generate_demand_scenario) rather than left in the demand pool,
# where they'd otherwise inflate "declined" counts once a service can't produce a
# sub-walking-distance offer for a trip nobody would have actually requested a ride for.
MIN_TRIP_DISTANCE_M = 500

sys.path.insert(0, REPO_ROOT)

from src.misc.globals import *
from utils.distributions import (HUB_SCHEDULE, HUB_TRIANGULAR, HUB_TRIANGULAR_2D,
                                 get_location_distribution, get_time_distribution)
from utils.network_utils import BOARDING_INFRA_NAME


def generate_demand_scenario(nw_name, rq_name, areal_density_pax_km2h, corridor_length_km, corridor_width_km,
                             dir_pct, seed, spatial_dist, temporal_dist,
                             user_group_shares, user_group_params, end_time,
                             overwrite=True, boarding_points=None, boarding_match_radius=None,
                             headway_s=None, ramp_s=None, network_speed_kmh=None):
    """
    Generate a demand scenario CSV with per-request user-group constraint columns.

    Parameters:
    - nw_name: Name of the network for which to generate demand scenario
    - rq_name: Name of the demand scenario file
    - areal_density_pax_km2h: Demand density in pax/km²/h; total arrival rate Λ is derived as
      areal_density_pax_km2h × corridor_length_km × corridor_width_km
    - corridor_length_km: Corridor length in km
    - corridor_width_km: Corridor width in km
    - dir_pct: Fraction of trips directed away from a hub (from-hub trips: hub -> non-hub);
      the remaining 1 - dir_pct are to-hub trips (non-hub -> hub)
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
    - headway_s, ramp_s: Hub-timetable parameters used only when temporal_dist is "hub_schedule"
      (scheduled-event spacing and one-sided ramp width, s). Fall back to the code defaults when
      None; ignored for other temporal distributions.
    - network_speed_kmh: Network free-flow speed (km/h), used only when temporal_dist is
      "hub_schedule" to floor a to-hub request's offset at the minimum physically possible travel
      time to the hub (see generate_demand_scenario's request loop) -- an event closer than that is
      impossible to reach regardless of service quality, not just unlikely. Required (>0) for
      "hub_schedule"; ignored for other temporal distributions.
    """
    output_dir = os.path.join(DEMAND_DIR, nw_name)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, rq_name + ".csv")
    if not overwrite and os.path.exists(output_path):
        # report the per-hub demand split (needed for the multi-hub proportional fleet split if needed later)
        return _hub_counts_from_csv(output_path, get_hubs_for_network(nw_name))

    rng = np.random.default_rng(seed)
    total_lambda_pax_h = areal_density_pax_km2h * corridor_length_km * corridor_width_km
    expected_requests = total_lambda_pax_h * end_time / SECONDS_PER_HOUR

    node_ids = get_node_ids_for_network(nw_name)
    node_coords = get_node_coordinates_for_network(nw_name)
    hubs = get_hubs_for_network(nw_name)

    if spatial_dist in (HUB_TRIANGULAR, HUB_TRIANGULAR_2D):
        if not hubs:
            raise ValueError(f"spatial_dist '{spatial_dist}' requires at least one hub in the network.")
        # Along-corridor triangular cutoff as a fraction of corridor length, derived from the hub
        # count: a single hub ramps down over the whole corridor (1.0, zero at the far end); two hubs
        # use 0.5 so the two ramps meet at zero at the corridor midpoint.
        hub_scale_frac = 1.0 if len(hubs) == 1 else 0.5
        # Exclude hub nodes as candidates so the non-hub end is never sampled exactly on a hub
        # (which would degenerate to a zero-length trip once the other end snaps to nearest_hub).
        hub_set = set(hubs)
        candidate_nodes = [n for n in node_ids if n not in hub_set]
        # Along-corridor (x) distance to the nearest hub.
        nearest_hub_dist = [min(_corridor_axis_dist(node_coords, n, hub) for hub in hubs)
                            for n in candidate_nodes]
        scale_m = hub_scale_frac * corridor_length_km * 1000
        if spatial_dist == HUB_TRIANGULAR:
            # Triangular along the length only; uniform across the width (y).
            location_distribution = get_location_distribution(
                spatial_dist, candidate_nodes, nearest_hub_dist=nearest_hub_dist, scale_m=scale_m)
        else:
            # Also triangular across the width: cross-corridor (y) distance to the hub row, decaying
            # to zero at the width edges (scale = half-width; the hubs sit on the centre row).
            hub_row_y = node_coords[int(hubs[0])][1]
            row_dist = [abs(node_coords[int(n)][1] - hub_row_y) for n in candidate_nodes]
            scale_y = corridor_width_km * 1000 / 2
            location_distribution = get_location_distribution(
                spatial_dist, candidate_nodes, nearest_hub_dist=nearest_hub_dist, scale_m=scale_m,
                row_dist=row_dist, scale_y=scale_y)
    else:
        location_distribution = get_location_distribution(spatial_dist, node_ids)

    time_distribution = get_time_distribution(temporal_dist, end_time, headway_s=headway_s, ramp_s=ramp_s)

    # The temporal model owns the count: both "uniform" and "hub_schedule" fix it to exactly
    # round(expected_requests) -- no run-to-run count variation, only arrival timing is randomized.
    num_requests = time_distribution.sample_count(expected_requests, rng)

    groups = list(user_group_shares.keys())
    probs = [user_group_shares[g] for g in groups]

    speed_ms = network_speed_kmh * 1000 / 3600 if network_speed_kmh else None

    hub_counts = {int(h): 0 for h in hubs}
    requests = []
    skipped_too_short = 0
    for _ in range(num_requests):
        # Direction is drawn first because schedule-anchored timing (HUB_SCHEDULE) depends on it:
        # to-hub requests cluster before a scheduled departure, from-hub after a scheduled arrival.
        # Direction-agnostic distributions (poisson) ignore the argument.
        direction = G_DIR_FROM_HUB if rng.random() < dir_pct else G_DIR_TO_HUB

        # Location is sampled before rq_time (HUB_SCHEDULE's offset floor below needs the
        # resulting distance-to-hub). Reject-and-resample the non-hub end if the trip is short
        # enough that the traveler would just walk the whole thing directly rather than ever
        # requesting a ride -- otherwise these trivially-short trips sit in the demand pool,
        # inflate the "declined" count once any service can't produce a sub-walking-distance
        # offer, and distort served_pct downward for a failure mode that was never a real one.
        # Threshold is a fixed constant (MIN_TRIP_DISTANCE_M), deliberately NOT the per-group
        # max_walking_dist -- this is a generic "anyone would just walk this" cutoff, not a
        # behavioral group-specific tolerance. Distance is measured to the HUB (not a specific
        # service's boarding point/station), since the demand file is shared across all service
        # types and the hub is the one endpoint common to all of them.
        for _attempt in range(200):
            if direction == G_DIR_FROM_HUB:
                non_hub_loc = end_loc = location_distribution.sample(rng)
                start_loc = hub_loc = nearest_hub(end_loc, hubs, node_coords)
            else:
                non_hub_loc = start_loc = location_distribution.sample(rng)
                end_loc = hub_loc = nearest_hub(start_loc, hubs, node_coords)
            hub_dist_m = _manhattan_dist(node_coords, non_hub_loc, hub_loc)
            if hub_dist_m >= MIN_TRIP_DISTANCE_M:
                break
        else:
            # couldn't find a far-enough candidate in 200 tries (e.g. a tiny network) -- keep the
            # last (too-short) draw rather than silently under-counting total demand, but track it
            skipped_too_short += 1

        # Floor a to-hub request's offset at the minimum physically possible travel time to the
        # hub (pure network distance / free-flow speed, no wait/detour/dwell) -- an event closer
        # than that isn't just unlikely to be reached, it's impossible regardless of service
        # quality, so HUB_SCHEDULE should never generate a request implying otherwise. from-hub
        # requests get no floor: that traveler is already at the hub when the event fires.
        min_offset = hub_dist_m / speed_ms if (direction == G_DIR_TO_HUB and speed_ms) else 0.0
        rq_time = time_distribution.sample(rng, direction, min_offset=min_offset)

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
            G_RQ_ACDT: gp.get(G_RQ_ACDT, 0.0),
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
        if int(hub_loc) in hub_counts:
            hub_counts[int(hub_loc)] += 1
        requests.append(rq)

    if skipped_too_short:
        print(f"WARNING: {rq_name}: {skipped_too_short}/{num_requests} requests kept a "
              f"shorter-than-max_walking_dist trip after 200 resample attempts each -- "
              f"network may be too small/dense to guarantee walk-worthy trips at this density.")

    write_demand_to_csv(requests, output_dir, rq_name, overwrite)
    return hub_counts


def _hub_counts_from_csv(output_path, hubs):
    """Count how many requests are snapped to each hub in an already-written demand CSV. Used to
    recover the per-hub demand split (for the multi-hub proportional fleet split) without
    regenerating the demand."""
    hub_set = {int(h) for h in hubs}
    hub_counts = {int(h): 0 for h in hubs}
    df = pd.read_csv(output_path)
    for start, end in zip(df["start"], df["end"]):
        # the trip's hub endpoint is whichever of start/end is a hub node
        for node in (int(start), int(end)):
            if node in hub_set:
                hub_counts[node] += 1
                break
    return hub_counts


def _manhattan_dist(node_coords, a, b):
    """Manhattan (L1) distance between two node IDs' projected coordinates."""
    ax, ay = node_coords[int(a)]
    bx, by = node_coords[int(b)]
    return abs(ax - bx) + abs(ay - by)


def _corridor_axis_dist(node_coords, a, b):
    """Distance along the corridor axis (x) between two node IDs' projected coordinates. The grid is
    laid out with the corridor length along x and the width along y (see network_utils.generate_nodes,
    with hubs placed at the mid-row corridor ends), so the x-difference alone measures displacement
    along the corridor, ignoring the cross-corridor (width) offset."""
    ax, _ = node_coords[int(a)]
    bx, _ = node_coords[int(b)]
    return abs(ax - bx)


def _nearest_node(location, candidates, node_coords, tie_break_ref=None):
    """
    Find the nearest candidate node to a location by Manhattan distance (node_coords are assumed
    already projected into a common metric CRS, see get_node_coordinates_for_network). Manhattan
    distance is used because the grid network only has axis-aligned edges (see
    network_utils.generate_edges), so it matches actual network/walking distance, unlike
    straight-line Euclidean distance which underestimates it whenever two nodes aren't aligned on
    the same row or column.

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

    dists = [(candidate, _manhattan_dist(node_coords, location, candidate)) for candidate in candidates]

    min_dist = min(dist for _, dist in dists)
    tied = [candidate for candidate, dist in dists if math.isclose(dist, min_dist, rel_tol=1e-9, abs_tol=1e-6)]

    if len(tied) == 1 or tie_break_ref is None:
        return tied[0]

    return min(tied, key=lambda candidate: _manhattan_dist(node_coords, candidate, tie_break_ref))


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

    in_range = [candidate for candidate in boarding_points
                if _manhattan_dist(node_coords, location, candidate) <= match_radius]

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


# --- demand scenario sweep (across the range grid) ---

def _demand_entry(nw_name, length_km, width_km, areal_density, spatial_dist, temporal_dist, profile_name, shares, direction_pct, seed, group_params, end_time, boarding_match_radius, headway_s=None, ramp_s=None, network_speed_kmh=None):
    # HUB_SCHEDULE's headway_s/ramp_s aren't otherwise reflected anywhere in the demand
    # filename -- two ranges files both using temporal_dist="hub_schedule" but different
    # headway_s/ramp_s would generate the IDENTICAL rq_name and silently overwrite (or,
    # combined into one scenario_cfg.csv, collide as duplicate scenario_names) each other's
    # demand CSV. Fold the values into the label so distinct hub_schedule configs are
    # distinct files; every other temporal_dist (headway_s/ramp_s both None) is unaffected.
    temporal_label = temporal_dist
    if temporal_dist == HUB_SCHEDULE and headway_s is not None and ramp_s is not None:
        temporal_label = f"{temporal_dist}_hw{int(headway_s)}_ramp{int(ramp_s)}"
    rq_name = f"{areal_density}pkm2h_dir{direction_pct}_seed{seed}_spatial_{spatial_dist}_temporal_{temporal_label}_user_{profile_name}"
    total_lambda = areal_density * length_km * width_km
    boarding_points = get_boarding_points_for_network(nw_name)
    hub_counts = generate_demand_scenario(
        nw_name, rq_name, areal_density, length_km, width_km, direction_pct, seed,
        spatial_dist, temporal_dist, shares, group_params, end_time,
        boarding_points=boarding_points, boarding_match_radius=boarding_match_radius,
        headway_s=headway_s, ramp_s=ramp_s, network_speed_kmh=network_speed_kmh)
    return {
        "network_name": nw_name,
        "rq_file": rq_name + ".csv",
        "areal_density": areal_density,
        "total_lambda": total_lambda,
        "spatial_distribution": spatial_dist,
        "temporal_distribution": temporal_dist,
        "user_profile": profile_name,
        "directionality": direction_pct,
        "seed": seed,
        # per-hub request counts, used for the multi-hub proportional fleet split
        "hub_counts": hub_counts or {},
    }


def generate_demand_scenarios(demand_ranges, networks, end_time, boarding_match_radius, network_speed_kmh=None):
    """Generate demand CSVs for all network/density combinations.

    Parameters:
    - networks: list of dicts {"name": str, "length_km": float, "width_km": float}
                as returned by generate_networks()
    - boarding_match_radius: uniform boarding-point match radius (m), see
      demand_utils.match_boarding_point
    - network_speed_kmh: network free-flow speed (km/h), passed through to
      generate_demand_scenario for the "hub_schedule" to-hub offset floor. A single global value
      (matching generate_networks/generate_pubtrans's own "not swept per-network" assumption), not
      per-network.
    """
    areal_densities = demand_ranges["areal_densities"]
    seeds = demand_ranges["seeds"]
    spatial_distributions = demand_ranges["spatial_distributions"]
    temporal_distributions = demand_ranges["temporal_distributions"]
    user_profiles = demand_ranges["user_profiles"]
    group_params = demand_ranges["user_group_params"]
    # Hub-timetable parameters for the "hub_schedule" temporal distribution (ignored by others);
    # fall back to the code defaults in get_time_distribution when absent.
    hub_schedule_cfg = demand_ranges.get("hub_schedule", {})
    headway_s = hub_schedule_cfg.get("headway_s")
    ramp_s = hub_schedule_cfg.get("ramp_s")

    demand_scenarios = []
    for nw in networks:
        for areal_density in areal_densities:
            for spatial_dist in spatial_distributions:
                for temporal_dist in temporal_distributions:
                    for profile_name, profile_cfg in user_profiles.items():
                        shares = {k: v for k, v in profile_cfg.items() if k != "directionality"}
                        for direction_pct in profile_cfg["directionality"]:
                            for seed in seeds:
                                demand_scenarios.append(_demand_entry(
                                    nw["name"], nw["length_km"], nw["width_km"],
                                    areal_density, spatial_dist, temporal_dist,
                                    profile_name, shares, direction_pct, seed,
                                    group_params, end_time, boarding_match_radius,
                                    headway_s=headway_s, ramp_s=ramp_s,
                                    network_speed_kmh=network_speed_kmh))
    return demand_scenarios

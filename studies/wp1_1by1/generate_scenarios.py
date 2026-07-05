from utils.demand_utils import generate_demand_scenario, get_hubs_for_network, get_boarding_points_for_network
from utils.network_utils import generate_networks
from utils.pubtrans_utils import generate_pubtrans
import yaml
import os
import sys
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)


STUDY_DIR = os.path.dirname(__file__)
SCENARIO_RANGES_FILE = os.path.join(STUDY_DIR, "scenario_ranges.yaml")
DEMAND_DIR = os.path.join(REPO_ROOT, "data", "demand", "agimo_wp1", "matched")
INIT_VEH_DIST_DIR = os.path.join(REPO_ROOT, "data", "fleetctrl", "initial_vehicle_distribution")
INIT_DIST_FILE_NAME = "hub_all.csv"

DEFAULT_RQ_TYPE = "UserGroupRequest"
PT_FIXED_LENGTH_SENTINEL_KM = 999999


def read_ranges():
    with open(SCENARIO_RANGES_FILE) as f:
        return yaml.safe_load(f)


# --- demand generation helpers ---

def _demand_entry(nw_name, length_km, width_km, areal_density, spatial_dist, temporal_dist, profile_name, shares, direction_pct, seed, group_params, end_time, boarding_match_radius):
    rq_name = f"{areal_density}pkm2h_dir{direction_pct}_seed{seed}_spatial_{spatial_dist}_temporal_{temporal_dist}_user_{profile_name}"
    total_lambda = areal_density * length_km * width_km
    boarding_points = get_boarding_points_for_network(nw_name)
    generate_demand_scenario(
        nw_name, rq_name, areal_density, length_km, width_km, direction_pct, seed,
        spatial_dist, temporal_dist, shares, group_params, end_time,
        boarding_points=boarding_points, boarding_match_radius=boarding_match_radius)
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
    }


def generate_demand_scenarios(demand_ranges, networks, end_time, boarding_match_radius):
    """Generate demand CSVs for all network/density combinations.

    Parameters:
    - networks: list of dicts {"name": str, "length_km": float, "width_km": float}
                as returned by generate_networks()
    - boarding_match_radius: uniform boarding-point match radius (m), see
      demand_utils.match_boarding_point
    """
    areal_densities = demand_ranges["areal_densities"]
    seeds = demand_ranges["seeds"]
    spatial_distributions = demand_ranges["spatial_distributions"]
    temporal_distributions = demand_ranges["temporal_distributions"]
    user_profiles = demand_ranges["user_profiles"]
    group_params = demand_ranges["user_group_params"]

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
                                    group_params, end_time, boarding_match_radius))
    return demand_scenarios


# --- scenario config helpers ---

def _fleet_entries(st_cfg, total_lambda):
    entries = [(n, f"n{n}") for n in st_cfg.get("fleet_sizes", [])]
    entries += [(max(1, round(r * total_lambda)), f"r{r}") for r in st_cfg.get("fleet_ratios", [])]
    return entries


def _scenario_row(scenario_name, dv, st_cfg, sim_end_time, n, size_tag, extra_cols=None):
    row = {
        "scenario_name": scenario_name,
        "network_name": dv["network_name"],
        "rq_file": dv["rq_file"],
        "end_time": sim_end_time,
        "op_module": st_cfg["op_module"],
        "op_vr_control_func_dict": yaml.dump(st_cfg["op_vr_control_func_dict"], default_flow_style=True).strip(),
        "op_fleet_composition": f"{st_cfg['veh_type']}:{n}",
        "op_init_veh_distribution": INIT_DIST_FILE_NAME,
    }
    row["rq_type"] = st_cfg.get("rq_type", DEFAULT_RQ_TYPE)
    if st_cfg.get("use_all_nodes_boarding"):
        row["op_use_all_nodes_boarding"] = True
    if "op_repo_method" in st_cfg:
        row["op_repo_method"] = st_cfg["op_repo_method"]
        row["op_repo_timestep"] = st_cfg.get("op_repo_timestep", 60)
    if extra_cols:
        row.update(extra_cols)
    return row


def _is_pt_line_service(st_cfg):
    return st_cfg.get("op_module") == "SemiOnDemandBatchAssignmentFleetcontrol"


def _pt_variant_lookup(pt_variants):
    return {(v["network_name"], v["station_spacing_m"], v["headway_min"]): v for v in pt_variants}


def _pt_extra_cols(st_cfg, pt_variant, headway_min, fixed_length_km, n_veh):
    return {
        "gtfs_name": pt_variant["pt_name"],
        "station_file": "stations.csv",
        "schedule_file": "schedules.csv",
        "alignment_file": "{line_id}_line_alignment.geojson",
        "terminus_id": pt_variant["terminus_station_id"],
        "line_id": st_cfg["line_id"],
        "pt_route_id": st_cfg["line_id"],
        "pt_regular_headway": headway_min * 60,
        "pt_fixed_length": fixed_length_km,
        "pt_flex_detour": st_cfg["pt_flex_detour"],
        "pt_zone_min_detour_time": st_cfg["pt_zone_min_detour_time"],
        "pt_zone_max_detour_time": st_cfg["pt_zone_max_detour_time"],
        "pt_dispatch_delay": st_cfg["pt_dispatch_delay"],
        "pt_n_veh": n_veh,
        
        "walking_speed": 4,
        
        "op_max_wait_time": 900,
        "op_max_wait_time_2": 1800,
        "op_max_detour_time_factor": 150,
        "op_add_constant_detour_time": 300,  # TODO double check
        # must be >= op_max_wait_time_2, or the traveler model auto-cancels (leaves_system) before
        # the retry mechanism gets a chance to match the request against the next dispatch
        "user_max_decision_time": 1800,
    }


def _pt_scenario_rows(base_name, dv, st_name, st_cfg, sim_end_time, pt_variant_lookup):
    """Build scenario rows for a PT-line service type (sod/fixed_line), sweeping station spacing,
    headway, fleet size, and (for sod only) the fixed-route/flexible split of the corridor."""
    rows = []
    is_fixed_line = st_cfg.get("fixed_line", False)

    for station_spacing_m in st_cfg["station_spacings_m"]:
        for headway_min in st_cfg["headways_min"]:
            key = (dv["network_name"], station_spacing_m, headway_min)
            pt_variant = pt_variant_lookup.get(key)
            if pt_variant is None:
                raise KeyError("No PT variant generated.")

            if is_fixed_line:
                fixed_length_variants = [(PT_FIXED_LENGTH_SENTINEL_KM, "full")]
            else:
                fixed_length_variants = [
                    (frac * pt_variant["route_length_km"], f"fl{frac}")
                    for frac in st_cfg.get("fixed_length_fractions", [0])
                ]
                # Stations are spaced out from the hub in exact station_spacing_m increments, so
                # the first non-hub station sits at station_spacing_m from the hub. If the fixed
                # zone doesn't reach that far, find_closest_station_to_x resolves the fixed/flex
                # boundary to the hub itself -- the "fixed route" segment silently degenerates to
                # zero length instead of covering the intended near-hub stations.
                for fixed_length_km, fl_tag in fixed_length_variants:
                    if fixed_length_km * 1000 < station_spacing_m:
                        raise ValueError(
                            f"{st_name}/{fl_tag} at sp{station_spacing_m}: fixed_length="
                            f"{fixed_length_km * 1000:.0f}m is shorter than the first non-hub "
                            f"station ({station_spacing_m}m from hub) -- the fixed-route segment "
                            f"would degenerate to just the hub. Raise this fixed_length_fraction "
                            f"or reduce station_spacing_m."
                        )

            for n, size_tag in _fleet_entries(st_cfg, dv["total_lambda"]):
                for fixed_length_km, fl_tag in fixed_length_variants:
                    scenario_name = (
                        f"{base_name}_{st_name}_sp{station_spacing_m}_hw{headway_min}_"
                        f"{fl_tag}_{size_tag}"
                    )
                    extra_cols = _pt_extra_cols(st_cfg, pt_variant, headway_min, fixed_length_km, n)
                    rows.append(_scenario_row(
                        scenario_name, dv, st_cfg, sim_end_time, n, size_tag, extra_cols=extra_cols))
    return rows


def generate_scenario_cfg(demand_scenarios, service_types, sim_end_time, pt_variants):
    """Generate scenario configuration CSV for all demand scenarios and service types."""
    scenarios_dir = os.path.join(STUDY_DIR, "scenarios")
    os.makedirs(scenarios_dir, exist_ok=True)

    pt_variant_lookup = _pt_variant_lookup(pt_variants)

    rows = []
    for dv in demand_scenarios:
        base_name = (
            f"{dv['network_name']}_{dv['areal_density']}pkm2h_{dv['directionality']}dir"
            f"_{dv['user_profile']}_seed{dv['seed']}"
        )
        for st_name, st_cfg in service_types.items():
            if _is_pt_line_service(st_cfg):
                rows.extend(_pt_scenario_rows(
                    base_name, dv, st_name, st_cfg, sim_end_time, pt_variant_lookup))
            else:
                for n, size_tag in _fleet_entries(st_cfg, dv["total_lambda"]):
                    rows.append(_scenario_row(
                        f"{base_name}_{st_name}_{size_tag}", dv, st_cfg, sim_end_time, n, size_tag))

    out_path = os.path.join(scenarios_dir, "scenario_cfg.csv")
    df = pd.DataFrame(rows)
    for col in ("terminus_id", "line_id", "pt_route_id", "pt_n_veh"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} scenarios to {out_path}")


def generate_initial_vehicle_distributions(nw_names):
    """Generate per-network init-distribution CSVs that place vehicles at hub nodes."""
    for nw_name in nw_names:
        hubs = get_hubs_for_network(nw_name)
        if not hubs:
            raise ValueError(f"Network {nw_name} has no hubs; cannot place vehicles at hub nodes.")
        prob = 1.0 / len(hubs)
        out_dir = os.path.join(INIT_VEH_DIST_DIR, nw_name)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([{"node_index": int(h), "probability": prob} for h in hubs]).to_csv(
            os.path.join(out_dir, INIT_DIST_FILE_NAME), index=False)


def generate_service_types(ranges):
    return ranges.get("service_types", {})


def main():
    ranges = read_ranges()
    sim_end_time = ranges["simulation"]["end_time"] + ranges["simulation"]["cool_time"]
    networks = generate_networks(ranges["network"])
    generate_initial_vehicle_distributions([nw["name"] for nw in networks])
    pt_variants = generate_pubtrans(ranges)
    boarding_match_radius = ranges["network"].get("boarding_point_spacing_m")
    demand_scenarios = generate_demand_scenarios(ranges["demand"], networks, sim_end_time, boarding_match_radius)
    service_types = generate_service_types(ranges)
    generate_scenario_cfg(demand_scenarios, service_types, sim_end_time, pt_variants)


if __name__ == "__main__":
    main()

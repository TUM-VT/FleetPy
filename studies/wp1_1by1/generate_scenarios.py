from utils.demand_utils import generate_demand_scenario, get_hubs_for_network
from utils.network_utils import generate_networks
# from utils.pubtrans_utils import generate_pubtrans, get_stations_for_network
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


def read_ranges():
    with open(SCENARIO_RANGES_FILE) as f:
        return yaml.safe_load(f)


# --- demand generation helpers ---

def _demand_entry(nw_name, length_km, width_km, areal_density, spatial_dist, temporal_dist, profile_name, shares, direction_pct, seed, group_params, end_time):
    rq_name = f"{areal_density}pkm2h_dir{direction_pct}_seed{seed}_spatial_{spatial_dist}_temporal_{temporal_dist}_user_{profile_name}"
    total_lambda = areal_density * length_km * width_km
    generate_demand_scenario(
        nw_name, rq_name, areal_density, length_km, width_km, direction_pct, seed,
        spatial_dist, temporal_dist, shares, group_params, end_time)
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


def generate_demand_scenarios(demand_ranges, networks, end_time):
    """Generate demand CSVs for all network/density combinations.

    Parameters:
    - networks: list of dicts {"name": str, "length_km": float, "width_km": float}
                as returned by generate_networks()
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
                                    group_params, end_time))
    return demand_scenarios


# --- scenario config helpers ---

def _fleet_entries(st_cfg, total_lambda):
    entries = [(n, f"n{n}") for n in st_cfg.get("fleet_sizes", [])]
    entries += [(max(1, round(r * total_lambda)), f"r{r}") for r in st_cfg.get("fleet_ratios", [])]
    return entries


def _scenario_row(base_name, dv, st_name, st_cfg, sim_end_time, n, size_tag):
    row = {
        "scenario_name": f"{base_name}_{st_name}_{size_tag}",
        "network_name": dv["network_name"],
        "rq_file": dv["rq_file"],
        "end_time": sim_end_time,
        "op_module": st_cfg["op_module"],
        "op_vr_control_func_dict": yaml.dump(st_cfg["op_vr_control_func_dict"], default_flow_style=True).strip(),
        "op_fleet_composition": f"{st_cfg['veh_type']}:{n}",
        "op_init_veh_distribution": INIT_DIST_FILE_NAME,
    }
    if "rq_type" in st_cfg:
        row["rq_type"] = st_cfg["rq_type"]
    if st_cfg.get("use_all_nodes_boarding"):
        row["op_use_all_nodes_boarding"] = True
    if "op_repo_method" in st_cfg:
        row["op_repo_method"] = st_cfg["op_repo_method"]
        row["op_repo_timestep"] = st_cfg.get("op_repo_timestep", 60)
    return row


def generate_scenario_cfg(demand_scenarios, service_types, sim_end_time):
    """Generate scenario configuration CSV for all demand scenarios and service types."""
    scenarios_dir = os.path.join(STUDY_DIR, "scenarios")
    os.makedirs(scenarios_dir, exist_ok=True)

    rows = []
    for dv in demand_scenarios:
        base_name = (
            f"{dv['network_name']}_{dv['areal_density']}pkm2h_{dv['directionality']}dir"
            f"_{dv['user_profile']}_seed{dv['seed']}"
        )
        for st_name, st_cfg in service_types.items():
            for n, size_tag in _fleet_entries(st_cfg, dv["total_lambda"]):
                rows.append(_scenario_row(
                    base_name, dv, st_name, st_cfg, sim_end_time,
                    n, size_tag))

    out_path = os.path.join(scenarios_dir, "scenario_cfg.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
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
    # TODO public transport services
    return ranges.get("service_types", {})


def main():
    ranges = read_ranges()
    sim_end_time = ranges["simulation"]["end_time"] + ranges["simulation"]["cool_time"]
    networks = generate_networks(ranges["network"])
    generate_initial_vehicle_distributions([nw["name"] for nw in networks])
    demand_scenarios = generate_demand_scenarios(ranges["demand"], networks, sim_end_time)
    service_types = generate_service_types(ranges)
    generate_scenario_cfg(demand_scenarios, service_types, sim_end_time)


if __name__ == "__main__":
    main()

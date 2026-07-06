"""Supply-side scenario helpers: fleet sizing, the per-scenario config row, and the initial
vehicle distribution files."""
import os

import pandas as pd
import yaml

from utils.demand_utils import get_hubs_for_network

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INIT_VEH_DIST_DIR = os.path.join(REPO_ROOT, "data", "fleetctrl", "initial_vehicle_distribution")
INIT_DIST_FILE_NAME = "hub_all.csv"

DEFAULT_RQ_TYPE = "UserGroupRequest"


def fleet_entries(st_cfg, total_lambda):
    """Fleet-size sweep for a service type: absolute sizes plus demand-proportional ratios.
    Returns a list of (n_vehicles, size_tag) tuples."""
    entries = [(n, f"n{n}") for n in st_cfg.get("fleet_sizes", [])]
    entries += [(max(1, round(r * total_lambda)), f"r{r}") for r in st_cfg.get("fleet_ratios", [])]
    return entries


def scenario_row(scenario_name, dv, st_cfg, sim_end_time, n, size_tag, extra_cols=None):
    """Assemble a single scenario_cfg.csv row from a demand scenario (dv) and service-type config."""
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

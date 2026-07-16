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
    min_fleet_size (default 1) floors the ratio-derived fleet -- at low demand,
    round(fleet_ratio * total_lambda) can floor-round to 1 vehicle across a wide density range
    and then jump to 2 all at once, which reads as a quality cliff rather than a real density
    effect; raising the floor smooths that out at the cost of a relatively larger fleet at the
    lowest densities swept.
    Returns a list of (n_vehicles, size_tag) tuples."""
    min_fleet_size = st_cfg.get("min_fleet_size", 1)
    entries = [(n, f"n{n}") for n in st_cfg.get("fleet_sizes", [])]
    entries += [(max(min_fleet_size, round(r * total_lambda)), f"r{r}") for r in st_cfg.get("fleet_ratios", [])]
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
        # Same operator-side matching-feasibility defaults used by PT-line services (see
        # pubtrans_utils._pt_extra_cols) -- unified here so dtd/stops/stops_8 get the same
        # search slack instead of silently falling back to const_cfg.yaml's much tighter
        # baseline (op_max_wait_time=600, op_max_detour_time_factor=40, no retry), which was
        # never deliberately tuned and matches the demand-side gate exactly (zero slack).
        # FleetControlBase.__init__ reads G_OP_MAX_WT/G_OP_MAX_DTF for every fleet control module
        # generically, and RidePoolingBatchAssignmentFleetcontrol (dtd) also reads G_OP_MAX_WT_2
        # for the same retry mechanism SoD uses -- these are real, not a no-op. Wait_time_2
        # disabled (0) and detour factor capped at 60% (the loosest demand-side max_rel_detour
        # across user groups) rather than the much looser 1800s/150% tried mid-session -- see
        # pubtrans_utils._pt_extra_cols's comment for why.
        "op_max_wait_time": st_cfg.get("op_max_wait_time", 900),
        "op_max_wait_time_2": st_cfg.get("op_max_wait_time_2", 0),
        "op_max_detour_time_factor": st_cfg.get("op_max_detour_time_factor", 60),
    }
    row["rq_type"] = st_cfg.get("rq_type", DEFAULT_RQ_TYPE)
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

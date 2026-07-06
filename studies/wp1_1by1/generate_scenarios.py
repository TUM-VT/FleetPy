import os
import sys

import pandas as pd
import yaml

REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils.demand_utils import generate_demand_scenarios
from utils.network_utils import generate_networks
from utils.pubtrans_utils import generate_pubtrans, is_pt_line_service, pt_scenario_rows, pt_variant_lookup
from utils.supply_utils import fleet_entries, generate_initial_vehicle_distributions, scenario_row

STUDY_DIR = os.path.dirname(__file__)
SCENARIO_RANGES_DIR = os.path.join(STUDY_DIR, "scenario_ranges")
SCENARIO_RANGES_FILE = os.path.join(SCENARIO_RANGES_DIR, "scenario_ranges_starter.yaml")


def read_ranges(path=None):
    with open(path or SCENARIO_RANGES_FILE) as f:
        return yaml.safe_load(f)


def get_ranges():
    """
    If a path is provided as the first command-line argument, it is used to read scenario ranges.
    Otherwise, the default SCENARIO_RANGES_FILE is used.
    """
    ranges_path = None
    if len(sys.argv) > 1:
        ranges_path = sys.argv[1]
        if not os.path.isabs(ranges_path):
            # resolve against the scenario_ranges/ directory, falling back to cwd
            in_dir = os.path.join(SCENARIO_RANGES_DIR, ranges_path)
            ranges_path = in_dir if os.path.exists(in_dir) else ranges_path
    return read_ranges(ranges_path)


def generate_scenario_cfg(demand_scenarios, service_types, sim_end_time, pt_variants):
    """Tie demand scenarios and service types together into the scenario_cfg.csv, dispatching
    PT-line service types (sod/fixed_line) to the pubtrans row builder and the on-demand ones
    (dtd/stops) to a plain per-fleet-size row."""
    scenarios_dir = os.path.join(STUDY_DIR, "scenarios")
    os.makedirs(scenarios_dir, exist_ok=True)

    variant_lookup = pt_variant_lookup(pt_variants)

    rows = []
    for dv in demand_scenarios:
        base_name = (
            f"{dv['network_name']}_{dv['areal_density']}pkm2h_{dv['directionality']}dir"
            f"_{dv['spatial_distribution']}_{dv['temporal_distribution']}"
            f"_{dv['user_profile']}_seed{dv['seed']}"
        )
        for st_name, st_cfg in service_types.items():
            if is_pt_line_service(st_cfg):
                rows.extend(pt_scenario_rows(
                    base_name, dv, st_name, st_cfg, sim_end_time, variant_lookup))
            else:
                for n, size_tag in fleet_entries(st_cfg, dv["total_lambda"]):
                    rows.append(scenario_row(
                        f"{base_name}_{st_name}_{size_tag}", dv, st_cfg, sim_end_time, n, size_tag))

    out_path = os.path.join(scenarios_dir, "scenario_cfg.csv")
    df = pd.DataFrame(rows)
    for col in ("line_id", "pt_route_id", "pt_n_veh"):
        if col in df.columns:
            try:
                # pt_n_veh is a "line:n,..." string for two-hub scenarios; leave those as-is
                df[col] = df[col].astype("Int64")
            except (ValueError, TypeError):
                pass
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} scenarios to {out_path}")


def main():
    #  1. Read scenario ranges
    ranges = get_ranges()

    # 2. Generate networks, initial vehicle distributions, and PT variants
    networks = generate_networks(ranges["network"])
    generate_initial_vehicle_distributions([nw["name"] for nw in networks])
    pt_variants = generate_pubtrans(ranges)

    # 3. Generate demand scenarios
    demand_window = ranges["simulation"]["end_time"]
    boarding_match_radius = ranges["network"].get("boarding_point_spacing_m")
    demand_scenarios = generate_demand_scenarios(ranges["demand"], networks, demand_window, boarding_match_radius)

    # 4. Generate scenario configuration file
    sim_end_time = demand_window + ranges["simulation"]["cool_time"]
    generate_scenario_cfg(demand_scenarios, ranges["service_types"], sim_end_time, pt_variants)


if __name__ == "__main__":
    main()

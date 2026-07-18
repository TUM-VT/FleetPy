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


def _resolve_ranges_path(ranges_path):
    """Resolve a ranges file argument against the scenario_ranges/ directory, falling back to cwd
    (or an absolute path unchanged)."""
    if os.path.isabs(ranges_path):
        return ranges_path
    in_dir = os.path.join(SCENARIO_RANGES_DIR, ranges_path)
    return in_dir if os.path.exists(in_dir) else ranges_path


def get_ranges_list():
    """Read one ranges dict per command-line argument (each resolved against scenario_ranges/,
    falling back to cwd). Multiple files let one generate_scenarios.py call combine several
    ranges files' scenarios into a single scenario_cfg.csv -- e.g. for a cluster job that wants
    everything in one run rather than one file at a time. Falls back to SCENARIO_RANGES_FILE if
    no arguments are given."""
    if len(sys.argv) > 1:
        return [read_ranges(_resolve_ranges_path(p)) for p in sys.argv[1:]]
    return [read_ranges()]


def build_scenario_rows(demand_scenarios, service_types, sim_end_time, pt_variants, default_line_id=1):
    """Tie demand scenarios and service types together into scenario_cfg.csv rows, dispatching
    PT-line service types (sod/fixed_line) to the pubtrans row builder and the on-demand ones
    (dtd/stops) to a plain per-fleet-size row. Returns the row list (no I/O) so callers can
    combine rows from multiple ranges files before writing."""
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
                    base_name, dv, st_name, st_cfg, sim_end_time, variant_lookup, default_line_id))
            else:
                for n, size_tag in fleet_entries(st_cfg, dv["total_lambda"]):
                    rows.append(scenario_row(
                        f"{base_name}_{st_name}_{size_tag}", dv, st_cfg, sim_end_time, n, size_tag))
    return rows


def write_scenario_cfg(rows, out_name="scenario_cfg.csv"):
    """Write accumulated scenario_cfg.csv rows (from one or more ranges files) to disk.
    out_name lets a one-off targeted run (e.g. a single ranges file's sweep) write to its own
    file instead of clobbering the shared scenario_cfg.csv other work may still depend on."""
    scenarios_dir = os.path.join(STUDY_DIR, "scenarios")
    os.makedirs(scenarios_dir, exist_ok=True)
    out_path = os.path.join(scenarios_dir, out_name)
    df = pd.DataFrame(rows)

    if df["scenario_name"].duplicated().any():
        dupes = df.loc[df["scenario_name"].duplicated(), "scenario_name"].tolist()
        raise ValueError(
            f"Duplicate scenario_name(s) across the combined ranges files: {dupes[:5]}"
            f"{'...' if len(dupes) > 5 else ''}. Each ranges file's scenarios must be unique "
            f"when combined -- check for overlapping network/density/service_type combos.")

    # A column only some rows set (e.g. one service type overriding a normally-unset config key)
    # comes out NaN for every other row once combined into one CSV. FleetPy's config loader
    # (src/misc/config.py: decode_config_str) converts that NaN to None, which then OVERRIDES the
    # matching const_cfg.yaml default instead of falling back to it -- silently corrupting (or,
    # for fields used in unguarded arithmetic, crashing) those rows. Backfill any such column from
    # const_cfg.yaml's own value so an absent override is truly a no-op, not a None override.
    const_cfg_path = os.path.join(scenarios_dir, "const_cfg.yaml")
    if os.path.isfile(const_cfg_path):
        with open(const_cfg_path) as f:
            const_cfg = yaml.safe_load(f)
        for col in df.columns:
            if col in const_cfg and df[col].isna().any():
                df[col] = df[col].fillna(const_cfg[col])

    for col in ("line_id", "pt_route_id", "pt_n_veh"):
        if col in df.columns:
            try:
                # pt_n_veh is a "line:n,..." string for two-hub scenarios; leave those as-is
                df[col] = df[col].astype("Int64")
            except (ValueError, TypeError):
                pass
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} scenarios to {out_path}")


def generate_from_ranges(ranges):
    """Run the network/demand/PT/scenario-row generation pipeline for one ranges dict. Returns
    the scenario_cfg rows (not yet written) so multiple ranges files can be combined before the
    final write."""
    # Generate networks, initial vehicle distributions, and PT variants
    networks = generate_networks(ranges["network"])
    generate_initial_vehicle_distributions([nw["name"] for nw in networks])
    pt_variants = generate_pubtrans(ranges)

    # Generate demand scenarios
    demand_window = ranges["simulation"]["end_time"]
    boarding_match_radius = ranges["network"].get("boarding_point_spacing_m")
    demand_scenarios = generate_demand_scenarios(ranges["demand"], networks, demand_window, boarding_match_radius,
                                                  network_speed_kmh=ranges["network"].get("default_speed"))

    # Build scenario configuration rows
    sim_end_time = demand_window + ranges["simulation"]["cool_time"]
    default_line_id = ranges.get("pubtrans", {}).get("line_id", 1)
    return build_scenario_rows(demand_scenarios, ranges["service_types"], sim_end_time, pt_variants,
                                default_line_id)


def main():
    ranges_list = get_ranges_list()
    all_rows = []
    for ranges in ranges_list:
        all_rows.extend(generate_from_ranges(ranges))
    write_scenario_cfg(all_rows)


if __name__ == "__main__":
    main()

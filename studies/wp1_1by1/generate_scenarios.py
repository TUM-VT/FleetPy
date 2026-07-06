import os
import sys

import yaml

REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils.demand_utils import generate_demand_scenarios
from utils.network_utils import generate_networks
from utils.pubtrans_utils import generate_pubtrans
from utils.scenario_cfg_utils import generate_initial_vehicle_distributions, generate_scenario_cfg

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

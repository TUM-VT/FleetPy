"""
Script to create a FleetPy network from a MATSim network file.

The FleetPy data directory is resolved automatically as <repo_root>/data.
The network is written to <repo_root>/data/networks/<fleetpy_network_name>/.

Usage:
    python -m src.preprocessing.networks.network_from_matsim \\
        --matsim_network_path <path/to/matsim_network.xml[.gz]> \\
        --fleetpy_network_name <desired_network_name> \\
        [--no_hash_check]

Arguments:
    matsim_network_path  Path to the MATSim network XML file (plain or gzipped).
    fleetpy_network_name Name for the resulting FleetPy network folder.
    --no_hash_check      Skip the hash check and always (re-)create the network.
                         Default: the hash check is enforced and an error is raised
                         if the source file changed since the last creation.
"""

import argparse
from pathlib import Path
import sys

# <repo_root>/data  (this file lives at src/preprocessing/networks/)
FLEETPY_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FLEETPY_PATH))
FLEETPY_DATA_PATH = FLEETPY_PATH / "data"

from src.coupling.MATSimEqasim.misc import create_fleetpy_network_from_matsim



def main():
    parser = argparse.ArgumentParser(
        description="Create a FleetPy network from a MATSim network XML file."
    )
    parser.add_argument(
        "--matsim_network_path",
        required=True,
        help="Path to the MATSim network file (.xml or .xml.gz).",
    )
    parser.add_argument(
        "--fleetpy_network_name",
        required=True,
        help="Name of the network to create (used as subfolder name under data/networks/).",
    )
    parser.add_argument(
        "--hash_check",
        action="store_true",
        default=False,
        help="Enable hash-similarity check (default behavior).",
    )
    args = parser.parse_args()

    create_fleetpy_network_from_matsim(
        matsim_network_path=args.matsim_network_path,
        fleetpy_data_path=str(FLEETPY_DATA_PATH),
        network_name=args.fleetpy_network_name,
        enforce_hash_similarity=args.hash_check,
    )


if __name__ == "__main__":
    main()

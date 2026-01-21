# FleetPy/src/routing/NetworkAVaS.py

import os
import logging
import pandas as pd

from src.routing.NetworkBasic import NetworkBasic
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_NetworkAVaS = {
    "doc": "Dynamic routing network reading travel time predictions from AVaS outputs",
    "inherit": "NetworkBasic",
    "input_parameters_mandatory": [G_NETWORK_NAME],
    "input_parameters_optional": [
        G_NW_DYNAMIC_F,          # keep compatibility
        "avas_tt_dir",           # path to AVaS travel time outputs
        "avas_file_pattern",     # e.g. "tt_{t}.csv" or "edges_td_att.csv"
        "avas_update_dt",        # update frequency in seconds (optional)
    ],
    "mandatory_modules": [],
    "optional_modules": [],
}


class NetworkAVaS(NetworkBasic):
    """
    NetworkBasic-compatible dynamic network that updates edge travel times from AVaS predictions.
    This supports piecewise-constant time-dependent routing (weights updated over time).
    """

    def __init__(
        self,
        network_name_dir,
        network_dynamics_file_name=None,
        scenario_time=None,
        avas_tt_dir=None,
        avas_file_pattern=None,
        avas_update_dt=None,
    ):
        # init base network (loads nodes/edges with base tt)
        super().__init__(network_name_dir, network_dynamics_file_name=network_dynamics_file_name, scenario_time=scenario_time)

        self.avas_tt_dir = avas_tt_dir
        self.avas_file_pattern = avas_file_pattern or "edges_td_att.csv"
        self.avas_update_dt = avas_update_dt  # can be None
        self._last_avas_update_time = None

        if self.avas_tt_dir is None:
            LOG.warning("NetworkAVaS: avas_tt_dir is None -> will behave like NetworkBasic unless update is provided.")

    def update_network(self, simulation_time, update_state=True):
        """
        Override: update from AVaS at the current sim time (or at a chosen update interval).
        Returns True if tt were updated, else False.
        """
        self.sim_time = simulation_time
        if not update_state:
            return False
        if self.avas_tt_dir is None:
            return False

        # Optional throttling: only update every avas_update_dt seconds
        if self.avas_update_dt is not None and self._last_avas_update_time is not None:
            if simulation_time - self._last_avas_update_time < self.avas_update_dt:
                return False

        updated = self.load_tt_from_avas(simulation_time)
        if updated:
            self._last_avas_update_time = simulation_time
        return updated

    def load_tt_from_avas(self, simulation_time) -> bool:
        """
        Load travel times predicted/estimated by AVaS and update edge objects.
        Expected file format: columns [from_node,to_node,edge_tt]
        """
        self._reset_internal_attributes_after_travel_time_update()

        # Case 1: single file that is overwritten continuously (latest snapshot)
        # e.g., avas_tt_dir/edges_td_att.csv
        f = os.path.join(self.avas_tt_dir, self.avas_file_pattern)

        # Case 2: time-stamped files
        # f = os.path.join(self.avas_tt_dir, self.avas_file_pattern.format(t=int(simulation_time)))

        if not os.path.exists(f):
            LOG.debug(f"NetworkAVaS: no AVaS tt file found at {f}")
            return False

        try:
            df = pd.read_csv(f)
        except Exception as e:
            LOG.warning(f"NetworkAVaS: failed to read {f}: {e}")
            return False

        # Require these columns
        required = {"from_node", "to_node", "edge_tt"}
        if not required.issubset(df.columns):
            LOG.warning(f"NetworkAVaS: file {f} missing columns {required}; got {set(df.columns)}")
            return False

        # Update edges
        # This uses NetworkBasic._set_edge_tt which also updates cached node travel_infos_*
        for _, row in df.iterrows():
            self._set_edge_tt(int(row["from_node"]), int(row["to_node"]), float(row["edge_tt"]))

        return True

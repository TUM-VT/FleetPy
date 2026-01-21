# FleetPy/src/routing/NetworkAVaS.py

import os
import logging
import numpy as np
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
        f = os.path.join(self.avas_tt_dir, self.avas_file_pattern)
        if not os.path.exists(f):
            return False

        df = pd.read_csv(f)

        # Identify horizon columns: edge_tt_0, edge_tt_30, ...
        tt_cols = [c for c in df.columns if c.startswith("edge_tt_")]
        if not tt_cols:
            LOG.warning(f"No tt_* horizon columns found in {f}")
            return False

        # Parse horizon seconds from column names
        horizons = np.array([int(c.split("_")[2]) for c in tt_cols], dtype=float)
        order = np.argsort(horizons)
        horizons = horizons[order]
        tt_cols = [tt_cols[i] for i in order]

        # Determine snapshot base time t0
        # Option 1: use simulation_time as t0 (works if file is updated exactly at that time)
        t0 = simulation_time

        # Option 2 (preferred): read from a meta file or a column
        # t0 = float(df["t0"].iloc[0])

        # Build dict: (from,to) -> array(tt at horizons)
        edge_tt = {}
        for _, row in df.iterrows():
            key = (int(row["from_node"]), int(row["to_node"]))
            edge_tt[key] = row[tt_cols].to_numpy(dtype=float)

        self._tt_t0 = float(t0)
        self._tt_horizons = horizons
        self._edge_tt_table = edge_tt

        return True
    
    def get_section_tt_at(self, o_node_index: int, d_node_index: int, depart_time: float) -> float:
        """
        Time-dependent travel time tt_e(depart_time) using the latest AVaS horizon snapshot.
        Falls back to static tt if no AVaS table loaded for this edge.
        """
        # fallback: if no dynamic table available
        if not hasattr(self, "_edge_tt_table") or self._edge_tt_table is None:
            return super().get_section_infos(o_node_index, d_node_index)[0]

        arr = self._edge_tt_table.get((o_node_index, d_node_index))
        if arr is None:
            return super().get_section_infos(o_node_index, d_node_index)[0]

        t0 = self._tt_t0
        H = self._tt_horizons
        dt = float(depart_time - t0)

        if dt <= H[0]:
            return float(arr[0])
        if dt >= H[-1]:
            return float(arr[-1])

        # linear interpolation
        # TODO: could be optimized
        idx = int(np.searchsorted(H, dt, side="right") - 1)
        h0, h1 = H[idx], H[idx + 1]
        tt0, tt1 = arr[idx], arr[idx + 1]
        w = (dt - h0) / (h1 - h0)
        return float(tt0 + w * (tt1 - tt0))
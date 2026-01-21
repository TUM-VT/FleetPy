# FleetPy/src/routing/NetworkAVaS.py

import os
import logging
import numpy as np
import pandas as pd

from src.routing.NetworkBasic import NetworkBasic
from src.misc.globals import *
from src.routing.routing_imports.Router import Router

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

    def return_travel_costs_1to1(self, origin_position, destination_position, customized_section_cost_function=None):
        """
        Same as NetworkBasic.return_travel_costs_1to1 but forces mode='time_dependent'
        and passes start_time into the router.
        """
        trivial_test = self.test_and_get_trivial_route_tt_and_dis(origin_position, destination_position)
        if trivial_test is not None:
            return trivial_test[1]

        origin_node = origin_position[0]
        origin_overhead = (0.0, 0.0, 0.0)
        if origin_position[1] is not None:
            origin_node = origin_position[1]
            origin_overhead = self.get_section_overhead(origin_position, from_start=False)

        destination_node = destination_position[0]
        destination_overhead = (0.0, 0.0, 0.0)
        if destination_position[1] is not None:
            destination_overhead = self.get_section_overhead(destination_position, from_start=True)

        # IMPORTANT: the routing "departure time" at origin_node should include overhead already traveled on the start edge
        # In NetworkBasic, this is handled by adding overhead afterwards (static tt).
        # For TD, the departure time matters, so we shift start_time by origin_overhead travel time.
        start_time_effective = float(self.sim_time + origin_overhead[1])

        R = Router(
            self,
            origin_node,
            destination_nodes=[destination_node],
            mode="time_dependent",
            forward_flag=True,
            customized_section_cost_function=customized_section_cost_function,
            start_time=start_time_effective,
        )

        s = R.compute(return_route=False)[0][1]  # (cfv, tt, dis) from origin_node to destination_node

        # Add overheads back (same semantics as NetworkBasic)
        res = (
            s[0] + origin_overhead[0] + destination_overhead[0],
            s[1] + origin_overhead[1] + destination_overhead[1],
            s[2] + origin_overhead[2] + destination_overhead[2],
        )
        return res


    def return_best_route_1to1(self, origin_position, destination_position, customized_section_cost_function=None):
        """
        Same as NetworkBasic.return_best_route_1to1 but forces mode='time_dependent'
        and passes start_time into the router.
        """
        trivial_test = self.test_and_get_trivial_route_tt_and_dis(origin_position, destination_position)
        if trivial_test is not None:
            return trivial_test[0]

        origin_node = origin_position[0]
        if origin_position[1] is not None:
            origin_node = origin_position[1]

        destination_node = destination_position[0]

        # Effective departure time at origin_node (accounts for remaining fraction on first edge if applicable)
        origin_overhead = (0.0, 0.0, 0.0)
        if origin_position[1] is not None:
            origin_overhead = self.get_section_overhead(origin_position, from_start=False)
        start_time_effective = float(self.sim_time + origin_overhead[1])

        R = Router(
            self,
            origin_node,
            destination_nodes=[destination_node],
            mode="time_dependent",
            forward_flag=True,
            customized_section_cost_function=customized_section_cost_function,
            start_time=start_time_effective,
        )

        node_list = R.compute(return_route=True)[0][0]

        # Same post-processing as NetworkBasic: include original origin/destination edge endpoints if needed
        if origin_node != origin_position[0]:
            node_list = [origin_position[0]] + node_list
        if destination_position[1] is not None:
            node_list.append(destination_position[1])

        return node_list
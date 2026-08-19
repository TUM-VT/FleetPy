"""Routing on a forecast that changes along the route.

Every other network module in FleetPy prices a whole route from one table. That
is the right model for a measured table, which only describes the bin it was
measured in, but it wastes what a forecast knows: the exported prediction covers
six 5-minute bins, and a leg fifteen minutes long crosses three of them. Read
statically, minute 15 of a route is priced by the forecast for minute 5.

This module loads all six horizons and lets the search pick per edge. The C++
forward Dijkstra already carries the elapsed travel time to every node it
settles, so that elapsed time selects the layer: an edge entered 40 s into a
route is priced by horizon 1, an edge entered 700 s in by horizon 3. Choosing
the path under those costs is what makes it time-dependent routing rather than
time-dependent pricing of a path chosen statically.

What this does NOT change, and why
----------------------------------
* **Backward searches stay on horizon 1.** ``return_travel_costs_Xto1`` searches
  from a destination backwards over candidate vehicles, and a backward search
  cannot know the arrival time it is solving for. Those queries screen vehicles
  for a pickup, which happens now, so horizon 1 is the right table for them
  anyway. **Their costs are kept out of the store**, because the parent writes
  every origin a backward search reaches into the same ``travel_time_infos`` the
  1to1 path reads *before* searching: the fleet control screens candidates with
  Xto1 and then asks about the same pairs with 1to1, so leaving them in handed
  the time-dependent query a static answer for exactly the pairs that matter, and
  the arm would have been its static twin wherever it was screened first.
* **A query's departure time is the current bin.** FleetPy's routing API takes no
  departure time, so a leg that will not start until after a pickup is still
  priced from now. Layer 0 therefore covers the first 300 s of the route rather
  than of the trip. The mean pickup wait in this study is under one bin.
* **The base table is horizon 1.** A directory of ``tt_<t>.csv`` files loaded by
  a static network module and by this one differ only in the search, not in the
  numbers, so the time-dependent arm is one scenario parameter away from its
  static twin.

Authors: extends the FleetPy modules of Roman Engelhardt and Florian Dandl.
"""
import glob
import logging
import os
import re

from src.routing.NetworkBasicWithStoreCpp import NetworkBasicWithStoreCpp

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_NetworkTimeDependentCpp = {
    "doc": """
        Routes on a multi-horizon travel-time forecast. Identical to
        NetworkBasicWithStoreCpp except that load_tt_file also picks up the
        sibling horizon files (tt_<t>_h2.csv ... tt_<t>_h6.csv) and the C++
        forward search prices each edge at the horizon covering the moment a
        vehicle enters it. Falls back to the static search bin by bin whenever
        the horizon files are absent.
        """,
    "inherit": "NetworkBasicWithStoreCpp",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

#: Seconds one horizon covers, when the export's own grid cannot be read.
#: This is a property of the EXPORT (the forecast is on 5-minute bins), not of
#: how often the simulation reloads. An earlier version inferred it from the gap
#: between two reloads, which is a different quantity: loading bins 1800 s apart
#: made it believe each horizon covered 1800 s, so no route was long enough to
#: leave layer 0 and the arm silently reproduced its static twin.
DEFAULT_LAYER_SECONDS = 300.0

#: How finely the intra-bin offset is tracked. The result store is keyed on
#: (origin, destination) with no time in it, so it has to be dropped whenever the
#: offset moves; quantising trades a bounded phase error for keeping that cache
#: useful within a bucket. At 60 s the residual is a fifth of a layer.
OFFSET_QUANTUM_SECONDS = 60.0

#: Horizons the exporter writes beside the base table. Horizon 1 IS the base
#: table, so the siblings run 2..6 and occupy C++ layers 1..5.
MAX_HORIZON = 6


def horizon_file(ext_path, horizon):
    """`.../tt_46800.csv`, 3 -> `.../tt_46800_h3.csv`."""
    root, ext = os.path.splitext(ext_path)
    return f"{root}_h{int(horizon)}{ext}"


class NetworkTimeDependentCpp(NetworkBasicWithStoreCpp):
    def __init__(self, network_name_dir, network_dynamics_file_name=None, scenario_time=None):
        super().__init__(network_name_dir,
                         network_dynamics_file_name=network_dynamics_file_name,
                         scenario_time=scenario_time)
        self._layer_seconds = DEFAULT_LAYER_SECONDS
        self._layer_seconds_source = None   # the directory it was measured on
        self._in_backward_query = False
        self._offset_bucket = None
        # Provenance, read back by the KPI aggregator: a cell that silently ran
        # static because its horizon files were missing must be visible.
        self.td_bins_loaded = 0
        self.td_layers_loaded = 0
        self.td_bins_without_layers = 0

    def load_tt_file(self, scenario_time, ext_path=None):
        """Load the bin's base table, then every horizon exported beside it."""
        if ext_path is None:
            # Folder-driven dynamics carry no horizons; stay static rather than
            # route on whatever the previous bin left in the layers. Switched off
            # BEFORE delegating, so a load that raises cannot leave the router
            # pricing this bin with the last bin's forecast.
            self.cpp_router.setLayerSeconds(-1.0)
            super().load_tt_file(scenario_time, ext_path=ext_path)
            return

        super().load_tt_file(scenario_time, ext_path=ext_path)

        self._update_layer_seconds(ext_path)

        # The base table was just refreshed for the links this bin lists; the
        # layers must not outlive it. A link the exporter drops from one bin's
        # horizon file would otherwise keep the previous bin's forecast beside
        # this bin's base value, and one route would mix two forecasts.
        self.cpp_router.clearAllLayers()

        loaded = 0
        for horizon in range(2, MAX_HORIZON + 1):
            path = horizon_file(ext_path, horizon)
            if not os.path.isfile(path):
                continue
            # C++ layer 0 is the base table, so horizon h lives in layer h-1.
            self.cpp_router.updateEdgeTravelTimesLayer(path.encode(), horizon - 1)
            loaded += 1

        # the layers now describe intervals measured from this bin, so the
        # reading point goes back to its start
        self._offset_bucket = 0
        self.cpp_router.setQueryOffset(0.0)

        self.td_bins_loaded += 1
        self.td_layers_loaded += loaded
        if loaded:
            self.cpp_router.setLayerSeconds(self._layer_seconds)
        else:
            # No horizons for this bin. Routing on the layers a previous bin left
            # behind would mix two forecasts, so this bin runs static.
            self.td_bins_without_layers += 1
            self.cpp_router.setLayerSeconds(-1.0)
            LOG.warning("no horizon files beside %s; this bin routes statically", ext_path)

    def return_travel_costs_Xto1(self, *args, **kwargs):
        """Screen candidate vehicles, without teaching the store a static answer.

        The parent stores every (origin, destination) a backward search reaches.
        Those costs are horizon-1 by construction, and `return_travel_costs_1to1`
        consults the store before it searches, so without this the layered search
        is skipped for precisely the pairs the fleet control looked at first.
        """
        self._in_backward_query = True
        try:
            return super().return_travel_costs_Xto1(*args, **kwargs)
        finally:
            self._in_backward_query = False

    def _add_to_database(self, o_node, d_node, cfv, tt, dis):
        """Cache a backward result only where it agrees with the forward one.

        A route that finishes inside the first layer is priced identically by
        both searches, so storing it is exact and keeps the fast path the fleet
        control leans on: with `op_max_wait_time` at 300 s every backward search
        this study issues is bounded by exactly one layer. A longer one would be
        a static answer to a time-dependent question, and the store is consulted
        before the layered search runs, so it must not go in.
        """
        if (getattr(self, "_in_backward_query", False)
                and self.cpp_router.getLayerSeconds() > 0
                and tt >= self._layer_seconds):
            return
        super()._add_to_database(o_node, d_node, cfv, tt, dis)

    def update_network(self, simulation_time, update_state=True):
        """Move the layers' reading point with the simulation clock.

        The layers describe absolute clock intervals measured from the bin they
        were loaded in, but the search measures elapsed travel time from the
        query. A request answered 240 s into a 300 s bin therefore reaches the
        second layer after 60 s of driving, not after 300, and without this
        offset every route is priced as if it had departed at the bin boundary:
        a phase lead averaging half a layer, always toward the nearer-term
        forecast the arm exists to improve on.

        The offset is quantised, and the result store is dropped when it moves,
        because that store is keyed on (origin, destination) alone and its
        entries are only valid for the offset they were computed at.
        """
        res = super().update_network(simulation_time, update_state=update_state)
        if self._last_tt_load_time is None or self.cpp_router.getLayerSeconds() <= 0:
            return res
        offset = max(0.0, float(simulation_time) - self._last_tt_load_time)
        bucket = int(offset // OFFSET_QUANTUM_SECONDS)
        if bucket != self._offset_bucket:
            self._offset_bucket = bucket
            self.cpp_router.setQueryOffset(bucket * OFFSET_QUANTUM_SECONDS)
            self._reset_internal_attributes_after_travel_time_update()
        return res

    def _update_layer_seconds(self, ext_path):
        """Measure one horizon's length from the export's own bin grid.

        The exporter files one forecast per bin and names each file by the
        simulation second it governs, so the spacing between those names IS the
        width of a horizon. Measured once per directory; falls back to the
        5-minute default when a directory holds a single bin.
        """
        d = os.path.dirname(os.path.abspath(ext_path))
        if self._layer_seconds_source == d:
            return
        times = sorted(
            int(m.group(1)) for m in
            (re.match(r"tt_(\d+)\.csv$", os.path.basename(p))
             for p in glob.glob(os.path.join(d, "tt_*.csv")))
            if m)
        gaps = {b - a for a, b in zip(times, times[1:]) if b > a}
        if gaps:
            self._layer_seconds = float(min(gaps))
        self._layer_seconds_source = d
        LOG.info("time-dependent layers are %.0f s wide (from the bin grid in %s)",
                 self._layer_seconds, d)

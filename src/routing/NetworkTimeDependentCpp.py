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
  anyway.
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

        loaded = 0
        for horizon in range(2, MAX_HORIZON + 1):
            path = horizon_file(ext_path, horizon)
            if not os.path.isfile(path):
                continue
            # C++ layer 0 is the base table, so horizon h lives in layer h-1.
            self.cpp_router.updateEdgeTravelTimesLayer(path.encode(), horizon - 1)
            loaded += 1

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

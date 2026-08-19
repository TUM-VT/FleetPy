"""Selects which travel-time CSV the co-simulation feeds to the routing engine.

The SUMO server measures travel times from probe vehicles and pushes them into
FleetPy every ``sumo_t_update`` seconds. That measured channel is the **R0** arm
of the predictive-routing study: no estimation, no prediction, just what the
probes saw.

**R1** (an LDD/AVaS current-state estimate) and **R2a/b/c** (GNN predictions)
enter through the *same* ``routing_engine.load_tt_file(sim_time, ext_path=...)``
call -- the only thing that differs between arms is which CSV it reads. This
module picks that file, so the arms stay one scenario parameter apart instead of
one code path apart.

Set ``sumo_tt_source_dir`` to a directory of per-bin CSVs named
``tt_<sim_time>.csv`` (what the AVaS repo's ``scripts/routing/export_gnn_tt.py``
writes) and the server routes on those. Leave it unset for R0.

Failure mode this exists to prevent
-----------------------------------
``NetworkBasic.load_tt_file`` reads ``ext_path`` with no existence check, and a
link absent from a loaded file simply keeps its previous travel time. A
directory whose file names sit off the request grid therefore has two ways to
ruin a cell: crash it mid-flight, or -- if the caller swallows the error --
leave every edge at its last value, so a prediction arm behaves exactly like the
baseline it is supposed to beat and the result reads as "prediction doesn't
help". :func:`missing_bins` checks the whole requested grid *before* SUMO
starts, which turns that into a one-second failure instead of a wasted ~50
minute run.
"""
from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Sequence

#: Per-bin file-name convention, shared with ``export_gnn_tt.py`` in the AVaS
#: repo. The producer and this consumer live in different git repositories, so
#: the contract is pinned by a cross-repo test rather than by a shared import.
TT_FILE_TEMPLATE = "tt_{}.csv"

#: Written into the results directory so a finished cell records which travel
#: times it actually routed on.
STATS_FILENAME = "tt_source_stats.json"

#: Scenario parameter names. Mirrored in src/misc/globals.py as
#: G_SUMO_TT_SRC_DIR / G_SUMO_TT_SRC_MIN_COV; kept here as plain strings so this
#: module stays importable on its own.
PARAM_SOURCE_DIR = "sumo_tt_source_dir"
PARAM_MIN_COVERAGE = "sumo_tt_source_min_coverage"


def tt_bin_filename(sim_time) -> str:
    """Name of the travel-time file for one update bin."""
    return TT_FILE_TEMPLATE.format(int(round(float(sim_time))))


def resolve_source_dir(raw, main_dir: Optional[str] = None) -> Optional[str]:
    """Normalise the ``sumo_tt_source_dir`` scenario parameter.

    Unset means the R0 arm, and pandas turns an empty config cell into ``nan``,
    so every falsy spelling has to resolve to the same thing. A relative path is
    read from the FleetPy root, which is where ``studies/<study>/...`` already
    hangs.
    """
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    if os.path.isabs(s):
        return os.path.normpath(s)
    if main_dir:
        return os.path.normpath(os.path.join(main_dir, s))
    return os.path.normpath(s)


def requested_bin_times(start_time, end_time, t_update) -> List[int]:
    """The simulation times at which the server reloads travel times.

    Mirrors the run loop's own condition (``sim_time % t_update == 0`` while
    ``sim_time <= end_time``) rather than assuming the grid starts at
    ``start_time``: for a start time that is not a multiple of ``t_update`` the
    two differ, and the exported file names have to match what is actually
    asked for, not what looks natural.
    """
    step = int(t_update)
    if step <= 0:
        raise ValueError("t_update must be positive, got {}".format(t_update))
    start = int(math.ceil(float(start_time)))
    end = int(math.floor(float(end_time)))
    first = start + (-start) % step
    return list(range(first, end + 1, step))


def missing_bins(source_dir: str, times: Sequence[int]) -> List[int]:
    """Which of ``times`` have no file in ``source_dir``."""
    return [t for t in times
            if not os.path.isfile(os.path.join(source_dir, tt_bin_filename(t)))]


def layer_bin_filename(sim_time, horizon: int) -> str:
    """Name of one forecast layer, mirroring ``export_gnn_tt.layer_filename``."""
    if int(horizon) <= 1:
        return tt_bin_filename(sim_time)
    return "tt_{}_h{}.csv".format(int(round(float(sim_time))), int(horizon))


def bins_without_layers(source_dir: str, times: Sequence[int],
                        horizons: Sequence[int] = (2, 3, 4, 5, 6)) -> List[int]:
    """Which of ``times`` have a base table but no forecast layers beside it.

    A time-dependent arm whose layers are absent does not fail: every edge is
    priced from the base table and the arm silently becomes its own static twin,
    which is the one outcome that would look like a finished experiment.
    """
    out = []
    for t in times:
        if not os.path.isfile(os.path.join(source_dir, tt_bin_filename(t))):
            continue
        if not all(os.path.isfile(os.path.join(source_dir, layer_bin_filename(t, h)))
                   for h in horizons):
            out.append(t)
    return out


def layer_coverage_error(source_dir: Optional[str], times: Sequence[int],
                         horizons: Sequence[int] = (2, 3, 4, 5, 6)) -> Optional[str]:
    """Why ``source_dir`` cannot serve a *time-dependent* run, or ``None``."""
    if source_dir is None:
        return ("a time-dependent routing engine needs {}, which is unset; "
                "without it there is nothing to route the later horizons on"
                .format(PARAM_SOURCE_DIR))
    if not os.path.isdir(source_dir):
        return "{}={} does not exist".format(PARAM_SOURCE_DIR, source_dir)
    incomplete = bins_without_layers(source_dir, times, horizons)
    if not incomplete:
        return None
    return (
        "{}={} is missing forecast layers for {} of {} bins, so those bins would "
        "route on the base table alone and the arm would silently be its static "
        "twin; first: {}. Export with --all-horizons."
    ).format(PARAM_SOURCE_DIR, source_dir, len(incomplete), len(times), incomplete[:5])


def coverage_error(source_dir: str, times: Sequence[int],
                   min_coverage: float = 1.0) -> Optional[str]:
    """Why ``source_dir`` cannot serve this run, or ``None`` if it can.

    Returns a message instead of raising so the decision is testable without a
    running simulation; the caller turns it into the exception that aborts
    setup.
    """
    if not os.path.isdir(source_dir):
        return "{}={} does not exist".format(PARAM_SOURCE_DIR, source_dir)
    if not times:
        return ("{}={} was given but the scenario asks for no travel-time "
                "updates at all".format(PARAM_SOURCE_DIR, source_dir))
    missing = missing_bins(source_dir, times)
    coverage = 1.0 - len(missing) / len(times)
    if coverage >= min_coverage:
        return None
    return (
        "{}={} supplies only {:.1%} of the {} travel-time updates this scenario asks for "
        "(minimum {:.1%}); first missing bins: {}. Those file names come from the "
        "exporter's --grid-start and --horizon; lower {} if the gap is intended."
    ).format(PARAM_SOURCE_DIR, source_dir, coverage, len(times), min_coverage,
             missing[:5], PARAM_MIN_COVERAGE)


def tt_file_for(source_dir: Optional[str], sim_time, measured_path: str) -> Optional[str]:
    """Path to load for ``sim_time``, or ``None`` to keep the current travel times.

    With no source directory this is the probe-measured file the server has just
    written (R0). With one, it is the matching per-bin CSV -- and when that bin
    is absent the answer is ``None``, deliberately *not* the measured file:
    falling back would blend the baseline into the arm under test one bin at a
    time and quietly make the comparison meaningless.
    """
    if source_dir is None:
        return measured_path
    path = os.path.join(source_dir, tt_bin_filename(sim_time))
    return path if os.path.isfile(path) else None


def write_source_stats(results_dir: str, source_dir: Optional[str],
                       bins_loaded: int, bins_missing: int,
                       routing_engine=None) -> str:
    """Record which travel times the run routed on, next to the bin files.

    Without this the arm a cell ran is only inferable from its scenario name --
    and a name is exactly what cannot be trusted here, since cells 001-009 are
    named ``R1`` and are all R0.
    """
    out_dir = os.path.join(results_dir, "EdgeTravelTimes")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, STATS_FILENAME)
    with open(path, "w") as fh:
        stats = {
            "tt_source": "external" if source_dir else "measured",
            "tt_source_dir": source_dir,
            "bins_loaded": int(bins_loaded),
            "bins_missing": int(bins_missing),
        }
        # A time-dependent engine that found no horizon files routes on its base
        # table alone and IS its static twin, silently and for the whole run.
        # The engine name alone cannot tell the two apart, so the counters that
        # can are written here beside it.
        if routing_engine is not None and hasattr(routing_engine, "td_bins_loaded"):
            stats["routing_engine"] = type(routing_engine).__name__
            stats["td_bins_loaded"] = int(routing_engine.td_bins_loaded)
            stats["td_layers_loaded"] = int(routing_engine.td_layers_loaded)
            stats["td_bins_without_layers"] = int(routing_engine.td_bins_without_layers)
            stats["td_layer_seconds"] = float(
                getattr(routing_engine, "_layer_seconds", -1.0))
        json.dump(stats, fh, indent=2)
    return path

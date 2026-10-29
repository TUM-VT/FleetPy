"""Helpers for combining a .sumocfg's own settings with the ones the co-sim
server adds on the command line.

SUMO's command-line options *override* the configuration file rather than
extending it. For scalar options that is the intent, but for the
``--additional-files`` list it means that passing ``-a`` with a generated file
silently discards everything the scenario declared: traffic-light programs,
WAUT switching, public-transport stops and schedules. The simulation still runs
and still produces output, which makes the mistake easy to miss.

Kept free of FleetPy and traci imports so it can be unit-tested on its own.
"""
from __future__ import annotations

import gzip
import math
import os
import re
import xml.etree.ElementTree as ET
from typing import List, Mapping, Optional


def additional_files_from_sumocfg(sumocfg_path: str) -> List[str]:
    """Absolute paths of the additional files a .sumocfg declares.

    Relative entries are resolved against the config's own directory, which is
    how SUMO itself interprets them.
    """
    base = os.path.dirname(os.path.abspath(sumocfg_path))
    root = ET.parse(sumocfg_path).getroot()
    files: List[str] = []
    for el in root.iter("additional-files"):
        for part in el.get("value", "").split(","):
            part = part.strip()
            if not part:
                continue
            files.append(
                part if os.path.isabs(part) else os.path.normpath(os.path.join(base, part))
            )
    return files


def merge_additional_files(sumocfg_path: str, *extra: str) -> str:
    """Value for ``-a`` that keeps the scenario's additional files and appends
    `extra`, preserving order and dropping duplicates.
    """
    merged: List[str] = []
    for path in list(additional_files_from_sumocfg(sumocfg_path)) + list(extra):
        if path and path not in merged:
            merged.append(path)
    return ",".join(merged)


def sumo_options(params: Mapping, seed) -> List[str]:
    """The SUMO options every co-simulation run sets from its scenario parameters.

    The run that saves a warm-start state takes the same list, so the traffic it
    hands over at the start time is the traffic the co-simulation would have made.
    Keys are the ``G_SUMO_*`` scenario parameters (src/misc/globals.py), spelled out
    to keep this module free of FleetPy imports.
    """
    return ["--step-length", "1",
            "--collision.action", "warn",
            "--seed", str(seed),
            "--no-warnings", str(True),
            "--route-steps", str(params.get("sumo_route-steps", 200)),
            "--no-internal-links", str(params.get("sumo_no-internal-links", False)),
            "--ignore-junction-blocker", str(params.get("sumo_ignore-junction-blocker", -1)),
            "--time-to-teleport", str(params.get("sumo_time-to-teleport", 300)),
            "--time-to-teleport.highways", str(params.get("sumo_time-to-teleport.highways", 0)),
            "--eager-insert", str(params.get("sumo_eager-insert", False)),
            # Event cells change lane speeds mid-run, which can leave a bus with no
            # path to its fixed stop; SUMO treats that as fatal and the whole
            # co-simulation aborts. This drops the un-routable vehicle instead.
            "--ignore-route-errors", str(True)]


def _state_header(path: str) -> str:
    """The text before SUMO's snapshot element: the saving run's options, then the snapshot tag."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        head = f.read(65536)
    end = head.find(">", head.find("<snapshot"))
    return head[:end + 1] if end > 0 else head


def load_state_args(raw, seed, start_time, sumocfg_path: str, main_dir: Optional[str] = None) -> List[str]:
    """``--load-state`` for a warm start, or nothing when the parameter is unset.

    Without a state SUMO starts the network empty at the start time: trips under
    way at that moment never enter it, and the main roads ran 5% faster than in
    the full-day simulations behind the travel-time tables (docs/17 2026-10-28).
    ``{seed}`` in the path takes the scenario's seed, since each seed needs its own
    state; a relative path is read from the FleetPy root. A state saved at another
    time, by another seed, from another day's routes or with other signal programs loads
    without complaint and simulates other traffic, so those are checked against its header here.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    path = s.format(seed=int(seed))
    if not os.path.isabs(path) and main_dir:
        path = os.path.join(main_dir, path)
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"warm-start state for seed {seed} not found: {path}")
    head = _state_header(path)
    t = re.search(r'<snapshot[^>]*\btime="([0-9.]+)"', head)
    if not t or abs(float(t.group(1)) - float(start_time)) > 0.5:
        raise ValueError(f"state {path} was saved at time {t.group(1) if t else '?'}, the run starts at {start_time}")
    sd = re.search(r'<seed value="([0-9]+)"', head)
    if not sd or int(sd.group(1)) != int(seed):
        raise ValueError(f"state {path} was saved by seed {sd.group(1) if sd else '?'}, the run uses seed {seed}")
    want = ET.parse(sumocfg_path).getroot().find(".//route-files")
    got = re.search(r'<route-files value="([^"]*)"', head)
    names = lambda v: sorted(os.path.basename(x.strip()) for x in v.split(",") if x.strip())
    if want is not None and (not got or names(got.group(1)) != names(want.get("value", ""))):
        raise ValueError(f"state {path} was saved from route files {got.group(1) if got else '?'}, "
                         f"the run loads {want.get('value')}")
    # the signal programs too: a state saved with the hourly plans (TL/) restarts the plans' signals
    # in a run that has dropped them (the 3 July states, docs/17 2026-10-29)
    signals = lambda v: sorted(os.path.basename(x.strip()) for x in v.split(",") if "TL/" in x)
    run_adds = ET.parse(sumocfg_path).getroot().find(".//additional-files")
    state_adds = re.search(r'<additional-files value="([^"]*)"', head)
    have = signals(state_adds.group(1)) if state_adds else []
    need = signals(run_adds.get("value", "")) if run_adds is not None else []
    if have != need:
        raise ValueError(f"state {path} was saved with signal files {have or 'none (built-in programs)'}, "
                         f"the run loads {need or 'none (built-in programs)'}")
    return ["--load-state", path]

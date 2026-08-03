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

import os
import xml.etree.ElementTree as ET
from typing import List


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

"""Re-route vehicles already driving only in chosen 5-min bins.

En-route re-routing (`sumo_enroute_min_gain`) runs after every travel-time table load. A
schedule (`sumo_enroute_bins_file`, a CSV with one `bin_end` column of simulation
seconds) restricts it to the loads at those times: for example the bins in which a
disruption alert, computed before the run, is on. Without a schedule nothing changes.

Pure Python, no traci, so it can be tested without SUMO.
"""
import csv
import os
from typing import Optional


class EnrouteSchedule:
    """The table-load times at which driving vehicles may be re-routed."""

    def __init__(self, bin_ends, path: str = ""):
        self.bin_ends = {int(round(float(t))) for t in bin_ends}
        self.path = path

    @classmethod
    def from_parameter(cls, value, main_dir: str) -> Optional["EnrouteSchedule"]:
        """The schedule a scenario names, or None when unset. A relative path is read
        from the FleetPy root; a missing file stops the run before it starts."""
        if value in (None, ""):
            return None
        path = value if os.path.isabs(value) else os.path.join(main_dir, value)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"sumo_enroute_bins_file {value!r} not found ({path})")
        with open(path, newline="") as fh:
            ends = [r["bin_end"] for r in csv.DictReader(fh) if r.get("bin_end") not in (None, "")]
        return cls(ends, path=str(value))

    def active(self, sim_time) -> bool:
        return int(round(float(sim_time))) in self.bin_ends

    def __len__(self) -> int:
        return len(self.bin_ends)


def reroute_now(min_gain, schedule: Optional[EnrouteSchedule], sim_time) -> bool:
    """Re-route after this table load? Only with re-routing on, and, given a schedule, in its bins."""
    if min_gain is None:
        return False
    return schedule is None or schedule.active(sim_time)

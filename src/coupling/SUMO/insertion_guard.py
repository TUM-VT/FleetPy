"""Fleet vehicles must not wait forever to enter SUMO.

Each leg of an externally moving fleet vehicle is a fresh SUMO insertion. On an
edge where background demand is also inserted and the road is jammed, the fleet
vehicle queues behind that backlog indefinitely: FleetPy believes it is driving,
its passenger is never delivered, and the fleet quietly shrinks during the run.
SUMO teleports a driving vehicle stuck longer than --time-to-teleport, but never
one that is still waiting to be inserted.

This module extends that rule to insertion. A fleet vehicle waiting longer than
the threshold skips its blocked first edge, the same jump a teleport makes, and
is re-inserted on the next edge; with nothing left it has arrived. A setRoute
cannot do this for a vehicle that has not departed (SUMO refuses), so the server
removes and re-adds it under the same id.

Pure Python, no traci, so it can be tested without SUMO.
"""
from typing import Dict, Iterable, List, Optional


class InsertionGuard:
    """How long each fleet vehicle has been waiting for insertion."""

    def __init__(self, max_wait_s: float):
        self.max_wait_s = float(max_wait_s)
        self._since: Dict[str, float] = {}

    @classmethod
    def from_parameter(cls, value) -> Optional["InsertionGuard"]:
        """A guard for the scenario parameter, or None when it is unset or <= 0."""
        if value in (None, ""):
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return cls(v) if v > 0 else None

    def overdue(self, sim_time: float, pending_ids: Iterable[str]) -> List[str]:
        """Fleet vehicles pending for at least max_wait_s; their clocks restart."""
        pending = {v for v in pending_ids if str(v).startswith("fp_")}
        for v in list(self._since):
            if v not in pending:
                del self._since[v]
        due = []
        for v in sorted(pending):
            t0 = self._since.setdefault(v, sim_time)
            if sim_time - t0 >= self.max_wait_s:
                due.append(v)
                self._since[v] = sim_time
        return due


def skip_blocked_edge(route_edges: Iterable[str]) -> Optional[List[str]]:
    """The route after the blocked first edge, or None when nothing is left."""
    rest = list(route_edges)[1:]
    return rest or None

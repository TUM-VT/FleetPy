"""Information value of a repositioning trip's route, and the repositioning program that trades it
against empty driving (vaas docs/19, stage 5).

No FleetPy imports (numpy and pandas only; gurobipy is imported by `solve`), so the vaas tests
load this file directly. `InformationSeekingPavoneFC` is the FleetPy glue.

The table is the value map: one row per link-interval with positive value, given by its interval
end (s; the interval is the 300 s before it), the FleetPy edge (from_node, to_node) of the link and
its value, the misestimated travel time a fresh observation corrects (vehicle-seconds). A target
is one such link-interval, keyed ((from_node, to_node), interval_end). A route passes an edge in
the interval in which it leaves the edge, i.e. at the time it reaches the edge's to_node (as vaas
`src/operation/fleet_coverage.py` counts the fleet's crossings).
"""
import math
from collections import defaultdict

import numpy as np

INTERVAL_S = 300
#: The bundled gurobipy licence solves models of at most 2000 variables and 2000 constraints.
GUROBI_SIZE_LIMIT = 2000


def interval_end(t, interval_s=INTERVAL_S):
    """End of the interval (end - interval_s, end] holding time t."""
    return int(math.ceil(float(t) / interval_s) * interval_s)


class InformationTable:
    """The value map, sorted by interval end so a window is two binary searches."""

    def __init__(self, ends, from_nodes, to_nodes, values, interval_s=INTERVAL_S):
        values = np.asarray(values, dtype=float)
        keep = values > 0
        ends = np.rint(np.asarray(ends, dtype=float)[keep]).astype(np.int64)
        order = np.argsort(ends, kind="stable")
        self.interval_s = interval_s
        self._end = ends[order]
        self._a = np.asarray(from_nodes, dtype=np.int64)[keep][order]
        self._b = np.asarray(to_nodes, dtype=np.int64)[keep][order]
        self._v = values[keep][order]

    @classmethod
    def from_rows(cls, rows, interval_s=INTERVAL_S):
        """:param rows: iterable of (interval_end, from_node, to_node, value)"""
        cols = list(zip(*rows)) or [(), (), (), ()]
        return cls(*cols, interval_s=interval_s)

    @classmethod
    def from_csv(cls, path, interval_s=INTERVAL_S):
        import pandas as pd
        df = pd.read_csv(path, usecols=["interval_end", "from_node", "to_node", "value"])
        return cls(df["interval_end"].values, df["from_node"].values, df["to_node"].values,
                   df["value"].values, interval_s=interval_s)

    def window(self, t0, t1):
        """{target: value} of the link-intervals whose interval overlaps [t0, t1)."""
        lo = np.searchsorted(self._end, t0, side="right")                    # end > t0
        hi = np.searchsorted(self._end, t1 + self.interval_s, side="left")   # end - s < t1
        out = {}
        for e, a, b, v in zip(self._end[lo:hi].tolist(), self._a[lo:hi].tolist(),
                              self._b[lo:hi].tolist(), self._v[lo:hi].tolist()):
            key = ((a, b), e)
            out[key] = out.get(key, 0.0) + v
        return out


def node_times(route, start, section_tt, start_fraction=0.0):
    """Time at each node of a node route the vehicle starts at `start`.

    :param section_tt: (from_node, to_node) -> travel time (s)
    :param start_fraction: share of the first edge already behind the vehicle
    :return: [start, time at route[1], time at route[2], ...]
    """
    times = [start]
    t = start
    for i, (u, v) in enumerate(zip(route[:-1], route[1:])):
        tt = section_tt(u, v)
        t = t + (tt * (1.0 - start_fraction) if i == 0 else tt)
        times.append(t)
    return times


def passes(route, times):
    """[target] the route passes: each edge in the interval in which the vehicle leaves it."""
    return [((u, v), interval_end(t)) for u, v, t in zip(route[:-1], route[1:], times[1:])]


def plan_legs(veh_pos, remaining_route, stops, sim_time, t1):
    """The legs a vehicle plans to start before t1, with their start times.

    :param veh_pos: the vehicle's position (from_node, to_node or None, fraction or None)
    :param remaining_route: nodes still to drive to the first stop (starts at veh_pos[1], or at
        the node after veh_pos[0]); empty when the vehicle is not driving
    :param stops: [(stop position, time the vehicle leaves the stop or None)], in plan order
    :param sim_time: now; the leg to the first stop starts now
    :param t1: horizon end (s)
    :return: [(kind, leg, start time, fraction of the first edge behind the vehicle)]:
        ("route", nodes, ...) for the leg being driven, ("query", (origin, destination), ...)
        for a leg whose route the router has to give; legs that do not move are left out
    """
    legs = []
    if not stops:
        return legs
    if remaining_route:
        fraction = veh_pos[2] if veh_pos[1] is not None and veh_pos[2] is not None else 0.0
        legs.append(("route", [veh_pos[0]] + list(remaining_route), sim_time, fraction))
    elif veh_pos[0] != stops[0][0][0]:
        legs.append(("query", (veh_pos, stops[0][0]), sim_time, 0.0))
    for (prev_pos, leave), (pos, _) in zip(stops[:-1], stops[1:]):
        if leave is None or leave >= t1:
            break
        if prev_pos[0] != pos[0]:
            legs.append(("query", (prev_pos, pos), leave, 0.0))
    return legs


def unmet(window, passed):
    """{target: value} of the window's targets no planned route passes in their interval."""
    passed = set(passed)
    return {k: v for k, v in window.items() if k not in passed}


def end_edges(unmet_targets, zone_of_node):
    """{zone: edge}: each zone's unmet target edge of largest value summed over its intervals.

    An edge belongs to the zone of its from_node (FleetPy's convention for positions); edges
    outside the zones are dropped; ties go to the smaller edge.
    """
    value = defaultdict(float)
    for (edge, _), v in unmet_targets.items():
        value[edge] += v
    best = {}
    for edge in sorted(value):
        zone = zone_of_node(edge[0])
        if zone is None or zone < 0:
            continue
        if zone not in best or value[edge] > value[best[zone]]:
            best[zone] = edge
    return best


def credit(route_passes, unmet_targets):
    """[target] a route collects: the unmet targets it passes in their interval, in any zone."""
    return [p for p in dict.fromkeys(route_passes) if p in unmet_targets]


def target_groups(credits, value):
    """Targets credited by the same set of trips, merged: [(pairs, value, number of targets)].

    Their credit variables would be bound by the same trips, so one variable per group is
    exact. Sorted by value, largest first (ties by pairs).
    """
    by_target = defaultdict(set)
    for od, targets in credits.items():
        for t in targets:
            by_target[t].add(od)
    groups = defaultdict(lambda: [0.0, 0])
    for t, pairs in by_target.items():
        g = groups[tuple(sorted(pairs))]
        g[0] += value[t]
        g[1] += 1
    return sorted(((p, v, n) for p, (v, n) in groups.items()), key=lambda x: (-x[1], x[0]))


def solve(zones, travel_time, excess, desired, idle, credits, value, weight, threads=1,
          size_limit=GUROBI_SIZE_LIMIT, demand=True):
    """Pavone's repositioning program (PavoneHailingRepositioningFC) with the information a trip's
    route collects:

        min  sum_od t_od n_od  -  weight * sum_g V_g y_g
        s.t. e_z + sum_o n_oz - sum_d n_zd >= d_z     keep the desired excess (as before)
             sum_d n_zd <= idle_z                       send only idle vehicles (as before)
             y_g <= sum_(od in S_g) n_od               a target is collected if a trip passes it
             0 <= y_g <= 1                              and pays once however many do

    g is a group of targets passed by the same candidate trips S_g (target_groups), V_g their
    value. n_od exists only for origins with idle vehicles: the others are held at 0 by the
    second constraint, and leaving them out leaves room for groups under the licence's size
    limit, which keeps the most valuable groups that fit. weight = 0 has no groups.

    :param travel_time: {(o, d): s} for o != d
    :param credits: {(o, d): [target]} the targets the candidate trip on (o, d) collects
    :param value: {target: value}
    :param weight: seconds of empty driving per vehicle-second of misestimated travel time
    :param demand: False drops the desired-excess constraint: data collection only, the 2 x 2
        design's corner without demand anticipation (vaas docs/19, 17 Nov)
    :return: {"ok", "trips": [(o, d)] one per vehicle, "drive_s": sum t_od n_od,
              "credited_targets", "credited_value": the targets and value the trips collect,
              "groups", "groups_dropped", "value_dropped": the groups the licence left out}
    """
    import gurobipy as gp

    pairs = [(o, d) for o in zones if idle.get(o, 0) > 0 for d in zones if o != d]
    groups = target_groups(credits, value) if weight > 0 else []
    rows = (len(zones) if demand else 0) + sum(1 for z in zones if idle.get(z, 0) > 0)
    room = max(0, min(size_limit - len(pairs), size_limit - rows))
    kept, dropped = groups[:room], groups[room:]
    out = {"ok": False, "trips": [], "drive_s": 0.0, "credited_targets": 0, "credited_value": 0.0,
           "groups": len(groups), "groups_dropped": len(dropped),
           "value_dropped": sum(v for _, v, _ in dropped)}
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model("information_seeking_repositioning", env=env)
        m.setParam(gp.GRB.Param.Threads, threads)
        n = {od: m.addVar(vtype=gp.GRB.INTEGER, lb=0, obj=travel_time[od], name=f"n[{od[0]}-{od[1]}]")
             for od in pairs}
        y = [m.addVar(lb=0, ub=1, obj=-weight * v, name=f"y[{i}]") for i, (_, v, _) in enumerate(kept)]
        for z in zones:
            into = gp.quicksum(n[(o, z)] for o in zones if (o, z) in n)
            out_z = [n[(z, d)] for d in zones if (z, d) in n]
            if demand:
                m.addConstr(excess.get(z, 0) + into - gp.quicksum(out_z) >= desired.get(z, 0))
            if out_z:
                m.addConstr(gp.quicksum(out_z) <= idle.get(z, 0))
        for var, (ods, _, _) in zip(y, kept):
            m.addConstr(var <= gp.quicksum(n[od] for od in ods))
        m.ModelSense = gp.GRB.MINIMIZE
        m.optimize()
        if m.Status != gp.GRB.Status.OPTIMAL:
            return out
        for od in pairs:
            out["trips"].extend([od] * int(round(n[od].X)))
        got = [round(var.X) for var in y]
        out.update(ok=True, drive_s=sum(travel_time[od] for od in out["trips"]),
                   credited_targets=int(sum(g * k for g, (_, _, k) in zip(got, kept))),
                   credited_value=sum(g * v for g, (_, v, _) in zip(got, kept)))
    return out

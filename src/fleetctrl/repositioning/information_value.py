"""Information value of unmet sensing targets per zone, and the repositioning program that
trades it against empty driving (vaas docs/19, steps 3-4).

Pure Python apart from gurobipy, which `solve` imports when called; no FleetPy imports, so the
vaas tests load this file directly. `InformationSeekingPavoneFC` is the FleetPy glue.

A table row is one target link-interval: the interval end (s; the interval is the 300 s before
it), the FleetPy edge (from_node, to_node) of the main-road link and its value, the expected
error reduction a fresh observation brings (vehicle-seconds of misestimated travel time). The
vaas script `scripts/operation/build_information_table.py` writes the table.
"""
import csv
from collections import defaultdict

INTERVAL_S = 300


class InformationTable:
    """Target link-intervals by interval end."""

    def __init__(self, rows, interval_s=INTERVAL_S):
        """:param rows: iterable of (interval_end, from_node, to_node, value)"""
        self.interval_s = interval_s
        self.by_end = defaultdict(list)
        for end, a, b, value in sorted(rows):
            self.by_end[int(end)].append(((int(a), int(b)), float(value)))
        self.ends = sorted(self.by_end)

    @classmethod
    def from_csv(cls, path):
        with open(path, newline="") as f:
            rows = [(int(float(r["interval_end"])), int(r["from_node"]), int(r["to_node"]),
                     float(r["value"])) for r in csv.DictReader(f)]
        return cls(rows)

    def window(self, t0, t1):
        """[(edge, value)] of the target link-intervals whose interval overlaps [t0, t1)."""
        return [x for end in self.ends if end > t0 and end - self.interval_s < t1
                for x in self.by_end[end]]


def plan_legs(veh_pos, remaining_route, stops, t1):
    """The legs a vehicle plans to start before t1.

    :param veh_pos: the vehicle's position (from_node, to_node or None, fraction or None)
    :param remaining_route: nodes still to drive to the first stop (starts at veh_pos[1]);
        empty when the vehicle is not driving
    :param stops: [(stop position, time the vehicle leaves the stop or None)], in plan order
    :param t1: horizon end (s)
    :return: [("route", nodes)] for the leg being driven, [("query", (origin, destination))]
        for a leg whose route the router has to give; legs that do not move are left out
    """
    legs = []
    if not stops:
        return legs
    if remaining_route:
        legs.append(("route", [veh_pos[0]] + list(remaining_route)))
    elif veh_pos[0] != stops[0][0][0]:
        legs.append(("query", (veh_pos, stops[0][0])))
    for (prev_pos, leave), (pos, _) in zip(stops[:-1], stops[1:]):
        if leave is None or leave >= t1:
            break
        if prev_pos[0] != pos[0]:
            legs.append(("query", (prev_pos, pos)))
    return legs


def route_edges(routes):
    """Set of (from_node, to_node) pairs along node routes."""
    edges = set()
    for r in routes:
        edges.update(zip(r[:-1], r[1:]))
    return edges


def zone_information(rows, planned_edges, zone_of_node):
    """{zone: (value of the unmet targets, number of distinct unmet target edges)}.

    A target counts as met when a planned route passes its edge; an edge belongs to the zone of
    its from_node (FleetPy's convention for positions); edges outside the zones are dropped.
    """
    value = defaultdict(float)
    edges = defaultdict(set)
    for edge, v in rows:
        if edge in planned_edges:
            continue
        zone = zone_of_node(edge[0])
        if zone is None or zone < 0:
            continue
        value[zone] += v
        edges[zone].add(edge)
    return {z: (value[z], len(edges[z])) for z in value}


def solve(zones, travel_time, excess, desired, idle, information, weight, threads=1):
    """Pavone's repositioning program (PavoneHailingRepositioningFC) with an information term:

        min  sum_od t_od n_od  -  weight * sum_z (V_z / c_z) y_z
        s.t. e_z + sum_o n_oz - sum_d n_zd >= d_z     keep the desired excess (as before)
             sum_d n_zd <= idle_z                       send only idle vehicles (as before)
             y_z <= sum_o n_oz,  0 <= y_z <= c_z       credit vehicles sent into z, at most one
                                                        per unmet target edge

    V_z is the value of zone z's unmet targets and c_z their number of distinct edges, so a
    vehicle sent into z is credited with one edge's mean value. weight = 0 is today's program.

    :param travel_time: {(o, d): s} for o != d
    :param information: {zone: (V_z, c_z)}
    :param weight: seconds of empty driving per vehicle-second of expected error reduction
    :return: {"ok", "trips": [(o, d)] one per vehicle, "info_vehicles": sum y_z,
              "drive_s": sum t_od n_od, "info_value": sum (V_z / c_z) y_z}
    """
    import gurobipy as gp

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model("information_seeking_repositioning", env=env)
        m.setParam(gp.GRB.Param.Threads, threads)
        n = {(o, d): m.addVar(vtype=gp.GRB.INTEGER, lb=0, obj=travel_time[(o, d)], name=f"n[{o}-{d}]")
             for o in zones for d in zones if o != d}
        y = {}
        for z in zones:
            v, c = information.get(z, (0.0, 0))
            if weight > 0 and c > 0 and v > 0:
                y[z] = m.addVar(lb=0, ub=c, obj=-weight * v / c, name=f"y[{z}]")
        for z in zones:
            into = gp.quicksum(n[(o, z)] for o in zones if o != z)
            out = gp.quicksum(n[(z, d)] for d in zones if d != z)
            m.addConstr(excess.get(z, 0) + into - out >= desired.get(z, 0))
            m.addConstr(out <= idle.get(z, 0))
            if z in y:
                m.addConstr(y[z] <= into)
        m.ModelSense = gp.GRB.MINIMIZE
        m.optimize()
        if m.Status != gp.GRB.Status.OPTIMAL:
            return {"ok": False, "trips": [], "info_vehicles": 0, "drive_s": 0.0, "info_value": 0.0}
        trips = []
        for od, var in n.items():
            trips.extend([od] * int(round(var.X)))
        return {"ok": True, "trips": trips,
                "info_vehicles": int(round(sum(v.X for v in y.values()))),
                "drive_s": sum(travel_time[od] for od in trips),
                "info_value": sum(information[z][0] / information[z][1] * v.X for z, v in y.items())}

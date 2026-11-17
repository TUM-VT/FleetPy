import os
import random
import time
import logging
from typing import NamedTuple

import numpy as np

from src.fleetctrl.planning.VehiclePlan import RoutingTargetPlanStop
from src.fleetctrl.repositioning.PavoneHailingFC import PavoneHailingRepositioningFC
from src.fleetctrl.repositioning import information_value as iv
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_InformationSeekingPavoneFC = {
    "doc": """PavoneHailingRepositioningFC with an information term (vaas docs/19, stage 5): a
    repositioning trip is credited with the unmet sensing targets its route passes, traded
    against empty driving through one weight, and the vehicle drives that route. Mode "route"
    (default): information decides where idle vehicles go; mode "detour": PavoneFC decides where,
    information only a detour on the way.""",
    "inherit": "PavoneHailingRepositioningFC",
    "input_parameters_mandatory": ["op_repo_info_table", "op_repo_info_weight"],
    "input_parameters_optional": ["op_repo_info_demand", "op_repo_info_mode"],
    "mandatory_modules": [],
    "optional_modules": []
}


def _flag(value, default):
    """A 0/1 scenario column; missing or empty gives the default."""
    if value is None or (isinstance(value, float) and value != value) or str(value).strip() == "":
        return default
    return bool(int(float(value)))


INFO_MODES = ("route", "detour")


def info_mode(value, weight, demand):
    """op_repo_info_mode: "route" (missing or empty) or "detour"; detour mode needs a weight > 0
    (at 0 it would be PavoneFC itself) and PavoneFC's demand rule (it decides where vehicles go)."""
    missing = value is None or (isinstance(value, float) and value != value) or str(value).strip() == ""
    mode = "route" if missing else str(value).strip()
    if mode not in INFO_MODES:
        raise IOError(f"op_repo_info_mode {mode!r} is none of {INFO_MODES}")
    if mode == "detour" and weight <= 0:
        raise IOError("op_repo_info_mode = detour needs op_repo_info_weight > 0")
    if mode == "detour" and not demand:
        raise IOError("op_repo_info_mode = detour needs op_repo_info_demand = 1")
    return mode


class Candidate(NamedTuple):
    """The trip one zone pair is credited with: vehicle, end edge, node route, targets collected."""
    veh: object
    edge: tuple
    route: list
    targets: list


class InformationSeekingPavoneFC(PavoneHailingRepositioningFC):
    """Every repositioning step, over the horizon [t0, t1) (op_repo_horizons):

    1. Unmet targets: the value map's link-intervals in the horizon that no vehicle's planned leg
       passes in their interval. The leg being driven starts now from the vehicle's position,
       later legs at the previous stop's departure (the router's best route); node times come
       from the routing table's section travel times.
    2. A candidate trip per zone pair (o, d), o != d, o with idle vehicles, d with unmet targets:
       its end edge is d's most valuable unmet target edge, its vehicle the idle vehicle in o
       nearest that edge, its route the router's best route there plus the edge, from now.
    3. The candidate collects every unmet target its route passes in its interval, in any zone.
    4. The program (information_value.solve) subtracts weight x the value collected, each target
       paid once, from the empty driving time of PavoneFC's program.
    5. The first trip on a pair with a candidate sends the candidate vehicle (else the nearest
       idle one) along the credited route: a stop at the end edge's from_node, then one at its
       to_node. Further trips on the pair go to the same edge; pairs without a candidate keep
       PavoneFC's random node.

    op_repo_info_weight = 0 is PavoneHailingRepositioningFC itself (its program and assignment);
    only the unmet targets are logged, and drive_s and ok are left blank.

    op_repo_info_mode = detour (vaas docs/19, 17 Nov) keeps PavoneFC's program and assignment
    (vehicle and random destination node) and lets information decide only a detour on the way:
    for each vehicle PavoneFC just sent, the options are its direct route and routes via an
    unmet target edge (a, b) whose detour T(vehicle -> a) + tt(a, b) + T(b -> destination) -
    T(vehicle -> destination) is at most DETOUR_CAP_S (one forward search from the vehicle, one
    backward search to its destination), the DETOUR_OPTIONS best by weight x the edge's value in
    the interval passed - detour. Each option's full route is credited as in route mode; unmet
    targets leave out the sent vehicles' own legs (they are idle until now). One program
    (information_value.solve_detours) picks one option per vehicle, a target paid once. A vehicle
    on a detour stops at a, then b, then its PavoneFC node; one on its direct route gets PavoneFC's
    plan. PavoneFC's plans are built after the program, with PavoneFC's own draws.

    Parameters (scenario columns):
        op_repo_info_table: value map, path relative to the FleetPy root
        op_repo_info_weight: seconds of empty driving per vehicle-second of misestimated travel time
        op_repo_info_demand: 1 (default) keeps PavoneFC's demand rule; 0 drops it, so idle vehicles
            move only for information (data collection only, the 2 x 2 design of vaas docs/19)
        op_repo_info_mode: "route" (default) or "detour"; detour needs weight > 0 and demand 1

    Writes, per step, 4-<op_id>_information_repositioning.csv (LOG_COLUMNS: the unmet targets, the
    candidate routes, the program's size and result, the trips routed to an end edge, and the
    seconds spent on planned routes, candidate routes, zone travel times and the solver), and per
    trip routed to an end edge 4-<op_id>_information_trips.csv (TRIP_COLUMNS, with the route the
    trip is planned on, to check against the trajectory it drove). Detour mode writes the same
    files with DETOUR_LOG_COLUMNS (vehicles sent, via options built, the program's groups,
    vehicles detoured, their detour seconds, the targets and value the chosen routes collect and
    those the direct routes alone would, and the seconds spent) and DETOUR_TRIP_COLUMNS (one row
    per vehicle sent: its PavoneFC node, the via edge or blank, the detour, the via options it
    had, the targets and value its route collects, and the route).
    """

    LOG_COLUMNS = ["sim_time", "idle", "targets", "unmet_targets", "unmet_value", "zones_with_value",
                   "candidate_routes", "groups", "groups_dropped", "value_dropped", "trips",
                   "routed_trips", "as_credited", "credited_targets", "credited_value", "drive_s",
                   "planned_s", "candidate_s", "travel_s", "solve_s", "ok"]
    TRIP_COLUMNS = ["sim_time", "vid", "o_zone", "d_zone", "from_node", "to_node", "as_credited",
                    "targets", "value", "route"]
    DETOUR_LOG_COLUMNS = ["sim_time", "idle", "targets", "unmet_targets", "unmet_value", "sent", "options",
                          "groups", "groups_dropped", "value_dropped", "detoured", "detour_s",
                          "credited_targets", "credited_value", "direct_targets", "direct_value",
                          "planned_s", "pavone_s", "search_s", "routes_s", "solve_s", "ok"]
    DETOUR_TRIP_COLUMNS = ["sim_time", "vid", "o_zone", "d_zone", "dest_node", "via_from", "via_to",
                           "detour_s", "options", "targets", "value", "route"]
    info_demand = True                # PavoneFC's demand rule kept unless op_repo_info_demand = 0
    info_mode = "route"               # op_repo_info_mode

    def __init__(self, fleetctrl, operator_attributes, dir_names, solver="Gurobi"):
        super().__init__(fleetctrl, operator_attributes, dir_names, solver=solver)
        if self.solver_key != "Gurobi":
            raise IOError("InformationSeekingPavoneFC needs Gurobi")
        table = os.path.join(dir_names[G_DIR_MAIN], operator_attributes["op_repo_info_table"])
        self.info_table = iv.InformationTable.from_csv(table)
        self.info_weight = float(operator_attributes["op_repo_info_weight"])
        self.info_demand = _flag(operator_attributes.get("op_repo_info_demand"), default=True)
        if not self.info_demand and self.info_weight <= 0:
            raise IOError("op_repo_info_demand = 0 needs op_repo_info_weight > 0")
        self.info_mode = info_mode(operator_attributes.get("op_repo_info_mode"), self.info_weight,
                                   self.info_demand)
        op = self.fleetctrl.op_id
        self.info_log = os.path.join(self.output_dir, f"4-{op}_information_repositioning.csv")
        self.info_trip_log = os.path.join(self.output_dir, f"4-{op}_information_trips.csv")
        self._init_logs()

    def _columns(self):
        if self.info_mode == "detour":
            return self.DETOUR_LOG_COLUMNS, self.DETOUR_TRIP_COLUMNS
        return self.LOG_COLUMNS, self.TRIP_COLUMNS

    def _init_logs(self):
        for path, cols in zip((self.info_log, self.info_trip_log), self._columns()):
            with open(path, "w") as f:
                f.write(",".join(cols) + "\n")

    # --- one step ---------------------------------------------------------------------------

    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        """PavoneFC's step; the information log is written once its trips are assigned."""
        self._step, self._candidates, self._unmet, self._tails = None, {}, {}, {}
        self._sent, self._trip_rows, self._pending = set(), [], []
        vids = super().determine_and_create_repositioning_plans(sim_time, lock=lock)
        if self._step is not None:
            if self.info_mode == "detour":
                self._detour_step(sim_time)
            else:
                col = self.TRIP_COLUMNS.index("as_credited")
                self._step["routed_trips"] = len(self._trip_rows)
                self._step["as_credited"] = sum(r[col] for r in self._trip_rows)
            with open(self.info_log, "a") as f:
                f.write(",".join(str(self._step[c]) for c in self._columns()[0]) + "\n")
            with open(self.info_trip_log, "a") as f:
                for r in self._trip_rows:
                    f.write(",".join(str(x) for x in r) + "\n")
        return vids

    def _get_current_veh_plan_arrivals_and_repo_idle_vehicles(self, t0, t1, node_level=False):
        """PavoneFC's count; the idle vehicles per zone are kept for the candidates."""
        zone_dict = super()._get_current_veh_plan_arrivals_and_repo_idle_vehicles(t0, t1, node_level=node_level)
        self._idle = {z: list(v[2]) for z, v in zone_dict.items() if z >= 0 and v[2]}
        return zone_dict

    def _optimization_gurobi(self, sim_time, list_zones, v_i_e_dict, v_i_d_dict, number_idle_vehicles, zone_dict):
        t1 = sim_time + self.list_horizons[1]
        c0 = time.perf_counter()
        window = self.info_table.window(sim_time + self.list_horizons[0], t1)
        self._unmet = iv.unmet(window, self._planned_passes(sim_time, t1))
        c1 = time.perf_counter()
        step = {"sim_time": sim_time, "idle": sum(number_idle_vehicles.values()), "targets": len(window),
                "unmet_targets": len(self._unmet), "unmet_value": round(sum(self._unmet.values()), 1),
                "planned_s": round(c1 - c0, 3)}
        if self.info_mode == "detour":
            # PavoneFC's program; the sent vehicles are idle, so no leg of theirs met a target
            alpha_od, trips = super()._optimization_gurobi(sim_time, list_zones, v_i_e_dict, v_i_d_dict,
                                                           number_idle_vehicles, zone_dict)
            step.update(sent=len(trips), pavone_s=round(time.perf_counter() - c1, 3))
            self._step = step
            return alpha_od, trips
        if self.info_weight == 0:
            alpha_od, trips = super()._optimization_gurobi(sim_time, list_zones, v_i_e_dict, v_i_d_dict,
                                                           number_idle_vehicles, zone_dict)
            c2 = time.perf_counter()
            res = {"trips": trips, "groups": 0, "groups_dropped": 0, "value_dropped": 0.0,
                   "credited_targets": 0, "credited_value": 0.0, "drive_s": "", "ok": ""}
            step.update(zones_with_value=len(iv.end_edges(self._unmet, self.zone_system.get_zone_from_node)),
                        candidate_s=0.0, travel_s="", solve_s=round(c2 - c1, 3))
        else:
            ends = iv.end_edges(self._unmet, self.zone_system.get_zone_from_node)
            self._candidates = self._candidate_trips(sim_time, ends)
            c2 = time.perf_counter()
            # the parent's pair order, so the random zone centroids are drawn alike
            travel = {(o, d): self._get_od_zone_travel_info(sim_time, o, d)[0]
                      for o in list_zones for d in list_zones if o != d}
            c3 = time.perf_counter()
            res = iv.solve(list_zones, travel, v_i_e_dict, v_i_d_dict, number_idle_vehicles,
                           {od: c.targets for od, c in self._candidates.items()}, self._unmet,
                           self.info_weight, threads=self.fleetctrl.n_cpu, demand=self.info_demand)
            c4 = time.perf_counter()
            if not res["ok"]:
                LOG.warning(f"Operator {self.fleetctrl.op_id}: no optimal information-seeking repositioning"
                            f" at {sim_time} -> no repositioning")
            alpha_od = np.zeros((len(list_zones), len(list_zones)))
            for o, d in res["trips"]:
                alpha_od[zone_dict[o], zone_dict[d]] += 1
            trips = res["trips"]
            res = dict(res, drive_s=round(res["drive_s"], 1), ok=int(res["ok"]))
            step.update(zones_with_value=len(ends), candidate_s=round(c2 - c1, 3),
                        travel_s=round(c3 - c2, 3), solve_s=round(c4 - c3, 3))
        step.update(candidate_routes=len(self._candidates), groups=res["groups"],
                    groups_dropped=res["groups_dropped"], value_dropped=round(res["value_dropped"], 1),
                    trips=len(trips), credited_targets=res["credited_targets"],
                    credited_value=round(res["credited_value"], 1), drive_s=res["drive_s"], ok=res["ok"])
        self._step = step
        return alpha_od, trips

    def _od_to_veh_plan_assignment(self, sim_time, origin_zone_id, destination_zone_id, list_veh_to_consider,
                                   destination_node=None, lock=True):
        """A pair with a candidate: its first trip takes the candidate vehicle along the credited
        route, if that vehicle is still idle; otherwise, and for further trips, the idle vehicle
        nearest the end edge drives there. Other pairs: PavoneFC's assignment. Detour mode:
        PavoneFC's vehicle and node, planned after the detour program (_detour_step)."""
        if self.info_mode == "detour":
            veh, node = self._pavone_choice(sim_time, destination_zone_id, list_veh_to_consider, destination_node)
            self._pending.append((veh, origin_zone_id, destination_zone_id, node, lock))
            return [veh]
        od = (origin_zone_id, destination_zone_id)
        cand = self._candidates.get(od)
        if cand is None:
            return super()._od_to_veh_plan_assignment(sim_time, origin_zone_id, destination_zone_id,
                                                      list_veh_to_consider, destination_node=destination_node,
                                                      lock=lock)
        as_credited = od not in self._sent and any(v is cand.veh for v in list_veh_to_consider)
        self._sent.add(od)
        if as_credited:
            veh, route = cand.veh, cand.route
        else:
            tt = self._travel_times_to(cand.edge[0], list_veh_to_consider)
            if not tt:
                return super()._od_to_veh_plan_assignment(sim_time, origin_zone_id, destination_zone_id,
                                                          list_veh_to_consider, lock=lock)
            veh = min(list_veh_to_consider, key=lambda v: (tt.get(v.vid, np.inf), v.vid))
            route = self._route_to_edge(veh.pos, cand.edge)
        plan = self.fleetctrl.veh_plans[veh.vid]
        stops = [cand.edge[1]] if tuple(veh.pos) == (cand.edge[0], None, None) else list(cand.edge)
        for node in stops:
            plan.add_plan_stop(RoutingTargetPlanStop((node, None, None), locked=lock,
                                                     planstop_state=G_PLANSTOP_STATES.REPO_TARGET),
                               veh, sim_time, self.routing_engine)
        self.fleetctrl.assign_vehicle_plan(veh, plan, sim_time)
        if lock:
            self.fleetctrl.lock_current_vehicle_plan(veh.vid)
        got = iv.credit(iv.passes(route, self._node_times(route, sim_time, veh.pos)), self._unmet)
        self._trip_rows.append([sim_time, veh.vid, origin_zone_id, destination_zone_id, cand.edge[0],
                                cand.edge[1], int(as_credited), len(got),
                                round(sum(self._unmet[t] for t in got), 1), ";".join(str(n) for n in route)])
        return [veh]

    # --- routes and times ---------------------------------------------------------------------

    def _section_tt(self, u, v):
        try:
            return self.routing_engine.get_section_infos(u, v)[0]
        except KeyError:  # a node pair that is no edge (a vehicle off its planned route)
            return 0.0

    def _node_times(self, route, start, pos=None):
        fraction = pos[2] if pos is not None and pos[1] is not None and pos[2] is not None else 0.0
        return iv.node_times(route, start, self._section_tt, fraction)

    def _planned_passes(self, sim_time, t1):
        """Targets every vehicle's planned legs pass, each in its interval."""
        passed = set()
        for vid, plan in self.fleetctrl.veh_plans.items():
            veh = self.fleetctrl.sim_vehicles[vid]
            stops = []
            for ps in plan.list_plan_stops:
                arr, dep = ps.get_planned_arrival_and_departure_time()
                stops.append((ps.get_pos(), dep if dep is not None else arr))
            for kind, leg, start, fraction in iv.plan_legs(veh.pos, veh.cl_remaining_route, stops, sim_time, t1):
                route = leg if kind == "route" else self.routing_engine.return_best_route_1to1(*leg)
                passed.update(iv.passes(route, iv.node_times(route, start, self._section_tt, fraction)))
        return passed

    def _travel_times_to(self, node, vehicles):
        """{vid: travel time to node} of the vehicles that reach it (one backward search)."""
        if not vehicles:
            return {}
        res = self.routing_engine.return_travel_costs_Xto1([v.pos for v in vehicles], (node, None, None))
        by_pos = {tuple(p): tt for p, _, tt, _ in res if np.isfinite(tt)}
        return {v.vid: by_pos[tuple(v.pos)] for v in vehicles if tuple(v.pos) in by_pos}

    def _route_to_edge(self, pos, edge):
        """The route a vehicle at pos drives to two stops, the edge's from_node and its to_node:
        the router's best route to from_node, then from there to to_node (the edge itself
        wherever the edge is its own best route)."""
        a, b = edge
        head = self.routing_engine.return_best_route_1to1(pos, (a, None, None)) or [pos[0]]
        if edge not in self._tails:
            self._tails[edge] = self.routing_engine.return_best_route_1to1((a, None, None), (b, None, None))
        return list(head) + list(self._tails[edge][1:])

    def _candidate_trips(self, sim_time, ends):
        """{(o, d): Candidate} for every zone o with idle vehicles and every other zone d with an
        end edge."""
        cands = {}
        for d, edge in sorted(ends.items()):
            others = [v for z, vs in sorted(self._idle.items()) if z != d for v in vs]
            tt = self._travel_times_to(edge[0], others)
            for o, vs in sorted(self._idle.items()):
                reach = [v for v in vs if v.vid in tt]
                if o == d or not reach:
                    continue
                veh = min(reach, key=lambda v: (tt[v.vid], v.vid))
                route = self._route_to_edge(veh.pos, edge)
                got = iv.credit(iv.passes(route, self._node_times(route, sim_time, veh.pos)), self._unmet)
                cands[(o, d)] = Candidate(veh, edge, route, got)
        return cands

    # --- detour mode ------------------------------------------------------------------------------

    def _pavone_choice(self, sim_time, destination_zone_id, list_veh_to_consider, destination_node=None):
        """The vehicle and destination node RepositioningBase._od_to_veh_plan_assignment draws, with
        the same draws in the same order (random.seed(sim_time), random.choice, the zone's random
        centroid node or else a random node), without building the plan."""
        random.seed(sim_time)
        veh = random.choice(list_veh_to_consider)
        if destination_node is None:
            destination_node = self.zone_system.get_random_centroid_node(destination_zone_id)
            if destination_node < 0:
                destination_node = self.zone_system.get_random_node(destination_zone_id)
        return veh, destination_node

    def _plan_stops(self, sim_time, veh, nodes, lock):
        """PavoneFC's plan for an idle vehicle (its plan is empty), with a routing stop per node."""
        plan = self.fleetctrl.veh_plans[veh.vid]
        for node in nodes:
            plan.add_plan_stop(RoutingTargetPlanStop((node, None, None), locked=lock,
                                                     planstop_state=G_PLANSTOP_STATES.REPO_TARGET),
                               veh, sim_time, self.routing_engine)
        self.fleetctrl.assign_vehicle_plan(veh, plan, sim_time)
        if lock:
            self.fleetctrl.lock_current_vehicle_plan(veh.vid)

    def _credit(self, route, sim_time, pos):
        return iv.credit(iv.passes(route, self._node_times(route, sim_time, pos)), self._unmet)

    def _via_options(self, sim_time, pos, dest, unmet_edges, froms, exclude, k):
        """iv.via_edges for a vehicle at pos sent to dest: one forward search from the vehicle to
        the target edges' from_nodes, one backward search from their to_nodes to dest, both capped
        at T(vehicle -> dest) + DETOUR_CAP_S."""
        re = self.routing_engine
        t_direct = re.return_travel_costs_1to1(pos, (dest, None, None))[1]
        if not np.isfinite(t_direct):
            return []
        cap = t_direct + iv.DETOUR_CAP_S
        t_from = {p[0]: tt for p, _, tt, _ in re.return_travel_costs_1toX(pos, froms, max_cost_value=cap)
                  if np.isfinite(tt)}
        tos = sorted({(b, None, None) for a, b in unmet_edges
                      if a in t_from and t_from[a] + self._section_tt(a, b) <= cap})
        t_to = {p[0]: tt for p, _, tt, _ in re.return_travel_costs_Xto1(tos, (dest, None, None), max_cost_value=cap)
                if np.isfinite(tt)}
        return iv.via_edges(t_from, t_to, self._section_tt, t_direct, unmet_edges, sim_time, self.info_weight,
                            exclude=exclude, k=k)

    def _detour_step(self, sim_time):
        """Detour mode, after PavoneFC's program and assignment: each sent vehicle's options, the
        detour program, then the plans (in PavoneFC's order) and the logs."""
        c0 = time.perf_counter()
        sent = self._pending
        unmet_edges = iv.by_edge(self._unmet)
        froms = sorted({(a, None, None) for a, _ in unmet_edges})
        # option variables stay within half the licence's limit, leaving the rest to target groups
        k = min(iv.DETOUR_OPTIONS, max(0, iv.GUROBI_SIZE_LIMIT // (2 * len(sent)) - 1)) if sent else 0
        detours, credits, routes, vias, best = {}, {}, {}, {}, {}
        for i, (veh, _, _, dest, _) in enumerate(sent):
            route = self.routing_engine.return_best_route_1to1(veh.pos, (dest, None, None)) or [veh.pos[0]]
            detours[(i, 0)], credits[(i, 0)], routes[(i, 0)] = 0.0, self._credit(route, sim_time, veh.pos), route
            best[i] = self._via_options(sim_time, veh.pos, dest, unmet_edges, froms, set(credits[(i, 0)]), k) if k else []
        c1 = time.perf_counter()
        for i, (veh, _, _, dest, _) in enumerate(sent):
            seen = {tuple(routes[(i, 0)])}
            for j, (edge, detour, _, _) in enumerate(best[i], 1):
                route = self._route_to_edge(veh.pos, edge)
                route += (self.routing_engine.return_best_route_1to1((edge[1], None, None), (dest, None, None))
                          or [edge[1]])[1:]
                got = self._credit(route, sim_time, veh.pos)
                # a route the direct one or another option already is, or one that collects nothing
                # the direct route does not, is no option
                if tuple(route) in seen or set(got) <= set(credits[(i, 0)]):
                    continue
                seen.add(tuple(route))
                detours[(i, j)], credits[(i, j)], routes[(i, j)], vias[(i, j)] = round(detour, 1), got, route, edge
        c2 = time.perf_counter()
        if sent:
            res = iv.solve_detours(detours, credits, self._unmet, self.info_weight, threads=self.fleetctrl.n_cpu)
            if not res["ok"]:
                LOG.warning(f"Operator {self.fleetctrl.op_id}: no optimal detour program at {sim_time}"
                            f" -> PavoneFC's plans")
        else:
            res = {"ok": True, "choice": {}, "detoured": 0, "detour_s": 0.0, "credited_targets": 0,
                   "credited_value": 0.0, "groups": 0, "groups_dropped": 0, "value_dropped": 0.0}
        c3 = time.perf_counter()
        direct = {t for i in range(len(sent)) for t in credits[(i, 0)]}
        for i, (veh, o, d, dest, lock) in enumerate(sent):
            j = res["choice"].get(i, 0)
            edge = vias.get((i, j))
            nodes = [dest]
            if edge is not None:
                nodes = list(edge) + [dest]
                if tuple(veh.pos) == (edge[0], None, None):
                    nodes = nodes[1:]
                nodes = [n for n, m in zip(nodes, [None] + nodes[:-1]) if n != m]
            self._plan_stops(sim_time, veh, nodes, lock)
            got = credits[(i, j)]
            self._trip_rows.append([sim_time, veh.vid, o, d, dest, edge[0] if edge else "", edge[1] if edge else "",
                                    round(detours[(i, j)], 1), sum(1 for (v, _) in vias if v == i), len(got),
                                    round(float(sum(self._unmet[t] for t in got)), 1),
                                    ";".join(str(n) for n in routes[(i, j)])])
        self._step.update(options=len(vias), groups=res["groups"], groups_dropped=res["groups_dropped"],
                          value_dropped=round(res["value_dropped"], 1), detoured=res["detoured"],
                          detour_s=round(res["detour_s"], 1), credited_targets=res["credited_targets"],
                          credited_value=round(res["credited_value"], 1), direct_targets=len(direct),
                          direct_value=round(float(sum(self._unmet[t] for t in direct)), 1),
                          search_s=round(c1 - c0, 3), routes_s=round(c2 - c1, 3), solve_s=round(c3 - c2, 3),
                          ok=int(res["ok"]))

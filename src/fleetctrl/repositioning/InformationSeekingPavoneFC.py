import os
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
    against empty driving through one weight, and the vehicle drives that route.""",
    "inherit": "PavoneHailingRepositioningFC",
    "input_parameters_mandatory": ["op_repo_info_table", "op_repo_info_weight"],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


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

    Parameters (scenario columns):
        op_repo_info_table: value map, path relative to the FleetPy root
        op_repo_info_weight: seconds of empty driving per vehicle-second of misestimated travel time

    Writes, per step, 4-<op_id>_information_repositioning.csv (LOG_COLUMNS: the unmet targets, the
    candidate routes, the program's size and result, the trips routed to an end edge, and the
    seconds spent on planned routes, candidate routes, zone travel times and the solver), and per
    trip routed to an end edge 4-<op_id>_information_trips.csv (TRIP_COLUMNS, with the route the
    trip is planned on, to check against the trajectory it drove).
    """

    LOG_COLUMNS = ["sim_time", "idle", "targets", "unmet_targets", "unmet_value", "zones_with_value",
                   "candidate_routes", "groups", "groups_dropped", "value_dropped", "trips",
                   "routed_trips", "as_credited", "credited_targets", "credited_value", "drive_s",
                   "planned_s", "candidate_s", "travel_s", "solve_s", "ok"]
    TRIP_COLUMNS = ["sim_time", "vid", "o_zone", "d_zone", "from_node", "to_node", "as_credited",
                    "targets", "value", "route"]

    def __init__(self, fleetctrl, operator_attributes, dir_names, solver="Gurobi"):
        super().__init__(fleetctrl, operator_attributes, dir_names, solver=solver)
        if self.solver_key != "Gurobi":
            raise IOError("InformationSeekingPavoneFC needs Gurobi")
        table = os.path.join(dir_names[G_DIR_MAIN], operator_attributes["op_repo_info_table"])
        self.info_table = iv.InformationTable.from_csv(table)
        self.info_weight = float(operator_attributes["op_repo_info_weight"])
        op = self.fleetctrl.op_id
        self.info_log = os.path.join(self.output_dir, f"4-{op}_information_repositioning.csv")
        self.info_trip_log = os.path.join(self.output_dir, f"4-{op}_information_trips.csv")
        self._init_logs()

    def _init_logs(self):
        for path, cols in ((self.info_log, self.LOG_COLUMNS), (self.info_trip_log, self.TRIP_COLUMNS)):
            with open(path, "w") as f:
                f.write(",".join(cols) + "\n")

    # --- one step ---------------------------------------------------------------------------

    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        """PavoneFC's step; the information log is written once its trips are assigned."""
        self._step, self._candidates, self._unmet, self._tails = None, {}, {}, {}
        self._sent, self._trip_rows = set(), []
        vids = super().determine_and_create_repositioning_plans(sim_time, lock=lock)
        if self._step is not None:
            col = self.TRIP_COLUMNS.index("as_credited")
            self._step["routed_trips"] = len(self._trip_rows)
            self._step["as_credited"] = sum(r[col] for r in self._trip_rows)
            with open(self.info_log, "a") as f:
                f.write(",".join(str(self._step[c]) for c in self.LOG_COLUMNS) + "\n")
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
                           self.info_weight, threads=self.fleetctrl.n_cpu)
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
        nearest the end edge drives there. Other pairs: PavoneFC's assignment."""
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

import os
import time
import logging

import numpy as np

from src.fleetctrl.repositioning.PavoneHailingFC import PavoneHailingRepositioningFC
from src.fleetctrl.repositioning import information_value as iv
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_InformationSeekingPavoneFC = {
    "doc": """PavoneHailingRepositioningFC with an information term (vaas docs/19, steps 3-4): idle
    vehicles are also sent to zones whose sensing targets in the repositioning horizon no planned
    route passes, traded against empty driving through one weight.""",
    "inherit": "PavoneHailingRepositioningFC",
    "input_parameters_mandatory": ["op_repo_info_table", "op_repo_info_weight"],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


class InformationSeekingPavoneFC(PavoneHailingRepositioningFC):
    """Every repositioning step: the targets of the horizon (op_repo_horizons) whose edge no planned
    route passes give each zone an information value (information_value.zone_information), and
    the program (information_value.solve) subtracts weight x the value collected from the empty
    driving time. op_repo_info_weight = 0 is PavoneHailingRepositioningFC's program.

    Parameters (scenario columns):
        op_repo_info_table: target table, path relative to the FleetPy root
        op_repo_info_weight: seconds of empty driving per vehicle-second of expected error reduction

    Writes 4-<op_id>_information_repositioning.csv: per step the unmet value, the vehicles sent
    and credited with information, and the time spent on planned routes, zone travel times and
    the solver.
    """

    LOG_COLUMNS = ["sim_time", "idle", "targets", "unmet_edges", "unmet_value", "zones_with_value",
                   "trips", "info_vehicles", "drive_s", "info_value", "routes_s", "travel_s", "solve_s", "ok"]

    def __init__(self, fleetctrl, operator_attributes, dir_names, solver="Gurobi"):
        super().__init__(fleetctrl, operator_attributes, dir_names, solver=solver)
        if self.solver_key != "Gurobi":
            raise IOError("InformationSeekingPavoneFC needs Gurobi")
        table = os.path.join(dir_names[G_DIR_MAIN], operator_attributes["op_repo_info_table"])
        self.info_table = iv.InformationTable.from_csv(table)
        self.info_weight = float(operator_attributes["op_repo_info_weight"])
        self.info_log = os.path.join(self.output_dir, f"4-{self.fleetctrl.op_id}_information_repositioning.csv")
        with open(self.info_log, "w") as f:
            f.write(",".join(self.LOG_COLUMNS) + "\n")

    def _planned_routes(self, t1):
        """Node routes of every leg a vehicle plans to start before t1."""
        routes = []
        for vid, plan in self.fleetctrl.veh_plans.items():
            veh = self.fleetctrl.sim_vehicles[vid]
            stops = []
            for ps in plan.list_plan_stops:
                arr, dep = ps.get_planned_arrival_and_departure_time()
                stops.append((ps.get_pos(), dep if dep is not None else arr))
            for kind, leg in iv.plan_legs(veh.pos, veh.cl_remaining_route, stops, t1):
                if kind == "route":
                    routes.append(leg)
                else:
                    routes.append(self.routing_engine.return_best_route_1to1(*leg))
        return routes

    def _optimization_gurobi(self, sim_time, list_zones, v_i_e_dict, v_i_d_dict, number_idle_vehicles, zone_dict):
        t0 = sim_time + self.list_horizons[0]
        t1 = sim_time + self.list_horizons[1]
        c0 = time.perf_counter()
        targets = self.info_table.window(t0, t1)
        planned = iv.route_edges(self._planned_routes(t1))
        info = iv.zone_information(targets, planned, self.zone_system.get_zone_from_node)
        c1 = time.perf_counter()
        # same pair order as the parent, so the random zone centroids are drawn alike
        travel = {(o, d): self._get_od_zone_travel_info(sim_time, o, d)[0]
                  for o in list_zones for d in list_zones if o != d}
        c2 = time.perf_counter()
        res = iv.solve(list_zones, travel, v_i_e_dict, v_i_d_dict, number_idle_vehicles, info,
                       self.info_weight, threads=self.fleetctrl.n_cpu)
        c3 = time.perf_counter()
        if not res["ok"]:
            LOG.warning(f"Operator {self.fleetctrl.op_id}: no optimal information-seeking repositioning"
                        f" at {sim_time} -> no repositioning")
        row = [sim_time, sum(number_idle_vehicles.values()), len(targets),
               sum(c for _, c in info.values()), round(sum(v for v, _ in info.values()), 1), len(info),
               len(res["trips"]), res["info_vehicles"], round(res["drive_s"], 1),
               round(res["info_value"], 1), round(c1 - c0, 3), round(c2 - c1, 3), round(c3 - c2, 3),
               int(res["ok"])]
        with open(self.info_log, "a") as f:
            f.write(",".join(str(x) for x in row) + "\n")
        alpha_od = np.zeros((len(list_zones), len(list_zones)))
        for o, d in res["trips"]:
            alpha_od[zone_dict[o], zone_dict[d]] += 1
        return alpha_od, res["trips"]

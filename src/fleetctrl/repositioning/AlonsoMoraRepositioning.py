from __future__ import annotations
import numpy as np
import os
import random
import traceback
import logging
from collections import Counter
from typing import TYPE_CHECKING
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.fleetctrl.planning.VehiclePlan import RoutingTargetPlanStop
from src.misc.globals import *

if TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.fleetctrl.planning.PlanRequest import PlanRequest

LOG = logging.getLogger(__name__)

TIME_OUT = 120

INPUT_PARAMETERS_AlonsoMoraRepositioning = {
    "doc" :     """ this class implements the reactive repositioning strategy proposed in 
    On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment
    Javier Alonso-Mora, Samitha Samaranayake, Alex Wallar, Emilio Frazzoli, and Daniela Rus (2016)
    -> vehicles are sent to locations where customers have been rejected since the last repo time-trigger
    """,
    "inherit" : "RepositioningBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [G_OP_REPO_RES_PUF],
    "mandatory_modules": [],
    "optional_modules": []
}

class AlonsoMoraRepositioning(RepositioningBase):
    """ this class implements the reactive repositioning strategy proposed in 
    On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment
    Javier Alonso-Mora, Samitha Samaranayake, Alex Wallar, Emilio Frazzoli, and Daniela Rus (2016)
    -> vehicles are sent to locations where customers have been rejected since the last repo time-trigger
    """
    def __init__(self, fleetctrl : FleetControlBase, operator_attributes: dict, dir_names: dict, solver: str = "Gurobi"):
        super().__init__(fleetctrl, operator_attributes, dir_names, solver=solver)
        self._rejected_customer_origins_since_last_step = []
        self.min_reservation_buffer = operator_attributes.get(G_OP_REPO_RES_PUF, 3600)  # TODO  # minimum time for service before a vehicle has a reserved trip
        
    def register_rejected_customer(self, planrequest : PlanRequest, sim_time):
        LOG.debug(f"new rejected customer at {planrequest.get_o_stop_info()[0][0]} time {sim_time}")
        self._rejected_customer_origins_since_last_step.append(planrequest.get_o_stop_info()[0][0])
        
    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        """This method determines and creates new repositioning plans. The repositioning plans are directly assigned
        to the vehicles.
        In order to allow further database processes, the vids of vehicles with new plans are returned.

        :param sim_time: current simulation time
        :param lock: indicates if vehplans should be locked
        :return: list[vid] of vehicles with changed plans
        """
        if len(self._rejected_customer_origins_since_last_step) == 0:
            return []
        self.sim_time = sim_time
        if lock is None:
            lock = self.lock_repo_assignments
        
        # possible repostionings (only idle vehicles)    
        vid_to_origin_to_tt = {}    # idle vehicle id -> possible origin of repo target -> travel time (to be minimized)
        origin_to_counts = Counter(self._rejected_customer_origins_since_last_step)   # origin -> number of occurances
        for veh in self.fleetctrl.sim_vehicles:
            if len(self.fleetctrl.veh_plans[veh.vid].list_plan_stops) == 0:
                vid_to_origin_to_tt[veh.vid] = {}
                for rej_o in origin_to_counts.keys():
                    _, tt, _ = self.fleetctrl.routing_engine.return_travel_costs_1to1(veh.pos, (rej_o, None, None) )
                    vid_to_origin_to_tt[veh.vid][rej_o] = tt
            elif len(self.fleetctrl.veh_plans[veh.vid].list_plan_stops) == 1 and self.fleetctrl.veh_plans[veh.vid].list_plan_stops[-1].is_locked_end():
                # reservation leg at end -> insert repo in between if far in future
                _, base_tt, _ = self.fleetctrl.routing_engine.return_travel_costs_1to1(veh.pos, self.fleetctrl.veh_plans[veh.vid].list_plan_stops[-1].get_pos() )
                est = self.fleetctrl.veh_plans[veh.vid].list_plan_stops[-1].get_earliest_start_time()
                if sim_time + base_tt + self.min_reservation_buffer < est:
                    for rej_o in origin_to_counts.keys():
                        _, o_tt_1, _ = self.fleetctrl.routing_engine.return_travel_costs_1to1(veh.pos, (rej_o, None, None) )
                        _, o_tt_2, _ = self.fleetctrl.routing_engine.return_travel_costs_1to1((rej_o, None, None), self.fleetctrl.veh_plans[veh.vid].list_plan_stops[-1].get_pos() )
                        if sim_time + o_tt_1 + o_tt_2 + self.min_reservation_buffer < est:  # check detour around repo target
                            try:
                                vid_to_origin_to_tt[veh.vid][rej_o] = o_tt_1 + o_tt_2 - base_tt     # base_tt has to be driven anyway
                            except KeyError:
                                vid_to_origin_to_tt[veh.vid] = {rej_o : o_tt_1 + o_tt_2 - base_tt}
                    
        LOG.debug(f"vids to origin tt: {vid_to_origin_to_tt}")
                    
        if self.solver_key == "Gurobi":
            vid_to_repo_target = self._solve_gurobi(vid_to_origin_to_tt, origin_to_counts, sim_time)
        else:
            raise NotImplementedError(f"optimizer {self.solver_key} not implemented here!")
        
        vid_changed_plans = []
        for vid, target_node in vid_to_repo_target.items():
            vid_changed_plans.append(vid)
            LOG.debug(f"repo {vid} -> {target_node} with tt {vid_to_origin_to_tt[vid][target_node]}")
            ps = RoutingTargetPlanStop((target_node, None, None), locked=lock, planstop_state=G_PLANSTOP_STATES.REPO_TARGET)
            veh_plan = self.fleetctrl.veh_plans[vid]
            veh_obj = self.fleetctrl.sim_vehicles[vid]
            if len(veh_plan.list_plan_stops) == 0 or not veh_plan.list_plan_stops[-1].is_locked_end():
                veh_plan.add_plan_stop(ps, veh_obj, sim_time, self.routing_engine)
            else:
                LOG.debug(f" -> insert before reservation ")
                veh_plan.list_plan_stops = veh_plan.list_plan_stops[:-2] + [ps] + [veh_plan.list_plan_stops[-1]]
                veh_plan.update_tt_and_check_plan(veh_obj, sim_time, self.fleetctrl.routing_engine)
            self.fleetctrl.assign_vehicle_plan(veh_obj, veh_plan, sim_time)
            if lock:
                self.fleetctrl.lock_current_vehicle_plan(veh_obj.vid)
                
        self._rejected_customer_origins_since_last_step = []
                
        return vid_changed_plans
            
    def _solve_gurobi(self, vid_to_origin_tt, origin_to_counts, sim_time):
        """ solves the assignment problem vehicle -> repo target using gurobi
        :param vid_to_origin_to_tt: dict vehicle_id -> repo target position -> travel time of all possibilities
        :param origin_to_counts: dict target position -> number of occurances
        :return dict vid ->  repo target (assignment)"""
        import gurobipy as gurobi
        
        model_name = f"AlonsoMoraRepositioning: _solve_gurobi {sim_time}"
        with gurobi.Env(empty=True) as env:
            if self.fleetctrl.log_gurobi:
                with open(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], "gurobi_log.log"), "a") as f:
                    f.write(f"\n\n{model_name}\n\n")
                env.setParam('OutputFlag', 1)
                env.setParam('LogToConsole', 0)
                env.setParam('LogFile', os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], "gurobi_log.log") )
                env.start()
            else:
                env.setParam('OutputFlag', 0)
                env.setParam('LogToConsole', 0)
                env.start()

            m = gurobi.Model(model_name, env = env)
            
            m.setParam(gurobi.GRB.param.Threads, self.fleetctrl.n_cpu)
            m.setParam('TimeLimit', TIME_OUT)
           
            expr = gurobi.LinExpr()   # building optimization objective
            vid_constr_dict = {}    # vid -> list var
            pos_constr_dict = {}    # pos_key -> list var
            for vid, o_dict in vid_to_origin_tt.items():
                for rej_o in o_dict.keys():
                    var = m.addVar(name = f"{vid}_{rej_o}", obj = o_dict[rej_o], vtype = gurobi.GRB.BINARY)
                    expr.add(var, o_dict[rej_o])
                    try:
                        vid_constr_dict[vid].append(var)
                    except KeyError:
                        vid_constr_dict[vid] = [var]
                    try:
                        pos_constr_dict[rej_o].append(var)
                    except KeyError:
                        pos_constr_dict[rej_o] = [var]
                    
            m.setObjective(expr, gurobi.GRB.MINIMIZE)

            assign_all_vehicles = False
            if len(vid_constr_dict.keys()) <= len(self._rejected_customer_origins_since_last_step):
                assign_all_vehicles = True

            for vid, varlist in vid_constr_dict.items():
                expr = gurobi.LinExpr()
                for var in varlist:
                    expr.add(var, 1)
                if assign_all_vehicles:
                    m.addConstr(expr, gurobi.GRB.EQUAL, 1, "c_{}".format(vid))
                else:
                    m.addConstr(expr, gurobi.GRB.LESS_EQUAL, 1, "c_{}".format(vid))
                    
            for rej_o, varlist in pos_constr_dict.items():
                expr = gurobi.LinExpr()
                for var in varlist:
                    expr.add(var, 1)
                if assign_all_vehicles:
                    m.addConstr(expr, gurobi.GRB.LESS_EQUAL, origin_to_counts[rej_o], "c_{}".format(rej_o))
                else:
                    m.addConstr(expr, gurobi.GRB.EQUAL, origin_to_counts[rej_o], "c_{}".format(rej_o))
                    
            m.optimize() #optimization
            
            #get solution
            if m.status == gurobi.GRB.Status.OPTIMAL:
                varnames = m.getAttr("VarName", m.getVars())
                solution = m.getAttr("X",m.getVars())
                
                new_assignments = {}
                for x in range(len(solution)):
                    if round(solution[x]) == 1:
                        key = varnames[x].split("_")
                        vid, rej_o = int(key[0]), int(key[1])
                        new_assignments[vid] = rej_o
            else:
                model_f = os.path.join(self.output_dir, f"70_repo_am_opt_model_infeasible_{sim_time}.lp")
                m.write(model_f)
                LOG.warning(f"Operator {self.fleetctrl.op_id}: No Optimal Solution! status {m.status}"
                            f" -> no repositioning")
                new_assignments = {}

            return new_assignments
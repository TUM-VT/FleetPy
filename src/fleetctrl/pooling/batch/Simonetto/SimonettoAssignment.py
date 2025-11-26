import logging
from typing import Callable, Dict, Any, List
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.misc.globals import G_DIR_OUTPUT

import time
import numpy as np

import src.fleetctrl.pooling.GeneralPoolingFunctions as GeneralPoolingFunctions
from src.fleetctrl.pooling.batch.BatchAssignmentAlgorithmBase import BatchAssignmentAlgorithmBase, SimulationVehicleStruct
from src.fleetctrl.planning.VehiclePlan import VehiclePlan, BoardingPlanStop
from src.fleetctrl.pooling.immediate.insertion import simple_remove, insert_prq_in_selected_veh_list
from src.misc.globals import *
from src.routing.NetworkBase import NetworkBase
from src.simulation.Legs import VehicleRouteLeg
from src.simulation.Vehicles import SimulationVehicle

from src.fleetctrl.pooling.immediate.singleVehicleDARP import solve_single_vehicle_DARP_exhaustive, solve_single_vehicle_DARP_LNS

LOG = logging.getLogger(__name__)
LARGE_INT = 100000
MAX_T = 1000000
TIME_OUT = 30
WRITE_PROBLEM = False

INPUT_PARAMETERS_SimonettoAssignment = {
    "doc" :  """This class implements the assingment algorithm of Simonetto et al. 2019.
                it assigns requests in batches. instead of finding optimal assignments, only schedules for new requests are computed. requests in the same batch cannot be assigned to the same vehicle, resulting in a linear assignment problem.
                Optional parameter : op_max_exhaustive_darp -> maximum number of requests for which the exhaustive DARP is solved. if the number of requests is larger, the requests are inserted into the vehicle plans of the vehicle.  base value: 4 (for speed-up use 1)""",
    "inherit" : "BatchAssignmentAlgorithmBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        G_RA_MAX_EXH_DARP
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class SimonettoAssignment(BatchAssignmentAlgorithmBase):
    def __init__(self, fleetcontrol: FleetControlBase, routing_engine: NetworkBase, sim_time: int, obj_function: Callable[..., Any], operator_attributes: dict, optimisation_cores: int = 1, seed: int = 6061992, veh_objs_to_build: Dict[int, SimulationVehicleStruct] = {}):
        """This class implements the assingment algorithm of Simonetto et al. 2019.
        it assigns requests in batches. instead of finding optimal assignments, only schedules for new requests are computed. requests in the same batch cannot be assigned to the same vehicle, resulting in a linear assignment problem.
        Optional parameter : op_max_exhaustive_darp -> maximum number of requests for which the exhaustive DARP is solved. if the number of requests is larger, the requests are inserted into the vehicle plans of the vehicle.  base value: 4 (for speed-up use 1)
        :param fleetcontrol : fleetcontrol object, which uses this assignment algorithm
        :param routing_engine : routing_engine object
        :param sim_time : (int) current simulation time
        :param obj_function : obj_function to rate a vehicle plan
        :param operator_attributes : input parameter dict for operator attributes
        :param seed : random seed
        :param veh_objs_to_build: dict vid -> SimulationVehicleStruct which will be considered in opt. if empty dict, vehicles from fleetcontrol will be taken
        """
        super().__init__(fleetcontrol, routing_engine, sim_time, obj_function, operator_attributes, optimisation_cores, seed=seed, veh_objs_to_build=veh_objs_to_build)
        
        self._max_prqs_exhaustive_DARP = self.operator_attributes.get(G_RA_MAX_EXH_DARP)
        if self._max_prqs_exhaustive_DARP is None or type(self._max_prqs_exhaustive_DARP) != int:
            if self._max_prqs_exhaustive_DARP is None:
                self._max_prqs_exhaustive_DARP = 4
            else:
                if np.isnan(self._max_prqs_exhaustive_DARP):
                    self._max_prqs_exhaustive_DARP = 4
                else:
                    self._max_prqs_exhaustive_DARP = int(self._max_prqs_exhaustive_DARP)
 
        self._optimisation_solutions = {}
        
    def compute_new_vehicle_assignments(self, sim_time: int, vid_to_list_passed_VRLs: Dict[int, List[VehicleRouteLeg]], veh_objs_to_build: Dict[int, SimulationVehicle] = {}, new_travel_times: bool = False, build_from_scratch: bool = False):
        LOG.debug(f"new assignments at time {sim_time} with requests {self.unassigned_requests.keys()}")
        #0) set database
        self.sim_time = sim_time
        self.veh_objs = {}
        if len(veh_objs_to_build.keys()) == 0:
            for veh_obj in self.fleetcontrol.sim_vehicles:
                veh_obj_struct = SimulationVehicleStruct(veh_obj, self.fleetcontrol.veh_plans.get(veh_obj.vid, VehiclePlan(veh_obj, self.sim_time, self.routing_engine, [])), sim_time, self.routing_engine)
                self.veh_objs[veh_obj.vid] = veh_obj_struct
        else:
            self.veh_objs = veh_objs_to_build
            
        non_repo_veh_plans = {}
        for vid, veh_obj in self.veh_objs.items():
            current_veh_p  : VehiclePlan = self.fleetcontrol.veh_plans.get(vid, VehiclePlan(veh_obj, self.sim_time, self.routing_engine, []))
            current_veh_p.update_tt_and_check_plan(veh_obj, sim_time, self.routing_engine, keep_feasible=True)
            obj = self.fleetcontrol.compute_VehiclePlan_utility(sim_time, veh_obj, current_veh_p)
            current_veh_p.set_utility(obj)
            veh_p = current_veh_p.copy_and_remove_empty_planstops(veh_obj, sim_time, self.routing_engine)
            obj = self.fleetcontrol.compute_VehiclePlan_utility(sim_time, veh_obj, veh_p)
            veh_p.set_utility(obj)
            non_repo_veh_plans[vid] = veh_p
        
        #1) compute RV for unassigned requests
        t = time.time()
        r2v = self._computeRV(self.unassigned_requests.keys())
        t_rv = time.time() - t
        #2) compute DARPS and cij for RV connections
        t = time.time()
        rid_to_vid_to_cost_and_plan = {}
        
        for rid, vehicle_dict in r2v.items():
            prq = self.active_requests[rid]
            if self.rid_to_consider_for_global_optimisation.get(rid) is None:
                continue
            for vid in vehicle_dict.keys():
                veh_p = non_repo_veh_plans[vid]
                veh_obj = self.veh_objs[vid]
                if len(veh_p.get_involved_request_ids()) <= self._max_prqs_exhaustive_DARP:
                    new_veh_p, cost = solve_single_vehicle_DARP_exhaustive(veh_obj, self.routing_engine, 
                                                                                [prq] + [self.active_requests[rid] for rid in veh_p.get_involved_request_ids()], 
                                                                                self.fleetcontrol, sim_time, veh_p)
                else:
                    r_list = insert_prq_in_selected_veh_list([veh_obj], {veh_obj.vid:veh_p}, prq, self.fleetcontrol.vr_ctrl_f, self.routing_engine, self.active_requests, sim_time, self.fleetcontrol.const_bt, self.fleetcontrol.add_bt)
                    if len(r_list) != 0:
                        _, new_veh_p, _ = min(r_list, key=lambda x:x[2])
                        cost = self.fleetcontrol.compute_VehiclePlan_utility(sim_time, veh_obj, new_veh_p)
                    else:
                        new_veh_p, cost = None, None
                if new_veh_p is not None:
                    cost_change = cost - veh_p.get_utility()
                    if cost_change < 0:
                        try:
                            rid_to_vid_to_cost_and_plan[rid][vid] = (cost_change, new_veh_p)
                        except KeyError:
                            rid_to_vid_to_cost_and_plan[rid] = {vid : (cost_change, new_veh_p)}                
        t_insertion = time.time() - t
        
        #3) solve assignment problem
        t = time.time()
        rid_to_vid_assignment = self._solve_assignment_problem_gurobi(sim_time, rid_to_vid_to_cost_and_plan)
        t_solve = time.time() - t
        
        times = {"sim_time" : self.sim_time, "rv" : t_rv, "build" : t_insertion, "opt" : t_solve, "all" : t_rv + t_insertion + t_solve}
        time_str = ",".join(["{};{}".format(a, b) for a, b in times.items()])
        LOG.info("OPT TIMES:{}".format(time_str))
        
        LOG.debug(f"rid to vid assignment: {rid_to_vid_assignment}")
        
        #4) assign vehicles and create offers
        for rid, vid in rid_to_vid_assignment.items():
            veh_plan = rid_to_vid_to_cost_and_plan[rid][vid][1]
            self._optimisation_solutions[vid] = veh_plan
            
        self.unassigned_requests = {} # only try once
        
        sum_obj = 0
        for vid, veh_plan in self.fleetcontrol.veh_plans.items():
            if self._optimisation_solutions.get(vid) is not None:
                sum_obj += self.fleetcontrol.compute_VehiclePlan_utility(sim_time, self.veh_objs[vid], self._optimisation_solutions[vid])
                #LOG.debug(f"assign to vid {vid} -> requests {self._optimisation_solutions[vid].get_involved_request_ids()} -> obj {self.fleetcontrol.compute_VehiclePlan_utility(sim_time, self.veh_objs[vid], self._optimisation_solutions[vid])} | {self._optimisation_solutions[vid]}")
            else:
                sum_obj += self.fleetcontrol.compute_VehiclePlan_utility(sim_time, self.veh_objs[vid], veh_plan)
        LOG.info(f"Objective value at time {sim_time} for Simonetto: {sum_obj}")
            
    
    def get_optimisation_solution(self, vid : int) -> VehiclePlan:
        """ returns optimisation solution for vid
        :param vid: vehicle id
        :return: vehicle plan object for the corresponding vehicle
        """
        plan = self._optimisation_solutions.get(vid)
        LOG.debug(f"plan for vid {vid} : {plan}")
        # LOG.debug("veh obj {}".format(self.veh_objs[vid]))
        #minimal_vehplan = VehiclePlan(self.veh_objs[vid], self.sim_time, self.routing_engine, self.veh_objs[vid].locked_planstops)
        if plan is None:
            return self.fleetcontrol.veh_plans[vid]
        else:
            return plan

    def set_assignment(self, vid : int, assigned_plan : VehiclePlan, is_external_vehicle_plan : bool = False, _is_init_sol=True):
        """ sets the vehicleplan as assigned in the algorithm database; if the plan is not computed within the this algorithm, the is_external_vehicle_plan flag should be set to true
        :param vid: vehicle id
        :param assigned_plan: vehicle plan object that has been assigned
        :param is_external_vehicle_plan: should be set to True, if the assigned_plan has not been computed within this algorithm
        """
        super().set_assignment(vid, assigned_plan, is_external_vehicle_plan=is_external_vehicle_plan)

    def get_current_assignment(self, vid : int) -> VehiclePlan: # TODO same as get_optimisation_solution (delete?)
        """ returns the vehicle plan assigned to vid currently
        :param vid: vehicle id
        :return: vehicle plan
        """
        return self.fleetcontrol.veh_plans[vid]
    
    def clear_databases(self):
        self._optimisation_solutions = {}
        self.unassigned_requests = {}
        return super().clear_databases()
        
    def _computeRV(self, rids_to_compute):
        """ this function computes all rv-connections from self.requests_to_compute with all active vehicles
        """
        veh_locations_to_vid = {}

        for vid, veh_obj in self.veh_objs.items():
            try:
                veh_locations_to_vid[veh_obj.pos].append(vid)
            except:
                veh_locations_to_vid[veh_obj.pos] = [vid]
        current_time = self.sim_time

        r2v = {}
        for rid in rids_to_compute:
            r2v[rid] = {}
            prq = self.active_requests[rid]
            o_pos, _, latest_pu = prq.get_o_stop_info()
            routing_results = self.routing_engine.return_travel_costs_Xto1(veh_locations_to_vid.keys(), o_pos,
                                                                            max_cost_value=latest_pu - current_time)
            for veh_loc, tt, _, _ in routing_results:
                for vid in veh_locations_to_vid[veh_loc]:
                    r2v[rid][vid] = tt

        return r2v
    
    def _solve_assignment_problem_gurobi(self, sim_time, rid_to_vid_to_cost_and_plan):
        import gurobipy as gurobi
        
        model_name = f"SimonettoAssignment: _solve_gurobi {sim_time}"
        with gurobi.Env(empty=True) as env:
            if self.fleetcontrol.log_gurobi:
                with open(os.path.join(self.fleetcontrol.dir_names[G_DIR_OUTPUT], "gurobi_log.log"), "a") as f:
                    f.write(f"\n\n{model_name}\n\n")
                env.setParam('OutputFlag', 1)
                env.setParam('LogToConsole', 0)
                env.setParam('LogFile', os.path.join(self.fleetcontrol.dir_names[G_DIR_OUTPUT], "gurobi_log.log") )
                env.start()
            else:
                env.setParam('OutputFlag', 0)
                env.setParam('LogToConsole', 0)
                env.start()

            m = gurobi.Model(model_name, env = env)
            
            m.setParam(gurobi.GRB.param.Threads, self.fleetcontrol.n_cpu)
            m.setParam('TimeLimit', TIME_OUT)
           
            vars = {}
            vid_constr_dict = {}    # vid -> list var
            rid_constr_dict = {}    # pos_key -> list var
            for rid, vid_dict in rid_to_vid_to_cost_and_plan.items():
                for vid, (cost, _) in vid_dict.items():
                    name = f"{rid}_{vid}"
                    var = m.addVar(name = name, obj = cost, vtype = gurobi.GRB.BINARY)
                    vars[name] = var
                    try:
                        vid_constr_dict[vid].append(var)
                    except KeyError:
                        vid_constr_dict[vid] = [var]
                    try:
                        rid_constr_dict[rid].append(var)
                    except KeyError:
                        rid_constr_dict[rid] = [var]

            for rid, var_list in rid_constr_dict.items():
                m.addConstr(sum(var_list) <= 1, name=f"c_rid_{rid}")
            for vid, var_list in vid_constr_dict.items():
                m.addConstr(sum(var_list) <= 1, name=f"c_vid_{vid}")
                
            if WRITE_PROBLEM:
                m.write(os.path.join(self.fleetcontrol.dir_names[G_DIR_OUTPUT], f"opt_problem_matching_{sim_time}.lp"))
                    
            m.optimize() #optimization
            
            #get solution
            if m.status == gurobi.GRB.Status.OPTIMAL:
                varnames = m.getAttr("VarName", m.getVars())
                solution = m.getAttr("X",m.getVars())
                
                new_assignments = {}
                for x in range(len(solution)):
                    if round(solution[x]) == 1:
                        key = varnames[x].split("_")
                        rid, vid = int(key[0]), int(key[1])
                        new_assignments[rid] = vid
            else:
                model_f = os.path.join(self.fleetcontrol.dir_names[G_DIR_OUTPUT], f"70_repo_am_opt_model_infeasible_{sim_time}.lp")
                m.write(model_f)
                LOG.error(f"Operator {self.fleetcontrol.op_id}: No Optimal Solution! status {m.status}"
                            f" -> no assignment")
                new_assignments = {}

            return new_assignments
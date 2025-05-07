from abc import abstractmethod
import os
import sys
import numpy as np
import pandas as pd
import time
from src.simulation.Offers import TravellerOffer
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.pooling.immediate.insertion import simple_remove
from src.routing.NetworkBase import NetworkBase
from src.fleetctrl.planning.VehiclePlan import PlanStopBase, RoutingTargetPlanStop
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.fleetctrl.reservation.RevelationHorizonBase import RevelationHorizonBase
from src.fleetctrl.reservation.misc.RequestGroup import RequestGroup, QuasiVehiclePlan, QuasiVehicle, VehiclePlanSupportingPoint, rg_key, get_lower_keys
from src.fleetctrl.reservation.ReservationRequestBatch import ReservationRequestBatch, merge_reservationRequestBatches
from src.misc.globals import *

from typing import Any, Dict, List, Tuple

import logging
LOG = logging.getLogger(__name__)

NEW_VID_PENALTY = 1000000   # penalty for introducing a new vehicle in case a match between batches is not possible
TIME_WINDOW_VIOLATION_PENALTY = 1000 # penalty for violating time constraints (€ per s)

INPUT_PARAMETERS_BatchSchedulingRevelationHorizonBase = {
    "doc" :     """         this class is used as base class for reservation algorithms that
        - offline compute a solution for reservation requests first, decoupled from the online state and
            infuses the solution into online solution (defined by RevelationHorizonBase)
        - assigns reservation requests to batches of size "op_res_batch_size"
        - computes possible VehiclePlans to serve requests within theses schedules
        - sorts these batches based on their time intervals
        - schedules theses vehicleplans one after another by solving an optimization problem to assign them
        
    subclasses of this baseclass should implment different ways of batching requests
    """,
    "inherit" : "RevelationHorizonBase",
    "input_parameters_mandatory": [G_RA_MAX_BATCH_SIZE, G_RA_MAX_BATCH_CONCAT],
    "input_parameters_optional": [G_RA_OP_NW_TYPE],
    "mandatory_modules": [],
    "optional_modules": []
}

class BatchSchedulingRevelationHorizonBase(RevelationHorizonBase):
    """ 
    this class is used as base class for reservation algorithms that
        - offline compute a solution for reservation requests first, decoupled from the online state and
            infuses the solution into online solution (defined by RevelationHorizonBase)
        - assigns reservation requests to batches of size "op_res_batch_size"
        - computes possible VehiclePlans to serve requests within theses schedules
        - sorts these batches based on their time intervals
        - schedules theses vehicleplans one after another by solving an optimization problem to assign them
        
    subclasses of this baseclass should implment different ways of batching requests
    """
    def __init__(self, fleetctrl : FleetControlBase, operator_attributes : dict, dir_names : dict, solver : str="Gurobi"):
        super().__init__(fleetctrl, operator_attributes, dir_names, solver=solver)

        self.solver = solver
        self.max_batch_size = int(operator_attributes[G_RA_MAX_BATCH_SIZE])
        self.N_batch_concat = int(operator_attributes.get(G_RA_MAX_BATCH_CONCAT, 1))
        
        self._use_own_routing_engine = False
        if operator_attributes.get(G_RA_OP_NW_TYPE):
            LOG.info(f"operator {self.fleetctrl.op_id} loads its own network for reservation!")
            if not operator_attributes.get(G_RA_OP_NW_NAME):
                raise IOError(f"parameter {G_RA_OP_NW_NAME} has to be given to load a network for operator {self.op_id}")
            from src.misc.init_modules import load_routing_engine
            self.routing_engine : NetworkBase = load_routing_engine(operator_attributes[G_RA_OP_NW_TYPE], os.path.join(dir_names[G_DIR_DATA], "networks", operator_attributes[G_RA_OP_NW_NAME]),
                                                      network_dynamics_file_name=operator_attributes.get(G_RA_OP_NW_DYN_F))
            self._use_own_routing_engine = True
        
    @abstractmethod
    def _create_sorted_request_batches(self, sim_time : int) -> List[ReservationRequestBatch]:
        """ this class uses the currently active reservation request to create batches and returns a sorted list of them.
        vehicleplans of this sorted batches will be scheduled one after another
        :param sim_time: current simulation time
        :return: list of ReservationRequestBatch"""
        raise NotImplementedError("This abstract method is not overwritten in its child class!")
        
    def _batch_optimisation(self, sim_time):
        LOG.debug("reservation fwd batch optimization!")
        if len(self._unprocessed_rids) != 0:
            # batch the requests
            sorted_rg_batches = self._create_sorted_request_batches(sim_time)
            # solve optimization problem to assign and connect schedules
            self._multi_forward_batch_optimization(sim_time, sorted_rg_batches, self.N_batch_concat)
            #exit()
            self._unprocessed_rids = {}
            
    def _multi_forward_batch_optimization(self, sim_time, batch_rg_list : List[ReservationRequestBatch], N_batch_concat):
        """ this function iterates through the list of batches and matches them together by solving a
        maximum priority matching problem. the expected start time of a request group has to exceed the expected end_time 
        of the former request group. as initial condition the vehicles and their currently assigned plans are used
        :param sim_time: current simulation time
        :param batch_rq_list: sorted list of request batches (only neighbouring batches are directly matched together)"""
        plan_id_to_part_best_plan : Dict[int, List[QuasiVehiclePlan]] = {}  # dict offline plan id to an ordered list of vehicle plans of the resulting assigned request groups
        # 1 to 1 ids after initial optimization
        plan_id_batch_constraints, current_assignment_horizon = self._get_vid_batch_constraints(sim_time)
        allready_assigned_rids = {} # rids that are assigned during the process are added here
                
        LOG.info("reoptimize reservation schedules completely")
        # batch optimization
        
        # get previous solutions
        assigned_subplans = self._get_assigned_subplans_from_off_plans()
        LOG.debug(f"prev assigned schedules: {assigned_subplans}")
        
        N_current_rids = 0
        current_rg_objectives = {}
        current_rg_constraints = {}
        current_assignment_rgs_batch = {}
        current_rids = {}
        last_routing_time = sim_time
        for i in range(len(batch_rg_list)):
            LOG.info(" ... process batch {}/{} with times {}".format(i, len(batch_rg_list), batch_rg_list[i].get_batch_time_window()))
            if self._use_own_routing_engine:
                batch_start_time = batch_rg_list[i].get_batch_time_window()[0]
                if batch_start_time is not None:
                    for t in range(last_routing_time, batch_start_time):
                        self.routing_engine.update_network(t)
                    last_routing_time = batch_start_time
            
            current_batch = batch_rg_list[i]
            rg_objectives, rg_constraints = current_batch.get_rg_obj_const()
            current_rg_objectives.update(rg_objectives)
            current_rg_constraints.update(rg_constraints)
            current_rids.update(current_batch.requests_in_batch)
            for rg in rg_constraints.keys():
                current_assignment_rgs_batch[rg] = i
            N_current_rids += current_batch.get_batch_size()
            if i+1 == len(batch_rg_list) or N_current_rids + batch_rg_list[i+1].get_batch_size() > self.max_batch_size:
                # new optimization
                LOG.info("start optimization with {} requests and {} schedules".format(N_current_rids, len(current_rg_constraints)))
                # add future batches to consider in optimization
                if i + 1 < len(batch_rg_list):
                    for j in range(i+1, len(batch_rg_list)):
                        if j  == len(batch_rg_list) or N_current_rids + batch_rg_list[j].get_batch_size() > self.max_batch_size * N_batch_concat:
                            LOG.info(" -> while considering {} requests and {} schedules".format(N_current_rids, len(current_rg_constraints)))
                            break
                        LOG.debug(" add batch with tw {}".format(batch_rg_list[j].get_batch_time_window()))
                        follow_batch = batch_rg_list[j]
                        rg_objectives, rg_constraints = follow_batch.get_rg_obj_const()
                        current_rg_objectives.update(rg_objectives)
                        current_rg_constraints.update(rg_constraints)
                        current_rids.update(follow_batch.requests_in_batch)
                        N_current_rids += follow_batch.get_batch_size()
                        for rg in rg_objectives.keys():
                            current_assignment_rgs_batch[rg] = j
                
                # define additional constraint to maintain feasibility also if some trips become infeasible i.e. due to changing travel times
                feasible_assignment_needed = {}
                rids_to_enforce = []
                for former_plan_id, list_vehplans in assigned_subplans.items():
                    assigned_rids = {}
                    first_plan = None
                    for vehplan in list_vehplans:
                        in_sup_batch = False
                        in_current_batch = False
                        rids_in_vehplan = {}
                        for ps in vehplan.list_plan_stops:
                            for rid in ps.get_list_boarding_rids():
                                rids_in_vehplan[rid] = 1
                            for rid in ps.get_list_alighting_rids():
                                rids_in_vehplan[rid] = 1
                        for rid in rids_in_vehplan.keys():
                            if current_batch.requests_in_batch.get(rid):
                                in_sup_batch = True
                                in_current_batch = True
                                LOG.debug(f"rid {rid} in current batch")
                            elif current_rids.get(rid):
                                in_sup_batch = True
                        if in_sup_batch:
                            LOG.debug(f" form {former_plan_id} in sup batch {in_sup_batch} in current batch {in_current_batch}: extend rids {rids_in_vehplan} : {vehplan}")
                            assigned_rids.update(rids_in_vehplan)
                            allready_assigned = []
                            for rid in rids_in_vehplan.keys():
                                if allready_assigned_rids.get(rid):
                                    allready_assigned.append(rid)
                            LOG.debug(f"allready assigned: {allready_assigned}")
                            if len(rids_in_vehplan) == len(allready_assigned): # doesnt need to be considered anymore
                                for rid in allready_assigned:
                                    del assigned_rids[rid]
                                continue
                            elif len(allready_assigned) != 0:
                                for rid in allready_assigned:
                                    LOG.debug(f"update plan for already assigned request {rid}")
                                    del assigned_rids[rid]
                                    o_vehplan = simple_remove(QuasiVehicle(vehplan.list_plan_stops[0].get_pos(), capacity=self.vehicle_capacity), vehplan, rid, sim_time, self.routing_engine, self.fleetctrl.vr_ctrl_f, self.active_reservation_requests, self.fleetctrl.const_bt, self.fleetctrl.add_bt)
                                    vehplan = QuasiVehiclePlan(self.routing_engine, o_vehplan.list_plan_stops, self.vehicle_capacity)
                            if first_plan is None:
                                first_plan = vehplan.copy()
                            elif in_current_batch:
                                first_plan.list_plan_stops += vehplan.list_plan_stops[:]
                    if first_plan is not None:
                        first_plan.update_tt_and_check_plan(QuasiVehicle(first_plan.list_plan_stops[0].get_pos(), capacity=self.vehicle_capacity), sim_time, self.routing_engine, keep_feasible=True)
                        first_plan.compute_obj_function(self.fleetctrl.vr_ctrl_f, self.routing_engine, self.active_reservation_requests)
                        feasible_assignment_needed[rg_key(assigned_rids.keys())] = first_plan
                        LOG.debug(f"add {rg_key(assigned_rids.keys())} : {first_plan}")
                        rids_to_enforce += list(assigned_rids.keys())

                        
                connections_to_add = self._create_forced_init_connections(sim_time, plan_id_batch_constraints, {rg_k : vp.get_start_end_constraints() for rg_k, vp in feasible_assignment_needed.items()})
                LOG.debug("connections to add: {}".format(connections_to_add))
                LOG.debug(f"rids in current batch: {current_batch.requests_in_batch.keys()}")
                LOG.debug(f"all considered rids: {current_rids.keys()}")
                #LOG.debug(f"unassigned reservation rids: {self._unprocessed_rids.keys()}")
                LOG.debug(f"rids to enforce {rids_to_enforce}")
                   
                # solve matching problem to schedule trips between batches 
                # LOG.debug(f"=========== solve next")
                # current_frontier = sorted([x[1] for x in plan_id_batch_constraints.values()])
                # current_targets = sorted([x[1] for x in current_rg_constraints.values()])
                # LOG.debug(f"start time constraints: {min(current_frontier)} {max(current_frontier)} | {current_frontier}")
                # LOG.debug(f"target time constraints: {min(current_targets)} {max(current_targets)} | {current_targets}")
                # LOG.debug(f"number vehicles with smaller frontier: {len([x for x in current_frontier if x < min(current_targets)])}")
                       
                plan_id_to_assigned_rgs = self._match_batch_rg_graph_to_start_constraints(current_rg_objectives, current_rg_constraints, 
                                                                                          plan_id_batch_constraints, current_assignment_rgs_batch,
                                                                                          connections_to_add, i)
                assigned_rids = {}
                for plan_id, assigned_rgs in plan_id_to_assigned_rgs.items():
                    for rg in assigned_rgs:
                        batch_index = current_assignment_rgs_batch.get(rg)
                        forced_assignment = False
                        if batch_index is None:
                            LOG.debug(f"could find {rg} in current batches -> forced assignment?")
                            if feasible_assignment_needed.get(rg):
                                LOG.debug(f" -> yes")
                                forced_assignment = True
                                # remove rids from following batches
                                rids_in_rg = feasible_assignment_needed[rg].get_involved_request_ids()
                                for k in range(i, len(batch_rg_list)):
                                    for rid in rids_in_rg:
                                        if batch_rg_list[k].requests_in_batch.get(rid):
                                            LOG.debug(f"remove {rid} from batch {k}")
                                            batch_rg_list[k].delete_request_from_batch(rid)
                        if not forced_assignment and (batch_index is None or batch_index > i):
                            break
                        if not forced_assignment:
                            best_plan = batch_rg_list[batch_index].get_best_plan_of_rg(rg, self.fleetctrl, self.routing_engine)
                            for rid in rg:
                                if allready_assigned_rids.get(rid):
                                    LOG.error(f"rid {rid} has already been assigned in a previous batch!")
                                    raise EnvironmentError(f"rid {rid} has already been assigned in a previous batch!")
                                allready_assigned_rids[rid] = 1
                        else:
                            best_plan = feasible_assignment_needed[rg]
                            for rid in best_plan.get_involved_request_ids():
                                if allready_assigned_rids.get(rid):
                                    LOG.error(f"rid {rid} has already been assigned in a previous batch!")
                                    raise EnvironmentError(f"rid {rid} has already been assigned in a previous batch!")
                                allready_assigned_rids[rid] = 1
                        _, _, end_pos, end_time = best_plan.get_start_end_constraints()
                        plan_id_batch_constraints[plan_id] = (end_pos, end_time)
                        try:
                            plan_id_to_part_best_plan[plan_id].append(best_plan)
                        except KeyError:
                            plan_id_to_part_best_plan[plan_id] = [best_plan]
                        for rid in rg:
                            assigned_rids[rid] = 1
                            
                LOG.debug(f"-> {len(assigned_rids)}/{len(current_batch.requests_in_batch)} assigned by reservation module")
                N_current_rids = 0
                current_rids = {}
                current_rg_objectives = {}
                current_rg_constraints = {}
                current_assignment_rgs_batch = {}
                                       
        # create full offline plans
        self._plan_id_to_off_plan = {}
        for plan_id, list_batch_plans in plan_id_to_part_best_plan.items():
            full_off_list_ps = []
            for plan in list_batch_plans:
                full_off_list_ps += plan.list_plan_stops
            self._plan_id_to_off_plan[plan_id] = QuasiVehiclePlan(self.routing_engine, full_off_list_ps, self.vehicle_capacity)
            
        if self._use_own_routing_engine:
            self.routing_engine.reset_network(sim_time)

        
    def _match_batch_rg_graph_to_start_constraints(self, batch_rg_objectives : Dict[Any, float], batch_rg_constraints : Dict[Any, Tuple[float, float, float, float]],
                                                   plan_id_batch_constraints : Dict[Any, Tuple[tuple, float]], current_assignment_rgs_batch : Dict[Any, int],
                                                   connections_to_add : List[Tuple[int, tuple, float, float]], c_batch_index) -> Dict[Any, List[Any]]:
        """ TODO
        :param batch_rg_objectives: dict rg key -> objective value
        :param batch_rg_constraints: dict rg key -> tuple of (start_pos, start_time, end_pos, end_time) of the plan
        :param plan_id_batch_constraints: dict vehicle id -> (end_pos, end_time) 
        :param current_assignment_rgs_batch: dict rg-key -> batch index
        :param connections_to_add: list of (init_plan_id, rg_key, objectve, time_window_violation) for connections that should be feasible
        :return: dict hypothetical vehicle id -> list of assigned request group ids"""
        if self.solver == "Gurobi":
            return self._match_batch_rg_graph_to_start_constraints_gurobi(batch_rg_objectives, batch_rg_constraints, plan_id_batch_constraints, current_assignment_rgs_batch, connections_to_add, c_batch_index)
        else:
            raise EnvironmentError(f"Solver {self.solver} not implemented for this class!")
        
    def _match_batch_rg_graph_to_start_constraints_gurobi(self, batch_rg_objectives : Dict[Any, float], batch_rg_constraints : Dict[Any, Tuple[float, float, float, float]],
                                                   plan_id_batch_constraints : Dict[Any, Tuple[tuple, float]], current_assignment_rgs_batch : Dict[Any, int],
                                                   connections_to_add : List[Tuple[int, tuple, float, float]], c_batch_index) -> Dict[Any, List[Any]]:
        """ TODO
        :param batch_rg_objectives: dict rg key -> objective value
        :param batch_rg_constraints: dict rg key -> tuple of (start_pos, start_time, end_pos, end_time) of the plan
        :param plan_id_batch_constraints: dict vehicle id -> (end_pos, end_time) 
        :param current_assignment_rgs_batch: dict rg-key -> batch index
        :param connections_to_add: list of (init_plan_id, rg_key, objectve, time_window_violation) for connections that should be feasible
        :return: dict hypothetical vehicle id -> list of assigned request group ids"""
        
        import gurobipy as gurobi
        model_name = f"BatchSchedulingRevelationHorizonBase: _match_batch_rg_graph_to_start_constraints_gurobi {c_batch_index} {self.fleetctrl.sim_time}"
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

            variables = {}  # var_key -> gurobi variable
            incoming_constr = {} # node -> var_key -> 1
            outgoing_constr = {}    # node -> var_key -> 1
            nodes = {}  # node -> 1
            rid_constr = {}  # rid -> var_key -> 1 
            
            last_end = {f"v_{vid}" : end_pos_end_time for vid, end_pos_end_time in plan_id_batch_constraints.items()}
            vids = list(last_end.keys())
            var_c = 0
            LOG.debug("optimizing batches")
            sorted_batch_rgs = sorted( (k for k in batch_rg_constraints.keys()), key=lambda x:batch_rg_constraints[x][1] )
            #LOG.debug("sorted batch rgs : {}".format([(rg, batch_rg_constraints[rg][1]) for rg in sorted_batch_rgs]))
            for i, rg in enumerate(sorted_batch_rgs):
                start_pos, start_time, end_pos, end_time = batch_rg_constraints[rg]
                rg_obj = batch_rg_objectives[rg]
                batch_index = current_assignment_rgs_batch[rg]
                
                # with start nodes
                for l_key, f_end_pos_end_time in last_end.items():
                    f_end_pos, f_end_time = f_end_pos_end_time
                    if f_end_time <= start_time:
                        _, tt, _ = self.routing_engine.return_travel_costs_1to1(f_end_pos, start_pos)
                        if tt <= start_time - f_end_time:
                            # Define Variable and Cost
                            # qvp = QuasiVehiclePlan(self.routing_engine, [RoutingTargetPlanStop(f_end_pos), RoutingTargetPlanStop(start_pos)], self.vehicle_capacity)
                            # cfv = qvp.compute_obj_function(self.fleetctrl.vr_ctrl_f, self.routing_engine, self.active_reservation_requests)
                            cfv = self.driving_leg_obj(None, self.routing_engine, f_end_pos, start_pos, f_end_time, start_time)
                            var = m.addVar(name = "{}_{}".format(l_key, rg), obj = rg_obj + cfv, vtype = gurobi.GRB.BINARY)
                            # LOG.debug(f"var {'{}_{}'.format(l_key, rg)} : {obj + cfv}")
                            var_key = (l_key, rg)
                            variables[var_key] = var
                            var_c += 1
                            # add nodes
                            nodes[l_key] = 1
                            nodes[rg] = 1
                            # add incoming constraint
                            try:
                                incoming_constr[rg][var_key] = 1
                            except KeyError:
                                incoming_constr[rg] = {var_key : 1}
                            # add outgoing constraint
                            try:
                                outgoing_constr[l_key][var_key] = 1
                            except KeyError:
                                outgoing_constr[l_key] = {var_key : 1}
                            # add rid constraint
                            for rid in rg:
                                try:
                                    rid_constr[rid][var_key] = 1
                                except KeyError:
                                    rid_constr[rid] = {var_key : 1}
                # with other rgs
                if i > 0:
                    for j in range(i):
                        f_rg = sorted_batch_rgs[j]
                        f_batch_index = current_assignment_rgs_batch[f_rg]
                        if batch_index < f_batch_index:
                            LOG.debug(f"forbid connection between {f_rg} and {rg} with batch indices {f_batch_index} and {batch_index}")
                            continue
                        l_key = f_rg
                        _, _, f_end_pos, f_end_time = batch_rg_constraints[f_rg]
                        #LOG.debug(f"{j} -> {i} : {f_end_pos} {f_end_time} -> {start_pos} {start_time} | {f_rg} -> {rg}")
                        if f_end_time <= start_time:
                            _, tt, _ = self.routing_engine.return_travel_costs_1to1(f_end_pos, start_pos)
                            if tt <= start_time - f_end_time:
                                # Define Variable and Cost
                                # qvp = QuasiVehiclePlan(self.routing_engine, [RoutingTargetPlanStop(f_end_pos), RoutingTargetPlanStop(start_pos)], self.vehicle_capacity)
                                # cfv2 = qvp.compute_obj_function(self.fleetctrl.vr_ctrl_f, self.routing_engine, self.active_reservation_requests)
                                cfv = self.driving_leg_obj(None, self.routing_engine, f_end_pos, start_pos, f_end_time, start_time)
                                #LOG.debug(f" -> cfv {cfv} {cfv2} | {rg_obj + cfv}")
                                var = m.addVar(name = "{}_{}".format(l_key, rg), obj = rg_obj + cfv, vtype = gurobi.GRB.BINARY)
                                # LOG.debug(f"var {'{}_{}'.format(l_key, rg)} : {obj + cfv}")
                                var_key = (l_key, rg)
                                variables[var_key] = var
                                var_c += 1
                                # add nodes
                                nodes[l_key] = 1
                                nodes[rg] = 1
                                # add incoming constraint
                                try:
                                    incoming_constr[rg][var_key] = 1
                                except KeyError:
                                    incoming_constr[rg] = {var_key : 1}
                                # add outgoing constraint
                                try:
                                    outgoing_constr[l_key][var_key] = 1
                                except KeyError:
                                    outgoing_constr[l_key] = {var_key : 1}
                                # add rid constraint
                                for rid in rg:
                                    try:
                                        rid_constr[rid][var_key] = 1
                                    except KeyError:
                                        rid_constr[rid] = {var_key : 1}
                                        
            # enforce feasibility
            for vid, rg, obj, _ in connections_to_add:
                l_key = f"v_{vid}"
                var_key = (l_key, rg)
                if variables.get(var_key):
                    continue
                var = m.addVar(name = "{}_{}".format(l_key, rg), obj = obj, vtype = gurobi.GRB.BINARY)
                variables[var_key] = var
                var_c += 1
                # add nodes
                nodes[l_key] = 1
                nodes[rg] = 1
                # add incoming constraint
                try:
                    incoming_constr[rg][var_key] = 1
                except KeyError:
                    incoming_constr[rg] = {var_key : 1}
                # add outgoing constraint
                try:
                    outgoing_constr[l_key][var_key] = 1
                except KeyError:
                    outgoing_constr[l_key] = {var_key : 1}
                # add rid constraint
                for rid in rg:
                    try:
                        rid_constr[rid][var_key] = 1
                    except KeyError:
                        rid_constr[rid] = {var_key : 1}
                                        
            # base node and connections to vehicles
            base_name = "base"
            # base node to vid and vid to base node
            for vid in vids:
                var = m.addVar(name = "{}_{}".format(base_name, vid), obj = 0, vtype = gurobi.GRB.BINARY)
                var_key = (base_name, vid)
                variables[var_key] = var
                var_c += 1
                # add incoming constraint
                try:
                    incoming_constr[vid][var_key] = 1
                except KeyError:
                    incoming_constr[vid] = {var_key : 1}
                # add outgoing constraint
                try:
                    outgoing_constr[base_name][var_key] = 1
                except KeyError:
                    outgoing_constr[base_name] = {var_key : 1}
                var = m.addVar(name = "{}_{}".format(vid, base_name), obj = 0, vtype = gurobi.GRB.BINARY)
                var_key = (vid, base_name)
                variables[var_key] = var
                var_c += 1
                # add incoming constraint
                try:
                    incoming_constr[base_name][var_key] = 1
                except KeyError:
                    incoming_constr[base_name] = {var_key : 1}
                # add outgoing constraint
                try:
                    outgoing_constr[vid][var_key] = 1
                except KeyError:
                    outgoing_constr[vid] = {var_key : 1}
            # all nodes to base node
            for l_key in nodes.keys():
                if type(l_key) == str and l_key.startswith("v"):
                    continue
                var = m.addVar(name = "{}_{}".format(l_key, base_name), obj = 0, vtype = gurobi.GRB.BINARY)
                var_key = (l_key, base_name)
                variables[var_key] = var
                var_c += 1
                # add incoming constraint
                try:
                    incoming_constr[base_name][var_key] = 1
                except KeyError:
                    incoming_constr[base_name] = {var_key : 1}
                # add outgoing constraint
                try:
                    outgoing_constr[l_key][var_key] = 1
                except KeyError:
                    outgoing_constr[l_key] = {var_key : 1}
            nodes[base_name] = 1
                        
            #define constraints
            
            #1) incoming constraints
            for node, var_dict in incoming_constr.items():
                if type(node) == str and node == base_name:
                    m.addConstr(sum(variables[x] for x in var_dict.keys()) == len(vids), name=f"in {node}")
                else:
                    m.addConstr(sum(variables[x] for x in var_dict.keys()) <= 1, name=f"in {node}")
            #2) outgoing constraints
            for node, var_dict in outgoing_constr.items():
                if type(node) == str and node == base_name:
                    m.addConstr(sum(variables[x] for x in var_dict.keys()) == len(vids), name=f"out {node}")
                else:
                    m.addConstr(sum(variables[x] for x in var_dict.keys()) <= 1, name=f"out {node}")
            #3) rid constraints
            for rid, var_dict in rid_constr.items():
                if self._unprocessed_rids.get(rid):
                    m.addConstr(sum(variables[x] for x in var_dict.keys()) <= 1, name=f"rid {rid}")
                else:
                    m.addConstr(sum(variables[x] for x in var_dict.keys()) == 1, name=f"rid {rid}")
            #4) flow constraints
            for node in nodes.keys():
                m.addConstr(sum(variables[x] for x in incoming_constr.get(node, {}).keys()) - sum(variables[x] for x in outgoing_constr.get(node, {}).keys()) == 0, name = f"flow {node}" )
            
            # optimize
            LOG.info("number variables: {}".format(var_c)) 
            #m.write(r'C:\Users\ge37ser\Documents\Coding\TUM_VT_FleetSimulation\tum-vt-fleet-simulation\FleetPy\studies\journal_reservation\m2.lp')
            #m.write(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], f"res_batch_matching_{c_batch_index}.lp"))
            m.optimize()
            
            # retrieve solution
            try:
                vals = m.getAttr('X', variables)
            except gurobi.GurobiError:
                m.computeIIS()
                m.write(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], "ilp_res_batch_matching.ilp"))
                raise gurobi.GurobiError("model infeasible!")
            sols = {x : 1 for x, v in vals.items() if int(np.round(v)) == 1}
            LOG.debug("opt sols:")
            LOG.debug(f"{list(sols.keys())}")

            sol_graph = {}
            for s, e in sols.keys():
                if type(s) == str and s == base_name:
                    continue
                sol_graph[s] = e
            sol_schedules = []
            plan_id_to_assigned_rgs = {}
            n_assigned_rids = 0
            for vid in vids:
                schedule = []
                LOG.debug(f"vid {vid}")
                cur = vid
                while type(sol_graph[cur]) != str and sol_graph[cur] != base_name:
                    cur = sol_graph[cur]
                    n_assigned_rids += len(cur)
                    schedule.append(cur)
                LOG.debug(f" -> schedule : {schedule}")
                sol_schedules.append(schedule)
                plan_id = int(vid.split("_")[1])
                plan_id_to_assigned_rgs[plan_id] = schedule
                
            return plan_id_to_assigned_rgs
                
    def _create_forced_init_connections(self, simulation_time, init_end_constraints : Dict[int, Tuple[tuple, int]], feasible_assignments_to_force : Dict[int, Tuple[tuple, int, tuple, int]]):
        """ this method solves a maximum matching problem the create connections between batches that are needed to create
        a feasible optimisation problem for batch matching in case connections got infeasible due to changing travel times
        :param simulation_time: current simulation time
        :param init_end_constraints: dict plan_id -> (pos, end_time)
        :param feasible_assignments_to_force: dict rg_key -> (start_pos, start_time, end_pos, end_time)
        :return: TODO"""
        if self.solver == "Gurobi":
            return self._create_forced_init_connections_gurobi(simulation_time, init_end_constraints, feasible_assignments_to_force)
        else:
            raise EnvironmentError(f"Solver {self.solver} not implemented for this class!")
        
    def _create_forced_init_connections_gurobi(self, simulation_time, init_end_constraints : Dict[int, Tuple[tuple, int]], feasible_assignments_to_force : Dict[int, Tuple[tuple, int, tuple, int]]) -> List[Tuple[int, tuple, float, int]]:
        """ this method solves a maximum matching problem using gurobi the create connections between batches that are needed to create
        a feasible optimisation problem for batch matching in case connections got infeasible due to changing travel times
        :param simulation_time: current simulation time
        :param init_end_constraints: dict plan_id -> (pos, end_time)
        :param feasible_assignments_to_force: dict rg_key -> (start_pos, start_time, end_pos, end_time)
        :return: list of connections to add tuples with (plan_id, rg_key, objective with time window violation, time window violation [s]"""
        import gurobipy as gurobi
        model_name = f"BatchSchedulingRevelationHorizonBase: _create_forced_init_connections_gurobi {simulation_time}"
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
            
            variables = {}  # var_key -> gurobi variable
            goal_constr = {}  # goal_key -> init_key -> 1
            init_constr = {}  # init_key -> goal_key -> 1
            effective_obj_and_tv = {}   # goal_key -> (objective, time constraint violation in s) 
            
            for init_key, init_constraints in init_end_constraints.items():
                for goal_key, goal_constraints in feasible_assignments_to_force.items():
                    objective = self.driving_leg_obj(simulation_time, self.routing_engine, init_constraints[0], goal_constraints[0], init_constraints[1], goal_constraints[1])
                    _, tt, _ = self.routing_engine.return_travel_costs_1to1(init_constraints[0], goal_constraints[0])
                    tv = 0
                    if init_constraints[1] + tt > goal_constraints[1]: # contraint violated
                        tv = init_constraints[1] + tt - goal_constraints[1]
                        objective += tv * TIME_WINDOW_VIOLATION_PENALTY
                        #LOG.debug("tc violation {} -> {} -> {} : obj {}".format(init_key, goal_key, tv, objective))
                    var = m.addVar(name = "{}_{}".format(init_key, goal_key), obj = objective, vtype = gurobi.GRB.BINARY)
                    var_key = (init_key, goal_key)
                    variables[var_key] = var
                    effective_obj_and_tv[var_key] = (objective, tv)
                    try:
                        goal_constr[goal_key][var_key] = 1
                    except KeyError:
                        goal_constr[goal_key] = {var_key : 1}
                    try:
                        init_constr[init_key][var_key] = 1
                    except KeyError:
                        init_constr[init_key] = {var_key : 1}
                        
            #define gurobi model
            
            #1) incoming constraints
            for goal_key, var_dict in goal_constr.items():
                m.addConstr(sum(variables[x] for x in var_dict.keys()) == 1, name=f"goal {goal_key}")
            #2) outgoing constraints
            for init_key, var_dict in init_constr.items():
                m.addConstr(sum(variables[x] for x in var_dict.keys()) <= 1, name=f"init {init_key}")
                
            #m.write(r'C:\Users\ge37ser\Documents\Coding\TUM_VT_FleetSimulation\tum-vt-fleet-simulation\FleetPy\studies\journal_reservation\m.lp')
                
            m.optimize()
            
            vals = m.getAttr('X', variables)
            sols = {x : 1 for x, v in vals.items() if int(np.round(v)) == 1}
            LOG.debug(f"sols {sols}")
            
            to_return = []
            for var in sols.keys():
                effective_obj_and_tv_var = effective_obj_and_tv[var]
                to_return.append( (var[0], var[1], effective_obj_and_tv_var[0], effective_obj_and_tv_var[1]))
            return to_return
        
    def _get_assigned_subplans_from_off_plans(self) -> Dict[int, List[QuasiVehiclePlan]]:
        """ this method iterates through all assigned offline plans and creates sub-schedules for each request (i.e. the parts where the request is scheduled and 
        the vehicle doesnt not get empty).
        this method can be used to infuse the last assigned solution into the matching problem to enforce feasiblity in case of a re-optimisation
        :return: dict plan_id -> list of quasi-vehicle plans"""
        plan_id_to_list_subplans = {}
        for plan_id, off_plan in self._plan_id_to_off_plan.items():
            plan_id_to_list_subplans[plan_id] = []
            current_occ = 0
            last_start_time = -1
            current_involved_rids = []
            current_ordered_planstops = []
            for i, ps in enumerate(off_plan.list_plan_stops):
                if current_occ == 0 and len(current_ordered_planstops) > 0:
                    sub_plan = QuasiVehiclePlan(self.routing_engine, current_ordered_planstops, self.vehicle_capacity, earliest_start_time=last_start_time)
                    LOG.debug(f" add suplan to {plan_id} : {sub_plan}") 
                    plan_id_to_list_subplans[plan_id].append(sub_plan)                   
                    last_start_time = ps.get_planned_arrival_and_departure_time()[0]
                    current_ordered_planstops = []
                    current_involved_rids = []
                    preceeding_qvp = sub_plan
                    
                current_occ += ps.get_change_nr_pax()
                for rid in ps.get_list_boarding_rids():
                    current_involved_rids.append(rid)
                current_ordered_planstops.append(ps)    
                    
            if current_occ != 0:
                raise EnvironmentError("Occupancy is not zero at end of plan??? {}".format(self))
            if current_occ == 0 and len(current_ordered_planstops) > 0:
                sub_plan = QuasiVehiclePlan(self.routing_engine, current_ordered_planstops, self.vehicle_capacity, earliest_start_time=last_start_time)  
                LOG.debug(f" add suplan to {plan_id} : {sub_plan}")                   
                plan_id_to_list_subplans[plan_id].append(sub_plan)   
                last_start_time = ps.get_planned_arrival_and_departure_time()[0]
                current_ordered_planstops = []
                current_involved_rids = []
            
        return plan_id_to_list_subplans
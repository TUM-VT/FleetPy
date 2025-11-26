import numpy as np
import time
import pandas as pd

from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.fleetctrl.reservation.BatchSchedulingRevelationHorizonBase import BatchSchedulingRevelationHorizonBase
from src.fleetctrl.reservation.misc.RequestGroup import QuasiVehiclePlan
from src.fleetctrl.reservation.ReservationRequestBatch import ReservationRequestBatch
from src.demand.TravelerModels import BasicRequest
from src.fleetctrl.planning.PlanRequest import ArtificialPlanRequest
from src.fleetctrl.planning.VehiclePlan import PlanStop, VehiclePlan, RoutingTargetPlanStop, BoardingPlanStop
from src.fleetctrl.pooling.batch.BatchAssignmentAlgorithmBase import SimulationVehicleStruct
from src.misc.globals import *
from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraAssignment import AlonsoMoraAssignment
from src.fleetctrl.pooling.batch.Simonetto.SimonettoAssignment import SimonettoAssignment

if TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.routing.NetworkBase import NetworkBase

import logging
LOG = logging.getLogger(__name__)

NEW_VID_PENALTY = 1000000   # penalty for introducing a new vehicle in case a match between batches is not possible
LARGE_INT = 1000000
MAX_TIME_DIFF_IN_BATCH = 60*60  # maximum time difference between requests in a batch

INPUT_PARAMETERS_ContinuousBatchRevelationReservation = {
    "doc" :     """   every reservation batch assignment, this algorithm assigns new reservation requests. 
    it first batches them based on overlapping earliest and latest pick-up times. 
    then it sorts the batches.
    for assignment, the current vehicle plans are progressed until the first request of the batch.
    it then treats the incoming requests similar to on-demand requests and solves the assignement problem(s).
    the corresponding assignment are used as long-term solutions for the vehicles.
    """,
    "inherit" : "BatchSchedulingRevelationHorizonBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [G_RA_REOPT_TS],
    "mandatory_modules": [],
    "optional_modules": []
}

def move_vehicle_according_to_plan(veh: SimulationVehicleStruct, veh_plan: VehiclePlan, t: int, time_step: int, routing_engine,
                                   rq_dict: Dict[int, ArtificialPlanRequest], vid_to_current_route, begin_approach_buffer_time):
    """ this method moves vehicles according to their currently assigned plan
    :param veh: vehicle object
    :param veh_plan: vehicle plan object
    :param t: current simulation time
    :param time_step: time step for moving
    :param routing_engine: routing engine
    :param rq_dict: dictionary of requests
    :param vid_to_current_route: dictionary of vehicle id -> current route
    :return: list of passed plan stops, vid_to_current_route (updated)"""
    passed_ps = []
    if len(veh_plan.list_plan_stops) > 0:
        #LOG.debug(f"move {veh.vid} at {t} - {t + time_step} pos {veh.pos}, {veh_plan}")
        cur_pos = veh.pos
        next_t = t + time_step
        last_t = t
        ps_ind = 0
        for i, ps in enumerate(veh_plan.list_plan_stops):
            arr, dep = ps.get_planned_arrival_and_departure_time()
            arr = max(arr, ps.get_earliest_start_time())
            #LOG.debug(f"arr dep {arr} {dep}")
            if next_t < arr:    # move along route
                target_pos = ps.get_pos()
                cur_pos = veh.pos
                if cur_pos != target_pos:
                    if ps.get_earliest_start_time() - next_t  > begin_approach_buffer_time and \
                        ps.get_earliest_start_time() - next_t - routing_engine.return_travel_costs_1to1(cur_pos, target_pos)[1] > begin_approach_buffer_time:
                        #LOG.debug(f"wait {veh.vid} at {cur_pos} for {target_pos} {ps.get_earliest_start_time()} - {next_t} > {begin_approach_buffer_time}")
                        pass
                    else:
                        route = vid_to_current_route.get(veh.vid, [])
                        #LOG.debug(f"old route {cur_pos} {route}")
                        if len(route) == 0 or route[-1] != target_pos[0]:
                            route = routing_engine.return_best_route_1to1(cur_pos, target_pos)
                            try:
                                route.remove(cur_pos[0])
                            except:
                                pass
                        #LOG.debug(f"new route {cur_pos} {route}")
                        new_pos, _, _, passed_nodes, _ = routing_engine.move_along_route(route, cur_pos, next_t - last_t, veh.vid, t)
                        veh.pos = new_pos
                        for node in passed_nodes:
                            route.remove(node)
                        vid_to_current_route[veh.vid] = route
                    ps_ind = i
                break
            else:   # boarding processes
                veh.pos = ps.get_pos()
                vid_to_current_route[veh.vid] = []
                to_remove = []
                to_add = []
                for rid in ps.get_list_alighting_rids():
                    #LOG.debug(f"check db {rid}")
                    for rq in veh.pax:
                        if rq.get_rid_struct() == rid:
                            #LOG.debug("ob")
                            to_remove.append(rq)
                            break
                for rid in ps.get_list_boarding_rids():
                    is_ob = False
                    #LOG.debug(f"check {rid}")
                    for rq in veh.pax:
                        if rq.get_rid_struct() == rid:
                            is_ob = True
                            break
                    if not is_ob:   # add new passenger (TODO dont like BasicRequest here)
                        #LOG.debug("not ob")
                        prq = rq_dict[rid]
                        traveller = BasicRequest(pd.Series({"start": prq.o_pos[0], "end": prq.d_pos[0], "rq_time": prq.rq_time, "request_id": rid, "latest_decision_time" : 1}),
                                                    routing_engine, time_step, {"latest_decision_time" : 1})
                        traveller.pu_time = arr
                        to_add.append(traveller)
                for pax in to_add:
                    veh.pax.append(pax)
                if next_t < dep:
                    ps.set_locked(True)
                    ps.set_started_at(arr)
                    LOG.debug(f"Move function: {next_t} < {dep} | {arr}")
                    LOG.debug(f"{veh}")
                    LOG.debug(f"Lock boarding: {ps}")
                    ps_ind = i
                    break
                else:
                    ps_ind += 1
                    last_t = dep
                    for pax in to_remove:
                        veh.pax.remove(pax)
        # track passed plan stops
        if ps_ind > 0:
            LOG.debug(f"old: {veh_plan}")
            for j in range(ps_ind):
                this_ps = veh_plan.list_plan_stops[j]
                this_ps.set_locked(False)
                passed_ps.append(this_ps)
            veh_plan.list_plan_stops = veh_plan.list_plan_stops[ps_ind:]
            LOG.debug(f"new {veh_plan}")
            LOG.debug(f"{veh}")
    # new state        
    if len(veh_plan.list_plan_stops) == 0:
        veh.status = VRL_STATES.IDLE
    else:
        if veh_plan.list_plan_stops[0].get_started_at() is None:
            veh.status = VRL_STATES.ROUTE
        else:
            if len(veh_plan.list_plan_stops[0].get_list_boarding_rids()) > 0 or len(veh_plan.list_plan_stops[0].get_list_alighting_rids()) > 0:
                veh.status = VRL_STATES.BOARDING
            else:
                veh.status = VRL_STATES.WAITING
                
    return passed_ps, vid_to_current_route

class ContinuousBatchRevelationReservation(BatchSchedulingRevelationHorizonBase):
    """ every reservation batch assignment, this algorithm assigns new reservation requests. 
    it first batches them based on overlapping earliest and latest pick-up times. 
    then it sorts the batches.
    for assignment, the current vehicle plans are progressed until the first request of the batch.
    it then treats the incoming requests similar to on-demand requests and solves the assignement problem(s).
    the corresponding assignment are used as long-term solutions for the vehicles."""
    def __init__(self, fleetctrl : FleetControlBase, operator_attributes : dict, dir_names : dict, solver : str="Gurobi"):
        super().__init__(fleetctrl, operator_attributes, dir_names, solver=solver)
        self._progress_time_step = operator_attributes.get(G_RA_REOPT_TS, 60)
        self._operator_attributes = operator_attributes.copy()
        
    def _create_sorted_request_batches(self, sim_time : int) -> List[List[PlanRequest]]:
        """ this class uses the currently active reservation request to create batches and returns a sorted list of them.
        vehicleplans of this sorted batches will be scheduled one after another
        :param sim_time: current simulation time
        :return: list of ReservationRequestBatch"""
        sorted_unprocessed_rids = sorted(self._unprocessed_rids.keys(), key=lambda x: self.active_reservation_requests[x].get_o_stop_info()[1])
        list_rid_batches = []
        current_batch = []
        
        insertion_assignment_rids = []
        
        last_time = None
        for rid in sorted_unprocessed_rids:
            req = self.active_reservation_requests[rid]
            LOG.debug(f"batching rid {rid} with epa {req.get_o_stop_info()[1]} |last_time {last_time} | current_batch {[rq.get_rid_struct() for rq in current_batch]}")
            if req.get_o_stop_info()[1] - sim_time <= self.assignment_horizon:
                insertion_assignment_rids.append(rid)
                continue
            if last_time is None:
                last_time = req.get_o_stop_info()[1]
            if req.get_o_stop_info()[1] - last_time > self.rolling_horizon:
                list_rid_batches.append(current_batch)
                current_batch = []
                last_time = None
            current_batch.append(req)
            if last_time is None:
                last_time = req.get_o_stop_info()[1]
        if len(current_batch) > 0:
            list_rid_batches.append(current_batch)
        # LOG.debug(f"sorted unprocessed rids {sorted_unprocessed_rids}")
        # LOG.debug(f"insertion_assignment_rids {insertion_assignment_rids}")
        # LOG.debug(f"list_rid_batches {list_rid_batches}")    
        return insertion_assignment_rids, list_rid_batches
        
    def _batch_optimisation(self, sim_time):
        LOG.debug(f"reservation batch optimization! rids to be assigned: {self._unprocessed_rids.keys()}")
        if len(self._unprocessed_rids) != 0:
            # batch the requests
            insertion_assignment_rids, list_rid_batches = self._create_sorted_request_batches(sim_time)
            # insert the insertion_assignment_rids into the batches
            LOG.debug(f"insertion_assignment_rids {insertion_assignment_rids}")
            for rid in insertion_assignment_rids:
                self.return_immediate_reservation_offer(rid, sim_time)
            # assign batches
            LOG.debug(f"start batch assignment")
            c = 0
            init_state = None
            for rq_batch in list_rid_batches:
                if c == len(list_rid_batches) - 1:
                    make_final_assignment = True
                else:
                    make_final_assignment = False
                LOG.debug(f"batch {c}/{len(list_rid_batches)} with {len(rq_batch)} requests (make assignment {make_final_assignment}): {[rq.get_rid_struct() for rq in rq_batch]}")
                init_state = self._batch_offline_assignment(rq_batch, sim_time, init_state=init_state, make_final_assignment=make_final_assignment)
                c += 1
            self._unprocessed_rids = {}
            
            # reset network
            if self.fleetctrl._use_own_routing_engine:
                self.routing_engine.reset_network(sim_time)
                
            return True
        else:
            return False
        
    def _batch_offline_assignment(self, rq_batch: List[PlanRequest], sim_time, init_state = None, make_final_assignment = True):
        """ assigns the given batch of reservation requests to the fleet
        by extrapolating current vehicle plans and solve static assignmentproblem at epa of first request
        :param rq_batch: list of reservation requests sorted by epa; diff of epa between first and last should not be larger than rolling horizon
        :param sim_time: current simulation time"""
        # get full vehicle plans
        if init_state is None:
            full_veh_plans = {}
            for vid, veh_plan in self.fleetctrl.veh_plans.items():
                if self._current_vids_with_supporting_points.get(vid):
                    plan_id = self._vid_to_plan_id[vid]
                    res_plan = self._plan_id_to_off_plan[plan_id]
                    fullplan = veh_plan.copy()
                    fullplan.list_plan_stops.pop()
                    fullplan.list_plan_stops += res_plan.list_plan_stops
                    self.fleetctrl.compute_VehiclePlan_utility(sim_time, self.fleetctrl.sim_vehicles[vid], fullplan)
                    full_veh_plans[vid] = fullplan
                    LOG.debug(f"full plan for vid {vid} : {fullplan}")
                else:
                    if veh_plan.get_utility() is None:
                        self.fleetctrl.compute_VehiclePlan_utility(sim_time, self.fleetctrl.sim_vehicles[vid], veh_plan)
                    full_veh_plans[vid] = veh_plan.copy()
                    LOG.debug(f"full plan for vid {vid} : {veh_plan}")
                    
            veh_objs = {vid : SimulationVehicleStruct(veh_obj, full_veh_plans[vid], sim_time, self.routing_engine) for vid, veh_obj in enumerate(self.fleetctrl.sim_vehicles)}
        else:
            full_veh_plans = init_state["full_veh_plans"]
            veh_objs = init_state["veh_objs"]
                
        # progress until first request
        vid_to_route = {}    # caching routes
        if init_state is None:
            vid_to_passed_ps = {}
        else:
            vid_to_passed_ps = init_state["vid_to_passed_ps"]
        if init_state is None:
            start_time = sim_time
        else:
            start_time = int(init_state["start_time"])
        prog_end_time_1 = int(rq_batch[0].get_o_stop_info()[1])
        prog_end_time_2 = int(rq_batch[-1].get_o_stop_info()[1]) - self.rolling_horizon
        prog_end_time = max(min(prog_end_time_1, prog_end_time_2), start_time)
        LOG.debug(f"batch prog end time: {prog_end_time} | {prog_end_time_1} | {prog_end_time_2} | {start_time}")
        for t in range(start_time, prog_end_time, self._progress_time_step):
            # update travel times
            if self.fleetctrl._use_own_routing_engine:
                self.routing_engine.update_network(t)
            # move vehicles
            for vid, veh in veh_objs.items():
                passed_ps, vid_to_route = move_vehicle_according_to_plan(veh, full_veh_plans[vid], t, self._progress_time_step, self.routing_engine, self.fleetctrl.rq_dict, vid_to_route, self.fleetctrl.begin_approach_buffer_time)
                if len(passed_ps) > 0:
                    if vid_to_passed_ps.get(veh.vid) is not None:
                        for ps in passed_ps:
                            vid_to_passed_ps[veh.vid].append(ps)
                    else:
                        vid_to_passed_ps[veh.vid] = passed_ps
                        
        # create vehplans for batch opt assignment
        opt_vehplans : Dict[int, VehiclePlan] = {}
        reservation_veh_stops = {}
        full_ass_requests = {} # full optization but assigned
        full_new_requests = {rq.get_rid_struct() : rq for rq in rq_batch} # full optimization assignment needed
        rids_to_send_offer = {rid : 1 for rid in full_new_requests.keys()} # for sending offers 
        rev_ass_requests = {} # lock assignment
        for vid, veh_p in full_veh_plans.items():
            first_stops, second_stops = self._full_split(prog_end_time, veh_p)
            if len(second_stops) > 0:
                reservation_veh_stops[vid] = second_stops
            online_vp = VehiclePlan(veh_objs[vid], prog_end_time, self.routing_engine, first_stops, copy=True)
            online_vp.vid = vid
            list_vrls = self.fleetctrl._build_VRLs(online_vp, veh_objs[vid], prog_end_time)
            veh_objs[vid].assigned_route = list_vrls
            online_vp.update_tt_and_check_plan(veh_objs[vid], prog_end_time, self.routing_engine, keep_feasible=True)
            
            vp_rids = online_vp.get_dedicated_rid_list()
            for rid in vp_rids:
                rq = self.fleetctrl.rq_dict[rid]
                epa = rq.get_o_stop_info()[1]
                if epa - prog_end_time <= self.rolling_horizon:
                    full_ass_requests[rid] = rq
                else:
                    rev_ass_requests[rid] = rq
            
            opt_vehplans[vid] = online_vp
            LOG.debug(f"online vp for vid {vid} and veh {veh_objs[vid]} : {online_vp}")
            
            
        # AM optimizer
        for vid, veh in veh_objs.items():
            veh.set_locked_vehplan(opt_vehplans[vid], prog_end_time, self.routing_engine) # TODO this is not updated dynamically
        AM_opt = AlonsoMoraAssignment(None, self.routing_engine, prog_end_time, self.fleetctrl.vr_ctrl_f, self._operator_attributes, veh_objs_to_build=veh_objs)
        AM_opt.max_rv_connections = 10        
        for vid, veh_p in opt_vehplans.items():
            AM_opt.set_assignment(vid, veh_p, is_external_vehicle_plan=True, _is_init_sol=True)

        for rid, rq in full_ass_requests.items():
            rq.compute_new_max_trip_time(self.routing_engine, boarding_time=self.fleetctrl.const_bt,
                                         max_detour_time_factor=self.fleetctrl.max_dtf, add_constant_detour_time=self.fleetctrl.add_cdt,
                                         max_constant_detour_time=self.fleetctrl.max_cdt)
            AM_opt.add_new_request(rid, rq, consider_for_global_optimisation=True, is_allready_assigned=True)
        for rid, rq in rev_ass_requests.items():
            rq.compute_new_max_trip_time(self.routing_engine, boarding_time=self.fleetctrl.const_bt,
                                         max_detour_time_factor=self.fleetctrl.max_dtf, add_constant_detour_time=self.fleetctrl.add_cdt,
                                         max_constant_detour_time=self.fleetctrl.max_cdt)
            AM_opt.add_new_request(rid, rq, consider_for_global_optimisation=False, is_allready_assigned=True)
        for rid, rq in full_new_requests.items():
            rq.compute_new_max_trip_time(self.routing_engine, boarding_time=self.fleetctrl.const_bt,
                                         max_detour_time_factor=self.fleetctrl.max_dtf, add_constant_detour_time=self.fleetctrl.add_cdt,
                                         max_constant_detour_time=self.fleetctrl.max_cdt)
            AM_opt.add_new_request(rid, rq, consider_for_global_optimisation=True, is_allready_assigned=False)
            
        for vid, veh in veh_objs.items():
            if len(veh.pax) > 0:
                for rq in veh.pax:
                    rid = rq.get_rid_struct()
                    AM_opt.set_database_in_case_of_boarding(rid, vid)
            
        AM_opt.compute_new_vehicle_assignments(prog_end_time, {}, veh_objs, build_from_scratch=True)
        
        # check for unassigned requests and idle vehicles
        vid_to_idle_times = {} # vid -> (last_idle_time, locked_end_ps/None)
        for vid, veh in veh_objs.items():
            cur_sol = AM_opt.get_optimisation_solution(vid)
            if cur_sol is None or len(cur_sol.list_plan_stops) == 0:
                passed_ps = vid_to_passed_ps.get(vid, [])
                if len(passed_ps) > 0:
                    last_end_time = passed_ps[-1].get_planned_arrival_and_departure_time()[1]
                    vid_to_idle_times[vid] = (last_end_time, None)
                else:
                    vid_to_idle_times[vid] = (sim_time, None)
            elif len(cur_sol.list_plan_stops) == 1 and cur_sol.list_plan_stops[-1].is_locked_end():
                passed_ps = vid_to_passed_ps.get(vid, [])
                if len(passed_ps) > 0:
                    last_end_time = passed_ps[-1].get_planned_arrival_and_departure_time()[1]
                    vid_to_idle_times[vid] = (last_end_time, cur_sol.list_plan_stops[-1])
                else:
                    vid_to_idle_times[vid] = (sim_time, cur_sol.list_plan_stops[-1])
            else:
                for rid, rq in list(full_new_requests.items()):
                    if rid in cur_sol.get_dedicated_rid_list():
                        del full_new_requests[rid]
                        LOG.debug(f" -> assigned reservation rid {rid} to vid {vid}")
        
        # check for possible repo trips for unassigned rids
        LOG.debug(f" -> unassigned rids {full_new_requests.keys()}")
        repo_assignment = {}
        for rid, rq in full_new_requests.items():
            best_idle_vid = None
            best_cost = float("inf")
            for vid, (last_idle_time, locked_end_ps) in vid_to_idle_times.items():
                veh_pos = veh_objs[vid].pos
                _, tt, dis = self.routing_engine.return_travel_costs_1to1(veh_pos, rq.get_o_stop_info()[0])
                cur_cost = 0
                if last_idle_time + tt < rq.get_o_stop_info()[2]:
                    cur_cost += dis
                    if locked_end_ps is not None:
                        pu_time = max(rq.get_o_stop_info()[1], last_idle_time + tt)
                        do_time = pu_time + self.routing_engine.return_travel_costs_1to1(rq.get_o_stop_info()[0], rq.get_d_stop_info()[0])[1] + 2*self.fleetctrl.const_bt
                        _, tt, dis = self.routing_engine.return_travel_costs_1to1(rq.get_d_stop_info()[0], locked_end_ps.get_pos())
                        if do_time + tt < locked_end_ps.get_earliest_start_time():
                            cur_cost += dis
                            cur_cost -= self.routing_engine.return_travel_costs_1to1(veh_pos, locked_end_ps.get_pos())[2]
                            if cur_cost < best_cost:
                                best_cost = cur_cost
                                best_idle_vid = vid
                    else:
                        if cur_cost < best_cost:
                            best_cost = cur_cost
                            best_idle_vid = vid
            if best_idle_vid is not None:
                prq_o_stop_pos, prq_t_pu_earliest, _ = rq.get_o_stop_info()
                bd_stop = BoardingPlanStop(prq_o_stop_pos, boarding_dict={1:[rid]}, earliest_pickup_time_dict={rid : prq_t_pu_earliest},
                                             latest_pickup_time_dict={rid : rid}, change_nr_pax=rq.nr_pax,
                                             duration=self.fleetctrl.const_bt)
                d_stop_pos, _, prq_max_trip_time = rq.get_d_stop_info()
                al_stop = BoardingPlanStop(d_stop_pos, boarding_dict={-1: [rid]}, max_trip_time_dict={rid : prq_max_trip_time},
                                                 change_nr_pax=-rq.nr_pax, duration=self.fleetctrl.const_bt)
                list_ps = [bd_stop, al_stop]
                if locked_end_ps is not None:
                    list_ps.append(locked_end_ps)
                repo_assignment[best_idle_vid] = VehiclePlan(veh_objs[best_idle_vid], sim_time, self.routing_engine, list_ps)
                del vid_to_idle_times[best_idle_vid]
                LOG.debug(f" -> assigned repo rid {rid} to vid {best_idle_vid}")
        
        # reassign support points
        vid_end_pos_time_dict = {}
        plan_id_start_constraints = {}
        for vid, veh in veh_objs.items():
            if repo_assignment.get(vid):
                cur_sol = repo_assignment[vid]
            else:
                cur_sol = AM_opt.get_optimisation_solution(vid)
            cur_ps = []
            if cur_sol is not None:
                cur_ps = cur_sol.list_plan_stops
            full_list_ps : List[PlanStop] = cur_ps
            if len(full_list_ps) > 0 and full_list_ps[-1].is_locked_end():
                full_list_ps.pop()
                
            if len(full_list_ps) > 0:
                p = full_list_ps[-1].get_pos()
                _, end_time = full_list_ps[-1].get_planned_arrival_and_departure_time()
                vid_end_pos_time_dict[vid] = (p, end_time)
            else:
                vid_end_pos_time_dict[vid] = (veh.pos, prog_end_time)
                
            if len(reservation_veh_stops.get(vid, [])) > 0:
                f_ps = reservation_veh_stops[vid][0]
                plan_id_start_constraints[vid] = (f_ps.get_pos(), f_ps.get_earliest_start_time())
        
        plan_id_vid_matches = self._future_supp_reassignment(vid_end_pos_time_dict, plan_id_start_constraints, sim_time)
        vid_to_plan_id = {vid : plan_id for plan_id, vid in plan_id_vid_matches}    
            
        # create final assignments
        next_init_state = {
            "full_veh_plans" : {},
            "veh_objs" : veh_objs,
            "vid_to_passed_ps" : vid_to_passed_ps,
            "start_time" : prog_end_time
        }    
        for vid, veh in veh_objs.items():
            if repo_assignment.get(vid):
                cur_sol = repo_assignment[vid]
            else:
                cur_sol = AM_opt.get_optimisation_solution(vid)
            cur_ps = []
            if cur_sol is not None:
                cur_ps = cur_sol.list_plan_stops
            full_list_ps : List[PlanStop] = cur_ps
            if len(full_list_ps) > 0 and full_list_ps[-1].is_locked_end():
                full_list_ps.pop()
            full_list_ps += reservation_veh_stops.get(vid_to_plan_id.get(vid, -1), [])
            
            next_init_state["full_veh_plans"][vid] = VehiclePlan(veh, prog_end_time, self.routing_engine, [ps.copy() for ps in full_list_ps])
            
            full_list_ps = vid_to_passed_ps.get(vid, []) + full_list_ps
            # check
            org_plan = self.fleetctrl.veh_plans[vid]
            for i, ps in enumerate(full_list_ps):
                # the move function introduces some locks, which have to be matched to the original plan
                if i < len(org_plan.list_plan_stops) and org_plan.list_plan_stops[i].get_pos() == ps.get_pos():
                    if org_plan.list_plan_stops[i].is_locked() and not ps.is_locked():
                        ps.set_locked(True)
                    if not org_plan.list_plan_stops[i].is_locked() and ps.is_locked():
                        LOG.debug(f"delock 1 {vid} {ps} | {org_plan.list_plan_stops[i]}")
                        ps.set_locked(False)
                else:
                    LOG.debug(f"delock 2 {vid} {ps}")
                    ps.set_locked(False)
                if ps.is_locked_end():
                    raise EnvironmentError(f"there should not be a lock end in {vid} | {full_list_ps}")
                
            new_vp = VehiclePlan(self.fleetctrl.sim_vehicles[vid], sim_time, self.routing_engine, full_list_ps)
            LOG.debug(f"new vp for vid {vid} and veh {self.fleetctrl.sim_vehicles[vid]} : {new_vp}")
            new_vp.update_tt_and_check_plan(self.fleetctrl.sim_vehicles[vid], sim_time, self.routing_engine, keep_feasible=True)
            
            if make_final_assignment:
                for rid in new_vp.get_dedicated_rid_list():
                    #if rids_to_send_offer.get(rid) is not None:
                    rq = self.fleetctrl.rq_dict[rid]
                    if rq.get_current_offer() is None:
                        _ = self.fleetctrl._create_user_offer(rq, sim_time, assigned_vehicle_plan=new_vp)
                self._assign_full_vehicleplan_after_insertion(vid, new_vp, sim_time)

        return next_init_state
            
                    
    def _full_split(self, split_time, full_veh_plan: QuasiVehiclePlan):
        current_occ = 0
        second_start_time = None
        current_rids = {}
        forced_stop = False
        index = len(full_veh_plan.list_plan_stops)
        split_index = index
        for ps in reversed(full_veh_plan.list_plan_stops):
            index -= 1
            current_occ -= ps.get_change_nr_pax()
            for rid in ps.get_list_boarding_rids():
                current_rids[rid] = 1
            if ps.is_locked() or ps.is_empty(): # rebalancing or locked parts should always be in online list
                forced_stop = True
            if current_occ == 0:
                possible_second_start_time, departure_time = ps.get_planned_arrival_and_departure_time()
                est = ps.get_earliest_start_time()
                if est > possible_second_start_time:
                    possible_second_start_time = est
                part_of_rid_revealed = False
                for rid in current_rids.keys():
                    if self._reavealed_rids.get(rid) or not self.active_reservation_requests.get(rid):
                        part_of_rid_revealed = True
                        break
                if not part_of_rid_revealed and not forced_stop and possible_second_start_time > split_time + self.assignment_horizon:    # possible checkpoint
                    second_start_time = possible_second_start_time
                    split_index = index
                    LOG.debug(f"possible break at {second_start_time} {split_index}")
                else:    # split detected
                    break
                current_rids = {}  
        LOG.debug(f" -> split index {split_index}")
        first_plan_stops = [full_veh_plan.list_plan_stops[i].copy() for i in range(split_index)] 
        second_plan_stops = [full_veh_plan.list_plan_stops[i].copy() for i in range(split_index, len(full_veh_plan.list_plan_stops))]
        LOG.debug(f"first plan stops: {[str(x) for x in first_plan_stops]}")
        LOG.debug(f"second plan stops: {[str(x) for x in second_plan_stops]}")
        #check for new revelations
        for ps in first_plan_stops:
            # for rid in ps.get_list_boarding_rids():
            #     if self.active_reservation_requests.get(rid) and not self._reavealed_rids.get(rid):
            #         plan_request = self.active_reservation_requests[rid]
            #         self._sorted_rids_with_epa.append( (plan_request.get_rid_struct(), plan_request.get_o_stop_info()[1]))
            #         self._reavealed_rids[rid] = 1
            #         LOG.debug(f"reveal soon: {plan_request.get_rid_struct()} at {plan_request.get_o_stop_info()[1]}")
            ps.direct_earliest_start_time = None
            ps.direct_latest_start_time = None
            
        # no more reservation part
        if second_start_time is None:
            pass
        else:   # reservation part
            sup_point_pos = second_plan_stops[0].get_pos()
            first_plan_stops.append( RoutingTargetPlanStop(sup_point_pos, earliest_start_time=second_start_time, latest_start_time=second_start_time, planstop_state=G_PLANSTOP_STATES.RESERVATION, duration=LARGE_INT, locked_end=True) )
            
        return first_plan_stops, second_plan_stops
    
    def _future_supp_reassignment(self, vid_end_pos_time_dict, plan_id_start_constraints, sim_time):
        matching_dict = {}
        # ensure feasibility
        for plan_id in plan_id_start_constraints.keys():
            try:
                matching_dict[plan_id][plan_id] = LARGE_INT * 0.1
            except KeyError:
                matching_dict[plan_id] = {plan_id : LARGE_INT * 0.1}
        for plan_id, s_pos_s_time in plan_id_start_constraints.items():
            if matching_dict.get(plan_id) is None:
                matching_dict[plan_id] = {}
            s_pos, s_time = s_pos_s_time
            _critical = False
            for vid, e_pos_e_time in vid_end_pos_time_dict.items():
                e_pos, e_time = e_pos_e_time
                _, tt, dis = self.routing_engine.return_travel_costs_1to1(e_pos, s_pos)
                if e_time + tt <= s_time:
                    matching_dict[plan_id][vid] = dis   # TODO dis as objective?
                else:
                    _critical = True
            if not _critical:
                for vid, obj in matching_dict[plan_id].items():
                    matching_dict[plan_id][vid] = obj / 100.0
        # match them together
        return self._match_plan_id_to_vid(matching_dict)
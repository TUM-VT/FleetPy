import numpy as np
import pandas as pd
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.routing.NetworkBase import NetworkBase
from src.fleetctrl.planning.VehiclePlan import PlanStopBase
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.fleetctrl.reservation.misc.RequestGroup import RequestGroup, QuasiVehiclePlan, VehiclePlanSupportingPoint, rg_key
from src.misc.globals import *

from typing import Any, Dict, List, Tuple

import logging
LOG = logging.getLogger(__name__)

def merge_reservationRequestBatches(list_reservationRequestBatch):
    """ this method returns a single ReservationRequestBatch obj the includes all request groups from the list of input batch
    doesnt adopt assignments!
    :param list_reservationRequestBatch: list of ReservationRequestBatch-objects to merg
    :return: ReservationRequestBatch that merged all input batches"""
    merged_batch = ReservationRequestBatch()
    for rb in list_reservationRequestBatch:
        merged_batch.requests_in_batch.update(rb.requests_in_batch)
        merged_batch.rg_constraints.update(rb.rg_constraints)
        merged_batch.rg_objectives.update(rb.rg_objectives)
        merged_batch.rid_to_rg.update(rb.rid_to_rg)
        for grade, grade_dict in rb.rg_graph.items():
            if merged_batch.rg_graph.get(grade) is None:
                merged_batch.rg_graph[grade] = {}
            for rg, rg_obj in grade_dict.items():
                merged_batch.rg_graph[grade][rg] = rg_obj
            if merged_batch.earliest_epa is None or merged_batch.earliest_epa > rb.earliest_epa:
                merged_batch.earliest_epa = rb.earliest_epa
            if merged_batch.latest_epa is None or merged_batch.latest_epa < rb.latest_epa:
                merged_batch.latest_epa = rb.latest_epa
    return merged_batch

class ReservationRequestBatch():
    """ this class is used as an collection of multiple reservation requests and to compute request groups within the batch """
    def __init__(self):
        """ this class is used as an collection of multiple reservation requests and to compute request groups within the batch 
        input parameters should only be used for internal usage (i.e. if a batch splits itself and new requests groups dont have the be recomputed)
        """
        self.requests_in_batch : Dict[Any, PlanRequest] = {} # rid -> plan request
        self.rg_graph: Dict[int, Dict[Any, RequestGroup]] = {1 : {}} # grade -> key -> rg_group obj (optional paramter only for internal use)
        self.rg_objectives : Dict[Any, float] = {}  # key -> objective
        self.rg_constraints : Dict[Any, Tuple[float, float, float, float]] = {} # key -> start_pos, start_time, end_pos, end_time of plan
        self.rid_to_rg = {}

        self.earliest_epa = None
        self.latest_epa = None

        self.current_assignments : Dict[Any, RequestGroup] = {}   # offline plan id -> request group
        self.rid_to_assigned_vid = {}   # rid -> offline plan id 
        
        self._store_for_rid = None
        self._stored_assignments = {}
        self._stored_rid_to_assigned_vid = {}

        if len(self.requests_in_batch) != 0:
            self.earliest_epa = min(prq.get_o_stop_info()[1] for prq in self.requests_in_batch.values())
            self.latest_epa = max(prq.get_o_stop_info()[1] for prq in self.requests_in_batch.values())
            for key in self.rg_objectives.keys():
                for rid in key:
                    try:
                        self.rid_to_rg[rid][key] = 1
                    except KeyError:
                        self.rid_to_rg[rid] = {key : 1}
            LOG.debug("ReservationRequestBatch : init after split:")
            LOG.debug("{}".format(self.requests_in_batch.keys()))
            LOG.debug("{}".format(self.rg_constraints.keys()))
            LOG.debug("{}".format(self))
            raise EnvironmentError

    def __str__(self):
        s = "ReservationRequestBatch: "
        s += "requests : {}".format(self.requests_in_batch.keys())
        s += "rgs : {}".format(self.rg_objectives.keys())
        return s

    def full_insertion_request_to_batch(self, prq : PlanRequest, routing_engine : NetworkBase, fleet_ctrl_ref : FleetControlBase, vehicle_capacity : int):
        """ this function adds a new request to the batch 
        :param prq: plan request object
        :param routing_engine: reference to routing engine
        :param fleet_ctrl_ref: reference to fleet control
        """
        new_rid_struct = prq.get_rid_struct()
        self.requests_in_batch[new_rid_struct] = prq
        epa = prq.get_o_stop_info()[1]
        if self.earliest_epa is None or epa < self.earliest_epa:
            self.earliest_epa = epa
        if self.latest_epa is None or epa > self.latest_epa:
            self.latest_epa = epa
        LOG.debug("full insertion of rid {}".format(new_rid_struct))
        rg_grad1 = RequestGroup(prq, routing_engine, fleet_ctrl_ref.const_bt, fleet_ctrl_ref.add_bt, vehicle_capacity)
        objective = rg_grad1.return_objective(fleet_ctrl_ref.vr_ctrl_f, routing_engine, self.requests_in_batch)
        constr = rg_grad1.return_best_plan(fleet_ctrl_ref.vr_ctrl_f, routing_engine, self.requests_in_batch).get_start_end_constraints()
        self._add_request_group_to_db(rg_grad1, objective, constr)
        # self.rg_objectives[rg_grad1.key] = objective
        # self.rg_constraints[rg_grad1.key] = rg_grad1.return_best_plan(fleet_ctrl_ref.vr_ctrl_f, routing_engine, self.requests_in_batch).get_start_end_constraints()
        # self.rg_graph[1][rg_grad1.key] = rg_grad1
        grade = 1
        while self.rg_graph.get(grade) is not None:
            rg_dict = self.rg_graph[grade]
            for key, rg_group in rg_dict.items():
                if new_rid_struct in key:
                    continue
                #LOG.debug("test {} for {}".format(key, prq.get_rid_struct()))
                new_rg_group = RequestGroup(prq, routing_engine, fleet_ctrl_ref.const_bt, fleet_ctrl_ref.add_bt, vehicle_capacity, lower_request_group=rg_group)
                if new_rg_group.is_feasible():
                    if self.rg_graph.get(grade + 1) is None:
                        self.rg_graph[grade + 1] = {}
                    #LOG.debug(" -> found {}".format(new_rg_group.key))
                    objective = new_rg_group.return_objective(fleet_ctrl_ref.vr_ctrl_f, routing_engine, self.requests_in_batch)
                    constr = new_rg_group.return_best_plan(fleet_ctrl_ref.vr_ctrl_f, routing_engine, self.requests_in_batch).get_start_end_constraints()
                    self._add_request_group_to_db(new_rg_group, objective, constr)
                    # self.rg_objectives[new_rg_group.key] = objective
                    # self.rg_constraints[new_rg_group.key] = new_rg_group.return_best_plan(fleet_ctrl_ref.vr_ctrl_f, routing_engine, self.requests_in_batch).get_start_end_constraints()
                    # self.rg_graph[grade+1][new_rg_group.key] = new_rg_group
            grade += 1
        LOG.debug(" -> {}".format(self.rg_constraints.keys()))

    def get_batch_size(self) -> int:
        """ this method returns the number of requests within the batch 
        :return: (int) number of requests in batch
        """
        return len(self.requests_in_batch)

    def get_batch_time_window(self) -> Tuple[float, float]:
        """ this method returns the minimum and maximum earliest pick up time of the requests in the batch
        :return: tuple (int, int) of minimum and maximum epa; (None, None) if batch is empty"""
        return self.earliest_epa, self.latest_epa
    
    def switch_assignments(self, dict_prev_to_new_imvid : Dict[int, int]):
        # self.current_assignments : Dict[Any, RequestGroup] = {}   # imaginary vid -> request group
        # self.rid_to_assigned_vid = {}   # rid -> imaginary vid
        for rid, im_vid in self.rid_to_assigned_vid.items():
            self.rid_to_assigned_vid[rid] = dict_prev_to_new_imvid.get(im_vid, im_vid)
        new_current_assignments = {}
        for prev_imvid, new_imvid in dict_prev_to_new_imvid.items():
            if self.current_assignments.get(prev_imvid) is not None:
                new_current_assignments[new_imvid] = self.current_assignments[prev_imvid]
        self.current_assignments = new_current_assignments

    def store_current_assignment(self, rid):
        self._store_for_rid = rid
        self._stored_assignments = {vid : rg.key for vid, rg in self.current_assignments.items()}
        self._stored_rid_to_assigned_vid = self.rid_to_assigned_vid.copy()
        
    def restore_current_assignments(self, rid):
        if self._store_for_rid is None or self._store_for_rid == rid:
            self.current_assignments = {vid : self.rg_graph[len(key)][key] for vid, key in self._stored_assignments.items()}
            self.rid_to_assigned_vid = self._stored_rid_to_assigned_vid

    def return_insertion_into_current_assignment(self, prq : PlanRequest, routing_engine : NetworkBase, 
                                                 fleet_ctrl_ref : FleetControlBase, vehicle_capacity : int) -> Tuple[QuasiVehiclePlan, int, bool]:
        """ this function test an insertion of a new plan request into current assignmened request groups of the batch
        and returns the best vehicle plan
        :param prq: plan request object
        :param routing_engine: reference to routing engine
        :param fleet_ctrl_ref: reference to fleet control
        :return: tuple of best insertion plan, current number of used vehicles in this batch, bool which is True in case a new vehicle is needed in the batch
        """
        best_obj = float("inf")
        best_rg = None
        best_vid = None
        new_rid_struct = prq.get_rid_struct()
        self.requests_in_batch[new_rid_struct] = prq
        for vid, assigned_rg in self.current_assignments.items():
            new_rg = RequestGroup(prq, routing_engine, fleet_ctrl_ref.const_bt, fleet_ctrl_ref.add_bt, vehicle_capacity, lower_request_group=assigned_rg)
            if new_rg.is_feasible():
                obj = new_rg.return_objective(fleet_ctrl_ref.vr_ctrl_f, routing_engine, fleet_ctrl_ref.rq_dict)
                constr = new_rg.return_best_plan(fleet_ctrl_ref.vr_ctrl_f, routing_engine, fleet_ctrl_ref.rq_dict).get_start_end_constraints()
                self._add_request_group_to_db(new_rg, obj, constr)
                if obj < best_obj:
                    best_obj = obj
                    best_vid = vid
                    best_rg = new_rg
        new_vid_needed = False
        if best_rg is None:
            best_rg = RequestGroup(prq, routing_engine, fleet_ctrl_ref.const_bt, fleet_ctrl_ref.add_bt, vehicle_capacity)
            obj = best_rg.return_objective(fleet_ctrl_ref.vr_ctrl_f, routing_engine, fleet_ctrl_ref.rq_dict)
            constr = best_rg.return_best_plan(fleet_ctrl_ref.vr_ctrl_f, routing_engine, fleet_ctrl_ref.rq_dict).get_start_end_constraints()
            self._add_request_group_to_db(best_rg, obj, constr)
            if len(self.current_assignments) > 0:
                best_vid = min(list(self.current_assignments.keys()) + [0]) - 1
            else:
                best_vid = -1
            new_vid_needed = True
        self._assign_rg_to_vid(best_vid, best_rg)
        LOG.debug(f"after batch insertion of {new_rid_struct}: best rg {best_rg.key} | assignments : { {vid : rg.key for vid, rg in self.current_assignments.items()} }")
        return best_rg.return_best_plan(fleet_ctrl_ref.vr_ctrl_f, routing_engine, fleet_ctrl_ref.rq_dict), len(self.current_assignments.keys()), new_vid_needed

    def _add_request_group_to_db(self, request_group : RequestGroup, obj : float, constraints : Tuple[float, float, float, float]):
        """ adds a request group to all database entries
        :param request_group: RequestGroup obj
        :param obj: value from request_group.return_objective(...)
        :param constraints: value from request_group.return_best_plan(...).get_start_end_constraints()
        """
        key = request_group.key
        grade = len(key)
        if self.rg_graph.get(grade) is None:
            self.rg_graph[grade] = {}
        self.rg_graph[grade][key] = request_group
        self.rg_constraints[key] = constraints
        self.rg_objectives[key] = obj
        for rid in key:
            try:
                self.rid_to_rg[rid][key] = 1
            except:
                self.rid_to_rg[rid] = {key : 1}

    def _assign_rg_to_vid(self, vid, rg : RequestGroup):
        """ adds all db entries for the assignment of a vid to a request group
        :param vid: imaginary vehicle id
        :param rg: RequestGroup obj
        """
        if rg is not None:
            self.current_assignments[vid] = rg 
            for rid in rg.key:
                self.rid_to_assigned_vid[rid] = vid
        else:
            try:
                del self.current_assignments[vid]
            except KeyError:
                pass

    def delete_request_from_batch(self, rid):
        """ deletes all entries of the request in this batch obj
        :param rid: request id
        """
        LOG.debug(f"delete requests : {rid}")
        LOG.debug(f"for {self.rid_to_rg.get(rid, {})}")
        #LOG.debug(str(self))
        for key in self.rid_to_rg.get(rid, {}).keys():
            g = len(key)
            try:
                del self.rg_graph[g][key]
            except KeyError:
                pass
            try:
                del self.rg_constraints[key]
            except KeyError:
                pass
            try:
                del self.rg_objectives[key]
            except KeyError:
                pass
            for orid in key:
                if rid != orid:
                    try:
                        del self.rid_to_rg[orid][key]
                    except KeyError:
                        pass
        try:
            del self.rid_to_rg[rid]
        except KeyError:
            pass
        if self.rid_to_assigned_vid.get(rid) is not None:
            # assigned_rg = self.current_assignments[self.rid_to_assigned_vid[rid]]
            # assigned_rids = list(assigned_rg.key)
            # assigned_rids.remove(rid)
            # if len(assigned_rids) > 0:
            #     new_key = rg_key(assigned_rids)
            #     try:
            #         new_rg = self.rg_graph[len(new_key)][new_key]
            #     except KeyError:
            #         raise NotImplementedError(f"couldnt find lower key for {new_key} | {self.rg_objectives.keys()}")
            #     self._assign_rg_to_vid(self.rid_to_assigned_vid[rid], new_rg)
            # else:
            #     self._assign_rg_to_vid(self.rid_to_assigned_vid[rid], None )
            del self.rid_to_assigned_vid[rid]
        try:
            del self.requests_in_batch[rid]
        except:
            pass

    def get_rg_obj_const(self) -> Tuple[Dict[Any, float], Dict[Any, Tuple[float, float, float, float]]]:
        """ returns the current objective dict and constraint dict of the batch
        :return: tuple of (dict rg-key -> obj value, dict rg-key -> constraints)
        """
        return self.rg_objectives, self.rg_constraints

    def set_assignments(self, vid_assignments):
        """ this function sets database entries for a new assignment from the optimization process
        :param vid_assignments: dict im vehicle id -> rg-key"""
        self.current_assignments = {}
        self.rid_to_assigned_vid = {}
        _ = {self._assign_rg_to_vid(vid, self.rg_graph[len(rg_key)][rg_key]) for vid, rg_key in vid_assignments.items()}

    def get_best_plan_of_rg(self, rg_key : tuple, fleet_ctrl_ref : FleetControlBase, routing_engine : NetworkBase) -> QuasiVehiclePlan:
        """ this function returns the best vehicle plan of the request group with key rg_key
        :param rg_key: key of request group
        :param fleet_ctrl_ref: reference to fleet control
        :param routing_engine: reference to routing engine
        :return: best vehicle plan of request group (QuasiVehiclePlan)
        """
        rg = self.rg_graph[len(rg_key)][rg_key]
        return rg.return_best_plan(fleet_ctrl_ref.vr_ctrl_f, routing_engine, self.requests_in_batch)
    
    def get_next_sup_point(self, vid : int, fleet_ctrl_ref : FleetControlBase, routing_engine : NetworkBase) -> VehiclePlanSupportingPoint:
        """ this function returns the next scheduled supporting point of the assigned imaginary vid
        :param vid: vehicle id which has been assigned in a previous assignment process
        :param fleet_ctrl_ref: reference to fleet control
        :param routing_engine: reference to routing engine
        :return: VehicleSupportingPoint"""
        assigned_rg = self.current_assignments[vid]
        best_plan = assigned_rg.return_best_plan(fleet_ctrl_ref.vr_ctrl_f, routing_engine, self.requests_in_batch)
        sup_points = best_plan.get_all_supporting_points()
        return sup_points[0]
    
    def reveal_until_next_sup(self, vid : int, fleet_ctrl_ref : FleetControlBase, routing_engine : NetworkBase,
                              vehicle_capacity : int) -> Tuple[List[PlanStopBase], VehiclePlanSupportingPoint]:
        """ this function returns the vehicle plan currently assigned until the nex supporting point
        and the following supporting point. if the next supporting point is not in the batch, None is returned
        this function also deletes all requests that are revealed with this supporting point
        additionally time constraints are updated to guarantee a feasible next sup
        :param im_vid: vehicle id which has been assigned in a previous assignment process
        :param fleet_ctrl_ref: reference to fleet control
        :param routing_engine: reference to routing engine
        :param vehicle_capacity: capacity of the vehicles
        :return: Tuple of (list plan stops, VehicleSupportingPoint) or (list plan stops, None)"""
        assigned_rg = self.current_assignments[vid]
        best_plan = assigned_rg.return_best_plan(fleet_ctrl_ref.vr_ctrl_f, routing_engine, self.requests_in_batch)
        sup_points = best_plan.get_all_supporting_points()
        first_plan_stops, second_plan_stops, second_start_time = best_plan.split_at_next_supporting_point()
        first_sup = sup_points[0]
        involved_rids = first_sup.return_involved_rids()
        
        LOG.debug("reveal until next sup for im_vid {}:".format(vid))
        LOG.debug("assigned: {}".format(assigned_rg.key))
        LOG.debug("sups: {}".format([str(x) for x in sup_points]))
        LOG.debug("first: {}".format([str(ps) for ps in first_plan_stops]))
        LOG.debug("second start time: {}".format(second_start_time))
        LOG.debug("second: {}".format([str(ps) for ps in second_plan_stops]))
        
        new_rg = None
        if second_start_time is not None:
            curr_rids = list(assigned_rg.key)
            new_rids = curr_rids[:]
            for rid in involved_rids:
                new_rids.remove(rid)
            if len(new_rids) > 0:
                new_key = rg_key(new_rids)
                try:
                    new_rg = self.rg_graph[len(new_key)][new_key]
                    LOG.debug("new rg1: {}".format(new_rg))
                    if not new_rg.update_start_time(second_start_time, routing_engine, vehicle_capacity):
                        new_rg = None
                except KeyError:
                    pass
                if new_rg is None:
                    new_rg = RequestGroup(None, routing_engine, fleet_ctrl_ref.const_bt, fleet_ctrl_ref.add_bt, vehicle_capacity, earliest_start_time=second_start_time, 
                                        list_quasi_vehicle_plans=[QuasiVehiclePlan(routing_engine, second_plan_stops, vehicle_capacity, earliest_start_time=second_start_time)])
                    LOG.debug("new rg2: {}".format(new_rg))
                objective = new_rg.return_objective(fleet_ctrl_ref.vr_ctrl_f, routing_engine, self.requests_in_batch)
                constr = new_rg.return_best_plan(fleet_ctrl_ref.vr_ctrl_f, routing_engine, self.requests_in_batch).get_start_end_constraints()
                self._add_request_group_to_db(new_rg, objective, constr)
                LOG.debug(" -> new key {}".format(new_key))
                LOG.debug(f" -> new rg {new_rg}")
            
        for rid in involved_rids:
            self.delete_request_from_batch(rid)
                
        self._assign_rg_to_vid(vid, new_rg)
        next_sup = None
        if new_rg is not None:
            next_best_plan = new_rg.return_best_plan(fleet_ctrl_ref.vr_ctrl_f, routing_engine, self.requests_in_batch)
            next_sup_points = next_best_plan.get_all_supporting_points()
            if len(next_sup_points) > 0:
                next_sup = next_sup_points[0]
            LOG.debug(" -> new sup: {}".format(next_sup))
            
        return first_plan_stops, next_sup        
            
            
    def assign_rg_without_rids(self, im_vid, list_rids_to_remove : List[Any], fleetctrl : FleetControlBase,
                               routing_engine : NetworkBase, vehicle_capacity : int, sim_time : int):
        current_rg = self.current_assignments[im_vid]
        curr_key = current_rg.key
        for rid in list_rids_to_remove:
            assigned_rids = list(curr_key)
            assigned_rids.remove(rid)
            if len(assigned_rids) > 0:
                prev_key = curr_key
                curr_key = rg_key(assigned_rids)
                try:
                    current_rg = self.rg_graph[len(curr_key)][curr_key]
                except KeyError:
                    LOG.debug(f"couldnt find lower key {curr_key} -> build lower by removing {rid}")
                    prev_rg = self.rg_graph[len(prev_key)][prev_key]
                    current_rg = prev_rg.create_lower_rg(self.requests_in_batch[rid], fleetctrl.vr_ctrl_f, routing_engine, fleetctrl.const_bt, fleetctrl.add_bt, vehicle_capacity, self.requests_in_batch)
                    obj = current_rg.return_objective(fleetctrl.vr_ctrl_f, routing_engine, fleetctrl.rq_dict)
                    constr = current_rg.return_best_plan(fleetctrl.vr_ctrl_f, routing_engine, fleetctrl.rq_dict).get_start_end_constraints()
                    self._add_request_group_to_db(current_rg, obj, constr)
            else:
                current_rg = None
                break
        self._assign_rg_to_vid(im_vid, current_rg)
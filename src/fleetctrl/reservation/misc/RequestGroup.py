from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.planning.VehiclePlan import VehiclePlan, BoardingPlanStop, PlanStopBase
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.routing.NetworkBase import NetworkBase
from src.fleetctrl.pooling.immediate.insertion import simple_insert, simple_remove
from src.misc.globals import *
from typing import Any, Callable, List, Dict, Tuple, Type

import logging
LOG = logging.getLogger(__name__)

def rg_key(list_rids : List[Any]) -> tuple:
    """ defines a key for groups of requests (tuple of sorted rids)
    :param list_rids: list of request ids
    :return: tuple representing the unique key of the list of requests"""
    return tuple(sorted(list_rids))

def get_lower_keys(key : tuple) -> List[tuple]:
    """returns a list of lower keys
    :param key: request group key
    :return: list of lower keys
    """
    r_list = []
    if len(key) == 1:
        return []
    prev_list = list(key)
    for rid in prev_list:
        new_list = prev_list[:]
        new_list.remove(rid)
        new_key = rg_key(new_list)
        r_list.append(new_key)
        r_list += get_lower_keys(new_key)
    return list(set(r_list))

class VehiclePlanSupportingPoint():
    """ this class is a collector supporting points in schedules 
    in a vehicle plan a supporting point is defined at plan stops at which in theory the vehicle to fullfill the schedule
    can be exchanged, i.e. the current occupancy becomes 0
    a supporting point is defined as the start location of the subtrip, the time a vehicle has to arrive there to fillfull the plans
    time constraints and the list of rids involved in the following subtrip """
    def __init__(self, start_pos : tuple, start_time : int, involved_rid_list : List[Any]):
        """
        :param start_pos: start position of supportin point
        :param start_time: time the upcoming trip is suppost to start
        :param involved_rid_list: list of requests involved in the upcoming trip
        """
        self.start_pos = start_pos
        self.start_time = start_time
        self.involved_rid_list = involved_rid_list

    def return_sup_point_constraint(self) -> Tuple[tuple, int]:
        """ this function returns start pos and start time of the supporting point 
        :return: tuple (start pos, start time)"""
        return (self.start_pos, self.start_time)

    def return_involved_rids(self) -> List[Any]:
        """ this function returns the list of requests that are involved in upcoming subtrip
        :return: list of rids"""
        return self.involved_rid_list
    
    def __str__(self):
        return f"VehSupPoint {self.start_pos} {self.start_time} {self.involved_rid_list}"

class QuasiVehicle():
    """ this class just serves as a helping class to define vehicle plans without an actual vehicle """
    def __init__(self, pos : Tuple, capacity : int = 4):
        self.op_id = 0
        self.vid = 0
        self.status = 0
        #
        self.max_pax = capacity
        self.max_parcels = 0 # TODO no reservation for parcels
        self.daily_fix_cost = 1
        self.distance_cost = 1
        self.battery_size = 100000000
        self.range = 100000000
        self.soc_per_m = 1/(self.range*1000)
        self.pos = pos
        self.soc = 1
        self.pax = []  # rq_obj

    def get_nr_pax_without_currently_boarding(self) -> int:
        return 0
        
    def get_nr_parcels_without_currently_boarding(self):
        return 0

    def compute_soc_consumption(self, distance : float) -> float:
        """This method returns the SOC change for driving a certain given distance.

        :param distance: driving distance in meters
        :type distance: float
        :return: delta SOC (positive value!)
        :rtype: float
        """
        return distance * self.soc_per_m    

class QuasiVehiclePlan(VehiclePlan):
    """ this function represents just a schedule for requests and doesnt involve an actual vehicle
        therefore a QuasiVehicle is placed at the starting location of the vehicle plan to be in line with
        the vehicle plan formalism """
    def __init__(self, routing_engine : NetworkBase, list_plan_stops : List[PlanStopBase], vehicle_capacity : int, 
                 earliest_start_time : int =-1, copy : bool=False, external_pax_info : dict={}):
        """  
        :param routing_engine: reference to routing engine obj
        :param list_plan_stops: list of plan stops corresponding to this plan
        :param vehicle_capacity: (int) max number of passengers per vehicle
        :param earliest_start_time: (int) optional. to give if an earliest start time has to be considered
        :param copy: (bool) optional; only use internally
        :param external_pax_info: only use internally
        """
        veh_obj = QuasiVehicle(list_plan_stops[0].get_pos(), capacity=vehicle_capacity)
        if list_plan_stops[0].get_earliest_start_time() > earliest_start_time:
            earliest_start_time = list_plan_stops[0].get_earliest_start_time()
        self.vehicle_capacity = vehicle_capacity
        self.earliest_start_time = earliest_start_time
        super().__init__(veh_obj, earliest_start_time, routing_engine, list_plan_stops, copy=copy, external_pax_info=external_pax_info)

    def copy(self):
        """
        see VehiclePlan
        """
        tmp_VehiclePlan = QuasiVehiclePlan(None, [ps.copy() for ps in self.list_plan_stops], self.vehicle_capacity, earliest_start_time=self.earliest_start_time, copy=True)
        tmp_VehiclePlan.vid = self.vid
        tmp_VehiclePlan.utility = self.utility
        tmp_VehiclePlan.pax_info = self.pax_info.copy()
        tmp_VehiclePlan.feasible = True
        return tmp_VehiclePlan

    def return_intermediary_plan_state(self, veh_obj : None, sim_time : int, routing_engine : NetworkBase, stop_index : int):
        """
        see VehiclePlan
        """
        veh_obj = QuasiVehicle(self.list_plan_stops[0].pos, capacity=self.vehicle_capacity)
        return super().return_intermediary_plan_state(veh_obj, self.earliest_start_time, routing_engine, stop_index)

    def update_tt_and_check_plan(self, veh_obj : None, sim_time : int, routing_engine : NetworkBase, init_plan_state=None, keep_feasible=False):
        """
        see VehiclePlan
        """
        veh_obj = QuasiVehicle(self.list_plan_stops[0].pos, capacity=self.vehicle_capacity)
        return super().update_tt_and_check_plan(veh_obj, self.earliest_start_time, routing_engine, init_plan_state=init_plan_state, keep_feasible=keep_feasible)

    def compute_obj_function(self, obj_function : Callable, routing_engine : NetworkBase, rq_dict : Dict[Any, PlanRequest]):
        """
        this function calculates the objective function of the schedule
        :param obj_function: control function to rate vehicle plans
        :param routing_engine: reference to routing_engine
        :param rq_dict: dictionary rid -> rq_obj with at least all requests involved in the plan
        :return: utility value (float)
        """
        veh_obj = QuasiVehicle(self.list_plan_stops[0].pos, capacity=self.vehicle_capacity)
        self.set_utility( obj_function(self.earliest_start_time, veh_obj, self, rq_dict, routing_engine) )
        return self.get_utility()

    def get_latest_start_time_and_end(self):
        """ this function returns expected start and end time of the schedule
        currently planned arrival times of first and last plan stop is returned
        TODO : is this the best way ???
        :return: tuple of (start_time, end_time) of the plan """
        start_time = self.list_plan_stops[0].get_planned_arrival_and_departure_time()[0]
        end_time = self.list_plan_stops[-1].get_planned_arrival_and_departure_time()[1]
        return start_time, end_time

    def get_start_end_pos(self):
        """ returns start and end position of the plan 
        :return: tuple of (start_pos, end_pos)"""
        return self.list_plan_stops[0].get_pos(), self.list_plan_stops[-1].get_pos()

    def get_start_end_constraints(self):
        """ returns the spatio temporal information of this vehicle plan
        :return: start_pos, start_time, end_pos, end_time"""
        start_pos, end_pos = self.get_start_end_pos()
        start_time, end_time = self.get_latest_start_time_and_end()
        return start_pos, start_time, end_pos, end_time

    def get_all_supporting_points(self) -> List[VehiclePlanSupportingPoint]:
        """ this function returns intermediate points in the plan where a different vehicle could in theory
        continue to fullfill the schedule. therefore all steps in the plan are evaluated where the vehicle occupancy would
        become 0 and can therefore be interpreted as starting constraints for a sub-schedule. 
        The informations are collected in the class VehiclePlanSupportingPoint
        :return: list of VehiclePlanSupportingPoint in corresponding plan order"""
        current_occ = 0
        list_supporting_points = []
        current_supporting_point_constraints = None
        current_involved_rids = []
        for ps in self.list_plan_stops:
            if current_occ == 0:
                if current_supporting_point_constraints is not None:
                    list_supporting_points.append(VehiclePlanSupportingPoint(current_supporting_point_constraints[0], current_supporting_point_constraints[1], current_involved_rids))
                current_supporting_point_constraints = (ps.get_pos(), ps.get_planned_arrival_and_departure_time()[0])
                current_involved_rids = []
            current_occ += ps.get_change_nr_pax()
            for rid in ps.get_list_boarding_rids():
                current_involved_rids.append(rid)
        if current_occ != 0:
            raise EnvironmentError("Occupancy is not zero hat end of plan??? {}".format(self))
        list_supporting_points.append(VehiclePlanSupportingPoint(current_supporting_point_constraints[0], current_supporting_point_constraints[1], current_involved_rids))

        return list_supporting_points
    
    def split_at_next_supporting_point(self) -> Tuple[List[PlanStopBase], List[PlanStopBase], float]:
        """ this function splits the assigned plan at the next supporting point (i.e. the next time the occupancy would become 0)
        it returns the first part of the vehicle plan (list plan stops), the second part of the vehicle plan and the time the second part of the plan is supposed to start
        :return: tuple of (first list plan stops, second list plan stops, second part start time/None if empty)
        """
        current_occ = 0
        first_plan_stops = []
        second_plan_stops = []
        second_start_time = None
        for i, ps in enumerate(self.list_plan_stops):
            if i != 0 and second_start_time is None and current_occ == 0:
                second_start_time, departure_time = ps.get_planned_arrival_and_departure_time()
                est = ps.get_earliest_start_time()
                if est > second_start_time:
                    second_start_time = est
            if second_start_time is None:
                first_plan_stops.append(ps)
            else:
                second_plan_stops.append(ps)
            current_occ += ps.get_change_nr_pax()
        return first_plan_stops, second_plan_stops, second_start_time

# =============================================================================== #

class RequestGroup():
    """ this class is a collection of QuasiVehiclePlans that serve the same set of requests 
    the grade of a request group  is defined by the number of requests served by this group"""
    def __init__(self, request_to_insert : PlanRequest, routing_engine : NetworkBase, std_bt : int, add_bt : int,
                 vehicle_capacity : int, earliest_start_time : int=-1, lower_request_group=None, list_quasi_vehicle_plans : List[QuasiVehiclePlan]=None):
        """ a RequestGroup can be initiallized by giving the corresponding request object to create
            a groupe of grade 1 (just one vehicle plan with pickup and drop off which is always feasible
            request groups of higher grade can be initialized by giving the lower request groupe where the new 
            request will be inserted into 
        :param request_to_insert: plan request object to add to the group or initialize the tree
        :param routing_engine: reference to routing engine
        :param std_bt: constant boarding time
        :param add_bt: additional boarding time per customer
        :param earliest_start_time: optional; if given the group's plans cannot start before this number
        :param lower_request_group: RequestGroup object. if given the new requests will be inserted into this group
                    otherwise a RequestGroup of grade 1 is created """
        self.list_quasi_vehicle_plans : List[QuasiVehiclePlan] = []
        self.feasible = True
        self.key = None
        self.objective = None
        self.best_plan_index = None
        self.start_end_time = None
        self.earliest_start_time = earliest_start_time
        if lower_request_group is None and list_quasi_vehicle_plans is None:
            prq_o_stop_pos, epa, lpa = request_to_insert.get_o_stop_info()
            prq_d_stop_pos, tdo, mtt = request_to_insert.get_d_stop_info()
            new_rid_struct = request_to_insert.get_rid_struct()
            b = BoardingPlanStop(prq_o_stop_pos, boarding_dict={1: [new_rid_struct]}, earliest_pickup_time_dict={new_rid_struct : epa},
                                latest_pickup_time_dict={new_rid_struct : lpa}, change_nr_pax=request_to_insert.nr_pax, duration=std_bt+add_bt)
            d = BoardingPlanStop(prq_d_stop_pos, boarding_dict={-1: [new_rid_struct]}, max_trip_time_dict={new_rid_struct: mtt},
                                change_nr_pax=-request_to_insert.nr_pax, duration=std_bt+add_bt)
            plan = QuasiVehiclePlan(routing_engine, [b, d], vehicle_capacity, earliest_start_time=earliest_start_time)
            self.list_quasi_vehicle_plans.append(plan)
            self.key = rg_key([new_rid_struct])
        elif lower_request_group is not None:
            for plan in lower_request_group.list_quasi_vehicle_plans: 
                for new_plan in simple_insert(routing_engine, earliest_start_time, QuasiVehicle(None, capacity=vehicle_capacity), plan, request_to_insert, std_bt, add_bt):
                    self.list_quasi_vehicle_plans.append(new_plan)
            if len(self.list_quasi_vehicle_plans) == 0:
                self.feasible = False
            self.key = rg_key([request_to_insert.get_rid_struct()] + list(lower_request_group.key))
        elif list_quasi_vehicle_plans is not None:
            rids = list_quasi_vehicle_plans[0].get_involved_request_ids()
            self.key = rg_key(rids)
            self.list_quasi_vehicle_plans = list_quasi_vehicle_plans

    def is_feasible(self) -> bool:
        return self.feasible

    def return_objective(self, obj_function : Callable, routing_engine : NetworkBase, rq_dict : Dict[Any, PlanRequest]) -> float:
        """ this method returns the objective of the best vehicle plan in this request group
        :param obj_function: fleet control objectve function
        :param routing_engine: reference to routing engine
        :param rq_dict: dict rid -> plan request obj of at least all request involved in this plan
        :return: objective value (float)"""
        if self.objective is None or self.best_plan_index is None:
            self.objective = float("inf")
            for i, p in enumerate(self.list_quasi_vehicle_plans):
                o = p.compute_obj_function(obj_function, routing_engine, rq_dict)
                if o < self.objective:
                    self.objective = o
                    self.best_plan_index = i
        return self.objective

    def return_best_plan(self, obj_function : Callable, routing_engine : NetworkBase, rq_dict : Dict[Any, PlanRequest]) -> QuasiVehiclePlan:
        """ this method returns the the best vehicle plan in this request group
        :param obj_function: fleet control objectve function
        :param routing_engine: reference to routing engine
        :param rq_dict: dict rid -> plan request obj of at least all request involved in this plan
        :return: QuasiVehiclePlan obj of best plan according to objective function"""
        _ = self.return_objective(obj_function, routing_engine, rq_dict)
        return self.list_quasi_vehicle_plans[self.best_plan_index]

    def create_lower_rg(self, prq_to_remove : PlanRequest, obj_function : Callable, routing_engine : NetworkBase,
                        std_bt : int, add_bt : int, vehicle_capacity : int, rq_dict : Dict[Any, PlanRequest]):
        lower_plan_list = []
        ps_found = {}
        for veh_plan in self.list_quasi_vehicle_plans:
            new_plan = simple_remove(QuasiVehicle(veh_plan.list_plan_stops[0].get_pos(), capacity=vehicle_capacity), veh_plan, prq_to_remove.get_rid_struct(), self.earliest_start_time, routing_engine, obj_function, rq_dict, std_bt, add_bt)
            new_plan = QuasiVehiclePlan(routing_engine, new_plan.list_plan_stops, vehicle_capacity, earliest_start_time=self.earliest_start_time)
            ps_tuple = tuple(ps.get_pos() for ps in new_plan.list_plan_stops)
            if ps_found.get(ps_tuple) is not None:
                continue
            ps_found[ps_tuple] = 1
            lower_plan_list.append(new_plan)
        new_rg = RequestGroup(prq_to_remove, routing_engine, std_bt, add_bt, vehicle_capacity, earliest_start_time=self.earliest_start_time)
        new_rg.list_quasi_vehicle_plans = lower_plan_list
        old_rid_list = list(self.key)
        old_rid_list.remove(prq_to_remove.get_rid_struct())
        new_rg.key = rg_key(old_rid_list)
        return new_rg
    
    def update_start_time(self, earliest_start_time : int, routing_engine : NetworkBase, vehicle_capacity : int) -> bool:
        """ this function sets a near earliest start_time for the request group and returns if the group is still feasible
        :param earliest_start_time: (int) earliest start time for request group
        :param routing_engine: reference to routing engine
        :param vehicle_capacity: capacity of vehicles
        :return: (bool) if rg is still feasible or not"""
        new_list_veh_plans = []
        self.earliest_start_time = earliest_start_time
        for veh_plan in self.list_quasi_vehicle_plans:
            new_plan = QuasiVehiclePlan(routing_engine, veh_plan.list_plan_stops, vehicle_capacity, earliest_start_time=self.earliest_start_time)
            if new_plan.update_tt_and_check_plan(None, None, routing_engine):
                new_list_veh_plans.append(new_plan)
        self.list_quasi_vehicle_plans = new_list_veh_plans
        if len(self.list_quasi_vehicle_plans) == 0:
            self.feasible = False
        self.objective = None
        self.best_plan_index = None
        return self.is_feasible()
    
    def __str__(self):
        s = [f"request group {self.key} with obj {self.objective} and best plan index {self.best_plan_index} | earliest start time {self.earliest_start_time}"]
        for p in self.list_quasi_vehicle_plans:
            s.append(str(p))
        return "\n".join(s)
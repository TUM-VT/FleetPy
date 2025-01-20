from __future__ import annotations
import os
import random
from abc import abstractmethod, ABCMeta
from src.misc.globals import G_RA_RES_BOPT_TS

from typing import TYPE_CHECKING, Dict, Any, List, Tuple
if TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.fleetctrl.planning.PlanRequest import PlanRequest
    from src.simulation.Offers import TravellerOffer
    from src.fleetctrl.planning.VehiclePlan import VehiclePlan

import logging
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_ReservationBase = {
    "doc" :  """This modules deals with request pre-booking a trip a long time in advance. 
            As these requests are hard to handle in the online assignment algorithm 
            (a lot of insertion possibilities into current vehicles schedules), 
            those requests are handeled in an additional sub-module. """,
    "inherit" : None,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class ReservationBase(metaclass=ABCMeta):
    def __init__(self, fleetctrl : FleetControlBase, operator_attributes : dict, dir_names : dict, solver : str="Gurobi"):
        """ this class is used as a base to treat reservations
        :param fleetctrl: reference to fleetcontrol
        :param operator_attributes: operator attribute dictionary
        :param dir_name: directory dictionary
        :param solver: optional attribute to specify the solver to solve optimization problems """
        self.fleetctrl = fleetctrl
        self.routing_engine = fleetctrl.routing_engine
        self.solver = solver
        self.active_reservation_requests : Dict[Any, PlanRequest] = {}

    def add_reservation_request(self, plan_request : PlanRequest, sim_time : int):
        """ this function adds a new request which is treated as reservation 
        :param plan_request: PlanRequest obj
        :param sim_time: current simulation time"""
        LOG.debug(f"new reservation request at time {sim_time}: {plan_request}")
        self.active_reservation_requests[plan_request.get_rid_struct()] = plan_request
        plan_request.set_reservation_flag(True)

    @abstractmethod
    def user_confirms_booking(self, rid : Any, sim_time : int):
        """ this function is triggered when a reservation request accepted the service
        :param rid: request id
        :param sim_time: current simulation time 
        """
        pass

    @abstractmethod
    def user_cancels_request(self, rid : Any, simulation_time : int):
        """This method is triggered when a reveration is cancelled. This can trigger some database processes.

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        pass

    @abstractmethod
    def reveal_requests_for_online_optimization(self, sim_time : int) -> List[Any]:
        """ this function is triggered during the simulation and returns a list of request ids that should be treated as online requests in the global optimisation and
        the assignment process
        :param sim_time: current simulation time
        :return: list of request ids"""
        pass

    @abstractmethod
    def return_availability_constraints(self, sim_time : int) -> List[Tuple[tuple, float]]:
        """ this function returns a list of network positions with times where a vehicle has to be available to fullfill future reservations
        this information can be included in the assignment process 
        :param sim_time: current simulation time
        :return: list of (position, latest arrival time)"""
        pass

    @abstractmethod
    def return_immediate_reservation_offer(self, rid : Any, sim_time : int) -> TravellerOffer:
        """ this function returns an offer if possible for an reservation request in case an immediate offer is needed for a reservation request
        :param rid: request id
        :param sim_time: current simulation time
        :return: offer for request """
        pass

    @abstractmethod
    def time_trigger(self, sim_time : int):
        """ this function is triggered during the simulation time and might trigger reoptimization processes for example 
        :param sim_time: simulation time """
        pass
    
    def get_full_vehicle_plan_until(self, vid, sim_time, until_time=None) -> Tuple[int, VehiclePlan]:
        """ this function returns the full vehicle plan of a vehicle until a certain time
            this method is relevant is the reservation module scheduled an long term offline plan
            if none is available, the currently assigned vehicle plan is returned
        :param vid: vehicle id
        :param sim_time: current simulation time
        :param until_time: (optional) time until the plan should be returned
        :return: reservation_plan_id (None if no reservation plan), VehiclePlan"""
        veh_plan = self.fleetctrl.veh_plans[vid].copy()
        veh_plan.update_tt_and_check_plan(self.fleetctrl.sim_vehicles[vid], sim_time, self.routing_engine, keep_feasible=True)
        plan_id = None
        return plan_id, veh_plan
    
    def reassign_supporting_points(self, sim_time, external_plan_vid_matches):
        """ this method can be used to reassign supporting points to vehicles
        it raises an error if the assignment is not feasible, i.e. a plan is not assigned to vehicle anymore
        this method is only required for a revelation based reservation system
        :param sim_time: current simulation time
        :param external_plan_vid_matches: dict of vid -> plan_id for assignment (this dictonary does not have to be complete. vehicles not part of the dict are not reassigned)"""
        pass
    
    def get_supporting_point(self, sim_time, vid=None, plan_id=None) -> Tuple[Tuple[int, int, float], float]:
        """ this method returns the next supporting point of a vehicle or reservation plan
        either vid or plan_id has to be given
        -> vid referse to the currently assigned vehicle with the corresponding reservation plan
        -> plan_id refers to the offline plan id
        :param sim_time: current simulation time
        :param vid: vehicle id
        :param plan_id: offline plan id
        :return: tuple (assigned_vid, plan_id, start location, planned starting time) of next supporting point"""
        return None, None, None, None
    
    def get_upcoming_unassigned_reservation_requests(self, t0, t1, with_assigned=False):
        """ this function returns exact future request attributes of unassigned reservation requests in the intervall  [t0, t1] which can be used for repositioning
        :param t0: start of forecast time horizon
        :type t0: float
        :param t1: end of forecast time horizon
        :type t1: float
        :param request_attribute: name of the attribute of the request class. if given, only returns requests with this attribute
        :type request_attribute: str
        :param attribute_value: if and request_attribute given: only returns future requests with this attribute value
        :type attribute_value: type(request_attribute)
        :param scale: (not for this class) scales forecast distribution by this values
        :type scale: float
        :return: list of (time, origin_node, destination_node) of future requests
        :rtype: list of 3-tuples
        """
        return []
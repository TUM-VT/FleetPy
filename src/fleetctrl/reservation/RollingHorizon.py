from __future__ import annotations
from typing import Dict

from src.misc.globals import *
from src.fleetctrl.reservation.ReservationBase import ReservationBase
from src.fleetctrl.pooling.immediate.insertion import reservation_insertion_with_heuristics, simple_remove

import logging
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_RollingHorizonReservation = {
    "doc" :  """ this reservation class treats reservation requests with a naive rolling horizon approach:
            innitially reservation requests are assigned to vehicles by an insertion heuristic;
            this assignment is kept until the simulation time approches the earliest pickup time within the rolling horizon;
            then requests are revealed to the global optimisation and removed from the reservation class
            """,
    "inherit" : "ReservationBase",
    "input_parameters_mandatory": [G_RA_OPT_HOR],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class RollingHorizonReservation(ReservationBase):
    """ this reservation class treats reservation requests with a naive rolling horizon approach:
    innitially reservation requests are assigned to vehicles by an insertion heuristic;
    this assignment is kept until the simulation time approches the earliest pickup time within the rolling horizon;
    then requests are revealed to the global optimisation and removed from the reservation class
    """
    def __init__(self, fleetctrl, operator_attributes, dir_names, solver="Gurobi"):
        super().__init__(fleetctrl, operator_attributes, dir_names, solver=solver)
        self.rolling_horizon = operator_attributes[G_RA_OPT_HOR]
        self.sorted_rids_with_epa = []  # list of (rid, earliest pick up time rid) sorted by epa
        self.rid_to_assigned_vid = {} # rid -> vid

    def add_reservation_request(self, plan_request, sim_time):
        """ this function adds a new request which is treated as reservation 
        :param plan_request: PlanRequest obj
        :param sim_time: current simulation time"""
        super().add_reservation_request(plan_request, sim_time)

    def reveal_requests_for_online_optimization(self, sim_time):
        """ this function is triggered during the simulation and returns a list of request ids that should be treated as online requests in the global optimisation and
        the assignment process
        :param sim_time: current simulation time
        :return: list of request ids"""
        reveal_index = 0
        while reveal_index < len(self.sorted_rids_with_epa) and self.sorted_rids_with_epa[reveal_index][1] <= sim_time + self.rolling_horizon:
            reveal_index += 1
        to_return = [self.sorted_rids_with_epa[x][0] for x in range(reveal_index)]
        self.sorted_rids_with_epa = self.sorted_rids_with_epa[reveal_index:]
        for rid in to_return:
            del self.active_reservation_requests[rid]
            try:
                del self.rid_to_assigned_vid[rid]
            except KeyError:
                pass
        LOG.debug(f"reveal following reservation requests at time {sim_time} : {to_return}")
        return to_return

    def return_availability_constraints(self, sim_time):
        """ this function returns a list of network positions with times where a vehicle has to be available to fullfill future reservations
        this information can be included in the assignment process 
        :param sim_time: current simulation time
        :return: list of (position, latest arrival time)"""
        return []

    def return_immediate_reservation_offer(self, rid, sim_time):
        """ this function returns an offer if possible for an reservation request which has been added to the reservation module before 
        in this implementation, an offer is always returned discribed by the earliest and latest pick up time
        :param rid: request id
        :param sim_time: current simulation time
        :return: offer for request """
        prq = self.active_reservation_requests[rid]
        tuple_list = reservation_insertion_with_heuristics(sim_time, prq, self.fleetctrl, force_feasible_assignment=True)
        if len(tuple_list) > 0:
            best_tuple = min(tuple_list, key=lambda x:x[2])
            best_vid, best_plan, _ = best_tuple
            offer = self.fleetctrl._create_user_offer(prq, sim_time, assigned_vehicle_plan=best_plan)
            self.fleetctrl.assign_vehicle_plan(self.fleetctrl.sim_vehicles[best_vid], best_plan, sim_time)
            self.rid_to_assigned_vid[rid] = best_vid
            LOG.debug(f"offer for reservation request {rid}: {offer}")
        else:
            offer = self.fleetctrl._create_rejection(prq, sim_time)
            LOG.debug(f"no offer for reservation request {rid}")
        return offer

    def user_confirms_booking(self, rid, sim_time):
        """ in this implementation nothing has to be done since the assignment is made in the "return_reservation_offer" methode
        :param rid: request id
        :param sim_time: current simulation time        
        """
        plan_request = self.active_reservation_requests[rid]
        self.sorted_rids_with_epa.append( (plan_request.get_rid_struct(), plan_request.get_o_stop_info()[1]))
        self.sorted_rids_with_epa.sort(key=lambda x:x[1])

    def user_cancels_request(self, rid, simulation_time):
        """ in case a reservation request which could be assigned earlier cancels the request
        this function removes the request from the assigned vehicle plan and deletes all entries in the database
        :param rid: request id
        :param simulation_time: current simulation time
        """
        if self.rid_to_assigned_vid.get(rid) is not None:
            vid = self.rid_to_assigned_vid.get[rid]
            assigned_plan = self.fleetctrl.veh_plans[vid]
            veh_obj = self.fleetctrl.veh_plans[vid]
            new_plan = simple_remove(veh_obj, assigned_plan, rid, simulation_time,
                self.routing_engine, self.fleetctrl.vr_ctrl_f, self.fleetctrl.rq_dict, self.fleetctrl.const_bt, self.fleetctrl.add_bt)
            self.fleetctrl.assign_vehicle_plan(veh_obj, new_plan, simulation_time)
            del self.rid_to_assigned_vid[rid]
            del self.active_reservation_requests[rid]

    def time_trigger(self, sim_time):
        """ this function is triggered during the simulation time and might trigger reoptimization processes for example 
        :param sim_time: simulation time """
        pass
    
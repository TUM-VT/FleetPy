from __future__ import annotations
from typing import Dict

from src.misc.globals import *
from src.fleetctrl.reservation.ReservationBase import ReservationBase
from src.fleetctrl.pooling.immediate.insertion import reservation_insertion_with_heuristics, simple_remove
from src.fleetctrl.planning.VehiclePlan import VehiclePlan, BoardingPlanStop
from src.fleetctrl.pooling.batch.BatchAssignmentAlgorithmBase import SimulationVehicleStruct

import logging
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_RollingHorizonNoGuarantee = {
    "doc" :  """ this reservation class treats reservation requests with a naive rolling horizon approach:
            reservation requests are not initially assigned;
            when their pickup falls below the rolling horizon, they are added to the global optimisation but with a high assignment reward;
            reservation requests are not guaranteed to be served, but can be rejected when approach the short-term horizon if no feasible assignment is found; a high assignment reward for reservation requests can be set.
            on the other hand this approach is very simple and computationally efficient.
            """,
    "inherit" : "ReservationBase",
    "input_parameters_mandatory": [G_RA_OPT_HOR],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class RollingHorizonNoGuarantee(ReservationBase):
    """ this reservation class treats reservation requests with a naive rolling horizon approach:
            reservation requests are not initially assigned;
            when their pickup falls below the rolling horizon, they are added to the global optimisation but with a high assignment reward;
            reservation requests are not guaranteed to be served, but can be rejected when approach the short-term horizon if no feasible assignment is found; a high assignment reward for reservation requests can be set.
            on the other hand this approach is very simple and computationally efficient.
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
            prq = self.active_reservation_requests[rid]
            prq.compute_new_max_trip_time(self.routing_engine, self.fleetctrl.const_bt, self.fleetctrl.max_dtf, self.fleetctrl.add_cdt, self.fleetctrl.min_dtw, self.fleetctrl.max_cdt)
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
        veh_strct = SimulationVehicleStruct(self.fleetctrl.sim_vehicles[0], None, sim_time, self.routing_engine, empty_init=True)
        o_pos, epa ,_ = prq.get_o_stop_info()
        d_pos, _, mtt = prq.get_d_stop_info()
        veh_strct.pos = o_pos
        ps_1 = BoardingPlanStop(o_pos, boarding_dict={1:[rid]}, earliest_pickup_time_dict={rid:epa})
        ps_2 = BoardingPlanStop(d_pos, boarding_dict={-1:[rid]}, max_trip_time_dict={rid:mtt})
        veh_p = VehiclePlan(veh_strct, sim_time, self.routing_engine, [ps_1, ps_2])
    
        offer = self.fleetctrl._create_user_offer(prq, sim_time, assigned_vehicle_plan=veh_p)
        
        plan_request = self.active_reservation_requests[rid]
        self.sorted_rids_with_epa.append( (plan_request.get_rid_struct(), plan_request.get_o_stop_info()[1]))
        self.sorted_rids_with_epa.sort(key=lambda x:x[1])

        return offer

    def user_confirms_booking(self, rid, sim_time):
        """ in this implementation nothing has to be done since the assignment is made in the "return_reservation_offer" methode
        :param rid: request id
        :param sim_time: current simulation time        
        """
        pass

    def user_cancels_request(self, rid, simulation_time):
        """ in case a reservation request which could be assigned earlier cancels the request
        this function removes the request from the assigned vehicle plan and deletes all entries in the database
        :param rid: request id
        :param simulation_time: current simulation time
        """
        del self.active_reservation_requests[rid]
        to_del = None
        for i, entry in enumerate(self.sorted_rids_with_epa):
            if entry[0] == rid:
                to_del = i
                break
        if to_del is not None:
            self.sorted_rids_with_epa.pop(to_del)

    def time_trigger(self, sim_time):
        """ this function is triggered during the simulation time and might trigger reoptimization processes for example 
        :param sim_time: simulation time """
        
        for rid, prq in self.active_reservation_requests.items():
            if not prq.get_current_offer():
                offer = self.return_immediate_reservation_offer(rid, sim_time)
                
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
        rid_list = []
        for rid, epa in self.sorted_rids_with_epa:
            if t0 <= epa <= t1:
                rid_list.append(rid)
            elif epa > t1:
                break
        return_list = []
        for rid in rid_list:
            prq = self.active_reservation_requests[rid]
            return_list.append((prq.get_o_stop_info()[1], prq.get_o_stop_info()[0][0], prq.get_d_stop_info()[0][0]))
        LOG.debug(f"upcoming unassigned reservation requests in intervall [{t0}, {t1}]: {return_list}")
        return return_list
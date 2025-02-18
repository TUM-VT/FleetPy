# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
from src.demand.TravelerModels import RequestBase
from src.routing.NetworkBase import NetworkBase
from src.simulation.Offers import TravellerOffer

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000

class PlanRequest:
    """ this class is the main class describing customer requests with hard time constraints which is
    used for planing in the fleetcontrol modules. in comparison to the traveler classes additional parameters
    and attributes can be defined which are unique for each operator defining the service of the operator (i.e. time constraints, walking, ...)"""
    def __init__(self, rq : RequestBase, routing_engine : NetworkBase, min_wait_time : int=0, 
                 max_wait_time : int=LARGE_INT, max_detour_time_factor : float=None,
                 max_constant_detour_time : int=None, add_constant_detour_time : int=None, min_detour_time_window : int=None,
                 boarding_time : int=0, pickup_pos : tuple=None, dropoff_pos : tuple=None, 
                 walking_time_start : float=0, walking_time_end : float=0, sub_rid_id=None):
        """
        :param rq: reference to traveller object which is requesting a trip
        :param routing_engine: reference to network object
        :param min_wait_time: defines earliest pickup_time from request time
        :param max_wait_time: defines latest pickup time from request time
        :param max_detour_time_factor: defines relative increase of maximum allowed travel time relative to direct route travel time in %
        :param max_constant_detour_time: defines absolute increase of maximum allowed travel time relative to direct route travel time
        :param add_constant_detour_time: this detour time is added upon the detour after evaluating the max_detour_time_factor
        :param min_detour_time_window: this detour time describes the minimum allowed detour
        :param boarding_time: time needed for customer to board the vehicle
        :param pickup_pos: network position tuple of pick up (used if pickup differs from request origin)
        :param dropoff_pos: network position tuple of drop off (used if dropoff differs from request destination)
        :param walking_time_start: walking time from origin to pickup
        :param walking_time_end: walking time from dropoff to destination
        :param sub_rid_id: id of this plan request that differs from the traveller id; usefull if one customer can be represented by multiple plan requests
        """
        # copy from rq
        self.rid = rq.get_rid_struct()
        self.nr_pax = rq.nr_pax
        if sub_rid_id is not None:
            self.sub_rid_struct = (self.rid, sub_rid_id)
        else:
            self.sub_rid_struct = self.rid
        self.rq_time = rq.rq_time
        #
        if pickup_pos is None:
            self.o_pos = rq.o_pos
        else:
            self.o_pos = pickup_pos
        if dropoff_pos is None:
            self.d_pos = rq.d_pos
        else:
            self.d_pos = dropoff_pos
        #
        self.walking_time_start = walking_time_start
        self.walking_time_end = walking_time_end
        #
        _, self.init_direct_tt, self.init_direct_td = routing_engine.return_travel_costs_1to1(self.o_pos, self.d_pos)
        # decision/output
        self.service_vehicle = None
        self.pu_time = None
        # constraints -> only in operator rq-class [pu: pick-up | do: drop-off, both start with boarding process]
        # TODO # -> information can be used for vehicle search, optimization and computation objective function value
        self.reservation_flag = False
        if min_wait_time is None:
            min_wait_time = 0
        self.t_pu_earliest = max(self.rq_time + min_wait_time, rq.earliest_start_time)
        if max_wait_time is None:
            max_wait_time = LARGE_INT
        if rq.latest_start_time is None:
            self.t_pu_latest = self.t_pu_earliest + max_wait_time
        else:
            self.t_pu_latest = min(self.t_pu_earliest + max_wait_time, rq.latest_start_time)
        self.t_do_latest = LARGE_INT
        if rq.max_trip_time is not None:
            self.max_trip_time = rq.max_trip_time
        else:
            max_trip_time = self.init_direct_tt + boarding_time
            # LOG.debug(f"max trip time: {max_trip_time}")
            if not pd.isnull(max_detour_time_factor):
                max_trip_time = (100 + max_detour_time_factor) * max_trip_time / 100
                # LOG.debug(f"max trip time {max_trip_time} -> max detour factor {max_detour_time_factor}")
            if not pd.isnull(add_constant_detour_time):
                max_trip_time += add_constant_detour_time
                # LOG.debug(f"max trip time {max_trip_time} -> add_constant_detour_time {add_constant_detour_time}")
            if not pd.isnull(min_detour_time_window):
                max_trip_time = max(self.init_direct_tt + boarding_time + min_detour_time_window, max_trip_time)
                # LOG.debug(f"max trip time {max_trip_time} -> min_detour_time_window {min_detour_time_window}")
            if not pd.isnull(max_constant_detour_time):
                max_trip_time = min(self.init_direct_tt + boarding_time + max_constant_detour_time, max_trip_time)
                # LOG.debug(f"max trip time {max_trip_time} -> max_constant_detour_time {max_constant_detour_time}")
            self.max_trip_time = max_trip_time
            if self.max_trip_time == self.init_direct_tt + boarding_time:
                self.max_trip_time = LARGE_INT
        self.t_do_latest = self.t_pu_latest + self.max_trip_time
        self.locked = False
        # LOG.debug(f"new PlanRequest: rid {self.rid}|{self.sub_rid_struct} start {self.o_pos} dest {self.d_pos} epa
        # {self.t_pu_earliest} lpa {self.t_pu_latest} dtt {self.init_direct_tt} mtt {self.max_trip_time}")
        # offer
        self.offer = None
        self.status = G_PRQS_NO_OFFER
        self.expected_pickup_time = None
        self.expected_dropoff_time = None

    def __str__(self):
        return f"new PlanRequest: rid {self.rid}|{self.sub_rid_struct} at {self.rq_time} start {self.o_pos} dest" \
               f"{self.d_pos} epa {self.t_pu_earliest} lpa {self.t_pu_latest} dtt {self.init_direct_tt} mtt" \
               f"{self.max_trip_time}"

    def set_reservation_flag(self, value : bool):
        """ this method sets a flag in case it is treated as reservation requests
        :param value: True, if reservation request"""
        self.reservation_flag = value

    def get_reservation_flag(self) -> bool:
        """ returns if request is treated as reservation request 
        :return: reservation flag"""
        return self.reservation_flag

    def get_rid_struct(self):
        """ this function returns the id of the plan request
        i.e. the traveler id, or the sub_rid id of the plan request if given
        :return: unique plan request id"""
        if not self.sub_rid_struct:
            return self.rid
        else:
            return self.sub_rid_struct

    def get_rid(self):
        """ this function returns the id of the traveller represented by the plan request
        :return: traveller id"""
        return self.rid

    def get_rq_time(self) -> float:
        """ returns the request time of the plan request
        :return: request time"""
        return self.rq_time

    def get_o_stop_info(self) -> tuple:
        """ returns a three tuple defining information about the request pick up
        :return: tuple of (origin position, earliest pickup time, latest pickup time)"""
        # LOG.debug("get o stop info {} {} {}".format(self.o_pos, self.t_pu_earliest, self.t_pu_latest))
        return self.o_pos, self.t_pu_earliest, self.t_pu_latest

    def get_d_stop_info(self):
        """ returns a three tuple defining information about request drop off
        :return: tuple of (destination pos, latest drop off time, maximum trip time)"""
        return self.d_pos, self.t_do_latest, self.max_trip_time

    def set_pickup(self, vid : int, simulation_time : int):
        """ this function should be called when the plan request is pickup up by a vehicle
        :param vid: vehicle id
        :param simulation_time: simulation time of pickup"""
        self.service_vehicle = vid
        self.pu_time = simulation_time
        self.status = G_PRQS_IN_VEH
        self.set_reservation_flag(False)

    def get_current_offer(self) -> TravellerOffer:
        """ returns the latest offer this plan request recieved
        :return: offer or None, if no offer has been made yet"""
        return self.offer

    def set_service_offered(self, offer : TravellerOffer):
        """ this method marks the plan request that a service has been offered
        :param offer: offer object
        """
        self.offer = offer
        LOG.debug(f"rid {self.get_rid_struct()} recieved offer : {offer}")
        if self.status < G_PRQS_INIT_OFFER:
            self.status = G_PRQS_INIT_OFFER

    def set_service_accepted(self):
        """ this method should be called when a request accepts an offer
        if this is the first time the status is set to 3
        if the requests already accepted an offer or is already assigned to a vehicle,
            the status is set to 5
        """
        if self.status < G_PRQS_ACC_OFFER:
            self.status = G_PRQS_ACC_OFFER

    def set_assigned(self, expected_pickup_time : int, expected_dropoff_time : int):
        """ on the one hand this method updates the status of the request to 4 in case the previous status
        was < 4 (status for unassigned requests)
        additionally the assigned_veh_plan is used to updated expected pick-up-times and expected drop-off-times
        :param expected_pickup_time: pickup time based on assigned plan
        :type expected_pickup_time: float
        :param expected_dropoff_time: dropoff time based on assigned plan
        :type expected_dropoff_time: float
        """
        self.expected_pickup_time = expected_pickup_time
        self.expected_dropoff_time = expected_dropoff_time

    def set_new_pickup_time_constraint(self, new_latest_pu_time : int, new_earliest_pu_time :int=None):
        """ this function is used to update pickup time constraints of the plan request
        :param new_latest_pu_time: new latest pickup time
        :param new_earliest_pu_time: (optional) new earliest pickup time if given
        """
        self.t_pu_latest = new_latest_pu_time
        if new_earliest_pu_time is not None:
            self.t_pu_earliest = new_earliest_pu_time
        # LOG.debug("after: {}".format(self))

    def set_new_max_trip_time(self, new_max_trip_time : float):
        """ this function updates the maximum trip time constraint
        :param new_max_trip_time: new maximum trip time"""
        self.max_trip_time = new_max_trip_time
        self.t_do_latest = self.t_pu_latest + self.max_trip_time

    def set_new_pickup_and_dropoff_positions(self, pickup_position : tuple, dropoff_position : tuple):
        """ this function is used to change pickup and drop off locations of the plan request
        :param pickup_position: network position of pickup
        :param dropoff_position: network position of drop off"""
        self.o_pos = pickup_position
        self.d_pos = dropoff_position

    def lock_request(self):
        """ this method set the request as locked"""
        if self.status < G_PRQS_LOCKED:
            self.status = G_PRQS_LOCKED
        self.locked = True

    def is_locked(self) -> bool:
        """ this function returns if the request is locked
        :return: True if locked"""
        return self.locked
    
    def is_parcel(self) -> bool:
        return False


class SoftConstraintPlanRequest(PlanRequest):
    """This class of PlanRequests has to be utilized for FleetControl classes with soft time windows, i.e.
    objective functions with time-window related penalty terms.
    Important: hard time constraints are still valid! Have to be set to large values if they should be ignored.
    """

    def __init__(self, rq : RequestBase, routing_engine : NetworkBase, min_wait_time : int=0, 
                 max_wait_time : int=LARGE_INT, max_detour_time_factor : float=None,
                 max_constant_detour_time : int=None, add_constant_detour_time : int=None, min_detour_time_window : int=None,
                 boarding_time : int=0, pickup_pos : tuple=None, dropoff_pos : tuple=None, 
                 walking_time_start : float=0, walking_time_end : float=0, sub_rid_id=None,
                 ept_soft : int=None, lpt_soft : int=None, edt_soft : int=None, ldt_soft : int=None):
        """ additional input parameters relative to PlanRequest:
        :param ept_soft: soft earliest pickup time
        :param lpt_soft: soft latest pickup time
        :param edt_soft: soft earliest drop off time
        :param ldt_soft: soft latest drop off time"""
        super().__init__(rq, routing_engine, min_wait_time=min_wait_time, max_wait_time=max_wait_time, max_detour_time_factor=max_detour_time_factor,
                         max_constant_detour_time=max_constant_detour_time, add_constant_detour_time=add_constant_detour_time, min_detour_time_window=min_detour_time_window,
                         boarding_time=boarding_time, pickup_pos=pickup_pos, dropoff_pos=dropoff_pos,
                         walking_time_start=walking_time_start, walking_time_end=walking_time_end, sub_rid_id=sub_rid_id)
        # soft pick-up time window
        self.ept_soft = ept_soft
        self.lpt_soft = lpt_soft
        # soft drop-off time window
        self.edt_soft = edt_soft
        self.ldt_soft = ldt_soft
        # soft max trip time
        self.max_trip_time_soft = self.max_trip_time

    def set_soft_pu_constraints(self, new_lpt_soft : int, new_ept_soft : int=None, remove_hard : bool=False):
        """ this methods set soft pickup time constraints
        :param new_lpt: new soft latest pick up time constraint
        :param new_ept_soft: (optional) new soft earliest pick up time constraint
        :param remove_hard: if True, the hard time constraints are removed"""
        if new_lpt_soft is not None:
            self.lpt_soft = new_lpt_soft
            if remove_hard:
                self.t_pu_latest = LARGE_INT
        if new_ept_soft is not None:
            self.ept_soft = new_ept_soft
            if remove_hard:
                self.t_pu_earliest = 0

    def get_soft_o_stop_info(self) -> tuple:
        """ returns a three tuple with information of the soft pick up
        :return: tuple of (pick up position, soft earliest pickup time, soft latest pickup time)"""
        return self.o_pos, self.ept_soft, self.lpt_soft

    def set_soft_do_constraints(self, new_ldt_soft : int, new_edt_soft : int=None, remove_hard=False):
        """ this method sets new soft drop off constraints
        :param new_ldt_soft: new soft latest dropoff time
        :param new_edt_soft: new soft earliest dropoff time
        :param remove_hard: if True, the hard time constraints are removed"""
        if new_ldt_soft is not None:
            self.ldt_soft = new_ldt_soft
            if remove_hard:
                self.t_do_latest = LARGE_INT
        if new_edt_soft is not None:
            # there is no hard constraint on earliest drop-off time
            self.edt_soft = new_edt_soft

    def get_soft_d_stop_info(self) -> tuple:
        """ returns a three tuple with information of the soft drop off up
        :return: tuple of (drop off position, soft earliest dropoff time, soft latest dropoff time)"""
        return self.d_pos, self.edt_soft, self.ldt_soft

    def set_soft_max_travel_time_constraint(self, new_soft_max_tt : float, remove_hard : bool=False):
        """ sets a new soft max travel time constraint
        :param new_sot_max_tt: new soft maximum travel time
        :param remove_hard: if True, hard time constraints are removed"""
        self.max_trip_time_soft = new_soft_max_tt
        if remove_hard:
            self.max_trip_time = LARGE_INT

    def get_soft_max_travel_time_constraint(self):
        """ returns the soft max travel time constraint
        :return: soft max travel time constraint"""
        return self.max_trip_time_soft
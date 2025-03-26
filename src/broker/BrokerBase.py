# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import time
import typing as tp
from abc import abstractmethod, ABCMeta

# additional module imports (> requirements)
# ------------------------------------------


# src imports
# -----------
if tp.TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.pt.PTControlBase import PTControlBase
    from src.demand.TravelerModels import RequestBase

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000
BUFFER_SIZE = 100

INPUT_PARAMETERS_BrokerBase = {
    "doc" : "this class is the base class representing a broker platform",
    "inherit" : None,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class BrokerBase():
    def __init__(self, n_amod_op: int, amod_operators: tp.List['FleetControlBase'], pt_operator: tp.Optional['PTControlBase'] = None):
        # TODO: The status update of the vehicles is not yet considered in the broker
        # TODO: The PT 
        self.n_amod_op = n_amod_op
        self.amod_operators = amod_operators
        self.pt_operator = pt_operator

    def inform_network_travel_time_update(self, sim_time):
        """This method informs the broker that the network travel times have been updated.
        This information is forwarded to the operators.
        """
        for op_id in range(self.n_amod_op):
            self.amod_operators[op_id].inform_network_travel_time_update(sim_time)
    
    def inform_request(self, rid, rq_obj, sim_time):
        """This method informs the broker that a new request has been made.
        This information is forwarded to the operators.
        """
        for op_id in range(self.n_amod_op):
            LOG.debug(f"Request {rid}: To operator {op_id} ...")
            self.amod_operators[op_id].user_request(rq_obj, sim_time)    

    def collect_offers(self, rid, rq_obj):
        """This method collects the offers from the operators.
        The return value is a list of tuples, where each tuple contains the operator id, the offer, and the simulation time.
        """
        amod_offers = []
        for op_id in range(self.n_amod_op):
            amod_offer = self.amod_operators[op_id].get_current_offer(rid)
            LOG.debug(f"amod offer {amod_offer}")
            if amod_offer is not None:
                amod_offers.append((op_id, amod_offer))
        return amod_offers

    def inform_user_booking(self, rid, rq_obj, sim_time, chosen_operator) -> tp.List[tuple[any, 'RequestBase']]:
        """This method informs the broker that the user has booked a trip.
        The return value can inform the FleetSimulationBase class about whether the sub request should be created and stored in the demand class
        """
        amod_confirmed_rids = []
        # amode_op_id: 0-n
        for i, operator in enumerate(self.amod_operators):
            if i != chosen_operator:
                operator.user_cancels_request(rid, sim_time)
            else:
                operator.user_confirms_booking(rid, sim_time)
                amod_confirmed_rids.append((rid, rq_obj))
        return amod_confirmed_rids

    def inform_user_leaving_system(self, rid, sim_time):
        """This method informs the broker that the user is leaving the system.
        """
        for _, operator in enumerate(self.amod_operators):
            operator.user_cancels_request(rid, sim_time)

    def inform_waiting_request_cancellations(self, chosen_operator, rid, sim_time):
        """This method informs the operators that the waiting requests have been cancelled.
        """
        self.amod_operators[chosen_operator].user_cancels_request(rid, sim_time)
    
    def acknowledge_user_boarding(self, op_id, rid, vid, boarding_time):
        """This method acknowledges the user boarding.
        """
        self.amod_operators[op_id].acknowledge_boarding(rid, vid, boarding_time)

    def acknowledge_user_alighting(self, op_id, rid, vid, alighting_time):
        """This method acknowledges the user alighting.
        """
        self.amod_operators[op_id].acknowledge_alighting(rid, vid, alighting_time)

    def receive_status_update(self, op_id, vid, sim_time, passed_VRL, force_update_plan):
        """This method receives the status update of the vehicles.
        """
        self.amod_operators[op_id].receive_status_update(vid, sim_time, passed_VRL, force_update_plan)


# TODO: Be careful with the request id in all methods
# TODO: Only the communication between the broker and the operators should be defined here, the rest should be in the FleetSimulationBase classs
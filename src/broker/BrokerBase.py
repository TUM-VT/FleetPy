# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import time
from abc import abstractmethod, ABCMeta

# additional module imports (> requirements)
# ------------------------------------------


# src imports
# -----------


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
    def __init__(self):
        # TODO: The status update of the vehicles is not yet considered in the broker
        pass
    
    def inform_network_travel_time_update(self, sim_time):
        """This method informs the broker that the network travel times have been updated.
        This information is forwarded to the operators.
        """
        for op_id in range(self.n_op):
            self.operators[op_id].inform_network_travel_time_update(sim_time)
    
    def inform_request(self, rid, rq_obj, sim_time):
        """This method informs the broker that a new request has been made.
        This information is forwarded to the operators.
        """
        for op_id in range(self.n_op):
            LOG.debug(f"Request {rid}: To operator {op_id} ...")
            self.operators[op_id].user_request(rq_obj, sim_time)    

    def collect_offers(self, rid, rq_obj, sim_time):
        """This method collects the offers from the operators.
        """
        amod_offers = []
        for op_id in range(self.n_op):
                amod_offer = self.operators[op_id].get_current_offer(rid)
                LOG.debug(f"amod offer {amod_offer}")
                if amod_offer is not None:
                    amod_offers.append((op_id, amod_offer, sim_time))
        return amod_offers

    def inform_user_booking(self, rid, rq_obj, sim_time, chosen_operator) -> list[str]:
        """This method informs the broker that the user has booked a trip.
        The return value can inform the FleetSimulationBase class about whether the sub request should be created and stored in the demand class
        """
        for i, operator in enumerate(self.operators):
            if i != chosen_operator:
                operator.user_cancels_request(rid, sim_time)
            else:
                operator.user_confirms_booking(rid, sim_time)
                # TODO: the line below should be in the FleetSimulationBase class
                self.demand.waiting_rq[rid] = rq_obj
        return []

    def inform_user_leaving_system(self, rid, rq_obj, sim_time):
        """This method informs the broker that the user is leaving the system.
        """
        for i, operator in enumerate(self.operators):
            operator.user_cancels_request(rid, sim_time)

    def inform_waiting_request_cancellations(self, chosen_operator, rid, sim_time):
        """This method informs the operators that the waiting requests have been cancelled.
        """
        self.operators[chosen_operator].user_cancels_request(rid, sim_time)
    
    def acknowledge_user_boarding(self, rid, vid, op_id, boarding_time):
        """This method acknowledges the user boarding.
        """
        self.operators[op_id].acknowledge_boarding(rid, vid, boarding_time)

    def acknowledge_user_alighting(self, rid, vid, op_id, alighting_time):
        """This method acknowledges the user alighting.
        """
        self.operators[op_id].acknowledge_alighting(rid, vid, alighting_time)

    def receive_status_update(self, vid, op_id, sim_time, passed_VRL, force_update_plan):
        """This method receives the status update of the vehicles.
        """
        self.operators[op_id].receive_status_update(vid, sim_time, passed_VRL, force_update_plan)


# TODO: Be careful with the request id in all methods
# TODO: Only the communication between the broker and the operators should be defined here, the rest should be in the FleetSimulationBase class
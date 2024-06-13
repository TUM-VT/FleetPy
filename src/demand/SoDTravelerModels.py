from src.demand.TravelerModels import BasicRequest, offer_str

import logging
import numpy as np
import math

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_SoDRequest = {
    "doc": "This request only performs a mode choice based on if it received an offer or not. if an offer is received,"
           " it accepts the offer. if multiple offers are received an error is thrown",
    "inherit": "RequestBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


class SoDRequest(BasicRequest):
    """This request performs a mode choice based on walking distance.
        if an offer is received, it accepts the offer
        if multiple offers are received an error is thrown"""
    type = "SoDRequest"

    def __init__(self, rq_row, routing_engine, simulation_time_step, scenario_parameters):
        super().__init__(rq_row, routing_engine, simulation_time_step, scenario_parameters)
        self.walk_logit_beta = scenario_parameters.get(G_PT_WALK_LOGIT_BETA, 0)

    def choose_offer(self, sc_parameters, simulation_time):
        test_all_decline = super().choose_offer(sc_parameters, simulation_time)
        if test_all_decline is not None and test_all_decline < 0:
            return -1
        if len(self.offer) == 0:
            return None
        opts = [offer_id for offer_id, operator_offer in self.offer.items() if
                operator_offer is not None and not operator_offer.service_declined()]
        LOG.debug(f"Basic request choose offer: {self.rid} : {offer_str(self.offer)} | {opts}")
        if len(opts) == 0:
            return None
        elif len(opts) == 1:
            self.fare = self.offer[opts[0]].get(G_OFFER_FARE, 0)

            total_walking_dist = (self.offer[opts[0]].get(G_OFFER_WALKING_DISTANCE_ORIGIN, 0) +
                                  self.offer[opts[0]].get(G_OFFER_WALKING_DISTANCE_DESTINATION, 0))

            # decide probability based on walking distance
            prob_accept = self.return_walk_logit_prob(total_walking_dist / 1000)

            LOG.debug(f"Basic request: {self.rid} ; walking distance: {total_walking_dist} | prob {prob_accept:2f}")
            # decide if to accept the offer
            if np.random.rand() < prob_accept:
                LOG.debug(f"Basic request: {self.rid} ; accepted offer {opts[0]}")
                return opts[0]
            else:
                LOG.debug(f"Basic request: {self.rid} ; declined offer {opts[0]} for walking distance")
                return -1
        else:
            LOG.error(f"not implemented {offer_str(self.offer)}")

    def return_walk_logit_prob(self, walking_dist):
        """
        return the probability of accepting the offer based on walking distance
        :param walking_dist: walking distance in km (or compatible with walk_logit_beta)
        """
        prob = math.exp(self.walk_logit_beta * walking_dist)
        return prob

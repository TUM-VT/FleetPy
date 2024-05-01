from src.demand.TravelerModels import BasicRequest, offer_str

import logging
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_SoDRequest = {
    "doc" : "This request only performs a mode choice based on if it recieved an offer or not. if an offer is recieved, it accepts the offer. if multiple offers are recieved an error is thrown",
    "inherit" : "RequestBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class SoDRequest(BasicRequest):
    """This request performs a mode choice based on walking distance.
        if an offer is recieved, it accepts the offer
        if multiple offers are recieved an error is thrown"""
    type = "SoDRequest"

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

            max_walking_dist = sc_parameters.get("max_walking_dist", 500)
            total_walking_dist = self.offer[opts[0]].get(G_OFFER_WALKING_DISTANCE_ORIGIN, 0) + \
                                 self.offer[opts[0]].get(G_OFFER_WALKING_DISTANCE_DESTINATION, 0)

            # decide probability based on walking distance
            prob_accept = 1 - total_walking_dist / max_walking_dist

            LOG.debug(f"Basic request: {self.rid} ; walking distance: {total_walking_dist} | prob {prob_accept:2f}")
            # decide if to accept the offer
            if np.random.rand() < prob_accept:
                LOG.debug(f"Basic request: {self.rid} ; accepted offer {opts[0]}")
                return opts[0]
            else:
                LOG.debug(f"Basic request: {self.rid} ; declined offer {opts[0]}")
                return -1
        else:
            LOG.error(f"not implemented {offer_str(self.offer)}")
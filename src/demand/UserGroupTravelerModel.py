from src.demand.TravelerModels import BasicRequest, offer_str

import logging

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_UserGroupRequest = {
    "doc": "",
    "inherit": "RequestBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


class UserGroupRequest(BasicRequest):
    """User group request class"""
    type = "UserGroupRequest"

    def __init__(self, rq_row, routing_engine, simulation_time_step, scenario_parameters):
        super().__init__(rq_row, routing_engine, simulation_time_step, scenario_parameters)
        self.max_wait_time = rq_row[G_AR_MAX_WT]
        self.latest_start_time = self.earliest_start_time + \
            self.max_wait_time
        self.set_direct_route_travel_infos(routing_engine)
        self.rel_detour = rq_row.get(G_RQ_MRD, scenario_parameters[G_OP_MAX_DTF])
        self.max_trip_time = (100 + self.rel_detour) * (self.direct_route_travel_time +
                                                   scenario_parameters.get(G_OP_CONST_BT, 0)) / 100


        self.walking_speed = rq_row[G_WALKING_SPEED]
        self.max_walking_distance = rq_row[G_MAX_WALKING_DIST]
        self.value_of_time = rq_row.get(G_MC_VOT, 0.0)
        self.value_of_walking_factor = rq_row.get(G_VOW_FACTOR, 1.0)
        self.value_of_waiting_factor = rq_row.get(G_V_WAIT_FACTOR, 1.0)

    def choose_offer(self, sc_parameters, simulation_time):
        """Accept the first operator offer that satisfies this user group's
        wait time, detour and walking distance thresholds; decline (-1) if none do.
        Utility-based comparison across accepted offers is done in post-processing."""
        test_all_decline = super().choose_offer(sc_parameters, simulation_time)
        if test_all_decline is not None and test_all_decline < 0:
            return -1
        sorted_amod_offer_ops = sorted([op_id for op_id in self.offer.keys() if op_id >= 0])
        if len(sorted_amod_offer_ops) == 0:
            return None
        for op in sorted_amod_offer_ops:
            offer = self.offer[op]
            if offer.service_declined():
                continue
            offered_pu_t = self.rq_time + offer[G_OFFER_WAIT]
            if offered_pu_t > self.latest_start_time:
                LOG.debug(f" -> decline offer {op}. too late pick-up {offered_pu_t} > {self.latest_start_time}")
                continue
            if offer[G_OFFER_DRIVE] > self.max_trip_time:
                LOG.debug(f" -> decline offer {op}. too much detour {offer[G_OFFER_DRIVE]} > {self.max_trip_time}")
                continue
            walking_dist = offer.get(G_OFFER_WALKING_DISTANCE_ORIGIN, 0) + \
                offer.get(G_OFFER_WALKING_DISTANCE_DESTINATION, 0)
            if walking_dist > self.max_walking_distance:
                LOG.debug(f" -> decline offer {op}. too far to walk {walking_dist} > {self.max_walking_distance}")
                continue
            LOG.debug(f" -> accept offer {op}")
            self.fare = offer.get(G_OFFER_FARE, 0)
            return op
        LOG.debug(f"all offers over threshold, decline: {offer_str(self.offer)}")
        return -1

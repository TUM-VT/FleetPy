from src.demand.TravelerModels import BasicRequest, offer_str
from src.infra.BoardingPointInfrastructure import routing_min_distance_cost_function

import logging

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_UserGroupRequest = {
    "doc": """User group request that accepts the first offer meeting its wait time, detour and walking
    distance thresholds, and records the disutility of the chosen offer (utility_chosen_mode output column)
    based on the group's value of time and its waiting/walking weighting factors. If no offer is acceptable,
    a per-group no-offer/reliability penalty is recorded instead, reflecting the group's sensitivity to
    unreliable service.""",
    "inherit": "RequestBase",
    "input_parameters_mandatory": [G_AR_MAX_WT, G_WALKING_SPEED, G_MAX_WALKING_DIST],
    "input_parameters_optional": [G_RQ_MRD, G_MC_VOT, G_VOW_FACTOR, G_V_WAIT_FACTOR, G_MC_NO_OFFER_PENALTY],
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
        self.no_offer_penalty = rq_row.get(G_MC_NO_OFFER_PENALTY, 0.0)
        self.utility_chosen_mode = None

    def _get_walking_distance(self, offer):
        """ returns the walking distance (in m) associated with an offer. Default implementation
        reads it off the offer's walking-distance fields (populated by fleet control modules that
        support boarding-point selection, e.g. SemiOnDemand*); subclasses that pre-match a fixed
        boarding point at demand-generation time (which plain pooling fleet control does not
        populate offers with) override this to return a precomputed distance instead.
        :param offer: TravelerOffer of the operator
        :return: walking distance (m)
        """
        return offer.get(G_OFFER_WALKING_DISTANCE_ORIGIN, 0) + \
            offer.get(G_OFFER_WALKING_DISTANCE_DESTINATION, 0)

    def _compute_utility(self, offer):
        """ computes the disutility (in time-equivalent units) of an offer based on this user group's
        value of time and its waiting/walking weighting factors:
        utility = - vot * (v_wait_factor * t_wait + t_drive + v_walk_factor * t_walk)
        :param offer: TravelerOffer of the operator
        :return: utility value (float)
        """
        t_wait = offer[G_OFFER_WAIT]
        t_drive = offer[G_OFFER_DRIVE]
        walking_dist = self._get_walking_distance(offer)
        t_walk = walking_dist / self.walking_speed if self.walking_speed else 0
        return - self.value_of_time * (self.value_of_waiting_factor * t_wait + t_drive +
                                        self.value_of_walking_factor * t_walk)

    def _add_record(self, record_dict):
        record_dict[G_RQ_C_UTIL] = self.utility_chosen_mode
        record_dict["user_group"] = getattr(self, "user_group", None)
        return super()._add_record(record_dict)

    def choose_offer(self, sc_parameters, simulation_time):
        """Accept the first operator offer that satisfies this user group's
        wait time, detour and walking distance thresholds; decline (-1) if none do.
        A declined outcome records a per-group no-offer/reliability penalty as its
        utility, instead of the usual wait/drive/walk disutility. Utility-based
        comparison across accepted offers is done in post-processing."""
        test_all_decline = super().choose_offer(sc_parameters, simulation_time)
        if test_all_decline is not None and test_all_decline < 0:
            self.utility_chosen_mode = -self.no_offer_penalty
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
            walking_dist = self._get_walking_distance(offer)
            if walking_dist > self.max_walking_distance:
                LOG.debug(f" -> decline offer {op}. too far to walk {walking_dist} > {self.max_walking_distance}")
                continue
            LOG.debug(f" -> accept offer {op}")
            self.fare = offer.get(G_OFFER_FARE, 0)
            self.utility_chosen_mode = self._compute_utility(offer)
            return op
        LOG.debug(f"all offers over threshold, decline: {offer_str(self.offer)}")
        self.utility_chosen_mode = -self.no_offer_penalty
        return -1


INPUT_PARAMETERS_StopBasedUserGroupRequest = {
    "doc": """Stop-based on-demand variant of UserGroupRequest: the vehicle serves a
    pre-matched boarding point (G_RQ_BOARDING_NODE, see demand_utils.generate_demand_scenario)
    on the non-hub side of the trip instead of the traveler's true location. The walking
    distance between the true location and that boarding point is computed once at request
    creation and used in place of offer-supplied walking distances (plain pooling fleet control
    does not populate those offer fields; only SemiOnDemand* fleet control does), so the usual
    UserGroupRequest wait/detour/walking-threshold and utility logic still applies unchanged.""",
    "inherit": "UserGroupRequest",
    "input_parameters_mandatory": [G_RQ_BOARDING_NODE, G_RQ_DIRECTION],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


class StopBasedUserGroupRequest(UserGroupRequest):
    """Stop-based on-demand request: served at a pre-matched boarding point instead of the
    traveler's true location on the non-hub side of the trip."""
    type = "StopBasedUserGroupRequest"

    def __init__(self, rq_row, routing_engine, simulation_time_step, scenario_parameters):
        true_o_node = int(rq_row[G_RQ_ORIGIN])
        true_d_node = int(rq_row[G_RQ_DESTINATION])
        boarding_node = int(rq_row[G_RQ_BOARDING_NODE])
        direction = rq_row[G_RQ_DIRECTION]

        stop_based_row = rq_row.copy()
        if direction == G_DIR_TO_HUB:
            stop_based_row[G_RQ_ORIGIN] = boarding_node
        else:
            stop_based_row[G_RQ_DESTINATION] = boarding_node

        super().__init__(stop_based_row, routing_engine, simulation_time_step, scenario_parameters)

        self.true_o_node = true_o_node
        self.true_d_node = true_d_node
        self.total_walking_distance = 0.0
        if true_o_node != self.o_node:
            self.total_walking_distance += self._walking_distance(
                routing_engine, routing_engine.return_node_position(true_o_node), self.o_pos)
        if true_d_node != self.d_node:
            self.total_walking_distance += self._walking_distance(
                routing_engine, routing_engine.return_node_position(true_d_node), self.d_pos)

    @staticmethod
    def _walking_distance(routing_engine, true_pos, boarding_pos):
        """ network walking distance (m) between a true request position and its matched
        boarding point, same distance-minimizing routing used by
        BoardingPointInfrastructure.return_walking_distance."""
        _, _, distance = routing_engine.return_travel_costs_1to1(
            true_pos, boarding_pos, customized_section_cost_function=routing_min_distance_cost_function)
        return distance

    def _get_walking_distance(self, offer):
        return self.total_walking_distance

    def _add_record(self, record_dict):
        record_dict["true_o_node"] = self.true_o_node
        record_dict["true_d_node"] = self.true_d_node
        record_dict["total_walking_distance"] = self.total_walking_distance
        return super()._add_record(record_dict)

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
    the disutility of walking the whole direct trip is recorded instead (the realistic fallback for a
    declined traveler), plus the group's no-offer/reliability penalty on top, reflecting the group's
    sensitivity to unreliable service.

    value_of_time is configured in EUR/HOUR (human-readable -- e.g. 10, 15, 5) and converted to
    EUR/second here at read time (divided by 3600), since utility_chosen_mode's time terms
    (t_wait/t_drive/t_walk) are all in seconds and need a per-second rate to come out monetized
    in EUR, negative (FleetPy's native utility sign, not yet flipped to a disutility);
    no_offer_penalty is in SECONDS (not EUR) -- a group-specific extra wait-equivalent penalty
    for being declined, monetized the same way ordinary wait time is (self.value_of_time *
    value_of_waiting_factor, both already per-second by that point), see _decline_utility.""",
    "inherit": "RequestBase",
    "input_parameters_mandatory": [G_AR_MAX_WT, G_WALKING_SPEED, G_MAX_WALKING_DIST],
    "input_parameters_optional": [G_RQ_MRD, G_RQ_ACDT, G_MC_VOT, G_VOW_FACTOR, G_V_WAIT_FACTOR, G_MC_NO_OFFER_PENALTY],
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
        # add_constant_detour_time: an ABSOLUTE addition on top of the percentage-based
        # max_rel_detour cap, applied per-request (unlike G_OP_ADD_CDT, the operator-side
        # equivalent in PlanRequest.py -- that one never applies here, since UserGroupRequest
        # always sets its own max_trip_time, see PlanRequest.py:81-82). Without this, a short
        # trip has very little absolute slack to absorb a fixed per-stop dwell-time overhead
        # (e.g. a headway-scheduled PT line's boarding time) before busting a % cap, while a
        # long trip comfortably absorbs the same fixed overhead as a smaller relative share --
        # structurally biasing declines against short trips. Defaults to 0 (no behavior change
        # unless a ranges file sets it).
        self.add_constant_detour_time = rq_row.get(G_RQ_ACDT, 0.0)
        self.max_trip_time = ((100 + self.rel_detour) * (self.direct_route_travel_time +
                                                   scenario_parameters.get(G_OP_CONST_BT, 0)) / 100
                               + self.add_constant_detour_time)


        self.walking_speed = rq_row[G_WALKING_SPEED]
        self.max_walking_distance = rq_row[G_MAX_WALKING_DIST]
        # G_MC_VOT is configured in EUR/hour (see class docstring) -- convert to EUR/second here,
        # once, so every downstream utility formula can keep working directly against seconds.
        self.value_of_time = rq_row.get(G_MC_VOT, 0.0) / 3600.0
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

    def _walking_fallback_utility(self):
        """ disutility of walking the whole direct trip instead of being served -- the realistic
        fallback for a declined request, using the same value-of-time/walking-factor weighting as
        _compute_utility's walking term, just applied to the full direct_route_travel_distance
        instead of an offer's (partial) walking_dist:
        utility = - vot * v_walk_factor * (direct_route_travel_distance / walking_speed)
        :return: utility value (float)
        """
        if not self.direct_route_travel_distance:
            return 0.0
        t_walk = self.direct_route_travel_distance / self.walking_speed if self.walking_speed else 0
        return - self.value_of_time * self.value_of_walking_factor * t_walk

    def _decline_utility(self):
        """ utility recorded for a declined/no-offer outcome: the walking-fallback disutility
        (what the traveler actually experiences -- walking the trip) plus the group's no-offer
        penalty (an additional, group-specific reliability penalty on top of that realistic
        fallback, in SECONDS -- 0 for groups that aren't reliability-sensitive, see
        G_MC_NO_OFFER_PENALTY), plus the disutility of how long the service actually took to
        answer with a decline (self.leave_system_time, set immediately before this is called,
        both for an active decline and for a request that gives up at its own decision deadline
        -- see leaves_system() -- covers both outcomes uniformly). Both extra terms are
        monetized the same way as ordinary wait time (value_of_time * value_of_waiting_factor):
        the traveler doesn't know they're being declined until this moment, so that time (plus
        the group's own reliability penalty) is spent waiting just like the wait leg of a trip
        that gets accepted.
        :return: utility value (float)
        """
        wait_for_answer = 0.0
        if self.leave_system_time is not None:
            wait_for_answer = self.leave_system_time - self.rq_time
        return (self._walking_fallback_utility()
                - self.value_of_time * self.value_of_waiting_factor
                * (self.no_offer_penalty + wait_for_answer))

    def leaves_system(self, sim_time):
        """ choose_offer() only calls _decline_utility() when it actively returns -1 (an
        all-offers-declined outcome). A request that never receives any offer at all instead
        stays "undecided" (choose_offer returns None) until its OWN user_max_decision_time
        deadline expires here -- without this override, utility_chosen_mode would stay at its
        None default for that outcome, silently dropping it (rather than recording its
        decline disutility) from any downstream utility average.
        :return: True/False, see RequestBase.leaves_system
        """
        left = super().leaves_system(sim_time)
        if left and self.utility_chosen_mode is None:
            self.utility_chosen_mode = self._decline_utility()
        return left

    def _add_record(self, record_dict):
        record_dict[G_RQ_C_UTIL] = self.utility_chosen_mode
        record_dict["user_group"] = getattr(self, "user_group", None)
        return super()._add_record(record_dict)

    def choose_offer(self, sc_parameters, simulation_time):
        """Accept the first operator offer that satisfies this user group's
        wait time, detour and walking distance thresholds; decline (-1) if none do.
        A declined outcome records the disutility of walking the trip instead (plus the group's
        no-offer/reliability penalty on top), instead of the usual wait/drive/walk disutility.
        Utility-based comparison across accepted offers is done in post-processing."""
        test_all_decline = super().choose_offer(sc_parameters, simulation_time)
        if test_all_decline is not None and test_all_decline < 0:
            self.leave_system_time = simulation_time
            self.utility_chosen_mode = self._decline_utility()
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
        self.leave_system_time = simulation_time
        self.utility_chosen_mode = self._decline_utility()
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
            access_walking_distance = self._walking_distance(
                routing_engine, routing_engine.return_node_position(true_o_node), self.o_pos)
            self.total_walking_distance += access_walking_distance
            if self.walking_speed:
                t_access_walk = access_walking_distance / self.walking_speed
                self.earliest_start_time += t_access_walk
                if self.latest_start_time is not None:
                    self.latest_start_time += t_access_walk
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

    def _walking_fallback_utility(self):
        """ direct_route_travel_distance here only covers the boarding-node<->hub leg (see
        __init__, which swaps one end of the trip for the matched boarding_node); the
        true-location<->boarding_node leg is tracked separately as total_walking_distance. Sum
        both for the full door-to-door distance a declined traveler would actually have to walk.
        """
        if not self.walking_speed:
            return 0.0
        full_distance = (self.direct_route_travel_distance or 0.0) + self.total_walking_distance
        t_walk = full_distance / self.walking_speed
        return - self.value_of_time * self.value_of_walking_factor * t_walk

    def _add_record(self, record_dict):
        record_dict["true_o_node"] = self.true_o_node
        record_dict["true_d_node"] = self.true_d_node
        record_dict["total_walking_distance"] = self.total_walking_distance
        return super()._add_record(record_dict)

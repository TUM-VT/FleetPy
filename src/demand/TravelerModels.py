# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import os
from copy import deepcopy
from abc import abstractmethod, ABCMeta

# additional module imports (> requirements)
# ------------------------------------------
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # TODO # disables warning when overwriting Dataframes

# src imports
# -----------
from src.misc.functions import PiecewiseContinuousLinearFunction
from src.routing.NetworkBase import return_position_str
# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------------------------- #


def offer_str(rq_offer):
    """ this function converts the offer_dict of travelers to a string for debugging """
    return ", ".join(["{}:{}".format(k, str(v)) for k, v in rq_offer.items()])


# -------------------------------------------------------------------------------------------------------------------- #
# Traveler Model Classes
# ----------------------
INPUT_PARAMETERS_RequestBase = {
    "doc" : "this is the base simulation class used for all traveler classes within FleetPy",
    "inherit" : None,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        G_AR_MIN_WT
    ],
    "mandatory_modules": [], 
    "optional_modules": []
}

class RequestBase(metaclass=ABCMeta):
    """Base class for customer requests."""
    type = "RequestBase"

    def __init__(self, rq_row, routing_engine, simulation_time_step, scenario_parameters):
        # input
        self.rid = int(rq_row.get(G_RQ_ID, rq_row.name))  # request id is index of dataframe
        self.sub_rid_struct = None
        self.is_parcel = False  # requests are usually persons
        self.rq_time = rq_row[G_RQ_TIME] - rq_row[G_RQ_TIME] % simulation_time_step
        self.latest_decision_time = rq_row[G_RQ_LDT]
        self.earliest_start_time = self.rq_time
        if rq_row.get(G_RQ_EPT):
            self.earliest_start_time = rq_row.get(G_RQ_EPT)
        elif scenario_parameters.get(G_AR_MIN_WT):  # TODO RPP : auslagern in ParcelBase + definieren neuer global variable (parcel_min_wait_time)
            self.earliest_start_time = self.rq_time + scenario_parameters.get(G_AR_MIN_WT)
        self.latest_start_time = None
        self.max_trip_time = None
        self.nr_pax = rq_row.get(G_RQ_PAX, 1)   # TODO RPP: neue attribute für größe/menge/gewicht
        #
        self.o_node = int(rq_row[G_RQ_ORIGIN])
        self.o_pos = routing_engine.return_node_position(self.o_node)
        self.d_node = int(rq_row[G_RQ_DESTINATION])
        self.d_pos = routing_engine.return_node_position(self.d_node)
        # store miscellaneous custom values from demand file
        for param, value in rq_row.drop([G_RQ_TIME, G_RQ_ID, G_RQ_ORIGIN, G_RQ_DESTINATION]).items():
            setattr(self, str(param), value)
        # offer: operator_id > offer class entity
        self.offer = {}
        # decision/output
        self.leave_system_time = None
        self.chosen_operator_id = None
        self.service_opid = None
        self.service_vid = None
        self.pu_time = None
        self.pu_pos = None
        self.t_access = None
        self.do_time = None
        self.do_pos = None
        self.t_egress = None
        self.fare = None
        # direct_route_infos
        self.direct_route_travel_time = None
        self.direct_route_travel_distance = None
        # 
        self.modal_state = G_RQ_STATE_MONOMODAL # mono-modal trip by default 

    def get_rid(self):
        return self.rid

    def get_rid_struct(self):
        if self.sub_rid_struct is None:
            return self.rid
        else:
            return self.sub_rid_struct

    def get_origin_pos(self):
        return self.o_pos

    def get_destination_pos(self):
        return self.d_pos

    def get_origin_node(self):
        return self.o_node

    def get_destination_node(self):
        return self.d_node

    def return_offer(self, op_id):
        return self.offer.get(op_id)

    def get_chosen_operator(self):
        return self.chosen_operator_id

    def record_data(self):
        record_dict = {}
        # input
        if self.sub_rid_struct is not None:
            rid_str = f"{self.sub_rid_struct}"
        else:
            rid_str = f"{self.rid}"
        record_dict[G_RQ_ID] = rid_str
        record_dict[G_RQ_TYPE] = self.type
        record_dict[G_RQ_PAX] = self.nr_pax
        record_dict[G_RQ_TIME] = self.rq_time
        record_dict[G_RQ_EPT] = self.earliest_start_time
        # # node output
        # record_dict[G_RQ_ORIGIN] = self.o_node
        # record_dict[G_RQ_DESTINATION] = self.d_node
        # position output
        record_dict[G_RQ_ORIGIN] = return_position_str(self.o_pos)
        record_dict[G_RQ_DESTINATION] = return_position_str(self.d_pos)
        if self.pu_pos is None or self.pu_pos == self.o_pos:
            record_dict[G_RQ_PUL] = ""
        else:
            record_dict[G_RQ_PUL] = return_position_str(self.pu_pos)
        if self.do_pos is None or self.do_pos == self.d_pos:
            record_dict[G_RQ_DOL] = ""
        else:
            record_dict[G_RQ_DOL] = return_position_str(self.do_pos)
        if self.t_access is None:
            record_dict[G_RQ_ACCESS] = ""
        else:
            record_dict[G_RQ_ACCESS] = self.t_access
        if self.t_egress is None:
            record_dict[G_RQ_EGRESS] = ""
        else:
            record_dict[G_RQ_EGRESS] = self.t_egress
        if self.direct_route_travel_time is not None:
            record_dict[G_RQ_DRT] = self.direct_route_travel_time
        if self.direct_route_travel_distance is not None:
            record_dict[G_RQ_DRD] = self.direct_route_travel_distance
        # offers
        all_offer_info = []
        for op_id, operator_offer in self.offer.items():
            all_offer_info.append(f"{op_id}:" + operator_offer.to_output_str())
        record_dict[G_RQ_OFFERS] = "|".join(all_offer_info)
        # decision-dependent
        record_dict[G_RQ_LEAVE_TIME] = self.leave_system_time  # TODO # when only adding stuff conditionally there will
        record_dict[G_RQ_CHOSEN_OP_ID] = self.chosen_operator_id  # TODO # be errors when evaluating
        record_dict[G_RQ_OP_ID] = self.service_opid
        record_dict[G_RQ_VID] = self.service_vid
        record_dict[G_RQ_PU] = self.pu_time
        record_dict[G_RQ_DO] = self.do_time
        record_dict[G_RQ_FARE] = self.fare
        record_dict[G_RQ_MODAL_STATE] = self.modal_state
        return self._add_record(record_dict)

    def receive_offer(self, operator_id, operator_offer, simulation_time, sc_parameters=None): # TODO remove sc_parameters here
        """ this function is used when a traveller recieves an offer from an operator
        :param operator_id: id of the corresponding operator
        :type operator_id: int
        :param operator_offer: entity of class TravelerOffer corresponding to the offer to the traveller
        :type operator_offer: TravelerOffer
        :param simulation_time: current simulation time
        :type simulation_time: int
        :param sc_parameters: scenario_parameter dict
        :type sc_parameters: dict
        """
        self.offer[operator_id] = operator_offer

    def retract_offer(self, operator_id):
        """ this function can be used to remove the offer of a specific operator
        :param operator_id: corresponding operator id
        :type operator_id: int
        """
        try:
            self.offer.pop(operator_id)
        except KeyError:
            LOG.warning("Attempting to retract non-existent offer!")

    def retract_all_offers(self):
        """ this function can be used to remove all earlier offers from a traveller
        """
        self.offer = {}

    def user_boards_vehicle(self, simulation_time, op_id, vid, pu_pos, t_access):
        self.pu_time = simulation_time
        self.service_opid = op_id
        self.service_vid = vid
        self.pu_pos = pu_pos
        self.t_access = t_access

    def user_leaves_vehicle(self, simulation_time, do_pos, t_egress):
        self.do_time = simulation_time
        self.do_pos = do_pos
        self.t_egress = t_egress

    def create_SubTripRequest(self, subtrip_id, mod_o_node=None, mod_d_node=None, mod_start_time=None, modal_state = None):
        """ this function creates subtriprequests (i.e. a customer sends multiple requests) based on a attributes of itself. different subtrip-customers
        can vary in start and target node, earlest start time and modal_state (monomodal, firstmile, lastmile, firstlastmile)
        :param subtrip_id: identifier of the subtrip (this is not the customer id!)
        :type subtrip_id: int
        :param mod_o_node: new origin node index of subtrip
        :type mod_o_node: int
        :param mod_d_node: new destination node index of subtrip
        :type mod_d_node: int
        :param mod_start_time: new earliest start time of the trip
        :type mod_start_time: int
        :param modal_state: indicator of modality (indicator if monomodal, first, last or firstlast mile trip)
        :type modal_state: int in G_RQ_STATE_MONOMODAL, G_RQ_STATE_FIRSTMILE, G_RQ_STATE_LASTMILE, G_RQ_STATE_FIRSTLASTMILE (globals)
        :return: new traveler with specified attributes
        :rtype: same as called from
        """
        sub_rq_obj = deepcopy(self)
        old_rid = sub_rq_obj.get_rid()
        sub_rq_obj.sub_rid_struct = f"{old_rid}_{subtrip_id}"
        if mod_o_node is not None:
            sub_rq_obj.o_node = mod_o_node
        if mod_d_node is not None:
            sub_rq_obj.d_node = mod_d_node
        if mod_start_time is not None:
            sub_rq_obj.earliest_start_time = mod_start_time
        if modal_state is not None:
            sub_rq_obj.modal_state = modal_state
        return sub_rq_obj

    def set_direct_route_travel_infos(self, routing_engine):
        """ this function set the current direct route travel time and distance for the later output
        should be called in time, when the request enters the system
        :param routing_engine: network object
        """
        _, tt, dis = routing_engine.return_travel_costs_1to1(self.o_pos, self.d_pos)
        self.direct_route_travel_distance = dis
        self.direct_route_travel_time = tt

    def _add_record(self, record_dict):
        """This method enables the output of Traveler Model specific output

        :param record_dict: standard record output
        :return: extended record output
        """
        return record_dict

    def get_service_vehicle(self):
        """ returns the vehicle the traveller is served in
        :return: tuple (op_id, vid) if traveller is on board of a mod vehicle, None else
        """
        if self.service_vid is not None:
            return (self.service_opid, self.service_vid)
        else:
            return None

    @abstractmethod
    def choose_offer(self, scenario_parameters, simulation_time):
        """This method returns the operator id of the chosen mode.
        0..n: MoD fleet provider
        None: not decided yet
        <0: decline all MoD
        :param scenario_parameters: scenario parameter dictionary
        :param simulation_time: current simulation time
        :return: operator_id of chosen offer; or -1 if all MoD offers are declined; None if decision not defined yet
        """
        declines = [offer_id for offer_id, operator_offer in self.offer.items() if operator_offer.service_declined()]
        if len(declines) == scenario_parameters[G_NR_OPERATORS]:
            return -1
        return None

    def leaves_system(self, sim_time):
        """This method can be used to model customers waiting for offers and request retries etc.

        :param sim_time: current simulation time
        :return: True/False
        """
        if sim_time >= self.latest_decision_time:
            self.leave_system_time = sim_time
            return True
        else:
            return False

    def cancels_booking(self, sim_time):
        """This method can be used to model customer cancellations after they already accepted an offer once. Remember
        to adapt self.leave_system_time if users are allowed to cancel a booking.

        :param sim_time: current simulation time
        :return: True/False
        """
        return False
# -------------------------------------------------------------------------------------------------------------------- #

INPUT_PARAMETERS_BasicRequest = {
    "doc" : "This request only performs a mode choice based on if it recieved an offer or not. if an offer is recieved, it accepts the offer. if multiple offers are recieved an error is thrown",
    "inherit" : "RequestBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [], 
    "optional_modules": []
}

class BasicRequest(RequestBase):
    """This request only performs a mode choice based on if it recieved an offer or not.
    if an offer is recieved, it accepts the offer
    if multiple offers are recieved an error is thrown"""
    type = "BasicRequest"

    def __init__(self, rq_row, routing_engine, simulation_time_step, scenario_parameters):
        super().__init__(rq_row, routing_engine, simulation_time_step, scenario_parameters)

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
            return opts[0]
        else:
            LOG.error(f"not implemented {offer_str(self.offer)}")

# -------------------------------------------------------------------------------------------------------------------- #

INPUT_PARAMETERS_IndividualConstraintRequest = {
    "doc" : """This request class makes decisions based on hard constraints; individual constraints can be read from demand file columns. If an operator offer
    satisfies these, it will be accepted. Moreover, it can be used to communicate earliest and latest pick-up time to the operators.""",
    "inherit" : "RequestBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [G_AR_MAX_WT, G_AR_MAX_DTF],
    "mandatory_modules": [], 
    "optional_modules": []
}

class IndividualConstraintRequest(RequestBase):
    """This request class makes decisions based on hard constraints; individual constraints can be read from demand file columns. If an operator offer
    satisfies these, it will be accepted. Moreover, it can be used to communicate earliest and latest pick-up time to the operators."""
    type = "IndividualConstraintRequest"

    def __init__(self, rq_row, routing_engine, simulation_time_step, scenario_parameters):
        super().__init__(rq_row, routing_engine, simulation_time_step, scenario_parameters)
        # columns for heterogeneous parameters from rq_file > scenario parameters for homogeneous parameters
        if rq_row.get(G_RQ_LPT):
            self.latest_start_time = rq_row.get(G_RQ_LPT)
        elif scenario_parameters.get(G_AR_MAX_WT):
            self.latest_start_time = self.earliest_start_time + scenario_parameters.get(G_AR_MAX_WT)
        self.set_direct_route_travel_infos(routing_engine)
        if rq_row.get(G_RQ_MRD):
            rel_dt_f = rq_row[G_RQ_MRD]
        elif not pd.isnull(scenario_parameters.get(G_AR_MAX_DTF)):
            rel_dt_f = scenario_parameters.get(G_AR_MAX_DTF)
        else:
            rel_dt_f = None
        if rel_dt_f is None:
            self.max_trip_time = None
        else:
            self.max_trip_time = (100 + rel_dt_f) * (self.direct_route_travel_time +
                                                     scenario_parameters.get(G_OP_CONST_BT, 0)) / 100

    def choose_offer(self, sc_parameters, simulation_time):
        test_all_decline = super().choose_offer(sc_parameters, simulation_time)
        if test_all_decline is not None and test_all_decline < 0: # all operators declined service
            return -1
        # simple sort by amod operator id
        sorted_amod_offer_ops = sorted([op_id for op_id in self.offer.keys() if op_id >= 0])
        if len(sorted_amod_offer_ops) == 0:
            return None
        else:
            # assume that there is only one operator -> decline if offer is not fitting
            for op in sorted_amod_offer_ops:
                offered_pu_t = self.rq_time + self.offer[op][G_OFFER_WAIT]
                if offered_pu_t < self.earliest_start_time:
                    LOG.debug(f" -> decline. too early pick-up {offered_pu_t} < {self.earliest_start_time}")
                    return -1
                if offered_pu_t > self.latest_start_time:
                    LOG.debug(f" -> decline. too late pick-up {offered_pu_t} > {self.latest_start_time}")
                    return -1
                if self.max_trip_time and self.offer[op][G_OFFER_DRIVE] > self.max_trip_time:
                    LOG.debug(F" -> decline. too much detour {self.offer[op][G_OFFER_DRIVE]} > {self.max_trip_time}")
                    return -1
                LOG.debug(f" -> accept")
                self.fare = self.offer[op].get(G_OFFER_FARE, 0)
                return op

INPUT_PARAMETERS_PriceSensitiveIndividualConstraintRequest = {
    "doc" : """This request class can be used to communicate earliest and latest pick-up time to the operators.
    Moreover, the requests have a maximum price they are willing to pay.""",
    "inherit" : "IndividualConstraintRequest",
    "input_parameters_mandatory": [G_RQ_MAX_FARE],
    "input_parameters_optional": [],
    "mandatory_modules": [], 
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
class PriceSensitiveIndividualConstraintRequest(IndividualConstraintRequest):
    """This request class can be used to communicate earliest and latest pick-up time to the operators.
    Moreover, the requests have a maximum price they are willing to pay."""
    type = "PriceSensitiveIndividualConstraintRequest"

    def __init__(self, rq_row, routing_engine, simulation_time_step, scenario_parameters):
        super().__init__(rq_row, routing_engine, simulation_time_step, scenario_parameters)
        # read max price column -> Throw error if it is not available!
        self.max_fare = rq_row[G_RQ_MAX_FARE]

    def choose_offer(self, sc_parameters, simulation_time):
        declines = [offer_id for offer_id, operator_offer in self.offer.items() if operator_offer.service_declined()]
        if len(declines) == sc_parameters[G_NR_OPERATORS]:
            return -1
        # simple sort by amod operator id
        sorted_amod_offer_ops = sorted([op_id for op_id in self.offer.keys() if op_id >= 0])
        if len(sorted_amod_offer_ops) == 0:
            return None
        else:
            # assume that there is only one operator -> decline if offer is not fitting
            for op in sorted_amod_offer_ops:
                offered_fare = self.offer[op].get(G_OFFER_FARE)
                if offered_fare is not None and offered_fare > self.max_fare:
                    LOG.debug(f" -> decline. too expensive offer {offered_fare} > {self.max_fare}")
                    return -1
                offered_pu_t = self.rq_time + self.offer[op][G_OFFER_WAIT]
                if offered_pu_t < self.earliest_start_time:
                    LOG.debug(f" -> decline. too early pick-up {offered_pu_t} < {self.earliest_start_time}")

                    return -1
                if offered_pu_t > self.latest_start_time:
                    LOG.debug(f" -> decline. too late pick-up {offered_pu_t} > {self.latest_start_time}")
                    return -1
                if self.max_trip_time and self.offer[op][G_OFFER_DRIVE] > self.max_trip_time:
                    LOG.debug(F" -> decline. too much detour {self.offer[op][G_OFFER_DRIVE]} > {self.max_trip_time}")
                    return -1
                LOG.debug(f" -> accept")
                self.fare = self.offer[op].get(G_OFFER_FARE, 0)
                return op
# -------------------------------------------------------------------------------------------------------------------- #

INPUT_PARAMETERS_WaitingTimeSensitiveLinearDeclineRequest = {
    "doc" :     """This request is sensitive to waiting_times:
    - all offers are accepted if waiting time is below G_AR_MAX_WT
    - all offers are decline if waiting time is higher than G_AR_MAX_WT_2
    - linear decrease of probability of acceptance between G_AR_MAX_WT and G_AR_MAX_WT_2
    """,
    "inherit" : "RequestBase",
    "input_parameters_mandatory": [G_AR_MAX_WT, G_AR_MAX_WT_2],
    "input_parameters_optional": [],
    "mandatory_modules": [], 
    "optional_modules": []
}

class WaitingTimeSensitiveLinearDeclineRequest(RequestBase):
    """This request is sensitive to waiting_times:
    - all offers are accepted if waiting time is below G_AR_MAX_WT
    - all offers are decline if waiting time is higher than G_AR_MAX_WT_2
    - linear decrease of probability of acceptance between G_AR_MAX_WT and G_AR_MAX_WT_2
    """
    type = "WaitingTimeSensitiveLinearDeclineRequest"

    def __init__(self, rq_row, routing_engine, simulation_time_step, scenario_parameters):
        super().__init__(rq_row, routing_engine, simulation_time_step, scenario_parameters)
        self.max_wt_1 = scenario_parameters[G_AR_MAX_WT]
        self.max_wt_2 = scenario_parameters[G_AR_MAX_WT_2]

    def choose_offer(self, sc_parameters, simulation_time):
        LOG.debug("choose offer {}".format(offer_str(self.offer)))
        test_all_decline = super().choose_offer(sc_parameters, simulation_time)
        if test_all_decline is not None and test_all_decline < 0:
            return -1
        if len(self.offer) == 0:
            return None
        elif len(self.offer) == 1:
            op = list(self.offer.keys())[0]
            if self.offer[op].service_declined():
                LOG.debug(" -> no offer!")
                return -1
            wt = self.offer[op][G_OFFER_WAIT]
            if wt <= self.max_wt_1:
                LOG.debug(f" -> accept {wt} <= {self.max_wt_1}")
                self.fare = self.offer[op].get(G_OFFER_FARE, 0)
                return op
            elif wt > self.max_wt_2:
                LOG.debug(f" -> decline. too long?? {wt} > {self.max_wt_2}")
                return -1
            else:
                acc_prob = (self.max_wt_2 - wt) / (
                            self.max_wt_2 - self.max_wt_1)
                r = np.random.random()
                LOG.debug(f" -> random prob {acc_prob}")
                if r < acc_prob:
                    LOG.debug(f" -> accept")
                    self.fare = self.offer[op].get(G_OFFER_FARE, 0)
                    return op
                else:
                    LOG.debug(f" -> decline")
                    return -1
        else:
            LOG.error(f"not implemented {offer_str(self.offer)}")
            raise NotImplementedError
        
# -------------------------------------------------------------------------------------------------------------------- #
# Broker Requests

INPUT_PARAMETERS_PreferredOperatorRequest = {
    "doc" :     """this request is used for the broker scenarios as base case of (quasi) independent operators 
    rid chooses:
    - self.preferred op, if an offer is recieved from this op
    - declines else
    this is used to meassure if the unpreferred op was able to create an offer
    requires simulation class PreferredOperatorSimulation !
    """,
    "inherit" : "RequestBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [], 
    "optional_modules": []
}

class PreferredOperatorRequest(RequestBase):
    """ this request is used for the broker scenarios as base case of (quasi) independent operators 
    rid chooses:
    - self.preferred op, if an offer is recieved from this op
    - declines else
    this is used to meassure if the unpreferred op was able to create an offer
    requires simulation class PreferredOperatorSimulation"""
    type = "PreferredOperatorRequest"
    
    def __init__(self, rq_row, routing_engine, simulation_time_step, scenario_parameters):
        super().__init__(rq_row, routing_engine, simulation_time_step, scenario_parameters)
        self.preferred_operator = None  # will be set in the simulation class
        
    def choose_offer(self, scenario_parameters, simulation_time):
        list_options = [i for i, off in self.offer.items() if not off.service_declined()]
        if self.preferred_operator in list_options:
            self.fare = self.offer[self.preferred_operator].get(G_OFFER_FARE, 0)
            return self.preferred_operator
        else:
            return None

INPUT_PARAMETERS_BrokerDecisionRequest = {
    "doc" :     """    
    This request class is used for the broker decision simulation where a broker instead of the customer decides on which offer to take.
    The broker marks offers, that it has been chosen by the flag G_OFFER_BROKER_FLAG which is unique.
    This request class will only accept these marked offers.
    Requires simulation class BrokerDecisionSimulation !
    """,
    "inherit" : "RequestBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [], 
    "optional_modules": []
}

class BrokerDecisionRequest(RequestBase):
    """
    This request class is used for the broker decision simulation where a broker instead of the customer decides on which offer to take.
    The broker marks offers, that it has been chosen by the flag G_OFFER_BROKER_FLAG which is unique.
    This request class will only accept these marked offers.
    Requires simulation class BrokerDecisionSimulation !
    """
    type = "BrokerDecisionRequest"

    def choose_offer(self, scenario_parameters, simulation_time):
        selected_offer = None
        selected_op = None
        for op_id, offer in self.offer.items():
            if offer.get(G_OFFER_BROKER_FLAG):
                selected_offer = offer
                selected_op = op_id
                break
        if selected_offer is not None:
            self.fare = selected_offer.get(G_OFFER_FARE, 0)
        return selected_op

INPUT_PARAMETERS_UserDecisionRequest = {
    "doc" :     """This request class chooses the offer with the lowest overall travel time
    """,
    "inherit" : "RequestBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [], 
    "optional_modules": []
}

class UserDecisionRequest(RequestBase):
    """
    This request class is used for the easyride user decision simulation.
    The user chooses the offer with the lowest overall travel time
    """
    type = "UserDecisionRequest"

    def choose_offer(self, scenario_parameters, simulation_time):
        selected_offer = None
        selected_op = None
        best_overall_tt = float("inf")
        for op_id, offer in self.offer.items():
            if not offer.service_declined():
                tt = offer[G_OFFER_WAIT] + offer[G_OFFER_DRIVE]
                if tt < best_overall_tt:
                    best_overall_tt = tt
                    selected_offer = offer
                    selected_op = op_id
                elif tt == best_overall_tt:
                    r = np.random.randint(2)
                    if r == 0:
                        best_overall_tt = tt
                        selected_offer = offer
                        selected_op = op_id
        if selected_offer is not None:
            self.fare = selected_offer.get(G_OFFER_FARE, 0)
        return selected_op

#----------------------------------------------------------------------------#

INPUT_PARAMETERS_MasterRandomChoiceRequest = {
    "doc" :     """This request class randomly chooses between options.
    """,
    "inherit" : "RequestBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [], 
    "optional_modules": []
}

class MasterRandomChoiceRequest(RequestBase):
    """This request class randomly chooses between options."""
    type = "MasterRandomChoiceRequest"

    def choose_offer(self, scenario_parameters, simulation_time):
        test_all_decline = super().choose_offer(scenario_parameters, simulation_time)
        if test_all_decline is not None and test_all_decline < 0:
            return -1
        list_options = [i for i, off in self.offer.items() if not off.service_declined()]
        if -1 not in list_options:
            list_options.append(-1)
        choice = np.random.choice(list_options)
        self.fare = self.offer[choice].get(G_OFFER_FARE, 0)
        LOG.debug(f"{self.get_rid_struct()} chooses offer {choice} from options {list_options} | offers {offer_str(self.offer)}")
        return choice
# -------------------------------------------------------------------------------------------------------------------- #

INPUT_PARAMETERS_SlaveRequest = {
    "doc" :     """This request class does not have any choice functionality.
    (i.e. is used when mode choice is performed outside of FleetPy)
    """,
    "inherit" : "RequestBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [], 
    "optional_modules": []
}

class SlaveRequest(RequestBase):
    """This request class does not have any choice functionality."""
    type = "SlaveRequest"

    def choose_offer(self, scenario_parameters, simulation_time):
        # method is not used
        raise AssertionError(f"Request class {self.type} cannot be used for choice decisions!")

    def user_boards_vehicle(self, simulation_time, op_id, vid, pu_pos, t_access):
        #LOG.info(f"user boards vehicle: {self.rid} | {self.sub_rid_struct} | {self.offer}")
        self.fare = self.offer[op_id].get(G_OFFER_FARE, 0)
        return super().user_boards_vehicle(simulation_time, op_id, vid, pu_pos, t_access)

# -------------------------------------------------------------------------------------------------------------------- #
# Parcel Requests #
# -------------------------------------------------------------------------------------------------------------------- #

INPUT_PARAMETERS_ParcelRequestBase = {
    "doc" : """This request class is the base class for parcel 'travelers'. Here specific attributes for parcels are defined (i.e. ID) or type
    """,
    "inherit" : "RequestBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [], 
    "optional_modules": []
}

class ParcelRequestBase(RequestBase):
    type = "ParcelRequestBase"
    """ here specific attributes for parcels are defined (i.e. ID) or type """
    def __init__(self, rq_row, routing_engine, simulation_time_step, scenario_parameters):
        # TODO RPP: Definiere globale Variablen für parcels
        self.parcel_size = None
        super().__init__(rq_row, routing_engine, simulation_time_step, scenario_parameters)
        self.is_parcel = True
        self.rid = f"p_{self.rid}"
        self.parcel_size = rq_row.get(G_RQ_PA_SIZE, 1)
        self.earliest_start_time = rq_row.get(G_RQ_PA_EPT, None)
        self.latest_start_time = rq_row.get(G_RQ_PA_LPT, None)
        self.earliest_drop_off_time = rq_row.get(G_RQ_PA_EDT, None)
        self.latest_drop_off_time = rq_row.get(G_RQ_PA_LDT, None)

INPUT_PARAMETERS_BasicParcelRequest = {
    "doc" : """ This parcel request can be used only for a single operator. It always accepts an offer coming from this operator.
    """,
    "inherit" : "ParcelRequestBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [], 
    "optional_modules": []
}

class BasicParcelRequest(ParcelRequestBase): # TODO
    type = "BasicParcelRequest"
    "This parcel request can be used only for a single operator. It always accepts an offer coming from this operator."
    def __init__(self, rq_row, routing_engine, simulation_time_step, scenario_parameters):
        # TODO RPP : für CL: zugehörige person request id
        # initialisierung für verschiedene globals
        super().__init__(rq_row, routing_engine, simulation_time_step, scenario_parameters)

    def choose_offer(self, scenario_parameters, simulation_time):
        """This method returns the operator id of the chosen mode.
        0..n: MoD fleet provider
        None: not decided yet
        <0: decline all MoD
        :param scenario_parameters: scenario parameter dictionary
        :param simulation_time: current simulation time
        :return: operator_id of chosen offer; or -1 if all MoD offers are declined; None if decision not defined yet
        """
        declines = [offer_id for offer_id, operator_offer in self.offer.items() if operator_offer.service_declined()]
        if len(declines) == scenario_parameters[G_NR_OPERATORS]:
            return -1
        elif len(self.offer) > 1:
            raise NotImplementedError("More than one offer?")
        else:
            return list(self.offer.keys())[0]
        return None
    
INPUT_PARAMETERS_SlaveParcelRequest = {
    "doc" : """This parcel request class does not have any choice functionality. For coupled frameworks only!
    """,
    "inherit" : "ParcelRequestBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [], 
    "optional_modules": []
}

class SlaveParcelRequest(ParcelRequestBase):
    """This parcel request class does not have any choice functionality. For coupled frameworks only!"""
    type = "SlaveParcelRequest"

    def choose_offer(self, scenario_parameters, simulation_time):
        # method is not used
        raise AssertionError(f"Request class {self.type} cannot be used for choice decisions!")

    def user_boards_vehicle(self, simulation_time, op_id, vid, pu_pos, t_access):
        #LOG.info(f"user boards vehicle: {self.rid} | {self.sub_rid_struct} | {self.offer}")
        self.fare = self.offer[op_id].get(G_OFFER_FARE, 0)
        return super().user_boards_vehicle(simulation_time, op_id, vid, pu_pos, t_access)

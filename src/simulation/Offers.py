# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import typing as tp

# src imports
# -----------

from src.routing.NetworkBase import return_position_str
from src.misc.globals import *

# -------------------------------------------------------------------------------------------------------------------- #
# Traveler Offer Class
# -----------


class TravellerOffer:
    def __init__(self, traveler_id, operator_id, offered_waiting_time, offered_driving_time, fare,
                 additional_parameters=None):
        """ this class collects all information of a trip offered by an operator for a specific customer request
        TravellerOffer entities will be created by mobility operators and send to travellers, who perform mode choices
        based on the corresponding entries
        if at least the offered_waiting_time is set to None the offer is treated as a decline by the operator
        :param traveler_id: traveler_id this offer is sent to
        :type traveler_id: int
        :param operator_id: id of operator who made the offer
        :type operator_id: int
        :param offered_waiting_time: absolute time [s] from request-time until expected pick-up time
        :type offered_waiting_time: float or None
        :param offered_driving_time: time [s] a request is expected to drive from origin to destination
        :type offered_driving_time: float or None
        :param fare: fare of the trip [ct]
        :type fare: int or None
        :param additional_parameters: dictionary of other offer-attributes that might influence the simulation flow
        :type additional_parameters: dict or None
        """
        if additional_parameters is None:
            additional_parameters = {}
        self.traveler_id = traveler_id
        self.operator_id = operator_id
        self.offered_waiting_time = offered_waiting_time
        self.offered_driving_time = offered_driving_time
        self.fare = fare
        self.additional_offer_parameters = additional_parameters.copy()

    def extend_offer(self, additional_offer_parameters):
        """ this function can be used to add parameters to the offer
        :param additional_offer_parameters: dictionary offer_variable (globals!) -> value
        :type additional_offer_parameters: dict
        """
        self.additional_offer_parameters.update(additional_offer_parameters)

    def service_declined(self):
        """ this function evaluates if the offer should be treated as a decline because the service is not possible
        :return: True if operator decline the service, False else
        :rtype: bool
        """
        if self.offered_waiting_time is None:
            return True
        else:
            return False

    def __getitem__(self, offer_attribute_str):
        """ this function can be used to access specific attributes of the offer
        :param offer_attribute_str: attribute_str of the offer parameter (see globals!)
        :type offer_attribute_str: str
        :return: value of the specific attribute within the offer. raises error if not specified!
        :rtype: not defined
        """   
        if offer_attribute_str == G_OFFER_WAIT:
            return self.offered_waiting_time
        elif offer_attribute_str == G_OFFER_DRIVE:
            return self.offered_driving_time
        elif offer_attribute_str == G_OFFER_FARE:
            return self.fare
        else:
            try:
                return self.additional_offer_parameters[offer_attribute_str]
            except KeyError:
                pass
        raise KeyError(type(self).__name__+" object has no attribute '"+offer_attribute_str+"'")

    def get(self, offer_attribute_str, other_wise=None):
        """ this function can be used to access specific attributes of the offer
        :param offer_attribute_str: attribute_str of the offer parameter (see globals!)
        :type offer_attribute_str: str
        :param other_wise: value of the corresponding offer_attribute_str in case it is not specified in the offer
        :type other_wise: not defined
        :return: value of the specific attribute within the offer
        :rtype: not defined
        """
        if offer_attribute_str == G_OFFER_WAIT:
            return self.offered_waiting_time
        elif offer_attribute_str == G_OFFER_DRIVE:
            return self.offered_driving_time
        elif offer_attribute_str == G_OFFER_FARE:
            return self.fare
        else:
            return self.additional_offer_parameters.get(offer_attribute_str, other_wise)

    def __contains__(self, offer_attribute_str):
        """ this function overwrites the "in" operator and can be used to test
        if the offer attribute is within the allready defined offer attributes
        :param offer_attribute_str: specific offer attribute key (globals!)
        :type offer_attribute_str: str
        :return: true, if offer attribute defined in offer; else false
        :rtype: bool
        """
        if offer_attribute_str == G_OFFER_WAIT or offer_attribute_str == G_OFFER_DRIVE or offer_attribute_str == G_OFFER_FARE:
            return True
        elif self.additional_offer_parameters.get(offer_attribute_str, None) is not None:
            return True
        else:
            return False

    def to_output_str(self):
        """ this function creates a string of the offer parameters for the output file
        in the form offer_param1:offer_value1;offer_param2_offer_value2;...
        if no service was offered an empty str is returned
        :return: string of the offer to write to the outputfile
        :rtype: str
        """
        if not self.service_declined():
            offer_info = [f"{G_OFFER_WAIT}:{self.offered_waiting_time}", f"{G_OFFER_DRIVE}:{self.offered_driving_time}", f"{G_OFFER_FARE}:{self.fare}"]
            for k, v in self.additional_offer_parameters.items():
                if k == G_OFFER_PU_POS or k == G_OFFER_DO_POS:
                    v = return_position_str(v)
                offer_info.append(f"{k}:{v}")
            return ";".join(offer_info)
        else:
            return ""

    def __str__(self):
        if self.service_declined():
            return "declined"
        else:
            return self.to_output_str()


class Rejection(TravellerOffer):
    """This class takes minimal input and creates an offer that represents a rejection."""
    def __init__(self, traveler_id, operator_id):
        super().__init__(traveler_id, operator_id, offered_waiting_time=None, offered_driving_time=None, fare=None)


class PTOffer(TravellerOffer):
    """This class represents a public transport offer.
    
    A PT offer contains the following information:
    - traveler_id (str): sub-request id struct of the parent request
    - operator_id (int): id of PT operator (-2)
    - source_station_id (str): id of the source station
    - target_station_id (str): id of the target station
    - source_station_arrival_time (int): absolute time [s] of the arrival at the source station
    - source_transfer_time (int): absolute time [s] of the transfer from the source station to the source station stop
    - offered_waiting_time (int): absolute time [s] from arrival at the source station until departure
    - offered_trip_time (int): absolute time [s] from departure at the source station until arrival at the target station
    - fare (int): fare of the offer
    - source_walking_time (int): absolute time [s] from origin street node to source station
    - target_walking_time (int): absolute time [s] from target station to destination street node
    - destination_node_arrival_time (int): absolute time [s] of the arrival at the destination street node
    - num_transfers (int): number of transfers in the PT journey
    - pt_journey_duration (int): absolute time [s] from arrival at the origin street node to arrival at the destination street node
    - detailed_journey_plan (dict): detailed journey plan (only if requested)
    """
    def __init__(
    self, 
    traveler_id: str,
    operator_id: int,
    source_station_id: str,
    target_station_id: str,
    source_station_arrival_time: int,
    source_transfer_time: int,
    offered_waiting_time: int,
    offered_trip_time: int,
    fare: int,
    source_walking_time: int,
    target_walking_time: int,
    num_transfers: int,
    detailed_journey_plan: tp.List[tp.Dict[str, tp.Any]],
):
        self.source_station_arrival_time = source_station_arrival_time
        self.detailed_journey_plan = detailed_journey_plan

        self.target_station_arrival_time = source_station_arrival_time + offered_waiting_time + offered_trip_time
        self.destination_node_arrival_time = self.target_station_arrival_time + target_walking_time

        additional_parameters = {
            G_PT_OFFER_SOURCE_STATION: source_station_id,
            G_PT_OFFER_TARGET_STATION: target_station_id,
            G_PT_OFFER_SOURCE_TRANSFER_TIME: source_transfer_time,
            G_PT_OFFER_SOURCE_WALK: source_walking_time,
            G_PT_OFFER_TARGET_WALK: target_walking_time,
            G_PT_OFFER_NUM_TRANSFERS: num_transfers,
        }

        super().__init__(traveler_id, operator_id, offered_waiting_time, offered_trip_time, fare, additional_parameters=additional_parameters)


class MultimodalOffer(TravellerOffer):
    """This class represents a multimodal offer that consists of multiple segments served by different operators."""
    def __init__(
        self, 
        traveler_id: int, 
        sub_trip_offers: tp.Dict[int, TravellerOffer], 
        rq_modal_state: RQ_MODAL_STATE, 
    ):
        """Initialize a multimodal offer that can include multiple sub-trips from different operators.
        
        :param traveler_id: traveler_id this offer is sent to
        :type traveler_id: int
        :param sub_trip_offers: dictionary of sub-trip offers {sub_trip_id: TravellerOffer}
        :type sub_trip_offers: dict
        :param rq_modal_state: modal state of the request
        :type rq_modal_state: RQ_MODAL_STATE
        :param additional_parameters: dictionary of other offer-attributes
        :type additional_parameters: dict or None
        """
        self.rq_modal_state = rq_modal_state 
        self.sub_trip_offers: tp.Dict[int, TravellerOffer] = sub_trip_offers

        # merge sub-trip offers
        aggregated_offer: tp.Dict[str, tp.Any] = self._merge_sub_trip_offers()
        operator_ids: tp.FrozenSet[tp.Tuple[int, int]] = aggregated_offer[G_MULTI_OFFER_OPERATOR_SUB_TRIP_TUPLE]  # {(operator_id, sub_trip_id)}
        offered_waiting_time: int = aggregated_offer[G_OFFER_WAIT]
        offered_driving_time: int = aggregated_offer[G_OFFER_DRIVE]
        fare: int = aggregated_offer[G_OFFER_FARE]

        additional_parameters = {
            G_PT_OFFER_SOURCE_WALK: aggregated_offer[G_PT_OFFER_SOURCE_WALK],
            G_PT_OFFER_WAIT: aggregated_offer[G_PT_OFFER_WAIT],
            G_PT_OFFER_TRIP: aggregated_offer[G_PT_OFFER_TRIP],
            G_PT_OFFER_TARGET_WALK: aggregated_offer[G_PT_OFFER_TARGET_WALK],
            G_PT_OFFER_NUM_TRANSFERS: aggregated_offer[G_PT_OFFER_NUM_TRANSFERS],
            G_PT_OFFER_SOURCE_STATION: aggregated_offer[G_PT_OFFER_SOURCE_STATION],
            G_PT_OFFER_TARGET_STATION: aggregated_offer[G_PT_OFFER_TARGET_STATION],
        }

        if self.rq_modal_state == RQ_MODAL_STATE.FIRSTLASTMILE:
            additional_parameters[G_OFFER_WAIT_0] = aggregated_offer[G_OFFER_WAIT_0]
            additional_parameters[G_OFFER_DRIVE_0] = aggregated_offer[G_OFFER_DRIVE_0]
            additional_parameters[G_OFFER_WAIT_1] = aggregated_offer[G_OFFER_WAIT_1]
            additional_parameters[G_OFFER_DRIVE_1] = aggregated_offer[G_OFFER_DRIVE_1]

        super().__init__(traveler_id, operator_ids, offered_waiting_time, offered_driving_time, fare, additional_parameters=additional_parameters)

    def get_sub_trip_offers(self) -> tp.Dict[int, TravellerOffer]:
        """Get the sub-trip offers for the multimodal offer."""
        return self.sub_trip_offers
    
    def _merge_sub_trip_offers(self) -> tp.Dict[str, tp.Any]:
        """Merge the sub-trip offers into a single offer."""
        # Initialize a dictionary to store aggregated values
        aggregated_offer = {G_MULTI_OFFER_OPERATOR_SUB_TRIP_TUPLE: []}
        
        fare = 0

        # Iterate through sub-trip offers and aggregate values
        for sub_trip_id, sub_trip_offer in self.sub_trip_offers.items():
            aggregated_offer[G_MULTI_OFFER_OPERATOR_SUB_TRIP_TUPLE].append((sub_trip_offer.operator_id, sub_trip_id))
            fare += sub_trip_offer.get(G_OFFER_FARE, 0)
        
        aggregated_offer[G_MULTI_OFFER_OPERATOR_SUB_TRIP_TUPLE] = tuple(aggregated_offer[G_MULTI_OFFER_OPERATOR_SUB_TRIP_TUPLE])
        aggregated_offer[G_OFFER_FARE] = fare
        
        if self.rq_modal_state == RQ_MODAL_STATE.FIRSTMILE:
            aggregated_offer[G_OFFER_WAIT] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FM_AMOD.value].get(G_OFFER_WAIT)
            aggregated_offer[G_OFFER_DRIVE] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FM_AMOD.value].get(G_OFFER_DRIVE)
            aggregated_offer[G_PT_OFFER_WAIT] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FM_PT.value].get(G_OFFER_WAIT)
            aggregated_offer[G_PT_OFFER_TRIP] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FM_PT.value].get(G_OFFER_DRIVE)
            aggregated_offer[G_PT_OFFER_SOURCE_WALK] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FM_PT.value].get(G_PT_OFFER_SOURCE_WALK)
            aggregated_offer[G_PT_OFFER_TARGET_WALK] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FM_PT.value].get(G_PT_OFFER_TARGET_WALK)
            aggregated_offer[G_PT_OFFER_SOURCE_STATION] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FM_PT.value].get(G_PT_OFFER_SOURCE_STATION)
            aggregated_offer[G_PT_OFFER_TARGET_STATION] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FM_PT.value].get(G_PT_OFFER_TARGET_STATION)
            aggregated_offer[G_PT_OFFER_NUM_TRANSFERS] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FM_PT.value].get(G_PT_OFFER_NUM_TRANSFERS)
        elif self.rq_modal_state == RQ_MODAL_STATE.LASTMILE:
            aggregated_offer[G_OFFER_WAIT] = self.sub_trip_offers[RQ_SUB_TRIP_ID.LM_AMOD.value].get(G_OFFER_WAIT)
            aggregated_offer[G_OFFER_DRIVE] = self.sub_trip_offers[RQ_SUB_TRIP_ID.LM_AMOD.value].get(G_OFFER_DRIVE)
            aggregated_offer[G_PT_OFFER_WAIT] = self.sub_trip_offers[RQ_SUB_TRIP_ID.LM_PT.value].get(G_OFFER_WAIT)
            aggregated_offer[G_PT_OFFER_TRIP] = self.sub_trip_offers[RQ_SUB_TRIP_ID.LM_PT.value].get(G_OFFER_DRIVE)
            aggregated_offer[G_PT_OFFER_SOURCE_WALK] = self.sub_trip_offers[RQ_SUB_TRIP_ID.LM_PT.value].get(G_PT_OFFER_SOURCE_WALK)
            aggregated_offer[G_PT_OFFER_TARGET_WALK] = self.sub_trip_offers[RQ_SUB_TRIP_ID.LM_PT.value].get(G_PT_OFFER_TARGET_WALK)
            aggregated_offer[G_PT_OFFER_SOURCE_STATION] = self.sub_trip_offers[RQ_SUB_TRIP_ID.LM_PT.value].get(G_PT_OFFER_SOURCE_STATION)
            aggregated_offer[G_PT_OFFER_TARGET_STATION] = self.sub_trip_offers[RQ_SUB_TRIP_ID.LM_PT.value].get(G_PT_OFFER_TARGET_STATION)
            aggregated_offer[G_PT_OFFER_NUM_TRANSFERS] = self.sub_trip_offers[RQ_SUB_TRIP_ID.LM_PT.value].get(G_PT_OFFER_NUM_TRANSFERS)
        elif self.rq_modal_state == RQ_MODAL_STATE.FIRSTLASTMILE:
            aggregated_offer[G_OFFER_WAIT_0] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_0.value].get(G_OFFER_WAIT)
            aggregated_offer[G_OFFER_WAIT_1] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_1.value].get(G_OFFER_WAIT)
            aggregated_offer[G_OFFER_DRIVE_0] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_0.value].get(G_OFFER_DRIVE)
            aggregated_offer[G_OFFER_DRIVE_1] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FLM_AMOD_1.value].get(G_OFFER_DRIVE)
            aggregated_offer[G_OFFER_WAIT] = aggregated_offer[G_OFFER_WAIT_0] + aggregated_offer[G_OFFER_WAIT_1]
            aggregated_offer[G_OFFER_DRIVE] = aggregated_offer[G_OFFER_DRIVE_0] + aggregated_offer[G_OFFER_DRIVE_1]
            aggregated_offer[G_PT_OFFER_WAIT] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FLM_PT.value].get(G_OFFER_WAIT)
            aggregated_offer[G_PT_OFFER_TRIP] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FLM_PT.value].get(G_OFFER_DRIVE)
            aggregated_offer[G_PT_OFFER_SOURCE_WALK] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FLM_PT.value].get(G_PT_OFFER_SOURCE_WALK)
            aggregated_offer[G_PT_OFFER_TARGET_WALK] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FLM_PT.value].get(G_PT_OFFER_TARGET_WALK)
            aggregated_offer[G_PT_OFFER_SOURCE_STATION] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FLM_PT.value].get(G_PT_OFFER_SOURCE_STATION)
            aggregated_offer[G_PT_OFFER_TARGET_STATION] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FLM_PT.value].get(G_PT_OFFER_TARGET_STATION)
            aggregated_offer[G_PT_OFFER_NUM_TRANSFERS] = self.sub_trip_offers[RQ_SUB_TRIP_ID.FLM_PT.value].get(G_PT_OFFER_NUM_TRANSFERS)
        return aggregated_offer
    


# src imports
# -----------
import typing as tp

from src.routing.road.NetworkBase import return_position_str
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
    - origin_node_arrival_time (int): absolute time [s] of the arrival at the origin street node
    - source_walking_time (int): walking time [s] from origin street node to source station
    - source_station_departure_time (int): absolute time [s] of the departure at the source station
    - source_transfer_time (int): transfer time [s] from the source station to the source stop
    - waiting_time (int): waiting time [s] from arrival at the source stop until departure; this value is used as the 'offered_waiting_time' in the TravellerOffer
    - pt_trip_time (int): travel time [s] from departure at the source stop until arrival at the target stop
    - fare (int): fare of the offer
    - target_transfer_time (int): transfer time [s] from the target stop to the target station
    - target_station_arrival_time (int): absolute time [s] of the arrival at the target station
    - target_walking_time (int): walking time [s] from target station to destination street node
    - destination_node_arrival_time (int): absolute time [s] of the arrival at the destination street node
    - num_transfers (int): number of transfers in the PT journey
    - pt_journey_duration (int): duration [s] from departure at the source station to arrival at the target station
    - pt_segment_duration (int): duration [s] from arrival at the origin street node to arrival at the destination street node
    - detailed_journey_plan (dict): detailed journey plan (only if requested)
    """
    def __init__(
        self, 
        traveler_id: str, operator_id: int,
        source_station_id: str, target_station_id: str,
        source_walking_time: int, source_station_departure_time: int, source_transfer_time: int,
        waiting_time: int, pt_trip_time: int, fare: int,
        target_transfer_time: int, target_station_arrival_time: int, target_walking_time: int,
        num_transfers: int, pt_journey_duration: int, detailed_journey_plan: tp.List[tp.Dict[str, tp.Any]],
    ):
        self.origin_node_arrival_time = source_station_departure_time - source_walking_time
        self.source_station_departure_time = source_station_departure_time

        self.target_station_arrival_time = target_station_arrival_time
        self.destination_node_arrival_time = self.target_station_arrival_time + target_walking_time

        self.pt_journey_duration = pt_journey_duration
        pt_segment_duration = self.destination_node_arrival_time - self.origin_node_arrival_time

        self.detailed_journey_plan = detailed_journey_plan

        offered_driving_time = pt_segment_duration - waiting_time

        additional_parameters = {
            G_PT_OFFER_SOURCE_STATION: source_station_id,
            G_PT_OFFER_TARGET_STATION: target_station_id,
            G_PT_OFFER_SOURCE_WALKING_TIME: source_walking_time,
            G_PT_OFFER_SOURCE_TRANSFER_TIME: source_transfer_time,
            G_PT_OFFER_TRIP_TIME: pt_trip_time,
            G_PT_OFFER_TARGET_TRANSFER_TIME: target_transfer_time,
            G_PT_OFFER_TARGET_WALKING_TIME: target_walking_time,
            G_PT_OFFER_NUM_TRANSFERS: num_transfers,
        }

        super().__init__(traveler_id, operator_id, waiting_time, offered_driving_time, fare, additional_parameters=additional_parameters)
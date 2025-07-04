# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import typing as tp
from abc import abstractmethod, ABCMeta

# additional module imports (> requirements)
# ------------------------------------------


# src imports
# -----------
if tp.TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.demand.TravelerModels import RequestBase
    from src.simulation.Legs import VehicleRouteLeg

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000
BUFFER_SIZE = 100

INPUT_PARAMETERS_BrokerBase = {
    "doc" : "this class is the base class representing a broker platform which serves as a communication interface between the FleetSimulation class and the operators",
    "inherit" : None,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class BrokerBase(metaclass=ABCMeta):
    def __init__(self, n_amod_op: int, amod_operators: tp.List['FleetControlBase']):
        """
        The general attributes for the broker are initialized.

        Args:
            n_amod_op (int): number of AMoD operators
            amod_operators (tp.List['FleetControlBase']): list of AMoD operators
        """
        self.n_amod_op: int = n_amod_op
        self.amod_operators: tp.List['FleetControlBase'] = amod_operators
        

    @abstractmethod
    def inform_network_travel_time_update(self, sim_time: int):
        """This method informs the broker that the network travel times have been updated.
        This information is forwarded to the operators.
        """
        pass
    
    @abstractmethod
    def inform_request(self, rid: int, rq_obj: 'RequestBase', sim_time: int):
        """This method informs the broker that a new request has been made.
        This information is forwarded to the operators.
        """
        pass

    @abstractmethod
    def collect_offers(self, rid: int) -> tp.Dict[int, 'RequestBase']:
        """This method collects the offers from the operators.
        The return value is a list of tuples, where each tuple contains the operator id, the offer, and the simulation time.
        """
        pass

    @abstractmethod
    def inform_user_booking(self, rid: int, rq_obj: 'RequestBase', sim_time: int, chosen_operator: int) -> tp.List[tp.Tuple[int, 'RequestBase']]:
        """This method informs the broker that the user has booked a trip.
        """
        pass

    @abstractmethod
    def inform_user_leaving_system(self, rid: int, sim_time: int):
        """This method informs the broker that the user is leaving the system.
        """
        pass

    @abstractmethod
    def inform_waiting_request_cancellations(self, chosen_operator: int, rid: int, sim_time: int):
        """This method informs the operators that the waiting requests have been cancelled.
        """
        pass

    @abstractmethod
    def acknowledge_user_boarding(self, op_id: int, rid: int, vid: int, boarding_time: int):
        """This method acknowledges the user boarding.
        """
        pass

    @abstractmethod
    def acknowledge_user_alighting(self, op_id: int, rid: int, vid: int, alighting_time: int):
        """This method acknowledges the user alighting.
        """
        pass

    @abstractmethod
    def receive_status_update(self, op_id: int, vid: int, sim_time: int, passed_VRL: tp.List['VehicleRouteLeg'], force_update_plan: bool):
        """This method receives the status update of the vehicles.
        """
        pass

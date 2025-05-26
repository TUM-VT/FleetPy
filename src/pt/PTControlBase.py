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
from src.simulation.Offers import TravellerOffer

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_PTControlBase = {
    "doc" : "this class is the base class representing an PT operator",
    "inherit" : None,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class PTControlBase(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _initialize_pt_router(self):
        """This method initializes the PT router.
        """
        pass

    @abstractmethod
    def return_fastest_pt_journey_1to1(self):
        """This method will return the fastest pt journey plan between two pt stations.
        A station may consist of multiple stops.
        """
        pass

    @abstractmethod
    def find_closest_pt_station(self):
        """This method will find the closest pt station to a given street node.
        """
        pass

    @abstractmethod
    def find_closest_street_node(self):
        """This method will find the closest street node to a given pt station.
        """
        pass

    @abstractmethod
    def record_pt_offer_db(self):
        """This method will create a TravellerOffer for the pt request and record it in the pt offer database.
        """
        pass

    @abstractmethod
    def get_current_offer(self):
        """This method will return the current offer for the pt request.
        """
        pass

    @abstractmethod
    def user_confirms_booking(self):
        """This method is used to confirm a customer booking. This can trigger some database processes.
        """
        pass

    @abstractmethod
    def _compute_fare(self):
        """This method will compute the fare for the pt request.
        """
        pass
    
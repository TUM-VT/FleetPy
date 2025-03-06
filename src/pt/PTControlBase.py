# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import time
from abc import abstractmethod, ABCMeta

# additional module imports (> requirements)
# ------------------------------------------


# src imports
# -----------


# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000
BUFFER_SIZE = 100

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
        # Important parameters
        self.walking_speed = 1.4  # [m/s]
        self.manhattan_distance_scaling_factor = 1.2  # used to scale the manhattan distance to the actual distance (optional)
        self.max_number_of_transfers = 3
        self.max_transfer_time = 1000  # [s]

    @abstractmethod
    def initialize_pt_router(self):
        """This method initializes the PT router.
        """
        pass

    @abstractmethod
    def return_pt_travel_costs_1to1(self):
        """This method will return the pt travel costs of the fastest journey between two pt stops.
        """
        pass

    @abstractmethod
    def return_pt_travel_costs_Xto1(self):
        """This method will return the pt travel costs between a list of possible pt origin stops
        and a certain pt destination stop.
        """
        pass

    @abstractmethod
    def return_pt_travel_costs_1toX(self):
        """This method will return the pt travel costs between a certain pt origin stop 
        and a list of possible pt destination stops.
        """
        pass

    @abstractmethod
    def return_best_pt_journey_1to1(self):
        """This method will return the best pt journey between two pt stops.
        """
        pass

    @abstractmethod
    def return_best_pt_journey_Xto1(self):
        """This method will return the best pt journey between a list of possible pt origin stops and a certain pt destination stop.
        """
        pass

    @abstractmethod
    def return_best_pt_journey_1toX(self):
        """This method will return the best pt journey between a certain pt origin stop and a list of possible pt destination stops.
        """
        pass

    @abstractmethod
    def _preprocess_transfers(self):
        """This method preprocesses the transfers for all PT OD pairs.
        """
        pass
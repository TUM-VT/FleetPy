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
        pass

    @abstractmethod
    def _initialize_pt_router(self):
        """This method initializes the PT router.
        """
        pass

    @abstractmethod
    def return_pt_travel_costs_1to1(self):
        """This method will return the pt travel costs (time) of the fastest journey between two pt stations.
        A station may consist of multiple stops.
        """
        pass

    @abstractmethod
    def return_fastest_pt_journey_1to1(self):
        """This method will return the fastest pt journey plan between two pt stations.
        A station may consist of multiple stops.
        """
        pass
    
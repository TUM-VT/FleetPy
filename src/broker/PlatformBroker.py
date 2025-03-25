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
from src.broker.BrokerBase import BrokerBase

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000
BUFFER_SIZE = 100

INPUT_PARAMETERS_PlatformBroker = {
    "doc" : "this class is the base class representing a regulatory platform",
    "inherit" : BrokerBase,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class PlatformBroker(BrokerBase):
    @abstractmethod
    def __init__(self):
        pass

    def collect_offers(self, rid, rq_obj, sim_time):
        """This method collects the offers from the operators.
        """
        amod_offers = super().collect_offers(rid, rq_obj, sim_time)
        # TODO: Functions for calculating the subsidies
        return amod_offers
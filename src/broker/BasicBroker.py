# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import time
import typing as tp
from abc import abstractmethod, ABCMeta

# additional module imports (> requirements)
# ------------------------------------------


# src imports
# -----------
from src.broker.BrokerBase import BrokerBase
if tp.TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.pt.PTControlBase import PTControlBase

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000
BUFFER_SIZE = 100

INPUT_PARAMETERS_BasicBroker = {
    "doc" : "this broker class will be used when no other broker is specified",
    "inherit" : None,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class BasicBroker(BrokerBase):
    def __init__(self, n_amod_op: int, amod_operators: tp.List['FleetControlBase'], pt_operator: tp.Optional['PTControlBase'] = None):
        super().__init__(n_amod_op, amod_operators, pt_operator)
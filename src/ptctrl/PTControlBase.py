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
    from src.routing.pt.RaptorRouterCpp import RaptorRouterCpp
    from src.simulation.Offers import PTOffer

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
        self.pt_router: RaptorRouterCpp = None
        self.pt_operator_id: int = None
        self.pt_offer_db: tp.Dict[str, 'PTOffer'] = {}
        self.gtfs_dir: str = None

    @abstractmethod
    def _load_pt_router(self):
        """This method will load and initialize the pt router instance.
        """
        pass

    @abstractmethod
    def return_fastest_pt_journey_1to1(self):
        """This method will return the fastest pt journey between an origin and a destination.
        """
        pass

    @abstractmethod
    def create_and_record_pt_offer_db(self):
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

    @abstractmethod
    def _update_gtfs_data(self):
        """This method will update the gtfs data of the pt router to reflect any changes in the pt network or schedule.
        """
        pass
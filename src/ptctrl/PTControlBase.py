# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import typing as tp
from abc import abstractmethod, ABCMeta
from datetime import datetime

# additional module imports (> requirements)
# ------------------------------------------

# src imports
# -----------
if tp.TYPE_CHECKING:
    from src.demand.TravelerModels import BasicIntermodalRequest
    from src.routing.pt.RaptorRouterCpp import RaptorRouterCpp
    from src.simulation.Offers import PTOffer, Rejection

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
        self.pt_offer_db: tp.Dict[tp.Tuple[str, tp.Optional[int]], tp.Union['PTOffer', 'Rejection']] = {}
        self.gtfs_dir: str = None

    @abstractmethod
    def _load_pt_router(self) -> 'RaptorRouterCpp':
        """This method will load and initialize the pt router instance.
        """
        pass

    @abstractmethod
    def return_fastest_pt_journey_1to1(
        self,
        source_station_id: str,
        target_station_id: str,
        source_station_departure_datetime: datetime,
        max_transfers: int = 999,
        detailed: bool = False,
    ) -> tp.Optional[tp.Dict[str, tp.Any]]:
        """Query from source-station readiness, returning None when no journey exists."""
        pass

    @abstractmethod
    def create_and_record_pt_offer_db(
        self,
        rid_struct: str,
        source_station_id: tp.Optional[str],
        target_station_id: tp.Optional[str],
        source_walking_time: tp.Optional[float],
        target_walking_time: tp.Optional[float],
        pt_journey_plan_dict: tp.Optional[tp.Dict[str, tp.Any]],
        firstmile_amod_operator_id: tp.Optional[int] = None,
    ) -> None:
        """Store this operator's PTOffer, or a Rejection for a None journey plan."""
        pass

    @abstractmethod
    def get_current_offer(
        self,
        rid_struct: str,
        firstmile_amod_operator_id: tp.Optional[int] = None,
    ) -> tp.Optional[tp.Union['PTOffer', 'Rejection']]:
        """Return the candidate's PTOffer or Rejection, or None for an unknown key."""
        pass

    @abstractmethod
    def user_confirms_booking(
        self,
        pt_sub_rq_obj: 'BasicIntermodalRequest',
        firstmile_amod_operator_id: tp.Optional[int] = None,
    ) -> None:
        """Bind one feasible candidate to its execution request and record PT service."""
        pass

    @abstractmethod
    def user_cancels_request(
        self,
        rid_struct: str,
        sim_time: float,
        firstmile_amod_operator_id: tp.Optional[int] = None,
        *,
        missed_connection: bool = False,
    ) -> bool:
        """Retract a proposal or unboarded booking at absolute simulation time.

        Return True when the candidate is inactive, including an already absent
        key; return False when the accepted binding must remain protected.
        """
        pass

    @abstractmethod
    def _compute_fare(self) -> int:
        """This method will compute the fare for the pt request.
        """
        pass

    @abstractmethod
    def _update_gtfs_data(self) -> None:
        """This method will update the gtfs data of the pt router to reflect any changes in the pt network or schedule.
        """
        pass

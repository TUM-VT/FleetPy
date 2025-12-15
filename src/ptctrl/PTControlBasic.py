# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import os
import logging
from datetime import datetime
import ast
import typing as tp
import pandas as pd

# additional module imports (> requirements)
# ------------------------------------------

# src imports
# -----------
from src.ptctrl.PTControlBase import PTControlBase
from src.routing.pt.RaptorRouterCpp import RaptorRouterCpp
from src.simulation.Offers import Rejection, PTOffer
if tp.TYPE_CHECKING:
    from src.demand.TravelerModels import BasicIntermodalRequest

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_PTControlBasic = {
    "doc" : "this class is the basic PT control class using C++ Raptor implementation",
    "inherit" : PTControlBase,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [RaptorRouterCpp, PTControlBase, Rejection, PTOffer],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class PTControlBasic(PTControlBase):
    def __init__(self, gtfs_dir: str, pt_operator_id: int = -2):
        super().__init__()

        self.pt_router: RaptorRouterCpp = self._load_pt_router(gtfs_dir)
        self.pt_operator_id: int = pt_operator_id

        self.pt_offer_db: tp.Dict[str, 'PTOffer'] = {}  # rid_struct -> PTOffer

        LOG.info("PT operator initialized successfully.")

    def _load_pt_router(self, gtfs_dir: str) -> RaptorRouterCpp:
        """This method will load and initialize the pt router instance.

        Args:
            gtfs_dir (str): the directory path where the GTFS files are stored.
        """
        return RaptorRouterCpp(gtfs_dir)
    
    def return_fastest_pt_journey_1to1(
        self,
        source_station_id: str, target_station_id: str,
        source_station_departure_time: int,
        max_transfers: int = 999,
        detailed: bool = False,
    ) -> tp.Union[tp.Dict[str, tp.Any], None]:
        """This method will return the fastest pt journey between an origin and a destination.

        Args:
            source_station_id (str): id of the source station.
            target_station_id (str): id of the target station.
            source_station_departure_time (int): departure timestamp [s] from source station.
            max_transfers (int, optional): maximum number of transfers allowed. Defaults to 999 (no limit).
            detailed (bool, optional): whether to return a detailed journey plan. Defaults to False.
        Returns:
            tp.Union[tp.Dict[str, tp.Any], None]: The pt journey plan dictionary or None if no journey is found.
        """
        pt_journey_plan_dict: tp.Union[tp.Dict[str, tp.Any], None] = self.pt_router.find_fastest_pt_journey_1to1(
            source_station_id = source_station_id,
            target_station_id = target_station_id,
            source_station_departure_time = source_station_departure_time,
            max_transfers = max_transfers,
            detailed = detailed,
        )
        return pt_journey_plan_dict
    
    def create_and_record_pt_offer_db(
        self,
        rid_struct: str, operator_id: int,
        source_station_id: str, target_station_id: str,
        source_walking_time: int,target_walking_time: int,
        source_station_departure_time: int,
        pt_journey_plan_dict: tp.Union[tp.Dict[str, tp.Any], None],
        previous_amod_operator_id: int = None,
    ):
        """This method will create a PTOffer for the pt request and record it in the pt offer database.

        Args:
            rid_struct (str): sub-request id struct of the journey.
            operator_id (int): id of PT operator (-2).
            source_station_id (str): id of the source station.
            target_station_id (str): id of the target station.
            source_walking_time (int): walking time [s] from origin street node to source station.
            target_walking_time (int): walking time [s] from target station to destination street node.
            source_station_departure_time (int): departure timestamp [s] from source station.
            pt_journey_plan_dict (tp.Union[tp.Dict[str, tp.Any], None]): The pt journey plan dictionary or None if no journey is found.
            previous_amod_operator_id (int, optional): The operator id of the previous amod operator. Defaults to None.
        """
        if pt_journey_plan_dict is None:
            self.pt_offer_db[(rid_struct, previous_amod_operator_id)] = Rejection(rid_struct, operator_id)
        else:
            fare: int = self._compute_fare()
            # old offer will always be overwritten
            self.pt_offer_db[(rid_struct, previous_amod_operator_id)] = PTOffer(
                traveler_id = rid_struct, operator_id = operator_id,
                source_station_id = source_station_id, target_station_id = target_station_id,
                source_walking_time = source_walking_time, source_station_departure_time = source_station_departure_time,
                source_transfer_time = pt_journey_plan_dict.get(G_PT_OFFER_SOURCE_TRANSFER_TIME, None),
                waiting_time = pt_journey_plan_dict.get(G_PT_OFFER_SOURCE_WAITING_TIME, None),
                trip_time = pt_journey_plan_dict.get(G_PT_OFFER_TRIP_TIME, None),
                fare = fare,
                target_transfer_time = pt_journey_plan_dict.get(G_PT_OFFER_TARGET_TRANSFER_TIME, None), 
                target_station_arrival_time = pt_journey_plan_dict.get(G_PT_OFFER_TARGET_STATION_ARRIVAL_TIME, None), 
                target_walking_time = target_walking_time,
                num_transfers = pt_journey_plan_dict.get(G_PT_OFFER_NUM_TRANSFERS, None), 
                pt_journey_duration = pt_journey_plan_dict.get(G_PT_OFFER_DURATION, None), 
                detailed_journey_plan = pt_journey_plan_dict.get(G_PT_OFFER_STEPS, None),
            )

    def _compute_fare(self) -> int:
        """This method will compute the fare for the pt request.

        For the basic implementation, this method returns 0.
        """
        return 0
    
    def get_current_offer(
        self, 
        rid_struct: str,
        previous_amod_operator_id: int = None,
    ) -> tp.Optional[PTOffer]:
        """This method will return the current offer for the pt request.

        Args:
            rid_struct (str): The sub-request id struct of the journey.
            previous_amod_operator_id (int, optional): The operator id of the previous amod operator. Defaults to None.
        Returns:
            tp.Optional[PTOffer]: The current offer for the pt request.
        """
        return self.pt_offer_db.get((rid_struct, previous_amod_operator_id), None)
    
    def user_confirms_booking(
        self,
        pt_sub_rq_obj: 'BasicIntermodalRequest',
        previous_amod_operator_id: int = None
    ):
        """This method is used to confirm a customer booking. This can trigger some database processes.

        Args:
            pt_sub_rq_obj (BasicMultimodalRequest): The pt sub-request object.
            previous_amod_operator_id (int, optional): The operator id of the previous amod operator. Defaults to None.
        """
        pt_rid_struct: str = pt_sub_rq_obj.get_rid_struct()
        pt_offer: 'PTOffer' = self.get_current_offer(pt_rid_struct, previous_amod_operator_id)
        pt_sub_rq_obj.user_boards_vehicle(
            simulation_time = pt_offer.get(G_PT_OFFER_SOURCE_STATION_DEPARTURE_TIME, None),
            op_id = self.pt_operator_id,
            vid = -1,
            pu_pos = None,
            t_access = pt_offer.get(G_PT_OFFER_SOURCE_WALKING_TIME, None),
        )
        pt_sub_rq_obj.user_leaves_vehicle(
            simulation_time = pt_offer.get(G_PT_OFFER_TARGET_STATION_ARRIVAL_TIME, None),
            do_pos = None,
            t_egress = pt_offer.get(G_PT_OFFER_TARGET_WALKING_TIME, None),
        )
        
    def _update_gtfs_data(self):
        """This method will update the gtfs data of the pt router to reflect any changes in the pt network or schedule.

        For the basic implementation, this method does nothing.
        """
        pass

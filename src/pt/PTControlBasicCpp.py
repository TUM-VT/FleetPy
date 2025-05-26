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
from src.pt.PTControlBase import PTControlBase
from src.pt.cpp_pt_router.PyPTRouter import PyPTRouter
from src.simulation.Offers import Rejection, PTOffer
if tp.TYPE_CHECKING:
    from src.demand.TravelerModels import BasicMultimodalRequest

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_PTControlBasicCpp = {
    "doc" : "this class is the basic PT control class using C++ Raptor implementation",
    "inherit" : PTControlBase,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class PTControlBasicCpp(PTControlBase):
    def __init__(self, fp_gtfs_dir: str, pt_operator_id: int = -2):
        super().__init__()

        self.pt_operator_id = pt_operator_id

        self.pt_offer_db: tp.Dict[str, 'PTOffer'] = {}  # rid_struct -> PTOffer

        # initialize the pt router
        self.pt_router = None
        self._initialize_pt_router(fp_gtfs_dir)

        # load the stations and street station transfers from the gtfs data
        self.stations_fp_df = self._load_stations_from_gtfs(fp_gtfs_dir)
        self.street_station_transfers_fp_df = self._load_street_station_transfers_from_gtfs(fp_gtfs_dir)

        LOG.info("PT operator initialized successfully.")

    def _initialize_pt_router(self, fp_gtfs_dir: str):
        """This method initializes the PT router.

        Args:
            fp_gtfs_dir (str): The directory containing the GTFS data of the operator.
        """
        # check if the directories exist and is all mandatory files present
        mandatory_files = [
            "agency_fp.txt", 
            "stops_fp.txt", 
            "trips_fp.txt", 
            "routes_fp.txt", 
            "calendar_fp.txt", 
            "stop_times_fp.txt", 
            "stations_fp.txt", 
            "transfers_fp.txt"
            ]
        if not os.path.exists(fp_gtfs_dir):
            raise FileNotFoundError(f"The directory {fp_gtfs_dir} does not exist.")
        for file in mandatory_files:
            if not os.path.exists(os.path.join(fp_gtfs_dir, file)):
                raise FileNotFoundError(f"The file {file} does not exist in the directory {fp_gtfs_dir}.")

        # initialize the pt router with the given gtfs data
        LOG.debug(f"Initializing the PT router with the given GTFS data in the directory: {fp_gtfs_dir}")
        self.pt_router = PyPTRouter(fp_gtfs_dir)
        LOG.debug("PT router initialized successfully.")

    def _load_stations_from_gtfs(self, fp_gtfs_dir: str) -> pd.DataFrame:
        """This method will load the stations from the GTFS data.

        Args:
            fp_gtfs_dir (str): The directory containing the GTFS data of the operator.

        Returns:
            pd.DataFrame: The PT stations data.
        """
        dtypes = {
            'station_id': 'str',
            'station_name': 'str',
            'station_lat': 'float',
            'station_lon': 'float',
            'stops_included': 'str',
            'station_stop_transfer_times': 'str',
            'num_stops_included': 'int',
        }
        return pd.read_csv(os.path.join(fp_gtfs_dir, "stations_fp.txt"), dtype=dtypes)
    
    def _load_street_station_transfers_from_gtfs(self, fp_gtfs_dir: str) -> pd.DataFrame:
        """This method will load the street station transfers from the GTFS data.

        Args:
            fp_gtfs_dir (str): The directory containing the GTFS data of the operator.

        Returns:
            pd.DataFrame: The transfer data between the street nodes and the pt stations.
        """
        dtypes = {
            'node_id': 'int',
            'closest_station_id': 'str',
            'street_transfer_time': 'int',
        }
        return pd.read_csv(os.path.join(fp_gtfs_dir, "street_station_transfers_fp.txt"), dtype=dtypes)

    def return_fastest_pt_journey_1to1(
        self,
        source_station_id: str,
        target_station_id: str,
        arrival_datetime: datetime,
        max_transfers: int=-1,
        detailed: bool=False,
    ) -> tp.Union[tp.Dict[str, tp.Any], None]:
        """This method will return the fastest PT journey plan between two PT stations.
        A station may consist of multiple stops.

        Args:
            source_station_id (str): The id of the source station.
            target_station_id (str): The id of the target station.
            arrival_datetime (datetime): The arrival datetime at the source station.
            max_transfers (int): The maximum number of transfers allowed in the journey, -1 for no limit.
            detailed (bool): Whether to return the detailed journey plan.
        Returns:
            tp.Union[tp.Dict[str, tp.Any], None]: The fastest PT journey plan or None if no journey is found.
        """
        # get all included stops for the source and target station
        included_sources = self._get_included_stops_and_transfer_times(source_station_id)
        included_targets = self._get_included_stops_and_transfer_times(target_station_id)
        
        return self.pt_router.return_fastest_pt_journey_1to1(arrival_datetime, included_sources, included_targets, max_transfers, detailed)
    
    def _get_included_stops_and_transfer_times(self, station_id: str) -> tp.Tuple[tp.List[str], tp.List[int]]:
        """This method will return the included stops and transfer times for a given station.

        Args:
            station_id (str): The id of the station.

        Returns:
            tp.Tuple[tp.List[str], tp.List[int]]: The included stops and transfer times.
        """
        station_data = self.stations_fp_df[self.stations_fp_df["station_id"] == station_id]
        
        if station_data.empty:
            raise ValueError(f"Station ID {station_id} not found in the stations data")
            
        included_ids_str = station_data["stops_included"].iloc[0]
        included_ids = ast.literal_eval(included_ids_str)
        transfer_times_str = station_data["station_stop_transfer_times"].iloc[0]
        transfer_times = ast.literal_eval(transfer_times_str)
        return [(stop_id, int(transfer_time)) for stop_id, transfer_time in zip(included_ids, transfer_times)]
    
    def find_closest_pt_station(self, street_node_id: int) -> tp.Tuple[str, int]:
        """This method finds the closest pt station from the street node id.

        Args:
            street_node_id (int): The street node id.

        Returns:
            tp.Tuple[str, int]: The closest pt station id and the walking time.
        """
        street_station_transfer = self.street_station_transfers_fp_df[self.street_station_transfers_fp_df["node_id"] == street_node_id]
        if street_station_transfer.empty:
            raise ValueError(f"Street node id {street_node_id} not found in the street station transfers file")
        closest_station_id: str = street_station_transfer["closest_station_id"].iloc[0]
        walking_time: int = street_station_transfer["street_station_transfer_time"].iloc[0]
        return closest_station_id, walking_time
    
    def find_closest_street_node(self, pt_station_id: str) -> tp.Tuple[int, int]:
        """This method finds the closest street node from the pt station id.

        Args:
            pt_station_id (str): The pt station id.

        Returns:
            tp.Tuple[int, int]: The closest street node id and the walking time.
        """
        street_station_transfers = self.street_station_transfers_fp_df[self.street_station_transfers_fp_df["closest_station_id"] == pt_station_id]
        if street_station_transfers.empty:
            raise ValueError(f"PT station id {pt_station_id} not found in the street station transfers file")
        # find the record with the minimum street_station_transfer_time
        min_transfer = street_station_transfers.loc[street_station_transfers["street_station_transfer_time"].idxmin()]
        closest_street_node_id: int = min_transfer["node_id"]
        walking_time: int = min_transfer["street_station_transfer_time"]
        return closest_street_node_id, walking_time
    
    def record_pt_offer_db(
        self,
        rid_struct: str,
        operator_id: int,
        source_station_id: str,
        target_station_id: str,
        source_walking_time: int,
        target_walking_time: int,
        pt_journey_plan_dict: tp.Union[tp.Dict[str, tp.Any], None],
        previous_amod_operator_id: int = None,
    ):
        """This method will create a PTOffer for the pt request and record it in the pt offer database.

        Args:
            rid_struct (str): The sub-request id struct of the journey.
            operator_id (int): The operator id.
            source_station_id (str): The id of the source station.
            target_station_id (str): The id of the target station.
            source_walking_time (int): The walking time from the street node to the source station.
            target_walking_time (int): The walking time from the target station to the street node.
            pt_journey_plan_dict (tp.Union[tp.Dict[str, tp.Any], None]): The pt journey plan dictionary or None if no journey is found.
            previous_amod_operator_id (int, optional): The operator id of the previous amod operator. Defaults to None.
        """
        if pt_journey_plan_dict is None:
            self.pt_offer_db[(rid_struct, previous_amod_operator_id)] = Rejection(rid_struct, operator_id)
        else:
            # TODO: compute pt fare
            fare: int = self._compute_fare()
            # old offer will always be overwritten
            self.pt_offer_db[(rid_struct, previous_amod_operator_id)] = PTOffer(
                                                                                traveler_id = rid_struct,
                                                                                operator_id = operator_id,
                                                                                source_station_id = source_station_id,
                                                                                target_station_id = target_station_id,
                                                                                source_station_arrival_time = pt_journey_plan_dict["departure_time"],
                                                                                source_transfer_time = pt_journey_plan_dict["source_transfer_time"],
                                                                                offered_waiting_time = pt_journey_plan_dict["waiting_time"],
                                                                                offered_trip_time = pt_journey_plan_dict["trip_time"],
                                                                                fare = fare,
                                                                                source_walking_time = source_walking_time,
                                                                                target_walking_time = target_walking_time,
                                                                                num_transfers = pt_journey_plan_dict["num_transfers"],
                                                                                detailed_journey_plan = pt_journey_plan_dict["steps"],
                                                                                )

    def _compute_fare(self) -> int:
        """This method will compute the fare for the pt request.
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
        pt_sub_rq_obj: 'BasicMultimodalRequest',
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
                                        simulation_time = pt_offer.source_station_arrival_time,  # source stop transfer time in not included
                                        op_id = self.pt_operator_id,
                                        vid = -1,
                                        pu_pos = None,
                                        t_access = pt_offer.get(G_PT_OFFER_SOURCE_WALK, None),
                                        )
        pt_sub_rq_obj.user_leaves_vehicle(
                                        simulation_time = pt_offer.target_station_arrival_time,
                                        do_pos = None,
                                        t_egress = pt_offer.get(G_PT_OFFER_TARGET_WALK, None),
                                        )
        

if __name__ == "__main__":
    # Test the pt control class： python -m src.pt.PTControlBasicCpp
    example_gtfs_dir = "data/pt/example_network/example_gtfs/matched"
    pt_control = PTControlBasicCpp(example_gtfs_dir)
    arrival_datetime = datetime(2024, 1, 1, 0, 4, 0)
    import time
    start_time = time.time()
    print(pt_control.return_fastest_pt_journey_1to1("s1", "s14", arrival_datetime, 1, detailed=False))
    print(f"Time taken: {time.time() - start_time} seconds")
    start_time = time.time()
    print(pt_control.return_fastest_pt_journey_1to1("s1", "s14", arrival_datetime, 1, detailed=True))
    print(f"Time taken: {time.time() - start_time} seconds")

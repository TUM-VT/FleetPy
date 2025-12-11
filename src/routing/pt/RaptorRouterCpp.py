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
from src.routing.pt.cpp_raptor_router.PyPTRouter import PyPTRouter

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_RaptorRouterCpp = {
    "doc" : "this class is the PT router class using C++ Raptor implementation",
    "inherit" : [],
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [PyPTRouter],
    "optional_modules": []
}

# -------------------------------------------------------------------------------------------------------------------- #
# main
# ----
class RaptorRouterCpp():
    def __init__(self, gtfs_dir: str):
        """
        Args:
            gtfs_dir (str): the directory path where the GTFS files are stored.
        """
        # initialize the pt router
        self.pt_router = None
        self._initialize_pt_router(gtfs_dir)

        # load the stations: used for station-stop mapping
        self.stations_fp_df = self._load_stations_from_gtfs(gtfs_dir)

        # load the street-station transfers: used for finding closest station to a street node, or vice versa
        self.street_station_transfers_fp_df = self._load_street_station_transfers_from_gtfs(gtfs_dir)

    def _initialize_pt_router(self, gtfs_dir: str):
        """This method initializes the PT router.

        Args:
            gtfs_dir (str): the directory path where the GTFS files are stored.
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
        if not os.path.exists(gtfs_dir):
            raise FileNotFoundError(f"The directory {gtfs_dir} does not exist.")
        for file in mandatory_files:
            if not os.path.exists(os.path.join(gtfs_dir, file)):
                raise FileNotFoundError(f"The file {file} does not exist in the directory {gtfs_dir}.")

        # initialize the pt router with the given gtfs data
        LOG.debug(f"Initializing the Raptor router (C++) with the given GTFS data in the directory: {gtfs_dir}")
        self.pt_router = PyPTRouter(gtfs_dir)
        LOG.debug("Raptor router (C++) initialized successfully.")

    def _load_stations_from_gtfs(self, gtfs_dir: str) -> pd.DataFrame:
        """This method loads the FleetPy-specific stations file.

        Args:
            gtfs_dir (str): The directory containing the GTFS data of the operator.
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
        return pd.read_csv(os.path.join(gtfs_dir, "stations_fp.txt"), dtype=dtypes)
    
    def _load_street_station_transfers_from_gtfs(self, gtfs_dir: str) -> pd.DataFrame:
        """This method loads the FleetPy-specific street station transfers file.

        Args:
            gtfs_dir (str): The directory containing the GTFS data of the operator.
        Returns:
            pd.DataFrame: The transfer data between the street nodes and the PT stations.
        """
        dtypes = {
            'node_id': 'int',
            'closest_station_id': 'str',
            'street_transfer_time': 'int',
        }
        return pd.read_csv(os.path.join(gtfs_dir, "street_station_transfers_fp.txt"), dtype=dtypes)
    
    def return_fastest_pt_journey_1to1(
        self,
        source_station_id: str,
        target_station_id: str,
        source_station_departure_datetime: datetime,
        max_transfers: int=999,
        detailed: bool=False,
    ) -> tp.Union[tp.Dict[str, tp.Any], None]:
        """This method returns the fastest PT journey plan between two PT stations.
        A station may consist of multiple stops.

        Args:
            source_station_id (str): The id of the source station.
            target_station_id (str): The id of the target station.
            source_station_departure_datetime (datetime): The departure datetime at the source station.
            max_transfers (int): The maximum number of transfers allowed in the journey, 999 for no limit.
            detailed (bool): Whether to return the detailed journey plan.
        Returns:
            tp.Union[tp.Dict[str, tp.Any], None]: The fastest PT journey plan or None if no journey is found.
        """
        # get all included stops for the source and target station
        included_sources = self._get_included_stops_and_transfer_times(source_station_id)
        included_targets = self._get_included_stops_and_transfer_times(target_station_id)
        
        return self.pt_router.return_fastest_pt_journey_1to1(source_station_departure_datetime, included_sources, included_targets, max_transfers, detailed)
    
    def _get_included_stops_and_transfer_times(self, station_id: str) -> tp.Tuple[tp.List[str], tp.List[int]]:
        """This method returns the included stops and transfer times for a given station.

        Args:
            station_id (str): The id of the station.
        Returns:
            tp.Tuple[tp.List[str], tp.List[int]]: The included stops and transfer times.
        """
        station_data = self.stations_fp_df[self.stations_fp_df["station_id"] == station_id]
        
        if station_data.empty:
            raise ValueError(f"Station ID {station_id} not found in the stations data")
            
        included_ids_str = station_data["stops_included"].iloc[0]
        included_ids_str = included_ids_str.replace(';', ',')
        included_ids_list = ast.literal_eval(included_ids_str)

        transfer_times_str = station_data["station_stop_transfer_times"].iloc[0]
        transfer_times_str = transfer_times_str.replace(';', ',')
        transfer_times_list = ast.literal_eval(transfer_times_str)
        return [(stop_id, int(transfer_time)) for stop_id, transfer_time in zip(included_ids_list, transfer_times_list)]



if __name__ == "__main__":
    # Test the pt router class： python -m src.routing.pt.CppTester

    gtfs_dir = r"data\pubtrans\example_network\example_gtfs\matched"
    router = RaptorRouterCpp(gtfs_dir)

    source_station_departure_datetime = datetime(2024, 1, 1, 0, 4, 0)
    import time
    start_time = time.time()
    print(router.return_fastest_pt_journey_1to1("s1", "s14", source_station_departure_datetime, 3, detailed=False))
    print(f"Time taken: {time.time() - start_time} seconds")
    start_time = time.time()
    print(router.return_fastest_pt_journey_1to1("s1", "s14", source_station_departure_datetime, 3, detailed=True))
    print(f"Time taken: {time.time() - start_time} seconds")
# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import os
import logging
from datetime import datetime
import ast
import typing as tp
from math import isfinite
from numbers import Integral, Real
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
        if not os.path.isdir(gtfs_dir):
            raise FileNotFoundError(f"The directory {gtfs_dir} does not exist.")
        for file in mandatory_files:
            if not os.path.isfile(os.path.join(gtfs_dir, file)):
                raise FileNotFoundError(f"The file {file} does not exist in the directory {gtfs_dir}.")

        # initialize the pt router with the given gtfs data
        LOG.debug(f"Initializing the Raptor router (C++) with the given GTFS data in the directory: {gtfs_dir}")
        try:
            self.pt_router = PyPTRouter(gtfs_dir)
        except Exception:
            LOG.exception("Failed to initialize the PT router from %s", gtfs_dir)
            raise
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
        path = os.path.join(gtfs_dir, "stations_fp.txt")
        stations = pd.read_csv(path, dtype=dtypes)
        required_columns = {
            "station_id", "stops_included", "station_stop_transfer_times", "num_stops_included",
        }
        missing_columns = required_columns.difference(stations.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in {path}: {sorted(missing_columns)}")
        if not stations["station_id"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Station IDs in {path} must be non-empty strings")
        duplicate_ids = stations.loc[stations["station_id"].duplicated(), "station_id"].tolist()
        if duplicate_ids:
            raise ValueError(f"Duplicate station IDs in {path}: {duplicate_ids}")
        return stations
    
    def find_fastest_pt_journey_1to1(
        self,
        source_station_id: str,
        target_station_id: str,
        source_station_departure_datetime: datetime,
        max_transfers: int = 999,
        detailed: bool = False,
    ) -> tp.Union[tp.Dict[str, tp.Any], None]:
        """This method returns the fastest PT journey plan between two PT stations.
        A station may consist of multiple stops.

        Args:
            source_station_id (str): The id of the source station.
            target_station_id (str): The id of the target station.
            source_station_departure_datetime (datetime): The departure datetime at the source station.
            max_transfers (int): Non-negative maximum PT transfers; 0 permits direct journeys only. Defaults to 999.
            detailed (bool): Whether to return the detailed journey plan.
        Returns:
            tp.Union[tp.Dict[str, tp.Any], None]: The fastest PT journey plan or None if no journey is found.
            
            The returned dictionary contains the following keys:
            * 'duration' (int): duration [s] from departure at the source station to arrival at the target station.
            * 'trip_time' (int): travel time [s] from departure at the source stop until arrival at the target stop.
            * 'num_transfers' (int): number of transfers.
            * 'source_station_departure_time' (int): departure timestamp [s] at the source station.
            * 'source_station_departure_day' (str): 'current_day' or 'next_day' depending on the departure time.
            * 'source_transfer_time' (int): transfer time [s] from the source station to the source stop.
            * 'source_waiting_time' (int): waiting time [s] from arrival at the source stop until departure of the trip., 
            * 'target_transfer_time' (int): transfer time [s] from the target stop to the target station.
            * 'target_station_arrival_time' (int):arrival timestamp [s] at the target station.
            * 'target_station_arrival_day' (str): 'current_day' or 'next_day' depending on the arrival time.
            * 'steps' (List[dict]): a list of journey steps, where each step contains:
                - 'duration' (int): step duration [s] from departure at the start stop until arrival at the end stop.
                - 'departure_time' (int): departure timestamp [s] at the start stop.
                - 'arrival_time' (int): arrival timestamp [s] at the end stop.
                - 'from_stop_id' (str): ID of the start stop.
                - 'to_stop_id' (str): ID of the end stop.
                - ... (other step keys, check the C++ Raptor router documentation for more details).
        """
        if not isinstance(source_station_departure_datetime, datetime):
            raise TypeError("source_station_departure_datetime must be a datetime")
        if isinstance(max_transfers, bool) or not isinstance(max_transfers, Real):
            raise TypeError("max_transfers must be an integer-valued number")
        if not isinstance(max_transfers, Integral) and (
            not isfinite(max_transfers) or int(max_transfers) != max_transfers
        ):
            raise ValueError("max_transfers must be a finite integer-valued number")
        if not 0 <= max_transfers <= 2147483647:
            raise ValueError("max_transfers must be between 0 and 2147483647")

        try:
            included_sources = self._get_included_stops_and_transfer_times(source_station_id)
            included_targets = self._get_included_stops_and_transfer_times(target_station_id)
            return self.pt_router.return_fastest_pt_journey_1to1(
                source_station_departure_datetime, included_sources, included_targets,
                int(max_transfers), detailed,
            )
        except Exception:
            LOG.exception(
                "PT query failed: source=%r, target=%r, departure=%s, max_transfers=%r",
                source_station_id, target_station_id, source_station_departure_datetime, max_transfers,
            )
            raise
    
    def _get_included_stops_and_transfer_times(self, station_id: str) -> tp.List[tp.Tuple[str, int]]:
        """This method returns the included stops and transfer times for a given station.

        The mapping DataFrame is read-only during operation. Assigning a replacement
        DataFrame, including after a reload, invalidates cached successful lookups.
        Each call returns its own list; cached stop/transfer pairs are immutable.

        Args:
            station_id (str): The id of the station.
        Returns:
            tp.List[tp.Tuple[str, int]]: Stop IDs paired with station-stop transfer times.
        """
        if not isinstance(station_id, str):
            raise TypeError("Station ID must be a string")
        if not station_id.strip():
            raise ValueError("Station ID must be a non-empty string")
        stations = self.stations_fp_df
        cache_state = getattr(self, "_station_stop_cache", None)
        if cache_state is None or cache_state[0] is not stations:
            cache_state = (stations, {})
            self._station_stop_cache = cache_state
        cache = cache_state[1]
        if station_id in cache:
            return list(cache[station_id])
        station_data = stations[stations["station_id"] == station_id]
        
        if station_data.empty:
            raise ValueError(f"Station ID {station_id} not found in the stations data")
            
        parsed = {}
        for column in ("stops_included", "station_stop_transfer_times"):
            raw_value = station_data[column].iloc[0]
            try:
                value = ast.literal_eval(raw_value.replace(';', ','))
            except (AttributeError, TypeError, ValueError, SyntaxError) as exc:
                raise ValueError(f"Invalid {column} for station {station_id}: expected a list") from exc
            if not isinstance(value, list):
                raise ValueError(f"Invalid {column} for station {station_id}: expected a list")
            parsed[column] = value

        included_ids = parsed["stops_included"]
        transfer_times = parsed["station_stop_transfer_times"]
        if not included_ids or len(included_ids) != len(transfer_times):
            raise ValueError(f"Station {station_id} must have non-empty stop and transfer-time lists of equal length")
        if station_data["num_stops_included"].iloc[0] != len(included_ids):
            raise ValueError(f"num_stops_included does not match the stop list for station {station_id}")
        if any(not isinstance(stop_id, str) or not stop_id.strip() for stop_id in included_ids):
            raise ValueError(f"Stop IDs for station {station_id} must be non-empty strings")
        if len(set(included_ids)) != len(included_ids):
            raise ValueError(f"Duplicate stop IDs for station {station_id}")
        if any(
            isinstance(time, bool) or not isinstance(time, Real)
            or not 0 <= time <= 2147483647 or int(time) != time
            for time in transfer_times
        ):
            raise ValueError(f"Transfer times for station {station_id} must be non-negative 32-bit integer values")
        included_stops = tuple((stop_id, int(time)) for stop_id, time in zip(included_ids, transfer_times))
        cache[station_id] = included_stops
        return list(included_stops)



if __name__ == "__main__":
    # Test/run the pt router module： python -m src.routing.pt.RaptorRouterCpp
    # To prepare the GTFS data, please check: src/preprocessing/pt/PTRouterGTFSPreperation.ipynb

    gtfs_dir = "data/pt/example_network/example_gtfs/matched"
    router = RaptorRouterCpp(gtfs_dir)

    source_station_departure_datetime = datetime(2024, 1, 1, 0, 4, 0)
    import time
    start_time = time.time()
    print(router.find_fastest_pt_journey_1to1("s1", "s14", source_station_departure_datetime, 3, detailed=False))
    print(f"Time taken: {time.time() - start_time} seconds")
    start_time = time.time()
    print(router.find_fastest_pt_journey_1to1("s1", "s14", source_station_departure_datetime, 3, detailed=True))
    print(f"Time taken: {time.time() - start_time} seconds")

# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
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

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *

LOG = logging.getLogger(__name__)
LARGE_INT = 100000000
BUFFER_SIZE = 100

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
    def __init__(self, fp_gtfs_dir: str):
        super().__init__()

        # initialize the pt router
        self.pt_router = None
        self._initialize_pt_router(fp_gtfs_dir)

        # load the stations and street station transfers from the gtfs data
        self.stations_fp_df = self._load_stations_from_gtfs(fp_gtfs_dir)
        self.street_station_transfers_fp_df = self._load_street_station_transfers_from_gtfs(fp_gtfs_dir)

        self.pt_rq_db: tp.Dict[str, tp.Tuple[int, int, int]] = {}  # key: rid_struct, value: (duration, departure_time, arrival_time)

        LOG.debug("PT operator initialized successfully.")

    def _initialize_pt_router(self, fp_gtfs_dir: str):
        """This method initializes the PT router.

        Args:
            fp_gtfs_dir (str): The directory containing the GTFS data of the operator, for example: "./data/gtfs/op1"
        """
        # check if the directories exist and is all mandatory files present
        mandatory_files = ["agency_fp.txt", "stops_fp.txt", "trips_fp.txt", "routes_fp.txt", "calendar_fp.txt", "stop_times_fp.txt", "stations_fp.txt", "transfers_fp.txt"]
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
        """
        dtypes = {
            'node_id': 'int',
            'closest_station_id': 'str',
            'street_transfer_time': 'int',
        }
        return pd.read_csv(os.path.join(fp_gtfs_dir, "street_station_transfers_fp.txt"), dtype=dtypes)

    def return_pt_travel_costs_1to1(
            self,
            rid_struct: str,
            source_station_id: str,
            target_station_id: str,
            arrival_datetime: datetime,
            max_transfers: int=-1,
        ) -> tp.Tuple[int, int, int]:
        """This method will return the pt travel costs (time) of the fastest journey between two pt stations.
        A station may consist of multiple stops.

        Args:
            rid_struct (str): The sub rid struct of the journey.
            source_station_id (str): The id of the source station.
            target_station_id (str): The id of the target station.
            arrival_datetime (datetime): The arrival datetime of the journey.
            max_transfers (int): The maximum number of transfers allowed in the journey, -1 for no limit.

        Returns:
            tp.Tuple[int, int, int]: (duration, departure_time, arrival_time)
        """
        # get all included stops for the source and target station
        included_sources = self._get_included_stops_and_transfer_times(source_station_id)
        included_targets = self._get_included_stops_and_transfer_times(target_station_id)

        duration, departure_time, arrival_time = self.pt_router.return_best_pt_costs_1to1(arrival_datetime, included_sources, included_targets, max_transfers)
        self._record_pt_rq_db(rid_struct, departure_time, arrival_time, duration)
        return duration, departure_time, arrival_time

    def return_fastest_pt_journey_1to1(
            self,
            source_station_id: str,
            target_station_id: str,
            arrival_datetime: datetime,
            max_transfers: int=-1,
        ) -> tp.Dict[str, tp.Any]:
        """This method will return the fastest pt journey plan between two pt stations.
        A station may consist of multiple stops.

        Args:
            source_station_id (str): The id of the source station.
            target_station_id (str): The id of the target station.
            arrival_datetime (datetime): The arrival datetime of the journey.
            max_transfers (int): The maximum number of transfers allowed in the journey, -1 for no limit.

        Returns:
            tp.Dict[str, tp.Any]: The fastest pt journey plan.
        """
        # get all included stops for the source and target station
        included_sources = self._get_included_stops_and_transfer_times(source_station_id)
        included_targets = self._get_included_stops_and_transfer_times(target_station_id)

        return self.pt_router.return_best_pt_journey_1to1(arrival_datetime, included_sources, included_targets, max_transfers)
    
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
    
    def _record_pt_rq_db(
            self,
            rid_struct: str,
            departure_time: int,
            arrival_time: int,
            duration: int,
        ):
        """This method will record the pt request in the pt request database.

        Args:
            rid_struct (str): The sub rid struct of the journey.
            departure_time (int): The departure time of the journey.
            arrival_time (int): The arrival time of the journey.
            duration (int): The duration of the journey.
        """
        if rid_struct not in self.pt_rq_db:
            self.pt_rq_db[rid_struct] = (duration, departure_time, arrival_time)
        elif self.pt_rq_db[rid_struct][1] != departure_time:
            LOG.debug(f"PT request {rid_struct} has a new departure time. Overwriting the existing request.")
            self.pt_rq_db[rid_struct] = (duration, departure_time, arrival_time)
        else:
            LOG.debug(f"PT request {rid_struct} already exists in the pt request database.")
            return

# if __name__ == "__main__":
#     # Test the pt control class
#     example_gtfs_dir = "/Users/dch/projects/fleetpy/github/pt/data/pt/example_network/example_gtfs/matched"
#     pt_control = PTControlBasicCpp(example_gtfs_dir)
#     arrival_datetime = datetime(2024, 1, 1, 0, 0, 1)
#     rid_struct = "1-8"
#     import time
#     start_time = time.time()
#     print(pt_control.return_pt_travel_costs_1to1(rid_struct, "s7", "s16", arrival_datetime, 1))
#     print(f"Time taken: {time.time() - start_time} seconds")
#     start_time = time.time()
#     print(pt_control.return_fastest_pt_journey_1to1("s7", "s16", arrival_datetime, 1))
#     print(f"Time taken: {time.time() - start_time} seconds")

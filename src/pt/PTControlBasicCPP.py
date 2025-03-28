# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import time
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

        # Initialize the pt router
        self.pt_router = None
        self._initialize_pt_router(fp_gtfs_dir)

        # load the stations from the gtfs data
        self.stations = self._load_stations_from_gtfs(fp_gtfs_dir)

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
        return pd.read_csv(os.path.join(fp_gtfs_dir, "stations_fp.txt"))

    def return_pt_travel_costs_1to1(
            self,
            source_station_id: str,
            target_station_id: str,
            year: int, month: int, day: int, hours: int, minutes: int, seconds: int,
            max_transfers: int=-1,
        ) -> tp.Tuple[int, int, int]:
        """This method will return the pt travel costs (time) of the fastest journey between two pt stations.
        A station may consist of multiple stops.

        Args:
            source_station_id (str): The id of the source station.
            target_station_id (str): The id of the target station.
            year (int): The year of the journey.
            month (int): The month of the journey.
            day (int): The day of the journey.
            hours (int): The hour of the journey.
            minutes (int): The minute of the journey.
            seconds (int): The second of the journey.
            max_transfers (int): The maximum number of transfers allowed in the journey, -1 for no limit.

        Returns:
            tp.Tuple[int, int, int]: (duration, departure_time, arrival_time)
        """
        # get all included stops for the source and target station
        included_sources = self._get_included_stops_and_transfer_times(source_station_id)
        included_targets = self._get_included_stops_and_transfer_times(target_station_id)

        return self.pt_router.return_best_pt_costs_1to1(year, month, day, hours, minutes, seconds, included_sources, included_targets, max_transfers)

    def return_best_pt_journey_1to1(
            self,
            source_station_id: str,
            target_station_id: str,
            year: int, month: int, day: int, hours: int, minutes: int, seconds: int,
            max_transfers: int=-1,
        ) -> tp.Dict[str, tp.Any]:
        """This method will return the best pt journey plan between two pt stations.
        A station may consist of multiple stops.
        """
        # get all included stops for the source and target station
        included_sources = self._get_included_stops_and_transfer_times(source_station_id)
        included_targets = self._get_included_stops_and_transfer_times(target_station_id)

        return self.pt_router.return_best_pt_journey_1to1(year, month, day, hours, minutes, seconds, included_sources, included_targets, max_transfers)
    
    def _get_included_stops_and_transfer_times(self, station_id: str) -> tp.Tuple[tp.List[str], tp.List[int]]:
        """This method will return the included stops and transfer times for a given station.
        """
        # TODO: the street transfer times are not included yet
        included_ids_str = self.stations_fp_df[self.stations_fp_df["station_id"] == station_id]["stops_included"].tolist()
        included_ids = ast.literal_eval(included_ids_str[0])
        transfer_times_str = self.stations_fp_df[self.stations_fp_df["station_id"] == station_id]["station_stop_transfer_times"].tolist()
        transfer_times = ast.literal_eval(transfer_times_str[0])
        return [(stop_id, int(transfer_time)) for stop_id, transfer_time in zip(included_ids, transfer_times)]
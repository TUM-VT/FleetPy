from __future__ import annotations

# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import os
import logging
import typing as tp
# additional module imports (> requirements)
# ------------------------------------------
import pandas as pd
import numpy as np

# src imports
# -----------
from src.fleetctrl.forecast.PerfectForecastZoning import PerfectForecastZoneSystem
from src.routing.NetworkBase import return_position_from_str
# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
from src.misc.globals import *
#from src.fleetctrl.FleetControlBase import PlanRequest # TODO # circular dependency!
# set log level to logging.DEBUG or logging.INFO for single simulations

if tp.TYPE_CHECKING:
    from src.fleetctrl.planning.PlanRequest import PlanRequest

LOG_LEVEL = logging.WARNING
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_MATSimIterationForecast = {
    "doc" :     """
    this class can be for use cases like MATSim coupling, where you want to produce forecasts for the next iteration based on the demand of the last iteration
    """,
    "inherit" : "PerfectForecastZoneSystem",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class AbsReq():
    def __init__(self, o_pos, d_pos):
        self.o_pos = o_pos
        self.d_pos = d_pos

class MATSimIterationForecast(PerfectForecastZoneSystem):
    """
    this class can be for use cases like MATSim coupling, where you want to produce forecasts for the next iteration based on the demand of the last iteration
    """
    def __init__(self, zone_network_dir, scenario_parameters, dir_names, operator_attributes):
        if operator_attributes.get(G_RA_FC_FNAME) is not None:
            LOG.warning("forecast file for forecast given. will not be loaded in this forecast system!")
        super().__init__(zone_network_dir, scenario_parameters, dir_names, operator_attributes)
        if self.fc_temp_resolution is None:
            self.fc_temp_resolution = operator_attributes[G_RA_FC_TR] # TODO ?
            
        current_matsim_iteration = operator_attributes.get("matsim_iteration")
        if current_matsim_iteration is None:
            current_matsim_iteration = operator_attributes.get("op_matsim_iteration")
        if current_matsim_iteration is None:
            raise EnvironmentError("matsim_iteration parameter not found in scenario_parameters. This is required for the MATSimIterationForecast.")
            
        self._last_iteration_requests = {} # dict of request_time -> list of dict {"o_node": o_node, "d_node": d_node} (to mimic the structure of future_requests in the demand object, but for past requests)
        
        c = 0
        if current_matsim_iteration > 0:
            # load the requests from the last iteration
            current_output_dir = dir_names[G_DIR_OUTPUT] # assume highest folder name is the current iteration
            last_iteration_output_dir = os.path.join(os.path.dirname(current_output_dir), f"{current_matsim_iteration-1}") # assume output folders are named like "0", "1", ...
            if os.path.exists(os.path.join(last_iteration_output_dir, "1_user-stats.csv")):
                LOG.info(f"Loading past requests from last iteration {current_matsim_iteration-1} for forecast. Looking in {last_iteration_output_dir}")
                past_request_df = pd.read_csv(os.path.join(last_iteration_output_dir, "1_user-stats.csv"))
                for earliest_pickup_time, start, end in zip(past_request_df["earliest_pickup_time"], past_request_df["start"], past_request_df["end"]):
                    o_pos = return_position_from_str(start)
                    d_pos = return_position_from_str(end)
                    if float(earliest_pickup_time) not in self._last_iteration_requests:
                        self._last_iteration_requests[float(earliest_pickup_time)] = {}
                    self._last_iteration_requests[float(earliest_pickup_time)][c] = AbsReq(o_pos, d_pos)
                    c += 1
            else:
                LOG.info(f"No past requests file found for iteration {current_matsim_iteration-1} in {last_iteration_output_dir}." )
        LOG.info(f"Initialized MATSimIterationForecast with {sum(len(requests) for requests in self._last_iteration_requests.values())} past requests.")
        
    def _get_future_requests(self, t):
        # for this forecast, we assume that the future requests are the same as the past requests in the last iteration (but only if they are in the future of the current time step)
        return self._last_iteration_requests.get(t, {})
            

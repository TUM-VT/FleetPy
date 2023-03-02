import os
import logging
import numpy as np
import pandas as pd
from src.fleetctrl.pricing.DynamicPricingBase import DynamicPrizingBase
from src.misc.functions import load_function

from src.misc.globals import *
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_UtilizationBasedDP = {
    "doc" : """This is a utilization dependent pricing scheme, which adjusts the general fare factor.
            After certain simulation intervals, the current fleet utilization is measured.
            If the utilization over-(under-)shoots a certain threshold, fares are in-(de-)creased.""",
    "inherit" : "DynamicPrizingBase",
    "input_parameters_mandatory": [G_OP_DYN_P_FUNC, G_OP_UTIL_EVAL_INT],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class UtilizationBasedDP(DynamicPrizingBase):
    """This is a utilization dependent pricing scheme, which adjusts the general fare factor.
    After certain simulation intervals, the current fleet utilization is measured.
    If the utilization over-(under-)shoots a certain threshold, fares are in-(de-)creased."""
    def __init__(self, fleetctrl, operator_attributes, solver="Gurobi"):
        """Initialization of dynamic pricing class.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param solver: solver for optimization problems
        """
        super().__init__(fleetctrl, operator_attributes, solver)
        # children classes:
        # - check of additionally required attributes from operator_attributes
        # - save these as class attributes
        self.eval_interval = operator_attributes.get(G_OP_UTIL_EVAL_INT)
        if not self.eval_interval:
            horizon = operator_attributes[G_OP_REPO_TH_DEF]
            self.eval_interval = horizon[1] - horizon[0]
        #
        self.current_base_fare_factor = 1.0
        self.current_distance_fare_factor = 1.0
        self.current_general_factor = 1.0
        #
        self.interval_start_key = None
        self.last_interval_utils = {} # sim_time -> value
        #
        self.util_price_f = load_function(operator_attributes[G_OP_DYN_P_FUNC])
        self.record_df_cols = ["sim_time", "utilization", "general_factor"]
        self.record_df_index_cols = self.record_df_cols[:1]
        self.record_df = pd.DataFrame([], columns=self.record_df_cols)
        self.record_df.set_index(self.record_df_index_cols, inplace=True)
        # TODO # check differences to other util-pricing scheme

    def get_elastic_price_factors(self, sim_time, expected_pu_time=None, o_pos=None, d_pos=None):
        """This method returns current time dependent fare scales. If it is called with sim_time, the price factors
        are updated first.

        :param sim_time: current simulation time
        :type sim_time: int
        :param expected_pu_time: expected pick-up time (re
        :type expected_pu_time: float
        :param o_pos: origin of request
        :type o_pos: tuple
        :param d_pos: destination of request
        :type d_pos: tuple
        :return: tuple of base_fare_factor, distance_fare_factor, general_factor
        :rtype: (float, float, float)
        """
        return self.current_base_fare_factor, self.current_distance_fare_factor, self.current_general_factor

    def update_current_price_factors(self, sim_time):
        """This method updates the current time dependent fare scales based on the dynamic pricing strategy.

        :param sim_time: current simulation time
        :type sim_time: int
        :return: None
        """
        if self.interval_start_key is None:
            self.interval_start_key = sim_time
        self.last_interval_utils[sim_time] = self.fleetctrl.compute_current_fleet_utilization(sim_time)[0]
        if sim_time - self.interval_start_key >= self.eval_interval:
            mean_util = np.mean(list(self.last_interval_utils.values()))
            self.current_general_factor = self.util_price_f.get_y(mean_util)
            self.record_df.loc[sim_time, ["utilization", "general_factor"]] = [mean_util, self.current_general_factor]
            LOG.debug(f"Simulation time {sim_time}: updated current general dynamic pricing factor:\n"
                      f"{self.current_general_factor} (mean utilization value: {mean_util})")
            self.interval_start_key = sim_time
            self.last_interval_utils = {}
            if os.path.isfile(self.record_f):
                write_mode = "a"
                write_header = False
            else:
                write_mode = "w"
                write_header = True
            self.record_df.to_csv(self.record_f, header=write_header, mode=write_mode)
            self.record_df = pd.DataFrame([], columns=self.record_df_cols)
            self.record_df.set_index(self.record_df_index_cols, inplace=True)

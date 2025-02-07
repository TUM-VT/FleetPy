import os
import logging
import numpy as np
import pandas as pd
from src.fleetctrl.pricing.DynamicPricingBase import DynamicPrizingBase
from src.misc.functions import load_function

from src.misc.globals import *
LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_DemoDP = {
    "doc" : """This is a utilization dependent pricing scheme, which adjusts the general fare factor.
            After certain simulation intervals, the current fleet utilization is measured.
            If the utilization over-(under-)shoots a certain threshold, fares are in-(de-)creased.""",
    "inherit" : "DynamicPrizingBase",
    "input_parameters_mandatory": [G_OP_DYN_P_FUNC, G_OP_UTIL_EVAL_INT],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class DemoDP(DynamicPrizingBase):
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

        # get the evaluation interval
        self.eval_interval = operator_attributes.get(G_OP_DP_EVAL_INT)
        if not self.eval_interval:
            horizon = operator_attributes[G_OP_REPO_TH_DEF]
            self.eval_interval = horizon[1] - horizon[0]
        self.interval_start_time = 0
        # initialize the price factors
        self.current_base_fare_factor = 1.0
        self.current_distance_fare_factor = 1.0
        self.current_general_factor = 1.0

    def get_elastic_price_factors(self, sim_time, expected_pu_time=None, o_pos=None, d_pos=None):
        """This method returns current time dependent fare scales to the operator.

        :param sim_time: current simulation time
        :type sim_time: int
        :param expected_pu_time: expected pick-up time
        :type expected_pu_time: float
        :param o_pos: origin of request
        :type o_pos: tuple
        :param d_pos: destination of request
        :type d_pos: tuple
        :return: tuple of base_fare_factor, distance_fare_factor, general_factor
        :rtype: (float, float, float)
        """
        # based on sim_time, expected_pu_time, o_pos, d_pos, different fare factors can be returned.
        return self.current_base_fare_factor, self.current_distance_fare_factor, self.current_general_factor

    def update_current_price_factors(self, sim_time):
        """This method updates the current time dependent fare scales based on the dynamic pricing strategy.

        :param sim_time: current simulation time
        :type sim_time: int
        :return: None
        """
        if sim_time - self.interval_start_time >= self.eval_interval:
            # TODO: implement your dynamic pricing logic here
            if sim_time >= 3600:
                # self.current_base_fare_factor = 0
                # self.current_distance_fare_factor = 0
                self.current_general_factor = 10000
            LOG.debug(f"Simulation time {sim_time}: updated current general dynamic pricing factor:\n"
                      f"Base fare factor: {self.current_base_fare_factor}\n"
                      f"Distance fare factor: {self.current_distance_fare_factor}\n"
                      f"General factor: {self.current_general_factor}")
            # update the interval start time    
            self.interval_start_time = sim_time
            # TODO: record the price factors to the record dataframe, if needed
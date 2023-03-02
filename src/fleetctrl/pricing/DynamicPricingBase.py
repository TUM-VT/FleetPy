from abc import abstractmethod, ABC
import os
import logging
import pandas as pd
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_DynamicPrizingBase = {
    "doc" :  """This sub-module implements strategies that dynamically set fares for trips. """,
    "inherit" : None,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class DynamicPrizingBase(ABC):
    def __init__(self, fleetctrl, operator_attributes, solver="Gurobi"):
        """This sub-module implements strategies that dynamically set fares for trips.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param solver: solver for optimization problems
        """
        self.fleetctrl = fleetctrl
        self.solver_key = solver
        # children classes:
        # - check of additionally required attributes from operator_attributes
        # - save these as class attributes
        self.output_dir = fleetctrl.dir_names[G_DIR_OUTPUT]
        self.record_f = os.path.join(self.output_dir, f"5-{self.fleetctrl.op_id}_dyn_pricing_info.csv")
        # columns can also be redefined in child classes
        self.record_df_cols = ["sim_time", "o_zone_id", "d_zone_id",
                               "base_fare_factor", "distance_fare_factor", "general_factor"]
        self.record_df_index_cols = self.record_df_cols[:3]
        self.record_df = pd.DataFrame([], columns= self.record_df_cols)
        self.record_df.set_index(self.record_df_index_cols, inplace=True)

    @abstractmethod
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
        base_fare_factor = 1.0
        distance_fare_factor = 1.0
        general_factor = 1.0
        return base_fare_factor, distance_fare_factor, general_factor

    @abstractmethod
    def update_current_price_factors(self, sim_time):
        """This method updates the current time dependent fare scales based on the dynamic pricing strategy.

        :param sim_time: current simulation time
        :type sim_time: int
        :return: None
        """
        pass

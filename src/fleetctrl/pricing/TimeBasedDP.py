from src.fleetctrl.pricing.DynamicPricingBase import DynamicPrizingBase

INPUT_PARAMETERS_TimeBasedDP = {
    "doc" :  """This strategy sets fares at specific times during the simulation based on an input file. """,
    "inherit" : "DynamicPrizingBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}

class TimeBasedDP(DynamicPrizingBase):
    def __init__(self, fleetctrl, operator_attributes, solver="Gurobi"):
        """This strategy sets fares at specific times during the simulation based on an input file.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param solver: solver for optimization problems
        """
        super().__init__(fleetctrl, operator_attributes, solver)
        # TODO # TimeBasedDP.__init__()
        # load input file
        # define current parameters as attributes
        raise NotImplementedError("TimeBaseDP is not implemented currently!")

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
        # TODO # TimeBasedDP.get_elastic_price_factors()
        # return current attribute values
        pass

    def update_current_price_factors(self, sim_time):
        """This method updates the current time dependent fare scales based on the dynamic pricing strategy.

        :param sim_time: current simulation time
        :type sim_time: int
        :return: None
        """
        # TODO # TimeBasedDP.update_current_price_factors()
        # check whether current attribute values need to be updated according to input file
        pass

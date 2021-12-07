from abc import abstractmethod, ABC
import logging

from src.misc.globals import *
LOG = logging.getLogger(__name__)


class DynamicFleetSizingBase(ABC):
    def __init__(self, fleetctrl, operator_attributes, solver="Gurobi"):
        """Initialization of repositioning class.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param solver: solver for optimization problems
        """
        self.fleetctrl = fleetctrl
        self.solver_key = solver
        # children classes:
        # - check of additionally required attributes from operator_attributes
        # - save these as class attributes

    @abstractmethod
    def check_and_change_fleet_size(self, sim_time):
        """This method checks whether fleet vehicles should be activated or deactivated.

        :param sim_time: current simulation time
        :return: net change in fleet size
        """
        return 0

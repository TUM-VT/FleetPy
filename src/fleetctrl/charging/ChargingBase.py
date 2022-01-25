from abc import abstractmethod, ABC
import logging
from typing import Dict, List, Any, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from src.infra.ChargingStation import ChargingInfrastructureOperator

LOG = logging.getLogger(__name__)


class ChargingBase(ABC):
    def __init__(self, fleetctrl, operator_attributes, solver="Gurobi"):
        """Initialization of charging class.

        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param solver: solver for optimization problems
        """
        self.fleetctrl = fleetctrl
        self.cm: ChargingInfrastructureOperator = fleetctrl.charging_management
        self.routing_engine = fleetctrl.routing_engine
        self.solver_key = solver
        # children classes:
        # - check of additionally required attributes from operator_attributes
        # - save these as class attributes

    @abstractmethod
    def time_triggered_charging_processes(self, sim_time):
        """This method can be used to apply a charging strategy and additionally, charge vehicles in depots if there are
        free slots.

        :param sim_time: current simulation time
        :return: None
        """
        pass

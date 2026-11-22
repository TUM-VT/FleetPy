from abc import abstractmethod, ABC
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.infra.Zoning import ZoneSystem
from src.routing.road.NetworkBase import NetworkBase
from src.FleetSimulationBase import FleetSimulationBase
from src.ml_gym.hooks_manager import Events


class AbstractObserver(ABC):

    def __init__(self, fleet_operator_id = None):
        """ The abstractor Observer class

        :param fleet_operator_id: The fleet operator ID is required when the observer has access to multiple fleet
        operators, such as inside FleetSimulationBase class. It specifies which operator to observe at the event.
        """
        self.fleet_operator_id = fleet_operator_id
        self.fleet_controller: FleetControlBase | None = None
        self.repositioning: RepositioningBase | None = None
        self.zone_system: ZoneSystem | None = None
        self.network: NetworkBase | None = None

    @abstractmethod
    def observe(self, fleetpy_module, event: Events, **kwargs):
        pass

    def extract_modules(self, fleetpy_module, event: Events):
        """ It is a helper method to extracts various other modules depending on the calling fleetpy module for observation """

        if isinstance(fleetpy_module, FleetControlBase):
            self.fleet_controller = fleetpy_module
            self.repositioning = self.fleet_controller.repo
            self.zone_system = self.fleet_controller.zones
            self.network = self.fleet_controller.routing_engine
        elif isinstance(fleetpy_module, RepositioningBase):
            self.repositioning = fleetpy_module
            self.fleet_controller = fleetpy_module.fleetctrl
            self.zone_system = self.fleet_controller.zones
            self.network = self.fleet_controller.routing_engine
        elif isinstance(fleetpy_module, FleetSimulationBase):
            if len(fleetpy_module.operators) > 1 and self.fleet_operator_id is None:
                raise ValueError(
                    f"Observer {self} is attached to event {event}, but {fleetpy_module} has multiple fleet "
                    f"operators: {fleetpy_module.operators}. Specify 'fleet_operator_id' when initializing the "
                    f"observer.")
            op_id = self.fleet_operator_id if len(fleetpy_module.operators) > 1 else 0
            self.fleet_controller = fleetpy_module.operators[op_id]
            self.repositioning = self.fleet_controller.repo
            self.zone_system = fleetpy_module.zones
            self.network = fleetpy_module.routing_engine
        else:
            raise TypeError(f"The extraction method for the observer {self} does not have an entry for the fleetpy "
                            f"module {fleetpy_module}")
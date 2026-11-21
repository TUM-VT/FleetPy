from abc import abstractmethod, ABC
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.infra.Zoning import ZoneSystem
from src.routing.road.NetworkBase import NetworkBase


class AbstractObserver(ABC):

    def __init__(self):
        self.fleet_controller: FleetControlBase | None = None
        self.repositioning: RepositioningBase | None = None
        self.zone_system: ZoneSystem | None = None
        self.network: NetworkBase | None = None

    @abstractmethod
    def observe(self, fleetpy_module, **kwargs):
        pass

    def extract_modules(self, fleetpy_module):
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
        else:
            raise TypeError(f"The extraction method for the observer {self} does not have an entry for the fleetpy "
                            f"module {fleetpy_module}")
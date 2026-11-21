from abc import abstractmethod, ABC
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.infra.Zoning import ZoneSystem
from src.routing.road.NetworkBase import NetworkBase


class AbstractObserver(ABC):

    @abstractmethod
    def observe(self, fleetpy_module, **kwargs):
        pass

    def extract_modules(self, fleetpy_module) -> tuple[FleetControlBase, RepositioningBase, ZoneSystem, NetworkBase]:
        """ It is a helper method to extracts various other modules depending on the calling fleetpy module for observation """
        
        if isinstance(fleetpy_module, FleetControlBase):
            fleet_controller = fleetpy_module
            repositioning = fleet_controller.repo
            zone_system = fleet_controller.zones
            network = fleet_controller.routing_engine
        elif isinstance(fleetpy_module, RepositioningBase):
            repositioning = fleetpy_module
            fleet_controller = fleetpy_module.fleetctrl
            zone_system = fleet_controller.zones
            network = fleet_controller.routing_engine
        else:
            raise TypeError(f"The extraction method for the observer {self} does not have an entry for the fleetpy "
                            f"module {fleetpy_module}")
        return fleet_controller, repositioning, zone_system, network
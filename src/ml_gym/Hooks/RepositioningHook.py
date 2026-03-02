from __future__ import annotations
# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import time
from abc import abstractmethod, ABCMeta
from typing import Dict, List, Any, Tuple, TYPE_CHECKING

# -------------------------------------------------------------------------------------------------------------------- #
# local imports
from src.fleetctrl.RidePoolingBatchAssignmentFleetcontrol import RidePoolingBatchAssignmentFleetcontrol
from src.ml_gym.Hooks.HookManager import Events, Hook
from src.misc.globals import G_FCTRL_CT_RES, G_FCTRL_CT_CH, G_FCTRL_CT_DFS, G_FCTRL_CT_REPO, G_FCTRL_CT_DP
if TYPE_CHECKING:
    from src.infra.ChargingInfrastructure import OperatorChargingAndDepotInfrastructure, PublicChargingInfrastructureOperator
    from src.infra.Zoning import ZoneSystem
    from src.ml_gym.Hooks.HookManager import HookManager
    from src.routing.NetworkBase import NetworkBase
    from src.simulation.Vehicles import SimulationVehicle
    from src.fleetctrl.repositioning import RepositioningBase
    
    
LOG = logging.getLogger(__name__)

class ZonebasedRepositioningHook(Hook):
    
    def on_event(self, event, repo_module: RepositioningBase, **kwargs):
        if event != Events.OBSERVE_BEFORE_REPOSITIONING:
            return
        return super().on_event(event, repo_module: RepositioningBase, **kwargs)
    
    def _observe_demand_forecast(self, repo_module: RepositioningBase):
        sim_time = repo_module.sim_time
        list_zones = repo_module.zone_system.get_all_zones()
        t0 = sim_time + self.list_horizons[0]
        t1 = sim_time + self.list_horizons[1]
        od_fc = repo_module.zone_system.get_trip_od_forecasts(t0, t1, scale=self._weight_on_forecast)
        dep_rate_s = {zone_id : sum(od_fc.get(zone_id, {}).values()) for zone_id in list_zones}
        arr_rate_s = {zone_id : 0 for zone_id in list_zones}
        for o_zone_id, d_zone_dict in od_fc.items():
            for d_zone_id, trips in d_zone_dict.items():
                arr_rate_s[d_zone_id] += trips
        # print(demand_fc_dict)
        # print(supply_fc_dict)
        
    def _observe_zonebase_vehicle_states(self, repo_module: RepositioningBase):
        sim_time = repo_module.sim_time
        list_zones = repo_module.zone_system.get_all_zones()
        t0 = sim_time + self.list_horizons[0]
        t1 = sim_time + self.list_horizons[1]
        
        cplan_arrival_idle_dict = repo_module._get_current_veh_plan_arrivals_and_repo_idle_vehicles(t0, t1)

        # compute imbalance values and constraints
        # ----------------------------------------
        vehicles_repo_to_zone = {k: len(v[1]) for k,v in cplan_arrival_idle_dict.items()}
        number_current_own_vehicles = {k: v[0] for k, v in cplan_arrival_idle_dict.items()}
        number_idle_vehicles = {k: len(v[2]) for k,v in cplan_arrival_idle_dict.items()}
        
        return {
            "zone_to_idle_vehilces" : number_idle_vehicles,
            "zone_to_overall_available_vehilces" : number_current_own_vehicles,
            "zone_to_current_repositioning_vehicles" : vehicles_repo_to_zone
        }
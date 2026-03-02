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
    
    
LOG = logging.getLogger(__name__)
    
    
class MLHook(Hook):
    def __init__(self, q_in, q_out):
        self._q_in = q_in
        self._q_out = q_out
        
    def on_event(self, event, sim: 'RLRepoFleetControl', **kwargs):
        if event == Events.ML_OBSERVE:
            observation = self._observe(sim, **kwargs)
            print(f"Hook: Put observation: {observation}")
            self._q_out.put(observation)
        elif event == Events.ML_ACTION:
            action = self._q_in.get()
            print(f"Hook: get action: {action}")
            self._act(sim, action, **kwargs)
            
    def _observe(self, sim: 'RLRepoFleetControl', **kwargs):
        # extract observation from sim
        return None
    
    def _act(self, sim: 'RLRepoFleetControl', action, **kwargs):
        # apply action to sim
        pass

class RLRepoFleetControl(RidePoolingBatchAssignmentFleetcontrol):
    def __init__(self, op_id : int, operator_attributes : Dict, list_vehicles : List[SimulationVehicle],
                 routing_engine : NetworkBase, zone_system : ZoneSystem, scenario_parameters : Dict,
                 dir_names : Dict, op_charge_depot_infra : OperatorChargingAndDepotInfrastructure=None,
                 list_pub_charging_infra: List[PublicChargingInfrastructureOperator]= [], hook_manager: 'HookManager' = None):
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                         dir_names, op_charge_depot_infra, list_pub_charging_infra, hook_manager)
        if self.hook_manager is not None:
            q_in, q_out = self.hook_manager.get_queues()
            ml_hook = MLHook(q_in, q_out)
            self.hook_manager.register(Events.ML_OBSERVE, ml_hook)
            self.hook_manager.register(Events.ML_ACTION, ml_hook)
        
    def _call_time_trigger_additional_tasks(self, sim_time):
        """This method can be used to trigger all fleet operational tasks that are not related to request assignment:
        - charging processes
        - changes to active fleet size
        - vehicle repositioning
        - dynamic pricing
        All these methods are controlled by scenario input parameters.

        :param sim_time: current simulation time
        :return: None
        """
        add_dyn_dict = {}

        # 1) Charging Processes
        # ---------------------
        if self.reservation_module:
            t0 = time.perf_counter()
            self.reservation_module.time_trigger(sim_time)
            add_dyn_dict[G_FCTRL_CT_RES] = round(time.perf_counter() - t0, 3)

        # 1) Charging Processes
        # ---------------------
        if self.charging_strategy:
            t0 = time.perf_counter()
            self.charging_strategy.time_triggered_charging_processes(sim_time)
            add_dyn_dict[G_FCTRL_CT_CH] = round(time.perf_counter() - t0, 3)

        # 2) Dynamic Fleet Sizing
        # -----------------------
        repo_activated_veh = False
        if self.dyn_fleet_sizing:
            t0 = time.perf_counter()
            change_in_fleet_size = self.dyn_fleet_sizing.check_and_change_fleet_size(sim_time)
            if change_in_fleet_size > 0:
                repo_activated_veh = True
            add_dyn_dict[G_FCTRL_CT_DFS] = round(time.perf_counter() - t0, 3)

        # 3) Repositioning
        # -------------------
        print(f"start triggers now {sim_time}")
        self.hook_manager.trigger(Events.ML_OBSERVE, sim=self, sim_time=sim_time)
        self.hook_manager.trigger(Events.ML_ACTION, sim=self, sim_time=sim_time)
        # if self.repo is not None and (sim_time % self.repo_time_step == 0 or repo_activated_veh):
        #     t0 = time.perf_counter()
        #     LOG.info("Calling repositioning algorithm! (because of activated vehicles? {})".format(repo_activated_veh))
        #     # vehplans no longer locked, because repo called very often
        #     self.repo.determine_and_create_repositioning_plans(sim_time)
        #     add_dyn_dict[G_FCTRL_CT_REPO] = round(time.perf_counter() - t0, 3)

        # 4) Dynamic Pricing
        # ------------------
        if self.dyn_pricing is not None:
            t0 = time.perf_counter()
            self.dyn_pricing.update_current_price_factors(sim_time)
            add_dyn_dict[G_FCTRL_CT_DP] = round(time.perf_counter() - t0, 3)

        # 5) Move idle vehicles if on-street parking is not allowed
        # ---------------------------------------------------------
        if not self.allow_on_street_parking:
            self.charging_management.move_idle_vehicles_to_nearest_depots(sim_time, self)

        # record
        if add_dyn_dict:
            self._add_to_dynamic_fleetcontrol_output(sim_time, add_dyn_dict)
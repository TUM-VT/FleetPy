from __future__ import annotations
# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import json
import logging
import os
from typing import Dict, List, TYPE_CHECKING

# -------------------------------------------------------------------------------------------------------------------- #
# local imports
from src.fleetctrl.PoolingIRSOnly import PoolingInsertionHeuristicOnly
from src.ml_gym.HookManager import Events, Hook
from src.misc.globals import *
if TYPE_CHECKING:
    from src.infra.ChargingInfrastructure import OperatorChargingAndDepotInfrastructure, PublicChargingInfrastructureOperator
    from src.infra.Zoning import ZoneSystem
    from src.ml_gym.HookManager import HookManager
    from src.routing.NetworkBase import NetworkBase
    from src.simulation.Vehicles import SimulationVehicle


LOG = logging.getLogger(__name__)


class FleetStateHook(Hook):
    """Accumulates per-vehicle state rows via OBSERVE events and writes
    the complete fleet state via OUTPUT events once all vehicles have reported."""

    def __init__(self, q_in, q_out, output_f, op_id, n_vehicles):
        self._q_in = q_in
        self._q_out = q_out
        self.output_f = output_f
        self._op_id = op_id
        self._n_vehicles = n_vehicles
        self._current_time = None
        self._vehicle_rows = []

    def on_event(self, event, sim: 'SLPoolingIRSOnly', **kwargs):
        if event == Events.OBSERVE_VEHICLE_STATUS_AFTER_RECEIVE_STATUS_UPDATE:
            vehicle_row = self._observe(sim, **kwargs)
            self._vehicle_rows.append(vehicle_row)
        elif event == Events.OUTPUT_FLEET_STATE_AFTER_RECEIVE_STATUS_UPDATE:
            self._output(**kwargs)

    def _observe(self, sim: 'SLPoolingIRSOnly', **kwargs):
        """Collect one vehicle's state and accumulate it."""
        sim_time = kwargs.get("sim_time")
        vid = kwargs.get("vid")
        self._current_time = sim_time
        vehicle_row = sim.collect_vehicle_state(vid, sim_time)
        return vehicle_row

    def _output(self, **kwargs):
        """If all vehicles have reported for the current time step, assemble and write."""
        if len(self._vehicle_rows) >= self._n_vehicles:
            self._flush()

    def _flush(self):
        """Assemble fleet state from accumulated rows, put on queue for MLEnv, write to file."""
        fleet_state = {
            "time": int(self._current_time),
            "op_id": self._op_id,
            "n_vehicles": len(self._vehicle_rows),
            "columns": VEH_COLUMNS,
            "leg_columns": LEG_COLUMNS,
            "stop_columns": STOP_COLUMNS,
            "vehicles": self._vehicle_rows,
        }
        self._q_out.put(fleet_state)
        with open(self.output_f, 'a') as f:
            f.write(json.dumps(fleet_state, ensure_ascii=False, default=str) + '\n')
        self._vehicle_rows = []


class SLPoolingIRSOnly(PoolingInsertionHeuristicOnly):
    def __init__(self, op_id : int, operator_attributes : Dict, list_vehicles : List['SimulationVehicle'],
                 routing_engine : 'NetworkBase', zone_system : 'ZoneSystem', scenario_parameters : Dict,
                 dir_names : Dict, op_charge_depot_infra : 'OperatorChargingAndDepotInfrastructure'=None,
                 list_pub_charging_infra: List['PublicChargingInfrastructureOperator']= [], hook_manager: 'HookManager' = None):
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                         dir_names, op_charge_depot_infra, list_pub_charging_infra, hook_manager)

        self._fleet_state_hook = None
        if self.hook_manager is not None:
            q_in, q_out = self.hook_manager.get_queues()
            output_f = os.path.join(dir_names[G_DIR_OUTPUT], f"5-{op_id}_fleet_state.jsonl")
            self._fleet_state_hook = FleetStateHook(q_in, q_out, output_f, op_id, len(list_vehicles))
            self.hook_manager.register(Events.OBSERVE_VEHICLE_STATUS_AFTER_RECEIVE_STATUS_UPDATE, self._fleet_state_hook)
            self.hook_manager.register(Events.OUTPUT_FLEET_STATE_AFTER_RECEIVE_STATUS_UPDATE, self._fleet_state_hook)

    def receive_status_update(self, vid, simulation_time, list_finished_VRL, force_update=True):
        super().receive_status_update(vid, simulation_time, list_finished_VRL, force_update)
        if self.hook_manager is not None:
            self.hook_manager.trigger(Events.OBSERVE_VEHICLE_STATUS_AFTER_RECEIVE_STATUS_UPDATE,
                                      sim=self, sim_time=simulation_time, vid=vid)
            self.hook_manager.trigger(Events.OUTPUT_FLEET_STATE_AFTER_RECEIVE_STATUS_UPDATE,
                                      sim=self, sim_time=simulation_time, vid=vid)

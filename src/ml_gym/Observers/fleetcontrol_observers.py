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
from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from src.ml_gym.Hooks.HookManager import Events, Hook
from src.misc.globals import G_FCTRL_CT_RES, G_FCTRL_CT_CH, G_FCTRL_CT_DFS, G_FCTRL_CT_REPO, G_FCTRL_CT_DP
if TYPE_CHECKING:
    from src.infra.ChargingInfrastructure import OperatorChargingAndDepotInfrastructure, PublicChargingInfrastructureOperator
    from src.infra.Zoning import ZoneSystem
    from src.ml_gym.Hooks.HookManager import HookManager
    from src.routing.NetworkBase import NetworkBase
    from src.simulation.Vehicles import SimulationVehicle
    from src.fleetctrl.repositioning import RepositioningBase
    from src.fleetctrl.FleetControlBase import FleetControlBase
    
# ML Fleet State Record Structure
VEH_COLUMNS = [
    "vid", "op_id", "veh_type",
    "status", "status_value", "pos",
    "soc", "battery_size", "range",
    "n_pax", "max_pax", "pax_rids",
    "cl_start_time", "cl_start_pos", "cl_start_soc",
    "cl_driven_distance", "cl_remaining_time",
    "cl_remaining_route_len", "cl_remaining_route",
    "cl_locked", "cumulative_distance",
    "n_assigned_legs", "assigned_route",
    "n_plan_stops", "plan_stops",
]

LEG_COLUMNS = [
    "status", "status_value", "destination_pos",
    "duration", "power",
    "earliest_start_time", "earliest_end_time",
    "locked", "started",
    "boarding_rids", "alighting_rids", "route_len",
]

STOP_COLUMNS = [
    "pos", "state",
    "boarding_rids", "alighting_rids",
    "planned_arrival_time", "planned_departure_time",
    "remaining_time_to_departure",
    "duration", "earliest_departure", "earliest_start_time",
    "locked", "change_nr_pax",
    "charging_power", "charging_task_id",
]

G_ML_WRITE_FLEET_STATE = "ml_write_fleet_state"
    
    
LOG = logging.getLogger(__name__)

def observe_fleet_state_exhaustive(fleetctrl_module: FleetControlBase):
    sim_time = fleetctrl_module.sim_time
    _vehicle_rows = []
    for veh in fleetctrl_module.sim_vehicles:
        vid = veh.vid
    
        vehicle_row = collect_vehicle_status(fleetctrl_module, vid, sim_time)
        _vehicle_rows.append(vehicle_row)

    # If all vehicles have reported for the current time step, assemble and write.
    fleet_state = {
        "time": int(sim_time),
        "op_id": fleetctrl_module.op_id,
        "n_vehicles": int(fleetctrl_module.nr_vehicles),
        "columns": VEH_COLUMNS,
        "leg_columns": LEG_COLUMNS,
        "stop_columns": STOP_COLUMNS,
        "vehicles": _vehicle_rows,
    }
    
    return fleet_state


        
def collect_vehicle_status(fleetctrl: FleetControlBase, vid, sim_time):
    """Collect a single vehicle's status as a compact value list (order matches VEH_COLUMNS)."""
    veh_obj: SimulationVehicle = fleetctrl.sim_vehicles[vid]
    vehicle_plan: VehiclePlan = fleetctrl.veh_plans[vid]

    return [
        vid,
        veh_obj.op_id,
        veh_obj.veh_type,
        veh_obj.status.display_name,
        veh_obj.status.value,
        list(veh_obj.pos) if veh_obj.pos else None,
        veh_obj.soc,
        veh_obj.battery_size,
        veh_obj.range,
        len(veh_obj.pax),
        veh_obj.max_pax,
        [pax.get_rid_struct() for pax in veh_obj.pax],
        veh_obj.cl_start_time,
        list(veh_obj.cl_start_pos) if veh_obj.cl_start_pos else None,
        veh_obj.cl_start_soc,
        veh_obj.cl_driven_distance,
        veh_obj.cl_remaining_time,
        len(veh_obj.cl_remaining_route) if veh_obj.cl_remaining_route else 0,
        veh_obj.cl_remaining_route if veh_obj.cl_remaining_route else [],
        veh_obj.cl_locked,
        veh_obj.cumulative_distance,
        len(veh_obj.assigned_route),
        _encode_assigned_route(veh_obj),
        len(vehicle_plan.list_plan_stops),
        _encode_plan_stops(vehicle_plan, sim_time),
    ]

def _encode_assigned_route(veh_obj: SimulationVehicle):
    """Encode assigned_route as a list of value arrays (order matches LEG_COLUMNS)."""
    legs = []
    for leg in veh_obj.assigned_route:
        legs.append([
            leg.status.display_name,
            leg.status.value,
            list(leg.destination_pos) if leg.destination_pos else None,
            leg.duration,
            leg.power,
            leg.earliest_start_time,
            leg.earliest_end_time,
            leg.locked,
            leg.started,
            [rq.get_rid_struct() for rq in leg.rq_dict.get(1, [])],
            [rq.get_rid_struct() for rq in leg.rq_dict.get(-1, [])],
            len(leg.route) if leg.route else 0,
        ])
    return legs

def _encode_plan_stops(vehicle_plan: VehiclePlan, sim_time):
    """Encode plan_stops as a list of value arrays (order matches STOP_COLUMNS)."""
    stops = []
    for ps in vehicle_plan.list_plan_stops:
        arr_time, dep_time = ps.get_planned_arrival_and_departure_time()
        dur, earliest_dep = ps.get_duration_and_earliest_departure()
        stops.append([
            list(ps.get_pos()) if ps.get_pos() else None,
            ps.get_state().name if ps.get_state() else None,
            ps.get_list_boarding_rids(),
            ps.get_list_alighting_rids(),
            arr_time,
            dep_time,
            round(dep_time - sim_time, 1) if dep_time else None,
            dur,
            earliest_dep,
            ps.get_earliest_start_time(),
            ps.is_locked(),
            ps.get_change_nr_pax(),
            ps.get_charging_power(),
            ps.get_charging_task_id(),
        ])
    return stops
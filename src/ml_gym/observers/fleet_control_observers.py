from src.ml_gym.Observers import AbstractObserver
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.simulation.Vehicles import SimulationVehicle
from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from typing import List, Optional, Any, Dict


# ---------------------------------------------------------------------------
# Field getter tables
# Each getter signature:
#   VEH:  (enc, veh_obj, vehicle_plan, sim_time) -> value
#   LEG:  (leg,) -> value
#   STOP: (ps, sim_time) -> value
# ---------------------------------------------------------------------------

class FieldGetters:
    VEH: Dict[str, Any] = {
        "vid":                    lambda enc, v, p, t: v.vid,
        "op_id":                  lambda enc, v, p, t: v.op_id,
        "veh_type":               lambda enc, v, p, t: v.veh_type,
        "status":                 lambda enc, v, p, t: v.status.display_name,
        "status_value":           lambda enc, v, p, t: v.status.value,
        "pos":                    lambda enc, v, p, t: list(v.pos) if v.pos else None,
        "soc":                    lambda enc, v, p, t: v.soc,
        "battery_size":           lambda enc, v, p, t: v.battery_size,
        "range":                  lambda enc, v, p, t: v.range,
        "n_pax":                  lambda enc, v, p, t: len(v.pax),
        "max_pax":                lambda enc, v, p, t: v.max_pax,
        "pax_rids":               lambda enc, v, p, t: [pax.get_rid_struct() for pax in v.pax],
        "cl_start_time":          lambda enc, v, p, t: v.cl_start_time,
        "cl_start_pos":           lambda enc, v, p, t: list(v.cl_start_pos) if v.cl_start_pos else None,
        "cl_start_soc":           lambda enc, v, p, t: v.cl_start_soc,
        "cl_driven_distance":     lambda enc, v, p, t: v.cl_driven_distance,
        "cl_remaining_time":      lambda enc, v, p, t: v.cl_remaining_time,
        "cl_remaining_route_len": lambda enc, v, p, t: len(v.cl_remaining_route) if v.cl_remaining_route else 0,
        "cl_remaining_route":     lambda enc, v, p, t: v.cl_remaining_route if v.cl_remaining_route else [],
        "cl_locked":              lambda enc, v, p, t: v.cl_locked,
        "cumulative_distance":    lambda enc, v, p, t: v.cumulative_distance,
        "n_assigned_legs":        lambda enc, v, p, t: len(v.assigned_route),
        "assigned_route":         lambda enc, v, p, t: enc._encode_assigned_route(v),
        "n_plan_stops":           lambda enc, v, p, t: len(p.list_plan_stops),
        "plan_stops":             lambda enc, v, p, t: enc._encode_plan_stops(p, t),
    }

    LEG: Dict[str, Any] = {
        "status":              lambda leg: leg.status.display_name,
        "status_value":        lambda leg: leg.status.value,
        "destination_pos":     lambda leg: list(leg.destination_pos) if leg.destination_pos else None,
        "duration":            lambda leg: leg.duration,
        "power":               lambda leg: leg.power,
        "earliest_start_time": lambda leg: leg.earliest_start_time,
        "earliest_end_time":   lambda leg: leg.earliest_end_time,
        "locked":              lambda leg: leg.locked,
        "started":             lambda leg: leg.started,
        "boarding_rids":       lambda leg: [rq.get_rid_struct() for rq in leg.rq_dict.get(1, [])],
        "alighting_rids":      lambda leg: [rq.get_rid_struct() for rq in leg.rq_dict.get(-1, [])],
        "route_len":           lambda leg: len(leg.route) if leg.route else 0,
    }

    STOP: Dict[str, Any] = {
        "pos":                        lambda ps, t: list(ps.get_pos()) if ps.get_pos() else None,
        "state":                      lambda ps, t: ps.get_state().name if ps.get_state() else None,
        "boarding_rids":              lambda ps, t: ps.get_list_boarding_rids(),
        "alighting_rids":             lambda ps, t: ps.get_list_alighting_rids(),
        "planned_arrival_time":       lambda ps, t: ps.get_planned_arrival_and_departure_time()[0],
        "planned_departure_time":     lambda ps, t: ps.get_planned_arrival_and_departure_time()[1],
        "remaining_time_to_departure":lambda ps, t: round(ps.get_planned_arrival_and_departure_time()[1] - t, 1) if ps.get_planned_arrival_and_departure_time()[1] else None,
        "duration":                   lambda ps, t: ps.get_duration_and_earliest_departure()[0],
        "earliest_departure":         lambda ps, t: ps.get_duration_and_earliest_departure()[1],
        "earliest_start_time":        lambda ps, t: ps.get_earliest_start_time(),
        "locked":                     lambda ps, t: ps.is_locked(),
        "change_nr_pax":              lambda ps, t: ps.get_change_nr_pax(),
        "charging_power":             lambda ps, t: ps.get_charging_power(),
        "charging_task_id":           lambda ps, t: ps.get_charging_task_id(),
    }


# ---------------------------------------------------------------------------
# Detail presets
# ---------------------------------------------------------------------------

DETAIL_PRESETS: Dict[str, Dict[str, List[str]]] = {
    "mini": {
        "veh":  ["vid", "status", "pos", "soc", "n_pax", "max_pax"],
        "leg":  [],
        "stop": [],
    },
    "medium": {
        "veh":  ["vid", "op_id", "status", "status_value", "pos", "soc",
                 "n_pax", "max_pax", "pax_rids", "cl_driven_distance",
                 "cl_remaining_time", "n_assigned_legs"],
        "leg":  ["status", "destination_pos", "duration",
                 "boarding_rids", "alighting_rids", "route_len", "locked"],
        "stop": ["pos", "state", "boarding_rids", "alighting_rids",
                 "planned_arrival_time", "planned_departure_time", "locked"],
    },
    "max": {
        "veh":  list(FieldGetters.VEH.keys()),
        "leg":  list(FieldGetters.LEG.keys()),
        "stop": list(FieldGetters.STOP.keys()),
    },
}


class FleetStateObserver(AbstractObserver):

    def __init__(self, detail_level: str = "mini", custom_fields: Optional[Dict[str, List[str]]] = None,
                 recording_interval: int = 1, sim_start_time: int = 0, sim_time_step: int = 1):
        # 1. Resolve field lists
        if detail_level == "custom":
            if custom_fields is None:
                raise ValueError("FleetStateObserver: custom_fields must be provided when detail_level='custom'")
            for key in ("veh", "leg", "stop"):
                if key not in custom_fields:
                    raise ValueError(f"FleetStateObserver: custom_fields must contain key '{key}'")
            self._veh_fields = list(custom_fields.get("veh", []))
            self._leg_fields = list(custom_fields.get("leg", []))
            self._stop_fields = list(custom_fields.get("stop", []))
        elif detail_level in DETAIL_PRESETS:
            preset = DETAIL_PRESETS[detail_level]
            self._veh_fields = list(preset["veh"])
            self._leg_fields = list(preset["leg"])
            self._stop_fields = list(preset["stop"])
        else:
            raise ValueError(f"FleetStateObserver: Unknown detail_level '{detail_level}'. Choose from {list(DETAIL_PRESETS)} or 'custom'.")

        # Validate field names
        for f in self._veh_fields:
            if f not in FieldGetters.VEH:
                raise ValueError(f"FleetStateObserver: Unknown vehicle field: '{f}'")
        for f in self._leg_fields:
            if f not in FieldGetters.LEG:
                raise ValueError(f"FleetStateObserver: Unknown leg field: '{f}'")
        for f in self._stop_fields:
            if f not in FieldGetters.STOP:
                raise ValueError(f"FleetStateObserver: Unknown stop field: '{f}'")

        # 2. Auto-inject container fields if sub-fields are requested
        if self._leg_fields and "assigned_route" not in self._veh_fields:
            self._veh_fields.append("assigned_route")
        if self._leg_fields and "n_assigned_legs" not in self._veh_fields:
            self._veh_fields.append("n_assigned_legs")
        if self._stop_fields and "plan_stops" not in self._veh_fields:
            self._veh_fields.append("plan_stops")
        if self._stop_fields and "n_plan_stops" not in self._veh_fields:
            self._veh_fields.append("n_plan_stops")

        # 3. Pre-resolve field names → getter functions (no per-call dict lookups)
        self._veh_getters  = [FieldGetters.VEH[f]  for f in self._veh_fields]
        self._leg_getters  = [FieldGetters.LEG[f]  for f in self._leg_fields]
        self._stop_getters = [FieldGetters.STOP[f] for f in self._stop_fields]

        # 4. Recording interval in seconds
        self._sim_start_time = sim_start_time
        self._recording_interval_seconds = recording_interval * sim_time_step

    # Nested encoding helpers — called by FieldGetters.VEH["assigned_route"] / ["plan_stops"]
    def _encode_assigned_route(self, veh_obj: SimulationVehicle) -> List[List[Any]]:
        return [[g(leg) for g in self._leg_getters] for leg in veh_obj.assigned_route]

    def _encode_plan_stops(self, vehicle_plan: VehiclePlan, sim_time: int) -> List[List[Any]]:
        return [[g(ps, sim_time) for g in self._stop_getters] for ps in vehicle_plan.list_plan_stops]

    def observe(self, fleetpy_module) -> dict:
        assert isinstance(fleetpy_module, FleetControlBase), "FleetStateObserver only works with FleetControl Module"

        sim_time = fleetpy_module.sim_time

        # Interval gating
        if self._recording_interval_seconds > 0:
            if (sim_time - self._sim_start_time) % self._recording_interval_seconds != 0:
                return {}

        vehicle_rows = [
            [g(self, fleetpy_module.sim_vehicles[vid], fleetpy_module.veh_plans[vid], sim_time)
             for g in self._veh_getters]
            for vid in range(fleetpy_module.nr_vehicles)
        ]

        return {
            "time": int(sim_time),
            "op_id": fleetpy_module.op_id,
            "n_vehicles": fleetpy_module.nr_vehicles,
            "columns": self._veh_fields,
            "leg_columns": self._leg_fields,
            "stop_columns": self._stop_fields,
            "vehicles": vehicle_rows,
        }


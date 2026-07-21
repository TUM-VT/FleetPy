"""Encode a FleetControl snapshot as a configurable, table-oriented observation.

The observer deliberately keeps FleetPy's domain objects out of its public
output.  Positions are converted to lists, request objects to request-id
strings, and vehicle plans to nested rows.  This makes the result suitable
for JSONL writers while preserving enough schema information to reconstruct
each row.

``FleetStateObserver`` only collects domain state.  Any conversion to a fixed
Gymnasium observation space remains the responsibility of the Gym adapter.
"""

from src.ml_gym.Observers import AbstractObserver
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.misc.globals import VRL_STATES
from src.simulation.Vehicles import SimulationVehicle
from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from typing import Any, Dict, Iterable, List, Optional


_VEHICLE_ROUTE_LEG_EARLIEST_START_SENTINELS = {-1000, -1}
_VEHICLE_ROUTE_LEG_EARLIEST_END_SENTINELS = {-1000, -100000000}
_PLAN_STOP_EARLIEST_START_SENTINELS = {-1}


def _public_rid(value: Any) -> str:
    """Return the one public representation used for every request id.

    ``value`` may be a FleetPy request object, a structured request id, or a
    NumPy-like scalar.  Resolving request objects here prevents vehicle, leg,
    stop, and top-level ids from acquiring different JSON representations.
    """
    get_rid_struct = getattr(value, "get_rid_struct", None)
    rid = get_rid_struct() if callable(get_rid_struct) else value
    scalar_item = getattr(rid, "item", None)
    if callable(scalar_item):
        rid = scalar_item()
    return str(rid)


def _position_or_none(value: Any) -> Optional[List[Any]]:
    """Convert a FleetPy position to a public list without losing node zero."""
    if value is None:
        return None
    return list(value)


def _absolute_time_or_none(value: Any, sentinels: set) -> Any:
    """Map one field's known missing-time sentinels to public ``None``."""
    if value is None or value in sentinels:
        return None
    return value


def _effective_passengers_after_current_boarding(vehicle: Any) -> List[Any]:
    """Return the certain passenger set after the active BOARDING leg.

    FleetPy adds boarders to ``vehicle.pax`` when a BOARDING leg starts but
    removes alighters only when that indivisible leg ends.  During that leg,
    the public state therefore keeps the already-present boarders and removes
    the current leg's alighters.  Outside an active BOARDING leg, ``pax`` is
    already the current certain state.
    """
    raw_passengers = list(vehicle.pax)
    if (
        vehicle.status != VRL_STATES.BOARDING
        or vehicle.cl_start_time is None
        or not vehicle.assigned_route
    ):
        return raw_passengers

    alighting_rids = {
        _public_rid(request)
        for request in vehicle.assigned_route[0].rq_dict.get(-1, [])
    }
    return [
        request
        for request in raw_passengers
        if _public_rid(request) not in alighting_rids
    ]


def _effective_passenger_state(vehicle: Any) -> tuple:
    """Return post-BOARDING passenger count and public request ids together."""
    passengers = _effective_passengers_after_current_boarding(vehicle)
    return (
        sum(request.nr_pax for request in passengers),
        [_public_rid(request) for request in passengers],
    )


def _normalise_json_value(value: Any) -> Any:
    """Convert NumPy-like scalars and nested tuples to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    scalar_item = getattr(value, "item", None)
    if callable(scalar_item):
        return _normalise_json_value(scalar_item())

    if isinstance(value, (list, tuple)):
        return [_normalise_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalise_json_value(item) for key, item in value.items()}

    return value


def _remaining_time_to_departure(plan_stop: Any, sim_time: int) -> Optional[float]:
    """Return the relative departure time, including a valid zero timestamp."""
    departure_time = plan_stop.get_planned_arrival_and_departure_time()[1]
    if departure_time is None:
        return None
    return round(departure_time - sim_time, 1)


# ---------------------------------------------------------------------------
# Field getter tables.  Keeping selection and extraction separate allows an
# observer to resolve its configured fields once during construction.
#
# Getter signatures:
#   VEH:  (observer, simulation_vehicle, vehicle_plan, sim_time,
#          effective_passenger_state) -> value
#   LEG:  (observer, simulation_vehicle, vehicle_route_leg, leg_index) -> value
#   STOP: (plan_stop, sim_time) -> value
#
# Some VEH getters do not need every argument; they retain the common signature
# so all selected fields can be evaluated by the same loop in observe().
# ---------------------------------------------------------------------------

class FieldGetters:
    """Registry of supported output fields and their extraction functions.

    The dictionary insertion order defines the column order of the ``max``
    preset.  Field names are also the public names accepted by
    ``custom_fields``, so renaming one changes the observation contract.
    """

    VEH: Dict[str, Any] = {
        # Vehicle identity.
        "vid":                    lambda enc, v, p, t, pax: v.vid,
        "op_id":                  lambda enc, v, p, t, pax: v.op_id,
        "veh_type":               lambda enc, v, p, t, pax: v.veh_type,

        # Current vehicle state.  Enum fields expose both a readable label and
        # the underlying numeric value for human- and model-facing consumers.
        "status":                 lambda enc, v, p, t, pax: v.status.display_name,
        "status_value":           lambda enc, v, p, t, pax: v.status.value,
        "pos":                    lambda enc, v, p, t, pax: _position_or_none(v.pos),
        "n_pax":                  lambda enc, v, p, t, pax: pax[0],
        "max_pax":                lambda enc, v, p, t, pax: v.max_pax,
        "pax_rids":               lambda enc, v, p, t, pax: pax[1],

        # ``cl`` fields describe the active/current leg, not the complete
        # assigned route.  Route nodes remain in FleetPy's native order.
        "cl_start_time":          lambda enc, v, p, t, pax: v.cl_start_time,
        "cl_start_pos":           lambda enc, v, p, t, pax: _position_or_none(v.cl_start_pos),
        "cl_driven_distance":     lambda enc, v, p, t, pax: v.cl_driven_distance,
        "cl_remaining_time":      lambda enc, v, p, t, pax: v.cl_remaining_time,
        "cl_remaining_route_len": lambda enc, v, p, t, pax: len(v.cl_remaining_route) if v.cl_remaining_route else 0,
        "cl_remaining_route":     lambda enc, v, p, t, pax: v.cl_remaining_route if v.cl_remaining_route else [],
        # ``assigned_route[0]`` is the lock source used by vehicle-plan
        # execution.  Only expose it after that leg has actually started;
        # an idle vehicle may already hold a future assigned route.
        "cl_locked":              lambda enc, v, p, t, pax: bool(v.assigned_route[0].locked)
                                                            if v.cl_start_time is not None and v.assigned_route
                                                            else False,
        # Include the distance already travelled on the active leg so this is
        # the vehicle's total distance at the observation timestamp.
        "cumulative_distance":    lambda enc, v, p, t, pax: v.cumulative_distance + v.cl_driven_distance,

        # Route and plan-stop containers are nested row arrays.  Their schemas
        # are published separately as leg_columns and stop_columns.
        "n_assigned_legs":        lambda enc, v, p, t, pax: len(v.assigned_route),
        "assigned_route":         lambda enc, v, p, t, pax: enc._encode_assigned_route(v),
        "n_plan_stops":           lambda enc, v, p, t, pax: len(p.list_plan_stops),
        "plan_stops":             lambda enc, v, p, t, pax: enc._encode_plan_stops(p, t),
    }

    LEG: Dict[str, Any] = {
        # A VehicleRouteLeg is the executable route representation assigned to
        # a SimulationVehicle.  Request objects are exported as stable ids.
        "status":              lambda enc, v, leg, i: leg.status.display_name,
        "status_value":        lambda enc, v, leg, i: leg.status.value,
        "destination_pos":     lambda enc, v, leg, i: _position_or_none(leg.destination_pos),
        "duration":            lambda enc, v, leg, i: leg.duration,
        "earliest_start_time": lambda enc, v, leg, i: _absolute_time_or_none(
            leg.earliest_start_time, _VEHICLE_ROUTE_LEG_EARLIEST_START_SENTINELS
        ),
        "earliest_end_time":   lambda enc, v, leg, i: _absolute_time_or_none(
            leg.earliest_end_time, _VEHICLE_ROUTE_LEG_EARLIEST_END_SENTINELS
        ),
        "locked":              lambda enc, v, leg, i: leg.locked,
        # The raw leg flag can be lost when FleetPy rebuilds an equivalent
        # route.  The current-leg fields are the reliable execution source.
        "started":             lambda enc, v, leg, i: i == 0 and v.cl_start_time is not None,
        "boarding_rids":       lambda enc, v, leg, i: [_public_rid(rq) for rq in leg.rq_dict.get(1, [])],
        "alighting_rids":      lambda enc, v, leg, i: [_public_rid(rq) for rq in leg.rq_dict.get(-1, [])],
    }

    STOP: Dict[str, Any] = {
        # A PlanStop is the planning-layer representation.  Planned times are
        # absolute simulation times; only the remaining-time field is relative
        # to the sim_time passed to observe().
        "pos":                        lambda ps, t: _position_or_none(ps.get_pos()),
        "state":                      lambda ps, t: ps.get_state().name if ps.get_state() else None,
        "boarding_rids":              lambda ps, t: [_public_rid(rid) for rid in ps.get_list_boarding_rids()],
        "alighting_rids":             lambda ps, t: [_public_rid(rid) for rid in ps.get_list_alighting_rids()],
        "planned_arrival_time":       lambda ps, t: ps.get_planned_arrival_and_departure_time()[0],
        "planned_departure_time":     lambda ps, t: ps.get_planned_arrival_and_departure_time()[1],
        "remaining_time_to_departure":lambda ps, t: _remaining_time_to_departure(ps, t),
        "duration":                   lambda ps, t: ps.get_duration_and_earliest_departure()[0],
        "earliest_departure":         lambda ps, t: ps.get_duration_and_earliest_departure()[1],
        "earliest_start_time":        lambda ps, t: _absolute_time_or_none(
            ps.get_earliest_start_time(), _PLAN_STOP_EARLIEST_START_SENTINELS
        ),
        "locked":                     lambda ps, t: ps.is_locked(),
        "change_nr_pax":              lambda ps, t: ps.get_change_nr_pax(),
    }


# ---------------------------------------------------------------------------
# Detail presets.  Lists are copied into each observer instance so automatic
# container injection never mutates these module-level definitions.
# ---------------------------------------------------------------------------

DETAIL_PRESETS: Dict[str, Dict[str, List[str]]] = {
    "mini": {
        "veh":  ["vid", "status", "pos", "n_pax", "max_pax"],
        "leg":  [],
        "stop": [],
    },
    "medium": {
        "veh":  ["vid", "op_id", "status", "status_value", "pos",
                 "n_pax", "max_pax", "pax_rids", "cl_driven_distance",
                 "cl_remaining_time", "n_assigned_legs"],
        "leg":  ["status", "destination_pos", "duration",
                 "boarding_rids", "alighting_rids", "locked", "started"],
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
    """Collect a configurable snapshot of every vehicle of one operator.

    Vehicle, route-leg, and plan-stop values use a row-oriented representation:
    ``vehicles[i][j]`` corresponds to ``columns[j]``.  Likewise, nested values
    in ``assigned_route`` and ``plan_stops`` correspond to ``leg_columns`` and
    ``stop_columns`` respectively.  If nested fields are requested, their
    container and count columns are appended automatically.

    ``detail_level`` may be ``mini``, ``medium``, ``max``, or ``custom``.  The
    custom form requires all three keys (``veh``, ``leg``, and ``stop``), even
    when one of their field lists is empty.  This makes omission explicit and
    prevents a misspelled section name from silently dropping data.
    """

    def __init__(self, detail_level: str = "medium",
                 custom_fields: Optional[Dict[str, List[str]]] = None):
        """Resolve and validate the schema used by subsequent observations.

        :param detail_level: Name of a predefined schema, or ``custom``.
        :param custom_fields: Field lists keyed by ``veh``, ``leg``, and
            ``stop``; used only for the custom detail level.
        :raises ValueError: If the detail level, custom sections, or field names
            are unknown.
        """

        # 1. Resolve field lists.  Copy caller-owned and preset lists because
        # the dependency injection below may append container fields.
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

        # 2. Validate field names
        for f in self._veh_fields:
            if f not in FieldGetters.VEH:
                raise ValueError(f"FleetStateObserver: Unknown vehicle field: '{f}'")
        for f in self._leg_fields:
            if f not in FieldGetters.LEG:
                raise ValueError(f"FleetStateObserver: Unknown leg field: '{f}'")
        for f in self._stop_fields:
            if f not in FieldGetters.STOP:
                raise ValueError(f"FleetStateObserver: Unknown stop field: '{f}'")

        # 3. Nested rows must be reachable from the vehicle row.  Also include
        # their counts so consumers can inspect cardinality without decoding
        # the complete nested container.
        if self._leg_fields and "assigned_route" not in self._veh_fields:
            self._veh_fields.append("assigned_route")
        if self._leg_fields and "n_assigned_legs" not in self._veh_fields:
            self._veh_fields.append("n_assigned_legs")
        if self._stop_fields and "plan_stops" not in self._veh_fields:
            self._veh_fields.append("plan_stops")
        if self._stop_fields and "n_plan_stops" not in self._veh_fields:
            self._veh_fields.append("n_plan_stops")

        # 4. Pre-resolve field names → getter functions.  Besides avoiding
        # repeated registry lookups, this fixes getter order to column order.
        self._resolve_getters()

    def _resolve_getters(self) -> None:
        """Build callable caches from the serializable field-name schema."""
        self._veh_getters  = [FieldGetters.VEH[f]  for f in self._veh_fields]
        self._leg_getters  = [FieldGetters.LEG[f]  for f in self._leg_fields]
        self._stop_getters = [FieldGetters.STOP[f] for f in self._stop_fields]

    def __getstate__(self) -> dict:
        """Exclude lambda getter caches when sending the Observer to a worker."""
        state = self.__dict__.copy()
        state.pop("_veh_getters", None)
        state.pop("_leg_getters", None)
        state.pop("_stop_getters", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore a worker-local getter cache after multiprocessing unpickling."""
        self.__dict__.update(state)
        self._resolve_getters()

    def _encode_assigned_route(self, veh_obj: SimulationVehicle) -> List[List[Any]]:
        """Encode assigned route legs as rows aligned with ``_leg_fields``."""
        return [
            [
                _normalise_json_value(getter(self, veh_obj, leg, leg_index))
                for getter in self._leg_getters
            ]
            for leg_index, leg in enumerate(veh_obj.assigned_route)
        ]

    def _encode_plan_stops(self, vehicle_plan: VehiclePlan, sim_time: int) -> List[List[Any]]:
        """Encode plan stops as rows aligned with ``_stop_fields``."""
        return [
            [_normalise_json_value(getter(ps, sim_time)) for getter in self._stop_getters]
            for ps in vehicle_plan.list_plan_stops
        ]

    def observe(self, fleetpy_module: FleetControlBase, *, sim_time: int,
                new_request_ids: Iterable[Any],
                optimization_request_ids: Iterable[Any],
                prediction_request_ids: Iterable[Any]) -> dict:
        """Return the fleet snapshot and its request timing contexts.

        :param fleetpy_module: Fleet controller whose vehicles and plans are
            sampled.  Vehicle ids are expected to be contiguous from zero to
            ``nr_vehicles - 1``, matching FleetControl's storage contract.
        :param sim_time: Time of the triggering simulation event.  It is
            keyword-only to prevent accidentally recording a stale controller
            timestamp.
        :param new_request_ids: Requests that entered the control flow at this
            simulation step.  An empty iterable is valid.
        :param optimization_request_ids: Requests whose assignment remains
            mutable in the insertion or batch optimisation at this step.  An
            empty iterable is valid.
        :param prediction_request_ids: Requests for which this pre-decision
            snapshot is the service-prediction input.  In Batch mode these are
            requests awaiting an offer in the optimisation that follows; in
            Immediate mode this is the current request.  An empty iterable is
            valid. Immediate mode supplies the same current id in all three
            request-id iterables.
        :return: Event metadata plus a ``fleet_state`` table.  The table's
            ``columns`` arrays define the order of all vehicle and nested rows.
        """

        # Compute each vehicle's post-BOARDING passenger state at most once in
        # this snapshot.  Both n_pax and pax_rids consume the same immutable
        # tuple, so the paired fields cannot diverge between two traversals.
        vehicle_ids = range(fleetpy_module.nr_vehicles)
        needs_passenger_state = bool(
            {"n_pax", "pax_rids"}.intersection(self._veh_fields)
        )
        passenger_states = (
            {
                vid: _effective_passenger_state(fleetpy_module.sim_vehicles[vid])
                for vid in vehicle_ids
            }
            if needs_passenger_state
            else {}
        )

        # Build one row per vehicle. The outer comprehension iterates over all
        # vehicle ids; the inner comprehension applies the pre-resolved field
        # getters in the same order as self._veh_fields.
        vehicle_rows = [
            [
                _normalise_json_value(
                    getter(
                        self,
                        fleetpy_module.sim_vehicles[vid],
                        fleetpy_module.veh_plans[vid],
                        sim_time,
                        passenger_states.get(vid),
                    )
                )
                for getter in self._veh_getters
            ]
            for vid in vehicle_ids
        ]

        fleet_state = {
            "time": int(sim_time),
            "op_id": fleetpy_module.op_id,
            "n_vehicles": fleetpy_module.nr_vehicles,
            "columns": self._veh_fields,
            "leg_columns": self._leg_fields,
            "stop_columns": self._stop_fields,
            "vehicles": vehicle_rows,
        }

        return {
            "op_id": fleetpy_module.op_id,
            "sim_time": int(sim_time),
            "new_request_ids": [_public_rid(rid) for rid in new_request_ids],
            "optimization_request_ids": [
                _public_rid(rid) for rid in optimization_request_ids
            ],
            "prediction_request_ids": [
                _public_rid(rid) for rid in prediction_request_ids
            ],
            "fleet_state": fleet_state,
        }

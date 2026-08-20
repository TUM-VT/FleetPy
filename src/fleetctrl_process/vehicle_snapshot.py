from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

from src.fleetctrl.RemoteVehicleStatusMixin import RemoteVRL
from src.misc.globals import VRL_STATES

if TYPE_CHECKING:
    from src.simulation.Vehicles import SimulationVehicle

""" Lightweight, picklable stand-in for SimulationVehicle, sent from the main process into
an operator's worker subprocess once per timestep (full refresh, no incremental sync -> no
drift). Field set mirrors what src.fleetctrl.pooling.batch.BatchAssignmentAlgorithmBase's
SimulationVehicleStruct already copies for the same "give fleetctrl a vehicle-like object"
purpose - reusing that class directly would additionally pull in VehiclePlan/locked_planstops
construction (needs a routing_engine reference and is specific to the AlonsoMora batch
matcher), which isn't needed for the core FleetControlBase methods (receive_status_update,
time_trigger, compute_VehiclePlan_utility) this is built for.

requests (pax / boarding / alighting) cross the boundary as bare rids, never as objects -
FleetControlBase.rq_dict on the worker side is the single source of truth for turning a rid
into a request-like object (same translation RemoteVehicleStatusMixin already relies on).
"""


@dataclass
class VRLSnapshot:
    status: VRL_STATES
    destination_pos: tuple
    boarding_rids: List[Any]
    alighting_rids: List[Any]
    power: float = 0
    duration: Any = None
    route: List[int] = field(default_factory=list)
    locked: bool = False
    earliest_start_time: float = 0


@dataclass
class VehicleSnapshot:
    op_id: int
    vid: int
    status: VRL_STATES
    pos: tuple
    soc: float
    veh_type: str
    max_pax: int
    max_parcels: int
    daily_fix_cost: float
    distance_cost: float
    battery_size: float
    range: float
    soc_per_m: float
    cl_start_time: float
    pax_rids: List[Any]
    assigned_route: List[VRLSnapshot]


def vrl_to_snapshot(vrl) -> VRLSnapshot:
    """ real VehicleRouteLeg (rq_dict holding real RequestBase objects) -> picklable
    VRLSnapshot (rq_dict entries replaced by bare rids). used both for a vehicle's
    assigned_route (build_vehicle_snapshot) and for the passed_VRL list
    FleetControlProcessProxy.receive_status_update() forwards from Broker. """
    return VRLSnapshot(
        status=vrl.status,
        destination_pos=vrl.destination_pos,
        boarding_rids=[rq.get_rid_struct() for rq in vrl.rq_dict.get(1, [])],
        alighting_rids=[rq.get_rid_struct() for rq in vrl.rq_dict.get(-1, [])],
        power=vrl.power,
        duration=vrl.duration,
        route=vrl.route,
        locked=vrl.locked,
        earliest_start_time=vrl.earliest_start_time,
    )


def build_vehicle_snapshot(veh_obj: "SimulationVehicle") -> VehicleSnapshot:
    """ main-process side: real SimulationVehicle -> picklable VehicleSnapshot """
    assigned_route = [vrl_to_snapshot(vrl) for vrl in veh_obj.assigned_route]
    return VehicleSnapshot(
        op_id=veh_obj.op_id,
        vid=veh_obj.vid,
        status=veh_obj.status,
        pos=veh_obj.pos,
        soc=veh_obj.soc,
        veh_type=veh_obj.veh_type,
        max_pax=veh_obj.max_pax,
        max_parcels=veh_obj.max_parcels,
        daily_fix_cost=veh_obj.daily_fix_cost,
        distance_cost=veh_obj.distance_cost,
        battery_size=veh_obj.battery_size,
        range=veh_obj.range,
        soc_per_m=veh_obj.soc_per_m,
        cl_start_time=veh_obj.cl_start_time,
        pax_rids=[rq.get_rid_struct() for rq in veh_obj.pax],
        assigned_route=assigned_route,
    )


class LocalVehicleMirror:
    """ worker-process side stand-in used as FleetControlBase.sim_vehicles[vid]. resolved from
    a VehicleSnapshot against the operator's own rq_dict (PlanRequest instances), never against
    real RequestBase objects - the worker doesn't have (and doesn't need) those. """

    def __init__(self, snapshot: VehicleSnapshot, rq_dict: Dict[Any, Any]):
        self.op_id = snapshot.op_id
        self.vid = snapshot.vid
        self.status = snapshot.status
        self.pos = snapshot.pos
        self.soc = snapshot.soc
        self.veh_type = snapshot.veh_type
        self.max_pax = snapshot.max_pax
        self.max_parcels = snapshot.max_parcels
        self.daily_fix_cost = snapshot.daily_fix_cost
        self.distance_cost = snapshot.distance_cost
        self.battery_size = snapshot.battery_size
        self.range = snapshot.range
        self.soc_per_m = snapshot.soc_per_m
        self.cl_start_time = snapshot.cl_start_time
        self.pax = [rq_dict[rid] for rid in snapshot.pax_rids]
        self.assigned_route = [
            RemoteVRL(
                status=vrl.status,
                destination_pos=vrl.destination_pos,
                rq_dict={1: [rq_dict[rid] for rid in vrl.boarding_rids],
                         -1: [rq_dict[rid] for rid in vrl.alighting_rids]},
                power=vrl.power,
                duration=vrl.duration,
                route=vrl.route,
                locked=vrl.locked,
                earliest_start_time=vrl.earliest_start_time,
            )
            for vrl in snapshot.assigned_route
        ]

    def __str__(self):
        return f"veh mirror {self.vid} at pos {self.pos} status {self.status} ob {self.pax}"

    def compute_soc_consumption(self, distance: float) -> float:
        return distance * self.soc_per_m

    def compute_soc_charging(self, power: float, duration: float) -> float:
        return power * duration / self.battery_size

    def get_nr_pax_without_currently_boarding(self) -> int:
        if self.status == VRL_STATES.BOARDING:
            return (sum(rq.nr_pax for rq in self.pax if not getattr(rq, "is_parcel", False))
                    - sum(rq.nr_pax for rq in self.assigned_route[0].rq_dict.get(1, []) if not getattr(rq, "is_parcel", False)))
        return sum(rq.nr_pax for rq in self.pax if not getattr(rq, "is_parcel", False))

    def get_nr_parcels_without_currently_boarding(self) -> int:
        if self.status == VRL_STATES.BOARDING:
            return (sum(rq.nr_pax for rq in self.pax if getattr(rq, "is_parcel", False))
                    - sum(rq.nr_pax for rq in self.assigned_route[0].rq_dict.get(1, []) if getattr(rq, "is_parcel", False)))
        return sum(rq.nr_pax for rq in self.pax if getattr(rq, "is_parcel", False))

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Tuple

from src.misc.globals import VRL_STATES

LARGE_INT = 100000000


@dataclass
class RemoteVRL:
    """ Minimal duck-typed stand-in for src.simulation.Legs.VehicleRouteLeg.

    Exposes exactly the attributes fleetctrl's read side actually touches
    (VehiclePlan.update_plan(), FleetControlBase.receive_status_update(),
    AlonsoMoraAssignment._setUpCurrentInformation()): status, destination_pos,
    rq_dict, power, duration, route, locked, earliest_start_time,
    earliest_end_time, stationary_process. A real VehicleRouteLeg carries much
    more (driven distance, replay data, ...) that fleetctrl never reads, so it
    isn't reproduced here.

    rq_dict values only need a get_rid_struct() method - fleetctrl's own
    PlanRequest objects satisfy that already, so a status report doesn't need
    real src.demand.TravelerModels.RequestBase objects.
    """
    status: VRL_STATES
    destination_pos: tuple
    rq_dict: dict  # {1: [boarding plan_requests], -1: [alighting plan_requests]}
    power: float = 0
    duration: Any = None
    route: List[int] = field(default_factory=list)
    locked: bool = False
    earliest_start_time: float = -LARGE_INT
    earliest_end_time: float = LARGE_INT
    stationary_process: Any = None


class RemoteVehicleStatusMixin:
    """ Counterpart to RemoteVehicleDispatchMixin: turns a status report coming from a
    real/remote vehicle into the call fleetctrl already expects (receive_status_update),
    instead of requiring the caller to construct real VehicleRouteLeg objects - which
    would pull in the simulation's Legs/Vehicles/RequestBase machinery just to report
    that a vehicle reached a stop.

    NOTE - this only replaces the *payload* (list_finished_VRL). receive_status_update()
    still reads self.sim_vehicles[vid] (assigned_route, pos, pax, soc, ...) internally
    via VehiclePlan.update_plan()/update_tt_and_check_plan(). For a real remote backend
    that vehicle-state object still needs to be kept up to date separately - that's the
    broader "read side" problem, not something this mixin solves.
    """

    def report_remote_leg_finished(self, vid: int, sim_time: int, finished_pos: tuple,
                                    boarding_rids: Iterable[Any] = (), alighting_rids: Iterable[Any] = (),
                                    locked: bool = False, force_update: bool = True):
        boarding = [self.rq_dict[rid] for rid in boarding_rids]
        alighting = [self.rq_dict[rid] for rid in alighting_rids]
        status = VRL_STATES.BOARDING if (boarding or alighting) else VRL_STATES.IDLE
        finished_leg = RemoteVRL(
            status=status,
            destination_pos=finished_pos,
            rq_dict={1: boarding, -1: alighting},
            locked=locked,
        )
        self.receive_status_update(vid, sim_time, [finished_leg], force_update=force_update)

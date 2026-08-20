from __future__ import annotations
import logging
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation.Vehicles import SimulationVehicle
    from src.fleetctrl.planning.VehiclePlan import VehiclePlan

LOG = logging.getLogger(__name__)


class RemoteFleetBackend:
    """ minimal interface a real vehicle/dispatch backend has to implement to be usable
    with RemoteVehicleDispatchMixin. reference implementation here only logs the schedule
    instead of talking to an actual vehicle. """

    def send_schedule(self, vid: int, sim_time: int, stops: List[Dict[str, Any]], force: bool = False):
        LOG.info(f"[RemoteFleetBackend] vid {vid} at {sim_time} (force={force}) new schedule:")
        for stop in stops:
            LOG.info(f"    {stop}")


class QueueFleetBackend(RemoteFleetBackend):
    """ RemoteFleetBackend implementation used inside an operator worker subprocess
    (see src/fleetctrl_process/worker.py): instead of mutating a local SimulationVehicle
    (there isn't one in the subprocess) or just logging, it puts the schedule on a
    multiprocessing.Queue that the main process drains once per timestep and applies to
    the real SimulationVehicle objects it owns. """

    def __init__(self, dispatch_queue, op_id: int):
        self.dispatch_queue = dispatch_queue
        self.op_id = op_id

    def send_schedule(self, vid: int, sim_time: int, stops: List[Dict[str, Any]], force: bool = False):
        self.dispatch_queue.put((self.op_id, vid, sim_time, stops, force))


class RemoteVehicleDispatchMixin:
    """ Mixin that redirects the single seam between fleetctrl's internal planning
    representation (VehiclePlan/PlanStop) and vehicle-side execution
    (VehicleRouteLeg/SimulationVehicle) to an external/remote backend instead of
    mutating a local SimulationVehicle directly.

    Combine with any concrete FleetControlBase subclass, e.g.:

        class RemoteRidePoolingControl(RemoteVehicleDispatchMixin, RidePoolingBatchAssignmentFleetcontrol):
            pass

    all other bookkeeping in assign_vehicle_plan() (charging tasks, rq_dict/veh_plans
    updates, assignment records) stays inherited unchanged; only the vehicle push is
    replaced.

    self.remote_backend can be set after construction (defaults to RemoteFleetBackend(),
    which just logs) or overridden in a subclass __init__.
    """

    remote_backend: RemoteFleetBackend = RemoteFleetBackend()

    def _dispatch_vehicle_plan(self, veh_obj: SimulationVehicle, vehicle_plan: VehiclePlan, sim_time: int,
                               force_assign: bool = False):
        # reuse the existing, already-correct PlanStop -> VehicleRouteLeg translation
        # (driving legs between stops, VRL_STATES classification for
        # charging/reservation/repositioning stops, buffer-time waiting, ...) instead of
        # re-deriving that logic here. only the *output* of _build_VRLs crosses the
        # boundary, as plain dicts with rid-only request references.
        new_list_vrls = self._build_VRLs(vehicle_plan, veh_obj, sim_time)
        stops = [
            {
                "status": vrl.status,
                "destination_pos": vrl.destination_pos,
                "boarding_rids": [rq.get_rid_struct() for rq in vrl.rq_dict.get(1, [])],
                "alighting_rids": [rq.get_rid_struct() for rq in vrl.rq_dict.get(-1, [])],
                "power": vrl.power,
                "duration": vrl.duration,
                "route": vrl.route,
                "locked": vrl.locked,
                "earliest_start_time": vrl.earliest_start_time,
            }
            for vrl in new_list_vrls
        ]
        self.remote_backend.send_schedule(veh_obj.vid, sim_time, stops, force=force_assign)

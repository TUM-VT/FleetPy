from __future__ import annotations
import itertools
import logging
import queue as queue_module
from typing import Any, Dict, List, Tuple

from src.fleetctrl_process.comcodes import COMCODE
from src.fleetctrl_process.vehicle_snapshot import build_vehicle_snapshot, vrl_to_snapshot

LOG = logging.getLogger(__name__)


class RemoteWorkerError(RuntimeError):
    """ raised in the main process when a worker subprocess reports COMCODE.ERROR for a
    call made via call_and_wait(). calls made via dispatch()/the domain methods below are
    fire-and-forget - their errors are only logged (via poll()), never raised, since no
    caller is left waiting to receive an exception. """


class FleetControlProcessProxy:
    """ Main-process stand-in for a FleetControlBase operator instance living in a worker
    subprocess (src/fleetctrl_process/worker.py).

    Every domain call here is fire-and-forget: dispatch() puts a message on the request
    queue and returns a call_id immediately, without waiting for the worker to process it.
    This is deliberate - the point of running fleet control in a subprocess is to be able
    to measure/tolerate its compute latency, which requires the simulation clock to keep
    advancing independently of when (or whether, within a given timestep) the worker
    responds. A caller that cares about a specific result (right now: only per-request
    offers) tracks its own call_ids and looks them up against poll()'s output; nothing
    here blocks waiting for a specific answer.

    call_and_wait() is kept as a narrow exception for one-off lifecycle calls that happen
    outside step() entirely (the initial vehicle-sync handshake at worker startup,
    shutdown) - never call it from inside a simulation step.
    """

    def __init__(self, op_id: int, process, req_queue, resp_queue, dispatch_queue, sim_vehicles: Dict):
        self.op_id = op_id
        self.process = process
        self.req_queue = req_queue
        self.resp_queue = resp_queue
        self.dispatch_queue = dispatch_queue
        # reference to FleetSimulationBase.sim_vehicles ((op_id, vid) -> SimulationVehicle),
        # kept so receive_status_update() can attach a fresh snapshot of the vehicle it's
        # reporting on without Broker/update_sim_state_fleets having to change at all -
        # receive_status_update is already called unconditionally once per vehicle per
        # timestep, so this is the natural (and only needed) sync point.
        self.sim_vehicles = sim_vehicles
        self._call_ids = itertools.count()

    # -- low-level: fire-and-forget dispatch, non-blocking poll -------------------------

    def dispatch(self, code: COMCODE, *args) -> int:
        call_id = next(self._call_ids)
        self.req_queue.put((call_id, code, args))
        return call_id

    def poll(self) -> List[Tuple[int, COMCODE, Any]]:
        """ drains every response the worker has produced so far, without blocking.
        COMCODE.ERROR entries are logged here and dropped, not returned - see
        RemoteWorkerError docstring for why. call this once per step (typically at the
        top, before doing anything else) so responses that arrived while the main process
        was busy with other operators/vehicle movement get picked up promptly. """
        results = []
        while True:
            try:
                call_id, code, payload = self.resp_queue.get_nowait()
            except queue_module.Empty:
                break
            if code == COMCODE.ERROR:
                err_repr, tb = payload
                LOG.error(f"op_id {self.op_id} worker raised {err_repr} for call {call_id}\n{tb}")
                continue
            results.append((call_id, code, payload))
        return results

    def call_and_wait(self, code: COMCODE, *args, timeout: float = None) -> Any:
        """ blocking request/response - lifecycle use only (see class docstring), never
        from inside step(). """
        call_id = self.dispatch(code, *args)
        while True:
            got_id, resp_code, payload = self.resp_queue.get(timeout=timeout)
            if got_id != call_id:
                # lifecycle calls happen strictly one at a time before/after step() runs,
                # so an unrelated response here would indicate a real ordering bug.
                raise RemoteWorkerError(f"op_id {self.op_id}: call_and_wait for call {call_id} "
                                        f"got unrelated response for call {got_id}")
            if resp_code == COMCODE.ERROR:
                err_repr, tb = payload
                raise RemoteWorkerError(f"op_id {self.op_id} worker raised {err_repr}\n{tb}")
            return payload

    # -- fire-and-forget domain calls -----------------------------------------------------
    # every method returns the call_id it dispatched with; callers that don't need the
    # result (everything except get_offer) can ignore it.

    def user_request(self, rq_obj, sim_time) -> int:
        return self.dispatch(COMCODE.USER_REQUEST, rq_obj, sim_time)

    def get_offer(self, rid) -> int:
        return self.dispatch(COMCODE.GET_OFFER, rid)

    def user_confirms_booking(self, rid, sim_time) -> int:
        return self.dispatch(COMCODE.CONFIRM_BOOKING, rid, sim_time)

    def user_cancels_request(self, rid, sim_time) -> int:
        return self.dispatch(COMCODE.CANCEL_REQUEST, rid, sim_time)

    def acknowledge_boarding(self, rid, vid, boarding_time) -> int:
        return self.dispatch(COMCODE.ACK_BOARDING, rid, vid, boarding_time)

    def acknowledge_alighting(self, rid, vid, alighting_time) -> int:
        return self.dispatch(COMCODE.ACK_ALIGHTING, rid, vid, alighting_time)

    def receive_status_update(self, vid, sim_time, passed_VRL, force_update_plan) -> int:
        # passed_VRL here are real VehicleRouteLeg objects (with real RequestBase in
        # rq_dict) - Broker/update_sim_state_fleets pass these unchanged. translate to
        # picklable VRLSnapshots and bundle a fresh snapshot of the reporting vehicle
        # itself, so the worker's LocalVehicleMirror for this vid is never stale. this is
        # local, in-process work (cheap) - only the dispatch() at the end touches IPC.
        veh_obj = self.sim_vehicles[(self.op_id, vid)]
        veh_snapshot = build_vehicle_snapshot(veh_obj)
        vrl_snapshots = [vrl_to_snapshot(vrl) for vrl in passed_VRL]
        return self.dispatch(COMCODE.RECEIVE_STATUS_UPDATE, vid, sim_time, vrl_snapshots, force_update_plan, veh_snapshot)

    def inform_network_travel_time_update(self, sim_time) -> int:
        return self.dispatch(COMCODE.INFORM_TT_UPDATE, sim_time)

    def time_trigger(self, sim_time) -> int:
        return self.dispatch(COMCODE.TIME_TRIGGER, sim_time)

    def record_dynamic_fleetcontrol_output(self, force=False) -> int:
        return self.dispatch(COMCODE.RECORD_STATS, force)

    # -- lifecycle (blocking) -------------------------------------------------------------

    def add_init(self, operator_attributes, scenario_parameters):
        return self.call_and_wait(COMCODE.ADD_INIT, operator_attributes, scenario_parameters)

    def sync_vehicles(self, snapshots: List):
        return self.call_and_wait(COMCODE.SYNC_VEHICLES, snapshots)

    def drain_dispatch_queue(self):
        """ returns all (vid, sim_time, stops, force) schedule messages the worker has
        pushed since the last drain (see QueueFleetBackend.send_schedule). the caller
        (AsyncFleetSimulationBase / a step() override) applies these to the real
        SimulationVehicle objects it owns. already non-blocking. """
        messages = []
        while not self.dispatch_queue.empty():
            op_id, vid, sim_time, stops, force = self.dispatch_queue.get()
            messages.append((vid, sim_time, stops, force))
        return messages

    def shutdown(self):
        self.call_and_wait(COMCODE.KILL)
        self.process.join(timeout=10)
        if self.process.is_alive():
            LOG.warning(f"op_id {self.op_id} worker did not shut down cleanly, terminating")
            self.process.terminate()

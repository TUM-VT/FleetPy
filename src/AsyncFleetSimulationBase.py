from __future__ import annotations
import logging
from multiprocessing import Process, Queue

from src.fleetctrl_process.worker import run_fleetcontrol_worker
from src.fleetctrl_process.proxy import FleetControlProcessProxy
from src.fleetctrl_process.vehicle_snapshot import build_vehicle_snapshot

LOG = logging.getLogger(__name__)

""" Mixin that moves every FleetControlBase operator instance built by
FleetSimulationBase._load_fleetctr_vehicles() into its own worker subprocess
(src/fleetctrl_process/worker.py), replacing self.operators[op_id] with a
FleetControlProcessProxy (src/fleetctrl_process/proxy.py). Broker/step() keep calling
the exact same method names on self.operators[op_id] as before - construction and
everything else in FleetSimulationBase.__init__ runs completely unchanged.

WINDOWS NOTE: multiprocessing uses "spawn" here (no fork), so whatever top-level script
constructs a simulation class combining this mixin MUST do so under
`if __name__ == "__main__":` - the same requirement the codebase's own
src/run_scenarios.py already follows for its scenario-level multiprocessing.

Usage: combine with any concrete FleetSimulationBase subclass and override step() to
actually drive the operators - see AsyncNonBlockingSimulation.py for the reference
pattern. Every FleetControlProcessProxy call is fire-and-forget (see proxy.py): step()
must not wait for a worker's response before advancing the simulation clock, since the
whole point of this architecture is to let fleet-control compute time run independently
of (and potentially behind) the simulated vehicles/demand - see
AsyncNonBlockingSimulation.py for how a step() built on that assumption looks.
"""


class AsyncFleetSimulationBase:
    def __init__(self, scenario_parameters: dict):
        super().__init__(scenario_parameters)
        self._operator_processes_started = False
        if not self._started:
            self._start_operator_processes()

    def _start_operator_processes(self):
        LOG.info("AsyncFleetSimulationBase: moving operators into worker subprocesses...")
        new_operators = []
        for op_id, in_process_operator in enumerate(self.operators):
            operator_attributes = self.list_op_dicts[op_id]
            op_dir_names = self.dir_names.copy()
            op_dir_names.update(self.dir_names.get(f"op_{op_id}", {}))

            req_queue = Queue()
            resp_queue = Queue()
            dispatch_queue = Queue()
            process = Process(target=run_fleetcontrol_worker,
                               args=(op_id, operator_attributes, self.scenario_parameters, op_dir_names,
                                     req_queue, resp_queue, dispatch_queue),
                               daemon=True)
            process.start()

            proxy = FleetControlProcessProxy(op_id, process, req_queue, resp_queue, dispatch_queue,
                                              sim_vehicles=self.sim_vehicles)
            op_vehicles = [veh_obj for (o, v), veh_obj in sorted(self.sim_vehicles.items())
                           if o == op_id]
            initial_snapshots = [build_vehicle_snapshot(veh_obj) for veh_obj in op_vehicles]
            proxy.sync_vehicles(initial_snapshots)  # blocking - doubles as "worker ready" handshake
            new_operators.append(proxy)

        self.operators = new_operators
        if self.broker is not None:
            # BrokerBase stores whatever list object it was given; re-point it explicitly
            # rather than relying on it having kept a reference to the same list.
            self.broker.amod_operators = self.operators
        self._operator_processes_started = True
        LOG.info(f"AsyncFleetSimulationBase: {len(self.operators)} operator worker(s) running.")

    def _apply_pending_vehicle_dispatches(self):
        """ drains every operator's dispatch_queue (schedules pushed by
        QueueFleetBackend.send_schedule in the worker, via RemoteVehicleDispatchMixin's
        _build_VRLs()-based translation) and applies them to the real SimulationVehicle
        objects this process owns - this replaces the direct in-process
        _dispatch_vehicle_plan() mutation that happens when an operator runs in-process.
        rid-only references are resolved against self.demand.rq_db (the real
        RequestBase objects), mirroring how the worker resolves them against its own
        rq_dict (PlanRequest objects) on the way in.

        a dispatch can be stale by the time it's applied here: the worker computed it
        against its own (possibly several steps old) LocalVehicleMirror snapshot, and the
        real vehicle may since have started an actual locked leg (e.g. a boarding already
        in progress) that this plan doesn't know about - SimulationVehicle.assign_vehicle_plan
        refuses that (AssertionError) unless force_ignore_lock is set. rather than forcing
        it through (which would silently corrupt the in-progress leg) or crashing the
        whole simulation, a stale dispatch is dropped and logged - this is exactly the
        kind of degradation-under-latency effect this architecture exists to make
        observable, not something to paper over. """
        from src.simulation.Legs import VehicleRouteLeg
        for op_id, proxy in enumerate(self.operators):
            for vid, sim_time, stops, force in proxy.drain_dispatch_queue():
                veh_obj = self.sim_vehicles[(op_id, vid)]
                vrls = [
                    VehicleRouteLeg(
                        stop["status"], stop["destination_pos"],
                        {1: [self.demand.rq_db[rid] for rid in stop["boarding_rids"]],
                         -1: [self.demand.rq_db[rid] for rid in stop["alighting_rids"]]},
                        power=stop["power"], duration=stop["duration"], route=stop["route"],
                        locked=stop["locked"], earliest_start_time=stop["earliest_start_time"],
                    )
                    for stop in stops
                ]
                try:
                    veh_obj.assign_vehicle_plan(vrls, sim_time, force_ignore_lock=force)
                except AssertionError as e:
                    LOG.warning(f"op_id {op_id} vid {vid}: dropped stale vehicle plan dispatch at "
                                f"sim_time {sim_time} - vehicle's current leg is locked/already "
                                f"started and moved on since this plan was computed ({e})")

    def poll_all_operators(self):
        """ non-blocking drain of every operator's response queue. returns
        {op_id: [(call_id, code, payload), ...]} for whatever has arrived so far (possibly
        empty per operator, possibly empty for all). worker-side errors are already logged
        and filtered out by FleetControlProcessProxy.poll() - never raised here, since
        nothing is left waiting for a specific call's result except the offer-tracking a
        concrete step() implements itself (see AsyncNonBlockingSimulation). """
        return {op_id: proxy.poll() for op_id, proxy in enumerate(self.operators)}

    def terminate_operator_processes(self):
        if not self._operator_processes_started:
            return
        for proxy in self.operators:
            try:
                proxy.shutdown()
            except Exception as e:
                LOG.warning(f"error shutting down operator worker op_id {proxy.op_id}: {e}")
        self._operator_processes_started = False

    def run(self, tqdm_position=0):
        try:
            super().run(tqdm_position=tqdm_position)
        finally:
            self.terminate_operator_processes()

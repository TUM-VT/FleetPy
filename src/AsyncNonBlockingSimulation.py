from __future__ import annotations
import logging
import time

from src.AsyncFleetSimulationBase import AsyncFleetSimulationBase
from src.FleetSimulationBase import FleetSimulationBase
from src.fleetctrl_process.comcodes import COMCODE
from src.misc.globals import G_AR_MAX_DEC_T, G_SIM_TIME_STEP

LOG = logging.getLogger(__name__)

""" step() that never waits for a FleetControl worker to respond, so the simulation clock
advances independently of (and can run ahead of, or fall behind) fleet-control compute
time - the point being to actually observe what happens to the system when FleetControl
is slow, not to hide that behind a blocking call.

Every dispatch to an operator (user_request, get_offer, time_trigger,
receive_status_update, ...) is fire-and-forget (see FleetControlProcessProxy - dispatch()
puts a message on the queue and returns immediately). Responses are picked up
opportunistically, whenever they've arrived, at the top of the NEXT step() call via
proxy.poll() (non-blocking). This is a genuine behavioral departure from
ImmediateDecisionsSimulation/BatchOfferSimulation (both of which block on
Broker.collect_offers() synchronously) - deliberately so; forcing this into their
per-rid-blocking shape would defeat the purpose.

Offer handling: a traveler whose offer(s) aren't all in yet by the time _rid_chooses_offer
is called simply stays undecided - this reuses FleetSimulationBase's existing
get_undecided_travelers()/leaves_system() retry mechanism unchanged (see
FleetSimulationBase._rid_chooses_offer): if rq_obj.choose_offer(...) can't decide yet
(no/incomplete offers), the traveler is re-evaluated next step once more offers have
arrived, or gives up per its own request model's patience - no new timeout logic needed.

Real-time pacing: step() measures its own non-blocking wall-clock duration and sleeps off
whatever's left of the (time_step / async_realtime_factor) budget, so - as long as
FleetControl keeps up - one simulated time_step takes about that many real seconds.
Set scenario parameter "async_realtime_factor" (e.g. 1.0 for real-time, 10.0 for 10x
real-time speed); omitted/0/None disables pacing entirely (default - important for
automated tests, which would otherwise run at real-world wall-clock speed). If a step
overruns its budget (e.g. FleetControl - or anything else - was too slow), the next step
starts immediately rather than trying to catch up: the resulting lag is exactly the
signal this class exists to make visible.

KNOWN LIMITATION - stale-dispatch state divergence: a vehicle plan a worker computed
(e.g. in user_confirms_booking()) can be stale by the time
AsyncFleetSimulationBase._apply_pending_vehicle_dispatches() tries to apply it to the
real SimulationVehicle - the vehicle may since have started an actual locked leg (e.g. a
boarding already in progress) that the plan doesn't know about. That case is caught and
the dispatch is dropped (logged as a warning), which keeps the real vehicle consistent -
but the OPERATOR's own bookkeeping (rq_dict, veh_plans, rid_to_assigned_vid, ...) already
assumed the assignment succeeded and is not rolled back, since FleetControlBase has no
"undo my last assignment" hook to call. Observed effect in testing: the affected
vehicle's *ride still completes correctly* (the real vehicle keeps executing whatever
plan it actually has), but every subsequent receive_status_update() for that vehicle
raises inside the worker's compute_VehiclePlan_utility()/objective function, because it's
now evaluating a plan the vehicle was never actually assigned - these are caught and
logged per-call by the worker's dispatch loop (never crash the process), so the
simulation keeps running in a degraded-but-functional state for that vehicle. Properly
reconciling this would need a real rollback/renegotiation protocol between worker and
operator, which is out of scope here - a longer user_max_decision_time (more time for a
confirmed booking's plan to go stale before being applied) or a slower/overloaded
FleetControl relative to async_realtime_factor make this more likely to occur; it's
exactly the kind of degradation this architecture exists to make observable, not
something to silently paper over.
"""


class AsyncNonBlockingSimulation(AsyncFleetSimulationBase, FleetSimulationBase):

    def check_sim_env_spec_inputs(self, scenario_parameters):
        # unlike ImmediateDecisionsSimulation, this class does NOT require
        # user_max_decision_time == 0 - to the contrary, since an offer can only ever
        # become visible via poll() at the earliest one time_step after it was requested
        # (see module docstring), user_max_decision_time == 0 means every traveler's
        # RequestBase.leaves_system() gives up before an offer could possibly have
        # arrived, and nothing ever gets booked. this is a scenario-config footgun, not
        # something this class can silently fix (the "right" decision-time budget depends
        # on the scenario), so it's surfaced loudly instead.
        max_dec_t = scenario_parameters.get(G_AR_MAX_DEC_T, 0)
        time_step = scenario_parameters.get(G_SIM_TIME_STEP, 1)
        if max_dec_t < time_step:
            LOG.warning(f"{G_AR_MAX_DEC_T}={max_dec_t} is less than one time_step ({time_step}) - "
                        f"every request will give up before an async offer can possibly arrive "
                        f"(offers are only ever picked up at the start of a later step()). "
                        f"set {G_AR_MAX_DEC_T} to at least a few multiples of time_step.")
        return scenario_parameters

    def add_evaluate(self):
        pass

    def __init__(self, scenario_parameters: dict):
        super().__init__(scenario_parameters)
        self.realtime_factor = scenario_parameters.get("async_realtime_factor", None)
        n_op = len(self.operators)
        # rid <-> in-flight/resolved GET_OFFER call tracking, per operator. never pruned
        # (known limitation, same tradeoff as worker.py's rid_lookup cache - fine for the
        # scenario sizes this was built/tested against, would need bounding for a real
        # long-running deployment).
        self._offer_call_to_rid = [dict() for _ in range(n_op)]   # op_id -> {call_id: rid}
        self._offer_pending = [dict() for _ in range(n_op)]       # op_id -> {rid: call_id}
        self._offer_resolved = {}                                  # rid -> {op_id, ...} already answered

    def step(self, sim_time):
        t_wall_start = time.perf_counter()

        # 0) pick up whatever became available since the last step - never blocks.
        self._poll_offers(sim_time)
        self._apply_pending_vehicle_dispatches()

        # 1) advance fleets and network. update_sim_state_fleets -> Broker ->
        # FleetControlProcessProxy.receive_status_update is fire-and-forget now, so this
        # no longer waits on fleet control either.
        self.update_sim_state_fleets(sim_time - self.time_step, sim_time)
        new_travel_times = self.routing_engine.update_network(sim_time)
        if new_travel_times:
            self.broker.inform_network_travel_time_update(sim_time)

        # 2) new + still-undecided travelers
        list_undecided_travelers = list(self.demand.get_undecided_travelers(sim_time))
        last_time = sim_time - self.time_step
        if last_time < self.start_time:
            last_time = None
        list_new_traveler_rid_obj = self.demand.get_new_travelers(sim_time, since=last_time)

        for rid, rq_obj in list_new_traveler_rid_obj:
            for operator in self.operators:
                operator.user_request(rq_obj, sim_time)

        for rid, rq_obj in list_undecided_travelers + list_new_traveler_rid_obj:
            self._request_missing_offers(rid)
            # decide with whatever offers have arrived so far (possibly none/incomplete);
            # if choose_offer() can't decide yet, this rid simply stays undecided and gets
            # revisited next step via get_undecided_travelers() - no new logic needed.
            self._rid_chooses_offer(rid, rq_obj, sim_time)

        # 3)
        self._check_waiting_request_cancellations(sim_time)

        # 4) time_trigger - one fire-and-forget dispatch per operator, no wait
        for operator in self.operators:
            operator.time_trigger(sim_time)

        # 5) charging - unchanged, still synchronous/in-process (public charging infra is
        # out of scope for AsyncFleetSimulationBase, see its module docstring)
        for ch_op_dict in self.charging_operator_dict.values():
            for ch_op in ch_op_dict.values():
                ch_op.time_trigger(sim_time)

        self.record_stats()

        self._pace_realtime(sim_time, t_wall_start)

    def _request_missing_offers(self, rid):
        resolved = self._offer_resolved.get(rid, set())
        for op_id, operator in enumerate(self.operators):
            if op_id in resolved or rid in self._offer_pending[op_id]:
                continue  # already answered, or already asked and awaiting poll()
            call_id = operator.get_offer(rid)
            self._offer_pending[op_id][rid] = call_id
            self._offer_call_to_rid[op_id][call_id] = rid

    def _poll_offers(self, sim_time):
        rq_db = self.demand.rq_db
        for op_id, proxy in enumerate(self.operators):
            for call_id, code, payload in proxy.poll():
                if code != COMCODE.RESULT:
                    continue  # an ACK from some other fire-and-forget call - nothing to do
                rid = self._offer_call_to_rid[op_id].pop(call_id, None)
                if rid is None:
                    continue  # a RESULT for a call we're no longer tracking (e.g. already resolved another way)
                self._offer_pending[op_id].pop(rid, None)
                if payload is None:
                    continue  # operator hasn't decided yet - _request_missing_offers will re-ask next step
                self._offer_resolved.setdefault(rid, set()).add(op_id)
                rq_obj = rq_db.get(rid)
                if rq_obj is not None:  # None if the request already left the system before this offer arrived
                    rq_obj.receive_offer(op_id, payload, sim_time)

    def _pace_realtime(self, sim_time, t_wall_start):
        if not self.realtime_factor:
            return
        budget = self.time_step / self.realtime_factor
        elapsed = time.perf_counter() - t_wall_start
        if elapsed < budget:
            time.sleep(budget - elapsed)
        else:
            LOG.warning(f"step {sim_time}: took {elapsed:.3f}s, over the {budget:.3f}s "
                        f"real-time budget by {elapsed - budget:.3f}s")

from __future__ import annotations
import logging
import os
import traceback
from typing import Any, Dict, List

from src.misc.globals import *
from src.misc.init_modules import load_routing_engine, load_fleet_control_module
from src.fleetctrl.RemoteVehicleDispatchMixin import RemoteVehicleDispatchMixin, QueueFleetBackend
from src.fleetctrl_process.comcodes import COMCODE
from src.fleetctrl_process.vehicle_snapshot import VehicleSnapshot, VRLSnapshot, LocalVehicleMirror

LOG = logging.getLogger(__name__)

""" Entry point for an operator worker subprocess (target of multiprocessing.Process).
Mirrors the pattern already used by
src/fleetctrl/pooling/batch/AlonsoMora/AlonsoMoraParallelization.py: only plain,
picklable construction params (dicts/strings/paths) cross the Windows-spawn process
boundary; the routing engine is reloaded from disk locally rather than pickled, since
that's the only approach that also works for the non-picklable Cython-backed routing
engine variants (NetworkBasicCpp).

Vehicle state crosses the boundary as VehicleSnapshot/VRLSnapshot (see
vehicle_snapshot.py) - never as a real SimulationVehicle/VehicleRouteLeg - and requests
cross as bare rids, resolved locally against a worker-local rid_lookup cache of the
operator's own PlanRequest instances (see _resolve_finished_legs() below for why this is
a separate cache rather than just reading operator.rq_dict), never as real RequestBase
objects.

known limitation: rid_lookup is never pruned, so it grows for the lifetime of the worker
process - fine for the scenario sizes this was built/tested against, but a real
long-running deployment would want to drop entries once a request's trip is fully
complete (mirroring whatever operator.rq_dict itself already tracks for that purpose).

known limitation: only the operator's own depot/charging infra
(operator_attributes[G_OP_DEPOT_F]) is reconstructed locally. Public charging operators
are scenario-wide/shared across operators in the normal in-process simulation; giving
each operator subprocess its own independent copy would silently desync shared station
capacity across operators, so it is intentionally left out of this first version rather
than "solved" incorrectly - operators configured with public charging are not yet
supported by AsyncFleetSimulationBase.
"""


def _resolve_finished_legs(rid_lookup: Dict[Any, Any], legs: List[VRLSnapshot]):
    """ resolves rids against rid_lookup, NOT operator.rq_dict directly: Broker calls
    acknowledge_alighting() (which several FleetControlBase subclasses use to del
    self.rq_dict[rid], e.g. PoolingIRSOnly.py:225) for a rid before the corresponding
    receive_status_update() call for the same vehicle/timestep - update_sim_state_fleets()
    processes alighting acks and the vehicle's status update in that order (see
    FleetSimulationBase.update_sim_state_fleets). The real in-process code never hits this
    because a real VehicleRouteLeg already carries its resolved RequestBase objects rather
    than looking them up again by rid at receive_status_update() time. rid_lookup is a
    worker-local cache populated at USER_REQUEST time, independent of operator.rq_dict's
    own (shorter) lifecycle. """
    from src.fleetctrl.RemoteVehicleStatusMixin import RemoteVRL
    resolved = []
    for vrl in legs:
        resolved.append(RemoteVRL(
            status=vrl.status,
            destination_pos=vrl.destination_pos,
            rq_dict={1: [rid_lookup[rid] for rid in vrl.boarding_rids],
                     -1: [rid_lookup[rid] for rid in vrl.alighting_rids]},
            power=vrl.power,
            duration=vrl.duration,
            route=vrl.route,
            locked=vrl.locked,
            earliest_start_time=vrl.earliest_start_time,
        ))
    return resolved


def _apply_vehicle_sync(operator, snapshots: List[VehicleSnapshot]):
    for snap in snapshots:
        operator.sim_vehicles[snap.vid] = LocalVehicleMirror(snap, operator.rq_dict)


def _build_local_routing_engine(scenario_parameters: Dict, dir_names: Dict):
    network_type = scenario_parameters[G_NETWORK_TYPE]
    network_dynamics_file = scenario_parameters.get(G_NW_DYNAMIC_F, None)
    return load_routing_engine(network_type, dir_names[G_DIR_NETWORK],
                                network_dynamics_file_name=network_dynamics_file)


def _build_local_charging_infra(op_id: int, operator_attributes: Dict, scenario_parameters: Dict,
                                 dir_names: Dict, routing_engine):
    depot_f_name = operator_attributes.get(G_OP_DEPOT_F)
    if depot_f_name is None or not dir_names.get(G_DIR_INFRA):
        return None
    from src.infra.ChargingInfrastructure import OperatorChargingAndDepotInfrastructure
    depot_f = os.path.join(dir_names[G_DIR_INFRA], depot_f_name)
    return OperatorChargingAndDepotInfrastructure(op_id, depot_f, operator_attributes, scenario_parameters,
                                                   dir_names, routing_engine)


def run_fleetcontrol_worker(op_id: int, operator_attributes: Dict, scenario_parameters: Dict, dir_names: Dict,
                             req_queue, resp_queue, dispatch_queue):
    """ subprocess target. blocks on req_queue for (call_id, code, args) messages, replies
    on resp_queue with (call_id, COMCODE.ACK/RESULT/ERROR, payload). """
    logging.basicConfig(level=logging.INFO)
    LOG.info(f"fleetcontrol worker for op_id {op_id} starting up")

    routing_engine = _build_local_routing_engine(scenario_parameters, dir_names)
    op_charge_depot_infra = _build_local_charging_infra(op_id, operator_attributes, scenario_parameters,
                                                          dir_names, routing_engine)

    # first message on the queue is always the initial fleet snapshot, so nr_vehicles /
    # veh_plans are correctly sized at FleetControlBase construction time (see proxy.py)
    call_id, code, args = req_queue.get()
    assert code == COMCODE.SYNC_VEHICLES, f"expected initial SYNC_VEHICLES message, got {code}"
    initial_snapshots: List[VehicleSnapshot] = args[0]

    BaseOpClass = load_fleet_control_module(operator_attributes[G_OP_MODULE])
    # RemoteVehicleDispatchMixin must come first in the MRO so its _dispatch_vehicle_plan()
    # (-> QueueFleetBackend) wins over FleetControlBase's default (-> real
    # veh_obj.assign_vehicle_plan(), which would fail here - there is no real
    # SimulationVehicle in this process, only LocalVehicleMirror).
    OpClass = type(f"Remote{BaseOpClass.__name__}", (RemoteVehicleDispatchMixin, BaseOpClass), {})
    # build the initial vehicle mirrors *before* construction (FleetControlBase.__init__
    # builds self.veh_plans/nr_vehicles directly from the given vehicle list - passing []
    # and back-filling sim_vehicles afterwards leaves veh_plans empty). rq_dict={} is safe
    # here: at simulation start vehicles carry no pax and only ever locked/blocking plan
    # stops with no boarding/alighting rids, so no rid lookups actually happen yet.
    initial_vehicles = sorted((LocalVehicleMirror(snap, {}) for snap in initial_snapshots), key=lambda v: v.vid)
    operator = OpClass(op_id, operator_attributes, initial_vehicles, routing_engine, None, scenario_parameters,
                        dir_names, op_charge_depot_infra, [])
    operator.remote_backend = QueueFleetBackend(dispatch_queue, op_id)
    resp_queue.put((call_id, COMCODE.ACK, None))

    LOG.info(f"fleetcontrol worker for op_id {op_id} ready ({operator.nr_vehicles} vehicles)")

    # rid -> PlanRequest cache independent of operator.rq_dict's own lifecycle, see
    # _resolve_finished_legs() docstring above for why this is necessary.
    rid_lookup: Dict[Any, Any] = {}

    while True:
        call_id, code, args = req_queue.get()
        try:
            if code == COMCODE.KILL:
                resp_queue.put((call_id, COMCODE.ACK, None))
                break
            elif code == COMCODE.USER_REQUEST:
                rq_obj, sim_time = args
                operator.user_request(rq_obj, sim_time)
                prq = operator.rq_dict.get(rq_obj.get_rid_struct())
                if prq is not None:
                    rid_lookup[rq_obj.get_rid_struct()] = prq
                resp_queue.put((call_id, COMCODE.ACK, None))
            elif code == COMCODE.GET_OFFER:
                (rid,) = args
                offer = operator.get_current_offer(rid)
                resp_queue.put((call_id, COMCODE.RESULT, offer))
            elif code == COMCODE.CONFIRM_BOOKING:
                rid, sim_time = args
                operator.user_confirms_booking(rid, sim_time)
                resp_queue.put((call_id, COMCODE.ACK, None))
            elif code == COMCODE.CANCEL_REQUEST:
                rid, sim_time = args
                operator.user_cancels_request(rid, sim_time)
                resp_queue.put((call_id, COMCODE.ACK, None))
            elif code == COMCODE.ACK_BOARDING:
                rid, vid, boarding_time = args
                operator.acknowledge_boarding(rid, vid, boarding_time)
                resp_queue.put((call_id, COMCODE.ACK, None))
            elif code == COMCODE.ACK_ALIGHTING:
                rid, vid, alighting_time = args
                operator.acknowledge_alighting(rid, vid, alighting_time)
                resp_queue.put((call_id, COMCODE.ACK, None))
            elif code == COMCODE.RECEIVE_STATUS_UPDATE:
                vid, sim_time, finished_legs, force_update, veh_snapshot = args
                operator.sim_vehicles[vid] = LocalVehicleMirror(veh_snapshot, operator.rq_dict)
                operator.receive_status_update(vid, sim_time, _resolve_finished_legs(rid_lookup, finished_legs),
                                                force_update)
                resp_queue.put((call_id, COMCODE.ACK, None))
            elif code == COMCODE.INFORM_TT_UPDATE:
                (sim_time,) = args
                routing_engine.update_network(sim_time)
                operator.inform_network_travel_time_update(sim_time)
                resp_queue.put((call_id, COMCODE.ACK, None))
            elif code == COMCODE.TIME_TRIGGER:
                (sim_time,) = args
                operator.time_trigger(sim_time)
                resp_queue.put((call_id, COMCODE.ACK, None))
            elif code == COMCODE.RECORD_STATS:
                (force,) = args
                operator.record_dynamic_fleetcontrol_output(force=force)
                resp_queue.put((call_id, COMCODE.ACK, None))
            elif code == COMCODE.ADD_INIT:
                op_attrs, scen_params = args
                operator.add_init(op_attrs, scen_params)
                resp_queue.put((call_id, COMCODE.ACK, None))
            elif code == COMCODE.SYNC_VEHICLES:
                (snapshots,) = args
                _apply_vehicle_sync(operator, snapshots)
                resp_queue.put((call_id, COMCODE.ACK, None))
            else:
                raise ValueError(f"unknown COMCODE {code}")
        except Exception as e:
            LOG.error(f"op_id {op_id}: error handling call {call_id} ({code}): {e}")
            resp_queue.put((call_id, COMCODE.ERROR, (repr(e), traceback.format_exc())))

    LOG.info(f"fleetcontrol worker for op_id {op_id} shutting down")

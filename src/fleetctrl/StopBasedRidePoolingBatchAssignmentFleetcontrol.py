import logging
import time

from src.fleetctrl.RidePoolingBatchAssignmentFleetcontrol import RidePoolingBatchAssignmentFleetcontrol
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_StopBasedRidePoolingBatchAssignmentFleetcontrol = {
    "doc": """RidePoolingBatchAssignmentFleetcontrol variant for the stop-based on-demand service
    (WP1). The base class always builds its PlanRequest with walking_time_start/walking_time_end
    left at their default of 0, so a control function that reads them (e.g.
    distance_and_user_times_with_walk) never actually sees any walking cost. This variant threads
    the traveler's walk-to/from-boarding-point time (StopBasedUserGroupRequest.
    total_walking_distance) into the PlanRequest instead, so the operator's own routing/insertion
    decisions can account for it, not just the traveler's own accept/decline threshold check.
    RidePoolingBatchOptimizationFleetControlBase and RidePoolingBatchAssignmentFleetcontrol are
    left untouched by this -- dtd, which shares those classes but has no walk phase, is
    unaffected.""",
    "inherit": "RidePoolingBatchAssignmentFleetcontrol",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


class StopBasedRidePoolingBatchAssignmentFleetcontrol(RidePoolingBatchAssignmentFleetcontrol):
    """RidePoolingBatchAssignmentFleetcontrol variant that passes a stop-based traveler's
    walk-to/from-boarding-point time into the PlanRequest, so control functions that read
    walking_time_start/walking_time_end (e.g. distance_and_user_times_with_walk) see a real value
    instead of the base class's implicit 0."""

    @staticmethod
    def _stop_based_walking_times(rq):
        """(walking_time_start, walking_time_end) for a StopBasedUserGroupRequest; (0, 0) for any
        other request type (e.g. plain UserGroupRequest, which has no walk phase)."""
        total_walking_distance = getattr(rq, "total_walking_distance", None)
        if not total_walking_distance:
            return 0.0, 0.0
        walking_speed = getattr(rq, "walking_speed", None)
        t_walk = total_walking_distance / walking_speed if walking_speed else 0.0
        if getattr(rq, "true_o_node", None) != getattr(rq, "o_node", None):
            return t_walk, 0.0
        return 0.0, t_walk

    def user_request(self, rq, sim_time):
        """Duplicates RidePoolingBatchOptimizationFleetControlBase.user_request +
        RidePoolingBatchAssignmentFleetcontrol.user_request's wrapper (rather than adding a hook
        to those base classes), with the traveler's stop-based walking time attached to the
        PlanRequest. Duplicated on purpose, to leave existing fleet control classes and their
        behavior for other service types (e.g. dtd) completely untouched."""
        LOG.debug(f"Incoming request {rq.__dict__} at time {sim_time}")
        if self.rq_dict.get(rq.get_rid_struct()):
            return
        t0 = time.perf_counter()
        self.sim_time = sim_time

        walking_time_start, walking_time_end = self._stop_based_walking_times(rq)
        prq = PlanRequest(rq, self.routing_engine, min_wait_time=self.min_wait_time,
                          max_wait_time=self.max_wait_time,
                          max_detour_time_factor=self.max_dtf, max_constant_detour_time=self.max_cdt,
                          add_constant_detour_time=self.add_cdt, min_detour_time_window=self.min_dtw,
                          boarding_time=self.const_bt,
                          walking_time_start=walking_time_start, walking_time_end=walking_time_end)
        rid_struct = rq.get_rid_struct()

        if prq.o_pos == prq.d_pos:
            LOG.debug(f"automatic decline for rid {rid_struct}!")
            self._create_rejection(prq, sim_time)
            return

        self.new_requests[rid_struct] = 1
        self.rq_dict[rid_struct] = prq

        o_pos, t_pu_earliest, t_pu_latest = prq.get_o_stop_info()
        if t_pu_earliest - sim_time > self.opt_horizon:
            self.reservation_module.add_reservation_request(prq, sim_time)
            LOG.debug(f"reservation rid {rid_struct}")
            prq.set_reservation_flag(True)
            self.RPBO_Module.add_new_request(rid_struct, prq, consider_for_global_optimisation=False)
        else:
            self.RPBO_Module.add_new_request(rid_struct, prq)

        if self.repo and not prq.get_reservation_flag():
            self.repo.register_user_request(prq, sim_time)

        # record cpu time
        dt = round(time.perf_counter() - t0, 5)
        old_dt = self._get_current_dynamic_fleetcontrol_value(sim_time, G_FCTRL_CT_RQU)
        if old_dt is None:
            new_dt = dt
        else:
            new_dt = old_dt + dt
        output_dict = {G_FCTRL_CT_RQU: new_dt}
        self._add_to_dynamic_fleetcontrol_output(sim_time, output_dict)

        if not self.rq_dict[rq.get_rid_struct()].get_reservation_flag():
            self.unassigned_requests_1[rq.get_rid_struct()] = 1
        return {}

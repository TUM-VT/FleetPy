from __future__ import annotations
import logging

from typing import Callable, Dict, List, Any, Tuple, TYPE_CHECKING

from src.fleetctrl.pooling.batch.InsertionHeuristic.BatchInsertionHeuristicAssignment import \
    BatchInsertionHeuristicAssignment
from src.fleetctrl.pooling.immediate.insertion import insert_prq_in_selected_veh_list

if TYPE_CHECKING:
    from src.simulation.Vehicles import SimulationVehicle
    from src.simulation.Legs import VehicleRouteLeg
    from src.fleetctrl.planning.PlanRequest import PlanRequest
    from src.fleetctrl.SoDZonalBatchAssignmentFleetcontrol import PtLineZonal

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_BatchZonalInsertionHeuristicAssignment = {
    "doc": """this class uses a simple insertion heuristic to assign zonal requests in batch that 
    havent been assigned before """,
    "inherit": "InsertionHeuristic",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
    ],
    "mandatory_modules": [],
    "optional_modules": []
}


class BatchZonalInsertionHeuristicAssignment(BatchInsertionHeuristicAssignment):
    def compute_new_vehicle_assignments(self, sim_time: int, vid_to_list_passed_VRLs: Dict[int, List[VehicleRouteLeg]],
                                        veh_objs_to_build: Dict[int, SimulationVehicle] = {},
                                        new_travel_times: bool = False, build_from_scratch: bool = False):
        """
        this function computes new vehicle assignments based on current fleet information
        (adapted implementation for zonal constraints)
                :param sim_time : current simulation time
                :param vid_to_list_passed_VRLs : (dict) vid -> list_passed_VRLs; needed to update database and V2RBs
                :param veh_objs_to_build: only these vehicles will be optimized (all if empty) dict vid
                -> SimVehicle obj only for special cases needed in current alonso mora module
                :param new_travel_times : bool; if traveltimes changed in the routing engine
                :param build_from_scratch : only for special cases needed in current alonso mora module
                """

        self.sim_time = sim_time
        for rid in list(self.unassigned_requests.keys()):
            if self.rid_to_consider_for_global_optimisation.get(rid) is None:
                continue

            vid_to_exclude = {}
            # check flexible portion time to add vehicles to excluded_vid
            pt_line: PtLineZonal = self.fleetcontrol.return_ptline()

            # add zonal constraints by adding veh not assigned to the same zone as the request in excluded_vid
            if self.fleetcontrol.n_zones > 1:
                pu_zone = pt_line.return_pos_zone(self.fleetcontrol.rq_dict[rid].o_pos)
                do_zone = pt_line.return_pos_zone(self.fleetcontrol.rq_dict[rid].d_pos)

                # if request pick-up & drop-off in fixed route, then consider all vehicles
                rq_zone = pt_line.return_rid_zone(rid)
                LOG.debug(f"rid {rid} pu_zone {pu_zone} do_zone {do_zone} rq_zone {rq_zone} "
                          f"pu_pos {self.fleetcontrol.rq_dict[rid].o_pos} "
                          f"do_pos {self.fleetcontrol.rq_dict[rid].d_pos}")

                if rq_zone is None:
                    pass
                # if pick-up & drop-off in different zones of flex routes, then ignore zonal vehicles
                elif rq_zone == -1:
                    for vid in self.fleetcontrol.veh_plans.keys():
                        if vid not in pt_line.veh_zone_assignment.keys():  # ignore vehicles not assigned to a zone
                            vid_to_exclude[vid] = 1
                        elif pt_line.veh_zone_assignment[vid] != -1:
                            vid_to_exclude[vid] = 1
                # otherwise, ignore all zonal vehicles but one zone
                else:
                    for vid in self.fleetcontrol.veh_plans.keys():
                        if vid not in pt_line.veh_zone_assignment.keys():  # ignore vehicles not assigned to a zone
                            vid_to_exclude[vid] = 1
                        else:
                            veh_zone = pt_line.veh_zone_assignment[vid]
                            if veh_zone != rq_zone and veh_zone != -1:  # include specific zonal & regular vehicles
                                # if veh_zone != max(pu_zone, do_zone): # ignore regular vehicles too
                                vid_to_exclude[vid] = 1

            selected_veh_list = [veh for veh in self.fleetcontrol.sim_vehicles if veh.vid not in vid_to_exclude]
            LOG.debug(f"selected vehicles: {[veh.vid for veh in selected_veh_list]}")

            r_list = insert_prq_in_selected_veh_list(
                selected_veh_list, self.fleetcontrol.veh_plans, self.active_requests[rid], self.fleetcontrol.vr_ctrl_f,
                self.fleetcontrol.routing_engine, self.fleetcontrol.rq_dict, sim_time,
                self.fleetcontrol.const_bt, self.fleetcontrol.add_bt,
                True, self.fleetcontrol.rv_heuristics
            )

            LOG.debug(f"solution for rid {rid}:")
            for vid, plan, obj in r_list:
                LOG.debug(f"vid {vid} with obj {obj}:\n plan {plan}")
                # original plan
                LOG.debug(
                    f"original obj {self.fleetcontrol.veh_plans[vid].get_utility()} "
                    f"plan {self.fleetcontrol.veh_plans[vid]}")
            if len(r_list) != 0:
                best_vid, best_plan, _ = min(r_list, key=lambda x: x[2])
                self.fleetcontrol.assign_vehicle_plan(self.fleetcontrol.sim_vehicles[best_vid], best_plan, sim_time)
                # update utility
                veh_obj = self.fleetcontrol.sim_vehicles[best_vid]
                upd_utility_val = self.fleetcontrol.compute_VehiclePlan_utility(sim_time, veh_obj,
                                                                                self.fleetcontrol.veh_plans[best_vid])
                self.fleetcontrol.veh_plans[best_vid].set_utility(upd_utility_val)

        self.unassigned_requests = {}  # only try once

    def register_change_in_time_constraints(self, rid: Any, prq: PlanRequest, assigned_vid: int = None,
                                            exceeds_former_time_windows: bool = True):
        # TODO: consider moving this to the base class (BatchAssignmentAlgorithmBase)
        raise NotImplementedError

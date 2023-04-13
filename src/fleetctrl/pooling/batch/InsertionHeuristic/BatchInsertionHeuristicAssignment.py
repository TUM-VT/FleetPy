from __future__ import annotations
import logging

import time
from typing import Callable, Dict, List, Any, Tuple, TYPE_CHECKING

from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from src.fleetctrl.pooling.batch.BatchAssignmentAlgorithmBase import BatchAssignmentAlgorithmBase
from src.fleetctrl.pooling.immediate.insertion import immediate_insertion_with_heuristics
from src.misc.globals import *
if TYPE_CHECKING:
    from src.simulation.Vehicles import SimulationVehicle
    from src.simulation.Legs import VehicleRouteLeg

LOG = logging.getLogger(__name__)
LARGE_INT = 100000
MAX_LENGTH_OF_TREES = 1024 # TODO
RETRY_TIME = 24*3600

INPUT_PARAMETERS_BatchInsertionHeuristicAssignment = {
    "doc" :  """this class uses a simple insertion heuristic to assign requests in batch that havent been assigned before """,
    "inherit" : "BatchAssignmentAlgorithmBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class BatchInsertionHeuristicAssignment(BatchAssignmentAlgorithmBase):
    """ this class uses a simple insertion heuristic to assign requests in batch that havent been assigned before """
    
    def compute_new_vehicle_assignments(self, sim_time : int, vid_to_list_passed_VRLs : Dict[int, List[VehicleRouteLeg]], veh_objs_to_build : Dict[int, SimulationVehicle] = {}, new_travel_times : bool = False, build_from_scratch : bool = False):
        """ this function computes new vehicle assignments based on current fleet information
        param sim_time : current simulation time
        param vid_to_list_passed_VRLs : (dict) vid -> list_passed_VRLs; needed to update database and V2RBs
        :param veh_objs_to_build: only these vehicles will be optimized (all if empty) dict vid -> SimVehicle obj only for special cases needed in current alonso mora module
        :param new_travel_times : bool; if traveltimes changed in the routing engine
        :param build_from_scratch : only for special cases needed in current alonso mora module
        """
        self.sim_time = sim_time
        if len(veh_objs_to_build) != 0:
            raise NotImplementedError
        for rid in list(self.unassigned_requests.keys()):
            if self.rid_to_consider_for_global_optimisation.get(rid) is None:
                continue
            r_list = immediate_insertion_with_heuristics(sim_time, self.active_requests[rid], self.fleetcontrol)
            if len(r_list) != 0:
                best_vid, best_plan, _ = min(r_list, key = lambda x:x[2])
                self.fleetcontrol.assign_vehicle_plan(self.fleetcontrol.sim_vehicles[best_vid], best_plan, sim_time)
        self.unassigned_requests = {} # only try once
            
    
    def get_optimisation_solution(self, vid : int) -> VehiclePlan:
        """ returns optimisation solution for vid
        :param vid: vehicle id
        :return: vehicle plan object for the corresponding vehicle
        """
        return self.fleetcontrol.veh_plans[vid]

    def set_assignment(self, vid : int, assigned_plan : VehiclePlan, is_external_vehicle_plan : bool = False):
        """ sets the vehicleplan as assigned in the algorithm database; if the plan is not computed within the this algorithm, the is_external_vehicle_plan flag should be set to true
        :param vid: vehicle id
        :param assigned_plan: vehicle plan object that has been assigned
        :param is_external_vehicle_plan: should be set to True, if the assigned_plan has not been computed within this algorithm
        """
        super().set_assignment(vid, assigned_plan, is_external_vehicle_plan=is_external_vehicle_plan)

    def get_current_assignment(self, vid : int) -> VehiclePlan: # TODO same as get_optimisation_solution (delete?)
        """ returns the vehicle plan assigned to vid currently
        :param vid: vehicle id
        :return: vehicle plan
        """
        return self.fleetcontrol.veh_plans[vid]
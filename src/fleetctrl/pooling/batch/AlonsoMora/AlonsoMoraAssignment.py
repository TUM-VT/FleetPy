from __future__ import annotations
import logging

import time
import numpy as np
from functools import cmp_to_key
from typing import Callable, Dict, List, Any, Tuple, TYPE_CHECKING

from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from src.fleetctrl.pooling.batch.BatchAssignmentAlgorithmBase import BatchAssignmentAlgorithmBase, SimulationVehicleStruct
from src.fleetctrl.pooling.GeneralPoolingFunctions import checkRRcomptibility
from src.fleetctrl.pooling.batch.AlonsoMora.V2RB import V2RB
from src.fleetctrl.pooling.immediate.insertion import simple_remove, single_insertion
from src.fleetctrl.pooling.immediate.SelectRV import filter_directionality, filter_least_number_tasks
from src.misc.globals import *
from src.simulation.Legs import VehicleRouteLeg
if TYPE_CHECKING:
    from src.routing.NetworkBase import NetworkBase
    from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraParallelization import ParallelizationManager
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.fleetctrl.planning.PlanRequest import PlanRequest
    from src.simulation.Vehicles import SimulationVehicle

LOG = logging.getLogger(__name__)
LARGE_INT = 100000
MAX_LENGTH_OF_TREES = 1024 # TODO
RETRY_TIME = 24*3600


# help functions
# --------------
def comp_key_entries(entry1 : Any, entry2 : Any) -> int:
    """ this function is used to sort keys with different data types (int, tuple, str)
    """
    if type(entry1) == type(entry2):
        if type(entry1) == tuple:
            if len(entry1) < len(entry2):
                return -1
            elif len(entry1) > len(entry2):
                return 1
            else:
                for x, y in zip(entry1, entry2):
                    c = comp_key_entries(x, y)
                    if c != 0:
                        return c
                return 0
        else:
            if entry1 < entry2:
                return -1
            elif entry1 > entry2:
                return 1
            else:
                return 0
    else:
        if type(entry1) == str:
            return -1
        elif type(entry2) == str:
            return 1
        else:
            if type(entry1) == int:
                return -1
            elif type(entry2) == int:
                return 1
    raise EnvironmentError("compare keys {} <-> {} : a new datatype within? -> not comparable".format(entry1, entry2))


def deleteRidFromRtv(rid : Any, rtv_key : tuple) -> tuple:
    """This method returns
    - rtv_key without rid        in case rids are left
    - None                       in case no rid is left
    """
    if rtv_key is None:
        return None
    vid = getVidFromRTVKey(rtv_key)
    list_rids = list(getRidsFromRTVKey(rtv_key))
    list_rids.remove(rid)
    if list_rids:
        return createRTVKey(vid, list_rids)
    else:
        return None


def getRRKey(rid1 : Any, rid2 : Any) -> tuple:
    """ this function returns an rr-key (ordered request_id pair)
    :param rid1: request_id of plan_request 1
    :param rid2: request_id of plan_request 2
    :return: ordered tuple of rid1 and rid2 """
    return tuple(sorted((rid1, rid2), key = cmp_to_key(comp_key_entries)))


def getRidsFromRTVKey(rtv_key) -> List[Any]:
    """ this function returns a list of plan_request_ids corresponding to the rtv_key
    :param rtv_key: rtv_key corresponding to an v2rb-obj
    :return: list of planrequest_ids """
    if rtv_key is None:
        return []
    return rtv_key[1:]


def getVidFromRTVKey(rtv_key : tuple) -> Any:
    """ this function returns the vehicle_id corresponding to the rtv_key
    :param rtv_key: rtv_key corresponding to an v2rb-obj
    :return: vehicle_id """
    return rtv_key[0]


def createRTVKey(vid : int, rid_list : List[Any]) -> tuple:
    """ this functions creates a new rtv_key from a vehicle_id and a list of plan_request_ids
    :param vid: vehicle id
    :param rid_list: list of plan_request_ids
    :return: type tuple : rtv_key """
    if len(rid_list) == 0:
        return None
    sorted_rid_list = tuple(sorted(rid_list, key = cmp_to_key(comp_key_entries)))
    return (vid, ) + sorted_rid_list


def createListLowerLevelKeys(build_key : tuple, new_rid : Any, do_not_remove_for_lower_keys : List[tuple]) -> List[tuple]:
    """This function creates keys from build_key that are one level lower than build_key
    by removing one of the rids.
    If one of the rids is in do_not_remove_for_lower_keys, no key is created for this rid.
    :param build_key: rtv_key of v2rbs to build on
    :param new_rid: plan_request_id of new request
    :param do_not_remove_for_lower_key: list of plan_request_ids that must be part of the lower key
    :return: list of lower rtv_keys"""
    return_keys = []
    vid = getVidFromRTVKey(build_key)
    list_rids = list(getRidsFromRTVKey(build_key))
    if new_rid in list_rids:
        return []
    for rid in list_rids:
        if rid not in do_not_remove_for_lower_keys:
            copy_of_list = list_rids[:]
            copy_of_list.remove(rid)
            if len(copy_of_list) == 0:
                return []
            copy_of_list.append(new_rid)
            new_key = createRTVKey(vid, copy_of_list)
            return_keys.append(new_key)
    return return_keys


def getNecessaryKeys(rtv_key : tuple, r_ob : List[Any]) -> List[tuple]:
    """ this functions computes all rtv_keys that must be existent for rtv_key to exist
    :param rtv_key: rtv_key of v2rb_obj
    :param r_ob: list of request_ids currently on board of the corresponding vehicle
    :return: iterator of necessary keys """
    if not rtv_key:
        return []
    vid = getVidFromRTVKey(rtv_key)
    ass_rids = getRidsFromRTVKey(rtv_key)
    v_r_ob = []
    for rid in ass_rids:
        if rid in r_ob:
            v_r_ob.append(rid)
    rid_combs = [v_r_ob]
    for rid in ass_rids:
        if rid in v_r_ob or rid in r_ob:
            continue
        #new_combs = []
        for comb in rid_combs:
            new_comb = comb[:]
            new_comb.append(rid)
            yield createRTVKey(vid, new_comb)
    #         new_combs.append(new_comb)
    #     rid_combs += new_combs
    # nec_key = [createRTVKey(vid, comb) for comb in rid_combs if len(comb) > 0]
    # return nec_key


def getRTVkeyFromVehPlan(veh_plan : VehiclePlan) -> tuple:
    """ creates the rtv_key based on a vehicle plan
    :param veh_plan: vehicle plan object in question
    :return: rtv_key
    """
    if veh_plan is None:
        return None
    rids = {}
    vid = veh_plan.vid
    for pstop in veh_plan.list_plan_stops:
        for rid in pstop.get_list_boarding_rids():
            rids[rid] = 1
        for rid in pstop.get_list_alighting_rids():
            rids[rid] = 1
    if len(rids.keys()) == 0:
        return None
    return createRTVKey(vid, rids.keys())

INPUT_PARAMETERS_AlonsoMoraAssignment = {
    "doc" :  """This algorithm is a variant of the publication
                On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment; Alonso-Mora, Javier; Samaranayake, Samitha; Wallar, Alex; Frazzoli, Emilio; Rus, Daniela (2017)
                the differences are described in
                Speed-up Heuristic for an On-Demand Ride-Pooling Algorithm; Engelhardt, Roman; Dandl, Florian; Bogenberger, Klaus (2020) https://arxiv.org/pdf/2007.14877 """,
    "inherit" : "BatchAssignmentAlgorithmBase",
    "input_parameters_mandatory": [G_RA_SOLVER],
    "input_parameters_optional": [
        G_RA_TB_TO_PER_VEH, G_RA_MAX_VR, G_RA_OPT_TO, G_RA_HEU, G_RVH_B_DIR, G_RVH_DIR, G_RVH_B_LWL, G_RVH_LWL, G_RVH_AM_RR, G_RVH_AM_TI
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class AlonsoMoraAssignment(BatchAssignmentAlgorithmBase):
    def __init__(self, fleetcontrol: FleetControlBase, routing_engine : NetworkBase, sim_time : int, obj_function : Callable, operator_attributes : dict, 
                 optimisation_cores : int = 1, seed : int = 6061992, veh_objs_to_build : Dict[int, SimulationVehicle] = {}):
        """This class is used to compute new vehicle assignments with Alonso-Mora-Algorithm
        this class should be initialized when the corresponding fleet controller is initialized
        if parallelization should be enabled, a alonso-mora-parallelization-manager object need to be given (AlonsoMoraParallelization.py)
        :param fleetcontrol : fleetcontrol object, which uses this assignment algorithm
        :param routing_engine : routing_engine object
        :param sim_time : (int) current simulation time
        :param obj_function: control function to rate vehicle plans
        :param operator_attributes: operator attribute dictionary
        :param veh_objs_to_build : (dict) vid -> veh_obj new routes are only build for theses vehicles, if none given all vehicles are used from fleetcontrol (needed for parallel computations)
        :param optimisation_cores: number of cpus/threads used for solving the ILP
        :param seed: random seed
        """
        super().__init__(fleetcontrol, routing_engine, sim_time, obj_function, operator_attributes, optimisation_cores=optimisation_cores, seed=seed, veh_objs_to_build=veh_objs_to_build)

        # implemented heuristics: # TODO # adapt description
        # "after_opt_rv_best_v2rb"    -> defined by op_max_rv_connections
        # "before_opt_rv_insertion"   -> defined by op_max_rv_connections
        # "before_opt_rv_mix"         -> dict input needed with "-" seperated int string (2 values before_opt_insertion first, then before_opt_nearest)
        # "before_opt_rv_nearest_rr"  -> defined by op_max_rv_connections
        # "single_plan_per_v2rb"      -> no parameter needed

        self.veh_tree_build_timeout : int = operator_attributes.get(G_RA_TB_TO_PER_VEH, None)
        self.optimisation_timeout : int = operator_attributes.get(G_RA_OPT_TO, None)
        self.max_rv_connections : int = operator_attributes.get(G_RA_MAX_VR, None)
        applied_heuristics = operator_attributes.get(G_RA_HEU, None)
        
        self.applied_heuristics = {}
        if applied_heuristics is not None:
            if type(applied_heuristics) == dict:
                self.applied_heuristics = applied_heuristics
                for name, values in self.applied_heuristics.items():
                    if type(values) == str:
                        values = [int(v) for v in values.split("-")]
                        self.applied_heuristics[name] = values
            elif type(applied_heuristics) == str:
                self.applied_heuristics = {applied_heuristics : 1}
            else:
                LOG.error("INVALID TYPE INPUT FOR HEURISTIC: NEITHER STR NOR DICT {}".format(applied_heuristics))
                raise EnvironmentError

        self.alonso_mora_parallelization_manager : ParallelizationManager = None # will be set in fleetcontrol

        self.vid_to_list_passed_VRLs : Dict[int, List[VehicleRouteLeg]] = {}

        self.rtv_obj : Dict[Any, V2RB] = {}                   # rtv_key -> V2RB-Object
        self.rtv_costs : Dict[Any, float] = {}                 # rtv_key -> cost-function-value
        self.current_assignments : Dict[Any, int] = {}       # vid -> rtv_key
        self.init_sol = {}
        self.optimisation_solutions : Dict[int, tuple] = {}    # vid -> rtv_key computed when solving optimisation problem
        self.active_requests : Dict[int, PlanRequest] = {}           # rid -> request-Object
        #
        self.new_requests : Dict[Any, 1] = {}  # rid -> 1
        self.unassigned_requests : Dict[Any, 1] = {}   # rid -> 1
        self.requests_to_remove : Dict[Any, 1] = {}    # rid -> 1
        #
        self.rtv_tree_N_v : Dict[int, Dict[int, Dict[tuple, 1]]] = {}         #vid -> number_of_requests -> rtv_key -> 1
        self.rtv_v : Dict[int, Dict[tuple, 1]] = {}         # vid -> {}: rtv_key -> 1
        self.rtv_r : Dict[Any, Dict[tuple, 1]] = {}         #rid -> {}: rtv_key -> 1
        self.v2r : Dict[int, Dict[Any, 1]] = {}            #vid -> rid -> 1 | generally available request-to-vehicle combinations
        self.v2r_locked : Dict[int, Dict[Any, 1]] = {}    #vid -> rid -> 1 | rids currently locked to vid
        self.r2v_locked : Dict[Any, Dict[int, 1]] = {}
        self.r2v : Dict[Any, Dict[int, 1]] = {}           #rid -> vid -> 1 | generally available request-to-vehicle combinations
        self.rr : Dict[tuple, 1] = {}            #rid1 -> rid2 -> 1/None | generally available 
        self.rebuild_rtv : Dict[int, 1] = {}   #vid -> 1 | vid is added if a request enters/leaves this vehicle --> flag to rebuild rtv-tree of vehicle from scratch
        self.requests_to_compute : Dict[Any, 1] = {}
        self.requests_to_compute_in_next_step : Dict[Any, 1] = {}
        # creation of permanent vehicle keys
        for vid in self.veh_objs.keys():
            self.rtv_v[vid] = {}
            self.rtv_tree_N_v[vid] = {}
            self.v2r[vid] = {}
            self.v2r_locked[vid] = {}

        self.external_assignments : Dict[int, Tuple[tuple, VehiclePlan]] = {} # assignments made outside of this algorithm
        self.untracked_boarding_detected = {}

        self.current_best_cfv = 0 #stores the global cost function value from last optimisation

    def register_parallelization_manager(self, alonsomora_parallelization_manager : ParallelizationManager):
        LOG.info("AM register parallelization manager")
        self.alonso_mora_parallelization_manager = alonsomora_parallelization_manager
        self.alonso_mora_parallelization_manager.initOp(self.fo_id, self.objective_function, self.operator_attributes)
        self.optimisation_cores = self.alonso_mora_parallelization_manager.number_cores

    def compute_new_vehicle_assignments(self, sim_time : int, vid_to_list_passed_VRLs : Dict[int, List[VehicleRouteLeg]], 
                                     veh_objs_to_build : Dict[int, SimulationVehicle] = {}, new_travel_times : bool = False, build_from_scratch : bool = False):
        """ this function computes new vehicle assignments based on current fleet information
        :param sim_time : current simulation time
        :param vid_to_list_passed_VRLs : (dict) vid -> list_passed_VRLs; needed to update database and V2RBs
        :param veh_objs_to_build: only these vehicles will be optimized (all if empty) dict vid -> SimVehicle obj
        :param new_travel_times : bool; if True, the database will be recomputed from scratch
        :param build_from_scratch : bool; if True, the whole database will be cleared and recomputed from scratch
        """
        t_start = time.time()
        if self.fleetcontrol is not None:
            for vid, ext_tuple in self.external_assignments.items():
                if ext_tuple[1] is None:
                    ext_veh_plan = self.fleetcontrol.veh_plans[vid]
                    rtv_key = getRTVkeyFromVehPlan(ext_veh_plan)
                    self.external_assignments[vid] = (rtv_key, ext_veh_plan)
            LOG.debug("external assignments : {}".format({x: (str(y[0]), str(y[1])) for x, y in self.external_assignments.items()}) )
        self.veh_objs = {}
        if len(veh_objs_to_build.keys()) == 0:
            for veh_obj in self.fleetcontrol.sim_vehicles:
                veh_obj_struct = SimulationVehicleStruct(veh_obj, self.fleetcontrol.veh_plans.get(veh_obj.vid, VehiclePlan(veh_obj, self.sim_time, self.routing_engine, [])), sim_time, self.routing_engine)
                self.veh_objs[veh_obj.vid] = veh_obj_struct
        else:
            self.veh_objs = veh_objs_to_build
        if build_from_scratch or new_travel_times:
            self.requests_to_compute = {rid:1 for rid in self.rid_to_consider_for_global_optimisation.keys()}
            self._clearV2RBDataBase()
        # # LOG.debug("setCurrentInformation")
        # # LOG.debug(f" -> t {sim_time} | vid list passt VRLS {vid_to_list_passed_VRLs}")
        # # LOG.debug(f" requests to compute: {self.requests_to_compute.keys()}")
        self._setUpCurrentInformation(sim_time, vid_to_list_passed_VRLs, new_travel_times=new_travel_times)
        ## LOG.debug("updateRequestDicts")
        self._updateRequestDicts(build_from_scratch = build_from_scratch)
        # # LOG.debug("ACTIVE REQUESTS:")
        # for rid, prq in self.active_requests.items():
        #     # LOG.debug("rid: {}".format(prq))
        # LOG.info("computeRR")
        t_setup = time.time()
        self._computeRR()
        t_rr = time.time()
        ## LOG.debug(f"new RR cons {self.rr}")
        # LOG.info("computeRV")
        self._computeRV()
        t_rv = time.time()
        ## LOG.debug(f"new RV cons {self.r2v}")
        if self.veh_tree_build_timeout is not None:
            self._set_init_solution_insertion()
       # # LOG.debug(f"check for untracked boardings")
        for vid in self.untracked_boarding_detected.keys():
            ## LOG.debug(f"untracked boarding for vid {vid}")
            assigned_key = self.current_assignments[vid]
            if not self.rtv_obj.get(assigned_key):
                ## LOG.debug(f"create v2rb {assigned_key}")
                assigned_plan = self.external_assignments[vid][1]
                assigned_v2rb = V2RB(self.routing_engine, self.active_requests, sim_time, assigned_key, self.veh_objs[vid], self.std_bt, self.add_bt, self.objective_function, orig_veh_plans=[assigned_plan])
                self._addRtvKey(assigned_key, assigned_v2rb)
        # LOG.info("build V2RBs")
        ## LOG.debug(f"old v2rbs {self.rtv_costs}")
        self._computeV2RBdatabase()
        t_build = time.time()
        ## LOG.debug(f"new v2rbs {self.rtv_costs}")
        # LOG.info("optimise")
        self._runOptimisation()
        t_opt = time.time()
        ## LOG.debug(f"after optimisation {self.optimisation_solutions}")
        times = {"sim_time" : self.sim_time, "setup" : t_setup - t_start, "rr" : t_rr - t_setup, "rv" : t_rv - t_rr, "build" : t_build - t_rv, "opt" : t_opt - t_build}
        #time_line = ["{};{}".format(x, y) for x,y in times.items()]
        # LOG.info("opt_times:{}".format(",".join(time_line)))
        if self.max_rv_connections is not None and self.applied_heuristics.get("after_opt_rv_best_v2rb"):
            self._after_opt_rv_best_v2rb_heuristic()
        t_opt = time.time()
        # LOG.debug(f"after optimisation {self.optimisation_solutions}")
        times = {"sim_time" : self.sim_time, "setup" : t_setup - t_start, "rr" : t_rr - t_setup, "rv" : t_rv - t_rr, "build" : t_build - t_rv, "opt" : t_opt - t_build, "all" : t_opt - t_start}
        time_str = ",".join(["{};{}".format(a, b) for a, b in times.items()])
        LOG.info("OPT TIMES:{}".format(time_str))
        LOG.info("Opt stats at sim time {} : opt duration {} | res cfv {}".format(self.sim_time, t_opt - t_start, self.current_best_cfv))

    def add_new_request(self, rid : Any, prq : PlanRequest, consider_for_global_optimisation : bool = True, is_allready_assigned : bool = False):
        """ this function adds a new request to the modules database and set entries that
        possible v2rbs are going to be computed in the next opt step.
        :param rid: plan_request_id
        :param prq: plan_request_obj 
        :param consider_for_global_optimisation: if false, it will not be looked for better solutions in global optimisation
                    but it is still part of the solution, if it is allready assigned
        :param is_allready_assigned: if not considered for global optimisation, this flag indicates, if the rid is allready assigned
            in the init solution"""
        if consider_for_global_optimisation:
            self.new_requests[rid] = 1
            self.requests_to_compute[rid] = 1
        if self.alonso_mora_parallelization_manager is not None:
            self.alonso_mora_parallelization_manager.add_new_request(self.fo_id, rid, prq, consider_for_global_optimisation = consider_for_global_optimisation)
        super().add_new_request(rid, prq, consider_for_global_optimisation=consider_for_global_optimisation, is_allready_assigned=is_allready_assigned)

    def set_mutually_exclusive_assignment_constraint(self, list_sub_rids : list, base_rid : Any):
        if self.alonso_mora_parallelization_manager is not None:
            self.alonso_mora_parallelization_manager.set_mutually_exclusive_assignment_constraint(self.fo_id, list_sub_rids, base_rid)
        return super().set_mutually_exclusive_assignment_constraint(list_sub_rids, base_rid)

    def set_request_assigned(self, rid : Any):
        """ this function marks a request as assigned. its assignment is therefor treatet as hard constraint in the optimization problem formulation
        also all requests with the same mutually_exclusive_cluster_id are set as assigned
        :param rid: plan_request_id """
        super().set_request_assigned(rid)

    def set_database_in_case_of_boarding(self, rid : Any, vid : int):
        """ deletes all rtvs without rid from vid (rid boarded to vid)
        deletes all rtvs with rid for all other vehicles 
        triggers database entries for boarding process
        :param rid: plan_request_id
        :param vid: vehicle obj id """
        LOG.debug(f"set_database_in_case_of_boarding rid {rid} -> vid {vid}")
        # LOG.debug(f"rtv_r[rid] : {self.rtv_r.get(rid, {})}")
        # LOG.debug(f"rtv_v[vid] : {self.rtv_v.get(vid, {})}")
        super().set_database_in_case_of_boarding(rid, vid)
        sub_rids = self._get_all_rids_representing_this_base_rid(rid)
        for sub_rid in sub_rids:
            for rtv_key in list(self.rtv_r.get(sub_rid,{}).keys()):
                if not self.rtv_v[vid].get(rtv_key):
                    # # LOG.debug(f"-> delete {rtv_key}")
                    self._delRtvKey(rtv_key)
        for rtv_key in list(self.rtv_v[vid].keys()):
            to_del = True
            for sub_rid in sub_rids:
                if self.rtv_r.get(sub_rid,{}).get(rtv_key) is not None:
                    to_del = False
                    break
            if to_del:
                self._delRtvKey(rtv_key)
        for sub_rid in sub_rids:
            for vid_o in list(self.r2v.get(sub_rid,{}).keys()):
                if vid == vid_o:
                    continue
                else:
                    try:
                        del self.r2v[sub_rid][vid_o]
                    except:
                        pass
                    try:
                        del self.v2r[vid_o][sub_rid]
                    except:
                        pass
        for sub_rid in sub_rids:
            if self.new_requests.get(sub_rid) is not None:
                self.rebuild_rtv[vid] = 1
                self.untracked_boarding_detected[vid] = 1


    def set_database_in_case_of_alighting(self, rid : Any, vid : int):
        """ this function deletes all database entries of rid and sets the new assignement by deleting rid of the currently
        assigned v2rb; the database of vid will be completely rebuild in the next opt step
        this function should be called in the fleet operators acknowledge alighting in case rid alights vid
        :param rid: plan_request_id
        :param vid: vehicle obj id """
        LOG.debug(f"set_database_in_case_of_alighting rid {rid} -> vid {vid}")

        self.rebuild_rtv[vid] = 1
        
        previous_rtv_key = self.current_assignments.get(vid)
        prev_v2rb_obj = None
        new_rtv_key = None
        LOG.debug(f"prev key: {previous_rtv_key}")
        sub_rids = self._get_all_rids_representing_this_base_rid(rid)
        if previous_rtv_key is not None:
            assigned_rid = None
            if len(sub_rids) > 1:
                assigned_rids = getRidsFromRTVKey(previous_rtv_key)
                for ass_rid in assigned_rids:
                    if ass_rid in sub_rids:
                        assigned_rid = ass_rid
                        break
            else:
                assigned_rid = list(sub_rids)[0]
            prev_v2rb_obj = self.rtv_obj.get(previous_rtv_key)
            LOG.debug("remove {} from key {}".format(assigned_rid, previous_rtv_key))
            new_rtv_key = deleteRidFromRtv(assigned_rid, previous_rtv_key)
            
            self.current_assignments[vid] = new_rtv_key
            # LOG.debug(f"prev v2rb {prev_v2rb_obj}")
            # LOG.debug(f"new key: {new_rtv_key}")
            
        for rtv_key in list(self.rtv_v.get(vid,{}).keys()):
            self._delRtvKey(rtv_key)
            
        # LOG.debug("prev_v2rb {}".format(prev_v2rb_obj))
        if new_rtv_key and prev_v2rb_obj:
            prev_v2rb_obj.rtv_key = new_rtv_key
            self._addRtvKey(new_rtv_key, prev_v2rb_obj)

        super().set_database_in_case_of_alighting(rid, vid)

    def _getV2RBobjList(self) -> List[V2RB]:
        """ returns a list of all v2rb_objs in the database
        this function is only used in AlonosoMoraParallelization.py and therefore considered as private
        :return: list of all V2RB-objects in database
        """
        return list(self.rtv_obj.values())

    def get_optimisation_solution(self, vid : int) -> VehiclePlan:
        """ returns the assigned vehicle plan from the last optimisation step
        :param vid: vehicle id
        :return: vehicle plan object for the corresponding vehicle
        """
        # # LOG.debug("get Optimisation solution: {} from {}".format(vid, self.optimisation_solutions))
        key = self.optimisation_solutions.get(vid)
        # LOG.debug("veh obj {}".format(self.veh_objs[vid]))
        #minimal_vehplan = VehiclePlan(self.veh_objs[vid], self.sim_time, self.routing_engine, self.veh_objs[vid].locked_planstops)
        if key is None:
            if self.veh_objs[vid].has_locked_vehplan() > 0:
                return self.veh_objs[vid].locked_vehplan
            else:
                return None
        else:
            veh_plan = self.rtv_obj[key].getBestPlan()
            return veh_plan.copy()

    def set_assignment(self, vid : int, assigned_plan : VehiclePlan, is_external_vehicle_plan : bool = False, _is_init_sol: bool = False):
        """ sets the vehicleplan as assigned in the algorithm database; if the plan is not computed within the this algorithm, the is_external_vehicle_plan flag should be set to true
        :param vid: vehicle id
        :param assigned_plan: vehicle plan object that has been assigned
        :param is_external_vehicle_plan: should be set to True, if the assigned_plan has not been computed within this algorithm
        :param _is_init_sol: used within the code, if the init solution creater set this solution 
        """
        super().set_assignment(vid, assigned_plan, is_external_vehicle_plan=is_external_vehicle_plan)
        if assigned_plan is None:
            self.current_assignments[vid] = None
        else:
            rtv_key = getRTVkeyFromVehPlan(assigned_plan)
            self.current_assignments[vid] = rtv_key
            LOG.debug(f"assign {vid} -> {rtv_key} | is external? {is_external_vehicle_plan}")
            if is_external_vehicle_plan and not _is_init_sol:
                self.external_assignments[vid] = (rtv_key, None)
                self.rebuild_rtv[vid] = 1
                self.delete_vehicle_database_entries(vid)
            elif _is_init_sol and not is_external_vehicle_plan:
                self.external_assignments[vid] = (rtv_key, assigned_plan)

    def get_current_assignment(self, vid : int) -> VehiclePlan:
        """ returns the vehicle plan assigned to vid currently
        :param vid: vehicle id
        :return: vehicle plan
        """
        currently_assigned_key = self.current_assignments.get(vid)
        if currently_assigned_key is not None:
            assigned_rtv_obj = self.rtv_obj.get(currently_assigned_key)
            if assigned_rtv_obj is not None:
                return assigned_rtv_obj.getBestPlan()
            else:
                return self.fleetcontrol.veh_plans[vid]
        else:
            return None

    def clear_databases(self):
        """ this function resets database entries after an optimisation step
        """
        self.new_requests = {}
        self.optimisation_solutions = {}
        self.rebuild_rtv = {}
        self.requests_to_compute = self.requests_to_compute_in_next_step.copy()
        self.requests_to_compute_in_next_step = {}
        self.external_assignments = {}
        self.untracked_boarding_detected = {}

    def delete_request(self, rid : Any):
        """ this function deletes a request from all databases
        :param rid: plan_request_id """
        for rtv_key in list(self.rtv_r.get(rid, {}).keys()):
            self._delRtvKey(rtv_key)
        for vid in self.r2v.get(rid, {}):
            del self.v2r[vid][rid]
        try:
            del self.r2v[rid]
        except:
            pass
        try:
            del self.rtv_r[rid]
        except:
            pass
        self._delRRcons(rid)
        try:
            del self.new_requests[rid]
        except:
            pass
        try:
            del self.requests_to_compute[rid]
        except:
            pass

        if self.alonso_mora_parallelization_manager is not None:
            self.alonso_mora_parallelization_manager.delete_request(self.fo_id, rid)

        super().delete_request(rid)

    def delete_vehicle_database_entries(self, vid : int):
        # LOG.debug("delete all rtv_entries for vid {}".format(vid))
        for rtv_key in list(self.rtv_v.get(vid, {}).keys()):
            self._delRtvKey(rtv_key)

    def lock_request_to_vehicle(self, rid : Any, vid : int):
        """locks the request to the assigned vehicle"""
        # LOG.info("locking rid to vid {} {}".format(rid, vid))
        super().lock_request_to_vehicle(rid, vid)
        sub_rids = self._get_all_rids_representing_this_base_rid(rid)
        for sub_rid in sub_rids:
            for rtv_key in list(self.rtv_r.get(sub_rid,{}).keys()):
                if not self.rtv_v[vid].get(rtv_key):
                    # # LOG.debug(f"-> delete {rtv_key}")
                    self._delRtvKey(rtv_key)
        for rtv_key in list(self.rtv_v[vid].keys()):
            to_del = True
            for sub_rid in sub_rids:
                if self.rtv_r.get(sub_rid,{}).get(rtv_key) is not None:
                    to_del = False
                    break
            if to_del:
                self._delRtvKey(rtv_key)
        for sub_rid in sub_rids:
            for vid_o in list(self.r2v.get(sub_rid,{}).keys()):
                if vid == vid_o:
                    continue
                else:
                    try:
                        del self.r2v[sub_rid][vid_o]
                    except:
                        pass
                    try:
                        del self.v2r[vid_o][sub_rid]
                    except:
                        pass
        for sub_rid in sub_rids:
            if self.new_requests.get(sub_rid) is not None:
                self.rebuild_rtv[vid] = 1

    def register_change_in_time_constraints(self, rid : Any, prq : PlanRequest, assigned_vid : int = None, exceeds_former_time_windows : bool = True):
        """ if time constraints on an requests changed the corresponding plan stop constraints have to be updated in the v2rbs
        if the new time constraints exceed former time constraints, the tree for the rids has to be rebuilded
        :param rid: request id
        :param prq: plan request obj
        :param assigned_vid: vehicle id, which is currently assigned to serve customer (none if none assigned)
        :param exceeds_former_time_windows: True: new time window is larger the old one, False otherwise
        """
        if not exceeds_former_time_windows:
            LOG.info("exceeds former time window alternative not implemented yet! rebuild!")
            exceeds_former_time_windows = True
        if exceeds_former_time_windows:
            v2rbs_2_keep = {}
            if assigned_vid is not None:
                assigned_key = self.current_assignments[assigned_vid]
                v2rbs_2_keep[assigned_key] = self.rtv_obj[assigned_key]
            for rtv_key in list(self.rtv_r.get(rid, {}).keys()):
                self._delRtvKey(rtv_key)
            for vid in self.r2v.get(rid, {}):
                self.rebuild_rtv[vid] = 1
                del self.v2r[vid][rid]
            try:
                del self.r2v[rid]
            except:
                pass
            try:
                del self.rtv_r[rid]
            except:
                pass
            # LOG.verbose("keep v2rbs {}".format(v2rbs_2_keep.keys()))
            for key, v2rb in v2rbs_2_keep.items():
                self._addRtvKey(key, v2rb)
            self.requests_to_compute[rid] = 1
            self.active_requests[rid] = prq         

    def get_vehicle_plan_without_rid(self, veh_obj : SimulationVehicle, vehicle_plan : VehiclePlan, rid_to_remove : Any, sim_time : int) -> VehiclePlan:
        """this function returns the best vehicle plan by removing the rid_to_remove from the vehicle plan
        :param veh_obj: corresponding vehicle obj
        :param vehicle_plan: vehicle_plan where rid_remove is included
        :param rid_to_remove: request_id that should be removed from the rtv_key
        :param sim_time: current simulation time
        :return: best_veh_plan if vehicle_plan rid_to_remove is part of vehicle_plan, None else
        """
        rtv_key = getRTVkeyFromVehPlan(vehicle_plan)
        new_rtv_key = deleteRidFromRtv(rid_to_remove, rtv_key)
        if new_rtv_key is None:
            return None
        new_v2rb = self.rtv_obj.get(new_rtv_key)
        if new_v2rb is not None:
            return new_v2rb.getBestPlan()
        else:
            LOG.warning(f"lower V2RB {new_rtv_key} by removing {rid_to_remove} not found -> build lower v2rb from {rtv_key}")
            old_v2rb = self.rtv_obj.get(rtv_key)
            if old_v2rb is not None:
                vid = getVidFromRTVKey(rtv_key)
                new_v2rb = old_v2rb.createLowerV2RB(new_rtv_key, self.sim_time, self.routing_engine, self.objective_function, self.active_requests, self.std_bt, self.add_bt)
                self._addRtvKey(new_rtv_key, new_v2rb)
                return new_v2rb.getBestPlan()
            else:
                new_veh_plan = simple_remove(veh_obj, vehicle_plan, rid_to_remove, sim_time, self.routing_engine, self.objective_function, self.active_requests, self.std_bt, self.add_bt)
                return new_veh_plan

    def _delRRcons(self, rid : Any, rid2 : Any = None):
        """ this function deletes rr-connections from the database
        if only one request_id is given, every rr-connection of this rid is deleted
        else only the specific one is deleted
        :param rid: plan_request_id
        :param rid2: (optional) other plan_request_id"""
        if rid2 is not None:
            rr_key = getRRKey(rid, rid2)
        else:
            for rid2 in self.active_requests.keys():
                rr_key = getRRKey(rid, rid2)
                if self.rr.get(rr_key) is not None:
                    del self.rr[rr_key]

    def _setUpCurrentInformation(self, sim_time : int, vid_to_list_passed_VRLs : Dict[int, List[VehicleRouteLeg]], new_travel_times : bool = False):
        """ this function processes information like new sim_time, boarding and alighting processes and defines which vehicle_databases need to be recomputed from scratch:
        if new_travel_times: every; else vehicles where someone deboareded the vehicle
        acknowledges boarding and alighting processes for the databases
        param sim_time : current simulation time 
        param vid_to_list_passed_VRLs : (dict) vid -> list_passed_VRLs; needed to update database and V2RBs
        param new_travel_times : bool; if True, the database will be recomputed from scratch
        return None
        """
        self.sim_time = sim_time
        self.vid_to_list_passed_VRLs = {}
        for vid, passed_VRLs in vid_to_list_passed_VRLs.items(): # remove stationary process
            new_passed_VRLs = [VehicleRouteLeg(x.status, x.destination_pos, x.rq_dict, power=x.power, duration = x.duration, route=x.route, locked=x.locked, earliest_start_time=x.earliest_start_time)
                               for x in passed_VRLs]
            self.vid_to_list_passed_VRLs[vid] = new_passed_VRLs
        if new_travel_times:
            self.rebuild_rtv = {vid : 1 for vid in self.veh_objs.keys()}


    def _updateRequestDicts(self, build_from_scratch : bool = False): # TODO #
        """ this function processes new request information and distributes them to the parallel processes if available"""
        if self.alonso_mora_parallelization_manager is None:
            if not build_from_scratch:
                return
            else:
                LOG.warning("here is a todo i dont remember why")   #TODO
        else:
            self.alonso_mora_parallelization_manager.setDynSimtimeAndActiveRequests(self.fo_id, self.sim_time)
            #self.alonso_mora_parallelization_manager.setSimtimeAndActiveRequests(self.fo_id, self.sim_time, self.active_requests, self.rid_to_mutually_exclusive_cluster_id, self.rid_to_consider_for_global_optimisation)

    def _addRtvKey(self, rtv_key : tuple, rtv_obj : V2RB):
        """this function adds entries to all necessery database dictionaries
        param rtv_key : key of rtv_obj
        param rtv_obj : rtv_obj/V2RB (look V2RB.py)
        return : None
        """
        LOG.debug(f"add rtv key {rtv_key} | {rtv_obj}")
        self.rtv_obj[rtv_key] = rtv_obj
        self.rtv_costs[rtv_key] = rtv_obj.cost_function_value
        list_rids = getRidsFromRTVKey(rtv_key)
        vid = getVidFromRTVKey(rtv_key)
        number_rids = len(list_rids)
        try:
            self.rtv_tree_N_v[vid][number_rids][rtv_key] = 1
        except:
            self.rtv_tree_N_v[vid][number_rids] = {rtv_key : 1}
        self.rtv_v[vid][rtv_key] = 1
        for rid in list_rids:
            if not self.rtv_r.get(rid):
                self.rtv_r[rid] = {}
            self.rtv_r[rid][rtv_key] = 1
        # # LOG.debug(f"rtv_tree: {self.rtv_tree_N_v}")

    def _updateV2RBcostInDataBase(self, rtv_key : tuple, v2rb : V2RB):
        if self.rtv_costs.get(rtv_key) is None:
            raise AssertionError(f"False use of this function! {rtv_key} is not in database yet!")
        self.rtv_obj[rtv_key] = v2rb
        self.rtv_costs[rtv_key] = v2rb.cost_function_value

    def _delRtvKey(self, rtv_key : tuple):
        """deletes rtv_key and v2rb-obj from all dicts and lists in database
        param rtv_key : key of rtv_obj
        """
        # # LOG.debug(f"del rtv key {rtv_key}")
        rid_list = getRidsFromRTVKey(rtv_key)
        vid = getVidFromRTVKey(rtv_key)
        for rid in rid_list:
            del self.rtv_r[rid][rtv_key]
        del self.rtv_tree_N_v[vid][len(rid_list)][rtv_key]
        del self.rtv_v[vid][rtv_key]
        del self.rtv_obj[rtv_key]
        del self.rtv_costs[rtv_key]

    def _clearV2RBDataBase(self, also_rr_and_rv : bool = True):
        """ this function resets the whole database in case of a rebuild
        """
        to_keep = []
        for vid, assigned_key in self.current_assignments.items():
            if assigned_key is not None:
                if self.rtv_obj.get(assigned_key):
                    to_keep.append( (vid, assigned_key, self.rtv_obj[assigned_key]))
        self.rtv_tree_N_v = {}         #vid -> number_of_requests -> rtv_key -> 1
        self.rtv_v = {}         # vid -> {}: rtv_key -> 1
        self.rtv_r = {}         #rid -> {}: rtv_key -> 1
        self.rtv_costs = {}
        self.rtv_obj = {}
        #
        # creation of permanent vehicle keys
        for vid in self.veh_objs.keys():
            self.rtv_v[vid] = {}
            self.rtv_tree_N_v[vid] = {}

        if also_rr_and_rv:
            self.v2r = {}            #vid -> rid -> 1 | generally available request-to-vehicle combinations
            self.r2v = {}           #rid -> vid -> 1 | generally available request-to-vehicle combinations
            self.rr = {}            #rid1 -> rid2 -> 1/None | generally available 
            for vid in self.veh_objs.keys():
                self.v2r[vid] = {}

        for vid, assigned_key, assigned_v2rb in to_keep:
            self._addRtvKey(assigned_key, assigned_v2rb)

    def _computeRR(self):
        """ this function computes all rr-connections from self.requests_to_compute with all active_requests
        """
        if not self.alonso_mora_parallelization_manager:
            for rid1 in self.requests_to_compute.keys():
                rq1 = self.active_requests[rid1]
                for rid2 in self.rid_to_consider_for_global_optimisation.keys():
                    if rid1 != rid2:
                        if not self._is_subrid(rid1) or not self._is_subrid(rid2) or self._get_associated_baserid(rid1) != self._get_associated_baserid(rid2):
                            rq2 = self.active_requests[rid2]
                            rr_comp = checkRRcomptibility(rq1, rq2, self.routing_engine, self.std_bt, dynamic_boarding_time=self.add_bt) #TODO definitions of boarding times!
                            if rr_comp:
                                self.rr[getRRKey(rid1, rid2)] = 1
        else:
            self.alonso_mora_parallelization_manager.computeRR(self.fo_id, self.requests_to_compute.keys())
            new_rr_entries = self.alonso_mora_parallelization_manager.fetch_computeRR()
            self.rr.update(new_rr_entries)
            self.alonso_mora_parallelization_manager.setRR(self.fo_id, new_rr_entries)
            # LOG.verbose(f"new rr entries: {new_rr_entries}")

    def _computeRV(self):
        """ this function computes all rv-connections from self.requests_to_compute with all active vehicles
        """
        veh_locations_to_vid = {}
        ca_rv = {}
        for vid, veh_obj in self.veh_objs.items():
            try:
                veh_locations_to_vid[veh_obj.pos].append(vid)
            except:
                veh_locations_to_vid[veh_obj.pos] = [vid]
        current_time = self.sim_time
        # # LOG.debug(f"compute RV: veh_locations_to_vid {veh_locations_to_vid}; to compute {self.requests_to_compute}")
        vid_dict = {}  # vid -> tt
        # prepare travel times in parallel processing
        if self.alonso_mora_parallelization_manager:
            rids_to_compute_to_rq = {rid: self.active_requests[rid] for rid in self.requests_to_compute.keys()}
            new_rv, travel_infos = self.alonso_mora_parallelization_manager.computeRV(veh_locations_to_vid,
                                                                                      rids_to_compute_to_rq,
                                                                                      self.sim_time)
            self.routing_engine.add_travel_infos_to_database(travel_infos)
        else:
            # for completeness of code
            new_rv = {}
            travel_infos = {}
            rids_to_compute_to_rq = {}
        for rid in self.requests_to_compute.keys():
            prq = self.active_requests[rid]
            if not self.alonso_mora_parallelization_manager:
                # get routing results in single processing
                o_pos, _, latest_pu = prq.get_o_stop_info()
                routing_results = self.routing_engine.return_travel_costs_Xto1(veh_locations_to_vid.keys(), o_pos,
                                                                               max_cost_value=latest_pu - current_time)
                for veh_loc, tt, _, _ in routing_results:
                    for vid in veh_locations_to_vid[veh_loc]:
                        vid_dict[vid] = tt
            else:
                # get prepared routing results in multi processing
                for vid in new_rv.get(rid, []):
                    veh_pos = self.veh_objs[vid].pos
                    rid_org = rids_to_compute_to_rq[rid].get_o_stop_info()[0]
                    tt = travel_infos[(veh_pos, rid_org)][0]
                    vid_dict[vid] = tt
            if self.fleetcontrol is not None:    # possible TODO for parallelization#
                # new structure for heuristics
                # ----------------------------
                # -> goal: try to keep some kind of consistency with /pooling/immediate/insertion
                # 1) pre-insertion vehicle-selection processes
                # generally available heuristics
                #   x) currently assigned vehicle [cannot be deselected!]
                #   a) G_RVH_DIR: directionality of currently assigned route compared to vector of prq origin-destination
                #   b) G_RVH_LWL: selection of vehicles with least workload
                # only for Alonso-Mora
                #   c) G_RVH_AM_RR: only if all rr-connections to assigned requests hold (starting from nearest vehicle)
                #   d) G_RVH_AM_TI: only keep best vehicles after testing insertions in currently assigned plan
                currently_assigned_veh = self.current_assignments
                heuristic_adopt_dict = {}
                rv_vehicles = [self.veh_objs[vid] for vid in vid_dict.keys()]
                # x) currently assigned vehicle
                ca_flag = False
                if self.fleetcontrol.rid_to_assigned_vid.get(rid) is not None:
                    veh_obj = self.veh_objs[self.fleetcontrol.rid_to_assigned_vid[rid]]
                    selected_veh = {veh_obj}
                    ca_flag = True
                else:
                    selected_veh = set([])
                # a) directionality of currently assigned route compared to vector of prq origin-destination
                number_directionality = self.fleetcontrol.rv_heuristics.get(G_RVH_B_DIR, None)
                if number_directionality is None:
                    number_directionality = self.fleetcontrol.rv_heuristics.get(G_RVH_DIR, 0)
                if number_directionality > 0:
                    veh_dir = filter_directionality(prq, rv_vehicles, number_directionality, self.routing_engine,
                                                    selected_veh)
                    for veh_obj in veh_dir:
                        selected_veh.add(veh_obj)
                # b) selection of vehicles with least workload
                number_least_load = self.fleetcontrol.rv_heuristics.get(G_RVH_B_LWL, None)
                if number_least_load is None:
                    number_least_load = self.fleetcontrol.rv_heuristics.get(G_RVH_LWL, 0)
                if number_least_load > 0:
                    veh_ll = filter_least_number_tasks(rv_vehicles, number_least_load, selected_veh)
                    for veh_obj in veh_ll:
                        selected_veh.add(veh_obj)
                if number_directionality + number_least_load > 0:
                    rv_keep = {veh_obj.vid: 1 for veh_obj in selected_veh}
                else:
                    rv_keep = {veh_obj.vid: 1 for veh_obj in rv_vehicles}
                # c) rr-connections
                nr_add_rr = self.fleetcontrol.rv_heuristics.get(G_RVH_AM_RR, 0)
                if nr_add_rr > 0:
                    max_rv = len(rv_keep) + nr_add_rr
                    rv_keep, _ = \
                        self._before_opt_rv_nearest_with_rr_heuristic(rid, vid_dict, already_vid_to_keep=rv_keep,
                                                                    prev_plans={}, max_rv_connections=max_rv)
                # d) testing insertions
                nr_add_ti = self.fleetcontrol.rv_heuristics.get(G_RVH_AM_TI, 0)
                if nr_add_ti > 0:
                    max_rv = len(rv_keep) + nr_add_ti
                    rv_keep, heuristic_adopt_dict = \
                        self._before_opt_rv_best_insertion_heuristic(rid, vid_dict, already_vid_to_keep=rv_keep,
                                                                    prev_plans={}, max_rv_connections=max_rv)
                # if heuristics were applied successfully, the number of vehicles in rv_keep is larger than 0
                if (ca_flag and len(rv_keep) == 1) or (not ca_flag and len(rv_keep) == 0):
                    to_keep_vids = {vid: 1 for vid in vid_dict.keys()}
                else:
                    to_keep_vids = rv_keep
            else:
                to_keep_vids = {vid: 1 for vid in vid_dict.keys()}
            for vid in to_keep_vids.keys():
                try:
                    self.r2v[rid][vid] = 1
                except:
                    self.r2v[rid] = {vid: 1}
                try:
                    self.v2r[vid][rid] = 1
                except:
                    self.v2r[vid] = {rid: 1}

    def _computeV2RBdatabase(self):
        """ this function computes the V2RB database for all vehicles """
        #if self.fleetcontrol is not None:
            # LOG.debug("alonso computeV2RBDataBase: at {}".format(self.sim_time))
            # # LOG.debug("rv connections: {}".format(self.v2r))
        if not self.alonso_mora_parallelization_manager:
            for vid, veh_obj in self.veh_objs.items():
                self._updateVehicleDataBase(vid)
                self._buildTreeForVid(vid)
        else:
            new_v2rbs_all = []
            batch_size = max(float(np.floor(len(self.veh_objs)/self.alonso_mora_parallelization_manager.number_cores/5.0)), 1)
            current_batch = []
            c = 1
            for vid, veh_obj in self.veh_objs.items():
                if not self.rebuild_rtv.get(vid):
                    vid_rids_to_compute = [rid for rid in self.requests_to_compute.keys() if self.v2r.get(vid, {}).get(rid) and self.r2v_locked.get(self._get_associated_baserid(rid), vid) == vid]
                else:
                    # LOG.debug("rebuild because of boarding in vid {}".format(vid))
                    vid_rids_to_compute = [rid for rid in self.v2r.get(vid, {}).keys() if self.rid_to_consider_for_global_optimisation.get(rid) is not None and self.r2v_locked.get(self._get_associated_baserid(rid), vid) == vid]
                    # LOG.debug("to (re) compute: {}".format(vid_rids_to_compute))
                    # LOG.debug("reference: {}".format([rid for rid in self.active_requests.keys() if self.v2r.get(vid, {}).get(rid) is not None]))
                # LOG.debug(f"build vid {vid} with rtv_keys {self.rtv_v.get(vid, {}).keys()}")
                # LOG.debug(f"all : {self.rtv_obj.keys()}")
                # LOG.debug(f"rtv_v : {self.rtv_v}")
                update_v2rbs = [self.rtv_obj[rtv_key] for rtv_key in self.rtv_v.get(vid, {}).keys()]
                #update_v2rbs_and_compute_new(self, fo_id, veh_obj, currently_assigned_key, ob_rids, rids_to_compute, vid_list_passed_VRLs = [], v2rb_list_to_be_updated = []):
                currently_assigned_key = self.current_assignments.get(vid, None)
                locked_rids = list(self.v2r_locked.get(vid, {}).keys())
                # LOG.debug(f"build parallel: vid {vid} veh_obj_vid {veh_obj.vid}")
                current_batch.append( (self.fo_id, veh_obj, currently_assigned_key, locked_rids, vid_rids_to_compute, self.vid_to_list_passed_VRLs.get(vid, []), update_v2rbs, self.external_assignments.get(vid, None)) )
                if c % batch_size == 0:
                    self.alonso_mora_parallelization_manager.batch_update_v2rbs_and_compute_new(current_batch)
                    current_batch = []
                c += 1
                #self.alonso_mora_parallelization_manager.update_v2rbs_and_compute_new(self.fo_id, veh_obj, currently_assigned_key, ob_rids, vid_rids_to_compute, vid_list_passed_VRLs = self.vid_to_list_passed_VRLs.get(vid, []), v2rb_list_to_be_updated = update_v2rbs, vid_external_assignment = self.external_assignments.get(vid, None))
            if len(current_batch) > 0:
                self.alonso_mora_parallelization_manager.batch_update_v2rbs_and_compute_new(current_batch)
            self._clearV2RBDataBase(also_rr_and_rv=False)
            new_v2rbs_all = self.alonso_mora_parallelization_manager.fetch_update_v2rbs_and_compute_new()
            #new_v2rbs_all.extend(new_v2rbs)
            for v2rb in new_v2rbs_all:
                self._addRtvKey(v2rb.rtv_key, v2rb)

    def _buildTreeForVid(self, vid : int):
        """ this method builds new V2RBS for all requests_to_compute for a single vehicle
        param vid : vehicle_id for vid to be build
        """
        assigned_key = self.current_assignments.get(vid, None)
        if assigned_key is not None:
            assigned_rids = getRidsFromRTVKey(assigned_key)
        else:
            assigned_rids = []
        if not self.rebuild_rtv.get(vid):
            rids_to_build = [rid for rid in self.requests_to_compute.keys() if self.rid_to_consider_for_global_optimisation.get(rid)]
        else:
            rids_to_build = [rid for rid in self.active_requests.keys() if self.rid_to_consider_for_global_optimisation.get(rid)]
        LOG.debug(f" ... build tree for vid {vid} with assigned key {assigned_key} for rids {rids_to_build}")
        rids_to_build_with_hierarchy = []
        for rid in rids_to_build:
            if self.r2v_locked.get(self._get_associated_baserid(rid), vid) == vid and self.v2r.get(vid, {}).get(rid):
                if rid in assigned_rids:
                    h = 1
                else: 
                    h = 0
                rids_to_build_with_hierarchy.append((rid, h))
        np.random.shuffle(rids_to_build_with_hierarchy)
        rids_to_build_with_hierarchy = sorted(rids_to_build_with_hierarchy, key = lambda x:x[1], reverse = True)
        t_all_vid = time.time()
        #LOG.debug("build tree for vid {} with rids {} | locked {}".format(vid, rids_to_build_with_hierarchy, self.r2v_locked))
        for rid, h in rids_to_build_with_hierarchy:
            t_c = time.time() - t_all_vid
            ## LOG.debug(f"build {rid} h {h}")
            if self.veh_tree_build_timeout and h != 1 and t_c > self.veh_tree_build_timeout:
                # LOG.debug(f"break tree building for vid {vid} after {t_c} s")
                break
            self._buildOnCurrentTree(vid, rid)

        self._checkForNecessaryV2RBsAndComputeMissing(vid)
        
    def _buildOnCurrentTree(self, vid : int, rid : Any):
        """This method adds rid to all currently available rtv_keys for vehicle
        IF the rid is matching with all on-board requests.
        
        To minimize the number of feasibility checks the tree is built from small bundles to large bundles with following considerations:
        1) self.rv_ob[vid] determines the lowest level of requests
        2) self.rr[vid] is considered for the lowest level
        3) the existence of lower level keys (including all requests) is necessary for the existence of higher level keys
        
        This method creates V2RB objects but does not return anything.

        :param vid: vehicle_id
        :param rid: plan_request_id
        """

        # LOG.verbose(f"build on current tree {rid} -> {vid}")
        associated_locked_rids = []
        for ob_rid in self.v2r_locked.get(vid, {}).keys():
            other_sub_rids = self._get_all_other_subrids_associated_to_this_subrid(ob_rid)
            associated_locked_rids.extend(list(other_sub_rids))
            feasible_found = False
            for sub_ob_rid in other_sub_rids:
                if self.rr.get(getRRKey(rid, sub_ob_rid)):
                    feasible_found = True
                    break
            if not feasible_found:
                return
        # check for assigned request not activated for global optimisation
        assigned_key = self.current_assignments.get(vid)
        if assigned_key is not None:
            o_rids = getRidsFromRTVKey(assigned_key)
            for o_rid in o_rids:
                if self.rid_to_consider_for_global_optimisation.get(o_rid) is None:
                    # LOG.verbose("additional rr check of not global opt {} <-> {}".format(rid, o_rid))
                    rr_comp = checkRRcomptibility(self.active_requests[o_rid], self.active_requests[rid], self.routing_engine, self.std_bt, dynamic_boarding_time=self.add_bt) 
                    if rr_comp:
                        # LOG.verbose(" -> 1")
                        self.rr[getRRKey(rid, o_rid)] = 1
        # check existing elements from lower to higher rid-number
        number_locked_rids = len(self.v2r_locked.get(vid, {}).keys())
        do_not_remove_for_lower_keys = [rid]
        do_not_remove_for_lower_keys.extend(associated_locked_rids)
        lower_keys_available = True
        do_not_build_on_rv_key = createRTVKey(vid, [rid])
        # check of activated heuristic
        max_tour_heuristic = False
        if self.applied_heuristics.get("single_plan_per_v2rb"):
            max_tour_heuristic = True
        #
        for i in range(max(number_locked_rids,1), MAX_LENGTH_OF_TREES):
            new_v2rb_found = False
            # # LOG.debug(f"build rid {rid} size {i}")
            for build_key in self.rtv_tree_N_v[vid].get(i, {}).keys():
                # # LOG.debug(f"build key {build_key}")
                lower_keys_available = True
                if build_key == do_not_build_on_rv_key:
                    continue
                if self.rtv_r.get(rid, {}).get(build_key) is not None:
                    # # LOG.debug(f"dont build on yourself {build_key}")
                    continue
                # check if lower key is available, otherwise match will not be possible
                # # LOG.debug(f"build on {build_key}")
                list_of_keys_to_test = createListLowerLevelKeys(build_key, rid, do_not_remove_for_lower_keys)
                
                for test_existence_rtv_key in list_of_keys_to_test:
                    if not self.rtv_obj.get(test_existence_rtv_key):
                        lower_keys_available = False
                        break
                if not lower_keys_available:
                    continue
                # test for feasibility by building on current V2RB object
                unsorted_rid_list = list(getRidsFromRTVKey(build_key))
                #check RR-compatibility! (?)
                rr_test = True
                for o_rid in unsorted_rid_list:
                    rr_test = self.rr.get(getRRKey(o_rid, rid))
                    # # LOG.debug(f"check rr {o_rid} {rid} -> {rr_test}")
                    if rr_test != 1:
                        rr_test = False
                        break
                if not rr_test:
                    continue
                
                unsorted_rid_list.append(rid)
                new_rtv_key = createRTVKey(vid, unsorted_rid_list)

                if self.rtv_obj.get(new_rtv_key, None) is not None:
                    continue

                # # LOG.debug(f"try building {build_key} | {rid} | ")
                if max_tour_heuristic:
                    test_new_V2RB = self._checkRTVFeasibilityAndReturnCreateV2RB_bestPlanHeuristic(vid, rid, build_key)
                else:
                    test_new_V2RB = self._checkRTVFeasibilityAndReturnCreateV2RB(vid, rid, build_key)

                if test_new_V2RB:
                    self._addRtvKey(new_rtv_key, test_new_V2RB)
                    new_v2rb_found = True
            if not new_v2rb_found:
                break

        if number_locked_rids == 0:
            rtv_key = createRTVKey(vid, [rid])
            if self.rtv_obj.get(rtv_key, None) is not None:
                return
            V2RB_obj = V2RB(self.routing_engine, self.active_requests, self.sim_time, rtv_key, self.veh_objs[vid], self.std_bt, self.add_bt, self.objective_function, new_prq_obj=self.active_requests[rid])
            if V2RB_obj.isFeasible():
                self._addRtvKey(rtv_key, V2RB_obj)

    def _checkRTVFeasibilityAndReturnCreateV2RB(self, vid : int, rid : Any, rtv_key : tuple) -> V2RB:
        """This method checks if the addition of rid to an existing V2RB object belonging to rtv_key.
        
        :param vid: vehicle_id
        :param rid: request_id to be added
        :param rtv_key: rtv_key of v2rb to be build on
        :return: V2RB_obj if feasible v2rb is found, else None
        """

        low_level_V2RB = self.rtv_obj[rtv_key]
        rid_list = list(getRidsFromRTVKey(rtv_key))
        rid_list.append(rid)
        new_rtv_key = createRTVKey(vid, rid_list)
        #new_prq_obj, new_rtv_key, routing_engine, rq_dict, sim_time, veh_obj, std_bt, add_bt
        return low_level_V2RB.addRequestAndCheckFeasibility(self.active_requests[rid], new_rtv_key, self.routing_engine, self.objective_function, self.active_requests, self.sim_time, self.veh_objs[vid], self.std_bt, self.add_bt)

    def _updateVehicleDataBase(self, vid : int):
        """ this function updates the v2rb-database of a specific vehicl from the last optimisation time-step
        and deletes v2rbs that are no longer feasible or updates plans corresponding the
        vehicle movements and boarding processes since the last opt-step
        :param vid: vehicle_id
        """
        veh_obj = self.veh_objs[vid]
        rv_rids = self.v2r.get(vid, {})
        assigned_key = self.current_assignments.get(vid)
        necessary_ob_rids = []
        if assigned_key is not None:
            base_ob_rids_dict = self.v2r_locked.get(vid, {})
            # LOG.verbose("base_ob_rids {} | assigned key {} | rid to mutually cluster {}".format(base_ob_rids_dict, assigned_key, self.rid_to_mutually_exclusive_cluster_id))
            for ass_rid in getRidsFromRTVKey(assigned_key):
                if base_ob_rids_dict.get( self._get_associated_baserid(ass_rid) ):
                    necessary_ob_rids.append(ass_rid)

        list_passed_VRLs = self.vid_to_list_passed_VRLs.get(vid, [])
        to_del_keys = {}
        # LOG.debug(f"updateVehicleDataBase {vid} assigned {assigned_key} ob {necessary_ob_rids}")
        for number_rtv_rids, rtv_key_dict in self.rtv_tree_N_v.get(vid, {}).items():
            for rtv_key in rtv_key_dict.keys():
                # # LOG.debug(f"check {rtv_key}")
                del_flag = False
                is_assigned = False
                if rtv_key == assigned_key:
                    is_assigned = True

                lower_keys_available = True
                if not is_assigned:
                    #necessary_keys = getNecessaryKeys(rtv_key, self.r_ob_orig)
                    for test_existence_rtv_key in getNecessaryKeys(rtv_key, necessary_ob_rids):
                        if test_existence_rtv_key == rtv_key:
                            continue
                        if not self.rtv_obj.get(test_existence_rtv_key):
                            lower_keys_available = False
                            # # LOG.debug(" -> key {} not available for {}".format(test_existence_rtv_key, rtv_key))
                            break
                if not lower_keys_available:
                    to_del_keys[rtv_key] = 1
                    continue

                v2rb_obj = self.rtv_obj[rtv_key]
                v2rb_obj.updateAndCheckFeasibility(self.routing_engine, self.objective_function, veh_obj, self.active_requests, self.sim_time, list_passed_VRLs = list_passed_VRLs, is_assigned = is_assigned)
                if not v2rb_obj.isFeasible():
                    to_del_keys[rtv_key] = 1
                else:
                    self._updateV2RBcostInDataBase(rtv_key, v2rb_obj)
                    # # LOG.debug(" -> still feasible")

        for rtv_key in to_del_keys.keys():
            self._delRtvKey(rtv_key)
        self._createNecessaryV2RBsBeforeBuildPhase(vid)

        # self._checkForNecessaryV2RBsAndComputeMissing(vid)    # TODO # think again if this is not necessary here!

    def _checkForNecessaryV2RBsAndComputeMissing(self, vid : int):
        assigned_key = self.current_assignments.get(vid, None)
        if not assigned_key:
            return
        necessary_ob_rids = []
        base_ob_rids_dict = self.v2r_locked.get(vid, {})
        for ass_rid in getRidsFromRTVKey(assigned_key):
            if base_ob_rids_dict.get( self._get_associated_baserid(ass_rid) ):
                necessary_ob_rids.append(ass_rid)
        necessary_keys = getNecessaryKeys(assigned_key, necessary_ob_rids)
        assigned_v2rb = self.rtv_obj.get(assigned_key)
        if assigned_v2rb is None:
            LOG.warning("assigned rtv-key not created after build! {} for vid {}".format(assigned_key, vid))
            LOG.warning("external assignments: {}".format({x: (str(y[0]), str(y[1])) for x, y in self.external_assignments.items()}))
            assigned_plan = self.external_assignments[vid][1]
            # try:
            #     feasible = assigned_plan.update_plan(self.veh_objs[vid], self.sim_time, self.routing_engine, keep_time_infeasible = True)
            # except:
            #     LOG.warning("update didnt work")
            #     feasible = False
            # if not feasible or assigned_key != self.external_assignments[vid][0]:
            #     LOG.warning("retry update")
            #     assigned_plan.update_plan(self.veh_objs[vid], self.sim_time, self.routing_engine, list_passed_VRLs=self.vid_to_list_passed_VRLs.get(vid, []), keep_time_infeasible = True)
            assigned_v2rb = V2RB(self.routing_engine, self.active_requests, self.sim_time, assigned_key, self.veh_objs[vid], self.std_bt, self.add_bt, self.objective_function, orig_veh_plans=[assigned_plan])
            self._addRtvKey(assigned_key, assigned_v2rb)
        for key in necessary_keys:
            if self.rtv_obj.get(key) is None:
                #LOG.debug(f"necessary key {key} not available for assignment {assigned_key} | ob: {self.v2r_locked.get(vid,{})}")
                new_v2rb = assigned_v2rb.createLowerV2RB(key, self.sim_time, self.routing_engine, self.objective_function, self.active_requests, self.std_bt, self.add_bt)
                self._addRtvKey(key, new_v2rb)

    def _createNecessaryV2RBsBeforeBuildPhase(self, vid : int):
        """ this function is needed in case of an deboarding. 
        it might happen that the necessery v2rb with just ob-rids is not there and will not be built even with rebuild.
        Additionally, missing v2rbs of requests that are not part of global optimisation are added to the database """
        # test for feasible ob v2rbs
        locked_rids = self.v2r_locked.get(vid, {})
        if len(locked_rids.keys()) > 0:
            assigned_key = self.current_assignments[vid]
            if assigned_key is None:
                LOG.warning("the database says someone is on board, but no v2rb is assigned!!")
                LOG.warning("locked board: {}".format(locked_rids))
                LOG.warning("vid {} {}".format(vid, self.veh_objs.get(vid)))
                LOG.warning("v2rbs: {}".format(self.rtv_v.get(vid)))
                return
            necessary_ob_rids = []
            base_ob_rids_dict = locked_rids
            for ass_rid in getRidsFromRTVKey(assigned_key):
                if base_ob_rids_dict.get( self._get_associated_baserid(ass_rid) ):
                    necessary_ob_rids.append(ass_rid)
            rtv_key = createRTVKey(vid, necessary_ob_rids)
            if self.rtv_obj.get(rtv_key, None) is None:
                LOG.debug("create ob v2rb: {} from {} | {} | {}".format(rtv_key, assigned_key, necessary_ob_rids, locked_rids))
                assigned_v2rb = self.rtv_obj.get(assigned_key)
                if assigned_v2rb is None:
                    LOG.warning("assigned rtv-key not here to create OBV2RB! {}".format(assigned_key))
                    assigned_plan = self.external_assignments[vid][1]
                    # try:
                    #     feasible = assigned_plan.update_plan(self.veh_objs[vid], self.sim_time, self.routing_engine, keep_time_infeasible = True)
                    # except:
                    #     LOG.warning("update didnt work")
                    #     feasible = False
                    # if not feasible or assigned_key != self.external_assignments[vid][0]:
                    #     LOG.warning("retry update")
                    #     assigned_plan.update_plan(self.veh_objs[vid], self.sim_time, self.routing_engine, list_passed_VRLs=self.vid_to_list_passed_VRLs.get(vid, []), keep_time_infeasible = True)
                    assigned_v2rb = V2RB(self.routing_engine, self.active_requests, self.sim_time, assigned_key, self.veh_objs[vid], self.std_bt, self.add_bt, self.objective_function, orig_veh_plans=[assigned_plan])
                    self._addRtvKey(assigned_key, assigned_v2rb)
                if rtv_key is not None:
                    ob_v2rb = assigned_v2rb.createLowerV2RB(rtv_key, self.sim_time, self.routing_engine, self.objective_function, self.active_requests, self.std_bt, self.add_bt)
                    self._addRtvKey(rtv_key, ob_v2rb)
        # test for feasible v2rbs of inactive rids
        assigned_key = self.current_assignments.get(vid)
        if assigned_key is not None:
            assigned_rids = getRidsFromRTVKey(assigned_key)
            check_needed = False
            for rid in assigned_rids:
                if self.rid_to_consider_for_global_optimisation.get(rid) is None:
                    check_needed = True
                    break
            if check_needed:
                # LOG.debug("rid inactive for global optimisation found {} -> check for missing v2rbs!".format(assigned_key))
                self._checkForNecessaryV2RBsAndComputeMissing(vid)

    def _set_init_solution_insertion(self):
        """ this function computes init solutions by an insertion heuristic
        the function needs to be called after the rv-step
        it will be used if an time-out is set to guarantee a certain solution quality
        """
        # LOG.debug("set initial solution insertion")
        current_insertion_solutions = {}    # vid -> veh-plan
        computed_cluster_id = {}    # base request id -> 1 if already computed
        for o_rid in self.requests_to_compute.keys():
            base_rid = self._get_associated_baserid(o_rid)
            if computed_cluster_id.get(base_rid) is not None:
                continue
            if self.unassigned_requests.get(o_rid) is not None:   # set init sol only for not yet assigned rids
                # LOG.debug("insert {}".format(o_rid))
                best_plan = None
                best_vid = None
                best_cfv = float("inf")
                for rid in self._get_all_rids_representing_this_base_rid(base_rid):
                    if not self.requests_to_compute.get(rid):
                        continue
                    if self.alonso_mora_parallelization_manager is None:
                        for vid in self.r2v.get(rid, {}).keys():
                            assigned_plan = current_insertion_solutions.get(vid)
                            if assigned_plan is None:
                                assigned_plan = self.fleetcontrol.veh_plans.get(vid, VehiclePlan(self.veh_objs[vid], self.sim_time, self.routing_engine, []))
                            ass_vid, ass_plan, delta_cfv = single_insertion([self.veh_objs[vid]], {vid : assigned_plan}, self.active_requests[rid], self.objective_function, self.routing_engine, self.active_requests, self.sim_time, self.std_bt, self.add_bt)
                            if ass_plan is not None and delta_cfv < best_cfv:
                                best_plan = ass_plan
                                best_cfv = delta_cfv
                                best_vid = ass_vid
                    else:
                        if len(self.r2v.get(rid, {}).keys()) > 0:
                            batch_size = max(int(np.floor(len(self.r2v.get(rid, {}).keys())/self.alonso_mora_parallelization_manager.number_cores/5.0)), 1)
                            c = 0
                            current_batch = []
                            for vid in self.r2v.get(rid, {}).keys():
                                assigned_plan = current_insertion_solutions.get(vid)
                                if assigned_plan is None:
                                    assigned_plan = self.fleetcontrol.veh_plans.get(vid, VehiclePlan(self.veh_objs[vid], self.sim_time, self.routing_engine, []))  
                                current_batch.append( ([self.veh_objs[vid]], {vid : assigned_plan}, self.active_requests[rid]) )
                                c += 1
                                if c % batch_size == 0:
                                    self.alonso_mora_parallelization_manager.batch_single_insertion(self.fo_id, current_batch)
                                    current_batch = []  
                            if len(current_batch) != 0:
                                self.alonso_mora_parallelization_manager.batch_single_insertion(self.fo_id, current_batch) 
                                current_batch = []
                            insertion_results = self.alonso_mora_parallelization_manager.fetch_batch_single_insertion()
                            for ass_vid, ass_plan, delta_cfv in insertion_results:
                                if ass_plan is not None and delta_cfv < best_cfv:
                                    best_plan = ass_plan
                                    best_cfv = delta_cfv
                                    best_vid = ass_vid
                computed_cluster_id[base_rid] = 1
                if best_plan is not None:
                    current_insertion_solutions[best_vid] = best_plan

        for vid, plan in current_insertion_solutions.items():
            # LOG.debug("set init sol: {} -> {}".format(vid, plan))
            self.set_assignment(vid, plan, _is_init_sol=True)



    #=========OPTIMISATION=====================
    def _runOptimisation(self):
        if self.solver == "Gurobi":
            self._runOptimisation_Gurobi()
        elif self.solver == "CPLEX":
            self._runOptimisation_CPLEX()
        else:
            raise EnvironmentError(f"False input for {G_RA_SOLVER}! Solver {self.solver} not found!")

    def _runOptimisation_Gurobi(self):
        """ this function uses gurobi to pick the best assignments from all v2rbs in the database
        by solving an ILP
        """
        import gurobipy as gurobi
        
        model_name = f"AlonsoMoraAssignment: assignment {self.sim_time}"
        
        vids = {}   #vid -> rtv_keys
        unassigned_rids = {}  #unassigned rid (or cluster_id) -> rtv_keys
        assigned_rids = {}  #assigned rid (or cluster_id) -> rtv_keys
        costs = {}  #rtv_key -> cost

        grb_available = False
        warning_created = False
        t0 = time.time()
        delta_t = time.time() - t0
        while not grb_available and delta_t <= RETRY_TIME:
            try:
                with gurobi.Env(empty=True) as env:
                    if self.fleetcontrol.log_gurobi:
                        import os
                        from src.misc.globals import G_DIR_OUTPUT
                        with open(os.path.join(self.fleetcontrol.dir_names[G_DIR_OUTPUT], "gurobi_log.log"), "a") as f:
                            f.write(f"\n\n{model_name}\n\n")
                        env.setParam('OutputFlag', 1)
                        env.setParam('LogToConsole', 0)
                        env.setParam('LogFile', os.path.join(self.fleetcontrol.dir_names[G_DIR_OUTPUT], "gurobi_log.log") )
                        env.start()
                    else:
                        env.setParam('OutputFlag', 0)
                        env.setParam('LogToConsole', 0)
                        env.start()

                    m = gurobi.Model(model_name, env = env)
                    grb_available = True

                    m.setParam(gurobi.GRB.param.Threads, self.optimisation_cores)
                    if self.optimisation_timeout:
                        m.setParam('TimeLimit', self.optimisation_timeout)
                    variables = {}  # rtv_key -> gurobi variable
                    
                    expr = gurobi.LinExpr()   # building optimization objective
                    key_to_varnames = {}
                    varnames_to_key = {}
                    for i, rtv_key in enumerate(self.rtv_costs.keys()):
                        rtv_cost = self.rtv_costs[rtv_key]
                        vid = getVidFromRTVKey(rtv_key)
                        rids = getRidsFromRTVKey(rtv_key)
                        cfv = rtv_cost

                        if cfv == float('inf') or np.isnan(cfv):
                            LOG.warning("v2rb with infinite cfv! no route found? {} {}".format(rtv_key, cfv))
                            continue

                        key_to_varnames[rtv_key] = str(i)
                        varnames_to_key[str(i)] = rtv_key

                        error_flag = False      #True if request not found on this core
                        try:
                            vids[vid].append(rtv_key)
                        except:
                            vids[vid] = [rtv_key]
                        for rid in rids:    
                            if not self.active_requests.get(rid):   #TODO dont know why i needed this check
                                error_flag = True
                                LOG.error(f"rid {rid} should not be here, also not key {rtv_key}")
                                vids[vid].remove(rtv_key)
                                self.delete_request(rid)

                        if not error_flag:
                            for rid in rids:   
                                v_rid = self._get_associated_baserid(rid) # requests with same mutually exclusive cluster ids are put into the same constraint later
                                #rid = GlobalFunctions.getOriginalRid(rid)
                                if self.unassigned_requests.get(rid):
                                    try:
                                        unassigned_rids[v_rid].append(rtv_key)
                                    except:
                                        unassigned_rids[v_rid] = [rtv_key]
                                else:
                                    try:
                                        assigned_rids[v_rid].append(rtv_key)
                                    except:
                                        assigned_rids[v_rid] = [rtv_key]

                        costs[rtv_key] = cfv
                        var = m.addVar(name = str(i), obj = cfv, vtype = gurobi.GRB.BINARY)
                        variables[rtv_key] = var
                        expr.add(var, cfv)
                        
                    m.setObjective(expr, gurobi.GRB.MINIMIZE)
                    # LOG.verbose("assignment assigned rids")
                    # LOG.verbose("{}".format(assigned_rids.keys()))
                    ## LOG.debug("{}".format(assigned_rids))
                    #vehicle constraint
                    for vid in vids.keys():
                        expr = gurobi.LinExpr()
                        for rtv in vids[vid]:
                            expr.add(variables[rtv], 1)
                        m.addConstr(expr, gurobi.GRB.LESS_EQUAL, 1, "c_{}".format(vid))
                        # TODO # seems to be wrong arguments but working anyway
                    #unassigned requests constraint
                    for rid in unassigned_rids.keys():
                        expr = gurobi.LinExpr()
                        for rtv in unassigned_rids[rid]:
                            expr.add(variables[rtv], 1)
                        m.addConstr(expr, gurobi.GRB.LESS_EQUAL, 1, "c_u_{}".format(rid))
                        # TODO # seems to be wrong arguments but working anyway
                    #assigned requests constraint
                    for rid in assigned_rids.keys():
                        expr = gurobi.LinExpr()
                        for rtv in assigned_rids[rid]:
                            expr.add(variables[rtv], 1)
                        m.addConstr(expr, gurobi.GRB.EQUAL, 1, "c_a_{}".format(rid))
                        # TODO # seems to be wrong arguments but working anyway
                    # set initial solution   # TODO set initial solution?
                    for vid, rtv_key in self.current_assignments.items():
                        if rtv_key is None:
                            continue
                        if not self.rtv_obj.get(rtv_key):
                            LOG.warning("current assignment {} not found for setting initial solution".format(rtv_key))
                        else:
                            variables[rtv_key].start = 1
                        
                    m.optimize() #optimization
                    LOG.info("=========")
                    LOG.info("OPT TIME {}:".format(self.sim_time))
                    LOG.info("solution status {}".format(m.status))
                    LOG.info("number solutions {}".format(m.SolCount))
                    LOG.info("number opt requests {} | number revealed requests {} | number active requests: {}".format(len(self.rid_to_consider_for_global_optimisation.keys()), len(unassigned_rids.keys()) + len(assigned_rids.keys()), len(self.active_requests.keys())))
                    LOG.info("number rtv_objs: {}".format(len(self.rtv_costs.keys())))
                    self.opt_stats = (m.status, m.SolCount)
                    if m.status != gurobi.GRB.Status.OPTIMAL and m.SolCount == 0:
                        if m.status == 3:
                            LOG.error("optimisation problem infeasible!")
                            m.computeIIS()
                            import os
                            from src.misc.globals import G_DIR_OUTPUT
                            p = os.path.join(self.fleetcontrol.dir_names[G_DIR_OUTPUT], r'iis.ilp')
                            m.write(p)
                            LOG.error("write iis to {}".format(p))
                            raise EnvironmentError
                        LOG.error("no solution within timeout found!")  #TODO fallback
                        LOG.error("the following lines should be adapted to the new version")
                        raise NotImplementedError
                        
                    #get solution
                    varnames = m.getAttr("VarName", m.getVars())
                    solution = m.getAttr("X",m.getVars())
                    
                    new_assignments = {}
                    sum_cfv = 0
                    for x in range(len(solution)):
                        if round(solution[x]) == 1:
                            key = varnames_to_key[varnames[x]]
                            vid = getVidFromRTVKey(key)
                            new_assignments[vid] = key
                            sum_cfv += self.rtv_costs[key]
                            # LOG.debug("{} -> {} : {}".format(vid, key, self.rtv_costs[key]))

                    self.current_best_cfv = sum_cfv
                    
                    self.optimisation_solutions = new_assignments

                    del m

            except gurobi.GurobiError:
                delta_t = time.time() - t0
                if not warning_created:
                    print("GUROBI ERROR: License Server not found or License not up to date!")
                    warning_created = True
        gurobi.disposeDefaultEnv()

    def _runOptimisation_CPLEX(self):
        """This method creates and solves the assignment optimization problem using CPLEX.
        #######################################################################
        # index description:
        # ------------------
        # optimization variable index: l = (i,k)
        # vehicle indices: i
        # rtv-bundle indices: k = [j1, j2, ...]
        # rid indices: j
        #######################################################################
        # optimization problem:
        # ---------------------
        # max     sum_l (c_l x_l) = sum_i sum_k (c_ik z_ik)
        # s.t.    sum_k (z_ik) <= 1                       for-all i        (C1)
        #         sum_i sum_{k in K(j)} (z_ik)  = 1       for-all j in R^a (C2)
        #         sum_i sum_{k in K(j)} (z_ik) <= 1       for-all j in R^u (C3)
        #######################################################################
        # the constraints have the following purpose:
        # (C1) each vehicle cannot be assigned more than once
        # (C2) each previously assigned request has to be assigned once again
        # (C3) each new request cannot be assigned more than once
        #
        # these constraints can be realized with the help of dictionaries
        # i2l -> all entries l that include vehicle i
        # j2l -> all entries l that include request j
        #######################################################################
        """
        import cplex
        from cplex.exceptions import CplexError
        from cplex.callbacks import MIPInfoCallback
        class StrictTimeLimitCallback(MIPInfoCallback):
            def __call__(self):
                if not self.aborted and self.has_incumbent():
                    # gap = 100.0 * self.get_MIP_relative_gap()
                    timeused = self.get_time() - self.starttime
                    if timeused > self.timelimit:
                        # self.log.log("Optimization approximate solution at {0} sec., gap = {1} %, quitting.".format(timeused, gap))
                        self.aborted = True
                        self.abort()

            # def setLog(self, log_instance):
                # self.log = log_instance
        # init optimization variable l and dictionaries
        l_counter = 0               # int | 1D optimization variable index counter
        sorted_l = []               # sorted list of optimization variable indices
        utility_function = {}       # l -> c_l (cost/utility of of optimization variable)
        l2rtv = {}                  # l -> rtv_key
        i2l = {}                    # vid -> list_of_l
        j2l = {}                    # rid -> list_of_l
        # setting variable and adjusting dictionaries
        assigned_rids = {}
        unassigned_rids = {}
        for rtv_key, rtv_cost in self.rtv_costs.items():

            if rtv_cost == float('inf') or np.isnan(rtv_cost):
                LOG.warning("v2rb with infinite cfv! no route found? {} {}".format(rtv_key, rtv_cost))
                continue
            utility_function[l_counter] = rtv_cost 

            l2rtv[l_counter] = rtv_key
            sorted_l.append(l_counter)
            #
            vid = getVidFromRTVKey(rtv_key)
            list_rids = getRidsFromRTVKey(rtv_key)
            try:
                i2l[vid].append(l_counter)
            except:
                i2l[vid] = [l_counter]
            for rid in list_rids:
                v_rid = self._get_associated_baserid(rid) # requests with same mutually exclusive cluster ids are put into the same constraint later
                try:
                    j2l[v_rid].append(l_counter)
                except:
                    j2l[v_rid] = [l_counter]
                if not self.unassigned_requests.get(rid):
                    assigned_rids[v_rid] = 1
                # if self.current_assignments[vid] is not None and rid in self.current_assignments[vid][1:]:
                    # assigned_rids[v_rid] = 1
            l_counter += 1
        if l_counter == 0:
            self.optimisation_solutions = {}
            return
        #
        # 0) init cplex problem
        self.prob = cplex.Cplex()
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)
        #
        # 1) set variables and objective function
        objective_list = []
        var_names = []
        var_types = "" # I for integer variable, C for continuous variable
        var_lb= []
        var_ub = []
        for l in sorted_l:
            objective_list.append(utility_function[l])
            var_names.append("z_{0}".format(l2rtv[l]).replace("|","_"))
            var_lb.append(0)
            var_ub.append(1)
            # var_types += "I"
            var_types += "B"
        self.prob.variables.add(obj = objective_list, lb=var_lb, ub=var_ub, names=var_names, types=var_types)
        # 2) define boundary condition equations
        rows = []
        r_names = []
        cols = []
        vals = []
        p_rhs = []
        sense_str = "" # E for equal, L for less or equal
        #
        # 2a) constraints (C1) for each vid_i
        c_counter = 0
        for vid_i in i2l.keys():
            p_rhs.append(1)
            for l in i2l[vid_i]:
                rows.append(c_counter)
                cols.append(l)
                vals.append(1)
            r_names.append("{0}".format(vid_i))
            sense_str += "L"
            c_counter += 1
        #
        # 2b) constraints (C2) / (C3) for each rid_j
        for rid_j in j2l.keys():
            p_rhs.append(1)
            for l in j2l[rid_j]:
                rows.append(c_counter)
                cols.append(l)
                vals.append(1)
            r_names.append("{0}".format(rid_j))
            if assigned_rids.get(rid_j):
                sense_str += "E"
            else:
                sense_str += "L"
            c_counter += 1
        #print(zip(rows, cols, vals))
        if len(rows)==0:
            return
        try:
            self.prob.linear_constraints.add(rhs=p_rhs, senses=sense_str, names=r_names)
            self.prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
        except CplexError as exc:
            err_str_list = ["Error in creation of user-constraints. j2l:"]
            for k, v in j2l.items():
                if len(v) != len(set(v)):
                    err_str_list.append("k: {0} | len(v): {1} | len(set(v)): {2} | v: {3}".format(k, len(v), len(set(v)), v))
            LOG.error("\n".join(err_str_list))
            raise EnvironmentError(exc)
        #
        # x) set timeout for non-optimal (heuristic) solution
        if self.optimisation_timeout:
            #timelim_cb = self.prob.register_callback(TimeLimitCallback)
            timelim_cb = self.prob.register_callback(StrictTimeLimitCallback)
            timelim_cb.starttime = self.prob.get_time()
            timelim_cb.timelimit = self.optimisation_timeout
            # timelim_cb.acceptablegap = 5
            timelim_cb.aborted = False
        #
        # 4) solve problem and re-transform l -> rtv_key
        new_assignments = {}
        sum_cfv = 0
        try:
            self.prob.set_results_stream(None)
            self.prob.solve()
            prt_str_list = []
            prt_str_list.append("Number of variables: {0}".format(l_counter))
            prt_str_list.append("Number of requests: {0}".format(len(j2l)))
            prt_str_list.append("Number of vehicles: {0}".format(len(i2l)))
            prt_str_list.append("Solution status = {0}".format(self.prob.solution.get_status()))
            prt_str_list.append("Solution value  = {0}".format(self.prob.solution.get_objective_value()))
            #numcols = self.prob.variables.get_num()
            #numrows = self.prob.linear_constraints.get_num()
            #slack = self.prob.solution.get_linear_slacks()
            x = self.prob.solution.get_values()
            prt_str_list.append("Assignments:")
            number_assigned_requests = 0
            for l in sorted_l:
                if x[l] > 0.999:
                    rtv_key = l2rtv[l]
                    number_assigned_requests += len(getRidsFromRTVKey(rtv_key))
                    vid = getVidFromRTVKey(rtv_key)
                    new_assignments[vid] = rtv_key
                    sum_cfv += self.rtv_costs[rtv_key]
            prt_str_list.append("Number of previous assignments before decision time step: {0}".format(len(self.current_assignments)))
            prt_str_list.append("Number of assignments after optimization: {0}".format(len(new_assignments)))
            prt_str_list.append("Number of assigned requests: {0}".format(number_assigned_requests))
            LOG.info("\n".join(prt_str_list))
        except CplexError as exc:
            LOG.error("CPLEX ERROR: {0} assignments set due to error in optimization!".format(len(new_assignments)))
            raise EnvironmentError(exc)
        self.current_best_cfv = sum_cfv
        
        self.optimisation_solutions = new_assignments
        self.prob.end()


    def _setAdditionalInitForParallelization(self, current_assignments, v2r_locked, requests_to_compute, rr, v2r, active_requests, external_assignments, rid_to_mutually_exclusive_cluster_id, mutually_exclusive_cluster_id_to_rids,rid_to_consider_for_global_optimisation):
        """ this function sets additional inits in the database if this class member is created in a parallel process
        this function is only needed in the AlonsoMoraParallelization, therefore treated as private!
        :param current_assignments: dict vid -> rtv_key of currently assign rtv_keys
        :param v2r_locked: dict vid -> rid -> 1 for request ids that are locked to vehicles with id vid
        :param requests_to_compute: dict rid -> 1 requests which rtv-tree has to be computed actively
        :param rr: rr_key -> 1 for request-request-combinations which are rr-feasible
        :param v2r: dict vid -> rid for feasible vehicle-rid-combinations
        :param active_requests: dict rid -> prq (Plan Request obj)
        :param vid_external_assignment: entry of the dictionary self.external_assignments (assignments computed outside of the alonsomora algorithm; used as fallback) 
        :param rid_to_mutually_exclusive_cluster_id:
        :param rid_to_consider_for_global_optimisation:
        """
        #
        #LOG.verbose("setAdditionalInitForParalellisation {}, {}, {}".format(current_assignments, v2r_locked, requests_to_compute))
        #
        self.current_assignments = current_assignments

        self.v2r_locked = v2r_locked
        self.r2v_locked = {}
        for vid, rid_dict in self.v2r_locked.items():
            for rid in rid_dict.keys():
                self.r2v_locked[rid] = vid
        self.requests_to_compute = requests_to_compute
        self.rr = rr
        self.v2r = v2r
        for vid, rid_dict in self.v2r.items():
            for rid in rid_dict.keys():
                try:
                    self.r2v[rid][vid] = 1
                except:
                    self.r2v[rid] = {vid : 1}
        self.active_requests = active_requests
        self.external_assignments = external_assignments
        self.rid_to_mutually_exclusive_cluster_id = rid_to_mutually_exclusive_cluster_id
        self.mutually_exclusive_cluster_id_to_rids = mutually_exclusive_cluster_id_to_rids
        # for rid, cluster_id in self.rid_to_mutually_exclusive_cluster_id.items():
        #     try:
        #         self.mutually_exclusive_cluster_id_to_rids[cluster_id][rid] = 1
        #     except:
        #         self.mutually_exclusive_cluster_id_to_rids[cluster_id] = {rid : 1}
        self.rid_to_consider_for_global_optimisation = rid_to_consider_for_global_optimisation

    ###=======================================================================================================###
    ### HEURISTICS
    ###=======================================================================================================###
    def _after_opt_rv_best_v2rb_heuristic(self):
        """ this heuristic has to be called after an optimisation
        only max_rv connections are kept for each request depending on the highest v2rb cfv with each veh
        and of course the current assignment """
        rid_to_opt_vid = {}
        for vid, rtv_key in self.optimisation_solutions.items():
            if rtv_key is not None:
                for rid in getRidsFromRTVKey(rtv_key):
                    rid_to_opt_vid[rid] = vid
                    for other_rid in self._get_all_other_subrids_associated_to_this_subrid(rid):
                        rid_to_opt_vid[other_rid] = vid
        to_del = {}
        for rid, vid_dict in self.r2v.items():
            if len(vid_dict.keys()) > self.max_rv_connections:
                vid_to_best_val = {}
                if rid_to_opt_vid.get(rid):
                    vid_to_best_val[rid_to_opt_vid[rid]] = -float('inf')
                for rtv_key in self.rtv_r.get(rid,{}):
                    vid = getVidFromRTVKey(rtv_key)
                    if vid_to_best_val.get(vid, 0) > self.rtv_costs[rtv_key]:
                        vid_to_best_val[vid] = self.rtv_costs[rtv_key]
                sort = sorted(list(vid_to_best_val.items()), key = lambda x : x[1])
                #print(sort)
                if len(sort) > self.max_rv_connections:
                    to_del[rid] = {}
                    #LOG.info("delete {} cons for rid {}".format(len(sort) - self.max_rv_connections, rid) )
                    for i in range(self.max_rv_connections, len(sort)):
                        ## LOG.debug(" -> {} : {} | {}".format(rid, sort[i][0], sort[i][1]))
                        if rid_to_opt_vid.get(rid, -1) == sort[i][0]:
                            continue
                        to_del[rid][sort[i][0]] = 1 

        for rid, vid_dict in to_del.items():
            for vid in vid_dict.keys():
                try:
                    del self.r2v[rid][vid]
                except:
                    pass
                try:
                    del self.v2r[vid][rid]
                except:
                    pass
            for rtv_key in list(self.rtv_r.get(rid, {})):
                if vid_dict.get(getVidFromRTVKey(rtv_key)):
                    #LOG.info("delete: {}".format(rtv_key))
                    self._delRtvKey(rtv_key)

    def _before_opt_rv_nearest_with_rr_heuristic(self, rid, vid_to_travel_time, already_vid_to_keep = {}, prev_plans = {}, max_rv_connections = None):
        """ this heuristic has to be called before an optimisation
        it is called within the rv-computation
        only max_rv connections are kept for each request depending on the closest vehicles
            if all rr-connections to assigned requests hold 
        rr connections have to be computed before rv-connections!
        :param rid: request id to check
        :param vid_to_travel_time: dict vehicle_id -> traveltime to reach request
        :param prev_plans: dummy variable to keep consitency
        :return: dict vid -> for vehicles to consider for rv-connections (maximum set by self.max_rv_connections), empty dict
        """
        vids_to_keep = already_vid_to_keep.copy()
        if max_rv_connections is None:
            max_rv = self.max_rv_connections
        else:
            max_rv = max_rv_connections
        #return all
        if len(vid_to_travel_time.keys()) <= max_rv:
            return {vid : 1 for vid in vid_to_travel_time.keys()}, prev_plans
        #check for vids and assigned rids sorted by travel time
        for vid, _ in sorted(vid_to_travel_time.items(), key = lambda x:x[1]):
            if vids_to_keep.get(vid):
                continue
            assigned = self.current_assignments.get(vid)
            if assigned is not None:
                assigned_rids = getRidsFromRTVKey(assigned)
                rr_not_found = False
                for other_rid in assigned_rids:
                    rr_key = getRRKey(rid, other_rid)
                    if not self.rr.get(rr_key):
                        rr_not_found = True
                        break
                if rr_not_found:
                    continue
            vids_to_keep[vid] = 1
            if len(vids_to_keep.keys()) >= max_rv:
                return vids_to_keep, prev_plans
        # fill if to small just ob rid check
        if len(vids_to_keep.keys()) < max_rv:
            for vid in vid_to_travel_time.keys():
                if not vids_to_keep.get(vid):
                    rr_not_found = False
                    for other_base_rid in self.v2r_locked.get(vid, {}):
                        feasible_base_rid = False
                        for other_rid in self._get_all_rids_representing_this_base_rid(other_base_rid):
                            rr_key = getRRKey(rid, other_rid)
                            if self.rr.get(rr_key):
                                feasible_base_rid = True
                                break
                        if not feasible_base_rid:
                            rr_not_found = True
                            break
                    if rr_not_found:
                        continue
                    vids_to_keep[vid] = 1
                    if len(vids_to_keep.keys()) >= max_rv:
                        break

        return vids_to_keep, prev_plans

    def _before_opt_rv_best_insertion_heuristic(self, rid, vid_to_travel_time, already_vid_to_keep = {}, prev_plans = {}, max_rv_connections = None):
        """ this heuristic has to be called before an optimisation
        choices are made by best insertions into the currently assigned tours (prev_plans) of a vehicle
        the vids with the best routes after insertion are picked for the heuristic
        :param rid: request id to check
        :param vid_to_travel_time: dict vehicle_id -> traveltime to reach request
        :return: tuple( dict vid -> for vehicles to consider for rv-connections (maximum set by self.max_rv_connections), vid -> vehicle plan for picked vehicles after insertion)
        """
        if max_rv_connections is None:
            max_rv = self.max_rv_connections
        else:
            max_rv = max_rv_connections
        vids_to_keep = already_vid_to_keep.copy()
        #return all
        if len(vid_to_travel_time.keys()) <= max_rv:
            return {vid : 1 for vid in vid_to_travel_time.keys()}, prev_plans
        vid_to_insertion_vals = {}
        already_assigned_vid = self.fleetcontrol.rid_to_assigned_vid.get(rid)
        if self.alonso_mora_parallelization_manager is None:
            for vid, _ in vid_to_travel_time.items():
                if already_assigned_vid is not None and vid == already_assigned_vid:
                    assigned_plan = self.fleetcontrol.veh_plans.get(vid, VehiclePlan(self.veh_objs[vid], self.sim_time, self.routing_engine, []))
                    vid_to_insertion_vals[vid] = (-float("inf"), assigned_plan)
                else:
                    assigned_plan = prev_plans.get(vid)
                    if assigned_plan is None:
                        assigned_plan = self.fleetcontrol.veh_plans.get(vid, VehiclePlan(self.veh_objs[vid], self.sim_time, self.routing_engine, []))
                    ass_vid, ass_plan, delta_cfv = single_insertion([self.veh_objs[vid]], {vid : assigned_plan}, self.active_requests[rid], self.objective_function, self.routing_engine, self.active_requests, self.sim_time, self.std_bt, self.add_bt)

                    if ass_plan is not None:
                        vid_to_insertion_vals[vid] = (delta_cfv, ass_plan)
        else:
            batch_size = max(int(np.floor(len(vid_to_travel_time.keys())/self.alonso_mora_parallelization_manager.number_cores/5.0)), 1)
            c = 0
            current_batch = []
            for vid, _ in vid_to_travel_time.items():
                if already_assigned_vid is not None and vid == already_assigned_vid:
                    assigned_plan = self.fleetcontrol.veh_plans.get(vid, VehiclePlan(self.veh_objs[vid], self.sim_time, self.routing_engine, []))
                    vid_to_insertion_vals[vid] = (-float("inf"), assigned_plan) 
                else:
                    assigned_plan = prev_plans.get(vid)
                    if assigned_plan is None:
                        assigned_plan = self.fleetcontrol.veh_plans.get(vid, VehiclePlan(self.veh_objs[vid], self.sim_time, self.routing_engine, []))  
                    current_batch.append( ([self.veh_objs[vid]], {vid : assigned_plan}, self.active_requests[rid]) )
                    c += 1
                    if c % batch_size == 0:
                        self.alonso_mora_parallelization_manager.batch_single_insertion(self.fo_id, current_batch)
                        current_batch = []  
            if len(current_batch) != 0:
                self.alonso_mora_parallelization_manager.batch_single_insertion(self.fo_id, current_batch) 
                current_batch = []
            insertion_results = self.alonso_mora_parallelization_manager.fetch_batch_single_insertion()
            for ass_vid, ass_plan, delta_cfv in insertion_results:
                if ass_plan is not None:
                    vid_to_insertion_vals[ass_vid] = (delta_cfv, ass_plan)     

        i = 0
        for vid, x in sorted(vid_to_insertion_vals.items(), key = lambda x:x[1][0]):
            if i == 0:
                prev_plans[vid] = x[1]
            if not vids_to_keep.get(vid):
                vids_to_keep[vid] = 1
                i += 1
                if i >= max_rv:
                    break
        return vids_to_keep, prev_plans

    def _before_opt_rv_mix_heuristic(self, rid, vid_to_travel_time, prev_plans = {}):
        # TODO # might be deprecated
        heu_attributs = self.applied_heuristics["before_opt_rv_mix"]
        t = time.time()
        # LOG.debug("heu rv mix for rid {} with atts {} | n vids in time range: {}".format(rid, heu_attributs, len(vid_to_travel_time.keys())))
        vids_to_keep, prev_plans = self._before_opt_rv_best_insertion_heuristic(rid, vid_to_travel_time, prev_plans=prev_plans, max_rv_connections=heu_attributs[0])
        # LOG.debug("after insertion heu: {} n prev plans {}".format(len(vids_to_keep.keys()), len(prev_plans.keys())))
        vids_to_keep, prev_plans = self._before_opt_rv_nearest_with_rr_heuristic(rid, vid_to_travel_time, already_vid_to_keep=vids_to_keep, prev_plans=prev_plans, max_rv_connections=heu_attributs[0]+heu_attributs[1])
        # LOG.debug("after nearest heu: {} | prev plans {}".format(len(vids_to_keep.keys()), len(prev_plans.keys())))
        # LOG.debug("took: {}".format(time.time() - t))
        return vids_to_keep, prev_plans


    def _checkRTVFeasibilityAndReturnCreateV2RB_bestPlanHeuristic(self, vid, rid, rtv_key):
        """This method checks if the addition of rid to an existing V2RB object belonging to rtv_key.
        This method is used for the "bestPlan" Heuristic: for each v2rb only the best plan is kept in store
        just apllying this heuristic would result a strong dependency on the order the v2rbs are built
        to prevent this, the v2rb is not just build by inserting rid in rtv_key, but all possible ways of inserting a rid into the corresponding low_level_v2rbs are tried
            that would result in the same v2rb. from this v2rb only the best route is stored
        
        :param vid: vehicle_id
        :param rid: request_id to be added
        :param rtv_key: rtv_key of v2rb to be build on
        :return: V2RB_obj if feasible v2rb is found, else None
        """
        low_level_V2RB = self.rtv_obj[rtv_key]
        prev_rid_list = list(getRidsFromRTVKey(rtv_key))
        rid_list =  prev_rid_list[:] + [rid]
        new_rtv_key = createRTVKey(vid, rid_list)
        #new_prq_obj, new_rtv_key, routing_engine, rq_dict, sim_time, veh_obj, std_bt, add_bt
        best_v2rb = low_level_V2RB.addRequestAndCheckFeasibility(self.active_requests[rid], new_rtv_key, self.routing_engine, self.objective_function, self.active_requests, self.sim_time, self.veh_objs[vid], self.std_bt, self.add_bt)
        # best_cfv = float("inf")
        # if best_v2rb is not None:
        #     best_cfv = best_v2rb.cost_function_value
        # # build other way around
        # for other_rid in prev_rid_list:
        #     low_level_rids = rid_list[:]
        #     low_level_rids.remove(other_rid)
        #     low_level_V2RB = self.rtv_obj.get(createRTVKey(vid, low_level_rids))
        #     if not low_level_V2RB:  # usual for grade 1 v2rbs (single rid v2rb is built after tree building, but shouldnt matter for the the permutation of 2 rids)
        #         continue
        #     new_v2rb = low_level_V2RB.addRequestAndCheckFeasibility(self.active_requests[other_rid], new_rtv_key, self.routing_engine, self.objective_function, self.active_requests, self.sim_time, self.veh_objs[vid], self.std_bt, self.add_bt)
        #     if new_v2rb is not None and new_v2rb.cost_function_value < best_cfv:
        #         best_v2rb = new_v2rb
        #         best_cfv = new_v2rb.cost_function_value
        # only keep best plan
        if best_v2rb is None:
            return None
        else:
            best_plan = best_v2rb.getBestPlan()
            best_v2rb.veh_plans = [best_plan]
            return best_v2rb

    ###=======================================================================================================###
    ### OTHER STUFF
    ###=======================================================================================================###

    def optimize_boarding_points_locally(self):
        """
        -> after optimization call
        -> for each assignment
        -> get complete tree of current_assignment
        -> create new AM subclass
        -> build complete tree of all associated sub rids
            -> don't build ob rids, but keep resulting v2rbs in memory
        -> take best rated v2rb instead of original one
        """
        # LOG.info("optimize boarding points locally!")
        if self.alonso_mora_parallelization_manager is None:
            for vid, assigned_key in self.optimisation_solutions.items():
                if assigned_key is not None:
                    current_rids = getRidsFromRTVKey(assigned_key)
                    necessary_ob_rids = []
                    for ass_rid in current_rids:
                        if self.v2r_locked.get(vid, {}).get( self._get_associated_baserid(ass_rid) ):
                            necessary_ob_rids.append(ass_rid)

                    complete_tree = [k for k in getNecessaryKeys(assigned_key, necessary_ob_rids ) ]
                    # LOG.debug("vid {} assigned key {}".format(vid, assigned_key))
                    # LOG.debug("vid complete tree {}".format(complete_tree))
                    # LOG.verbose("mutually exclusiv to rid {}".format(self.mutually_exclusive_cluster_id_to_rids))
                    # LOG.verbose("other way around {}".format(self.rid_to_mutually_exclusive_cluster_id))

                    rids_to_build = []
                    for ass_sub_rid in current_rids:
                        for other_sub_rid in self._get_all_other_subrids_associated_to_this_subrid(ass_sub_rid):
                            rids_to_build.append(other_sub_rid)

                    v2rb_obj_list = []
                    for key in complete_tree:
                        try:
                            v2rb_obj_list.append( self.rtv_obj[key] )
                        except:
                            LOG.warning("Necessary key not in database {} | think of adopting checkForNecessaryKeys... | {}".format(key, complete_tree))
                    for rid in rids_to_build:
                        if rid in current_rids:
                            continue
                        rid_rtvs = self.rtv_r.get(rid, {})
                        for key in rid_rtvs.keys():
                            if not getVidFromRTVKey(key) == vid:
                                continue
                            other_rids = getRidsFromRTVKey(key)
                            only_included_rids = True
                            for other_rid in other_rids:
                                if not other_rid in rids_to_build:
                                    only_included_rids = False
                                    break
                            if only_included_rids:
                                v2rb_obj_list.append( self.rtv_obj[key] )

                    mutually_excl_cluster_dict = {self._get_associated_baserid(rid) : self._get_all_other_subrids_associated_to_this_subrid(rid) for rid in current_rids }
                    new_v2rb_list = optimize_boarding_points_of_vid_assigned_v2rb(None, self.routing_engine, self.sim_time, self.std_bt, self.add_bt, self.objective_function,
                        self.veh_objs[vid], assigned_key, mutually_excl_cluster_dict, self.r2v_locked, self.v2r_locked,
                        self.active_requests, v2rb_obj_list, self.veh_tree_build_timeout)

                    best_val = self.rtv_costs[assigned_key]
                    prev_best_val = best_val
                    best_key = assigned_key
                    for rtv_obj in new_v2rb_list:
                        key = rtv_obj.rtv_key
                        self._addRtvKey(key, rtv_obj)
                        # LOG.verbose("found: {}".format(key))
                        if self.rtv_costs[key] < best_val:
                            best_val = self.rtv_costs[key]
                            best_key = key
                    self.optimisation_solutions[vid] = best_key
                    # LOG.debug("vid route cfv: before {} after {} delta {}".format(prev_best_val, best_val, best_val - prev_best_val))

                    new_assigned_rids = getRidsFromRTVKey(best_key)
                    for rid in new_assigned_rids:
                        if not rid in current_rids:
                            self.rid_to_consider_for_global_optimisation[rid] = 1
                            self.requests_to_compute_in_next_step[rid] = 1
                    for rid in current_rids:
                        if not rid in new_assigned_rids:
                            try:
                                del self.rid_to_consider_for_global_optimisation[rid]
                            except:
                                pass
        else:
            new_v2rbs_all = []
            batch_size = 1# max(int(np.floor(len(self.veh_objs)/self.alonso_mora_parallelization_manager.number_cores/15.0)), 1)
            current_batch = []
            c = 1
            for vid, assigned_key in self.optimisation_solutions.items():
                if assigned_key is not None:
                    current_rids = getRidsFromRTVKey(assigned_key)
                    necessary_ob_rids = []
                    for ass_rid in current_rids:
                        if self.v2r_locked.get(vid, {}).get( self._get_associated_baserid(ass_rid) ):
                            necessary_ob_rids.append(ass_rid)

                    complete_tree = [k for k in getNecessaryKeys(assigned_key, necessary_ob_rids ) ]
                    # LOG.debug("vid {} assigned key {}".format(vid, assigned_key))
                    # LOG.debug("vid complete tree {}".format(complete_tree))
                    # LOG.verbose("mutually exclusiv to rid {}".format(self.mutually_exclusive_cluster_id_to_rids))
                    # LOG.verbose("other way around {}".format(self.rid_to_mutually_exclusive_cluster_id))

                    rids_to_build = []
                    for ass_sub_rid in current_rids:
                        for other_sub_rid in self._get_all_other_subrids_associated_to_this_subrid(ass_sub_rid):
                            rids_to_build.append(other_sub_rid)

                    v2rb_obj_list = []
                    for key in complete_tree:
                        try:
                            v2rb_obj_list.append( self.rtv_obj[key] )
                        except:
                            LOG.warning("Necessary key not in database {} | think of adopting checkForNecessaryKeys...| {}".format(key, complete_tree))
                    for rid in rids_to_build:
                        if rid in current_rids:
                            continue
                        rid_rtvs = self.rtv_r.get(rid, {})
                        for key in rid_rtvs.keys():
                            if not getVidFromRTVKey(key) == vid:
                                continue
                            other_rids = getRidsFromRTVKey(key)
                            only_included_rids = True
                            for other_rid in other_rids:
                                if not other_rid in rids_to_build:
                                    only_included_rids = False
                                    break
                            if only_included_rids:
                                v2rb_obj_list.append( self.rtv_obj[key] )
                    #mutually_excl_cluster_dict = {self._get_associated_baserid(rid) : self._get_all_other_subrids_associated_to_this_subrid(rid) for rid in current_rids }
                    current_batch.append( (self.fleetcontrol.op_id, self.veh_objs[vid], assigned_key, self.v2r_locked.get(vid, {}), v2rb_obj_list, self.veh_tree_build_timeout) )
                    c += 1
                    if c % batch_size == 0:
                        self.alonso_mora_parallelization_manager.batch_optimize_boarding_points_of_vid_assigned_v2rb(current_batch)
                        current_batch = []
            if len(current_batch) > 0:
                self.alonso_mora_parallelization_manager.batch_optimize_boarding_points_of_vid_assigned_v2rb(current_batch)
                current_batch = []
            vid_v2rb_list = self.alonso_mora_parallelization_manager.fetch_optimize_boarding_points_of_vid_assigned_v2rb()
            for vid, new_v2rb_list in vid_v2rb_list:
                assigned_key = self.optimisation_solutions[vid]
                current_rids = getRidsFromRTVKey(assigned_key)
                best_val = self.rtv_costs[assigned_key]
                prev_best_val = best_val
                best_key = assigned_key
                for rtv_obj in new_v2rb_list:
                    key = rtv_obj.rtv_key
                    self._addRtvKey(key, rtv_obj)
                    # LOG.verbose("found: {}".format(key))
                    if self.rtv_costs[key] < best_val:
                        best_val = self.rtv_costs[key]
                        best_key = key
                self.optimisation_solutions[vid] = best_key
                # LOG.debug("vid route cfv: before {} after {} delta {}".format(prev_best_val, best_val, best_val - prev_best_val))

                new_assigned_rids = getRidsFromRTVKey(best_key)
                for rid in new_assigned_rids:
                    if not rid in current_rids:
                        self.rid_to_consider_for_global_optimisation[rid] = 1
                        self.requests_to_compute_in_next_step[rid] = 1
                for rid in current_rids:
                    if not rid in new_assigned_rids:
                        try:
                            del self.rid_to_consider_for_global_optimisation[rid]
                        except:
                            pass


def optimize_boarding_points_of_vid_assigned_v2rb(fleetcontrol, routing_engine, sim_time, operator_attributes, objective_function,
                veh_obj, assigned_key, mutually_exclusive_cluster_id_to_rids,
                r2v_locked, v2r_locked, active_requests, v2rb_obj_to_add):
    """ this function optimizes the boarding locations for a single vehicle and its assigned v2rb
    returns a list of all feasible tours serving the same physical customers like the input v2rb but with different imaginary requests (mutually_cluster_ids_to_rids)
    :param fleetcontrol: fleetcontrol reference
    :param routing_engine: routing engine reference
    :param sim_time: current simulation time
    :param operator_attributes: dictionary of op attributes
    :param objective_function: reference to control function
    :param veh_obj: veh_obj reference
    :param assigned_key: key of assigned v2rb
    :param mutually_exclusive_cluster_id_to_rids: dict base_rid -> sub_rid_iterable (collection of sub_rids corresponding to same customer)
    :param r2v_locked: rid -> vid -> 1, for rid locked to vid
    :param v2r_locked: vid -> rid -> 1, for rid locked to vid
    :param active_requests: rid -> prq_obj for all active requests
    :param v2rb_obj_to_add: list v2rb objs that are added before building the tree (wont be built again)
    """
    #t = time.time()
    vid = veh_obj.vid
    sub_am_module = AlonsoMoraAssignment(fleetcontrol, routing_engine, sim_time, operator_attributes, objective_function, veh_objs_to_build={veh_obj.vid : veh_obj})
    sub_am_module.applied_heuristics = None
    for base_rid, sub_rid_iterable in mutually_exclusive_cluster_id_to_rids.items():
        sub_am_module.set_mutually_exclusive_assignment_constraint(sub_rid_iterable, base_rid)
    current_rids = getRidsFromRTVKey(assigned_key)
    rids_to_build = []
    for ass_sub_rid in current_rids:
        base_rid = sub_am_module._get_associated_baserid(ass_sub_rid)
        # LOG.verbose("ass sub rid {} | base rid {}".format(ass_sub_rid, base_rid))
        for other_sub_rid in sub_am_module._get_all_rids_representing_this_base_rid(base_rid):
            # LOG.verbose("other sub rid {}".format(other_sub_rid))
            rids_to_build.append(other_sub_rid)
        if r2v_locked.get(base_rid):
            sub_am_module.r2v_locked[base_rid] = r2v_locked[base_rid]
    # LOG.debug("optimize boarding points for vid {} with rids to build: {}".format(vid, rids_to_build))
    sub_am_module.v2r_locked[vid] = v2r_locked[vid]

    for rid in rids_to_build:
        sub_am_module.add_new_request(rid, active_requests[rid], is_allready_assigned=True)

    for v2rb in v2rb_obj_to_add:
        key = v2rb.rtv_key
        sub_am_module._addRtvKey(key, v2rb)

    sub_am_module._computeRR()
    sub_am_module._computeRV()
    sub_am_module._buildTreeForVid(vid)
    # LOG.info("local boarding point optimization for vid {} with {} rids took {}".format(vid, len(rids_to_build), time.time()-t))
    # if time.time() - t > 30.0:
    #     v2rb_list = sub_am_module._getV2RBobjList()
    #     for v2rb in v2rb_list:
    #         LOG.info("found {} alltogether: {}".format(v2rb.rtv_key, len(v2rb_list)))
    return sub_am_module._getV2RBobjList()

                

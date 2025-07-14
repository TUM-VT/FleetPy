import logging
from typing import Callable, Dict, Any, List
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.misc.globals import G_DIR_OUTPUT

import time
import numpy as np

import src.fleetctrl.pooling.GeneralPoolingFunctions as GeneralPoolingFunctions
from src.fleetctrl.pooling.batch.BatchAssignmentAlgorithmBase import BatchAssignmentAlgorithmBase, SimulationVehicleStruct
from src.fleetctrl.pooling.batch.AlonsoMora.misc import *
from src.fleetctrl.pooling.GeneralPoolingFunctions import checkRRcomptibility
from src.fleetctrl.planning.VehiclePlan import VehiclePlan, BoardingPlanStop
from src.fleetctrl.pooling.immediate.insertion import simple_remove, insert_prq_in_selected_veh_list
from src.misc.globals import *
from src.routing.NetworkBase import NetworkBase
from src.simulation.Legs import VehicleRouteLeg
from src.simulation.Vehicles import SimulationVehicle

from src.fleetctrl.pooling.immediate.singleVehicleDARP import solve_single_vehicle_DARP_exhaustive, solve_single_vehicle_DARP_LNS

if TYPE_CHECKING:
    from src.routing.NetworkBase import NetworkBase
    from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraParallelization import ParallelizationManager
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.fleetctrl.planning.PlanRequest import PlanRequest
    from src.simulation.Vehicles import SimulationVehicle

LOG = logging.getLogger(__name__)
LARGE_INT = 100000
MAX_LENGTH_OF_TREES = 1024
TIME_OUT = 30
WRITE_PROBLEM = True
RETRY_TIME = 24*3600

class AlonsoMoraAssignmentOriginal(BatchAssignmentAlgorithmBase):
    def __init__(self, fleetcontrol: FleetControlBase, routing_engine: NetworkBase, sim_time: int, obj_function: Callable[..., Any], operator_attributes: dict, optimisation_cores: int = 1, seed: int = 6061992, veh_objs_to_build: Dict[int, SimulationVehicleStruct] = {}):
        super().__init__(fleetcontrol, routing_engine, sim_time, obj_function, operator_attributes, optimisation_cores, seed=seed, veh_objs_to_build=veh_objs_to_build)
        self.veh_tree_build_timeout : int = operator_attributes.get(G_RA_TB_TO_PER_VEH, None)
        self.optimisation_timeout : int = operator_attributes.get(G_RA_OPT_TO, None)
        self.max_rv_connections : int = operator_attributes.get(G_RA_MAX_VR, None)
        applied_heuristics = operator_attributes.get(G_RA_HEU, None)
        self._max_prqs_exhaustive_DARP = 4
        
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

        self.rtv_obj : Dict[Any, VehiclePlan] = {}                   # rtv_key -> VehiclePlan-Object
        self.rtv_costs : Dict[Any, float] = {}                 # rtv_key -> cost-function-value
        self.current_assignments : Dict[Any, int] = {}       # vid -> rtv_key
        self.optimisation_solutions : Dict[int, tuple] = {}    # vid -> rtv_key computed when solving optimisation problem
        self.active_requests : Dict[int, PlanRequest] = {}           # rid -> request-Object
        #
        self.unassigned_requests : Dict[Any, 1] = {}   # rid -> 1
        #
        self.rtv_tree_N_v : Dict[int, Dict[int, Dict[tuple, 1]]] = {}         #vid -> number_of_requests -> rtv_key -> 1
        self.rtv_v : Dict[int, Dict[tuple, 1]] = {}         # vid -> {}: rtv_key -> 1
        self.rtv_r : Dict[Any, Dict[tuple, 1]] = {}         #rid -> {}: rtv_key -> 1
        self.v2r : Dict[int, Dict[Any, 1]] = {}            #vid -> rid -> 1 | generally available request-to-vehicle combinations
        self.v2r_locked : Dict[int, Dict[Any, 1]] = {}    #vid -> rid -> 1 | rids currently locked to vid
        self.r2v_locked : Dict[Any, Dict[int, 1]] = {}
        self.r2v : Dict[Any, Dict[int, 1]] = {}           #rid -> vid -> 1 | generally available request-to-vehicle combinations
        self.rr : Dict[tuple, 1] = {}            #rid1 -> rid2 -> 1/None | generally available 
       # creation of permanent vehicle keys
        for vid in self.veh_objs.keys():
            self.rtv_v[vid] = {}
            self.rtv_tree_N_v[vid] = {}
            self.v2r[vid] = {}
            self.v2r_locked[vid] = {}

        
    def compute_new_vehicle_assignments(self, sim_time: int, vid_to_list_passed_VRLs: Dict[int, List[VehicleRouteLeg]], veh_objs_to_build: Dict[int, SimulationVehicle] = {}, new_travel_times: bool = False, build_from_scratch: bool = False):
        LOG.debug(f"new assignments at time {sim_time} with requests {self.unassigned_requests.keys()}")
        #0) set database
        self.sim_time = sim_time
        self.veh_objs = {}
        if len(veh_objs_to_build.keys()) == 0:
            for veh_obj in self.fleetcontrol.sim_vehicles:
                veh_obj_struct = SimulationVehicleStruct(veh_obj, self.fleetcontrol.veh_plans.get(veh_obj.vid, VehiclePlan(veh_obj, self.sim_time, self.routing_engine, [])), sim_time, self.routing_engine)
                self.veh_objs[veh_obj.vid] = veh_obj_struct
        else:
            self.veh_objs = veh_objs_to_build
        
        #1) compute RV for all requests
        self._computeRV(self.rid_to_consider_for_global_optimisation.keys()) # TODO rids locked to vids?
        
        #2) compute RR for all requests
        self._computeRR(self.rid_to_consider_for_global_optimisation.keys())
        
        #3) build RTV graph
        for vid, r_dict in self.v2r.items():
            # TODO hoda: use ML to prune the rids to build on with vr graph predictions
            self._buildTreeForVid(vid, r_dict.keys())
            
        #4) solve assignment
        self._runOptimisation()
        
    def _computeRV(self, rids_to_compute):
        """ this function computes all rv-connections from self.requests_to_compute with all active vehicles
        """
        veh_locations_to_vid = {}

        for vid, veh_obj in self.veh_objs.items():
            try:
                veh_locations_to_vid[veh_obj.pos].append(vid)
            except:
                veh_locations_to_vid[veh_obj.pos] = [vid]
        current_time = self.sim_time

        for rid in rids_to_compute:
            self.r2v[rid] = {}
            prq = self.active_requests[rid]
            o_pos, _, latest_pu = prq.get_o_stop_info()
            routing_results = self.routing_engine.return_travel_costs_Xto1(veh_locations_to_vid.keys(), o_pos,
                                                                            max_cost_value=latest_pu - current_time)
            for veh_loc, tt, _, _ in routing_results:
                for vid in veh_locations_to_vid[veh_loc]:
                    vid_locked = self.r2v_locked.get(rid, None)
                    if vid_locked is not None and vid_locked != vid:
                        continue
                    self.r2v[rid][vid] = tt
                    try:
                        self.v2r[vid][rid] = tt
                    except KeyError:
                        self.v2r[vid] = {rid : tt}

    
    def _computeRR(self, list_rids):
        for rid1 in list_rids:
            rq1 = self.active_requests[rid1]
            for rid2 in list_rids:
                if rid1 != rid2:
                    if not self._is_subrid(rid1) or not self._is_subrid(rid2) or self._get_associated_baserid(rid1) != self._get_associated_baserid(rid2):
                        rq2 = self.active_requests[rid2]
                        rr_comp = checkRRcomptibility(rq1, rq2, self.routing_engine, self.std_bt, dynamic_boarding_time=self.add_bt) #TODO definitions of boarding times!
                        if rr_comp:
                            self.rr[getRRKey(rid1, rid2)] = 1
    
    def _buildTreeForVid(self, vid : int, rids_to_build):
        """ this method builds new V2RBS for all requests_to_compute for a single vehicle
        param vid : vehicle_id for vid to be build
        """
        assigned_plan = self.fleetcontrol.veh_plans[vid]
        obj = self.fleetcontrol.compute_VehiclePlan_utility(self.sim_time, self.veh_objs[vid], assigned_plan)
        assigned_key = getRTVkeyFromVehPlan(assigned_plan)
        if assigned_key is not None:
            self._addRtvKey(assigned_key, assigned_plan, obj)
            self.current_assignments[vid] = assigned_key
        LOG.debug(f" ... build tree for vid {vid} with assigned key {assigned_key} for rids {rids_to_build}")
        t_all_vid = time.time()
        #LOG.debug("build tree for vid {} with rids {} | locked {}".format(vid, rids_to_build_with_hierarchy, self.r2v_locked))
        for rid in rids_to_build:
            # TODO hoda: use ML to prune the rids to build on with vr graph predictions
            # TODO hoda check ML model prediction for feasibility
            # TODO pseudocode: concatenate vid features and rid features to generate a feature vector
            # TODO pseudocode: load pretrained vr model
            # TODO pseudocode: if model.predict_proba(X) < self.rr_threshold: continue
            t_c = time.time() - t_all_vid
            ## LOG.debug(f"build {rid} h {h}")
            if self.veh_tree_build_timeout and  t_c > self.veh_tree_build_timeout:
                # LOG.debug(f"break tree building for vid {vid} after {t_c} s")
                break
            self._buildOnCurrentTree(vid, rid)
     
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
                    # TODO hoda check ML model prediction for feasibility
                    # TODO pseudocode: concatenate o_rid features and rid features to generate a feature vector
                    # TODO pseudocode: load pretrained model
                    # TODO pseudocode: if model.predict_proba(X) < self.rr_threshold:
                    # TODO pseudocode: rr_test = False
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
                if i <= self._max_prqs_exhaustive_DARP:
                    new_veh_p, cost = solve_single_vehicle_DARP_exhaustive(self.veh_objs[vid], self.routing_engine, [self.active_requests[rid] for rid in unsorted_rid_list], self.fleetcontrol, self.sim_time, self.fleetcontrol.veh_plans[vid])
                else:
                    low_level_vehplan = self.rtv_obj[build_key]
                    rid_list = list(getRidsFromRTVKey(build_key))
                    rid_list.append(rid)
                    new_rtv_key = createRTVKey(vid, rid_list)
                    r_list = insert_prq_in_selected_veh_list([self.veh_objs[vid]], {vid:low_level_vehplan}, self.active_requests[rid], 
                                                    self.fleetcontrol.vr_ctrl_f, self.routing_engine, self.active_requests, self.sim_time, self.fleetcontrol.const_bt,
                                                    self.fleetcontrol.add_bt)
                    if len(r_list) != 0:
                        _, new_veh_p, _ = min(r_list, key=lambda x:x[2])
                        cost = self.fleetcontrol.compute_VehiclePlan_utility(self.sim_time, self.veh_objs[vid], new_veh_p)
                    else:
                        new_veh_p, cost = None, None

                if new_veh_p is not None:
                    self._addRtvKey(new_rtv_key, new_veh_p, cost)
                    new_v2rb_found = True
            if not new_v2rb_found:
                break

        if number_locked_rids == 0:
            rtv_key = createRTVKey(vid, [rid])
            if self.rtv_obj.get(rtv_key, None) is not None:
                return
            vehplan, obj = solve_single_vehicle_DARP_exhaustive(self.veh_objs[vid], self.routing_engine, [self.active_requests[rid]], self.fleetcontrol, self.sim_time, self.fleetcontrol.veh_plans[vid])
            if vehplan is not None:
                self._addRtvKey(rtv_key, vehplan, obj)
                
    def _addRtvKey(self, rtv_key, veh_plan, obj):
        """this function adds entries to all necessery database dictionaries
        param rtv_key : key of rtv_obj
        param rtv_obj : rtv_obj/V2RB (look V2RB.py)
        return : None
        """
        LOG.debug(f"add rtv key {rtv_key} | {veh_plan}")
        self.rtv_obj[rtv_key] = veh_plan
        self.rtv_costs[rtv_key] = obj
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
            
    #=========OPTIMISATION=====================
    def _runOptimisation(self):
        if self.solver == "Gurobi":
            self._runOptimisation_Gurobi()
        else:
            raise EnvironmentError(f"False input for {G_RA_SOLVER}! Solver {self.solver} not found!")

    def _runOptimisation_Gurobi(self):
        """ this function uses gurobi to pick the best assignments from all v2rbs in the database
        by solving an ILP
        """
        import gurobipy as gurobi
        
        model_name = f"AlonsoMoraAssignmentOriginal: assignment {self.sim_time}"
        
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
                        m.addConstr(expr <= 1, "c_{}".format(vid))
                        # TODO # seems to be wrong arguments but working anyway
                    #unassigned requests constraint
                    for rid in unassigned_rids.keys():
                        expr = gurobi.LinExpr()
                        for rtv in unassigned_rids[rid]:
                            expr.add(variables[rtv], 1)
                        m.addConstr(expr <= 1, "c_u_{}".format(rid))
                        # TODO # seems to be wrong arguments but working anyway
                    #assigned requests constraint
                    for rid in assigned_rids.keys():
                        expr = gurobi.LinExpr()
                        for rtv in assigned_rids[rid]:
                            expr.add(variables[rtv], 1)
                        m.addConstr(expr == 1, "c_a_{}".format(rid))
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
        
    def clear_databases(self):
        self.rtv_obj : Dict[Any, VehiclePlan] = {}                   # rtv_key -> V2RB-Object
        self.rtv_costs : Dict[Any, float] = {}                 # rtv_key -> cost-function-value
        self.current_assignments : Dict[Any, int] = {}       # vid -> rtv_key
        self.optimisation_solutions : Dict[int, tuple] = {}    # vid -> rtv_key computed when solving optimisation problem
        #
        self.unassigned_requests : Dict[Any, 1] = {}   # rid -> 1
        #
        self.rtv_tree_N_v : Dict[int, Dict[int, Dict[tuple, 1]]] = {}         #vid -> number_of_requests -> rtv_key -> 1
        self.rtv_v : Dict[int, Dict[tuple, 1]] = {}         # vid -> {}: rtv_key -> 1
        self.rtv_r : Dict[Any, Dict[tuple, 1]] = {}         #rid -> {}: rtv_key -> 1
        self.v2r : Dict[int, Dict[Any, 1]] = {}            #vid -> rid -> 1 | generally available request-to-vehicle combinations
        self.r2v : Dict[Any, Dict[int, 1]] = {}           #rid -> vid -> 1 | generally available request-to-vehicle combinations
        self.rr : Dict[tuple, 1] = {}            #rid1 -> rid2 -> 1/None | generally available 
       # creation of permanent vehicle keys
        for vid in self.veh_objs.keys():
            self.rtv_v[vid] = {}
            self.rtv_tree_N_v[vid] = {}
            self.v2r[vid] = {}
        return super().clear_databases()
    
    def get_optimisation_solution(self, vid : int) -> VehiclePlan:
        """ returns optimisation solution for vid
        :param vid: vehicle id
        :return: vehicle plan object for the corresponding vehicle
        """
        key = self.optimisation_solutions.get(vid)
        # LOG.debug("veh obj {}".format(self.veh_objs[vid]))
        #minimal_vehplan = VehiclePlan(self.veh_objs[vid], self.sim_time, self.routing_engine, self.veh_objs[vid].locked_planstops)
        if key is None:
            if self.veh_objs[vid].has_locked_vehplan() > 0:
                return self.veh_objs[vid].locked_vehplan
            else:
                return None
        else:
            return self.rtv_obj[key]
        
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
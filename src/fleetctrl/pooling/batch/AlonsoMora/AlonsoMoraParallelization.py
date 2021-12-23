import importlib
import os
from src.FleetSimulationBase import DEFAULT_LOG_LEVEL
import traceback
from multiprocessing import Process, Queue, Pipe
import time
import math
import dill as pickle

from src.misc.globals import *
from src.misc.init_modules import load_routing_engine
import src.fleetctrl.pooling.GeneralPoolingFunctions as GeneralPoolingFunctions
from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraAssignment import *
from src.fleetctrl.pooling.immediate.insertion import single_insertion

import logging 
LOG = logging.getLogger(__name__)

#======================================================================
#COMMUNICATION CODES
from src.fleetctrl.pooling.batch.AlonsoMora.comcodes import *
#=====================================================================

def startProcess(q_in, q_out, process_id, scenario_parameters, dir_names):
    PP = ParallelProcess(q_in, q_out, process_id, scenario_parameters, dir_names)
    LOG.info(f"time to run PP {process_id}")
    PP.run()

class ParallelizationManager():
    def __init__(self, number_cores, scenario_parameters, dir_names):   #TODO define data_staff (for loading all the data (nw e.g. cant be transmitted directly))
        """ this class is used to manage the parallelization of functions from the AlonsoMoraAssignment module
        function calls are batched and sent to parallel processes via multiprocessing.Queue() objects
        on each multiprocessing.Process() a ParallelProcess class is initialized which is active during the whole time of simulation
            therefore the network must not be loaded during each optimisation step but is initialized on the parallel cores if initialization for the simulation
            -> this class has to be initialized before(!) the AlonsoMoraAssignment-Module since it may be inptu in its initialization
        
        communication is made with two Queues():
            q_in : this class puts tasks to be computed on the parallel processes
                    a task is defined by the tuple (communication_code, (function_arguments)) the communication codes defined as globals above define the function to be called on the parallel cores
            q_out : here all the outputs from the parallel cores are collected

        :param number_cores: number of parallel processes
        :param scenario_parameters: dictionary initialized in the beginning of the simulation for all simulation parameters
        :param dir_names: dictionary of input/ouput directories initialzied in the beginning of the simulation
        """
        self.number_cores = number_cores

        self.q_in = Queue() #communication queues
        self.q_out = Queue()

        self.processes = [Process(target = startProcess, args = (self.q_in, self.q_out, i, scenario_parameters, dir_names)) for i in range(self.number_cores)]    # start processes
        for p in self.processes:
            p.daemon = True    
            p.start()

        self.last_function_call = None  #last task sent to the parallel cores
        self.number_currently_pending_results = 0   #number of results that are expected to come from the parallel cores
        self.fetched_result_list = []       # buffer for fetched results from parallel cores

        self.fo_new_requests_for_next_opt_step = {}
        self.fo_rids_to_delete_for_next_opt_step = {}
        self.fo_rid_to_mutually_exclusive_cluster_id_for_next_opt_step = {} 
        self.fo_rid_to_consider_for_global_optimisation_for_next_opt_step = {}

        self.last_update_network_call = -1



    def killProcesses(self):
        """ this function is supposed to kill all parallel processes """
        for i in range(self.N_cores):
            self.q_in.put( (KILL,) )
            
        self.q_in.close()
        self.q_in.join_thread()
        self.q_out.close()
        self.q_out.join_thread()

        
        for p in self.processes:
            p.join()

    def _checkFunctionCall(self, function_id = -1):
        """ check if this function call is feasible respective to the last function call and if all results are fetched
        raises error if not
        :param function_id: corresponding communication code
        """
        if function_id != self.last_function_call:
            if self.number_currently_pending_results > 0 or len(self.fetched_result_list) > 0:
                print("results or computations from parallel processes are not cleared! use fetch functions to retrieve results!")
                print("not retrieved results from : {}".format(self.last_function_call))
                raise SyntaxError
    
    def _checkFetchCall(self, function_id):
        """ checks if all results have been fetched in case of a new function call
        raises error if not
        :param function_id: corresponding communcation code
        """
        if function_id != self.last_function_call:
            if not self.number_currently_pending_results == 0:
                LOG.error("results from former computation not fetched! use fetch-functions!")
                LOG.error("not retrieved results from : {} | current call {}".format(self.last_function_call, function_id))
                raise SyntaxError

    def initOp(self, fo_id, obj_function, operator_attributes):
        """ initializes an operator with its specific attributes on the parallel processes
        :param fo_id: fleet control id
        :param obj_function: objective function to rate vehicle plans
        :param operator_attributes: dictionary of operator attributes
        """
        c = 0
        for i in range(self.number_cores):
            # LOG.debug(f"Queue put {INIT_OP}")
            obj_function_pickle = pickle.dumps(obj_function)
            self.q_in.put( (INIT_OP, (fo_id, obj_function_pickle, operator_attributes) ) )
            c += 1
        self.last_function_call = INIT_OP
        self.fo_new_requests_for_next_opt_step[fo_id] = {}
        self.fo_rids_to_delete_for_next_opt_step[fo_id] = {}
        self.fo_rid_to_mutually_exclusive_cluster_id_for_next_opt_step[fo_id] = {} 
        self.fo_rid_to_consider_for_global_optimisation_for_next_opt_step[fo_id] = {}
        while c > 0:
            x = self.q_out.get()
            c -= 1
            # LOG.debug(f"initOp got {x} rest {c}")

    def delete_request(self, fo_id, rid):
        """ this function is used to communicate the deletion of a rid to the parallel processes
        :param fo_id: fleetctrl id
        :param rid: request id
        """
        if self.fo_new_requests_for_next_opt_step[fo_id].get(rid) is not None:
            del self.fo_new_requests_for_next_opt_step[fo_id][rid]
        else:
            self.fo_rids_to_delete_for_next_opt_step[fo_id][rid] = 1
        try:
            del self.fo_rid_to_mutually_exclusive_cluster_id_for_next_opt_step[fo_id][rid]
        except:
            pass
        try:
            del self.fo_rid_to_consider_for_global_optimisation_for_next_opt_step[fo_id][rid]
        except:
            pass

    def add_new_request(self, fo_id, rid, prq, consider_for_global_optimisation = True, is_allready_assigned = False):
        """ this function used to add a new request at parallel processes
        :param fo_id: flletctrl id
        :param rid: request id
        :param prq: plan request obj
        """
        self.fo_new_requests_for_next_opt_step[fo_id][rid] = prq
        if consider_for_global_optimisation:
            self.fo_rid_to_consider_for_global_optimisation_for_next_opt_step[fo_id][rid] = 1

    def set_mutually_exclusive_assignment_constraint(self, fo_id, list_sub_rids, base_rid):
        for s_rid in list_sub_rids:
            self.fo_rid_to_mutually_exclusive_cluster_id_for_next_opt_step[fo_id][s_rid] = base_rid

    def setSimtimeAndActiveRequests(self, fo_id, sim_time, active_requests_dict, rid_to_mutually_exclusive_cluster_id, rid_to_consider_for_global_optimisation):
        """ updates simulation time and active requests on the parallel cores
        :param fo_id: fleet control id
        :param sim_time: current simulation time
        :param active_requests_dict: dict rid -> planrequests objects of all currently active requests
        :param rid_to_mutually_exclusive_cluster_id: dict rid -> cluster_id (see init alonosomoraassignment)
        :param rid_to_consider_for_global_optimisation: dict rid -> 1 which are currently active to build the overall tree for
        """
        c = 0
        for i in range(self.number_cores):
            # LOG.debug(f"Queue put {SET_SIMTIME_AND_ACTIVE_REQUESTS}")
            self.q_in.put((SET_SIMTIME_AND_ACTIVE_REQUESTS, (fo_id, sim_time, active_requests_dict, rid_to_mutually_exclusive_cluster_id, rid_to_consider_for_global_optimisation)) )
            c += 1
        self.last_function_call = SET_SIMTIME_AND_ACTIVE_REQUESTS
        while c > 0:
            x = self.q_out.get()
            c -= 1
            # LOG.debug(f"setSimtimeAndActiveRequests got {x} rest {c}")

    def setDynSimtimeAndActiveRequests(self, fo_id, sim_time):
        """ updates simulation time and active requests on the parallel cores
        :param fo_id: fleet control id
        :param sim_time: current simulation time
        :param active_requests_dict: dict rid -> planrequests objects of all currently active requests
        :param rid_to_mutually_exclusive_cluster_id: dict rid -> cluster_id (see init alonosomoraassignment)
        :param rid_to_consider_for_global_optimisation: dict rid -> 1 which are currently active to build the overall tree for
        """
        c = 0
        for i in range(self.number_cores):
            LOG.debug(f"Queue put {SET_DYN_SIMTIME_AND_ACTIVE_REQUESTS}")
            self.q_in.put((SET_DYN_SIMTIME_AND_ACTIVE_REQUESTS, (fo_id, sim_time, self.fo_new_requests_for_next_opt_step[fo_id], self.fo_rids_to_delete_for_next_opt_step[fo_id], self.fo_rid_to_mutually_exclusive_cluster_id_for_next_opt_step[fo_id], self.fo_rid_to_consider_for_global_optimisation_for_next_opt_step[fo_id])) )
            c += 1
        self.last_function_call = SET_DYN_SIMTIME_AND_ACTIVE_REQUESTS
        while c > 0:
            x = self.q_out.get()
            c -= 1
            LOG.debug(f"setSimtimeAndActiveRequests got {x} rest {c}")
        self.fo_new_requests_for_next_opt_step[fo_id] = {}
        self.fo_rids_to_delete_for_next_opt_step[fo_id] = {}
        self.fo_rid_to_mutually_exclusive_cluster_id_for_next_opt_step[fo_id] = {}
        self.fo_rid_to_consider_for_global_optimisation_for_next_opt_step[fo_id] = {}

    def update_network(self, sim_time):
        """ this method communicates the parallel processes to update their network
        """
        if sim_time != self.last_update_network_call:
            c = 0
            for i in range(self.number_cores):
                LOG.debug(f"Queue put {LOAD_NEW_NETWORK_TRAVEL_TIMES}")
                self.q_in.put((LOAD_NEW_NETWORK_TRAVEL_TIMES, (sim_time, )) )
                c += 1
            self.last_function_call = LOAD_NEW_NETWORK_TRAVEL_TIMES
            while c > 0:
                x = self.q_out.get()
                c -= 1
                LOG.debug(f"update_network got {x} rest {c}")
            self.last_update_network_call = sim_time

    def computeRR(self, fo_id, rid_list):
        """ compute RR dict for fleetcontroller fo_id for all requests in rid_list with all active requests
        no return; use fetch_computeRR() to retrieve results
        :param fo_id: (int) fo_id of corresponding fleetcontroller
        :param rid_list: (list) list of all rids for rr-graphs to be computed
        """
        self._checkFunctionCall()
        length = len(rid_list)
        sub_batch_size = 4
        batch_length = math.floor(float(length)/self.number_cores/sub_batch_size)
        if batch_length == 0:
            batch_length = 1
        #all_batch_length = batch_length * self.N_cores * sub_batch_size
        batch_rid_list = []
        for rid in rid_list:
            batch_rid_list.append(rid)
            if len(batch_rid_list) == batch_length:
                # LOG.debug(f"Queue put {COMPUTE_RR}")
                self.q_in.put( (COMPUTE_RR, (fo_id, batch_rid_list) ) )
                self.number_currently_pending_results += 1
                batch_rid_list = []
        if len(batch_rid_list) != 0:
            # LOG.debug(f"Queue put {COMPUTE_RR}")
            self.q_in.put( (COMPUTE_RR, (fo_id, batch_rid_list) ) )
            self.number_currently_pending_results += 1
        self.last_function_call = COMPUTE_RR

    def fetch_computeRR(self):
        """ fetches result from parallel processes of the previously called function "computeRR"
        return : rr_dict (dict); rr_key -> 1 for rid-pairs with rr-connections
        """
        self._checkFetchCall(COMPUTE_RR)
        c = 0
        rr_dict = {}
        while c < self.number_currently_pending_results:
            x = self.q_out.get()
            rr_dict.update(x)
            c += 1
        self.number_currently_pending_results = 0
        return rr_dict

    def setRR(self, fo_id, rr_dict):
        """ set rr-dict information on all parallel processes to synchronize information
        :param fo_id: fleetcontroll id of corresponding rr_dict
        :param rr_dict: (dict) ; rr_key -> 1 for rid-pairs with rr-connections
        """
        self._checkFunctionCall()
        c = 0
        for i in range(self.number_cores):
            # LOG.debug(f"Queue put {SET_RR}")
            self.q_in.put( (SET_RR, (fo_id, rr_dict)) )
            c += 1
        self.last_function_call = SET_RR
        while c > 0:
            x = self.q_out.get()
            c -= 1
            # LOG.debug(f"setRR got {x} rest {c}")

    def computeRV(self, veh_locations_to_vid, rids_to_compute_to_rq, sim_time):
        """ this function computes the RV-graph in parallel
        :param veh_locations_to_vid: dict network position -> list of vehicle ids
        :param rids_to_compute_to_rq: dict request id -> plan request object
        :param sim_time: current simulation time
        :return: rv dict rid -> vid -> 1 if rv-connection exists (request rid can be reached by vehicle vid in time)
        """
        self._checkFunctionCall()
        self._setXto1TargetLocations(list(veh_locations_to_vid.keys()))
        length = len(rids_to_compute_to_rq.keys())
        sub_batch_size = 4
        batch_length = math.floor(float(length)/self.number_cores/sub_batch_size)
        if batch_length == 0:
            batch_length = 1
        #all_batch_length = batch_length * self.N_cores * sub_batch_size
        batch_target_postions = []
        for rid, rq in rids_to_compute_to_rq.items():
            o_pos, _, latest_pu = rq.get_o_stop_info()
            max_time = latest_pu - sim_time
            batch_target_postions.append( (rid, o_pos, max_time) )
            if len(batch_target_postions) == batch_length:
                # LOG.debug(f"Queue put {RETURN_TARGETPOS_IN_TIMERANGE}")
                self.q_in.put( (RETURN_TARGETPOS_IN_TIMERANGE, batch_target_postions) )
                self.number_currently_pending_results += 1
                batch_target_postions = []
        if len(batch_target_postions) != 0:
            # LOG.debug(f"Queue put {RETURN_TARGETPOS_IN_TIMERANGE}")
            self.q_in.put( (RETURN_TARGETPOS_IN_TIMERANGE, batch_target_postions) )
            self.number_currently_pending_results += 1
        self.last_function_call = RETURN_TARGETPOS_IN_TIMERANGE

        self._checkFetchCall(RETURN_TARGETPOS_IN_TIMERANGE)
        c = 0
        new_rv = {}
        travel_infos = {}
        while c < self.number_currently_pending_results:
            x = self.q_out.get()
            for rid, o_pos, routing_results in x:
                for target_pos, cfv, tt, dis in routing_results:
                    travel_infos[(target_pos, o_pos)] = (cfv, tt, dis)
                    for vid in veh_locations_to_vid[target_pos]:
                        try:
                            new_rv[rid][vid] = 1
                        except:
                            new_rv[rid] = {vid: 1}
            c += 1
        self.number_currently_pending_results = 0
        self._setRVroutingResults(travel_infos)
        return new_rv, travel_infos

    def _setXto1TargetLocations(self, list_target_pos):
        """ this function sets the target locations for rv-computations (current vehicle positions)
        :param list_target_pos: list of all current vehicle network positions
        """
        self._checkFunctionCall()
        c = 0
        for i in range(self.number_cores):
            # LOG.debug(f"Queue put {SET_XTO1_TARGET_LOCATIONS}")
            self.q_in.put( (SET_XTO1_TARGET_LOCATIONS, list_target_pos) )
            c += 1
        self.last_function_call = SET_XTO1_TARGET_LOCATIONS 
        while c > 0:
            x = self.q_out.get()
            c -= 1
            # LOG.debug(f"setXto1Target got {x} rest {c}")     

    def _setRVroutingResults(self, travel_info_dict):
        """ this function sends the parallel processed routing results to all cores to update their database
        :param travel_info_dict : dictionary (o_node, d_node) -> (cfv, tt, dis)
        """
        self._checkFunctionCall()
        c = 0
        for i in range(self.number_cores):
            # LOG.debug(f"Queue put {SET_RV_RESULTS}")
            self.q_in.put( (SET_RV_RESULTS, (travel_info_dict,)) )
            c += 1
        self.last_function_call = SET_RV_RESULTS
        while c > 0:
            x = self.q_out.get()
            c -= 1
            # LOG.debug(f"set routing results got {x} rest {c}")

    def update_v2rbs_and_compute_new(self, fo_id, veh_obj, currently_assigned_key, ob_rids, locked_rids, v2r_rids_to_compute, vid_list_passed_VRLs = [], v2rb_list_to_be_updated = [], vid_external_assignment = None):
        """This function computes new v2rbs for ONE single vehicle. To compute v2rbs for many vehicles, this function can be called multiple times.
        For best perfomance, only one call of fetch_update_v2rbs_and_compute_new. is recommended after all vehicles are computed
        this method doesnt return anything; use use fetch_update_v2rbs_and_compute_new to retrieve results after all tasks have been scheduled
        :param fo_id: id of corresponding fleetcontroller
        :param veh_obj: SimulationVehicleStruct of corresponding vehicle to compute
        :param std_bt: int constant boarding time
        :param add_bt: int additional boarding time parameter
        :param v2r_rids_to_compute: list of rids to be computed for corresponding vehicle (including v2r-connections)
        :param vid_list_passed_VRLs:  list_passt_VRLs since last opt-step (for updating v2rbs)
        :param v2rb_list_to_be_updated: list v2rbs from last opt-step
        :param vid_external_assignment: entry of the dictionary AMA.external_assignments (assignments computed outside of the alonsomora algorithm; used as fallback) 
        """
        self._checkFunctionCall(function_id=UPDATE_V2RBS_AND_COMPUTE_NEW)
        # LOG.debug(f"Queue put {UPDATE_V2RBS_AND_COMPUTE_NEW}")
        self.q_in.put((UPDATE_V2RBS_AND_COMPUTE_NEW, (fo_id, veh_obj, currently_assigned_key, ob_rids, locked_rids, v2r_rids_to_compute, vid_list_passed_VRLs, v2rb_list_to_be_updated, vid_external_assignment) ))
        self.number_currently_pending_results += 1
        self.last_function_call = UPDATE_V2RBS_AND_COMPUTE_NEW

    def batch_update_v2rbs_and_compute_new(self, batch_list):
        """ same function as above but multiple inputs are stored in a batch_list """
        self._checkFunctionCall(function_id=UPDATE_V2RBS_AND_COMPUTE_NEW)
        self.q_in.put((UPDATE_V2RBS_AND_COMPUTE_NEW, batch_list))
        self.number_currently_pending_results += 1
        self.last_function_call = UPDATE_V2RBS_AND_COMPUTE_NEW

    def fetch_update_v2rbs_and_compute_new(self):
        """
        function collect results from "update_v2rbs_and_compute_new" and returns these results
        :return: list of all new computed v2rbs (for all vehicles that have been scheduled)
        """
        self._checkFetchCall(UPDATE_V2RBS_AND_COMPUTE_NEW)
        new_v2rb_list = []
        c = 0
        # LOG.debug("fetch new v2rbs: pending {}".format(self.number_currently_pending_results))
        while c < self.number_currently_pending_results:
            x = self.q_out.get()
            ## LOG.debug(f"fetch recieved: {[y.rtv_key for y in x]}")
            new_v2rb_list += x
            c += 1
        self.number_currently_pending_results = 0
        return new_v2rb_list

    def batch_single_insertion(self, fo_id, batch_list, rv_routing_results = {}):
        """
        input: either
            veh_obj_list, vid_assigned_plan_dict, rq_obj, other_plan_rq_dict (rid -> rq_dict for all rids in assigned plan) and additional optional parameters compute_rv and skip_first
            or
            veh_obj_list, vid_assigned_plan_dict, rq_obj
            as list entries of batch_list
        rv_routing_results: dict (o_pos, d_pos) -> (cfv, tt, dis)
        """
        self._checkFunctionCall(function_id=BATCH_SINGLE_INSERTION)
        self.q_in.put((BATCH_SINGLE_INSERTION, (fo_id, batch_list, rv_routing_results)))
        self.number_currently_pending_results += 1
        self.last_function_call = BATCH_SINGLE_INSERTION

    def fetch_batch_single_insertion(self):
        """
        function collect results from "batch_single_insertion" and returns these results
        :return: list of (ass_vid, ass_plan, delta_cfv)
        """
        self._checkFetchCall(BATCH_SINGLE_INSERTION)
        new_insertion_results = []
        c = 0
        # LOG.debug("fetch insertion results: pending {}".format(self.number_currently_pending_results))
        while c < self.number_currently_pending_results:
            x = self.q_out.get()
            ## LOG.debug(f"fetch recieved: {[y.rtv_key for y in x]}")
            for res in x:
                new_insertion_results.append(res)
            c += 1
        self.number_currently_pending_results = 0
        return new_insertion_results

    def batch_optimize_boarding_points_of_vid_assigned_v2rb(self, batch_list):
        self._checkFunctionCall(function_id=OPTIMIZE_BOARDING_POINTS_VID)
        self.q_in.put((OPTIMIZE_BOARDING_POINTS_VID, batch_list))
        self.number_currently_pending_results += 1
        self.last_function_call = OPTIMIZE_BOARDING_POINTS_VID

    def fetch_optimize_boarding_points_of_vid_assigned_v2rb(self):
        """
        function collect results from "batch_optimize_boarding_points_of_vid_assigned_v2rb" and returns these results
        :return: list of (vid, [list of alternative v2rbs for assigned route])
        """
        self._checkFetchCall(OPTIMIZE_BOARDING_POINTS_VID)
        new_v2rb_list = []
        c = 0
        # LOG.debug("fetch new v2rbs: pending {}".format(self.number_currently_pending_results))
        while c < self.number_currently_pending_results:
            x = self.q_out.get()
            ## LOG.debug(f"fetch recieved: {[y.rtv_key for y in x]}")
            for res in x:
                new_v2rb_list.append(res)
            c += 1
        self.number_currently_pending_results = 0
        return new_v2rb_list


#===============================================================================================================#

class ParallelProcess():
    def __init__(self, q_in, q_out, process_id, scenario_parameters, dir_names):
        """ this class carries out the computation tasks distributed from the parallelization manager
        communication is made via two multiprocessing.Queue() objects; the functions to be excuted are communcated via the global communication codes
        this process mimics AlonsoMoraAssignment-classes with only a subset of vehicles (mostly one); 
        preprocessing steps and tree building are carried out in parallel thereby
        :param q_in: mulitprocessing.Queue() -> only inputs from the main core are put here
        :param q_out: multiprocessing.Queue() -> only outputs to the main core are put here
        :param process_id: id defined in the manager class of this process class
        :param scenario_parameters: scenario parameter entries to set up the process
        :param dir_names: dir_name paths from the simulation environment to set up the process
        """
        # routing engine
        self.sleep_time = 0.1   # short waiting time in case another process is still busy

        self.process_id = process_id
        self.q_in = q_in
        self.q_out = q_out


        self.scenario_parameters = scenario_parameters
        self.dir_names = dir_names

        # start log file
        logging.VERBOSE = 5
        logging.addLevelName(logging.VERBOSE, "VERBOSE")
        logging.Logger.verbose = lambda inst, msg, *args, **kwargs: inst.log(logging.VERBOSE, msg, *args, **kwargs)
        logging.LoggerAdapter.verbose = lambda inst, msg, *args, **kwargs: inst.log(logging.VERBOSE, msg, *args, **kwargs)
        logging.verbose = lambda msg, *args, **kwargs: logging.log(logging.VERBOSE, msg, *args, **kwargs)
        if self.scenario_parameters["log_level"]:
            level_str = self.scenario_parameters["log_level"]
            if level_str == "verbose":
                log_level = logging.VERBOSE
            elif level_str == "debug":
                log_level = logging.DEBUG
            elif level_str == "info":
                log_level = logging.INFO
            elif level_str == "warning":
                log_level = logging.WARNING
            else:
                log_level = DEFAULT_LOG_LEVEL
        else:
            log_level = DEFAULT_LOG_LEVEL
        self.log_file = os.path.join(self.dir_names[G_DIR_OUTPUT], f"00_simulation_par_{self.process_id}.log")  # logging from this process into an extra file
        if log_level < logging.INFO:
            streams = [logging.FileHandler(self.log_file), logging.StreamHandler()]
        else: # TODO # progress bar 
            print("Only minimum output to console -> see log-file")
            streams = [logging.FileHandler(self.log_file)]
        logging.basicConfig(handlers=streams,
                            level=log_level, format='%(process)d-%(name)s-%(levelname)s-%(message)s')

        LOG.info(f"Initialization of network and routing engine... on {self.process_id}")   # load the network TODO this should be communicated in a better fashion since this is allready defined
        network_type = self.scenario_parameters[G_NETWORK_TYPE]
        network_dynamics_file = self.scenario_parameters.get(G_NW_DYNAMIC_F, None)
        self.routing_engine = load_routing_engine(network_type, self.dir_names[G_DIR_NETWORK], network_dynamics_file_name=network_dynamics_file)

        self.new_routing_data_loaded = False    # flag to tell if network changed

        self.fo_data = {}   #fo_id -> parameters= (see self.initOp())

        self.sim_time = -1  # simulation time

        #alonso mora preprocessing graphs; also possible for multiple operators (in theory TODO check)
        self.fo_active_requests = {}    #fo_id -> rid -> rq-obj
        self.fo_rid_to_mutually_exclusive_cluster_id = {}   #fo_id -> rid -> cluster_id (copy of dict in alonsomora assigment -> see there for explenation)
        self.fo_mutually_exclusive_cluster_id_to_rids = {}   #fo_id -> cluster_id -> rid -> 1
        self.fo_rid_to_consider_for_global_optimisation = {}    #fo_id -> rid -> 1 for rids that are considered for global optimisation (copy of dict in alonso mora assignment)
        self.fo_rr = {}     #fo_id -> rr_key -> 1
        self.fo_r2v = {}    #fo_id -> rid -> vid -> 1
        self.fo_v2r = {}    #fo_id -> vid -> rid -> 1

        self.current_Xto1_targets = []  # sets targets to compute Xto1 routing queries in parallel

        # LOG.debug("_____________________________________")
        # LOG.debug(f"PARALLEL PROCESS INITIALIZED! on {self.process_id}")
        # LOG.debug("_____________________________________")
        self.time_meassure = time.time()
        self.last_order = None   # some tasks might be recieved double but shouldnt; its checked with this parameter
        self.last_order_fo_id = None

    def run(self):
        """ this is the main function of the parallel process which is supposed to run until the simulation terminates
        here communcatios via the Queue() objects are treated and computation tasks are carried out
        tasks from the main process have the form (function_code, (function_arguments)); the function codes are defined as globals at the top of the file
        """
        try:
            LOG.info("Process {} started work!".format(self.process_id))
            while True:
                x = self.q_in.get() # recieve new task from main core
                # LOG.debug("Queue got {}".format(x[0]))
                if x[0] == KILL:
                    LOG.warning("Process {} got killed!".format(self.process_id))
                    return
                elif x[0] == LOAD_NEW_NETWORK_TRAVEL_TIMES: # update travel times
                    if self.last_order == x[0]:  # information allready here, but missing on another process -> put it back into the queue and wait for small time
                        self.q_in.put(x)
                        time.sleep(self.sleep_time)
                    else:
                        self.new_routing_data_loaded = True
                        self.routing_engine.update_network(*x[1], update_state = True)
                        self.q_out.put(LOAD_NEW_NETWORK_TRAVEL_TIMES)   # show that message is processed
                elif x[0] == INIT_OP: # initialize operator
                    if self.last_order == x[0] and self.last_order_fo_id == x[1][0]:# information allready here, but missing on another process -> put it back into the queue and wait for small time TODO still feasible for multiple operators?
                        # LOG.debug(f"pid {self.process_id} : {INIT_OP}")
                        self.q_in.put(x)
                        time.sleep(self.sleep_time)
                    else:
                        # if x[0] != INIT_OP:
                        #     LOG.error("wrong input for {}! : {}".format(INIT_OP, x))
                        # LOG.verbose(f"first recieved {x}")
                        self._initOp(*x[1])
                        self.last_order_fo_id = x[1][0]
                        self.q_out.put(INIT_OP) # show that message is processed
                elif x[0] == SET_SIMTIME_AND_ACTIVE_REQUESTS:   # update simulation time and recive active requests
                    if self.last_order == x[0]: # information allready here, but missing on another process
                        self.q_in.put(x)
                        time.sleep(self.sleep_time)
                    else:
                        # LOG.verbose(f"first recieved {x}")
                        self._setSimtimeAndActiveRequests(*x[1])
                        self.q_out.put(SET_SIMTIME_AND_ACTIVE_REQUESTS)
                elif x[0] == SET_DYN_SIMTIME_AND_ACTIVE_REQUESTS:   # update simulation time and recive active requests
                    if self.last_order == x[0]: # information allready here, but missing on another process
                        self.q_in.put(x)
                        time.sleep(self.sleep_time)
                    else:
                        LOG.verbose(f"first recieved {x}")
                        self._setDynSimtimeAndActiveRequests(*x[1])
                        self.q_out.put(SET_DYN_SIMTIME_AND_ACTIVE_REQUESTS)
                elif x[0] == COMPUTE_RR:    # task for computing some rr connections
                    res = self._computeRR(*x[1])
                    self.q_out.put(res)
                elif x[0] == SET_RR:    # set all rr connections
                    if self.last_order == x[0]:
                        self.q_in.put(x)
                        time.sleep(self.sleep_time)
                    else:
                        self._setRR(*x[1])
                        self.q_out.put(SET_RR)
                elif x[0] == SET_XTO1_TARGET_LOCATIONS:     # sets targets for which an Xto1-routing query should be computed
                    if self.last_order == x[0]:
                        self.q_in.put(x)
                        time.sleep(self.sleep_time)
                    else:
                        self._setXto1TargetLocations(x[1])
                        self.q_out.put(SET_XTO1_TARGET_LOCATIONS)
                elif x[0] == RETURN_TARGETPOS_IN_TIMERANGE: # return Xto1- routing query results
                    arg_list = x[1]
                    res = []
                    for rid, o_pos, max_range in arg_list:
                        r = self.routing_engine.return_travel_costs_Xto1(self.current_Xto1_targets, o_pos, max_cost_value = max_range)
                        res.append( (rid, o_pos, r) )
                    self.q_out.put(res)
                elif x[0] == SET_RV_RESULTS:
                    if self.last_order == x[0]:
                        self.q_in.put(x)
                        time.sleep(self.sleep_time)
                    else:
                        self.q_out.put(SET_RV_RESULTS)
                        self.routing_engine.add_travel_infos_to_database(*x[1])
                elif x[0] == UPDATE_V2RBS_AND_COMPUTE_NEW:  # update und compute v2rb database for a set of vehicles
                    if type(x[1]) == tuple:
                        res = self._update_v2rbs_and_compute_new(*x[1])
                    else:
                        res = []
                        for args in x[1]:
                            res += self._update_v2rbs_and_compute_new(*args)
                    self.q_out.put(res)
                elif x[0] == OPTIMIZE_BOARDING_POINTS_VID:
                    res = []
                    for args in x[1]:
                        res.append( self._optimize_boarding_points_for_vid(*args) )
                    self.q_out.put(res)
                elif x[0] == BATCH_SINGLE_INSERTION:
                    res = self._batch_single_insertion(*x[1])
                    self.q_out.put(res)
                else:
                    if not self._add_com(x):
                        error_str = "I dont know what this means!\n{}\nOh boy, here I go killing myself again, {}".format(str(x), self.process_id)
                        LOG.error(error_str)
                        return
                self.last_order = x[0]
        except:
            try:
                exc_info = os.sys.exc_info()
                try:
                    for x in range(100):
                        self.q_in.put("kill")
                except:
                    LOG.warning("break breaked!")
            finally:
                traceback.print_exception(*exc_info)
                del exc_info

                return

    def _add_com(self, x):
        """ this function can be overwritten in child classes to define new communication codes
        :param x: stuff recieved from queue
        :return True, if everything worked out, False if error in communication
        """
        return False

    def _initOp(self, fo_id, obj_function_pickle, operator_attributes):
        """  this function initialize operator attributes
        :param fo_id: fleet control id
        :param obj_function_pickle: with pickle serialized objective function for rating vehicle plans
        :param operator_attributes: operator attribute dictionary
        """
        obj_function = pickle.loads(obj_function_pickle)
        # LOG.verbose(f"unpickeld obj func {obj_function}")
        self.fo_data[fo_id] = {"obj_function" : obj_function, "std_bt" : operator_attributes[G_OP_CONST_BT], "add_bt" : operator_attributes[G_OP_ADD_BT],
             "operator_attributes" : operator_attributes}

        self.fo_active_requests[fo_id] = {}    #fo_id -> rid -> rq-obj
        self.fo_rid_to_mutually_exclusive_cluster_id[fo_id] = {}   #fo_id -> rid -> cluster_id (copy of dict in alonsomora assigment -> see there for explenation)
        self.fo_mutually_exclusive_cluster_id_to_rids[fo_id] = {}   #fo_id -> cluster_id -> rid -> 1
        self.fo_rid_to_consider_for_global_optimisation[fo_id] = {}    #fo_id -> rid -> 1 for rids that are considered for global optimisation (copy of dict in alonso mora assignment)
        self.fo_rr[fo_id] = {}     #fo_id -> rr_key -> 1
        self.fo_r2v[fo_id] = {}    #fo_id -> rid -> vid -> 1
        self.fo_v2r[fo_id] = {}    #fo_id -> vid -> rid -> 1

    def _setSimtimeAndActiveRequests(self, fo_id, sim_time, active_requests_dict, rid_to_mutually_exclusive_cluster_id, rid_to_consider_for_global_optimisation):
        """ this function updates the request dictionaries for an operator
        request_to_delete_on_par_processes and del_rr_cons are currently not implemented but are supposed to be used to update the dictionaries dynamically and not from scratch like currently implemented TODO
        :param fo_id: fleet controll id
        :param sim_time: current simulation time
        :param active_request_dict: dict rid -> plan request objects of all currently active requests
        :param rid_to_mutually_exclusive_cluster_id: dict rid -> cluster_id (see init alonosomoraassignment)
        """
        self.sim_time = sim_time
        # LOG.debug("______________________")
        # LOG.debug("New Sim Time {}".format(sim_time))
        # LOG.debug("___________________")
        self.fo_active_requests[fo_id] = active_requests_dict
        self.fo_rid_to_mutually_exclusive_cluster_id[fo_id] = rid_to_mutually_exclusive_cluster_id
        self.fo_mutually_exclusive_cluster_id_to_rids[fo_id] = {}
        for rid, cluster_id in self.fo_rid_to_mutually_exclusive_cluster_id[fo_id].items():
            try:
                self.fo_mutually_exclusive_cluster_id_to_rids[fo_id][cluster_id][rid] = 1
            except:
                self.fo_mutually_exclusive_cluster_id_to_rids[fo_id][cluster_id] = {rid : 1}
        self.fo_rid_to_consider_for_global_optimisation[fo_id] = rid_to_consider_for_global_optimisation

    def _setDynSimtimeAndActiveRequests(self, fo_id, sim_time, new_active_requests_dict, rids_to_del_dict, new_rid_to_mutually_exclusive_cluster_id, new_rid_to_consider_for_global_optimisation):
        """ this function updates the request dictionaries for an operator
        
        """
        self.sim_time = sim_time
        LOG.debug("______________________")
        LOG.debug("New Sim Time {}".format(sim_time))
        LOG.debug("___________________")
        for rid in rids_to_del_dict.keys():
            try:
                del self.fo_active_requests[fo_id][rid]
            except:
                pass
            cluster_rid = self.fo_rid_to_mutually_exclusive_cluster_id[fo_id].get(rid)
            if cluster_rid is not None:
                try:
                    del self.fo_mutually_exclusive_cluster_id_to_rids[fo_id][cluster_rid][rid]
                except KeyError:
                    pass
                if len(self.fo_mutually_exclusive_cluster_id_to_rids[fo_id].get(cluster_rid, {}).keys()) == 0:
                    try:
                        self.fo_mutually_exclusive_cluster_id_to_rids[fo_id][cluster_rid]
                    except KeyError:
                        pass
                try:
                    del self.fo_rid_to_mutually_exclusive_cluster_id[fo_id][rid]
                except KeyError:
                    pass
            try:
                del self.fo_rid_to_consider_for_global_optimisation[fo_id][rid]
            except KeyError:
                pass

        for rid, prq in new_active_requests_dict.items():
            self.fo_active_requests[fo_id][rid] = prq

        for rid, cluster_rid in new_rid_to_mutually_exclusive_cluster_id.items():
            self.fo_rid_to_mutually_exclusive_cluster_id[fo_id][rid] = cluster_rid
            try:
                self.fo_mutually_exclusive_cluster_id_to_rids[fo_id][cluster_rid][rid] = 1
            except:
                self.fo_mutually_exclusive_cluster_id_to_rids[fo_id][cluster_rid] = {rid : 1}

        for rid in new_rid_to_consider_for_global_optimisation.keys():
            self.fo_rid_to_consider_for_global_optimisation[fo_id][rid] = 1


    def _computeRR(self, fo_id, rid_list):
        """ this function computes rr-connections for all rids in rid_list with all active requests for a single operator and returns the graph
        :param fo_id: fleet control id
        :param rid_list: list of request ids to compute rr-connections with all other active requests
        :return: dictionary rr_key -> 1 if rr-connection exists
        """
        return_rr = {}
        active_requests = self.fo_active_requests[fo_id]
        fo_data = self.fo_data[fo_id]
        rid_to_consider_for_global_optimisation = self.fo_rid_to_consider_for_global_optimisation[fo_id]
        # TODO #
        rid_to_mutually_exclusive_cluster_id = self.fo_rid_to_mutually_exclusive_cluster_id.get(fo_id, {})
        # raise NotImplementedError
        const_bt = fo_data["std_bt"]
        add_bt = fo_data["add_bt"]
        for rid in rid_list:
            rq1 = active_requests[rid]
            for rid2 in rid_to_consider_for_global_optimisation.keys():
                if rid != rid2:
                    if rid_to_mutually_exclusive_cluster_id.get(rid) is None or rid_to_mutually_exclusive_cluster_id.get(rid2) is None or rid_to_mutually_exclusive_cluster_id.get(rid) != rid_to_mutually_exclusive_cluster_id.get(rid2):
                        rq2 = active_requests[rid2]
                        if GeneralPoolingFunctions.checkRRcomptibility(rq1, rq2, self.routing_engine, const_bt, dynamic_boarding_time=add_bt):
                            rr_key = getRRKey(rid, rid2)
                            return_rr[rr_key] = 1
        return return_rr

    def _setRR(self, fo_id, rr):
        """ updates the rr-graph for a certain fleet control operator
        :param fo_id: fleet operator id
        :param rr: dictionary rr_key -> 1 if rr-connection exists (this contains ALL feasible rr-connections of the current optimisation problem)
        """
        # try:
        #     self.fo_rr[fo_id].update(rr)    # TODO check if they are also deleted again!
        # except:
        self.fo_rr[fo_id] = rr

    def _setXto1TargetLocations(self, target_locations):
        """ this function sets targets that have to be computed by a routing Xto1 query (mostly vehicle locations)
        :param target_locations: target locations for the routing query (the 1)
        """
        self.current_Xto1_targets = target_locations

    def _return_travel_costs_XtoTargets_in_time_range(self, list_origin_positions, list_target_positions, time_range):
        """ computes a travel time matrix from list_origin_positions to list_target_postions if they are within a certain time_range
        :param list_origin_positions: list of network position as starting point for routing queries
        :param list_target_positions: list of network positions as target points for routing queries
        :param time_range: maximum travel time allowed from being included in the solution
        :return: dict origin_position -> destination_position -> (travel_time, travel_distance) if travel_time <= time_range
        """
        res_dict = {}   #origin -> target -> (tt, dis)
        for target in list_target_positions:
            routing_res = self.routing_engine.return_travel_costs_Xto1(list_origin_positions, target, max_cost_value = time_range)
            for o_pos, _, tt, dis in routing_res:
                try:
                    res_dict[o_pos][target] = (tt, dis) 
                except:
                    res_dict[o_pos] = {target : (tt, dis)}
        return res_dict

    def _update_v2rbs_and_compute_new(self, fo_id, veh_obj, currently_assigned_key, locked_rids, rids_to_compute, vid_list_passed_VRLs = [], v2rb_list_to_be_updated = [], vid_external_assignment = None):
        """ this function recalculates the v2rb-graph for a single vehicle
        :param fo_id: fleet control id
        :param veh_obj: simulation vehicle (struct) object
        :param currently_assigned_key: rtv_key of currently assigned vehicle plan
        :param locked_rids: list of requests ids that are currently locked to the vehicle
        :param rids_to_compute: list of requests ids which rtv-tree have to be built actively
        :param vid_list_passed_VRLs: list of VehicleRouteLegs that have been passed since last optimisation
        :param v2rb_list_to_be_updated: list of v2rb-objects from last optimisation_step that have to be checked if they are still feasible
        :param vid_external_assignment: entry of the dictionary AMA.external_assignments (assignments computed outside of the alonsomora algorithm; used as fallback) 
        :return: list of all new v2rb_objects for this vehicle
        """
        t_start = time.time()
        # TODO fleetcontrol from fo_id or delete?
        # TODO flag for new rounting data needed?
        fleetcontrol_data = self.fo_data[fo_id]
        obj_func = fleetcontrol_data["obj_function"]
        operator_attributes = fleetcontrol_data["operator_attributes"]
        vid = veh_obj.vid

        # LOG.debug(f"build for fo {fo_id} vehicle {vid} trees for rids {rids_to_compute}")

        AMA = AlonsoMoraAssignment(None, self.routing_engine, self.sim_time, obj_func, operator_attributes, veh_objs_to_build={vid : veh_obj})
        vid_to_list_passed_VRLs = {veh_obj.vid : vid_list_passed_VRLs}
        AMA._setUpCurrentInformation(self.sim_time, vid_to_list_passed_VRLs, new_travel_times=self.new_routing_data_loaded)
        
        current_assignments = {vid : currently_assigned_key}
        v2r_locked = {vid: {rid : 1 for rid in locked_rids}}
        v2r = {vid : {rid : 1 for rid in rids_to_compute}} # assumes that rids in rids to compute always have rv!
        request_to_compute = {rid : 1 for rid in rids_to_compute}
        external_assignments = {}
        if vid_external_assignment is not None:
            external_assignments[vid] = vid_external_assignment
        AMA._setAdditionalInitForParallelization(current_assignments, v2r_locked, request_to_compute, self.fo_rr[fo_id], v2r, self.fo_active_requests[fo_id], external_assignments, self.fo_rid_to_mutually_exclusive_cluster_id[fo_id], self.fo_mutually_exclusive_cluster_id_to_rids[fo_id], self.fo_rid_to_consider_for_global_optimisation[fo_id])
        
        t_setup = time.time()
        # # LOG.debug("current rr connections {}".format(self.fo_rr[fo_id]))
        # # LOG.debug("current vr connections {}".format(v2r))
        # # LOG.debug("current ob {}".format(v2r_ob))

        # # LOG.debug("keys before update: {} | incoming {}".format(AMA.rtv_obj.keys(), len(v2rb_list_to_be_updated)))
        if len(v2rb_list_to_be_updated) > 0:
            for v2rb in v2rb_list_to_be_updated:
                AMA._addRtvKey(v2rb.rtv_key, v2rb)
            AMA._updateVehicleDataBase(veh_obj.vid)
        ## LOG.debug("keys after update: {}".format(AMA.rtv_obj.keys()))
        AMA._buildTreeForVid(veh_obj.vid)
        # # LOG.debug("keys after build: {}".format(AMA.rtv_obj.keys()))
        # # LOG.debug(f"number new v2rbs: {len(AMA.getV2RBobjList())}")
        # # LOG.debug("currently assigned: {}".format(currently_assigned_key))
        t_done = time.time()
        # LOG.debug("building took {} | setup took {} | since last call {}".format(t_done - t_setup, t_setup - t_start, t_start - self.time_meassure))
        self.time_meassure = time.time()
        return AMA._getV2RBobjList()

    def _optimize_boarding_points_for_vid(self, fo_id, veh_obj, assigned_key, vid_v2r_locked_dict, v2rb_obj_list, veh_build_tree_time_out):
        """ this function is used to calculate "optimize_boarding_points_of_vid_assigned_v2rb" from AlonsoMoraAssignment in parallel
        check when called for inputs
        :return: vid, list of alternative v2rbs for other boarding points
        """
        fleetcontrol_data = self.fo_data[fo_id]
        operator_attributes = fleetcontrol_data["operator_attributes"]
        obj_func = fleetcontrol_data["obj_function"]
        vid = veh_obj.vid
        v2r_locked = {vid : vid_v2r_locked_dict}
        r2v_locked = {}
        for vid, rid_dict in v2r_locked.items():
            for rid in rid_dict.keys():
                r2v_locked[rid] = vid
        mutually_excl_cluster_dict = {}
        for rid in getRidsFromRTVKey(assigned_key):
            base_rid = self.fo_rid_to_mutually_exclusive_cluster_id[fo_id].get(rid, rid)
            sub_rids = self.fo_mutually_exclusive_cluster_id_to_rids[fo_id].get(base_rid, {rid : 1}).keys()
            mutually_excl_cluster_dict[base_rid] = sub_rids
        alternative_v2rb_list = optimize_boarding_points_of_vid_assigned_v2rb(None, self.routing_engine, self.sim_time, operator_attributes, obj_func, veh_obj,
                assigned_key, mutually_excl_cluster_dict,
                r2v_locked, v2r_locked, self.fo_active_requests[fo_id], v2rb_obj_list)
        return (vid, alternative_v2rb_list)

    def _batch_single_insertion(self, fo_id, batch_list, rv_routing_results):
        """ this function computes an insertion heuristic for each entry in the batch list
        if an insertion is not possible, the results is not returned
        :return: list of (ass_vid, ass_plan, change in cfv)
        """
        self.routing_engine.add_travel_infos_to_database(rv_routing_results)
        #single_insertion([self.veh_objs[vid]], {vid : assigned_plan}, self.active_requests[rid], self.objective_function, self.routing_engine, self.active_requests, self.sim_time, self.std_bt, self.add_bt)
        fleetcontrol_data = self.fo_data[fo_id]
        std_bt = fleetcontrol_data["std_bt"]
        add_bt = fleetcontrol_data["add_bt"]
        obj_func = fleetcontrol_data["obj_function"]
        res_list = []
        for x in batch_list:
            if len(x) == 3:
                veh_obj_list, vid_assigned_plan_dict, rq_obj = x
                ass_vid, ass_plan, delta_cfv = single_insertion(veh_obj_list, vid_assigned_plan_dict, rq_obj, obj_func, self.routing_engine, self.fo_active_requests[fo_id], self.sim_time, std_bt, add_bt)
                if ass_plan is not None:
                    res_list.append( (ass_vid, ass_plan, delta_cfv) )
            else:
                veh_obj_list, vid_assigned_plan_dict, rq_obj, other_plan_rq_dict = x
                check_rv=True
                skip_first_position_insertion=False
                if len(x) >= 5:
                    check_rv = x[4]
                if len(x) >= 6:
                    skip_first_position_insertion=x[6]
                # LOG.debug("single insertion batch")
                # LOG.debug("{} {} {} {}".format(veh_obj_list, vid_assigned_plan_dict, rq_obj, other_plan_rq_dict))
                ass_vid, ass_plan, delta_cfv = single_insertion(veh_obj_list, vid_assigned_plan_dict, rq_obj, obj_func, self.routing_engine, other_plan_rq_dict, self.sim_time, std_bt, add_bt, check_rv=check_rv, skip_first_position_insertion=skip_first_position_insertion)
                if ass_plan is not None:
                    res_list.append( (ass_vid, ass_plan, delta_cfv) )
        return res_list

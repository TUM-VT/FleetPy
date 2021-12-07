""" this file is only used for the Moia-simulation framework 
to parallize first-last-mile offer creations """

from multiprocessing import Process, Queue, Pipe
import time
import math
import dill as pickle
import pandas as pd
from src.misc.globals import *
from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraParallelization import ParallelizationManager, ParallelProcess
from src.fleetctrl.pooling.immediate.insertion import single_insertion
from src.fleetctrl.pooling.immediate.searchVehicles import veh_search_for_immediate_request, veh_search_for_reservation_request

import logging 
LOG = logging.getLogger(__name__)

from src.fleetctrl.pooling.batch.AlonsoMora.comcodes import *

def startProcess(q_in, q_out, process_id, scenario_parameters, dir_names, veh_timeout_per_tree):
    PP = MoiaParallelProcess(q_in, q_out, process_id, scenario_parameters, dir_names, veh_timeout_per_tree)
    LOG.info(f"time to run PP {process_id}")
    PP.run()

class MoiaParallelizationManager(ParallelizationManager):
    def __init__(self, number_cores, scenario_parameters, dir_names, veh_timeout_per_tree):
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
        :param veh_timeout_per_tree: timeout in seconds after which the building of the rtv-tree of a single vehicle is aborted
        """
        self.number_cores = number_cores

        self.q_in = Queue() #communication queues
        self.q_out = Queue()

        self.processes = [Process(target = startProcess, args = (self.q_in, self.q_out, i, scenario_parameters, dir_names, veh_timeout_per_tree)) for i in range(self.number_cores)]    # start processes
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
        self.update_offer_id = 0

    def _checkFunctionCall(self, function_id = -1):
        """ check if this function call is feasible respective to the last function call and if all results are fetched
        raises error if not
        :param function_id: corresponding communication code
        """
        if function_id != self.last_function_call:
            if (function_id == RESERVATION_INSERTION and self.last_function_call != IMMEDIATE_INSERTION) or (function_id == IMMEDIATE_INSERTION and self.last_function_call != RESERVATION_INSERTION):
                if self.number_currently_pending_results > 0 or len(self.fetched_result_list) > 0:
                    print("results or computations from parallel processes are not cleared! use fetch functions to retrieve results!")
                    print("not retrieved results from : {}".format(self.last_function_call))
                    raise SyntaxError

    def additionalInitOp(self, fo_id, operator_attributes):
        """ initializes additional operator attributes to create offers in parallel for moia on the parallel processes
        :param fo_id: fleet control id
        :param operator_attributes: operator attributes dict
        """
        c = 0
        for i in range(self.number_cores):
            # LOG.debug(f"Queue put {INIT_OP}")
            self.q_in.put( (ADD_INIT_OP, (fo_id, operator_attributes) ) )
            c += 1
        self.last_function_call = ADD_INIT_OP
        while c > 0:
            x = self.q_out.get()
            c -= 1

    def updateOfferVehiclePlans(self, fo_id, sim_vehicles, new_vid_to_vehplans_wo_reloc, new_accept_rqs):
        """ updates vehicle plan dictionary if assignment changed either for a new request phase or if a request accepted
        an offer and is finally assigned to a vehicle plan
        :param fo_id: operator id
        :param sim_vehicles: list of simulator vehicle structs (if empty, no update is performed)
        :param new_vid_to_vehplans_wo_reloc: dict vid -> vehplan_without_reloc (only changes necessary)
        :param new_accepted_rqs: dict rid -> rq object if rq accepted an offer
        """
        c = 0
        for i in range(self.number_cores):
            LOG.debug(f"Queue put {UPDATE_VEHICLE_PLANS_OFFER}")
            self.q_in.put((UPDATE_VEHICLE_PLANS_OFFER, (fo_id, sim_vehicles, new_vid_to_vehplans_wo_reloc, new_accept_rqs), self.update_offer_id) )
            c += 1
        self.last_function_call = UPDATE_VEHICLE_PLANS_OFFER
        self.update_offer_id += 1
        if self.update_offer_id > 1000:
            self.update_offer_id = 0
        while c > 0:
            x = self.q_out.get()
            c -= 1

    def reservationPrqInsertion(self, fo_id, prq, sim_time):
        """ this function sends insertion tasks to the parallel processes
        results can be retrieved with "fetchInsertionResults"
        :param fo_id: operator id
        :param prq: plan request object
        :param sim_time: simulation time
        """
        self._checkFunctionCall()
        LOG.debug(f"Queue put {RESERVATION_INSERTION}")
        self.q_in.put( (RESERVATION_INSERTION, (fo_id, prq, sim_time) ) )
        self.number_currently_pending_results += 1

        self.last_function_call = RESERVATION_INSERTION

    def immediatePrqInsertion(self, fo_id, prq, sim_time):
        """ this function sends insertion tasks to the parallel processes
        results can be retrieved with "fetchInsertionResults"
        :param fo_id: operator id
        :param prq: plan request object
        :param sim_time: simulation time
        """
        self._checkFunctionCall()
        LOG.debug(f"Queue put {IMMEDIATE_INSERTION}")
        self.q_in.put( (IMMEDIATE_INSERTION, (fo_id, prq, sim_time) ) )
        self.number_currently_pending_results += 1

        self.last_function_call = IMMEDIATE_INSERTION

    def fetchInsertionResults(self):
        """ this function retrieves all results from insertion tasks
        :return: list of (rid, best_ass_vid, best_plan, best_obj)
        """
        #self._checkFetchCall(COMPUTE_RR)
        LOG.debug("fetch {} insertion results".format(self.number_currently_pending_results))
        c = 0
        insertion_list = []
        while c < self.number_currently_pending_results:
            x = self.q_out.get()
            insertion_list.append(x)
            c += 1
        self.number_currently_pending_results = 0
        return insertion_list


# ========================================================================================= #
class MoiaParallelProcess(ParallelProcess):
    def __init__(self, q_in, q_out, process_id, scenario_parameters, dir_names, veh_tree_build_timeout):
        super().__init__(q_in, q_out, process_id, scenario_parameters, dir_names, veh_tree_build_timeout)
        self.fo_vid_to_vehplans_without_offer = {}
        self.fo_to_fltctrl_mimic = {}
        self.last_update_offer = -1

    def _add_com(self, x):
        #print("recieve", x)
        if x[0] == ADD_INIT_OP: # add init op for moia
            if self.last_order == x[0] and self.last_order_fo_id == x[1][0]:# information allready here, but missing on another process -> put it back into the queue and wait for small time
                self.q_in.put(x)
                time.sleep(self.sleep_time)
            else:
                self._addInitOp(*x[1])
                self.last_order_fo_id = x[1][0]
                self.q_out.put(ADD_INIT_OP) # show that message is processed
            return True
        elif x[0] == UPDATE_VEHICLE_PLANS_OFFER: # add init op for moia
            if self.last_order == x[0] and self.last_order_fo_id == x[1][0] and self.last_update_offer == x[2]:# information allready here, but missing on another process -> put it back into the queue and wait for small time
                self.q_in.put(x)
                time.sleep(self.sleep_time)
            else:
                self._updateOfferVehiclePlans(*x[1])
                self.last_order_fo_id = x[1][0]
                self.q_out.put(UPDATE_VEHICLE_PLANS_OFFER) # show that message is processed
                self.last_update_offer = x[2]
            return True
        elif x[0] == RESERVATION_INSERTION:    # task for computing reservation insertion
            res = self._reservationInsertion(*x[1])
            self.q_out.put(res)
            return True
        elif x[0] == IMMEDIATE_INSERTION:    # task for computing immediate insertion
            res = self._immediateInsertion(*x[1])
            self.q_out.put(res)
            return True
        else:
            return False # unknown message -> raise error in parent class
        return True

    def _addInitOp(self, fo_id, operator_attributes):
        """ this function creates instances of FleetControlMimic to use function that need a fleetctrl class as input
        :param fo_id: operator id
        :param operator_attributes: op atts dict
        """
        self.fo_to_fltctrl_mimic[fo_id] = FleetControlMimic(fo_id, operator_attributes, self.routing_engine)
        self.fo_vid_to_vehplans_without_offer[fo_id] = {}

    def _updateOfferVehiclePlans(self, fo_id, sim_vehicles, new_vid_to_vehplans_wo_reloc, new_accept_rqs):
        """ updates vehicle plan dictionary if assignment changed either for a new request phase or if a request accepted
        an offer and is finally assigned to a vehicle plan
        :param fo_id: operator id
        :param sim_vehicles: list of SimVehicleStructs
        :param new_vid_to_vehplans_wo_reloc: dict vid -> vehplan_without_reloc (only changes necessary)
        :param new_accepted_rqs: dict rid -> rq object if rq accepted an offer
        """
        LOG.debug("updateOfferVehiclePlans")
        fltctrl_mimic = self.fo_to_fltctrl_mimic[fo_id]
        if len(sim_vehicles) > 0:
            fltctrl_mimic.sim_vehicles = sim_vehicles
        for vid, vehplan_wo_reloc in new_vid_to_vehplans_wo_reloc.items():
            self.fo_vid_to_vehplans_without_offer[fo_id][vid] = vehplan_wo_reloc
            fltctrl_mimic.veh_plans[vid] = vehplan_wo_reloc
        for rid, prq in new_accept_rqs.items():
            self.fo_active_requests[fo_id][rid] = prq

    def _reservationInsertion(self, fo_id, prq, sim_time):
        """ this function calculates the insertion of a reservation prq into the current vehplans
        :param fo_id: operator id
        :param prq: plan request obj
        :param sim_time: simulation time
        :return: tuple (rid, best_ass_vid, best_plan, best_obj)
        """
        rid = prq.get_rid_struct()
        LOG.debug(f"reservation insertion {fo_id} {rid}, {sim_time}")
        self.fo_active_requests[fo_id][rid] = prq
        fleetcontrol_data = self.fo_data[fo_id]
        std_bt = fleetcontrol_data["std_bt"]
        add_bt = fleetcontrol_data["add_bt"]
        obj_func = fleetcontrol_data["obj_function"]
        vehicles_to_consider, _ = self._get_vehicles_to_consider_for_reservation_request(fo_id, prq,
                                                                                            sim_time)
        ass_vid, ass_plan, delta_obj = single_insertion(vehicles_to_consider,
                                                        {veh.vid: self.fo_vid_to_vehplans_without_offer[fo_id][veh.vid] for veh in
                                                            vehicles_to_consider}, prq, obj_func,
                                                        self.routing_engine, self.fo_active_requests[fo_id], sim_time,
                                                        std_bt, add_bt, check_rv=False,
                                                        skip_first_position_insertion=True)
        del self.fo_active_requests[fo_id][rid]
        LOG.info("reservation insertion {} {} {}".format(ass_vid, ass_plan, delta_obj))
        return rid, ass_vid, ass_plan, delta_obj

    def _immediateInsertion(self, fo_id, prq, sim_time):
        """ this function calculates the insertion of an immediate prq into the current vehplans
        :param fo_id: operator id
        :param prq: plan request obj
        :param sim_time: simulation time
        :return: tuple (rid, best_ass_vid, best_plan, best_obj)
        """
        rid = prq.get_rid_struct()
        LOG.debug(f"immediate insertion {fo_id} {rid}, {sim_time}")
        self.fo_active_requests[fo_id][rid] = prq
        fleetcontrol_data = self.fo_data[fo_id]
        std_bt = fleetcontrol_data["std_bt"]
        add_bt = fleetcontrol_data["add_bt"]
        obj_func = fleetcontrol_data["obj_function"]
        vehicles_to_consider, _ = self._get_vehicles_to_consider_for_immediate_request(fo_id, prq,
                                                                                            sim_time)
        ass_vid, ass_plan, delta_obj = single_insertion(vehicles_to_consider,
                                                        {veh.vid: self.fo_vid_to_vehplans_without_offer[fo_id][veh.vid] for veh in
                                                            vehicles_to_consider}, prq, obj_func,
                                                        self.routing_engine, self.fo_active_requests[fo_id], sim_time,
                                                        std_bt, add_bt, check_rv=False)
        del self.fo_active_requests[fo_id][rid]
        LOG.info("immediate insertion {} {} {}".format(ass_vid, ass_plan, delta_obj))
        return rid, ass_vid, ass_plan, delta_obj

    def _get_vehicles_to_consider_for_immediate_request(self, fo_id, prq, simulation_time):
        """ this function returns a list of vehicles that should be considered for insertion of a plan request
        which should be assigned immediately
        :param fo_id: operator_id
        :param prq: corresponding plan request
        :type prq: PlanRequest
        :return: list of vehicle objects considered for assignment, routing_results_dict ( (o_pos, d_pos) -> (cfv, tt, dis))
        :rtype: tuple of list of SimulationVehicle, dict
        """
        fleetctr_mimic = self.fo_to_fltctrl_mimic[fo_id]
        rv_vehicles, rv_results_dict = veh_search_for_immediate_request(simulation_time, prq, fleetctr_mimic)
        LOG.info("found {} for immediate rq".format(len(rv_vehicles)))
        return rv_vehicles, rv_results_dict

    def _get_vehicles_to_consider_for_reservation_request(self, fo_id, prq, simulation_time):
        """ this function returns a list of vehicles that should be considered for insertion of a plan request
        which pick up is far in the future
        :param fo_id: operator_id
        :param prq: corresponding plan request
        :type prq: PlanRequest
        :return: list of vehicle objects considered for assignment, routing_results_dict ( (o_pos, d_pos) -> (cfv, tt, dis))
        :rtype: tuple of list of SimulationVehicle, dict
        """
        fleetctr_mimic = self.fo_to_fltctrl_mimic[fo_id]
        if fleetctr_mimic.rv_heuristics.get(G_RH_R_NWS) is not None:
            rv_vehicles_dict = veh_search_for_reservation_request(simulation_time, prq, fleetctr_mimic)
            LOG.info("found {} for reservation rq".format(len(rv_vehicles_dict.keys())))
            return [fleetctr_mimic.sim_vehicles[vid] for vid in rv_vehicles_dict.keys()], {}
        else:
            return self._get_vehicles_to_consider_for_immediate_request(fo_id, prq, simulation_time)


# ======================================================================================== #
class FleetControlMimic():
    def __init__(self, op_id, operator_attributes, routing_engine):
        self.op_id = op_id
        self.routing_engine = routing_engine
        #
        self.sim_vehicles = []
        self.veh_plans = {}
        # RV and insertion heuristics
        # ---------------------------
        self.rv_heuristics = {}
        self.insertion_heuristics = {}
        rv_im_max_routes = operator_attributes.get(G_RH_I_NWS)
        if not pd.isnull(rv_im_max_routes):
            self.rv_heuristics[G_RH_I_NWS] = int(rv_im_max_routes)
        rv_res_max_routes = operator_attributes.get(G_RH_R_NWS)
        if not pd.isnull(rv_res_max_routes):
            self.rv_heuristics[G_RH_R_NWS] = int(rv_res_max_routes)
        rv_nr_veh_direction = operator_attributes.get(G_RVH_DIR)
        if not pd.isnull(rv_nr_veh_direction):
            self.rv_heuristics[G_RVH_DIR] = int(rv_nr_veh_direction)
        rv_nr_least_load = operator_attributes.get(G_RVH_LWL)
        if not pd.isnull(rv_nr_least_load):
            self.rv_heuristics[G_RVH_LWL] = rv_nr_least_load
        rv_nr_rr = operator_attributes.get(G_RVH_AM_RR)
        if not pd.isnull(rv_nr_rr):
            self.rv_heuristics[G_RVH_AM_RR] = rv_nr_rr
        rv_nr_ti = operator_attributes.get(G_RVH_AM_TI)
        if not pd.isnull(rv_nr_ti):
            self.rv_heuristics[G_RVH_AM_TI] = rv_nr_ti
        ih_keep_x_best_plans_per_veh = operator_attributes.get(G_VPI_KEEP)
        if not pd.isnull(ih_keep_x_best_plans_per_veh):
            self.insertion_heuristics[G_VPI_KEEP] = int(ih_keep_x_best_plans_per_veh)
        max_rv_con = operator_attributes.get(G_RA_MAX_VR)
        if not pd.isnull(max_rv_con):
            self.rv_heuristics[G_RA_MAX_VR] = int(max_rv_con)
        max_req_plans = operator_attributes.get(G_RA_MAX_RP)
        if not pd.isnull(max_req_plans):
            self.rv_heuristics[G_RA_MAX_RP] = int(max_req_plans)

        self.zones = None
        self.pos_veh_dict_time = -1
        self.pos_veh_dict = {}
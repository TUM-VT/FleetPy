from abc import abstractmethod, ABCMeta
import logging
from typing import Callable, Dict, List, Any
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.fleetctrl.pooling.immediate.insertion import simple_remove
import numpy as np
from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from src.simulation.Legs import VehicleRouteLeg
from src.simulation.Vehicles import SimulationVehicle
from src.routing.NetworkBase import NetworkBase
from src.misc.globals import *
LOG = logging.getLogger(__name__)

""" The idea is to have a fixed necessary set of functions for all BatchAssignmentAlgorithms
when implementing a new ride-pooling assginment algorithm these functions have to be implemented to create the optimisation problem
-> only these functions should be non-private functions
"""

class SimulationVehicleStruct():
    """ this class can be used to get basic vehicle information for optimisation """
    def __init__(self, simulation_vehicle : SimulationVehicle, assigned_veh_plan : VehiclePlan, sim_time : int, routing_engine : NetworkBase):
        self.op_id = simulation_vehicle.op_id
        self.vid = simulation_vehicle.vid

        self.status = simulation_vehicle.status
        self.pos = simulation_vehicle.pos
        self.soc = simulation_vehicle.soc
        self.pax = simulation_vehicle.pax[:]    # rq_obj

        self.cl_start_time = simulation_vehicle.cl_start_time

        self.veh_type = simulation_vehicle.veh_type
        self.max_pax = simulation_vehicle.max_pax
        self.max_parcels = simulation_vehicle.max_parcels
        self.daily_fix_cost = simulation_vehicle.daily_fix_cost
        self.distance_cost = simulation_vehicle.distance_cost
        self.battery_size = simulation_vehicle.battery_size
        self.range = simulation_vehicle.range
        self.soc_per_m = simulation_vehicle.soc_per_m

        # assigned route = list of assigned vehicle legs (copy and remove stationary process (TODO?))
        self.assigned_route = [VehicleRouteLeg(x.status, x.destination_pos, x.rq_dict, power=x.power, duration=x.duration, route=x.route, locked=x.locked, earliest_start_time=x.earliest_start_time)
                               for x in simulation_vehicle.assigned_route]

        self.locked_planstops = VehiclePlan(self, sim_time, routing_engine, [])
        self.set_locked_vehplan(assigned_veh_plan, sim_time, routing_engine)

    def __str__(self):
        return f"veh struct {self.vid} at pos {self.pos} leg status {self.status} ob {[rq.get_rid_struct() for rq in self.pax]}"

    def compute_soc_charging(self, power : float, duration : float) -> float:
        """This method returns the SOC change for charging a certain amount of power for a given duration.

        :param power: power of charging process
        :type power: float
        :param duration: duration of charging process
        :type duration: float
        :return: delta SOC
        :rtype: float
        """
        return power * duration / self.battery_size

    def compute_soc_consumption(self, distance : float) -> float:
        """This method returns the SOC change for driving a certain given distance.

        :param distance: driving distance in meters
        :type distance: float
        :return: delta SOC (positive value!)
        :rtype: float
        """
        return distance * self.soc_per_m

    def set_locked_vehplan(self, assigned_veh_plan : VehiclePlan, sim_time : float, routing_engine : NetworkBase):
        """ this method filters the currently locked vehicle plan from the currently assigned vehicle plan
        in fleet control to keep these vehicle plans after a new assignment
        :param assigned_veh_plan: currently assigned vehicle plan
        :type assigned_veh_plan: VehiclePlan
        """
        if assigned_veh_plan is None:
            self.locked_vehplan = VehiclePlan(self, sim_time, routing_engine, [])
        else:
            locked_ps = []
            for ps in assigned_veh_plan.list_plan_stops:
                if ps.is_locked() or ps.is_locked_end():
                    locked_ps.append(ps.copy())
            if len(locked_ps) > 0:
                LOG.debug("locked ps for vid {}: {}".format(self.vid, [str(x) for x in locked_ps]))
            self.locked_vehplan = VehiclePlan(self, sim_time, routing_engine, locked_ps, external_pax_info=assigned_veh_plan.pax_info.copy())

    def has_locked_vehplan(self) -> bool:
        if len(self.locked_vehplan.list_plan_stops) > 0:
            return True
        else:
            return False

    def get_nr_pax_without_currently_boarding(self) -> int:
        """ this method returns the current number of pax for the use of setting the inititial stats for 
        the update_tt_... function in fleetControlBase.py.
        In case the vehicle is currently boarding, this method doesnt count the number of currently boarding customers
        the reason is that boarding and deboarding of customers is recognized during different timesteps of the vcl
        :return: number of pax without currently boarding ones
        :rtype: int
        """
        if self.status == VRL_STATES.BOARDING:
            return sum([rq.nr_pax for rq in self.pax if not rq.is_parcel]) - sum([rq.nr_pax for rq in self.assigned_route[0].rq_dict.get(1, []) if not rq.is_parcel])
        else:
            return sum([rq.nr_pax for rq in self.pax if not rq.is_parcel if not rq.is_parcel])
        
    def get_nr_parcels_without_currently_boarding(self) -> int:
        """ this method returns the current number of parcels for the use of setting the inititial stats for 
        the update_tt_... function in fleetControlBase.py.
        In case the vehicle is currently boarding, this method doesnt count the number of currently boarding parcels
        the reason is that boarding and deboarding of parcels is recognized during different timesteps of the vcl
        :return: number of parcels without currently boarding ones
        :rtype: int
        """
        if self.status == VRL_STATES.BOARDING:
            return sum([rq.nr_pax for rq in self.pax if rq.is_parcel]) - sum([rq.nr_pax for rq in self.assigned_route[0].rq_dict.get(1, []) if rq.is_parcel])
        else:
            return sum([rq.nr_pax for rq in self.pax if rq.is_parcel])

INPUT_PARAMETERS_BatchAssignmentAlgorithmBase = {
    "doc" :  """This class is used to compute new vehicle assignments with an algorithm
        this class should be initialized when the corresponding fleet controller is initialized """,
    "inherit" : None,
    "input_parameters_mandatory": [],
    "input_parameters_optional": [
        ],
    "mandatory_modules": [],
    "optional_modules": []
}

class BatchAssignmentAlgorithmBase(metaclass=ABCMeta):

    def __init__(self, fleetcontrol : FleetControlBase, routing_engine : NetworkBase, sim_time : int, obj_function : Callable,
                 operator_attributes : dict, optimisation_cores : int=1, seed :int = 6061992, veh_objs_to_build : Dict[int, SimulationVehicleStruct]={}):
        """This class is used to compute new vehicle assignments with an algorithm
        this class should be initialized when the corresponding fleet controller is initialized
        :param fleetcontrol : fleetcontrol object, which uses this assignment algorithm
        :param routing_engine : routing_engine object
        :param sim_time : (int) current simulation time
        :param obj_function : obj_function to rate a vehicle plan
        :param operator_attributes : input parameter dict for operator attributes
        :param seed : random seed
        :param veh_objs_to_build: dict vid -> SimulationVehicleStruct which will be considered in opt. if empty dict, vehicles from fleetcontrol will be taken
        """
        self.fleetcontrol = fleetcontrol                      
        if fleetcontrol is not None:
            self.fo_id = fleetcontrol.op_id
            self.solver = fleetcontrol.solver
        self.routing_engine = routing_engine
        self.sim_time = sim_time
        self.std_bt = operator_attributes.get(G_OP_CONST_BT, 0)
        self.add_bt = operator_attributes.get(G_OP_ADD_BT, 0)
        self.objective_function = obj_function
        self.optimisation_cores = optimisation_cores
        self.operator_attributes = operator_attributes

        np.random.seed(seed)

        self.active_requests : Dict[Any, PlanRequest] = {}           # rid -> request-Object
        self.veh_objs : Dict[int, SimulationVehicleStruct] = {}
        if len(veh_objs_to_build.keys()) == 0 and self.fleetcontrol is not None:
            for veh_obj in self.fleetcontrol.sim_vehicles:
                veh_obj_struct = SimulationVehicleStruct(veh_obj, self.fleetcontrol.veh_plans.get(veh_obj.vid, VehiclePlan(veh_obj, self.sim_time, self.routing_engine, [])), sim_time, self.routing_engine)
                self.veh_objs[veh_obj.vid] = veh_obj_struct
        else:
            self.veh_objs = veh_objs_to_build

        #
        self.rid_to_consider_for_global_optimisation : Dict[Any, int] = {}      # rid -> 1 if the request is actively treated in the global optimisation (useful for local optimisation later)

        # constraints for optimisation
        # 1) requests locked to single vehicle (i.e. on-board) | rid is always a base_rid!
        self.v2r_locked : Dict[int, Dict[Any, int]] = {}    #vid -> rid -> 1 | rids currently locked to vid
        self.r2v_locked  : Dict[Any, Dict[int, int]]= {}    #rid -> vid -> 1 | rids currently locked to vid
        for vid in self.veh_objs.keys():
            self.v2r_locked[vid] = {}
        # 2) assignment constraint (defines if requests have to be assigned) | rid is always a base_rid!
        self.unassigned_requests : Dict[Any, int] = {}   # rid -> 1  # for requests that dont have to be assigned (i.e. no accepted offer yet)
        # 3) only one of those requests is allowed to be assigned (i.e. variable boarding locations)
        self.mutually_exclusive_cluster_id_to_rids : Dict[Any, Dict[Any, 1]] = {}    # base_id -> rid -> 1 if rid is part of a cluster of rids which are mutually exclusive (only one of them can be assigned)
        self.rid_to_mutually_exclusive_cluster_id : Dict[Any, Any] = {}      # rid -> base_id if rid is part of this cluster of mutually exclusive rids

    def register_parallelization_manager(self, parallelzation_manager):
        """ this method is used to register a parallelization manager for a batch optimization algorithm. this is only needed
        if the parallelization is done via seperated processes being active over the whole simulation"""
        pass

    @abstractmethod
    def compute_new_vehicle_assignments(self, sim_time : int, vid_to_list_passed_VRLs : Dict[int, List[VehicleRouteLeg]], veh_objs_to_build : Dict[int, SimulationVehicle] = {}, new_travel_times : bool = False, build_from_scratch : bool = False):
        """ this function computes new vehicle assignments based on current fleet information
        param sim_time : current simulation time
        param vid_to_list_passed_VRLs : (dict) vid -> list_passed_VRLs; needed to update database and V2RBs
        :param veh_objs_to_build: only these vehicles will be optimized (all if empty) dict vid -> SimVehicle obj only for special cases needed in current alonso mora module
        :param new_travel_times : bool; if traveltimes changed in the routing engine
        :param build_from_scratch : only for special cases needed in current alonso mora module
        """
        pass

    def add_new_request(self, rid : Any, prq : PlanRequest, consider_for_global_optimisation : bool = True, is_allready_assigned : bool = False):
        """ this function adds a new request to the modules database and set entries that
        possible v2rbs are going to be computed in the next opt step.
        :param rid: plan_request_id
        :param prq: plan_request_obj 
        :param consider_for_global_optimisation: if false, it will not be looked for better solutions in global optimisation
                    but it is still part of the solution, if it is allready assigned
        :param is_allready_assigned: if not considered for global optimisation, this flag indicates, if the rid is allready assigned
            in the init solution"""
        LOG.debug("new request enters pooling optimisation system: {}".format(rid))
        LOG.debug("rq: {}".format(prq))
        self.active_requests[rid] = prq
        if consider_for_global_optimisation:
            self.rid_to_consider_for_global_optimisation[rid] = 1
        if not is_allready_assigned:
            self.unassigned_requests[rid] = 1

    def set_request_assigned(self, rid : Any):
        """ this function marks a request as assigned. its assignment is therefor treatet as hard constraint in the optimization problem formulation
        also all requests with the same mutually_exclusive_cluster_id are set as assigned
        :param rid: plan_request_id """
        LOG.debug(f"set request {rid} as assigned!")
        try:
            del self.unassigned_requests[rid]
        except:
            pass
        for other_rid in self._get_all_other_subrids_associated_to_this_subrid(rid):
            if other_rid != rid:
                try:
                    del self.unassigned_requests[rid]
                except KeyError:
                    pass

    def set_database_in_case_of_boarding(self, rid : Any, vid : int):
        """ deletes all rtvs without rid from vid (rid boarded to vid)
        deletes all rtvs with rid for all other vehicles 
        this function should be called in the fleet operators acknowledge boarding in case rid boards vid
        :param rid: plan_request_id
        :param vid: vehicle obj id """
        try:
            self.v2r_locked[vid][rid] = 1
        except KeyError:
            self.v2r_locked[vid] = {rid : 1}
        self.r2v_locked[rid] = vid

    def set_database_in_case_of_alighting(self, rid : Any, vid : int):
        """ this function deletes all database entries of rid and sets the new assignement by deleting rid of the currently
        assigned v2rb; the database of vid will be completely rebuild in the next opt step
        this function should be called in the fleet operators acknowledge alighting in case rid alights vid
        :param rid: plan_request_id
        :param vid: vehicle obj id """
        LOG.debug("locked dicts: {} | {}".format(self.v2r_locked, self.r2v_locked))
        del self.v2r_locked[vid][rid]
        del self.r2v_locked[rid]
        sub_rids = self._get_all_rids_representing_this_base_rid(rid)
        for sub_rid in list(sub_rids):
            self.delete_request(sub_rid)

    @abstractmethod
    def get_optimisation_solution(self, vid : int) -> VehiclePlan:
        """ returns optimisation solution for vid
        :param vid: vehicle id
        :return: vehicle plan object for the corresponding vehicle
        """
        pass

    @abstractmethod
    def set_assignment(self, vid : int, assigned_plan : VehiclePlan, is_external_vehicle_plan : bool = False):
        """ sets the vehicleplan as assigned in the algorithm database; if the plan is not computed within the this algorithm, the is_external_vehicle_plan flag should be set to true
        :param vid: vehicle id
        :param assigned_plan: vehicle plan object that has been assigned
        :param is_external_vehicle_plan: should be set to True, if the assigned_plan has not been computed within this algorithm
        """
        if assigned_plan is not None:
            for rid in assigned_plan.get_involved_request_ids():
                self.set_request_assigned(rid)

    @abstractmethod
    def get_current_assignment(self, vid : int) -> VehiclePlan: # TODO same as get_optimisation_solution (delete?)
        """ returns the vehicle plan assigned to vid currently
        :param vid: vehicle id
        :return: vehicle plan
        """
        pass

    def clear_databases(self):
        """ this function resets database entries after an optimisation step
        """
        pass

    def delete_request(self, rid : Any):
        """ this function deletes a request from all databases
        :param rid: plan_request_id """
        try:
            del self.active_requests[rid]
        except:
            pass
        try:
            del self.unassigned_requests[rid]
        except:
            pass
        self._delete_subrid(rid)
        try:
            del self.rid_to_consider_for_global_optimisation[rid]
        except:
            pass

    def get_vehicle_plan_without_rid(self, veh_obj : SimulationVehicle, vehicle_plan : VehiclePlan, rid_to_remove : Any, sim_time : int) -> VehiclePlan:
        """this function returns the best vehicle plan by removing the rid_to_remove from the vehicle plan
        :param veh_obj: corresponding vehicle obj
        :param vehicle_plan: vehicle_plan where rid_remove is included
        :param rid_to_remove: request_id that should be removed from the rtv_key
        :param sim_time: current simulation time
        :return: best_veh_plan if vehicle_plan rid_to_remove is part of vehicle_plan, None else
        """
        return simple_remove(veh_obj, vehicle_plan, rid_to_remove, sim_time, self.routing_engine, self.objective_function, self.active_requests, self.std_bt, self.add_bt)

    def lock_request_to_vehicle(self, rid : Any, vid : int):
        """locks the request to the assigned vehicle and therefore prevents future re-assignments
        :param rid: request id
        :param vid: vehicle id
        """
        try:
            self.v2r_locked[vid][rid] = 1
        except:
            self.v2r_locked[vid] = {rid : 1}
        self.r2v_locked[rid] = vid

    def set_mutually_exclusive_assignment_constraint(self, list_sub_rids : List[Any], base_rid : Any):
        """ when this function is called, only one of requests from list_rids is allowed being part of the solution of the assignment constraints
        all requests of list_sub_rids have to be added by add_new_request before calling this method
        :param list_sub_rids: list of request_ids
        :param base_rid: id of request, that will actually board the vehicle in the simulation (the sub_rids represent variations of this request)
        """
        if len(list_sub_rids) == 0:
            return
        cluster_id = base_rid
        for rid in list_sub_rids:
            try:
                self.mutually_exclusive_cluster_id_to_rids[cluster_id][rid] = 1
            except KeyError:
                self.mutually_exclusive_cluster_id_to_rids[cluster_id] = {rid : 1}
            self.rid_to_mutually_exclusive_cluster_id[rid] = cluster_id

    def _get_all_rids_representing_this_base_rid(self, base_rid : Any) -> List[Any]:
        """ this function returns an iterable of all sub_rid_ids representing the base_rid
        if the base_rid is represented by itsself, just itself is also returned
        :param base_rid: request id of physical customer
        :return: iterable of sub_rid ids
        """
        return self.mutually_exclusive_cluster_id_to_rids.get(base_rid, {base_rid : 1}).keys()

    def _get_associated_baserid(self, sub_rid : Any) -> Any:
        """ returns the base_rid associated to this sub_rid
        if sub_rid is a base_rid itself, just itself is return
        :param sub_rid: id of corresponding sub_rid
        :return: id of base_rid
        """
        return self.rid_to_mutually_exclusive_cluster_id.get(sub_rid, sub_rid)

    def _get_all_other_subrids_associated_to_this_subrid(self, sub_rid : Any) -> List[Any]:
        """ this function returns an iterable of all sub_rids belongig ot the same base_rid
        as the sub_rid
        :param sub_rid: id of corresponding sub_rid
        :return: iterable of sub_rid ids
        """
        base_rid = self._get_associated_baserid(sub_rid)
        return self._get_all_rids_representing_this_base_rid(base_rid)

    def _is_subrid(self, sub_rid : Any) -> bool:
        """ this function tests if the sub_rid is a sub_rid or a base_rid
        :param sub_rid: id of corresponding sub_rid
        :return: True, if sub_rid is a sub_rid, else False
        """
        if self.rid_to_mutually_exclusive_cluster_id.get(sub_rid) is None:
            return False
        else:
            return True

    def _delete_subrid(self, sub_rid : Any):
        """ this function removes sub_rids from dict entries
        if this rid is a base_rid, nothing happens
        :param sub_rid: id of sub_rid
        """
        mut_cluster_id = self.rid_to_mutually_exclusive_cluster_id.get(sub_rid, None)
        if mut_cluster_id is not None:
            try:
                del self.rid_to_mutually_exclusive_cluster_id[sub_rid]
            except:
                pass
            try:
                del self.mutually_exclusive_cluster_id_to_rids[mut_cluster_id][sub_rid]
            except:
                pass
            if len(self.mutually_exclusive_cluster_id_to_rids.get(mut_cluster_id, {}).keys()) == 0:
                try:
                    del self.mutually_exclusive_cluster_id_to_rids[mut_cluster_id]
                except:
                    pass
                
    def delete_vehicle_database_entries(self, vid):
        """ triggered when all database entries of vehicle vid should be deleted"""
        pass


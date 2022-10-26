from __future__ import annotations
import logging
import time
from typing import Dict, List, Any, TYPE_CHECKING, Tuple

from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.fleetctrl.pooling.objectives import return_pooling_objective_function
from src.misc.init_modules import load_ride_pooling_batch_optimizer
from src.simulation.Offers import Rejection, TravellerOffer
from src.simulation.Legs import VehicleRouteLeg
from src.fleetctrl.pooling.GeneralPoolingFunctions import get_assigned_rids_from_vehplan
from src.misc.globals import *

if TYPE_CHECKING:
    from src.routing.NetworkBase import NetworkBase
    from src.fleetctrl.pooling.batch.BatchAssignmentAlgorithmBase import BatchAssignmentAlgorithmBase
    from src.simulation.Vehicles import SimulationVehicle
    from src.infra.ChargingInfrastructure import OperatorChargingAndDepotInfrastructure, PublicChargingInfrastructureOperator
    from src.infra.Zoning import ZoneSystem
    from src.demand.TravelerModels import RequestBase
    from src.simulation.StationaryProcess import ChargingProcess


LOG = logging.getLogger(__name__)
LARGE_INT = 100000

def load_parallelization_manager(rp_batch_optimizer_str):
    """ this function is used to load the parallelization manager class corresponding to the corresponding batch optimizer
    which will be used in the optimizer itself. managers can be shared across multiple operators in case they use the same
    optimization algorithm
    :param rp_batch_optimizer_str: input string of the optimizer (operator_attributes[G_RA_RP_BATCH_OPT])
    :return: parallelization manager class (can be only used by the corresponding batch opt algorithm!)"""
    if rp_batch_optimizer_str == "AlonsoMora":
        from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraParallelization import ParallelizationManager
        return ParallelizationManager
    elif rp_batch_optimizer_str == "ParallelTempering":
        from dev.fleetctrl.pooling.batch.ParallelTempering.ParallelTemperingParallelization import ParallelizationManager
        return ParallelizationManager
    else:
        return None

INPUT_PARAMETERS_RidePoolingBatchOptimizationFleetControlBase = {
    "doc" : """THIS CLASS IS FOR INHERITANCE ONLY.
        this class can be used for common ride-pooling studies using a batch assignmant algorithm for optimisation
        triggered in the _time_trigger_request_batch() method""",
    "inherit" : "FleetControlBase",
    "input_parameters_mandatory": [G_SLAVE_CPU, G_RA_REOPT_TS],
    "input_parameters_optional": [
        ],
    "mandatory_modules": [G_RA_RP_BATCH_OPT],
    "optional_modules": []
}

class RidePoolingBatchOptimizationFleetControlBase(FleetControlBase):
    def __init__(self, op_id : int, operator_attributes : dict, list_vehicles : List[SimulationVehicle],
                 routing_engine : NetworkBase, zone_system : ZoneSystem, scenario_parameters : dict,
                 dir_names : dict, op_charge_depot_infra : OperatorChargingAndDepotInfrastructure=None,
                 list_pub_charging_infra: List[PublicChargingInfrastructureOperator]= []):
        """The specific attributes for the fleet control module are initialized. Strategy specific attributes are
        introduced in the children classes.

        THIS CLASS IS FOR INHERITANCE ONLY.
        this class can be used for common ride-pooling studies using a batch assignmant algorithm for optimisation
        triggered in the _time_trigger_request_batch() method
        customers are introduced by the user_request() function, for each customer requesting a trip, either the method
        user_confirms_booking() or user_cancels_request has to be called!


        DEPENDING ON THE MODELLED CUSTOMER-FLEETOPERATOR-INTERACTION-FOLLOWING METHODS HAVE TO BE EXTENDED.
        - user_request()
        - user_confirms_booking()
        - user_cancels_request()
        - time_trigger()

        :param op_id: operator id
        :type op_id: int
        :param operator_attributes: dictionary with keys from globals and respective values
        :type operator_attributes: dict
        :param list_vehicles: simulation vehicles; their assigned plans should be instances of the VehicleRouteLeg class
        :type list_vehicles: list
        :param routing_engine: routing engine
        :type routing_engine: Network
        :param scenario_parameters: access to all scenario parameters (if necessary)
        :type scenario_parameters: dict
        :param op_charge_depot_infra: reference to a OperatorChargingAndDepotInfrastructure class (optional) (unique for each operator)
        :type OperatorChargingAndDepotInfrastructure: OperatorChargingAndDepotInfrastructure
        :param list_pub_charging_infra: list of PublicChargingInfrastructureOperator classes (optional) (accesible for all agents)
        :type list_pub_charging_infra: list of PublicChargingInfrastructureOperator
        """
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters,
                         dir_names=dir_names, op_charge_depot_infra=op_charge_depot_infra, list_pub_charging_infra=list_pub_charging_infra)
        self.sim_time = scenario_parameters[G_SIM_START_TIME]
        self.rid_to_assigned_vid : Dict[Any, int] = {}
        self.pos_veh_dict : Dict[tuple, List[SimulationVehicle]] = {}  # pos -> list_veh
        # additional control scenario input parameters
        # define vr-assignment control objective function
        self.vr_ctrl_f = return_pooling_objective_function(operator_attributes[G_OP_VR_CTRL_F])

        self.Parallelization_Manager = None
        n_cores = scenario_parameters[G_SLAVE_CPU]

        RPBO_class = load_ride_pooling_batch_optimizer(operator_attributes.get(G_RA_RP_BATCH_OPT, "AlonsoMora"))
        self.RPBO_Module : BatchAssignmentAlgorithmBase = RPBO_class(self, self.routing_engine, self.sim_time, self.vr_ctrl_f, operator_attributes, optimisation_cores=n_cores, seed=scenario_parameters[G_RANDOM_SEED])

        self.optimisation_time_step = operator_attributes[G_RA_REOPT_TS]
        self.max_rv_con = operator_attributes.get(G_RA_MAX_VR, None)
        self.applied_heuristic = operator_attributes.get(G_RA_HEU, None)
        
        # dynamic dicts to update database
        self.new_requests : Dict[Any, PlanRequest] = {}  # rid -> prq (new)
        self.requests_that_changed : Dict[Any, PlanRequest] = {}  # rid -> prq (already here but new constraints)
        self.new_travel_times_loaded = False    # indicates if new travel times have been loaded on the routing engine

        # init dynamic output -> children fleet controls should check correct usage
        self._init_dynamic_fleetcontrol_output_key(G_FCTRL_CT_RQU)
        self._init_dynamic_fleetcontrol_output_key(G_FCTRL_CT_RQB)

    def add_init(self, operator_attributes, scenario_parameters):
        super().add_init(operator_attributes, scenario_parameters)
        n_cores = scenario_parameters[G_SLAVE_CPU]
        LOG.info("add init: {}".format(n_cores))
        if n_cores > 1 and self.Parallelization_Manager is None:
            LOG.info("initialize Parallelization Manager")
            pm_class = load_parallelization_manager(operator_attributes.get(G_RA_RP_BATCH_OPT, "AlonsoMora"))
            if pm_class is not None:
                self.Parallelization_Manager = pm_class(n_cores, scenario_parameters, self.dir_names)
                LOG.info(" -> success")

        if self.Parallelization_Manager is not None:
            self.RPBO_Module.register_parallelization_manager(self.Parallelization_Manager)

    def register_parallelization_manager(self, Parallelization_Manager):
        """ this method can be used within the add_init of the fleet simulation to define
        a Parallelization Manager that is shared between multiple operators 
        (add_init of the fleetcontrol has to be called after this one)
        :param Parallelization_Manager: object to manage parallelization in AM algorithm
        :type Parallelization_Manager: src.pooling.batch.AlonsoMora.AlonsoMoraParallelization.ParallelizationManager
        """
        self.Parallelization_Manager = Parallelization_Manager

    def receive_status_update(self, vid : int, simulation_time : int, list_finished_VRL : List[VehicleRouteLeg], force_update : bool=True):
        """This method can be used to update plans and trigger processes whenever a simulation vehicle finished some
         VehicleRouteLegs.

        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        :param list_finished_VRL: list of VehicleRouteLeg objects
        :type list_finished_VRL: list
        :param force_update: force vehicle plan update (can be turned off in normal update step)
        :type force_update: bool
        """
        if simulation_time%self.optimisation_time_step == 0:
            force_update=True
        super().receive_status_update(vid, simulation_time, list_finished_VRL, force_update=force_update)
        self.sim_time = simulation_time
        veh_obj = self.sim_vehicles[vid]
        # track done VRLs for updating DB in optimisation-step
        try:
            self.vid_finished_VRLs[vid] += list_finished_VRL
        except KeyError:
            self.vid_finished_VRLs[vid] = list_finished_VRL
        LOG.debug(f"veh {veh_obj} | after status update: {self.veh_plans[vid]}")

    def user_request(self, rq : RequestBase, sim_time : int):
        """This method is triggered for a new incoming request. It generally adds the rq to the database.
        WHEN INHERITING THIS FUNCTION AN ADDITIONAL CONTROL STRUCTURE TO CREATE OFFERS NEED TO BE IMPLEMENTED IF NEEDED
        (e.g. the functionality of creating an offer might be extended here)

        :param rq: request object containing all request information
        :type rq: RequestBase
        :param sim_time: current simulation time
        :type sim_time: float

        """
        LOG.debug(f"Incoming request {rq.__dict__} at time {sim_time}")
        if self.rq_dict.get(rq.get_rid_struct()):
            return
        t0 = time.perf_counter()
        self.sim_time = sim_time

        self.sim_time = sim_time
        prq = PlanRequest(rq, self.routing_engine, min_wait_time=self.min_wait_time,
                          max_wait_time=self.max_wait_time,
                          max_detour_time_factor=self.max_dtf, max_constant_detour_time=self.max_cdt,
                          add_constant_detour_time=self.add_cdt, min_detour_time_window=self.min_dtw,
                          boarding_time=self.const_bt)
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

        # record cpu time
        dt = round(time.perf_counter() - t0, 5)
        old_dt = self._get_current_dynamic_fleetcontrol_value(sim_time, G_FCTRL_CT_RQU)
        if old_dt is None:
            new_dt = dt
        else:
            new_dt = old_dt + dt
        output_dict = {G_FCTRL_CT_RQU: new_dt}
        self._add_to_dynamic_fleetcontrol_output(sim_time, output_dict)

    def user_confirms_booking(self, rid : Any, simulation_time : int):
        """This method is used to confirm a customer booking. This can trigger some database processes.

        WHEN INHERITING THIS FUNCTION ADDITIONAL CONTROL STRUCTURES WHICH DEFINE THE ASSIGNED PLAN MIGHT BE NEEDED
        DEPENDING ON WHERE OFFERS ARE CREATED THEY HAVE TO BE ADDED TO THE OFFERS OF PLANREQUESTS

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        LOG.debug(f"user confirms booking {rid} at {simulation_time}")
        super().user_confirms_booking(rid, simulation_time)
        self.sim_time = simulation_time
        vid = self.rid_to_assigned_vid.get(rid)
        prq = self.rq_dict.get(rid)
        if vid is not None and prq is not None and prq.get_reservation_flag():
            try:
                self.vid_with_reserved_rids[vid].append(rid)
            except KeyError:
                self.vid_with_reserved_rids[vid] = [rid]
        self.RPBO_Module.set_request_assigned(rid)

    def user_cancels_request(self, rid : Any, simulation_time : int):
        """This method is used to confirm a customer cancellation. This can trigger some database processes.

        WHEN INHERITING THIS FUNCTION AN ADDITIONAL CONTROL STRUCTURE DEFINING WHICH VEHICLE ROUTE SHOULD BE PICKED
        INSTEAD NEEDS TO BE IMPLEMENTED!
        if the currently assigned tour for the rid is needed, first retrieve it by selecting
        self.rid_to_assigned_vid.get(rid) after that call super().user_cancels_request
        additionally a new vehicle plan without the rid has to be registered, in case the optimisation (time_trigger)
        has been called since user_request(rid)
        -> use assign_vehicle_plan() to register the new tour for the optimisation problem

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        self.sim_time = simulation_time
        LOG.debug(f"user cancels request {rid} at {simulation_time}")
        prev_vid = self.rid_to_assigned_vid.get(rid)
        prq = self.rq_dict.get(rid)
        if prev_vid is not None and prq is not None and prq.get_reservation_flag():
            list_reserved_rids = self.vid_with_reserved_rids.get(prev_vid, [])
            if rid in list_reserved_rids:
                list_reserved_rids.remove(rid)
                if list_reserved_rids:
                    self.vid_with_reserved_rids[prev_vid] = list_reserved_rids
                else:
                    del self.vid_with_reserved_rids[prev_vid]
        if prq.get_reservation_flag():
            self.reservation_module.user_cancels_request(rid, simulation_time)
        try:
            del self.rq_dict[rid]
        except KeyError:
            pass
        try:
            del self.rid_to_assigned_vid[rid]
        except KeyError:
            pass
        self.RPBO_Module.delete_request(rid)

    def acknowledge_boarding(self, rid : Any, vid : int, simulation_time : int):
        """This method can trigger some database processes whenever a passenger is starting to board a vehicle.

        :param rid: request id
        :type rid: int
        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        self.sim_time = simulation_time
        LOG.debug(f"acknowledge boarding {rid} in {vid} at {simulation_time}")
        self.rq_dict[rid].set_pickup(vid, simulation_time)
        self.RPBO_Module.set_database_in_case_of_boarding(rid, vid)

    def acknowledge_alighting(self, rid : Any, vid : int, simulation_time : int):
        """This method can trigger some database processes whenever a passenger is finishing to alight a vehicle.

        :param rid: request id
        :type rid: int
        :param vid: vehicle id
        :type vid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        self.sim_time = simulation_time
        LOG.debug(f"acknowledge alighting {rid} from {vid} at {simulation_time}")
        self.RPBO_Module.set_database_in_case_of_alighting(rid, vid)
        del self.rq_dict[rid]
        try:
            del self.rid_to_assigned_vid[rid]
        except KeyError:
            pass

    def _prq_from_reservation_to_immediate(self, rid : Any, sim_time : int):
        """This method is triggered when a reservation request becomes an immediate request.
        All database relevant methods can be triggered from here.

        :param rid: request id
        :param sim_time: current simulation time
        :return: None
        """
        if self.rid_to_assigned_vid.get(rid) is not None:
            self.rq_dict[rid].set_reservation_flag(False)
            self.RPBO_Module.add_new_request(rid, self.rq_dict[rid], is_allready_assigned=True)
        else:
            self.RPBO_Module.add_new_request(rid, self.rq_dict[rid])

    def _call_time_trigger_request_batch(self, simulation_time : int):
        """This method can be used to perform time-triggered processes, e.g. the optimization of the current
        assignments of simulation vehicles of the fleet.

        WHEN INHERITING THIS FUNCTION AN ADDITIONAL CONTROL STRUCTURE TO CREATE OFFERS NEED TO BE IMPLEMENTED IF NEEDED
        DEPENDING ON WHERE OFFERS ARE CREATED THEY HAVE TO BE ADDED TO THE DICT self.active_request_offers

        when overwriting this method super().time_trigger(simulation_time) should be called first

        :param simulation_time: current simulation time
        :type simulation_time: int
        """

        t0 = time.perf_counter()
        self.sim_time = simulation_time
        if self.sim_time % self.optimisation_time_step == 0:
            # LOG.info(f"time for new optimisation at {simulation_time}")
            self.RPBO_Module.compute_new_vehicle_assignments(self.sim_time, self.vid_finished_VRLs, build_from_scratch=False,
                                                        new_travel_times=self.new_travel_times_loaded)
            # LOG.info(f"new assignments computed")
            self._set_new_assignments()
            self._clearDataBases()
            self.RPBO_Module.clear_databases()
            dt = round(time.perf_counter() - t0, 5)
            output_dict = {G_FCTRL_CT_RQB: dt}
            self._add_to_dynamic_fleetcontrol_output(simulation_time, output_dict)

    def compute_VehiclePlan_utility(self, simulation_time : int, veh_obj : SimulationVehicle, vehicle_plan : VehiclePlan) -> float:
        """This method computes the utility of a given plan and returns the value.

        :param simulation_time: current simulation time
        :type simulation_time: float
        :param veh_obj: vehicle object
        :type veh_obj: SimulationVehicle
        :param vehicle_plan: vehicle plan in question
        :type vehicle_plan: VehiclePlan
        :return: utility of vehicle plan
        :rtype: float
        """
        return self.vr_ctrl_f(simulation_time, veh_obj, vehicle_plan, self.rq_dict, self.routing_engine)

    def assign_vehicle_plan(self, veh_obj : SimulationVehicle, vehicle_plan : VehiclePlan, sim_time : int, force_assign : bool=False
                            , assigned_charging_task: Tuple[Tuple[str, int], ChargingProcess]=None , add_arg : bool=None):
        """ this method should be used to assign a new vehicle plan to a vehicle

        WHEN OVERWRITING THIS FUNCTION MAKE SURE TO CALL AT LEAST THE LINES BELOW (i.e. super())

        :param veh_obj: vehicle obj to assign vehicle plan to
        :type veh_obj: SimulationVehicle
        :param vehicle_plan: vehicle plan that should be assigned
        :type vehicle_plan: VehiclePlan
        :param sim_time: current simulation time in seconds
        :type sim_time: int
        :param force_assign: this parameter can be used to enforce the assignment, when a plan is (partially) locked
        :type force_assign: bool
        :param add_arg: set to True, if the vehicle plan is assigned internally by AM-assignment
        :type add_arg: not defined here
        """
        LOG.debug(f"assign vehicle plan for {veh_obj} addarg {add_arg} : {vehicle_plan}")
        super().assign_vehicle_plan(veh_obj, vehicle_plan, sim_time, force_assign=force_assign, assigned_charging_task=assigned_charging_task, add_arg=add_arg)
        if add_arg is None:
            veh_plan_without_rel = vehicle_plan.copy_and_remove_empty_planstops(veh_obj, sim_time, self.routing_engine)
            self.RPBO_Module.set_assignment(veh_obj.vid, veh_plan_without_rel, is_external_vehicle_plan=True)
        else:
            self.RPBO_Module.set_assignment(veh_obj.vid, vehicle_plan)

    def _set_new_assignments(self):
        """ this function sets the new assignments computed in the alonso-mora-module
        """
        LOG.debug("global opt sols:")
        for vid, veh_obj in enumerate(self.sim_vehicles):
            assigned_plan = self.RPBO_Module.get_optimisation_solution(vid)
            LOG.debug("vid: {} {}".format(vid, assigned_plan))
            rids = get_assigned_rids_from_vehplan(assigned_plan)
            if len(rids) == 0 and len(get_assigned_rids_from_vehplan(self.veh_plans[vid])) == 0:
                #LOG.debug("ignore assignment")
                self.RPBO_Module.set_assignment(vid, None)
                continue
            if assigned_plan is not None:
                #LOG.debug(f"assigning new plan for vid {vid} : {assigned_plan}")
                self.assign_vehicle_plan(veh_obj, assigned_plan, self.sim_time, add_arg=True)
            else:
                #LOG.debug(f"removing assignment from {vid}")
                assigned_plan = VehiclePlan(veh_obj, self.sim_time, self.routing_engine, [])
                self.assign_vehicle_plan(veh_obj, assigned_plan, self.sim_time, add_arg=True)

    def lock_rid_vid_assignments(self):
        """ this function locks all assignments of new assigned requests to the corresponding vid
        and prevents them from reassignment in the next opt-steps
        """
        for vid, veh_obj in enumerate(self.sim_vehicles):
            assigned_plan = self.RPBO_Module.get_optimisation_solution(vid)
            # LOG.debug("vid: {} {}".format(vid, assigned_plan))
            rids = get_assigned_rids_from_vehplan(assigned_plan)
            for rid in rids:
                if self.new_requests.get(rid):
                    # LOG.info("lock rid {} to vid {}".format(rid, vid))
                    self.RPBO_Module.lock_request_to_vehicle(rid, vid)

    def _clearDataBases(self):
        """ this function clears dynamic data base entries in fleet control 
        should be called after the optimisation step
        """
        self.new_requests = {}
        self.requests_that_changed = {}
        self.vid_finished_VRLs = {}
        self.new_travel_times_loaded = False

    def inform_network_travel_time_update(self, simulation_time : int):
        """ triggered if new travel times are available;
        -> the AM database needs to be recomputed
        -> networks on parallel cores need to be synchronized
        """
        self.sim_time = simulation_time
        self.new_travel_times_loaded = True
        if self.Parallelization_Manager is not None:
            self.Parallelization_Manager.update_network(simulation_time)

    def lock_current_vehicle_plan(self, vid : int):
        super().lock_current_vehicle_plan(vid)
        if hasattr(self, "RPBO_Module"):
            LOG.debug(" -> also lock in RPBO_Module")
            assigned_plan = self.veh_plans.get(vid, None)
            if assigned_plan is not None:
                self.RPBO_Module.set_assignment(vid, assigned_plan, is_external_vehicle_plan=True)
            self.RPBO_Module.delete_vehicle_database_entries(vid)
            for rid in get_assigned_rids_from_vehplan(assigned_plan):
                self.RPBO_Module.lock_request_to_vehicle(rid, vid)

    def _lock_vid_rid_pickup(self, sim_time : int, vid : int, rid : Any):
        """This method constrains the pick-up of a rid. In the pooling case, the pick-up time is constrained to a very
        short time window. In the hailing case, the Task to serve rid is locked for the vehicle.

        :param sim_time: current simulation time
        :param vid: vehicle id
        :param rid: PlanRequest id
        :return: None
        """
        super()._lock_vid_rid_pickup(sim_time, vid, rid)
        self.RPBO_Module.lock_request_to_vehicle(rid, vid)

    def change_prq_time_constraints(self, sim_time : int, rid : Any, new_lpt : float, new_ept : float=None):
        """this function registers if time constraints of a requests is changed during the simulation"""
        LOG.debug("change time constraints for rid {}".format(rid))
        prq = self.rq_dict[rid]
        exceed_tw = True
        if new_lpt <= prq.t_pu_latest:
            if new_ept is None or new_ept >= prq.t_pu_earliest:
                exceed_tw = False
        prq.set_new_pickup_time_constraint(new_lpt, new_earliest_pu_time=new_ept)
        ass_vid = self.rid_to_assigned_vid.get(rid)
        if ass_vid is not None:
            self.veh_plans[ass_vid].update_prq_hard_constraints(self.sim_vehicles[ass_vid], sim_time,
                                                                self.routing_engine, prq, new_lpt, new_ept=new_ept,
                                                                keep_feasible=True)
        self.RPBO_Module.register_change_in_time_constraints(rid, prq, assigned_vid=ass_vid,
                                                           exceeds_former_time_windows=exceed_tw)

    def _create_user_offer(self, prq : PlanRequest, simulation_time : int, assigned_vehicle_plan : VehiclePlan=None,
                           offer_dict_without_plan : dict={}) -> TravellerOffer:
        """ creating the offer for a requests

        :param prq: plan request
        :type prq: PlanRequest obj
        :param simulation_time: current simulation time
        :type simulation_time: int
        :param assigned_vehicle_plan: vehicle plan of initial solution to serve this request
        :type assigned_vehicle_plan: VehiclePlan None
        :param offer_dict_without_plan: can be used to create an offer that is not derived from a vehicle plan
                    entries will be used to create/extend offer
        :type offer_dict_without_plan: dict or None
        :return: offer for request
        :rtype: TravellerOffer
        """
        if assigned_vehicle_plan is not None:
            pu_time, do_time = assigned_vehicle_plan.pax_info.get(prq.get_rid_struct())
            offer = TravellerOffer(prq.get_rid_struct(), self.op_id, pu_time - prq.rq_time, do_time - pu_time,
                                   self._compute_fare(simulation_time, prq, assigned_vehicle_plan))
            prq.set_service_offered(offer)  # has to be called
        else:
            offer = self._create_rejection(prq, simulation_time)
        return offer
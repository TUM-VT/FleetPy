from __future__ import annotations
# -------------------------------------------------------------------------------------------------------------------- #
# standard distribution imports
# -----------------------------
import logging
import time
from abc import abstractmethod, ABCMeta
from typing import Dict, List, Any, Tuple, TYPE_CHECKING

import logging
import time

from src.fleetctrl.FleetControlBase import VehiclePlan, PlanRequest
from src.fleetctrl.RidePoolingBatchOptimizationFleetControlBase import RidePoolingBatchOptimizationFleetControlBase
from src.fleetctrl.pooling.objectives import return_pooling_objective_function
from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraParallelization import ParallelizationManager
from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraAssignment import AlonsoMoraAssignment 
from src.simulation.Offers import TravellerOffer
from src.fleetctrl.pooling.GeneralPoolingFunctions import get_assigned_rids_from_vehplan
from src.misc.globals import *
from src.fleetctrl.pooling.immediate.insertion import insert_prq_in_selected_veh_list

from src.simulation.FreelancerSimulationVehicle import FreelancerSimulationVehicle

if TYPE_CHECKING:
    from src.routing.NetworkBase import NetworkBase
    from src.infra.Zoning import ZoneSystem
    from src.infra.ChargingInfrastructure import OperatorChargingAndDepotInfrastructure, PublicChargingInfrastructureOperator
    from src.simulation.StationaryProcess import ChargingProcess


LOG = logging.getLogger(__name__)
LARGE_INT = 100000


class PlatformFleetControlBase(RidePoolingBatchOptimizationFleetControlBase):
    def __init__(self, op_id: int, operator_attributes: Dict, list_vehicles: List[FreelancerSimulationVehicle], routing_engine: NetworkBase, zone_system: ZoneSystem, scenario_parameters: Dict, dir_names: Dict, op_charge_depot_infra: OperatorChargingAndDepotInfrastructure = None, list_pub_charging_infra: List[PublicChargingInfrastructureOperator] = ...):
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters, dir_names, op_charge_depot_infra, list_pub_charging_infra)
        """ 
        this control provides functionality to check for available drivers + provides correct vehicle plan assignment + checks for driver acceptance
        -> needs FreelancerSimulationVehicle-objects
        """
        if type(self.sim_vehicles[0]) != FreelancerSimulationVehicle:
            raise EnvironmentError("PlatfromFleetControlBase requires FreelancerSimulationVehicle-Objects!")
        
        self._last_check_for_available_vehicles = None
        self._pos_to_available_vehicles = {} # TODO think about .pos_veh_dict in fleetcontrol_base (is used in insertion heuristic) but problem: would not capture if vehicle is assigned to other op during offer phase
        self._available_vehicles = {}
        
    def _update_available_vehicles(self, sim_time):
        LOG.debug(f"check available vehicles for op {self.op_id}")
        if self._last_check_for_available_vehicles is None or self._last_check_for_available_vehicles != sim_time:
            self._last_check_for_available_vehicles = sim_time
            self._pos_to_available_vehicles = {}
            self._available_vehicles = {}
            for veh in self.sim_vehicles:
                LOG.debug(f"check vehicle {veh}")
                if self.op_id in veh.current_op_id_options:
                    LOG.debug(f" -> available for operator")
                    if veh.check_vehicle_acceptance(sim_time):
                        LOG.debug(f" -> also currently available")
                        pos = veh.pos 
                        self._available_vehicles[veh.vid] = veh
                        try:
                            self._pos_to_available_vehicles[pos].append(veh.vid)
                        except KeyError:
                            self._pos_to_available_vehicles[pos] = [veh.vid]

class FreelancerFleetControl(PlatformFleetControlBase):
    def __init__(self, op_id: int, operator_attributes: Dict, list_vehicles: List[FreelancerSimulationVehicle], routing_engine: NetworkBase, zone_system: ZoneSystem, scenario_parameters: Dict, dir_names: Dict, op_charge_depot_infra: OperatorChargingAndDepotInfrastructure = None, list_pub_charging_infra: List[PublicChargingInfrastructureOperator] = ...):
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters, dir_names, op_charge_depot_infra, list_pub_charging_infra)
        """ 
        this control handles the control of vehicles that are currently not assigned to any operator (private drivers)
        main tasks:
        - greedy rebalancing -> rebalancing modul (ggf automatisch geladen)
        - charging ?
        - output of vehicle movements that dont belong to any operator (platform) [is created in the vehicle, not in the control itself]
        """
        
    def user_request(self, rq, sim_time: int):
        """ this fleetcontrol does not offer a mobility service"""
        self._update_available_vehicles(sim_time)
        
    def user_confirms_booking(self, rid: Any, simulation_time: int):
        raise NotImplementedError("This should not happen")
    
    def user_cancels_request(self, rid: Any, simulation_time: int):
        pass
        
    def _call_time_trigger_request_batch(self, simulation_time: int):
        """ nothing to do here"""
        pass
        
class RidePoolingPlatformFleetControl(PlatformFleetControlBase):
    def __init__(self, op_id: int, operator_attributes: Dict, list_vehicles: List[FreelancerSimulationVehicle], routing_engine: NetworkBase, zone_system: ZoneSystem, scenario_parameters: Dict, dir_names: Dict, op_charge_depot_infra: OperatorChargingAndDepotInfrastructure = None, list_pub_charging_infra: List[PublicChargingInfrastructureOperator] = ...):
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters, dir_names, op_charge_depot_infra, list_pub_charging_infra)
        """ 
        provides control for a platform of a ridepooling service with private drivers 
        wie "normale fleetcontrol", nur
            - kein rebalancing ?
            - ggf dynmaic pricing + freelancer reagieren
            - dynamische flottengröße (müssen auf verfügbare driver checken)
            - driver müssen assigned route nicht annehmen
            
        kritische stelle: reaktionszeit von fahrern ? fallback für fahrer rejects
        """
        self.tmp_assignment = {}  # rid -> possible veh_plan
        
    def user_request(self, rq, sim_time):
        """This method is triggered for a new incoming request. It generally adds the rq to the database. It has to
        return an offer to the user. This operator class only works with immediate responses and therefore either
        sends an offer or a rejection.

        :param rq: request object containing all request information
        :type rq: RequestDesign
        :param sim_time: current simulation time
        :type sim_time: float
        :return: offer
        :rtype: TravellerOffer
        """
        # TODO # think about way to call super().user_request() again! -> add_new_request should not be called twice
        # check if request is already in database (do nothing in this case)
        if self.rq_dict.get(rq.get_rid_struct()):
            return
        
        self._update_available_vehicles(sim_time)
        
        t0 = time.perf_counter()
        self.sim_time = sim_time

        LOG.debug(f"Incoming request {rq.__dict__} at time {sim_time}")
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
            raise EnvironmentError("Not defined how reservation requests should be treated with freelancer drivers!")
            self.reservation_module.add_reservation_request(prq, sim_time)
            offer = self.reservation_module.return_immediate_reservation_offer(prq.get_rid_struct(), sim_time)
            LOG.debug(f"reservation offer for rid {rid_struct} : {offer}")
            prq.set_reservation_flag(True)
            self.RPBO_Module.add_new_request(rid_struct, prq, consider_for_global_optimisation=False)
        else:
            self.RPBO_Module.add_new_request(rid_struct, prq)
            
            # backwards Dijkstra
            rv_routing = self.routing_engine.return_travel_costs_Xto1(self._pos_to_available_vehicles.keys(), o_pos, max_cost_value=prq.t_pu_latest - sim_time)
            selected_veh_list = []
            for r in rv_routing:
                pos = r[0]
                for vid in self._pos_to_available_vehicles[pos]:
                    veh = self._available_vehicles[vid]
                    if self.op_id in veh.current_op_id_options:
                        selected_veh_list.append(veh)
            
            list_tuples = insert_prq_in_selected_veh_list(selected_veh_list, self.veh_plans, prq, self.vr_ctrl_f, self.routing_engine, self.rq_dict, sim_time, self.const_bt, self.add_bt)
            offered = False
            if len(list_tuples) > 0:
                list_tuples.sort(key=lambda x:x[2])
                for vid, vehplan, _ in list_tuples:
                    veh = self._available_vehicles[vid]
                    if veh.check_vehicle_acceptance(sim_time, vehicle_plan=vehplan):
                        self.tmp_assignment[rid_struct] = vehplan
                        offer = self._create_user_offer(prq, sim_time, vehplan)
                        LOG.debug(f"new offer for rid {rid_struct} : {offer}")
                        offered=True
                        break
            if not offered:
                LOG.debug(f"rejection for rid {rid_struct}")
                self._create_rejection(prq, sim_time)                            

        # record cpu time
        dt = round(time.perf_counter() - t0, 5)
        old_dt = self._get_current_dynamic_fleetcontrol_value(sim_time, G_FCTRL_CT_RQU)
        if old_dt is None:
            new_dt = dt
        else:
            new_dt = old_dt + dt
        output_dict = {G_FCTRL_CT_RQU: new_dt}
        self._add_to_dynamic_fleetcontrol_output(sim_time, output_dict)
        
    def assign_vehicle_plan(self, veh_obj: FreelancerSimulationVehicle, vehicle_plan: VehiclePlan, sim_time: int, force_assign: bool = False, assigned_charging_task: Tuple[Tuple[str, int], ChargingProcess] = None, add_arg: bool = None):
        if not veh_obj.check_vehicle_acceptance(sim_time, vehicle_plan=vehicle_plan):
            raise EnvironmentError("Now the vehplan is rejected? should not be possible")
        veh_obj.assign_operator(self.op_id)
        return super().assign_vehicle_plan(veh_obj, vehicle_plan, sim_time, force_assign, assigned_charging_task, add_arg)
    
    def _call_time_trigger_request_batch(self, simulation_time : int):
        """This method can be used to perform time-triggered processes, e.g. the optimization of the current
        assignments of simulation vehicles of the fleet.

        WHEN INHERITING THIS FUNCTION AN ADDITIONAL CONTROL STRUCTURE TO CREATE OFFERS NEED TO BE IMPLEMENTED IF NEEDED
        DEPENDING ON WHERE OFFERS ARE CREATED THEY HAVE TO BE ADDED TO THE DICT self.active_request_offers

        when overwriting this method super().time_trigger(simulation_time) should be called first
        
        update here: only vehicles that have assignments from this platform should be available for re-assignment (if at all)

        :param simulation_time: current simulation time
        :type simulation_time: int
        """

        t0 = time.perf_counter()
        self.sim_time = simulation_time
        if self.sim_time % self.optimisation_time_step == 0:
            # LOG.info(f"time for new optimisation at {simulation_time}")
            veh_objs_to_build = {}
            self._update_available_vehicles(simulation_time)
            for vid, veh in self._available_vehicles.items():
                if len(veh.current_op_id_options) == 1 and self.op_id in veh.current_op_id_options:
                    veh_objs_to_build[vid] = veh
            self.RPBO_Module.compute_new_vehicle_assignments(self.sim_time, self.vid_finished_VRLs, build_from_scratch=False,
                                                        new_travel_times=self.new_travel_times_loaded, veh_objs_to_build=veh_objs_to_build)
            # LOG.info(f"new assignments computed")
            self._set_new_assignments()
            self._clearDataBases()
            self.RPBO_Module.clear_databases()
            dt = round(time.perf_counter() - t0, 5)
            output_dict = {G_FCTRL_CT_RQB: dt}
            self._add_to_dynamic_fleetcontrol_output(simulation_time, output_dict)
            
    def _set_new_assignments(self):
        """ this function sets the new assignments computed in the alonso-mora-module
        it has to updated to not querry the alonso-mora-module for all vehicles but only for those that are available
        """
        LOG.debug("global opt sols:")
        for vid, veh_obj in enumerate(self.sim_vehicles):
            if len(veh_obj.current_op_id_options) == 1 and self.op_id in veh_obj.current_op_id_options: # check if vehicle was part of optimization
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
            
    def user_confirms_booking(self, rid, simulation_time):
        """This method is used to confirm a customer booking. This can trigger some database processes.

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        self.sim_time = simulation_time
        LOG.debug(f"user confirms booking {rid} at {simulation_time}")
        super().user_confirms_booking(rid, simulation_time)
        prq = self.rq_dict[rid]
        if prq.get_reservation_flag():
            self.reservation_module.user_confirms_booking(rid, simulation_time)
        else:
            assigned_plan = self.tmp_assignment[rid]
            vid = assigned_plan.vid
            veh_obj = self.sim_vehicles[vid]
            self.assign_vehicle_plan(veh_obj, assigned_plan, simulation_time)
            del self.tmp_assignment[rid]

    def user_cancels_request(self, rid, simulation_time):
        """This method is used to confirm a customer cancellation. This can trigger some database processes.

        :param rid: request id
        :type rid: int
        :param simulation_time: current simulation time
        :type simulation_time: float
        """
        self.sim_time = simulation_time
        LOG.debug(f"user cancels request {rid} at {simulation_time}")
        prq = self.rq_dict[rid]
        if prq.get_reservation_flag():
            self.reservation_module.user_cancels_request(rid, simulation_time)
        else:
            prev_assignment = self.tmp_assignment.get(rid)
            if prev_assignment:
                del self.tmp_assignment[rid]
        super().user_cancels_request(rid, simulation_time)
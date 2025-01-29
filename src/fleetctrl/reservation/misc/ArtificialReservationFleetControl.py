from src.fleetctrl.RidePoolingBatchAssignmentFleetcontrol import RidePoolingBatchAssignmentFleetcontrol
from src.fleetctrl.planning.VehiclePlan import VehiclePlan, BoardingPlanStop, PlanStopBase, RoutingTargetPlanStop
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.routing.NetworkBase import NetworkBase
from src.fleetctrl.pooling.immediate.insertion import simple_insert, simple_remove
from src.misc.globals import *
from typing import Any, Callable, List, Dict, Tuple, Type
import time

import logging

from src.simulation.Legs import VehicleRouteLeg
from src.simulation.StationaryProcess import ChargingProcess
from src.simulation.Vehicles import SimulationVehicle
LOG = logging.getLogger(__name__)

from dev.fleetctrl.reservation.misc.RequestGroup import QuasiVehicle

class ArtificialReservationFleetControl(RidePoolingBatchAssignmentFleetcontrol):
    def __init__(self, op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters, dir_names, op_charge_depot_infra=None, list_pub_charging_infra=...):
        super().__init__(op_id, operator_attributes, list_vehicles, routing_engine, zone_system, scenario_parameters, dir_names, op_charge_depot_infra, list_pub_charging_infra)
        self._vid_to_idle_since = {vid : scenario_parameters[G_SIM_START_TIME] for vid in range(len(self.sim_vehicles))}    # vid -> time it got idle the last time
        self._vid_to_idle_pos = {} # can be differend from veh.pos in case veh is driving to next reservation stop
        self.repo = None    # repo by beaming

    def receive_status_update(self, vid: int, simulation_time: int, list_finished_VRL: List[VehicleRouteLeg], force_update: bool = True):
        x = super().receive_status_update(vid, simulation_time, list_finished_VRL, force_update)
        if len(list_finished_VRL) > 0 and self._vid_is_idle(vid):
            self._vid_to_idle_since[vid] = simulation_time
            self._vid_to_idle_pos[vid] = self.sim_vehicles[vid].pos
        return x
    
    def assign_vehicle_plan(self, veh_obj: SimulationVehicle, vehicle_plan: VehiclePlan, sim_time: int, force_assign: bool = False, assigned_charging_task: Tuple[Tuple[str, int], ChargingProcess] = None, add_arg: bool = None):
        if not self._vid_is_idle(veh_obj.vid, vehicle_plan=vehicle_plan) and self._vid_to_idle_since.get(veh_obj.vid) is not None:
            try:
                del self._vid_to_idle_since[veh_obj.vid]
            except KeyError:
                pass
            try:
                del self._vid_to_idle_pos[veh_obj.vid]
            except KeyError:
                pass
        elif self._vid_is_idle(veh_obj.vid, vehicle_plan=vehicle_plan) and self._vid_to_idle_since.get(veh_obj.vid) is None:
            self._vid_to_idle_since[veh_obj.vid] = sim_time
            self._vid_to_idle_pos[veh_obj.vid] = veh_obj.pos
        x = super().assign_vehicle_plan(veh_obj, vehicle_plan, sim_time, force_assign, assigned_charging_task, add_arg)  
        return x      
    
    def _vid_is_idle(self, vid, vehicle_plan=None):
        if vehicle_plan is not None:
            veh_p = vehicle_plan
        else:
            veh_p = self.veh_plans[vid]
        if len(veh_p.list_plan_stops) == 0:
            return True
        elif len(veh_p.list_plan_stops) == 1 and veh_p.list_plan_stops[0].is_empty() and  veh_p.list_plan_stops[0].is_locked_end():
            if veh_p.list_plan_stops[0].is_locked():
                return False
            else:
                return True
        else:
            return False
    
    def _call_time_trigger_request_batch(self, simulation_time):
        """ this function first triggers the upper level batch optimisation
        based on the optimisation solution offers to newly assigned requests are created in the second step with following logic:
        if a request remained unassigned, an idle vehicle is search that could have served it if started driving in time.
            if this is the case, it is "teleported" to the request and serves it

        :param simulation_time: current time in simulation
        :return: dictionary rid -> offer for each unassigned request, that will recieve an answer. (offer: dictionary with plan specific entries; empty if no offer can be made)
        :rtype: dict
        """
        t0 = time.perf_counter()
        self.sim_time = simulation_time
        if self.sim_time % self.optimisation_time_step == 0:
            # LOG.info(f"time for new optimisation at {simulation_time}")
            self.RPBO_Module.compute_new_vehicle_assignments(self.sim_time, self.vid_finished_VRLs, build_from_scratch=False,
                                                        new_travel_times=self.new_travel_times_loaded)
            # LOG.info(f"new assignments computed")
            self._set_new_assignments()
            dt = round(time.perf_counter() - t0, 5)
            output_dict = {G_FCTRL_CT_RQB: dt}
            self._add_to_dynamic_fleetcontrol_output(simulation_time, output_dict)
            self.RPBO_Module.clear_databases()

            # rids to be assigned in first try
            for rid in self.unassigned_requests_1.keys():
                assigned_vid = self.rid_to_assigned_vid.get(rid, None)
                prq = self.rq_dict[rid]
                if assigned_vid is None:
                    if self.new_travel_times_loaded:
                        LOG.debug(f" -> update max travel time for rid {rid}")
                        prq.compute_new_max_trip_time(self.routing_engine, 
                          max_detour_time_factor=self.max_dtf, max_constant_detour_time=self.max_cdt,
                          add_constant_detour_time=self.add_cdt, min_detour_time_window=self.min_dtw,
                          boarding_time=self.const_bt)
                    o_pos, earliest_pu, latest_pu = prq.get_o_stop_info()
                    best_other_vid = None
                    tt_best = float("inf")
                    for vid, idle_time in self._vid_to_idle_since.items():
                        vid_pos = self._vid_to_idle_pos.get(vid, self.sim_vehicles[vid].pos)
                        tt = self.routing_engine.return_travel_costs_1to1(vid_pos, o_pos)[0]
                        if idle_time + tt < latest_pu:
                            # if len(self.veh_plans[vid].list_plan_stops) != 0:
                            #     latest_arrival = self.veh_plans[vid].list_plan_stops[0].get_earliest_start_time()
                            #     if idle_time + tt + 2* self.const_bt + prq.init_direct_tt + \
                            #             self.routing_engine.return_travel_costs_1to1(prq.get_d_stop_info()[0], self.veh_plans[vid].list_plan_stops[0].get_pos())[0] > latest_arrival:
                            #         continue
                            if tt < tt_best:
                                if len(self.veh_plans[vid].list_plan_stops) != 0:
                                    d_pos, latest_do, max_tt = prq.get_d_stop_info()
                                    list_ps = []
                                    if idle_time + tt_best > earliest_pu:
                                        lock_stop = RoutingTargetPlanStop(o_pos, duration=idle_time+tt-simulation_time,locked=True)
                                        list_ps.append(lock_stop)
                                    o_stop = BoardingPlanStop(o_pos, boarding_dict={1:[rid]}, earliest_pickup_time_dict={rid : earliest_pu}, latest_pickup_time_dict={rid : latest_pu}, duration=self.const_bt, change_nr_pax=prq.nr_pax)
                                    d_stop = BoardingPlanStop(d_pos, boarding_dict={-1 : [rid]}, max_trip_time_dict={rid : max_tt}, latest_arrival_time_dict={rid : latest_do}, change_nr_pax=-prq.nr_pax, duration=self.const_bt)
                                    list_ps.append(o_stop)
                                    list_ps.append(d_stop)
                                    veh = QuasiVehicle(o_pos, capacity=self.sim_vehicles[vid].max_pax)
                                    qvp = VehiclePlan(veh, simulation_time, self.routing_engine, list_ps + [self.veh_plans[vid].list_plan_stops[-1].copy()])
                                    feasible = qvp.update_tt_and_check_plan(veh, simulation_time, self.routing_engine)
                                    if feasible:
                                        tt_best = tt
                                        best_other_vid = vid
                                else:
                                    tt_best = tt
                                    best_other_vid = vid
                    if best_other_vid is not None:
                        LOG.info(f"alternative vehicle to beam found! rid {rid} - vid {best_other_vid}")
                        LOG.info(f"vehicle idle since {self._vid_to_idle_since[best_other_vid]} with tt {tt_best}")
                        idle_time = self._vid_to_idle_since[best_other_vid]
                        d_pos, latest_do, max_tt = prq.get_d_stop_info()
                        list_ps = []
                        if idle_time + tt_best > earliest_pu:
                            lock_stop = RoutingTargetPlanStop(o_pos, duration=idle_time+tt_best-simulation_time,locked=True)
                            list_ps.append(lock_stop)
                        o_stop = BoardingPlanStop(o_pos, boarding_dict={1:[rid]}, earliest_pickup_time_dict={rid : earliest_pu}, latest_pickup_time_dict={rid : latest_pu}, duration=self.const_bt, change_nr_pax=prq.nr_pax)
                        d_stop = BoardingPlanStop(d_pos, boarding_dict={-1 : [rid]}, max_trip_time_dict={rid : max_tt}, latest_arrival_time_dict={rid : latest_do}, change_nr_pax=-prq.nr_pax, duration=self.const_bt)
                        list_ps.append(o_stop)
                        list_ps.append(d_stop)
                        LOG.info(f"teleport vehicle from {self.sim_vehicles[best_other_vid].pos} to {o_pos}")
                        LOG.info(f"currently assigned plan: {self.veh_plans[best_other_vid]}")
                        self.sim_vehicles[best_other_vid].pos = o_pos
                        if len(self.veh_plans[best_other_vid].list_plan_stops) == 0:
                            veh_plan = VehiclePlan(self.sim_vehicles[best_other_vid], simulation_time, self.routing_engine, list_ps)
                        else:
                            veh_plan = VehiclePlan(self.sim_vehicles[best_other_vid], simulation_time, self.routing_engine, list_ps + self.veh_plans[best_other_vid].list_plan_stops[:])
                        feasible = veh_plan.update_tt_and_check_plan(self.sim_vehicles[best_other_vid], simulation_time, self.routing_engine)
                        LOG.info(f"new plan assigned plan: {veh_plan}")
                        if not feasible:
                            LOG.info(f"infeasible plan: {veh_plan}")
                            raise EnvironmentError("not feasible!")
                        self.assign_vehicle_plan(self.sim_vehicles[best_other_vid], veh_plan, simulation_time)
                        self._create_user_offer(prq, simulation_time, assigned_vehicle_plan=veh_plan)
                    else:   # no retry, rid declined
                        self._create_user_offer(prq, simulation_time)
                else:
                    assigned_plan = self.veh_plans[assigned_vid]
                    self._create_user_offer(prq, simulation_time, assigned_vehicle_plan=assigned_plan)
            self.unassigned_requests_1 = {}
            
            self._clearDataBases()

    
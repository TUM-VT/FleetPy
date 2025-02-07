from __future__ import annotations

import numpy as np
import os
import random
import traceback
import logging
import time

from src.fleetctrl.FleetControlBase import FleetControlBase

from src.demand.TravelerModels import BasicRequest
from src.fleetctrl.planning.PlanRequest import ArtificialPlanRequest
from src.fleetctrl.planning.VehiclePlan import VehiclePlan
from src.fleetctrl.pooling.immediate.insertion import insert_prq_in_selected_veh_list, simple_remove
from src.fleetctrl.pooling.batch.BatchAssignmentAlgorithmBase import SimulationVehicleStruct
import pandas as pd

from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.fleetctrl.forecast.ODForecastZoneSystem import ODForecastZoneSystem
from src.fleetctrl.planning.VehiclePlan import RoutingTargetPlanStop
from src.misc.globals import *
LOG = logging.getLogger(__name__)

from typing import TYPE_CHECKING, List, Dict, Tuple, Callable, Any
if TYPE_CHECKING:
    from src.fleetctrl.FleetControlBase import FleetControlBase
    from src.routing.NetworkBase import NetworkBase

OPT_TIME_LIMIT = 120
WRITE_SOL = True
WRITE_PROBLEM = False
SMALL_VALUE = 10
LARGE_VALUE = 10**6
GUROBI_MIPGAP = 10**-8

INPUT_PARAMETERS_FullSamplingRidePoolingRebalancingMultiStageReservation = {
    "doc" :     """ this class implements the sampling based repositioning method for ride-pooling
        described in roman's thesis (https://mediatum.ub.tum.de/?id=1755168).
        The method is based on sampling future requests and simulating the future vehicle states.
        based on future fleet states, repositioning trips are assigned to close supply gaps.
        In contrast to 'FullSamplingRidePoolingRebalancingMultiStage', this method should be used if reservations which are treated with waypoints as 
        these waypoints are considered in the repositioning optimization.
    """,
    "inherit" : "RepositioningBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [G_OP_REPO_SAMPLE_GAMMA, G_OP_REPO_NR_SAMPLES],
    "mandatory_modules": [],
    "optional_modules": []
}

def move_vehicle_according_to_plan(veh: SimulationVehicleStruct, veh_plan: VehiclePlan, t: int, time_step: int, routing_engine: NetworkBase,
                                   rq_dict: Dict[int, ArtificialPlanRequest], vid_to_current_route, begin_approach_buffer_time):
    """ this method moves vehicles according to their currently assigned plan
    :param veh: vehicle object
    :param veh_plan: vehicle plan object
    :param t: current simulation time
    :param time_step: time step for moving
    :param routing_engine: routing engine
    :param rq_dict: dictionary of requests
    :param vid_to_current_route: dictionary of vehicle id -> current route
    :return: list of passed plan stops, vid_to_current_route (updated)"""
    passed_ps = []
    if len(veh_plan.list_plan_stops) > 0:
        #LOG.debug(f"move {veh.vid} at {t} - {t + time_step} pos {veh.pos}, {veh_plan}")
        cur_pos = veh.pos
        next_t = t + time_step
        last_t = t
        ps_ind = 0
        for i, ps in enumerate(veh_plan.list_plan_stops):
            arr, dep = ps.get_planned_arrival_and_departure_time()
            #LOG.debug(f"arr dep {arr} {dep}")
            if next_t < arr:    # move along route
                target_pos = ps.get_pos()
                cur_pos = veh.pos
                if cur_pos != target_pos:
                    if ps.get_earliest_start_time() - next_t  > begin_approach_buffer_time and \
                        ps.get_earliest_start_time() - next_t - routing_engine.return_travel_costs_1to1(cur_pos, target_pos)[1] > begin_approach_buffer_time:
                        #LOG.debug(f"wait {veh.vid} at {cur_pos} for {target_pos} {ps.get_earliest_start_time()} - {next_t} > {begin_approach_buffer_time}")
                        pass
                    else:
                        route = vid_to_current_route.get(veh.vid, [])
                        #LOG.debug(f"old route {cur_pos} {route}")
                        if len(route) == 0 or route[-1] != target_pos[0]:
                            route = routing_engine.return_best_route_1to1(cur_pos, target_pos)
                            try:
                                route.remove(cur_pos[0])
                            except:
                                pass
                        #LOG.debug(f"new route {cur_pos} {route}")
                        new_pos, _, _, passed_nodes, _ = routing_engine.move_along_route(route, cur_pos, next_t - last_t, veh.vid, t)
                        veh.pos = new_pos
                        for node in passed_nodes:
                            route.remove(node)
                        vid_to_current_route[veh.vid] = route
                    ps_ind = i
                break
            else:   # boarding processes
                veh.pos = ps.get_pos()
                vid_to_current_route[veh.vid] = []
                to_remove = []
                to_add = []
                for rid in ps.get_list_alighting_rids():
                    #LOG.debug(f"check db {rid}")
                    for rq in veh.pax:
                        if rq.get_rid_struct() == rid:
                            #LOG.debug("ob")
                            to_remove.append(rq)
                            break
                for rid in ps.get_list_boarding_rids():
                    is_ob = False
                    #LOG.debug(f"check {rid}")
                    for rq in veh.pax:
                        if rq.get_rid_struct() == rid:
                            is_ob = True
                            break
                    if not is_ob:   # add new passenger (TODO dont like BasicRequest here)
                        #LOG.debug("not ob")
                        prq = rq_dict[rid]
                        traveller = BasicRequest(pd.Series({"start": prq.o_pos[0], "end": prq.d_pos[0], "rq_time": prq.rq_time, "request_id": rid, "latest_decision_time" : 1}),
                                                    routing_engine, time_step, {"latest_decision_time" : 1})
                        traveller.pu_time = arr
                        to_add.append(traveller)
                for pax in to_add:
                    veh.pax.append(pax)
                if next_t < dep:
                    ps.set_locked(True)
                    ps.set_started_at(arr)
                    #LOG.debug(f"new {vehplan}")
                    #LOG.debug(f"{veh}")
                    ps_ind = i
                    break
                else:
                    ps_ind += 1
                    last_t = dep
                    for pax in to_remove:
                        veh.pax.remove(pax)
        # track passed plan stops
        if ps_ind > 0:
            #LOG.debug(f"old: {vehplan}")
            for j in range(ps_ind):
                this_ps = veh_plan.list_plan_stops[j]
                this_ps.set_locked(False)
                passed_ps.append(this_ps)
            veh_plan.list_plan_stops = veh_plan.list_plan_stops[ps_ind:]
            #LOG.debug(f"new {vehplan}")
            #LOG.debug(f"{veh}")
            veh_plan.update_tt_and_check_plan(veh, t, routing_engine, keep_feasible=True)
    return passed_ps, vid_to_current_route
    

class FullSamplingRidePoolingRebalancingMultiStageReservation(RepositioningBase):
    def __init__(self, fleetctrl: FleetControlBase, operator_attributes: dict, dir_names: dict, solver: str = "Gurobi"):
        """ 
        this class implements the sampling based repositioning method for ride-pooling
        described in roman's thesis (https://mediatum.ub.tum.de/?id=1755168).
        The method is based on sampling future requests and simulating the future vehicle states.
        based on future fleet states, repositioning trips are assigned to close supply gaps.
        In contrast to 'FullSamplingRidePoolingRebalancingMultiStage', this method should be used if reservations which are treated with waypoints as 
        these waypoints are considered in the repositioning optimization.
        :param fleetctrl: FleetControl class
        :param operator_attributes: operator dictionary that can contain additionally required parameters
        :param dir_names: directory structure dict
        :param solver: solver for optimization problems
        """
        super().__init__(fleetctrl, operator_attributes, dir_names, solver)
        self.N_samples = int(operator_attributes.get(G_OP_REPO_NR_SAMPLES,1)) # number of samples used for repositioning
        #self._exploration_cost_weight = operator_attributes.get("op_rp_rebal_exploration_cost", None)
        #self._exploration_request_per_sample = 1
        self._sampling_ctrl_function = self.fleetctrl.vr_ctrl_f# update_ctrl_function(self.fleetctrl.vr_ctrl_f)
        self._gamma = operator_attributes.get(G_OP_REPO_SAMPLE_GAMMA, 0.5) # weight of future rewards
        self._progress_time_step = operator_attributes.get(G_RA_REOPT_TS, 60)
        self._repo_time_step = fleetctrl.repo_time_step
        if operator_attributes.get(G_RA_OP_REPO_ZONE_SYSTEM):
            from src.fleetctrl.forecast.PerfectForecastZoning import PerfectForecastDistributionZoneSystem
            self._repo_zone_system = PerfectForecastDistributionZoneSystem(dir_names[G_RA_OP_REPO_ZONE_SYSTEM], {}, dir_names, operator_attributes)
        else:
            self._repo_zone_system = self.zone_system
        self.min_reservation_buffer = max(operator_attributes.get(G_OP_REPO_RES_PUF, 3600), self.list_horizons[1])
        self._prioritize_reservations = operator_attributes.get(G_OP_REPO_RES_PRIORITIZE, True)
        
        self._repo_fallback = FullSamplingRidePoolingRebalancingMultiStage_Fallback(fleetctrl, operator_attributes, dir_names, solver) # TODO this is not awesome

        
    def determine_and_create_repositioning_plans(self, sim_time: int, lock: bool=None) -> List[int]:
        """ computes and assigns new repositioning plans
        :param sim_time: current simulation time
        :param lock: bool if repositioning should be locked
        :return: list of vehicle ids with adopted schedules (new repositioning targets)"""
        try:
            self.sim_time = sim_time
            self.zone_system.time_trigger(sim_time)
            self._repo_zone_system.time_trigger(sim_time)
            if self._sampling_ctrl_function is None:
                self._sampling_ctrl_function = self.fleetctrl.vr_ctrl_f# update_ctrl_function(self.fleetctrl.vr_ctrl_f)
            if lock is None:
                lock = self.lock_repo_assignments
                
            t0_min = sim_time + self.list_horizons[0]
            t1_max = sim_time + self.list_horizons[1]
            
            _sampling_times = []
            
            sample_tour_id_subtour_id_parameters = []
            sample_start_bin_to_ozone_to_tour_ids = []
            sample_end_bin_to_dzone_to_tour_ids = []
            sample_timebin_to_zone_to_idle_vehs = []
            sample_tour_id_to_plan_id = []
            sample_bin_to_zone_to_idle_veh_objs = []
            
            for _ in range(self.N_samples): # do for each sample
                tm = time.time()
                future_rq_atts = []
                for t0 in range(t0_min, t1_max, self._repo_time_step):
                    # sampled future requests
                    new_future_rq_atts = self.zone_system.draw_future_request_sample(t0, t0 + self._repo_time_step)
                    time_bin_int = int( (t0 - sim_time + self.list_horizons[0])/self._repo_time_step )
                    #priority = self._gamma ** time_bin_int  # priority of future requests
                    new_future_rq_atts = [(*x, False) for x in new_future_rq_atts]
                    future_rq_atts += new_future_rq_atts
                    if self.fleetctrl.reservation_module is not None:
                        upcoming_reservation_requests = \
                            self.fleetctrl.reservation_module.get_upcoming_unassigned_reservation_requests(t0, t0 + self._repo_time_step)
                        if self._prioritize_reservations:
                            priority = True
                            future_rq_atts += [(*x, priority) for x in upcoming_reservation_requests]
                        else:
                            future_rq_atts += [(*x, False) for x in upcoming_reservation_requests]

                # simulate future vehicle states and create input parameters for matching
                tour_id_subtour_id_parameters, start_bin_to_ozone_to_tour_ids, end_bin_to_dzone_to_tour_ids, timebin_to_zone_to_idle_vehs, tour_id_to_plan_id, bin_to_zone_to_idle_veh_objs = \
                    self._progress_vehicles(future_rq_atts, sim_time, t1_max)
                    
                LOG.debug(f"tour_id_subtour_id_parameters {tour_id_subtour_id_parameters}")
                LOG.debug(f"start_bin_to_ozone_to_tour_ids {start_bin_to_ozone_to_tour_ids}")
                LOG.debug(f"end_bin_to_dzone_to_tour_ids {end_bin_to_dzone_to_tour_ids}")
                LOG.debug(f"timebin_to_zone_to_idle_vehs {timebin_to_zone_to_idle_vehs}")
                LOG.debug(f"tour_id_to_plan_id {tour_id_to_plan_id}")
                sample_tour_id_subtour_id_parameters.append(tour_id_subtour_id_parameters)
                sample_start_bin_to_ozone_to_tour_ids.append(start_bin_to_ozone_to_tour_ids)
                sample_end_bin_to_dzone_to_tour_ids.append(end_bin_to_dzone_to_tour_ids)
                sample_timebin_to_zone_to_idle_vehs.append(timebin_to_zone_to_idle_vehs)
                sample_tour_id_to_plan_id.append(tour_id_to_plan_id)
                sample_bin_to_zone_to_idle_veh_objs.append(bin_to_zone_to_idle_veh_objs)
                LOG.debug(f"sampling took {time.time() - tm}")
                _sampling_times.append(time.time() - tm)
            
            tm = time.time()
            # get forecast of vehicle availabilities
            avail_for_rebal_vehicles = sample_bin_to_zone_to_idle_veh_objs[0][sim_time]      
            #_, _, avail_for_rebal_vehicles, _ = self._get_vehs_with_vp_available_for_repo(sim_time)
            
            # solve matching problem to reposition idle vehicles to zones
            od_number_rebal, plan_id_to_repo, plan_id_to_idle_bin = self._solve_sampling_repositioning(sim_time, avail_for_rebal_vehicles, sample_tour_id_subtour_id_parameters, 
                                                                    sample_start_bin_to_ozone_to_tour_ids, sample_end_bin_to_dzone_to_tour_ids,
                                                                    sample_timebin_to_zone_to_idle_vehs, sample_tour_id_to_plan_id)
            
            list_veh_with_changes = self._assign_repositioning_plans(sim_time, od_number_rebal,
                                                                    plan_id_to_repo, plan_id_to_idle_bin, sample_bin_to_zone_to_idle_veh_objs[0], lock=self.lock_repo_assignments)
            _solve_time = time.time() - tm
            LOG.info(f"Sampling took {sum(_sampling_times)} | Matching took {_solve_time} | Total {sum(_sampling_times) + _solve_time} | Single sample times: {_sampling_times}")       
            return list_veh_with_changes
        except Exception as e:
            LOG.warning(f"Error in repositioning {sim_time}: {e}")
            LOG.warning(f"Traceback: {traceback.format_exc()}")
            traceback.print_exc()
            LOG.warning(f"Use Fallback")
            return self._repo_fallback.determine_and_create_repositioning_plans(sim_time, lock)
    
    def _get_vehs_with_vp_available_for_repo(self, sim_time : int, use_vehicles = None, use_vehplans = None) \
            -> Tuple[Dict[int, SimulationVehicleStruct], Dict[int, VehiclePlan], Dict[int, List[SimulationVehicleStruct]], Dict[int, VehiclePlan]]:
        """ 
        prepares input parameters for vehicles for sampling based repositioning in combination with reservations
        identifies idle vehicles and vehicles with currently assigned vehicle plans until the end of the forecast horizon
        :param sim_time: current simulation time
        :param use_vehicles: list of vehicle objects to be used (if None, fleetctrl vehicles are used)
        :param use_vehplans: dict of vehicle plan objects to be used (if None, fleetctrl vehicle plans are used)
        :return: 
            veh_objs: dict of vehicle id -> vehicle object with current state (copies for preprogression)
            veh_plans: dict of vehicle id -> vehicle plan object (copies for preprogression)
            zone_to_idle_vehs: dict of zone -> list of vehicle objects that are idle (within the forecast horizon)
            unassigned_reservation_plans: dict of plan id (in reservation module) -> full vehicle plan object for vehicles with reservation plans that are not assigned yet
        """
        n_idle = 0
        veh_objs: Dict[int, SimulationVehicleStruct] = {}
        veh_plans: Dict[int, VehiclePlan] = {}
        zone_to_idle_vehs = {}
        unassigned_reservation_plans = {}
        fetch_reservation_plans = False
        if use_vehicles is None:
            use_vehicles = self.fleetctrl.sim_vehicles
            fetch_reservation_plans = True # otherwise, the reservation plans are already fetched
        if use_vehplans is None:
            use_vehplans = self.fleetctrl.veh_plans
            fetch_reservation_plans = True
        for veh_obj in use_vehicles:
            vp = use_vehplans.get(veh_obj.vid)
            if not fetch_reservation_plans and self.fleetctrl.reservation_module is not None and self.fleetctrl.reservation_module.get_supporting_point(sim_time, vid=veh_obj.vid)[0] is not None:
                LOG.debug(f" reservation stops of veh {veh_obj.vid} at time {sim_time} have been treated before | not idle")
                veh_objs[veh_obj.vid] = veh_obj
                veh_plans[veh_obj.vid] = vp
            else:
                if vp is not None and len(vp.list_plan_stops) > 0: # and not vp.list_plan_stops[-1].is_locked_end() and not vp.list_plan_stops[-1].is_locked():
                    if not (len(vp.list_plan_stops) == 1 and vp.list_plan_stops[-1].is_locked_end()):
                        veh_objs[veh_obj.vid] = SimulationVehicleStruct(veh_obj, vp, sim_time, self.routing_engine)
                        veh_plans[veh_obj.vid] = vp.copy()
                    else:
                        LOG.debug(f"veh {veh_obj.vid} has reservation plan stops at time {sim_time}")
                        est = vp.list_plan_stops[-1].get_earliest_start_time()
                        plan_id, full_plan = self.fleetctrl.reservation_module.get_full_vehicle_plan_until( veh_obj.vid, sim_time, until_time=sim_time + self.list_horizons[1] + self.min_reservation_buffer)
                        LOG.debug(f" -> with first stop earliest start time {full_plan.list_plan_stops[0].get_earliest_start_time()} (plan_id {plan_id})")
                        if est > sim_time + self.min_reservation_buffer: # 
                            n_idle += 1
                            zone = self._repo_zone_system.get_zone_from_pos(veh_obj.pos)
                            try:
                                zone_to_idle_vehs[zone].append(veh_obj)
                            except KeyError:
                                zone_to_idle_vehs[zone] = [veh_obj]
                            # dont keep assignment but force the assignment in sampling process
                            if fetch_reservation_plans:
                                LOG.debug(f" -> for reassignment {est} -> plan_id {plan_id} (fetch)")
                                unassigned_reservation_plans[plan_id] = full_plan.copy()
                            else:
                                LOG.debug(f" -> for reassignment {est} -> plan_id {plan_id} (no fetch)")
                                unassigned_reservation_plans[plan_id] = vp.copy()
                        else:
                            if fetch_reservation_plans:
                                veh_objs[veh_obj.vid] = SimulationVehicleStruct(veh_obj, full_plan, sim_time, self.routing_engine)
                                veh_plans[veh_obj.vid] = full_plan.copy()
                                LOG.debug(f"veh {veh_objs[veh_obj.vid]} | {veh_obj} has reservation plan stops at time {sim_time} -> {full_plan}")
                                veh_plans[veh_obj.vid].update_tt_and_check_plan(veh_objs[veh_obj.vid], sim_time, self.routing_engine, keep_feasible=True)
                                LOG.debug(f" -> but close (plan_id {plan_id}) (fetch) -> not idle {full_plan}")
                            else:
                                veh_objs[veh_obj.vid] = veh_obj
                                veh_plans[veh_obj.vid] = vp
                                LOG.debug(f" -> but close (plan_id {plan_id}) (no fetch) -> not idle {vp}")
                else:
                    n_idle +=1
                    zone = self._repo_zone_system.get_zone_from_pos(veh_obj.pos)
                    try:
                        zone_to_idle_vehs[zone].append(veh_obj)
                    except KeyError:
                        zone_to_idle_vehs[zone] = [veh_obj]
        LOG.debug(f" -> new idle: {n_idle} | new unassigned reservation plans {len(unassigned_reservation_plans)}")
        return veh_objs, veh_plans, zone_to_idle_vehs, unassigned_reservation_plans
    
    def _assign_repositioning_plans(self, sim_time, od_number_rebal, plan_id_to_repo, plan_id_to_idle_bin, bin_to_zone_to_idle_veh_objs, lock=False) -> List[int]:
        LOG.debug(f"assign new repo plans at time {sim_time}:")
        LOG.debug(f"od_number_rebal {od_number_rebal}")
        LOG.debug(f"plan_id_to_repo {plan_id_to_repo}")
        LOG.debug(f"plan_id_to_idle_bin {plan_id_to_idle_bin}")
        LOG.debug(f"bin_to_zone_to_idle_veh_objs {bin_to_zone_to_idle_veh_objs}")
        avail_for_rebal_vehicles = bin_to_zone_to_idle_veh_objs[sim_time]
        repo_to_plan_ids = {}
        for plan_id, repo in plan_id_to_repo.items():
            try:
                repo_to_plan_ids[repo].append(plan_id)
            except KeyError:
                repo_to_plan_ids[repo] = [plan_id]
        od_repo_dict = {}
        for (o, d), number in od_number_rebal.items():
            if od_repo_dict.get(o) is None:
                od_repo_dict[o] = {}
            try:
                od_repo_dict[o][d] = number
            except KeyError:
                od_repo_dict[o] = {d: number}
        # assign repositioning plans
        list_veh_with_changes = []
        vid_to_plan_id = {}
        for o, d_dict in od_repo_dict.items():
            avail_vehicles = avail_for_rebal_vehicles[o]
            shuffle_list = list(d_dict.items())
            np.random.shuffle(shuffle_list)
            for d, number in shuffle_list:
                LOG.info(f"rebal {o} {d} : {number} | {[v.vid for v in avail_vehicles]}")
                plan_ids = repo_to_plan_ids.get( (o, d), [])
                list_veh_obj_with_repos = []
                if o == d and len(plan_ids) > 0:
                    for i, plan_id in enumerate(plan_ids):
                        veh_obj = avail_vehicles[i]
                        vid_to_plan_id[veh_obj.vid] = plan_id
                        list_veh_obj_with_repos.append(veh_obj)
                        LOG.debug(f"assign {veh_obj.vid} with reservation plan {plan_id} (same zone repo)")
                else:
                    for i in range(number):
                        veh_obj = avail_vehicles[i]
                        assign_repo = True
                        if i < len(plan_ids):
                            plan_id = plan_ids[i]
                            vid_to_plan_id[veh_obj.vid] = plan_id
                            LOG.debug(f"assign {veh_obj.vid} with reservation plan {plan_id} (different zone repo)")
                            if self.fleetctrl.reservation_module is not None:
                                _, _, start_pos, start_time = self.fleetctrl.reservation_module.get_supporting_point(sim_time, plan_id=plan_id)
                                start_zone = self._repo_zone_system.get_zone_from_pos(start_pos)
                                # if start_zone == d:
                                #     assign_repo = False
                                #     LOG.debug(f" -> supporting point is in destination zone {d} -> no repositioning")
                        if assign_repo:
                            list_veh_obj_with_repos.extend(self._od_to_veh_plan_assignment(sim_time, o, d, [veh_obj], lock=lock))
                            list_veh_with_changes.extend([veh_obj.vid for veh_obj in list_veh_obj_with_repos])
                        else:
                            list_veh_with_changes.append(veh_obj.vid)
                            list_veh_obj_with_repos.append(veh_obj)
                for veh_obj in list_veh_obj_with_repos:
                    LOG.info(f" -> assign {veh_obj.vid}")
                    avail_vehicles.remove(veh_obj)
        # assign future idle vehicles to reservation plans
        LOG.debug(f"idle vehicles:")
        for time, zone_dict in bin_to_zone_to_idle_veh_objs.items():
            LOG.debug(f" -> at time {time}")
            for zone, veh_objs in zone_dict.items():
                LOG.debug(f"     -> zone {zone} -> {[veh_obj.vid for veh_obj in veh_objs]}")
        list_sorted_time_bins = list(sorted(bin_to_zone_to_idle_veh_objs.keys()))
        for plan_id, bin in plan_id_to_idle_bin.items():
            LOG.debug(f"assign future idle vehicles to reservation plan {plan_id} at {bin}")
            zone, time_bin = bin
            avail_vehicles = None
            for tb in reversed(list_sorted_time_bins):
                if tb <= time_bin and len(bin_to_zone_to_idle_veh_objs.get(tb, {}).get(zone, [])) != 0:
                    avail_vehicles = bin_to_zone_to_idle_veh_objs[tb][zone]
                    break
            if avail_vehicles is None:
                raise EnvironmentError(f"no vehicles available for reservation plan {plan_id} at {bin}")
            veh_obj = random.choice(avail_vehicles)
            vid_to_plan_id[veh_obj.vid] = plan_id
            avail_vehicles.remove(veh_obj)
            LOG.debug(f" -> assign {veh_obj.vid} with reservation plan {plan_id} (future idle)")
        LOG.debug(f"assigned plans {vid_to_plan_id}")
        if self.fleetctrl.reservation_module is not None:
            self.fleetctrl.reservation_module.reassign_supporting_points(sim_time, vid_to_plan_id)
        return list_veh_with_changes
    
    def _progress_vehicles(self, future_rq_atts, sim_time, prog_end_time):
        """ in this method future vehicle states are simulated and input parameters for rebalancing optimization are created
        in each time step
        1) new requests are inserted
            if no vehicle is close by, a new vehicle is created (their traces of the period are stored and used as possible repositioning targets)
        2) vehicles are moved according to their plans
        3) after a repo time bin, new idle vehicles are tracked
        
        at the end, input parameters for the optimization are created and returned
        
        :param future_rq_atts: list of future requests
        :param sim_time: current simulation time
        :param prog_end_time: end time of the simulation
        :return: input parameters for the optimization
            tour_id_subtour_id_parameters: tour_id -> subtour_id -> (o_zone, cost_gain, arr_at_zone, end_zone, start_bin, end_bin)
            start_bin_to_ozone_to_tour_ids: start_bin -> o_zone -> list of (tour_id, subtour_id)
            end_bin_to_dzone_to_tour_ids: end_bin -> d_zone -> list of (tour_id, subtour_id)
            timebin_to_zone_to_idle_vehs: time bin -> zone -> number of idle vehicles
            tour_id_to_plan_id: tour_id -> plan_id # this tour also assigns a reservation plan with plan_id in the reservation module
            bin_to_zone_to_idle_veh_objs: time bin -> zone -> list of idle vehicle objects
        """
        # only currently driving vehicles (idle vehicles will be created)
        veh_objs, veh_plans, idle_veh_objs, unassigned_reservation_plans = self._get_vehs_with_vp_available_for_repo(sim_time)
        
        bin_to_zone_to_idle_veh_objs = {}
        bin_to_zone_to_idle_veh_objs[sim_time] = idle_veh_objs
        
        opt_time_step = self._progress_time_step
        
        insert_time_to_reservation_plan_id = {}
        for plan_id, full_plan in unassigned_reservation_plans.items():
            insert_time = max(full_plan.list_plan_stops[0].get_earliest_start_time() - self._repo_time_step, sim_time)
            insert_time = int(np.floor(insert_time/opt_time_step)) * opt_time_step
            if insert_time > prog_end_time - opt_time_step: # insert plans that are outside of the horizon at the end
                insert_time = prog_end_time - opt_time_step
            try:
                insert_time_to_reservation_plan_id[insert_time].append(plan_id)
            except KeyError:
                insert_time_to_reservation_plan_id[insert_time] = [plan_id]

        rq_dict = self.fleetctrl.rq_dict.copy()
        
        current_new_vid = len(self.fleetctrl.sim_vehicles) + 1
        new_vids_to_ps = {}
        
        vids_with_assignment = {}
        
        pos_to_vids = {}
        plan_id_to_assigned_vid = {}
        
        if len(rq_dict) > 0:
            start_id = max(rq_dict.keys()) + 1
        else:
            start_id = 0

        prqs = {}
        rid_order = {}
        for atts in future_rq_atts:
            if len(atts) == 3:
                t, o_n, d_n = atts
                priority = None
            else:
                t, o_n, d_n, priority = atts
            rid = start_id + len(prqs)
            opt_time = int(np.floor(t/opt_time_step)) * opt_time_step
            try:
                rid_order[opt_time].append(rid)
            except:
                rid_order[opt_time] = [rid]
            prqs[rid] = ArtificialPlanRequest(rid, t, (o_n, None, None), (d_n, None, None), self.routing_engine,
                                                max_wait_time=self.fleetctrl.max_wait_time, max_detour_time_factor=self.fleetctrl.max_dtf,
                                                max_constant_detour_time=self.fleetctrl.max_cdt, boarding_time=self.fleetctrl.const_bt)
            if priority:
                prqs[rid].set_reservation_flag(True)
        
        timebin_to_zone_to_idle_vehs = {} # time bin -> zone -> number of idle vehicles
        
        vid_to_route = {}    # caching routes
        for t in range(sim_time, prog_end_time, opt_time_step):
            # update travel times
            if self.fleetctrl._use_own_routing_engine:
                self.routing_engine.update_network(t)
            # track new idle vehicles at start of time bin
            if t != sim_time and t%self._repo_time_step == 0:
                # update currently idle vehicles
                timebin_to_zone_to_idle_vehs[t] = {}
                LOG.debug(f"update idle vehicles at {t} : large prog end time? {t >= prog_end_time}")
                use_vehicles = [veh for veh in veh_objs.values() if veh.vid  < len(self.fleetctrl.sim_vehicles)]
                veh_objs_new, veh_plans_new, zone_to_idle_vehs, unassigned_reservation_plans_new = \
                    self._get_vehs_with_vp_available_for_repo(t, use_vehicles=use_vehicles, use_vehplans=veh_plans)
                bin_to_zone_to_idle_veh_objs[t] = zone_to_idle_vehs
                for vid, veh_obj_new in veh_objs_new.items():
                    veh_objs[vid] = veh_obj_new
                    veh_plans[vid] = veh_plans_new[vid]
                    LOG.debug(f"update idle veh {veh_obj_new} at {t} -> {veh_plans[vid]}")
                for zone, vehs in zone_to_idle_vehs.items():
                    try:
                        timebin_to_zone_to_idle_vehs[t][zone] += len(vehs)
                    except KeyError:
                        timebin_to_zone_to_idle_vehs[t][zone] = len(vehs)
                    for veh in vehs:
                        del veh_objs[veh.vid]
                        del veh_plans[veh.vid]
                    
                for plan_id, full_plan in unassigned_reservation_plans_new.items():
                    insert_time = max(full_plan.list_plan_stops[0].get_earliest_start_time() - self._repo_time_step, sim_time)
                    insert_time = int(np.floor(insert_time/opt_time_step)) * opt_time_step
                    if insert_time > prog_end_time - opt_time_step: # insert plans that are outside of the horizon at the end
                        insert_time = prog_end_time - opt_time_step
                    try:
                        insert_time_to_reservation_plan_id[insert_time].append(plan_id)
                    except KeyError:
                        insert_time_to_reservation_plan_id[insert_time] = [plan_id]
                unassigned_reservation_plans.update(unassigned_reservation_plans_new)
                LOG.debug(f"new unassigned reservation plans {unassigned_reservation_plans_new}")
            
            # insert offline plans
            new_offline_plans = insert_time_to_reservation_plan_id.get(t)
            if new_offline_plans is not None:
                LOG.debug(f"assign offline plans {new_offline_plans} at {t}")
                for plan_id in new_offline_plans:
                    full_plan = unassigned_reservation_plans[plan_id]
                    full_plan_start_time = full_plan.list_plan_stops[0].get_earliest_start_time()
                    full_plan_start_pos = full_plan.list_plan_stops[0].get_pos()
                    for rid in full_plan.get_involved_request_ids():
                        rq_dict[rid].compute_new_max_trip_time(self.routing_engine, boarding_time=self.fleetctrl.const_bt, max_detour_time_factor=self.fleetctrl.max_dtf,
                                                max_constant_detour_time=self.fleetctrl.max_cdt)
                    best_vid = None
                    best_cfv = float("inf")
                    best_plan = None
                    for vid, veh in veh_objs.items():
                        if vid < len(self.fleetctrl.sim_vehicles):
                            continue
                        current_vp = veh_plans[vid]
                        if len(current_vp.list_plan_stops) != 0:
                            last_ps = current_vp.list_plan_stops[-1]
                            if last_ps.is_locked_end():
                                continue
                            last_ps_end_time = last_ps.get_planned_arrival_and_departure_time()[1]
                            last_ps_end_pos = last_ps.get_pos()
                        else:
                            last_ps_end_time = t
                            last_ps_end_pos = veh.pos
                        if last_ps_end_time > t:
                            continue
                        _, tt, dis = self.routing_engine.return_travel_costs_1to1(last_ps_end_pos, full_plan_start_pos)
                        if tt + last_ps_end_time <= full_plan_start_time:
                            new_vp = current_vp.copy()
                            new_vp.list_plan_stops += [ps.copy() for ps in full_plan.list_plan_stops]
                            feasible = new_vp.update_tt_and_check_plan(veh, t, self.routing_engine)
                            if feasible:
                                cfv = self._sampling_ctrl_function(t, veh, new_vp, rq_dict, self.routing_engine)
                                if cfv < best_cfv:
                                    best_vid = vid
                                    best_cfv = cfv
                                    best_plan = new_vp
                    if best_vid is None:
                        o_zone = self._repo_zone_system.get_zone_from_pos(full_plan_start_pos)
                        centroid = self._repo_zone_system.get_random_centroid_node(o_zone)
                        any_vehicle = self.fleetctrl.sim_vehicles[0]
                        new_veh = SimulationVehicleStruct(any_vehicle, full_plan, t, self.routing_engine, empty_init=True)
                        new_veh.pos = (centroid, None, None)
                        new_veh.vid = current_new_vid
                        new_veh.status = VRL_STATES.IDLE
                        best_vid = current_new_vid
                        best_plan = full_plan.copy()
                        best_plan.update_tt_and_check_plan(new_veh, t, self.routing_engine, keep_feasible=True)
                        veh_objs[best_vid] = new_veh
                        veh_plans[best_vid] = best_plan
                        new_vids_to_ps[best_vid] = []
                        current_new_vid += 1
                    else:
                        vids_with_assignment[best_vid] = 1
                        veh_plans[best_vid] = best_plan
                         
                    LOG.debug(f"reservation plan {plan_id} assigned to {best_vid} at {t} -> {best_plan}")
                    plan_id_to_assigned_vid[plan_id] = best_vid
                        
            # insert new requests
            new_rids = rid_order.get(t)
            if new_rids is not None:
                pos_to_vids = {} # position -> list of vehicle ids
                for veh in veh_objs.values():
                    pos = veh.pos
                    try:
                        pos_to_vids[pos].append(veh.vid)
                    except KeyError:
                        pos_to_vids[pos] = [veh.vid]
                LOG.debug(f"current vids : { [(vid, veh.vid) for vid, veh in veh_objs.items()]}")
                for rid in new_rids: 
                    #LOG.debug(f"insert {rid}")       
                    prq : ArtificialPlanRequest = prqs[rid]
                    rq_dict[rid] = prq
                    #LOG.debug(f"check timing: {prq.get_o_stop_info()[2] - t}")
                    
                    # check for close by vehicles
                    r = self.routing_engine.return_travel_costs_Xto1(pos_to_vids.keys(), prq.get_o_stop_info()[0], 
                                                                        max_cost_value= self.fleetctrl.max_wait_time)
                    rid_vehs = []
                    rid_veh_plans = {}
                    for o_pos, _ ,_,_ in r:
                        vids = pos_to_vids[o_pos] 
                        for vid in vids:
                            rid_vehs.append(veh_objs[vid])  
                            rid_veh_plans[vid] = veh_plans[vid] 
                            
                    # add possible new idle vehicle
                    o_zone = self._repo_zone_system.get_zone_from_pos(prq.get_o_stop_info()[0])
                    centroid = self._repo_zone_system.get_random_centroid_node(o_zone)
                    any_vehicle = self.fleetctrl.sim_vehicles[0]
                    new_veh = SimulationVehicleStruct(any_vehicle, VehiclePlan(any_vehicle, t, self.routing_engine, []), t, self.routing_engine, empty_init=True)
                    new_veh.pos = (centroid, None, None)
                    new_veh.vid = current_new_vid
                    new_veh.status = VRL_STATES.IDLE
                    
                    rid_vehs.append(new_veh)
                    rid_veh_plans[current_new_vid] = VehiclePlan(new_veh, t, self.routing_engine, [])
                                
                    insert_sols = insert_prq_in_selected_veh_list(rid_vehs, rid_veh_plans, prq,
                                                                    self._sampling_ctrl_function, self.routing_engine, rq_dict,
                                                                    t, self.fleetctrl.const_bt, self.fleetctrl.add_bt)
                    if len(insert_sols) == 0:
                        LOG.warning(f"no sampling solution found? {prq}")
                        continue
                    best_sol = min(insert_sols, key=lambda x:x[2])
                    best_vid, best_plan, _ = best_sol
                    if best_vid == current_new_vid:
                        if len(insert_sols) > 1:
                            LOG.debug(f"new vid would be best, but other option")
                            best_vid, best_plan, _ = sorted(insert_sols, key = lambda x:x[2])[1]
                    veh_plans[best_vid] = best_plan
                    LOG.debug(f"fc prq {prq} assigned to {best_vid} which is new? {best_vid == current_new_vid}")
                    if best_vid == current_new_vid: # new vid is introduced
                        veh_objs[new_veh.vid] = new_veh
                        pos = new_veh.pos
                        try:
                            pos_to_vids[pos].append(new_veh.vid)
                        except KeyError:
                            pos_to_vids[pos] = [new_veh.vid]
                            
                        new_vids_to_ps[new_veh.vid] = []
                        current_new_vid += 1
                    else: # rid is accomodated by onroute vehicle
                        vids_with_assignment[vid] = 1
                        
            # move vehicles
            for vid, veh in veh_objs.items():
                passed_ps, vid_to_route = move_vehicle_according_to_plan(veh, veh_plans[vid], t, opt_time_step, self.routing_engine, rq_dict, vid_to_route, self.fleetctrl.begin_approach_buffer_time)
                if len(passed_ps) > 0:
                    if new_vids_to_ps.get(veh.vid) is not None:
                        for ps in passed_ps:
                            new_vids_to_ps[veh.vid].append(ps)

        if self.fleetctrl._use_own_routing_engine:
            self.routing_engine.reset_network(sim_time)

        # create optimization input parameters
        #tour_id = 0
        tour_id_subtour_id_parameters = {} # tour_id -> subtour_id -> (o_zone, cost_gain, arr_at_zone, end_zone, start_bin, end_bin)
        start_bin_to_ozone_to_tour_ids = {} # start_bin -> o_zone -> list of (tour_id, subtour_id)
        end_bin_to_dzone_to_tour_ids = {} # end_bin -> d_zone -> list of (tour_id, subtour_id)
        tour_id_to_plan_id = {val : key for key, val in plan_id_to_assigned_vid.items()}
        LOG.debug(f"tour ids to be assigned: {tour_id_to_plan_id}")
        for vid, list_ps in new_vids_to_ps.items():
            tour_id = vid
            is_reservation_tour = False
            if tour_id_to_plan_id.get(vid) is not None:
                is_reservation_tour = True
            tour_id_subtour_id_parameters[tour_id] = {}
            subtour_id = 0
            
            vp: VehiclePlan = veh_plans.get(vid, VehiclePlan(self.fleetctrl.sim_vehicles[0], sim_time, self.routing_engine, []))
            LOG.debug(f"create input params for {vid}: has reservations? {is_reservation_tour}")
            start_time = sim_time
            if len(list_ps) > 0:
                o_z = self._repo_zone_system.get_zone_from_pos(list_ps[0].get_pos())
            else:
                o_z = self._repo_zone_system.get_zone_from_pos(vp.list_plan_stops[0].get_pos())
                
            centroid = self._repo_zone_system.get_random_centroid_node(o_z)
            any_vehicle = next(iter(veh_objs.values()))
            new_veh = SimulationVehicleStruct(any_vehicle, VehiclePlan(any_vehicle, sim_time, self.routing_engine, []), sim_time, self.routing_engine, empty_init=True)
            new_veh.pos = (centroid, None, None)
            new_veh.vid = -1
            new_veh.status = VRL_STATES.IDLE

            full_vp = VehiclePlan(new_veh, start_time, self.routing_engine, list_ps + vp.list_plan_stops )
            full_vp.update_tt_and_check_plan(new_veh, start_time, self.routing_engine, keep_feasible=True)
            LOG.debug(f" -- full vp {full_vp}")
            #cost_gain = self.fleetctrl.compute_VehiclePlan_utility(start_time, new_veh, full_vp)
            cost_gain = self._sampling_ctrl_function(start_time, new_veh, full_vp, rq_dict, self.routing_engine)
            arr_time = max(full_vp.list_plan_stops[0].get_planned_arrival_and_departure_time()[0], full_vp.list_plan_stops[0].get_earliest_start_time())
            arr_at_zone = arr_time - self.routing_engine.return_travel_costs_1to1(new_veh.pos, full_vp.list_plan_stops[0].get_pos())[0]
            
            end_time = full_vp.list_plan_stops[-1].get_planned_arrival_and_departure_time()[1]
            start_pos = full_vp.list_plan_stops[0].get_pos()
            end_pos = full_vp.list_plan_stops[-1].get_pos()
            end_zone = self._repo_zone_system.get_zone_from_pos(end_pos)
            
            LOG.debug(f" -- add base tour {(tour_id, subtour_id)} {(o_z, cost_gain, arr_at_zone)}")
            
            start_bin = int(arr_at_zone / self._repo_time_step) * self._repo_time_step
            end_bin = int(end_time / self._repo_time_step) * self._repo_time_step
            LOG.debug(f" -> end time and bin: {end_time} {end_bin}")
            
            tour_id_subtour_id_parameters[tour_id][subtour_id] = (o_z, cost_gain, arr_at_zone, end_zone, start_bin, end_bin)
            
            try:
                start_bin_to_ozone_to_tour_ids[start_bin][o_z].append( (tour_id, subtour_id) )
            except KeyError:
                try:
                    start_bin_to_ozone_to_tour_ids[start_bin][o_z] = [(tour_id, subtour_id)]
                except KeyError:
                    start_bin_to_ozone_to_tour_ids[start_bin] = {o_z : [(tour_id, subtour_id)]}
            try:
                end_bin_to_dzone_to_tour_ids[end_bin][end_zone].append( (tour_id, subtour_id) )
            except KeyError:
                try:
                    end_bin_to_dzone_to_tour_ids[end_bin][end_zone] = [(tour_id, subtour_id)]
                except KeyError:
                    end_bin_to_dzone_to_tour_ids[end_bin] = {end_zone : [(tour_id, subtour_id)]}
                
            subtour_id += 1
            
            def add_subtour(full_vp: VehiclePlan, subtour_id):
                o_z = self._repo_zone_system.get_zone_from_pos(full_vp.list_plan_stops[0].get_pos())
                
                centroid = self._repo_zone_system.get_random_centroid_node(o_z)
                any_vehicle = next(iter(veh_objs.values()))
                new_veh = SimulationVehicleStruct(any_vehicle, VehiclePlan(any_vehicle, sim_time, self.routing_engine, []), sim_time, self.routing_engine, empty_init=True)
                new_veh.pos = (centroid, None, None)
                new_veh.vid = -1
                new_veh.status = VRL_STATES.IDLE

                full_vp = VehiclePlan(new_veh, start_time, self.routing_engine, full_vp.list_plan_stops )
                LOG.debug(f" -- check subtour {full_vp}")
                #cost_gain = self.fleetctrl.compute_VehiclePlan_utility(start_time, new_veh, full_vp)
                cost_gain = self._sampling_ctrl_function(start_time, new_veh, full_vp, rq_dict, self.routing_engine)
                arr_time = max(full_vp.list_plan_stops[0].get_planned_arrival_and_departure_time()[0], full_vp.list_plan_stops[0].get_earliest_start_time())
                arr_at_zone = arr_time - self.routing_engine.return_travel_costs_1to1(new_veh.pos, full_vp.list_plan_stops[0].get_pos())[0]
                LOG.debug(f" -- add subtour {(tour_id, subtour_id)} {(o_z, cost_gain, arr_at_zone)}")
                
                start_bin = int(arr_at_zone / self._repo_time_step) * self._repo_time_step
                end_bin = int(end_time / self._repo_time_step) * self._repo_time_step
                
                tour_id_subtour_id_parameters[tour_id][subtour_id] = (o_z, cost_gain, arr_at_zone, end_zone, start_bin, end_bin)
                
                try:
                    start_bin_to_ozone_to_tour_ids[start_bin][o_z].append( (tour_id, subtour_id) )
                except KeyError:
                    try:
                        start_bin_to_ozone_to_tour_ids[start_bin][o_z] = [(tour_id, subtour_id)]
                    except KeyError:
                        start_bin_to_ozone_to_tour_ids[start_bin] = {o_z : [(tour_id, subtour_id)]}
                try:
                    end_bin_to_dzone_to_tour_ids[end_bin][end_zone].append( (tour_id, subtour_id) )
                except KeyError:
                    try:
                        end_bin_to_dzone_to_tour_ids[end_bin][end_zone] = [(tour_id, subtour_id)]
                    except KeyError:
                        end_bin_to_dzone_to_tour_ids[end_bin] = {end_zone : [(tour_id, subtour_id)]}
                
            
            # create also parameters for subplans
            if len(full_vp.get_involved_request_ids()) > 0:
                #to_remove = full_vp.list_plan_stops[0].get_list_boarding_rids()
                while True:
                    arrived_at_reservation = False
                    for i, ps in enumerate(full_vp.list_plan_stops):
                        found = False
                        for rid in ps.get_list_boarding_rids():
                            if rq_dict[rid].get_reservation_flag():
                                arrived_at_reservation = True
                                LOG.debug(f" -- arrived at reservation {rid}")
                                break
                        if arrived_at_reservation:
                            break
                        for rid in ps.get_list_boarding_rids():
                            full_vp = simple_remove(new_veh, full_vp, rid, sim_time, self.routing_engine, self._sampling_ctrl_function, rq_dict, self.fleetctrl.const_bt, self.fleetctrl.add_bt)
                            found = True
                        if found:
                            break
                    if len(full_vp.list_plan_stops) == 0 or len(full_vp.get_involved_request_ids()) == 0 or arrived_at_reservation:
                        if len(full_vp.get_involved_request_ids()) == 0 and len(full_vp.list_plan_stops) == 1 and full_vp.list_plan_stops[0].is_locked_end():
                            add_subtour(full_vp, subtour_id)
                            subtour_id += 1
                        break
                    add_subtour(full_vp, subtour_id)
                    subtour_id += 1
        
        #exit()           
        return tour_id_subtour_id_parameters, start_bin_to_ozone_to_tour_ids, end_bin_to_dzone_to_tour_ids, timebin_to_zone_to_idle_vehs, tour_id_to_plan_id, bin_to_zone_to_idle_veh_objs
    
    def _solve_sampling_repositioning(self, sim_time: int, avail_for_rebal_vehicles: Dict[int, List[int]], sample_tour_id_subtour_id_parameters, 
                                                                   sample_start_bin_to_ozone_to_tour_ids, sample_end_bin_to_dzone_to_tour_ids,
                                                                   sample_timebin_to_zone_to_add_idle_vehs, sample_tour_id_to_plan_id):
        """ solves the repositioning problem with the given input data 
        :param sim_time: current simulation time
        :param avail_for_rebal_vehicles: zone -> list of vehicle ids
        :param sample_tour_id_subtour_id_parameters: list of tour_id -> subtour_id -> (o_zone, cost_gain, arr_at_zone, end_zone, start_bin, end_bin) per sample
        :param sample_start_bin_to_ozone_to_tour_ids: list of start_bin -> o_zone -> list of (tour_id, subtour_id) per sample
        :param sample_end_bin_to_dzone_to_tour_ids: list of end_bin -> d_zone -> list of (tour_id, subtour_id) per sample
        :param sample_timebin_to_zone_to_add_idle_vehs: list of time bin -> zone -> number of idle vehicles per sample
        :param sample_tour_id_to_plan_id: list of tour_id -> plan_id per sample (reservation plan that has to be assigned)
        :param lock: bool if repositioning should be locked
        :return: dict (o_zone, d_zone) -> number of rebalancing vehicles (new repositioning targets), 
                dict plan_id -> (o_zone, d_zone) (this rebal trips has to be combined with reservation plan plan_id,
                dict plan_id -> (o_zone, time_bin) a vehicle becoming idle in this time bin has to be combined with reservation plan plan_id"""
        
        import gurobipy as grp
        
        model_name = f"rp_rebal_solve_matching_{sim_time}"
        with grp.Env(empty=True) as env:
            if self.fleetctrl.log_gurobi:
                with open(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], "gurobi_log.log"), "a") as f:
                    f.write(f"\n\n{model_name}\n\n")
                env.setParam('OutputFlag', 1)
                env.setParam('LogToConsole', 0)
                env.setParam('LogFile', os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], "gurobi_log.log") )
                env.start()
            else:
                env.setParam('OutputFlag', 0)
                env.setParam('LogToConsole', 0)
                env.start()

            m = grp.Model(model_name, env = env)

            m.setParam(grp.GRB.param.Threads, self.fleetctrl.n_cpu)
            m.setParam('TimeLimit', OPT_TIME_LIMIT)
            m.setParam("MIPGap", GUROBI_MIPGAP)
        
            vars = {}
            current_T_od_vars = {} # time bin -> (o, d) -> name
            future_T_od_vars = {}  # sample -> time_bin -> (o,d) -> name
            current_T_from_o_vars = {}
            current_T_to_d_vars = {}
            future_T_from_o_vars = {}
            future_T_to_d_vars = {}
            T_od_to_sample_to_tours_vars = {}
            sample_to_tours_to_subtours_vars = {}
            T_to_sample_to_ozone_subtour_vars = {}
            T_to_sample_to_dzone_subtour_vars = {}
            
            # create variables
            for sample, tour_id_subtour_id_parameters in enumerate(sample_tour_id_subtour_id_parameters):
                future_T_od_vars[sample] = {}
                future_T_from_o_vars[sample] = {}
                future_T_to_d_vars[sample] = {}
                for tour_id, subtour_id_parameters in tour_id_subtour_id_parameters.items():
                    for subtour_id, (o_z, cost_gain, arr_at_zone, end_zone, start_bin, end_bin) in subtour_id_parameters.items():
                        if start_bin > sim_time + self.list_horizons[1]:
                            add_bin = max(start_bin - 3600, sim_time + self.list_horizons[1])
                            time_bin_iterator = [tb for tb in range(sim_time + self.list_horizons[0], sim_time + self.list_horizons[1], self._repo_time_step)] + [add_bin]
                        else:
                            time_bin_iterator = range(sim_time + self.list_horizons[0], sim_time + self.list_horizons[1], self._repo_time_step)
                        for time_bin in time_bin_iterator:
                            time_bin_int = int( (time_bin - sim_time + self.list_horizons[0])/self._repo_time_step )
                            factor = self._gamma ** time_bin_int
                            if time_bin == sim_time + self.list_horizons[0]:
                                iterator = avail_for_rebal_vehicles.keys()
                            else:
                                iterator = self._repo_zone_system.get_all_zones()
                            for zone in iterator:
                                tt = self._get_od_zone_travel_info(sim_time, zone, o_z)[0]
                                if time_bin + tt <= arr_at_zone:
                                    # od variable
                                    if time_bin_int == 0:
                                        od_name = f"od_cur_{time_bin}_{zone}_{o_z}"
                                        if vars.get(od_name) is None:
                                            var = m.addVar(name=od_name, obj=tt, vtype=grp.GRB.INTEGER)
                                            vars[od_name] = var
                                            try:
                                                current_T_od_vars[time_bin][(zone, o_z)] = od_name
                                            except KeyError:
                                                current_T_od_vars[time_bin] = {(zone, o_z): od_name}
                                            try:
                                                current_T_from_o_vars[time_bin][zone].append(od_name)
                                            except KeyError:
                                                try:
                                                    current_T_from_o_vars[time_bin][zone] = [od_name]
                                                except KeyError:
                                                    current_T_from_o_vars[time_bin] = {zone: [od_name]}
                                            try:
                                                current_T_to_d_vars[time_bin][o_z].append(od_name)
                                            except KeyError:
                                                try:
                                                    current_T_to_d_vars[time_bin][o_z] = [od_name]
                                                except KeyError:
                                                    current_T_to_d_vars[time_bin] = {o_z: [od_name]}
                                    else:
                                        # future rebalancing trips per sample
                                        od_name = f"od_future_{sample}_{time_bin}_{zone}_{o_z}"
                                        if vars.get(od_name) is None:
                                            var = m.addVar(name=od_name, obj=tt * factor / self.N_samples, vtype=grp.GRB.INTEGER)
                                            vars[od_name] = var
                                            try:
                                                future_T_od_vars[sample][time_bin][(zone, o_z)] = od_name
                                            except KeyError:
                                                future_T_od_vars[sample][time_bin] = {(zone, o_z): od_name}
                                            try:
                                                future_T_from_o_vars[sample][time_bin][zone].append(od_name)
                                            except KeyError:
                                                try:
                                                    future_T_from_o_vars[sample][time_bin][zone] = [od_name]
                                                except KeyError:
                                                    future_T_from_o_vars[sample][time_bin] = {zone: [od_name]}
                                            try:
                                                future_T_to_d_vars[sample][time_bin][o_z].append(od_name)
                                            except KeyError:
                                                try:
                                                    future_T_to_d_vars[sample][time_bin][o_z] = [od_name]
                                                except KeyError:
                                                    future_T_to_d_vars[sample][time_bin] = {o_z: [od_name]}
                                    
                                    # tour variable        
                                    name = f"t_z_{zone}_{time_bin}_s_{sample}_ti_{tour_id}_sti_{subtour_id}"
                                    var = m.addVar(name=name, obj=cost_gain * factor  / self.N_samples , vtype=grp.GRB.BINARY)
                                    vars[name] = var
                                    try:
                                        sample_to_tours_to_subtours_vars[sample][tour_id].append(name)
                                    except KeyError:
                                        try:
                                            sample_to_tours_to_subtours_vars[sample][tour_id] = [name]
                                        except KeyError:
                                            sample_to_tours_to_subtours_vars[sample] = {tour_id: [name]}
                                            
                                    try:
                                        T_od_to_sample_to_tours_vars[(time_bin, zone, o_z)][sample].append(name)
                                    except KeyError:
                                        try:
                                            T_od_to_sample_to_tours_vars[(time_bin, zone, o_z)][sample] = [name]
                                        except KeyError:
                                            T_od_to_sample_to_tours_vars[(time_bin, zone, o_z)] = {sample: [name]}
                                            
                                    try:
                                        T_to_sample_to_ozone_subtour_vars[(start_bin, o_z)][sample].append(name)
                                    except KeyError:
                                        try:
                                            T_to_sample_to_ozone_subtour_vars[(start_bin, o_z)][sample] = [name]
                                        except KeyError:
                                            T_to_sample_to_ozone_subtour_vars[(start_bin, o_z)] = {sample: [name]}
                                    
                                    if sample_tour_id_to_plan_id[sample].get(tour_id) is None:        
                                        try:
                                            T_to_sample_to_dzone_subtour_vars[(end_bin, end_zone)][sample].append(name)
                                        except KeyError:
                                            try:
                                                T_to_sample_to_dzone_subtour_vars[(end_bin, end_zone)][sample] = [name]
                                            except KeyError:
                                                T_to_sample_to_dzone_subtour_vars[(end_bin, end_zone)] = {sample: [name]}                      
            
            # create constraints
            
            # number of rebalancing vehicles
            
            # current
            for time_bin, o_to_list_vars in current_T_from_o_vars.items():
                for o_zone, list_vars in o_to_list_vars.items():
                    lhs = sum(vars[x] for x in list_vars)
                    V_0_idle = len(avail_for_rebal_vehicles.get(o_zone, []))
                    m.addConstr(lhs <= V_0_idle, name = f"o_constr_c_{time_bin}_{o_zone}" )
                
            # future sample
            for sample, T_from_o_vars in future_T_from_o_vars.items():
                for time_bin, o_to_list_vars in T_from_o_vars.items():
                    for o_zone, list_vars in o_to_list_vars.items():
                        lhs = sum(vars[x] for x in list_vars)
                        V_0_idle = len(avail_for_rebal_vehicles.get(o_zone, []))
                        prev_terms = grp.LinExpr()
                        for time_bin_2 in range(sim_time + self.list_horizons[0], time_bin, self._repo_time_step):
                            prev_rebalanced_vehicles = sum(vars[x] for x in T_from_o_vars.get(time_bin_2, {}).get(o_zone, [])) + sum(vars[x] for x in current_T_from_o_vars.get(time_bin_2, {}).get(o_zone, []))
                            prev_terms -= prev_rebalanced_vehicles
                            
                            new_idle_vehs = sample_timebin_to_zone_to_add_idle_vehs[sample].get(time_bin_2, {}).get(o_zone, 0)
                            
                            ending_tours = sum(vars[x] for x in T_to_sample_to_dzone_subtour_vars.get((time_bin_2, o_zone), {}).get(sample, []))
                            
                            prev_terms += new_idle_vehs + ending_tours
                        m.addConstr(lhs <= V_0_idle + prev_terms, name = f"o_constr_fut_{time_bin}_{o_zone}_{sample}" )

            # reachable tours
            for (time_bin, o_zone, d_zone), sample_list_vars in T_od_to_sample_to_tours_vars.items():
                for sample, list_vars in sample_list_vars.items():
                    lhs = sum(vars[x] for x in list_vars)
                    T_od_var = current_T_od_vars.get(time_bin, {}).get((o_zone, d_zone))
                    if T_od_var is None:
                        T_od_var = future_T_od_vars.get(sample, {}).get(time_bin, {}).get((o_zone, d_zone))
                    if T_od_var is not None:
                        m.addConstr(lhs <= vars[T_od_var], name = f"tour_constr_{time_bin}_{o_zone}_{d_zone}_{sample}" )
                    else:
                        m.addConstr(lhs == 0, name = f"tour_constr_{time_bin}_{o_zone}_{d_zone}_{sample}" )
                        
            # only one subtour per tour and sample
            for sample, tours_to_subtours_vars in sample_to_tours_to_subtours_vars.items():
                for tour, subtours_vars in tours_to_subtours_vars.items():
                    lhs = sum(vars[x] for x in subtours_vars)
                    if sample_tour_id_to_plan_id[sample].get(tour) is not None:
                        m.addConstr(lhs == 1, name = f"subtour_constr_{sample}_{tour}_force" )
                    else:
                        m.addConstr(lhs <= 1, name = f"subtour_constr_{sample}_{tour}" )
                                                        
            if WRITE_PROBLEM:
                m.write(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], f"opt_problem_matching_{sim_time}.lp"))
                
            m.optimize()

            # retrieve solution
            sample_to_look_for = 0

            vals = m.getAttr('X', vars)
            #print(vals)
            od_number_rebal = {}
            future_od_number_rebal = {}
            vid_found = {}
            sample_assigned_reservation_tour_variables = []
            sample_assigned_repo_variables = []
            sample_all_tour_variables = []
            LOG.debug("ODs with rebalancing vehicles:")
            for x in vals:
                v = vals[x]
                #LOG.debug(f"{v}, {x}")
                v = int(np.round(v))
                if v == 0:
                    continue
                if x.startswith(f"od"):
                    v = int(np.round(v))
                    if v == 0:
                        continue
                    x = x.split("_")
                    if len(x) == 5:
                        _, _, time_bin, o_z, d_z = x
                        sample = 0
                    else:
                        _, _, sample, time_bin, o_z, d_z = x
                    o_z = int(o_z)
                    d_z = int(d_z)
                    time_bin = int(time_bin)
                    # if o_z == d_z:
                    #     continue
                    if len(x) == 5:
                        od_number_rebal[(o_z, d_z)] = v
                    else:
                        future_od_number_rebal[(time_bin, o_z, d_z)] = v/self.N_samples
                    if sample == sample_to_look_for:
                        sample_assigned_repo_variables.append( (x, v) )
                # tour variables f"t_z_{zone}_{time_bin}_s_{sample}_ti_{tour_id}_sti_{subtour_id}"
                elif x.startswith("t_z"):
                    tour_id = int(x.split("_")[-3])
                    sample = int(x.split("_")[-5])
                    if sample_tour_id_to_plan_id[sample].get(tour_id) is not None:
                        LOG.debug(f"tour {tour_id} assigned to {sample_tour_id_to_plan_id[sample][tour_id]} in variable {x}")
                        LOG.debug(f"tours in sample: {sample_tour_id_to_plan_id[sample]}")
                        if sample == sample_to_look_for:
                            sample_assigned_reservation_tour_variables.append(x)
                    elif sample == sample_to_look_for:
                        sample_all_tour_variables.append(x)
            LOG.debug(f"assigned reservation tours: {sample_assigned_reservation_tour_variables}  {sample_tour_id_to_plan_id}")
            
            # reconstruct assignment to find vehicles assigned to reservation tour
            tour_to_sol_vars = {} # tour id -> list assigned rebal variables
            od_to_origins = {} # zone (zone_id, time_bin) -> (assigned_rebal_vars, incoming tours, idle_vehicles)
            tour_to_number_end_zones = {}
            
            # check flow of constraints
            for constr in m.getConstrs():
                if constr.ConstrName.startswith("subtour"):
                    continue
                if constr.ConstrName.startswith("o_constr"): # defines incoming and outgoing trips per zone/time_bin with idle vehicles
                    if constr.ConstrName.startswith("o_constr_fut"): # assignment of current rebal trips is trivial (defined by tour_constr)
                        sample = int(constr.ConstrName.split("_")[-1])
                        if sample != sample_to_look_for:
                            continue
                        lexpr = m.getRow(constr)
                        time_bin = int(constr.ConstrName.split("_")[3])
                        origin_zone = int(constr.ConstrName.split("_")[4])
                        idle_vehicles = int(round(constr.RHS))
                        incoming_tours = []
                        od_vars = []
                        #print("")
                        #print(constr.ConstrName, constr.RHS)
                        for i in range(lexpr.size()):
                            var = lexpr.getVar(i)
                            coeff = lexpr.getCoeff(i)
                            sol = var.x
                            sol = int(round(sol))
                            if sol != 0:
                                #print(var.varName, coeff, sol)
                                if var.varName.startswith("od"):
                                    var_time_bin = int(var.varName.split("_")[-3])
                                    if var_time_bin < time_bin:
                                        idle_vehicles -= sol
                                    if var_time_bin == time_bin:
                                        od_vars.append( (var.varName, sol) )
                                else:
                                    incoming_tours.append( var.varName )
                                    tour_to_number_end_zones[var.varName] = tour_to_number_end_zones.get(var.varName, 0) + sol
                        if len(od_vars) == 0:
                            continue
                        # print(constr)
                        # print("remaining idle vehicles", idle_vehicles)
                        # print("incoming tours", incoming_tours)
                        # print("od vars", od_vars)
                        # print("")
                        od_to_origins[(origin_zone, time_bin)] = (od_vars, incoming_tours, idle_vehicles)
                        # print()
                        # continue
                if constr.ConstrName.startswith("tour_constr"): # tour -> rebal trip
                    sample = int(constr.ConstrName.split("_")[-1])
                    if sample != sample_to_look_for:
                        continue
                    # print(constr.ConstrName, constr.RHS)
                    # print(m.getRow(constr))
                    # print("")
                    lexpr = m.getRow(constr)
                    assigned_tours = []
                    assigned_rebals = []
                    for i in range(lexpr.size()):
                        var = lexpr.getVar(i)
                        coeff = lexpr.getCoeff(i)
                        sol = var.x
                        sol = int(round(sol))
                        #print(var, coeff)
                        if sol != 0:
                            if var.varName.startswith("t_z"):
                                assigned_tours.append(var.varName)
                            elif var.varName.startswith("od"):
                                assigned_rebals.append( (var.varName, sol) )
                    if len(assigned_tours) > 0:
                        for tour in assigned_tours:
                            tour_to_sol_vars[tour] = assigned_rebals
            
            # retrace assigned reservation trip to originating rebal trip (if future rebal assigned, this has to point to an vehicle becoming idle in the future)                
            already_assigned = {}
            def trace_back(tour_variable):
                """ traces back to the predecessing rebal variable"""
                # check if od_cur is in assigned rebals
                #print("trace back", tour_variable, tour_to_sol_vars[tour_variable])
                od_cur_var = None
                for od_var, sol in tour_to_sol_vars[tour_variable]:
                    if od_var.startswith("od_cur"):
                        if already_assigned.get(od_var, 0) < sol:
                            od_cur_var = (od_var, sol)
                            break
                if od_cur_var is not None:
                    return ("rebal", od_cur_var)
                # trace back future rebal trip to originating trips
                considered_var = None
                for od_var, sol in tour_to_sol_vars[tour_variable]:
                    if already_assigned.get(od_var, 0) < sol:
                        considered_var = (od_var, sol)
                        break
                if considered_var is None:
                    raise Exception("No feasible assignment found")
                
                o_zone = int(considered_var[0].split("_")[4])
                time_bin = int(considered_var[0].split("_")[3])
                zone_vars = od_to_origins[(o_zone, time_bin)]
                
                if zone_vars[2] > already_assigned.get( (o_zone, time_bin) , 0):
                    return ("idle", (o_zone, time_bin))
                
                for tour in sorted(zone_vars[1], key=lambda x: tour_to_number_end_zones.get(x, 0), reverse=False):
                    if already_assigned.get(tour) is None:
                        return ("tour", tour)
                
                LOG.warning(f"tour: {tour_variable}")
                LOG.warning(f"tour to sol: {tour_to_sol_vars[tour_variable]}")
                LOG.warning(f"already assigned: {already_assigned}")
                LOG.warning(f"zones: {zone_vars}")
                raise Exception("No feasible assignment found")
                
            LOG.debug("retraced tours:")
            plan_id_to_repo_trip = {}   #plan_id -> (o_zone, d_zone)
            plan_id_to_idle_bin = {}    #plan_id -> (o_zone, time_bin) (where vehicles become idle)
            for assigned_tour in sample_assigned_reservation_tour_variables:
                #print("evaluate assigned tour: ", assigned_tour)
                prev_tour = ("tour", assigned_tour)
                all_tour = [prev_tour]
                while prev_tour[0] == "tour":
                    prev_tour = trace_back(prev_tour[1])
                    all_tour.append(prev_tour)
                all_tour = [all_tour[i] for i in range(len(all_tour)-1, -1, -1)]
                LOG.debug(f"{all_tour}")
                for tour in all_tour:
                    already_assigned[tour[1]] = already_assigned.get(tour[1], 0) + 1
                
                sample_tour_id = int(assigned_tour.split("_")[-3])
                plan_id = sample_tour_id_to_plan_id[sample_to_look_for][sample_tour_id]
                if all_tour[0][0] == "rebal": #('rebal', ('od_cur_0_29_29', 1))
                    u = all_tour[0][1][0].split("_")
                    o_zone = int(u[3])
                    d_zone = int(u[4])
                    plan_id_to_repo_trip[plan_id] = (o_zone, d_zone)
                elif all_tour[0][0] == "idle": #('idle', (2, 1800))
                    plan_id_to_idle_bin[plan_id] = all_tour[0][1]
                

            if WRITE_SOL:
                import pandas as pd
                avail_sol_df_list = []
                for od, repos in od_number_rebal.items():
                    avail_sol_df_list.append({
                        "o_zone_id" : od[0],
                        "d_zone_id" : od[1],
                        "repos" : repos,
                        "time_bin" : sim_time + self.list_horizons[0],
                    })
                for tod, repos in future_od_number_rebal.items():
                    avail_sol_df_list.append({
                        "o_zone_id" : tod[1],
                        "d_zone_id" : tod[2],
                        "repos" : repos,
                        "time_bin" : tod[0],
                    })
                pd.DataFrame(avail_sol_df_list).to_csv(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], f"repo_results_{sim_time}.csv"), index=False)

        return od_number_rebal, plan_id_to_repo_trip, plan_id_to_idle_bin 
    
    def _od_to_veh_plan_assignment(self, sim_time, origin_zone_id, destination_zone_id, list_veh_to_consider,
                                   destination_node=None, lock = True):
        """This method translates an OD repositioning assignment for one of the vehicles in list_veh_to_consider.
        This method adds a PlanStop at a random node in the destination zone, thereby calling
        the fleet control class to make all required data base entries.

        :param sim_time: current simulation time
        :param origin_zone_id: origin zone id
        :param destination_zone_id: destination zone id
        :param list_veh_to_consider: list of (idle/repositioning) simulation vehicle objects
        :param destination_node: if destination node is given, it will be prioritized over picking a random node in
                                    the destination zone
        :param lock: indicates if vehplan should be locked
        :return: list_veh_objs with new repositioning assignments
        """
        # v0) randomly pick vehicle, randomly pick destination node in destination zone
        random.seed(sim_time)
        veh_obj = random.choice(list_veh_to_consider)
        veh_plan = self.fleetctrl.veh_plans[veh_obj.vid]
        if destination_node is None:
            destination_node = self._repo_zone_system.get_random_centroid_node(destination_zone_id)
            LOG.debug("repositioning {} to zone {} with centroid {}".format(veh_obj.vid, destination_zone_id,
                                                                            destination_node))
            if destination_node < 0:
                destination_node = self._repo_zone_system.get_random_node(destination_zone_id)
        ps = RoutingTargetPlanStop((destination_node, None, None), locked=lock, planstop_state=G_PLANSTOP_STATES.REPO_TARGET)
        if len(veh_plan.list_plan_stops) == 0 or not veh_plan.list_plan_stops[-1].is_locked_end():
            veh_plan.add_plan_stop(ps, veh_obj, sim_time, self.routing_engine)
        else:
            new_list_plan_stops = veh_plan.list_plan_stops[:-1] + [ps] + veh_plan.list_plan_stops[-1:]
            veh_plan.list_plan_stops = new_list_plan_stops
            veh_plan.update_tt_and_check_plan(veh_obj, sim_time, self.routing_engine, keep_feasible=True)
            if not veh_plan.is_feasible():
                LOG.warning("veh plan not feasible after assigning repo with reservation! {}".format(veh_plan))
        self.fleetctrl.assign_vehicle_plan(veh_obj, veh_plan, sim_time)
        if lock:
            self.fleetctrl.lock_current_vehicle_plan(veh_obj.vid)
        return [veh_obj]
    
    def _get_od_zone_travel_info(self, sim_time, o_zone_id, d_zone_id):
        """This method returns OD travel times on zone level.

        :param o_zone_id: origin zone id
        :param d_zone_id: destination zone id
        :return: tt, dist
        """
        # v0) pick random node and compute route
        loop_iter = 0
        while True:
            o_pos = self.routing_engine.return_node_position(self._repo_zone_system.get_random_centroid_node(o_zone_id))
            d_pos = self.routing_engine.return_node_position(self._repo_zone_system.get_random_centroid_node(d_zone_id))
            if o_pos[0] >= 0 and d_pos[0] >= 0:
                route_info = self.routing_engine.return_travel_costs_1to1(o_pos, d_pos)
                if route_info:
                    return route_info[1], route_info[2]
                loop_iter += 1
                if loop_iter == 10:
                    break
            else:
                break
        # TODO # v1) think about the use of centroids!
        return 10000000, 10000000
    
    
# Fallback variant in case an error occurs

class FullSamplingRidePoolingRebalancingMultiStage_Fallback(RepositioningBase):
    def __init__(self, fleetctrl: FleetControlBase, operator_attributes: dict, dir_names: dict, solver: str = "Gurobi"):
        super().__init__(fleetctrl, operator_attributes, dir_names, solver)
        self.N_samples = int(operator_attributes.get(G_OP_REPO_NR_SAMPLES, 1)) # number of samples used for repositioning
        #self._exploration_cost_weight = operator_attributes.get("op_rp_rebal_exploration_cost", None)
        #self._exploration_request_per_sample = 1
        self._sampling_ctrl_function = self.fleetctrl.vr_ctrl_f# update_ctrl_function(self.fleetctrl.vr_ctrl_f)
        self._gamma = operator_attributes.get(G_OP_REPO_SAMPLE_GAMMA, 0.5) # weight of future rewards
        self._progress_time_step = operator_attributes.get(G_RA_REOPT_TS, 60)
        self._repo_time_step = fleetctrl.repo_time_step
        self.min_reservation_buffer = max(operator_attributes.get(G_OP_REPO_RES_PUF, 3600), self.list_horizons[1])
        self._prioritize_reservations = operator_attributes.get(G_OP_REPO_RES_PRIORITIZE, True)
        
    def determine_and_create_repositioning_plans(self, sim_time: int, lock: bool=None) -> List[int]:
        """ computes and assigns new repositioning plans
        :param sim_time: current simulation time
        :param lock: bool if repositioning should be locked
        :return: list of vehicle ids with adopted schedules (new repositioning targets)"""
        self.sim_time = sim_time
        self.zone_system.time_trigger(sim_time)
        self._repo_zone_system.time_trigger(sim_time)
        if self._sampling_ctrl_function is None:
            self._sampling_ctrl_function = self.fleetctrl.vr_ctrl_f# update_ctrl_function(self.fleetctrl.vr_ctrl_f)
        if lock is None:
            lock = self.lock_repo_assignments
            
        t0_min = sim_time + self.list_horizons[0]
        t1_max = sim_time + self.list_horizons[1]
        
        _sampling_times = []
        
        sample_tour_id_subtour_id_parameters = []
        sample_start_bin_to_ozone_to_tour_ids = []
        sample_end_bin_to_dzone_to_tour_ids = []
        sample_timebin_to_zone_to_idle_vehs = []
        sample_tour_id_to_plan_id = []
        sample_bin_to_zone_to_idle_veh_objs = []
        
        use_veh_plans = self._remove_reservation_stops_for_sampling(sim_time)
        
        for _ in range(self.N_samples): # do for each sample
            tm = time.time()
            future_rq_atts = []
            for t0 in range(t0_min, t1_max, self._repo_time_step):
                # sampled future requests
                new_future_rq_atts = self.zone_system.draw_future_request_sample(t0, t0 + self._repo_time_step)
                time_bin_int = int( (t0 - sim_time + self.list_horizons[0])/self._repo_time_step )
                #priority = self._gamma ** time_bin_int  # priority of future requests
                new_future_rq_atts = [(*x, False) for x in new_future_rq_atts]
                future_rq_atts += new_future_rq_atts
                if self.fleetctrl.reservation_module is not None:
                    upcoming_reservation_requests = \
                        self.fleetctrl.reservation_module.get_upcoming_unassigned_reservation_requests(t0, t0 + self._repo_time_step, with_assigned=True)
                    if self._prioritize_reservations:
                        priority = True
                        future_rq_atts += [(*x, priority) for x in upcoming_reservation_requests]
                    else:
                        future_rq_atts += [(*x, False) for x in upcoming_reservation_requests]
                LOG.debug(f"Sampled Requests for bin {t0} - {t0 + self._repo_time_step} : odm: {len(new_future_rq_atts)} | res: {len(upcoming_reservation_requests)} | all till now: {len(future_rq_atts)}")

            # simulate future vehicle states and create input parameters for matching
            tour_id_subtour_id_parameters, start_bin_to_ozone_to_tour_ids, end_bin_to_dzone_to_tour_ids, timebin_to_zone_to_idle_vehs, tour_id_to_plan_id, bin_to_zone_to_idle_veh_objs = \
                self._progress_vehicles(future_rq_atts, sim_time, t1_max, use_veh_plans=use_veh_plans)
                
            LOG.debug(f"tour_id_subtour_id_parameters {tour_id_subtour_id_parameters}")
            LOG.debug(f"start_bin_to_ozone_to_tour_ids {start_bin_to_ozone_to_tour_ids}")
            LOG.debug(f"end_bin_to_dzone_to_tour_ids {end_bin_to_dzone_to_tour_ids}")
            LOG.debug(f"timebin_to_zone_to_idle_vehs {timebin_to_zone_to_idle_vehs}")
            LOG.debug(f"tour_id_to_plan_id {tour_id_to_plan_id}")
            sample_tour_id_subtour_id_parameters.append(tour_id_subtour_id_parameters)
            sample_start_bin_to_ozone_to_tour_ids.append(start_bin_to_ozone_to_tour_ids)
            sample_end_bin_to_dzone_to_tour_ids.append(end_bin_to_dzone_to_tour_ids)
            sample_timebin_to_zone_to_idle_vehs.append(timebin_to_zone_to_idle_vehs)
            sample_tour_id_to_plan_id.append(tour_id_to_plan_id)
            sample_bin_to_zone_to_idle_veh_objs.append(bin_to_zone_to_idle_veh_objs)
            LOG.debug(f"sampling took {time.time() - tm}")
            _sampling_times.append(time.time() - tm)
        
        tm = time.time()
        # get forecast of vehicle availabilities
        avail_for_rebal_vehicles = sample_bin_to_zone_to_idle_veh_objs[0][sim_time]      
        #_, _, avail_for_rebal_vehicles, _ = self._get_vehs_with_vp_available_for_repo(sim_time)
        
        # solve matching problem to reposition idle vehicles to zones
        od_number_rebal, plan_id_to_repo, plan_id_to_idle_bin = self._solve_sampling_repositioning(sim_time, avail_for_rebal_vehicles, sample_tour_id_subtour_id_parameters, 
                                                                   sample_start_bin_to_ozone_to_tour_ids, sample_end_bin_to_dzone_to_tour_ids,
                                                                   sample_timebin_to_zone_to_idle_vehs, sample_tour_id_to_plan_id)
        
        list_veh_with_changes = self._assign_repositioning_plans(sim_time, od_number_rebal,
                                                                 plan_id_to_repo, plan_id_to_idle_bin, sample_bin_to_zone_to_idle_veh_objs[0], lock=self.lock_repo_assignments)
        _solve_time = time.time() - tm
        LOG.info(f"Sampling took {sum(_sampling_times)} | Matching took {_solve_time} | Total {sum(_sampling_times) + _solve_time} | Single sample times: {_sampling_times}")
                    
        return list_veh_with_changes
    
    def _remove_reservation_stops_for_sampling(self, sim_time):
        """ removes reservation stops that are not assigned yet and are in the future
        :param sim_time: current simulation time"""
        new_veh_plans = {}
        for vid, vp in self.fleetctrl.veh_plans.items():
            new_vep = vp.copy()
            if len(new_vep.list_plan_stops) > 0 and new_vep.list_plan_stops[-1].is_locked_end():
                new_vep.list_plan_stops = new_vep.list_plan_stops[:-1]
                new_vep.update_tt_and_check_plan(self.fleetctrl.sim_vehicles[vid], sim_time, self.routing_engine, keep_feasible=True)
            new_veh_plans[vid] = new_vep
        return new_veh_plans
    
    def _get_vehs_with_vp_available_for_repo(self, sim_time : int, use_vehicles = None, use_vehplans = None) \
            -> Tuple[Dict[int, SimulationVehicleStruct], Dict[int, VehiclePlan], Dict[int, List[SimulationVehicleStruct]], Dict[int, VehiclePlan]]:
        """ 
        prepares input parameters for vehicles for sampling based repositioning in combination with reservations
        identifies idle vehicles and vehicles with currently assigned vehicle plans until the end of the forecast horizon
        :param sim_time: current simulation time
        :param use_vehicles: list of vehicle objects to be used (if None, fleetctrl vehicles are used)
        :param use_vehplans: dict of vehicle plan objects to be used (if None, fleetctrl vehicle plans are used)
        :return: 
            veh_objs: dict of vehicle id -> vehicle object with current state (copies for preprogression)
            veh_plans: dict of vehicle id -> vehicle plan object (copies for preprogression)
            zone_to_idle_vehs: dict of zone -> list of vehicle objects that are idle (within the forecast horizon)
            unassigned_reservation_plans: dict of plan id (in reservation module) -> full vehicle plan object for vehicles with reservation plans that are not assigned yet
        """
        n_idle = 0
        veh_objs: Dict[int, SimulationVehicleStruct] = {}
        veh_plans: Dict[int, VehiclePlan] = {}
        zone_to_idle_vehs = {}
        unassigned_reservation_plans = {}
        if use_vehicles is None:
            use_vehicles = self.fleetctrl.sim_vehicles
        if use_vehplans is None:
            use_vehplans = self.fleetctrl.veh_plans
        for veh_obj in use_vehicles:
            vp = use_vehplans.get(veh_obj.vid)
            if vp is not None and len(vp.list_plan_stops) > 0: # and not vp.list_plan_stops[-1].is_locked_end() and not vp.list_plan_stops[-1].is_locked():
                veh_objs[veh_obj.vid] = SimulationVehicleStruct(veh_obj, vp, sim_time, self.routing_engine)
                veh_plans[veh_obj.vid] = vp.copy()
            else:
                n_idle +=1
                zone = self._repo_zone_system.get_zone_from_pos(veh_obj.pos)
                try:
                    zone_to_idle_vehs[zone].append(veh_obj)
                except KeyError:
                    zone_to_idle_vehs[zone] = [veh_obj]
        LOG.debug(f" -> new idle: {n_idle} | new unassigned reservation plans {len(unassigned_reservation_plans)}")
        return veh_objs, veh_plans, zone_to_idle_vehs, unassigned_reservation_plans
    
    def _assign_repositioning_plans(self, sim_time, od_number_rebal, plan_id_to_repo, plan_id_to_idle_bin, bin_to_zone_to_idle_veh_objs, lock=False) -> List[int]:
        LOG.debug(f"assign new repo plans at time {sim_time}:")
        LOG.debug(f"od_number_rebal {od_number_rebal}")
        LOG.debug(f"plan_id_to_repo {plan_id_to_repo}")
        LOG.debug(f"plan_id_to_idle_bin {plan_id_to_idle_bin}")
        LOG.debug(f"bin_to_zone_to_idle_veh_objs {bin_to_zone_to_idle_veh_objs}")
        avail_for_rebal_vehicles = bin_to_zone_to_idle_veh_objs[sim_time]

        supporting_points = {}  # plan_id -> (pos, time, vid)
        if self.fleetctrl.reservation_module is not None:
            for vid in range(len(self.fleetctrl.sim_vehicles)):
                assigned_vid, plan_id, pos, time = self.fleetctrl.reservation_module.get_supporting_point(sim_time, vid=vid)
                if assigned_vid is not None:
                    supporting_points[plan_id] = (pos, time, vid)
        
        if len(supporting_points) != 0:
            import gurobipy as grp
            
            model_name = f"rp_rebal_solve_od_res_{sim_time}"
            with grp.Env(empty=True) as env:
                if self.fleetctrl.log_gurobi:
                    with open(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], "gurobi_log.log"), "a") as f:
                        f.write(f"\n\n{model_name}\n\n")
                    env.setParam('OutputFlag', 1)
                    env.setParam('LogToConsole', 0)
                    env.setParam('LogFile', os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], "gurobi_log.log") )
                    env.start()
                else:
                    env.setParam('OutputFlag', 0)
                    env.setParam('LogToConsole', 0)
                    env.start()

                m = grp.Model(model_name, env = env)

                m.setParam(grp.GRB.param.Threads, self.fleetctrl.n_cpu)
                m.setParam('TimeLimit', OPT_TIME_LIMIT)
                m.setParam("MIPGap", GUROBI_MIPGAP)
            
                vars = {} # var_name -> var   
                sup_vars = {}   # plan_id -> var_name
                od_repo_vars = {}   # (o, d) -> var_name
                veh_vars = {}   # vid -> var_name
                _repo_prio = 100000
                
                od_repo_dict = {}
                number_repos = 0
                for (o, d), number in od_number_rebal.items():
                    if od_repo_dict.get(o) is None:
                        od_repo_dict[o] = {}
                    try:
                        od_repo_dict[o][d] = number
                    except KeyError:
                        od_repo_dict[o] = {d: number}
                    number_repos += number
                        
                    avail_vehicles = avail_for_rebal_vehicles[o]
                    np.random.shuffle(avail_vehicles)
                    
                    for veh in avail_vehicles:
                        # only repo
                        var_name = f"veh_repo_{veh.vid}_{o}_{d}"
                        var = m.addVar(vtype=grp.GRB.BINARY, name=var_name, obj=_repo_prio)
                        vars[var_name] = var
                        try:
                            od_repo_vars[(o, d)].append(var_name)
                        except:
                            od_repo_vars[(o, d)] = [var_name]
                        try:
                            veh_vars[veh.vid].append(var_name)
                        except:
                            veh_vars[veh.vid] = [var_name]
                            
                        # repo + supporting point
                        repo_dest = self._repo_zone_system.get_random_centroid_node(d)
                        repo_dest = (repo_dest, None, None)
                        for plan_id, sup in supporting_points.items(): #TODO
                            arrival_time = sim_time + self.routing_engine.return_travel_costs_1to1(veh.pos, repo_dest)[1] + \
                                    self.routing_engine.return_travel_costs_1to1(repo_dest, sup[0])[1]
                            start_time = sup[1]
                            if arrival_time <= start_time:
                                var_name = f"veh_repo_sup_{veh.vid}_{o}_{d}_{plan_id}"
                                var = m.addVar(vtype=grp.GRB.BINARY, name=var_name, obj=_repo_prio + start_time - arrival_time)
                                vars[var_name] = var
                                try:
                                    od_repo_vars[(o, d)].append(var_name)
                                except:
                                    od_repo_vars[(o, d)] = [var_name]
                                try:
                                    veh_vars[veh.vid].append(var_name)
                                except:
                                    veh_vars[veh.vid] = [var_name]
                                try:
                                    sup_vars[plan_id].append(var_name)
                                except:
                                    sup_vars[plan_id] = [var_name]
                # only sup points
                for vid, veh_plan in self.fleetctrl.veh_plans.items():
                    if len(veh_plan.list_plan_stops) > 0:
                        if veh_plan.list_plan_stops[-1].is_locked_end():
                            if len(veh_plan.list_plan_stops) > 1:
                                last_ps = veh_plan.list_plan_stops[-2]
                                end_pos = last_ps.get_pos()
                                end_time = last_ps.get_planned_arrival_and_departure_time()[1]
                            else:
                                end_pos = self.fleetctrl.sim_vehicles[vid].pos
                                end_time = sim_time
                        else:
                            last_ps = veh_plan.list_plan_stops[-1]
                            end_pos = last_ps.get_pos()
                            end_time = last_ps.get_planned_arrival_and_departure_time()[1]
                    else:
                        end_pos = self.fleetctrl.sim_vehicles[vid].pos
                        end_time = sim_time
                    for plan_id, sup in supporting_points.items():
                        arrival_time = end_time + self.routing_engine.return_travel_costs_1to1(end_pos, sup[0])[1]
                        start_time = sup[1]
                        if arrival_time <= start_time or self.fleetctrl.reservation_module.get_supporting_point(sim_time, plan_id=plan_id)[0] == vid:
                            var_name = f"veh_sup_{vid}_{plan_id}"
                            var = m.addVar(vtype=grp.GRB.BINARY, name=var_name, obj= (start_time - arrival_time)/2)
                            vars[var_name] = var
                            try:
                                veh_vars[vid].append(var_name)
                            except:
                                veh_vars[vid] = [var_name]
                            try:
                                sup_vars[plan_id].append(var_name)
                            except:
                                sup_vars[plan_id] = [var_name]
                                
                # constaint 1) vehicles can only be assigned to one repo
                for vid, var_names in veh_vars.items():
                    m.addConstr(grp.quicksum(vars[var_name] for var_name in var_names) <= 1, name=f"veh_{vid}")
                # constraint 2) each od pair can only be assigned to one vehicle
                for (o, d), var_names in od_repo_vars.items():
                    n_repos = od_repo_dict[o][d]
                    m.addConstr(grp.quicksum(vars[var_name] for var_name in var_names) <= n_repos, name=f"od_{o}_{d}")
                # constraint 3) each supporting point must be assigned to one vehicle
                for plan_id, var_names in sup_vars.items():
                    m.addConstr(grp.quicksum(vars[var_name] for var_name in var_names) == 1, name=f"sup_{plan_id}")
                    
                # set objective and optimize
                m.modelSense = grp.GRB.MAXIMIZE
                m.update()
                if WRITE_PROBLEM:
                    m.write(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], f"opt_problem_repo_reservation_{sim_time}.lp"))
                    
                m.optimize()

                # retrieve solution
                vals = m.getAttr('X', vars)
                new_vid_to_plan_id = {}
                list_veh_with_changes = []
                number_assigned_repos = 0
                for x in vals:
                    v = vals[x]
                    #LOG.debug(f"{v}, {x}")
                    v = int(np.round(v))
                    if v == 0:
                        continue
                    p = x.split("_")
                    if "repo" in p and "sup" in p:
                        vid, o, d, plan_id = int(p[-4]), int(p[-3]), int(p[-2]), int(p[-1])
                        LOG.debug(f" -> repo and sup {vid} {o} {d} {plan_id}")
                        veh_obj = self.fleetctrl.sim_vehicles[vid]
                        if o != d:
                            list_veh_with_changes += self._od_to_veh_plan_assignment(sim_time, o, d, [veh_obj], lock=lock)
                        new_vid_to_plan_id[vid] = plan_id
                        number_assigned_repos += 1
                    elif "repo" in p:
                        vid, o, d = int(p[-3]), int(p[-2]), int(p[-1])
                        LOG.debug(f" -> repo {vid} {o} {d}")
                        veh_obj = self.fleetctrl.sim_vehicles[vid]
                        if o != d:
                            list_veh_with_changes += self._od_to_veh_plan_assignment(sim_time, o, d, [veh_obj], lock=lock)
                        number_assigned_repos += 1
                    elif "sup" in p:
                        LOG.debug(f" -> sup {p}")
                        vid, plan_id = int(p[-2]), int(p[-1])
                        new_vid_to_plan_id[vid] = plan_id
                        
                self.fleetctrl.reservation_module.reassign_supporting_points(sim_time, new_vid_to_plan_id)
                LOG.debug(f"after repo in {sim_time} assigned {number_assigned_repos} / {number_repos} repos")
        else:
            od_repo_dict = {}
            for (o, d), number in od_number_rebal.items():
                if od_repo_dict.get(o) is None:
                    od_repo_dict[o] = {}
                try:
                    od_repo_dict[o][d] = number
                except KeyError:
                    od_repo_dict[o] = {d: number}
            # assign repositioning plans
            list_veh_with_changes = []
            for o, d_dict in od_repo_dict.items():
                avail_vehicles = avail_for_rebal_vehicles[o]
                shuffle_list = list(d_dict.items())
                np.random.shuffle(shuffle_list)
                for d, number in shuffle_list:
                    LOG.info(f"rebal {o} {d} : {number}")
                    vehs_with_priority = []
                    for veh in avail_vehicles:
                        vp = self.fleetctrl.veh_plans.get(veh.vid)
                        if len(vp.list_plan_stops) == 0 or not vp.list_plan_stops[-1].is_locked_end():
                            vehs_with_priority.append( (veh, LARGE_VALUE) )
                        else:
                            centroid_pos = (self.zone_system.get_random_centroid_node(d), None, None)
                            buffer_time = vp.list_plan_stops[-1].get_earliest_start_time() - sim_time - \
                                self.routing_engine.return_travel_costs_1to1(veh.pos, centroid_pos)[0] - \
                                self.routing_engine.return_travel_costs_1to1(centroid_pos, vp.list_plan_stops[-1].get_pos())[0]
                            if buffer_time > 0:
                                vehs_with_priority.append( (veh, buffer_time) )
                            else:
                                LOG.warning(f"veh {veh.vid} has no buffer time {buffer_time} for repo {o} -> {d}")
                    vehs_with_priority.sort(key=lambda x:-x[1])
                    for i in range(min(number, len(vehs_with_priority))):
                        veh_obj, prio = vehs_with_priority[i]
                        list_veh_obj_with_repos = self._od_to_veh_plan_assignment(sim_time, o, d, [veh_obj], lock=lock)
                        list_veh_with_changes.extend([veh_obj.vid for veh_obj in list_veh_obj_with_repos])
                        for veh_obj in list_veh_obj_with_repos:
                            LOG.info(f" -> assign {veh_obj.vid} with priority {prio}")
                            avail_vehicles.remove(veh_obj)

        return list_veh_with_changes
    
    def _progress_vehicles(self, future_rq_atts, sim_time, prog_end_time, use_veh_plans=None):
        """ in this method future vehicle states are simulated and input parameters for rebalancing optimization are created
        in each time step
        1) new requests are inserted
            if no vehicle is close by, a new vehicle is created (their traces of the period are stored and used as possible repositioning targets)
        2) vehicles are moved according to their plans
        3) after a repo time bin, new idle vehicles are tracked
        
        at the end, input parameters for the optimization are created and returned
        
        :param future_rq_atts: list of future requests
        :param sim_time: current simulation time
        :param prog_end_time: end time of the simulation
        :return: input parameters for the optimization
            tour_id_subtour_id_parameters: tour_id -> subtour_id -> (o_zone, cost_gain, arr_at_zone, end_zone, start_bin, end_bin)
            start_bin_to_ozone_to_tour_ids: start_bin -> o_zone -> list of (tour_id, subtour_id)
            end_bin_to_dzone_to_tour_ids: end_bin -> d_zone -> list of (tour_id, subtour_id)
            timebin_to_zone_to_idle_vehs: time bin -> zone -> number of idle vehicles
            tour_id_to_plan_id: tour_id -> plan_id # this tour also assigns a reservation plan with plan_id in the reservation module
            bin_to_zone_to_idle_veh_objs: time bin -> zone -> list of idle vehicle objects
        """
        # only currently driving vehicles (idle vehicles will be created)
           
        veh_objs, veh_plans, idle_veh_objs, unassigned_reservation_plans = self._get_vehs_with_vp_available_for_repo(sim_time, use_vehplans=use_veh_plans)
        
        bin_to_zone_to_idle_veh_objs = {}
        bin_to_zone_to_idle_veh_objs[sim_time] = idle_veh_objs
        
        opt_time_step = self._progress_time_step
        
        insert_time_to_reservation_plan_id = {}
        for plan_id, full_plan in unassigned_reservation_plans.items():
            insert_time = max(full_plan.list_plan_stops[0].get_earliest_start_time() - self._repo_time_step, sim_time)
            insert_time = int(np.floor(insert_time/opt_time_step)) * opt_time_step
            if insert_time > prog_end_time - opt_time_step: # insert plans that are outside of the horizon at the end
                insert_time = prog_end_time - opt_time_step
            try:
                insert_time_to_reservation_plan_id[insert_time].append(plan_id)
            except KeyError:
                insert_time_to_reservation_plan_id[insert_time] = [plan_id]

        rq_dict = self.fleetctrl.rq_dict.copy()
        
        current_new_vid = len(self.fleetctrl.sim_vehicles) + 1
        new_vids_to_ps = {}
        
        vids_with_assignment = {}
        
        pos_to_vids = {}
        plan_id_to_assigned_vid = {}
        
        if len(rq_dict) > 0:
            start_id = max(rq_dict.keys()) + 1
        else:
            start_id = 0

        prqs = {}
        rid_order = {}
        for atts in future_rq_atts:
            if len(atts) == 3:
                t, o_n, d_n = atts
                priority = None
            else:
                t, o_n, d_n, priority = atts
            rid = start_id + len(prqs)
            opt_time = int(np.floor(t/opt_time_step)) * opt_time_step
            try:
                rid_order[opt_time].append(rid)
            except:
                rid_order[opt_time] = [rid]
            prqs[rid] = ArtificialPlanRequest(rid, t, (o_n, None, None), (d_n, None, None), self.routing_engine,
                                                max_wait_time=self.fleetctrl.max_wait_time, max_detour_time_factor=self.fleetctrl.max_dtf,
                                                max_constant_detour_time=self.fleetctrl.max_cdt, boarding_time=self.fleetctrl.const_bt)
            if priority:
                prqs[rid].set_reservation_flag(True)
        
        timebin_to_zone_to_idle_vehs = {} # time bin -> zone -> number of idle vehicles
        
        vid_to_route = {}    # caching routes
        for t in range(sim_time, prog_end_time, opt_time_step):
            # update travel times
            if self.fleetctrl._use_own_routing_engine:
                self.routing_engine.update_network(t)
            # track new idle vehicles at start of time bin
            if t != sim_time and t%self._repo_time_step == 0:
                # update currently idle vehicles
                timebin_to_zone_to_idle_vehs[t] = {}
                LOG.debug(f"update idle vehicles at {t} : large prog end time? {t >= prog_end_time}")
                use_vehicles = [veh for veh in veh_objs.values() if veh.vid  < len(self.fleetctrl.sim_vehicles)]
                veh_objs_new, veh_plans_new, zone_to_idle_vehs, unassigned_reservation_plans_new = \
                    self._get_vehs_with_vp_available_for_repo(t, use_vehicles=use_vehicles, use_vehplans=veh_plans)
                bin_to_zone_to_idle_veh_objs[t] = zone_to_idle_vehs
                for vid, veh_obj_new in veh_objs_new.items():
                    veh_objs[vid] = veh_obj_new
                    veh_plans[vid] = veh_plans_new[vid]
                    LOG.debug(f"update idle veh {veh_obj_new} at {t} -> {veh_plans[vid]}")
                for zone, vehs in zone_to_idle_vehs.items():
                    try:
                        timebin_to_zone_to_idle_vehs[t][zone] += len(vehs)
                    except KeyError:
                        timebin_to_zone_to_idle_vehs[t][zone] = len(vehs)
                    for veh in vehs:
                        del veh_objs[veh.vid]
                        del veh_plans[veh.vid]
                    
                for plan_id, full_plan in unassigned_reservation_plans_new.items():
                    insert_time = max(full_plan.list_plan_stops[0].get_earliest_start_time() - self._repo_time_step, sim_time)
                    insert_time = int(np.floor(insert_time/opt_time_step)) * opt_time_step
                    if insert_time > prog_end_time - opt_time_step: # insert plans that are outside of the horizon at the end
                        insert_time = prog_end_time - opt_time_step
                    try:
                        insert_time_to_reservation_plan_id[insert_time].append(plan_id)
                    except KeyError:
                        insert_time_to_reservation_plan_id[insert_time] = [plan_id]
                unassigned_reservation_plans.update(unassigned_reservation_plans_new)
                LOG.debug(f"new unassigned reservation plans {unassigned_reservation_plans_new}")
            
            # insert offline plans
            new_offline_plans = insert_time_to_reservation_plan_id.get(t)
            if new_offline_plans is not None:
                LOG.debug(f"assign offline plans {new_offline_plans} at {t}")
                for plan_id in new_offline_plans:
                    full_plan = unassigned_reservation_plans[plan_id]
                    full_plan_start_time = full_plan.list_plan_stops[0].get_earliest_start_time()
                    full_plan_start_pos = full_plan.list_plan_stops[0].get_pos()
                    for rid in full_plan.get_involved_request_ids():
                        rq_dict[rid].compute_new_max_trip_time(self.routing_engine, boarding_time=self.fleetctrl.const_bt, max_detour_time_factor=self.fleetctrl.max_dtf,
                                                max_constant_detour_time=self.fleetctrl.max_cdt)
                    best_vid = None
                    best_cfv = float("inf")
                    best_plan = None
                    for vid, veh in veh_objs.items():
                        if vid < len(self.fleetctrl.sim_vehicles):
                            continue
                        current_vp = veh_plans[vid]
                        if len(current_vp.list_plan_stops) != 0:
                            last_ps = current_vp.list_plan_stops[-1]
                            if last_ps.is_locked_end():
                                continue
                            last_ps_end_time = last_ps.get_planned_arrival_and_departure_time()[1]
                            last_ps_end_pos = last_ps.get_pos()
                        else:
                            last_ps_end_time = t
                            last_ps_end_pos = veh.pos
                        if last_ps_end_time > t:
                            continue
                        _, tt, dis = self.routing_engine.return_travel_costs_1to1(last_ps_end_pos, full_plan_start_pos)
                        if tt + last_ps_end_time <= full_plan_start_time:
                            new_vp = current_vp.copy()
                            new_vp.list_plan_stops += [ps.copy() for ps in full_plan.list_plan_stops]
                            feasible = new_vp.update_tt_and_check_plan(veh, t, self.routing_engine)
                            if feasible:
                                cfv = self._sampling_ctrl_function(t, veh, new_vp, rq_dict, self.routing_engine)
                                if cfv < best_cfv:
                                    best_vid = vid
                                    best_cfv = cfv
                                    best_plan = new_vp
                    if best_vid is None:
                        o_zone = self._repo_zone_system.get_zone_from_pos(full_plan_start_pos)
                        centroid = self._repo_zone_system.get_random_centroid_node(o_zone)
                        any_vehicle = self.fleetctrl.sim_vehicles[0]
                        new_veh = SimulationVehicleStruct(any_vehicle, full_plan, t, self.routing_engine, empty_init=True)
                        new_veh.pos = (centroid, None, None)
                        new_veh.vid = current_new_vid
                        new_veh.status = VRL_STATES.IDLE
                        best_vid = current_new_vid
                        best_plan = full_plan.copy()
                        best_plan.update_tt_and_check_plan(new_veh, t, self.routing_engine, keep_feasible=True)
                        veh_objs[best_vid] = new_veh
                        veh_plans[best_vid] = best_plan
                        new_vids_to_ps[best_vid] = []
                        current_new_vid += 1
                    else:
                        vids_with_assignment[best_vid] = 1
                        veh_plans[best_vid] = best_plan
                         
                    LOG.debug(f"reservation plan {plan_id} assigned to {best_vid} at {t} -> {best_plan}")
                    plan_id_to_assigned_vid[plan_id] = best_vid
                        
            # insert new requests
            new_rids = rid_order.get(t)
            if new_rids is not None:
                pos_to_vids = {} # position -> list of vehicle ids
                for veh in veh_objs.values():
                    pos = veh.pos
                    try:
                        pos_to_vids[pos].append(veh.vid)
                    except KeyError:
                        pos_to_vids[pos] = [veh.vid]
                LOG.debug(f"current vids : { [(vid, veh.vid) for vid, veh in veh_objs.items()]}")
                for rid in new_rids: 
                    #LOG.debug(f"insert {rid}")       
                    prq : ArtificialPlanRequest = prqs[rid]
                    rq_dict[rid] = prq
                    #LOG.debug(f"check timing: {prq.get_o_stop_info()[2] - t}")
                    
                    # check for close by vehicles
                    r = self.routing_engine.return_travel_costs_Xto1(pos_to_vids.keys(), prq.get_o_stop_info()[0], 
                                                                        max_cost_value= self.fleetctrl.max_wait_time)
                    rid_vehs = []
                    rid_veh_plans = {}
                    for o_pos, _ ,_,_ in r:
                        vids = pos_to_vids[o_pos] 
                        for vid in vids:
                            rid_vehs.append(veh_objs[vid])  
                            rid_veh_plans[vid] = veh_plans[vid] 
                            
                    # add possible new idle vehicle
                    o_zone = self._repo_zone_system.get_zone_from_pos(prq.get_o_stop_info()[0])
                    centroid = self._repo_zone_system.get_random_centroid_node(o_zone)
                    any_vehicle = self.fleetctrl.sim_vehicles[0]
                    new_veh = SimulationVehicleStruct(any_vehicle, VehiclePlan(any_vehicle, t, self.routing_engine, []), t, self.routing_engine, empty_init=True)
                    new_veh.pos = (centroid, None, None)
                    new_veh.vid = current_new_vid
                    new_veh.status = VRL_STATES.IDLE
                    
                    rid_vehs.append(new_veh)
                    rid_veh_plans[current_new_vid] = VehiclePlan(new_veh, t, self.routing_engine, [])
                                
                    insert_sols = insert_prq_in_selected_veh_list(rid_vehs, rid_veh_plans, prq,
                                                                    self._sampling_ctrl_function, self.routing_engine, rq_dict,
                                                                    t, self.fleetctrl.const_bt, self.fleetctrl.add_bt)
                    if len(insert_sols) == 0:
                        LOG.warning(f"no sampling solution found? {prq}")
                        continue
                    best_sol = min(insert_sols, key=lambda x:x[2])
                    best_vid, best_plan, _ = best_sol
                    if best_vid == current_new_vid:
                        if len(insert_sols) > 1:
                            LOG.debug(f"new vid would be best, but other option")
                            best_vid, best_plan, _ = sorted(insert_sols, key = lambda x:x[2])[1]
                    veh_plans[best_vid] = best_plan
                    LOG.debug(f"fc prq {prq} assigned to {best_vid} which is new? {best_vid == current_new_vid}")
                    if best_vid == current_new_vid: # new vid is introduced
                        veh_objs[new_veh.vid] = new_veh
                        pos = new_veh.pos
                        try:
                            pos_to_vids[pos].append(new_veh.vid)
                        except KeyError:
                            pos_to_vids[pos] = [new_veh.vid]
                            
                        new_vids_to_ps[new_veh.vid] = []
                        current_new_vid += 1
                    else: # rid is accomodated by onroute vehicle
                        vids_with_assignment[vid] = 1
                        
            # move vehicles
            for vid, veh in veh_objs.items():
                passed_ps, vid_to_route = move_vehicle_according_to_plan(veh, veh_plans[vid], t, opt_time_step, self.routing_engine, rq_dict, vid_to_route, self.fleetctrl.begin_approach_buffer_time)
                if len(passed_ps) > 0:
                    if new_vids_to_ps.get(veh.vid) is not None:
                        for ps in passed_ps:
                            new_vids_to_ps[veh.vid].append(ps)

        if self.fleetctrl._use_own_routing_engine:
            self.routing_engine.reset_network(sim_time)

        # create optimization input parameters
        #tour_id = 0
        tour_id_subtour_id_parameters = {} # tour_id -> subtour_id -> (o_zone, cost_gain, arr_at_zone, end_zone, start_bin, end_bin)
        start_bin_to_ozone_to_tour_ids = {} # start_bin -> o_zone -> list of (tour_id, subtour_id)
        end_bin_to_dzone_to_tour_ids = {} # end_bin -> d_zone -> list of (tour_id, subtour_id)
        tour_id_to_plan_id = {val : key for key, val in plan_id_to_assigned_vid.items()}
        LOG.debug(f"tour ids to be assigned: {tour_id_to_plan_id}")
        for vid, list_ps in new_vids_to_ps.items():
            tour_id = vid
            is_reservation_tour = False
            if tour_id_to_plan_id.get(vid) is not None:
                is_reservation_tour = True
            tour_id_subtour_id_parameters[tour_id] = {}
            subtour_id = 0
            
            vp: VehiclePlan = veh_plans.get(vid, VehiclePlan(self.fleetctrl.sim_vehicles[0], sim_time, self.routing_engine, []))
            LOG.debug(f"create input params for {vid}: has reservations? {is_reservation_tour}")
            start_time = sim_time
            if len(list_ps) > 0:
                o_z = self._repo_zone_system.get_zone_from_pos(list_ps[0].get_pos())
            else:
                o_z = self._repo_zone_system.get_zone_from_pos(vp.list_plan_stops[0].get_pos())
                
            centroid = self._repo_zone_system.get_random_centroid_node(o_z)
            any_vehicle = next(iter(veh_objs.values()))
            new_veh = SimulationVehicleStruct(any_vehicle, VehiclePlan(any_vehicle, sim_time, self.routing_engine, []), sim_time, self.routing_engine, empty_init=True)
            new_veh.pos = (centroid, None, None)
            new_veh.vid = -1
            new_veh.status = VRL_STATES.IDLE

            full_vp = VehiclePlan(new_veh, start_time, self.routing_engine, list_ps + vp.list_plan_stops )
            full_vp.update_tt_and_check_plan(new_veh, start_time, self.routing_engine, keep_feasible=True)
            LOG.debug(f" -- full vp {full_vp}")
            #cost_gain = self.fleetctrl.compute_VehiclePlan_utility(start_time, new_veh, full_vp)
            cost_gain = self._sampling_ctrl_function(start_time, new_veh, full_vp, rq_dict, self.routing_engine)
            arr_time = max(full_vp.list_plan_stops[0].get_planned_arrival_and_departure_time()[0], full_vp.list_plan_stops[0].get_earliest_start_time())
            arr_at_zone = arr_time - self.routing_engine.return_travel_costs_1to1(new_veh.pos, full_vp.list_plan_stops[0].get_pos())[0]
            
            end_time = full_vp.list_plan_stops[-1].get_planned_arrival_and_departure_time()[1]
            start_pos = full_vp.list_plan_stops[0].get_pos()
            end_pos = full_vp.list_plan_stops[-1].get_pos()
            end_zone = self._repo_zone_system.get_zone_from_pos(end_pos)
            
            LOG.debug(f" -- add base tour {(tour_id, subtour_id)} {(o_z, cost_gain, arr_at_zone)}")
            
            start_bin = int(arr_at_zone / self._repo_time_step) * self._repo_time_step
            end_bin = int(end_time / self._repo_time_step) * self._repo_time_step
            LOG.debug(f" -> end time and bin: {end_time} {end_bin}")
            
            tour_id_subtour_id_parameters[tour_id][subtour_id] = (o_z, cost_gain, arr_at_zone, end_zone, start_bin, end_bin)
            
            try:
                start_bin_to_ozone_to_tour_ids[start_bin][o_z].append( (tour_id, subtour_id) )
            except KeyError:
                try:
                    start_bin_to_ozone_to_tour_ids[start_bin][o_z] = [(tour_id, subtour_id)]
                except KeyError:
                    start_bin_to_ozone_to_tour_ids[start_bin] = {o_z : [(tour_id, subtour_id)]}
            try:
                end_bin_to_dzone_to_tour_ids[end_bin][end_zone].append( (tour_id, subtour_id) )
            except KeyError:
                try:
                    end_bin_to_dzone_to_tour_ids[end_bin][end_zone] = [(tour_id, subtour_id)]
                except KeyError:
                    end_bin_to_dzone_to_tour_ids[end_bin] = {end_zone : [(tour_id, subtour_id)]}
                
            subtour_id += 1
            
            def add_subtour(full_vp: VehiclePlan, subtour_id):
                o_z = self._repo_zone_system.get_zone_from_pos(full_vp.list_plan_stops[0].get_pos())
                
                centroid = self._repo_zone_system.get_random_centroid_node(o_z)
                any_vehicle = next(iter(veh_objs.values()))
                new_veh = SimulationVehicleStruct(any_vehicle, VehiclePlan(any_vehicle, sim_time, self.routing_engine, []), sim_time, self.routing_engine, empty_init=True)
                new_veh.pos = (centroid, None, None)
                new_veh.vid = -1
                new_veh.status = VRL_STATES.IDLE

                full_vp = VehiclePlan(new_veh, start_time, self.routing_engine, full_vp.list_plan_stops )
                LOG.debug(f" -- check subtour {full_vp}")
                #cost_gain = self.fleetctrl.compute_VehiclePlan_utility(start_time, new_veh, full_vp)
                cost_gain = self._sampling_ctrl_function(start_time, new_veh, full_vp, rq_dict, self.routing_engine)
                arr_time = max(full_vp.list_plan_stops[0].get_planned_arrival_and_departure_time()[0], full_vp.list_plan_stops[0].get_earliest_start_time())
                arr_at_zone = arr_time - self.routing_engine.return_travel_costs_1to1(new_veh.pos, full_vp.list_plan_stops[0].get_pos())[0]
                LOG.debug(f" -- add subtour {(tour_id, subtour_id)} {(o_z, cost_gain, arr_at_zone)}")
                
                start_bin = int(arr_at_zone / self._repo_time_step) * self._repo_time_step
                end_bin = int(end_time / self._repo_time_step) * self._repo_time_step
                
                tour_id_subtour_id_parameters[tour_id][subtour_id] = (o_z, cost_gain, arr_at_zone, end_zone, start_bin, end_bin)
                
                try:
                    start_bin_to_ozone_to_tour_ids[start_bin][o_z].append( (tour_id, subtour_id) )
                except KeyError:
                    try:
                        start_bin_to_ozone_to_tour_ids[start_bin][o_z] = [(tour_id, subtour_id)]
                    except KeyError:
                        start_bin_to_ozone_to_tour_ids[start_bin] = {o_z : [(tour_id, subtour_id)]}
                try:
                    end_bin_to_dzone_to_tour_ids[end_bin][end_zone].append( (tour_id, subtour_id) )
                except KeyError:
                    try:
                        end_bin_to_dzone_to_tour_ids[end_bin][end_zone] = [(tour_id, subtour_id)]
                    except KeyError:
                        end_bin_to_dzone_to_tour_ids[end_bin] = {end_zone : [(tour_id, subtour_id)]}
                
            
            # create also parameters for subplans
            if len(full_vp.get_involved_request_ids()) > 0:
                #to_remove = full_vp.list_plan_stops[0].get_list_boarding_rids()
                while True:
                    arrived_at_reservation = False
                    for i, ps in enumerate(full_vp.list_plan_stops):
                        found = False
                        for rid in ps.get_list_boarding_rids():
                            if rq_dict[rid].get_reservation_flag():
                                arrived_at_reservation = True
                                LOG.debug(f" -- arrived at reservation {rid}")
                                break
                        if arrived_at_reservation:
                            break
                        for rid in ps.get_list_boarding_rids():
                            full_vp = simple_remove(new_veh, full_vp, rid, sim_time, self.routing_engine, self._sampling_ctrl_function, rq_dict, self.fleetctrl.const_bt, self.fleetctrl.add_bt)
                            found = True
                        if found:
                            break
                    if len(full_vp.list_plan_stops) == 0 or len(full_vp.get_involved_request_ids()) == 0 or arrived_at_reservation:
                        if len(full_vp.get_involved_request_ids()) == 0 and len(full_vp.list_plan_stops) == 1 and full_vp.list_plan_stops[0].is_locked_end():
                            add_subtour(full_vp, subtour_id)
                            subtour_id += 1
                        break
                    add_subtour(full_vp, subtour_id)
                    subtour_id += 1
        
        #exit()           
        return tour_id_subtour_id_parameters, start_bin_to_ozone_to_tour_ids, end_bin_to_dzone_to_tour_ids, timebin_to_zone_to_idle_vehs, tour_id_to_plan_id, bin_to_zone_to_idle_veh_objs
    
    def _solve_sampling_repositioning(self, sim_time: int, avail_for_rebal_vehicles: Dict[int, List[int]], sample_tour_id_subtour_id_parameters, 
                                                                   sample_start_bin_to_ozone_to_tour_ids, sample_end_bin_to_dzone_to_tour_ids,
                                                                   sample_timebin_to_zone_to_add_idle_vehs, sample_tour_id_to_plan_id):
        """ solves the repositioning problem with the given input data 
        :param sim_time: current simulation time
        :param avail_for_rebal_vehicles: zone -> list of vehicle ids
        :param sample_tour_id_subtour_id_parameters: list of tour_id -> subtour_id -> (o_zone, cost_gain, arr_at_zone, end_zone, start_bin, end_bin) per sample
        :param sample_start_bin_to_ozone_to_tour_ids: list of start_bin -> o_zone -> list of (tour_id, subtour_id) per sample
        :param sample_end_bin_to_dzone_to_tour_ids: list of end_bin -> d_zone -> list of (tour_id, subtour_id) per sample
        :param sample_timebin_to_zone_to_add_idle_vehs: list of time bin -> zone -> number of idle vehicles per sample
        :param sample_tour_id_to_plan_id: list of tour_id -> plan_id per sample (reservation plan that has to be assigned)
        :param lock: bool if repositioning should be locked
        :return: dict (o_zone, d_zone) -> number of rebalancing vehicles (new repositioning targets), 
                dict plan_id -> (o_zone, d_zone) (this rebal trips has to be combined with reservation plan plan_id,
                dict plan_id -> (o_zone, time_bin) a vehicle becoming idle in this time bin has to be combined with reservation plan plan_id"""
        
        import gurobipy as grp
        
        model_name = f"rp_rebal_solve_matching_{sim_time}"
        with grp.Env(empty=True) as env:
            if self.fleetctrl.log_gurobi:
                with open(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], "gurobi_log.log"), "a") as f:
                    f.write(f"\n\n{model_name}\n\n")
                env.setParam('OutputFlag', 1)
                env.setParam('LogToConsole', 0)
                env.setParam('LogFile', os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], "gurobi_log.log") )
                env.start()
            else:
                env.setParam('OutputFlag', 0)
                env.setParam('LogToConsole', 0)
                env.start()

            m = grp.Model(model_name, env = env)

            m.setParam(grp.GRB.param.Threads, self.fleetctrl.n_cpu)
            m.setParam('TimeLimit', OPT_TIME_LIMIT)
            m.setParam("MIPGap", GUROBI_MIPGAP)
        
            vars = {}
            current_T_od_vars = {} # time bin -> (o, d) -> name
            future_T_od_vars = {}  # sample -> time_bin -> (o,d) -> name
            current_T_from_o_vars = {}
            current_T_to_d_vars = {}
            future_T_from_o_vars = {}
            future_T_to_d_vars = {}
            T_od_to_sample_to_tours_vars = {}
            sample_to_tours_to_subtours_vars = {}
            T_to_sample_to_ozone_subtour_vars = {}
            T_to_sample_to_dzone_subtour_vars = {}
            
            # create variables
            for sample, tour_id_subtour_id_parameters in enumerate(sample_tour_id_subtour_id_parameters):
                future_T_od_vars[sample] = {}
                future_T_from_o_vars[sample] = {}
                future_T_to_d_vars[sample] = {}
                for tour_id, subtour_id_parameters in tour_id_subtour_id_parameters.items():
                    for subtour_id, (o_z, cost_gain, arr_at_zone, end_zone, start_bin, end_bin) in subtour_id_parameters.items():
                        if start_bin > sim_time + self.list_horizons[1]:
                            add_bin = max(start_bin - 3600, sim_time + self.list_horizons[1])
                            time_bin_iterator = [tb for tb in range(sim_time + self.list_horizons[0], sim_time + self.list_horizons[1], self._repo_time_step)] + [add_bin]
                        else:
                            time_bin_iterator = range(sim_time + self.list_horizons[0], sim_time + self.list_horizons[1], self._repo_time_step)
                        for time_bin in time_bin_iterator:
                            time_bin_int = int( (time_bin - sim_time + self.list_horizons[0])/self._repo_time_step )
                            factor = self._gamma ** time_bin_int
                            if time_bin == sim_time + self.list_horizons[0]:
                                iterator = avail_for_rebal_vehicles.keys()
                            else:
                                iterator = self._repo_zone_system.get_all_zones()
                            for zone in iterator:
                                tt = self._get_od_zone_travel_info(sim_time, zone, o_z)[0]
                                if time_bin + tt <= arr_at_zone:
                                    # od variable
                                    if time_bin_int == 0:
                                        od_name = f"od_cur_{time_bin}_{zone}_{o_z}"
                                        if vars.get(od_name) is None:
                                            var = m.addVar(name=od_name, obj=tt, vtype=grp.GRB.INTEGER)
                                            vars[od_name] = var
                                            try:
                                                current_T_od_vars[time_bin][(zone, o_z)] = od_name
                                            except KeyError:
                                                current_T_od_vars[time_bin] = {(zone, o_z): od_name}
                                            try:
                                                current_T_from_o_vars[time_bin][zone].append(od_name)
                                            except KeyError:
                                                try:
                                                    current_T_from_o_vars[time_bin][zone] = [od_name]
                                                except KeyError:
                                                    current_T_from_o_vars[time_bin] = {zone: [od_name]}
                                            try:
                                                current_T_to_d_vars[time_bin][o_z].append(od_name)
                                            except KeyError:
                                                try:
                                                    current_T_to_d_vars[time_bin][o_z] = [od_name]
                                                except KeyError:
                                                    current_T_to_d_vars[time_bin] = {o_z: [od_name]}
                                    else:
                                        # future rebalancing trips per sample
                                        od_name = f"od_future_{sample}_{time_bin}_{zone}_{o_z}"
                                        if vars.get(od_name) is None:
                                            var = m.addVar(name=od_name, obj=tt * factor / self.N_samples, vtype=grp.GRB.INTEGER)
                                            vars[od_name] = var
                                            try:
                                                future_T_od_vars[sample][time_bin][(zone, o_z)] = od_name
                                            except KeyError:
                                                future_T_od_vars[sample][time_bin] = {(zone, o_z): od_name}
                                            try:
                                                future_T_from_o_vars[sample][time_bin][zone].append(od_name)
                                            except KeyError:
                                                try:
                                                    future_T_from_o_vars[sample][time_bin][zone] = [od_name]
                                                except KeyError:
                                                    future_T_from_o_vars[sample][time_bin] = {zone: [od_name]}
                                            try:
                                                future_T_to_d_vars[sample][time_bin][o_z].append(od_name)
                                            except KeyError:
                                                try:
                                                    future_T_to_d_vars[sample][time_bin][o_z] = [od_name]
                                                except KeyError:
                                                    future_T_to_d_vars[sample][time_bin] = {o_z: [od_name]}
                                    
                                    # tour variable        
                                    name = f"t_z_{zone}_{time_bin}_s_{sample}_ti_{tour_id}_sti_{subtour_id}"
                                    var = m.addVar(name=name, obj=cost_gain * factor  / self.N_samples , vtype=grp.GRB.BINARY)
                                    vars[name] = var
                                    try:
                                        sample_to_tours_to_subtours_vars[sample][tour_id].append(name)
                                    except KeyError:
                                        try:
                                            sample_to_tours_to_subtours_vars[sample][tour_id] = [name]
                                        except KeyError:
                                            sample_to_tours_to_subtours_vars[sample] = {tour_id: [name]}
                                            
                                    try:
                                        T_od_to_sample_to_tours_vars[(time_bin, zone, o_z)][sample].append(name)
                                    except KeyError:
                                        try:
                                            T_od_to_sample_to_tours_vars[(time_bin, zone, o_z)][sample] = [name]
                                        except KeyError:
                                            T_od_to_sample_to_tours_vars[(time_bin, zone, o_z)] = {sample: [name]}
                                            
                                    try:
                                        T_to_sample_to_ozone_subtour_vars[(start_bin, o_z)][sample].append(name)
                                    except KeyError:
                                        try:
                                            T_to_sample_to_ozone_subtour_vars[(start_bin, o_z)][sample] = [name]
                                        except KeyError:
                                            T_to_sample_to_ozone_subtour_vars[(start_bin, o_z)] = {sample: [name]}
                                    
                                    if sample_tour_id_to_plan_id[sample].get(tour_id) is None:        
                                        try:
                                            T_to_sample_to_dzone_subtour_vars[(end_bin, end_zone)][sample].append(name)
                                        except KeyError:
                                            try:
                                                T_to_sample_to_dzone_subtour_vars[(end_bin, end_zone)][sample] = [name]
                                            except KeyError:
                                                T_to_sample_to_dzone_subtour_vars[(end_bin, end_zone)] = {sample: [name]}                      
            
            # create constraints
            
            # number of rebalancing vehicles
            
            # current
            for time_bin, o_to_list_vars in current_T_from_o_vars.items():
                for o_zone, list_vars in o_to_list_vars.items():
                    lhs = sum(vars[x] for x in list_vars)
                    V_0_idle = len(avail_for_rebal_vehicles.get(o_zone, []))
                    m.addConstr(lhs <= V_0_idle, name = f"o_constr_c_{time_bin}_{o_zone}" )
                
            # future sample
            for sample, T_from_o_vars in future_T_from_o_vars.items():
                for time_bin, o_to_list_vars in T_from_o_vars.items():
                    for o_zone, list_vars in o_to_list_vars.items():
                        lhs = sum(vars[x] for x in list_vars)
                        V_0_idle = len(avail_for_rebal_vehicles.get(o_zone, []))
                        prev_terms = grp.LinExpr()
                        for time_bin_2 in range(sim_time + self.list_horizons[0], time_bin, self._repo_time_step):
                            prev_rebalanced_vehicles = sum(vars[x] for x in T_from_o_vars.get(time_bin_2, {}).get(o_zone, [])) + sum(vars[x] for x in current_T_from_o_vars.get(time_bin_2, {}).get(o_zone, []))
                            prev_terms -= prev_rebalanced_vehicles
                            
                            new_idle_vehs = sample_timebin_to_zone_to_add_idle_vehs[sample].get(time_bin_2, {}).get(o_zone, 0)
                            
                            ending_tours = sum(vars[x] for x in T_to_sample_to_dzone_subtour_vars.get((time_bin_2, o_zone), {}).get(sample, []))
                            
                            prev_terms += new_idle_vehs + ending_tours
                        m.addConstr(lhs <= V_0_idle + prev_terms, name = f"o_constr_fut_{time_bin}_{o_zone}_{sample}" )

            # reachable tours
            for (time_bin, o_zone, d_zone), sample_list_vars in T_od_to_sample_to_tours_vars.items():
                for sample, list_vars in sample_list_vars.items():
                    lhs = sum(vars[x] for x in list_vars)
                    T_od_var = current_T_od_vars.get(time_bin, {}).get((o_zone, d_zone))
                    if T_od_var is None:
                        T_od_var = future_T_od_vars.get(sample, {}).get(time_bin, {}).get((o_zone, d_zone))
                    if T_od_var is not None:
                        m.addConstr(lhs <= vars[T_od_var], name = f"tour_constr_{time_bin}_{o_zone}_{d_zone}_{sample}" )
                    else:
                        m.addConstr(lhs == 0, name = f"tour_constr_{time_bin}_{o_zone}_{d_zone}_{sample}" )
                        
            # only one subtour per tour and sample
            for sample, tours_to_subtours_vars in sample_to_tours_to_subtours_vars.items():
                for tour, subtours_vars in tours_to_subtours_vars.items():
                    lhs = sum(vars[x] for x in subtours_vars)
                    if sample_tour_id_to_plan_id[sample].get(tour) is not None:
                        m.addConstr(lhs == 1, name = f"subtour_constr_{sample}_{tour}_force" )
                    else:
                        m.addConstr(lhs <= 1, name = f"subtour_constr_{sample}_{tour}" )
                                                        
            if WRITE_PROBLEM:
                m.write(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], f"opt_problem_matching_{sim_time}.lp"))
                
            m.optimize()

            # retrieve solution
            sample_to_look_for = 0

            vals = m.getAttr('X', vars)
            #print(vals)
            od_number_rebal = {}
            future_od_number_rebal = {}
            vid_found = {}
            sample_assigned_reservation_tour_variables = []
            sample_assigned_repo_variables = []
            sample_all_tour_variables = []
            LOG.debug("ODs with rebalancing vehicles:")
            for x in vals:
                v = vals[x]
                #LOG.debug(f"{v}, {x}")
                v = int(np.round(v))
                if v == 0:
                    continue
                if x.startswith(f"od"):
                    v = int(np.round(v))
                    if v == 0:
                        continue
                    x = x.split("_")
                    if len(x) == 5:
                        _, _, time_bin, o_z, d_z = x
                        sample = 0
                    else:
                        _, _, sample, time_bin, o_z, d_z = x
                    o_z = int(o_z)
                    d_z = int(d_z)
                    time_bin = int(time_bin)
                    # if o_z == d_z:
                    #     continue
                    if len(x) == 5:
                        od_number_rebal[(o_z, d_z)] = v
                    else:
                        future_od_number_rebal[(time_bin, o_z, d_z)] = v/self.N_samples
                    if sample == sample_to_look_for:
                        sample_assigned_repo_variables.append( (x, v) )
                # tour variables f"t_z_{zone}_{time_bin}_s_{sample}_ti_{tour_id}_sti_{subtour_id}"
                elif x.startswith("t_z"):
                    tour_id = int(x.split("_")[-3])
                    sample = int(x.split("_")[-5])
                    if sample_tour_id_to_plan_id[sample].get(tour_id) is not None:
                        LOG.debug(f"tour {tour_id} assigned to {sample_tour_id_to_plan_id[sample][tour_id]} in variable {x}")
                        LOG.debug(f"tours in sample: {sample_tour_id_to_plan_id[sample]}")
                        if sample == sample_to_look_for:
                            sample_assigned_reservation_tour_variables.append(x)
                    elif sample == sample_to_look_for:
                        sample_all_tour_variables.append(x)
            LOG.debug(f"assigned reservation tours: {sample_assigned_reservation_tour_variables}  {sample_tour_id_to_plan_id}")
            
            # reconstruct assignment to find vehicles assigned to reservation tour
            tour_to_sol_vars = {} # tour id -> list assigned rebal variables
            od_to_origins = {} # zone (zone_id, time_bin) -> (assigned_rebal_vars, incoming tours, idle_vehicles)
            tour_to_number_end_zones = {}
            
            # check flow of constraints
            for constr in m.getConstrs():
                if constr.ConstrName.startswith("subtour"):
                    continue
                if constr.ConstrName.startswith("o_constr"): # defines incoming and outgoing trips per zone/time_bin with idle vehicles
                    if constr.ConstrName.startswith("o_constr_fut"): # assignment of current rebal trips is trivial (defined by tour_constr)
                        sample = int(constr.ConstrName.split("_")[-1])
                        if sample != sample_to_look_for:
                            continue
                        lexpr = m.getRow(constr)
                        time_bin = int(constr.ConstrName.split("_")[3])
                        origin_zone = int(constr.ConstrName.split("_")[4])
                        idle_vehicles = int(round(constr.RHS))
                        incoming_tours = []
                        od_vars = []
                        #print("")
                        #print(constr.ConstrName, constr.RHS)
                        for i in range(lexpr.size()):
                            var = lexpr.getVar(i)
                            coeff = lexpr.getCoeff(i)
                            sol = var.x
                            sol = int(round(sol))
                            if sol != 0:
                                #print(var.varName, coeff, sol)
                                if var.varName.startswith("od"):
                                    var_time_bin = int(var.varName.split("_")[-3])
                                    if var_time_bin < time_bin:
                                        idle_vehicles -= sol
                                    if var_time_bin == time_bin:
                                        od_vars.append( (var.varName, sol) )
                                else:
                                    incoming_tours.append( var.varName )
                                    tour_to_number_end_zones[var.varName] = tour_to_number_end_zones.get(var.varName, 0) + sol
                        if len(od_vars) == 0:
                            continue
                        # print(constr)
                        # print("remaining idle vehicles", idle_vehicles)
                        # print("incoming tours", incoming_tours)
                        # print("od vars", od_vars)
                        # print("")
                        od_to_origins[(origin_zone, time_bin)] = (od_vars, incoming_tours, idle_vehicles)
                        # print()
                        # continue
                if constr.ConstrName.startswith("tour_constr"): # tour -> rebal trip
                    sample = int(constr.ConstrName.split("_")[-1])
                    if sample != sample_to_look_for:
                        continue
                    # print(constr.ConstrName, constr.RHS)
                    # print(m.getRow(constr))
                    # print("")
                    lexpr = m.getRow(constr)
                    assigned_tours = []
                    assigned_rebals = []
                    for i in range(lexpr.size()):
                        var = lexpr.getVar(i)
                        coeff = lexpr.getCoeff(i)
                        sol = var.x
                        sol = int(round(sol))
                        #print(var, coeff)
                        if sol != 0:
                            if var.varName.startswith("t_z"):
                                assigned_tours.append(var.varName)
                            elif var.varName.startswith("od"):
                                assigned_rebals.append( (var.varName, sol) )
                    if len(assigned_tours) > 0:
                        for tour in assigned_tours:
                            tour_to_sol_vars[tour] = assigned_rebals
            
            # retrace assigned reservation trip to originating rebal trip (if future rebal assigned, this has to point to an vehicle becoming idle in the future)                
            already_assigned = {}
            def trace_back(tour_variable):
                """ traces back to the predecessing rebal variable"""
                # check if od_cur is in assigned rebals
                #print("trace back", tour_variable, tour_to_sol_vars[tour_variable])
                od_cur_var = None
                for od_var, sol in tour_to_sol_vars[tour_variable]:
                    if od_var.startswith("od_cur"):
                        if already_assigned.get(od_var, 0) < sol:
                            od_cur_var = (od_var, sol)
                            break
                if od_cur_var is not None:
                    return ("rebal", od_cur_var)
                # trace back future rebal trip to originating trips
                considered_var = None
                for od_var, sol in tour_to_sol_vars[tour_variable]:
                    if already_assigned.get(od_var, 0) < sol:
                        considered_var = (od_var, sol)
                        break
                if considered_var is None:
                    raise Exception("No feasible assignment found")
                
                o_zone = int(considered_var[0].split("_")[4])
                time_bin = int(considered_var[0].split("_")[3])
                zone_vars = od_to_origins[(o_zone, time_bin)]
                
                if zone_vars[2] > already_assigned.get( (o_zone, time_bin) , 0):
                    return ("idle", (o_zone, time_bin))
                
                for tour in sorted(zone_vars[1], key=lambda x: tour_to_number_end_zones.get(x, 0), reverse=False):
                    if already_assigned.get(tour) is None:
                        return ("tour", tour)
                
                LOG.warning("tour", tour_variable)
                LOG.warning("tour to sol", tour_to_sol_vars[tour_variable])
                LOG.warning("already assigned", already_assigned)
                LOG.warning("zones", zone_vars)
                raise Exception("No feasible assignment found")
                
            LOG.debug("retraced tours:")
            plan_id_to_repo_trip = {}   #plan_id -> (o_zone, d_zone)
            plan_id_to_idle_bin = {}    #plan_id -> (o_zone, time_bin) (where vehicles become idle)
            for assigned_tour in sample_assigned_reservation_tour_variables:
                #print("evaluate assigned tour: ", assigned_tour)
                prev_tour = ("tour", assigned_tour)
                all_tour = [prev_tour]
                while prev_tour[0] == "tour":
                    prev_tour = trace_back(prev_tour[1])
                    all_tour.append(prev_tour)
                all_tour = [all_tour[i] for i in range(len(all_tour)-1, -1, -1)]
                LOG.debug(f"{all_tour}")
                for tour in all_tour:
                    already_assigned[tour[1]] = already_assigned.get(tour[1], 0) + 1
                
                sample_tour_id = int(assigned_tour.split("_")[-3])
                plan_id = sample_tour_id_to_plan_id[sample_to_look_for][sample_tour_id]
                if all_tour[0][0] == "rebal": #('rebal', ('od_cur_0_29_29', 1))
                    u = all_tour[0][1][0].split("_")
                    o_zone = int(u[3])
                    d_zone = int(u[4])
                    plan_id_to_repo_trip[plan_id] = (o_zone, d_zone)
                elif all_tour[0][0] == "idle": #('idle', (2, 1800))
                    plan_id_to_idle_bin[plan_id] = all_tour[0][1]
                

            if WRITE_SOL:
                import pandas as pd
                avail_sol_df_list = []
                for od, repos in od_number_rebal.items():
                    avail_sol_df_list.append({
                        "o_zone_id" : od[0],
                        "d_zone_id" : od[1],
                        "repos" : repos,
                        "time_bin" : sim_time + self.list_horizons[0],
                    })
                for tod, repos in future_od_number_rebal.items():
                    avail_sol_df_list.append({
                        "o_zone_id" : tod[1],
                        "d_zone_id" : tod[2],
                        "repos" : repos,
                        "time_bin" : tod[0],
                    })
                pd.DataFrame(avail_sol_df_list).to_csv(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], f"repo_results_{sim_time}.csv"), index=False)

        return od_number_rebal, plan_id_to_repo_trip, plan_id_to_idle_bin 
    
    def _od_to_veh_plan_assignment(self, sim_time, origin_zone_id, destination_zone_id, list_veh_to_consider,
                                   destination_node=None, lock = True):
        """This method translates an OD repositioning assignment for one of the vehicles in list_veh_to_consider.
        This method adds a PlanStop at a random node in the destination zone, thereby calling
        the fleet control class to make all required data base entries.

        :param sim_time: current simulation time
        :param origin_zone_id: origin zone id
        :param destination_zone_id: destination zone id
        :param list_veh_to_consider: list of (idle/repositioning) simulation vehicle objects
        :param destination_node: if destination node is given, it will be prioritized over picking a random node in
                                    the destination zone
        :param lock: indicates if vehplan should be locked
        :return: list_veh_objs with new repositioning assignments
        """
        # v0) randomly pick vehicle, randomly pick destination node in destination zone
        random.seed(sim_time)
        veh_obj = random.choice(list_veh_to_consider)
        veh_plan = self.fleetctrl.veh_plans[veh_obj.vid]
        if destination_node is None:
            destination_node = self._repo_zone_system.get_random_centroid_node(destination_zone_id)
            LOG.debug("repositioning {} to zone {} with centroid {}".format(veh_obj.vid, destination_zone_id,
                                                                            destination_node))
            if destination_node < 0:
                destination_node = self._repo_zone_system.get_random_node(destination_zone_id)
        ps = RoutingTargetPlanStop((destination_node, None, None), locked=lock, planstop_state=G_PLANSTOP_STATES.REPO_TARGET)
        if len(veh_plan.list_plan_stops) == 0 or not veh_plan.list_plan_stops[-1].is_locked_end():
            veh_plan.add_plan_stop(ps, veh_obj, sim_time, self.routing_engine)
        else:
            new_list_plan_stops = veh_plan.list_plan_stops[:-1] + [ps] + veh_plan.list_plan_stops[-1:]
            veh_plan.list_plan_stops = new_list_plan_stops
            veh_plan.update_tt_and_check_plan(veh_obj, sim_time, self.routing_engine, keep_feasible=True)
            if not veh_plan.is_feasible():
                LOG.warning("veh plan not feasible after assigning repo with reservation! {}".format(veh_plan))
        self.fleetctrl.assign_vehicle_plan(veh_obj, veh_plan, sim_time)
        if lock:
            self.fleetctrl.lock_current_vehicle_plan(veh_obj.vid)
        return [veh_obj]
    
    def _get_od_zone_travel_info(self, sim_time, o_zone_id, d_zone_id):
        """This method returns OD travel times on zone level.

        :param o_zone_id: origin zone id
        :param d_zone_id: destination zone id
        :return: tt, dist
        """
        # v0) pick random node and compute route
        loop_iter = 0
        while True:
            o_pos = self.routing_engine.return_node_position(self._repo_zone_system.get_random_centroid_node(o_zone_id))
            d_pos = self.routing_engine.return_node_position(self._repo_zone_system.get_random_centroid_node(d_zone_id))
            if o_pos[0] >= 0 and d_pos[0] >= 0:
                route_info = self.routing_engine.return_travel_costs_1to1(o_pos, d_pos)
                if route_info:
                    return route_info[1], route_info[2]
                loop_iter += 1
                if loop_iter == 10:
                    break
            else:
                break
        # TODO # v1) think about the use of centroids!
        return 10000000, 10000000
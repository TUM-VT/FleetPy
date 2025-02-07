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
WRITE_PROBLEM = True
SMALL_VALUE = 10
LARGE_VALUE = 10**6
GUROBI_MIPGAP = 10**-8

INPUT_PARAMETERS_FullSamplingRidePoolingRebalancingMultiStage = {
    "doc" :     """ this class implements the sampling based repositioning method for ride-pooling
    described in roman's thesis (https://mediatum.ub.tum.de/?id=1755168).
    The method is based on sampling future requests and simulating the future vehicle states.
    based on future fleet states, repositioning trips are assigned to close supply gaps.
    """,
    "inherit" : "RepositioningBase",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [G_OP_REPO_SAMPLE_GAMMA, G_OP_REPO_NR_SAMPLES, G_RA_FC_TYPE, G_RA_FC_TYPE, G_OP_REPO_RES_PUF],
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
        LOG.debug(f"move {veh.vid} at {t} - {t + time_step} pos {veh.pos}, {veh_plan}")
        cur_pos = veh.pos
        next_t = t + time_step
        last_t = t
        ps_ind = 0
        for i, ps in enumerate(veh_plan.list_plan_stops):
            arr, dep = ps.get_planned_arrival_and_departure_time()
            LOG.debug(f"arr dep {arr} {dep}")
            if next_t < arr:    # move along route
                target_pos = ps.get_pos()
                cur_pos = veh.pos
                if cur_pos != target_pos:
                    if ps.get_earliest_start_time() - next_t  > begin_approach_buffer_time and \
                        ps.get_earliest_start_time() - next_t - routing_engine.return_travel_costs_1to1(cur_pos, target_pos)[1] > begin_approach_buffer_time:
                        LOG.debug(f"wait {veh.vid} at {cur_pos} for {target_pos} {ps.get_earliest_start_time()} - {next_t} > {begin_approach_buffer_time}")
                    else:
                        route = vid_to_current_route.get(veh.vid, [])
                        LOG.debug(f"old route {cur_pos} {route}")
                        if len(route) == 0 or route[-1] != target_pos[0]:
                            route = routing_engine.return_best_route_1to1(cur_pos, target_pos)
                            try:
                                route.remove(cur_pos[0])
                            except:
                                pass
                        LOG.debug(f"new route {cur_pos} {route}")
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
                    LOG.debug(f"check db {rid}")
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
    

class FullSamplingRidePoolingRebalancingMultiStage(RepositioningBase):
    def __init__(self, fleetctrl: FleetControlBase, operator_attributes: dict, dir_names: dict, solver: str = "Gurobi"):
        """
        this class implements the sampling based repositioning method for ride-pooling
        described in roman's thesis (https://mediatum.ub.tum.de/?id=1755168).
        The method is based on sampling future requests and simulating the future vehicle states.
        based on future fleet states, repositioning trips are assigned to close supply gaps.
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
        self._pregress_time_step = operator_attributes.get(G_RA_REOPT_TS, 60)
        self.min_reservation_buffer = max(operator_attributes.get(G_OP_REPO_RES_PUF, 3600), self.list_horizons[1])
        
    def determine_and_create_repositioning_plans(self, sim_time: int, lock: bool=None) -> List[int]:
        """ computes and assigns new repositioning plans
        :param sim_time: current simulation time
        :param lock: bool if repositioning should be locked
        :return: list of vehicle ids with adopted schedules (new repositioning targets)"""
        self.sim_time = sim_time
        self.zone_system.time_trigger(sim_time)
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
        
        for _ in range(self.N_samples): # do for each sample
            tm = time.time()
            future_rq_atts = []
            for t0 in range(t0_min, t1_max, self.zone_system.fc_temp_resolution):
                # sampled future requests
                new_future_rq_atts = self.zone_system.draw_future_request_sample(t0, t0 + self.zone_system.fc_temp_resolution)
                time_bin_int = int( (t0 - sim_time + self.list_horizons[0])/self.zone_system.fc_temp_resolution )
                priority = self._gamma ** time_bin_int  # priority of future requests
                new_future_rq_atts = [(*x, priority) for x in new_future_rq_atts]
                future_rq_atts += new_future_rq_atts

            # simulate future vehicle states and create input parameters for matching
            tour_id_subtour_id_parameters, start_bin_to_ozone_to_tour_ids, end_bin_to_dzone_to_tour_ids, timebin_to_zone_to_idle_vehs = \
                self._progress_vehicles(future_rq_atts, sim_time, t1_max)
                
            LOG.debug(f"tour_id_subtour_id_parameters {tour_id_subtour_id_parameters}")
            LOG.debug(f"start_bin_to_ozone_to_tour_ids {start_bin_to_ozone_to_tour_ids}")
            LOG.debug(f"end_bin_to_dzone_to_tour_ids {end_bin_to_dzone_to_tour_ids}")
            LOG.debug(f"timebin_to_zone_to_idle_vehs {timebin_to_zone_to_idle_vehs}")
            sample_tour_id_subtour_id_parameters.append(tour_id_subtour_id_parameters)
            sample_start_bin_to_ozone_to_tour_ids.append(start_bin_to_ozone_to_tour_ids)
            sample_end_bin_to_dzone_to_tour_ids.append(end_bin_to_dzone_to_tour_ids)
            sample_timebin_to_zone_to_idle_vehs.append(timebin_to_zone_to_idle_vehs)
            LOG.debug(f"sampling took {time.time() - tm}")
            _sampling_times.append(time.time() - tm)
        
        tm = time.time()
        # get forecast of vehicle availabilities      
        _, _, avail_for_rebal_vehicles = self._get_vehs_with_vp_available_for_repo(sim_time)
        
        # solve matching problem to reposition idle vehicles to zones
        od_number_rebal = self._solve_sampling_repositioning(sim_time, avail_for_rebal_vehicles, sample_tour_id_subtour_id_parameters, 
                                                                   sample_start_bin_to_ozone_to_tour_ids, sample_end_bin_to_dzone_to_tour_ids,
                                                                   sample_timebin_to_zone_to_idle_vehs)
        
        list_veh_with_changes = self._assign_repositioning_plans(sim_time, od_number_rebal, avail_for_rebal_vehicles, lock=self.lock_repo_assignments)
        _solve_time = time.time() - tm
        LOG.info(f"Sampling took {sum(_sampling_times)} | Matching took {_solve_time} | Total {sum(_sampling_times) + _solve_time} | Single sample times: {_sampling_times}")
                    
        return list_veh_with_changes
    
    def _get_vehs_with_vp_available_for_repo(self, sim_time : int) -> Tuple[Dict[int, SimulationVehicleStruct], Dict[int, VehiclePlan], Dict[int, List[SimulationVehicleStruct]]]:
        """ 
        prepares input parameters for vehicles for sampling based repositioning in combination with reservations
        identifies idle vehicles and vehicles with currently assigned vehicle plans until the end of the forecast horizon
        :param sim_time: current simulation time
        :return: 
            veh_objs: dict of vehicle id -> vehicle object with current state (copies for preprogression)
            veh_plans: dict of vehicle id -> vehicle plan object (copies for preprogression)
            zone_to_idle_vehs: dict of zone -> list of vehicle objects that are idle (within the forecast horizon)
        """
        n_idle = 0
        veh_objs: Dict[int, SimulationVehicleStruct] = {}
        veh_plans: Dict[int, VehiclePlan] = {}
        zone_to_idle_vehs = {}
        for veh_obj in self.fleetctrl.sim_vehicles:
            vp = self.fleetctrl.veh_plans.get(veh_obj.vid)
            if vp is not None and len(vp.list_plan_stops) > 0: # and not vp.list_plan_stops[-1].is_locked_end() and not vp.list_plan_stops[-1].is_locked():
                #TODO recheck for this if incase of reservations!
                if not (len(vp.list_plan_stops) == 1 and vp.list_plan_stops[-1].is_locked_end()):
                    veh_objs[veh_obj.vid] = SimulationVehicleStruct(veh_obj, vp, sim_time, self.routing_engine)
                    veh_plans[veh_obj.vid] = vp.copy()
                else:
                    LOG.debug(f"veh {veh_obj.vid} has reservation plan stops at time {sim_time}")
                    LOG.error(" this module should not treat reservations! ")
                    raise EnvironmentError(" this module should not treat reservations! ")
                    est = vp.list_plan_stops[-1].get_earliest_start_time()
                    if est > sim_time + self.min_reservation_buffer:
                        LOG.debug(f" -> but far away {est}")
                        n_idle += 1
                        zone = self.zone_system.get_zone_from_pos(veh_obj.pos)
                        try:
                            zone_to_idle_vehs[zone].append(veh_obj)
                        except KeyError:
                            zone_to_idle_vehs[zone] = [veh_obj]
                    else:
                        plan_id, full_plan = self.fleetctrl.reservation_module.get_full_vehicle_plan_until( veh_obj.vid, sim_time, until_time=sim_time + self.min_reservation_buffer)
                        veh_objs[veh_obj.vid] = SimulationVehicleStruct(veh_obj, vp, sim_time, self.routing_engine)
                        veh_plans[veh_obj.vid] = full_plan.copy()
                        LOG.debug(f" -> but close -> not idle {full_plan}")
            else:
                n_idle +=1
                zone = self.zone_system.get_zone_from_pos(veh_obj.pos)
                try:
                    zone_to_idle_vehs[zone].append(veh_obj)
                except KeyError:
                    zone_to_idle_vehs[zone] = [veh_obj]
        return veh_objs, veh_plans, zone_to_idle_vehs
    
    def _assign_repositioning_plans(self, sim_time, od_number_rebal, avail_for_rebal_vehicles, lock=False) -> List[int]:
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
        """
        # only currently driving vehicles (idle vehicles will be created)
        veh_objs, veh_plans, _ = self._get_vehs_with_vp_available_for_repo(sim_time)

        rq_dict = self.fleetctrl.rq_dict.copy()
        
        current_new_vid = len(self.fleetctrl.sim_vehicles) + 1
        new_vids_to_ps = {}
        
        vids_with_assignment = {}
        
        pos_to_vids = {}
        
        opt_time_step = self._pregress_time_step
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
            if priority is not None:
                prqs[rid].priority = priority
        
        timebin_to_zone_to_idle_vehs = {} # time bin -> zone -> number of idle vehicles
        
        vid_to_route = {}    # caching routes
        for t in range(sim_time, prog_end_time, opt_time_step):
            # track new idle vehicles at start of time bin
            if t != sim_time and t%self.zone_system.fc_temp_resolution == 0:
                # update currently idle vehicles
                timebin_to_zone_to_idle_vehs[t] = {}
                
                for vid, veh_obj in list(veh_objs.items()):
                    if vid < len(self.fleetctrl.sim_vehicles):
                        if len(veh_plans[vid].list_plan_stops) == 0:
                            LOG.debug(f"veh {vid} has no plan stops at time {t}")
                            zone = self.zone_system.get_zone_from_pos(veh_obj.pos)
                            try:
                                timebin_to_zone_to_idle_vehs[t][zone] += 1
                            except KeyError:
                                timebin_to_zone_to_idle_vehs[t] = {zone: 1}
                            del veh_objs[vid]
                            del veh_plans[vid]
                        elif len(veh_plans[vid].list_plan_stops) == 1 and veh_plans[vid].list_plan_stops[-1].is_locked_end():
                            LOG.debug(f"veh {vid} has reservation plan stop")
                            est = veh_plans[vid].list_plan_stops[-1].get_earliest_start_time()
                            if est > t + self.min_reservation_buffer:
                                LOG.debug(f" -> but far away {est}")
                                zone = self.zone_system.get_zone_from_pos(veh_obj.pos)
                                try:
                                    timebin_to_zone_to_idle_vehs[t][zone] += 1
                                except KeyError:
                                    timebin_to_zone_to_idle_vehs[t] = {zone: 1}
                                del veh_objs[vid]
                                del veh_plans[vid]
                        
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
                    o_zone = self.zone_system.get_zone_from_pos(prq.get_o_stop_info()[0])
                    centroid = self.zone_system.get_random_centroid_node(o_zone)
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

        # create optimization input parameters
        tour_id = 0
        tour_id_subtour_id_parameters = {} # tour_id -> subtour_id -> (o_zone, cost_gain, arr_at_zone, end_zone, start_bin, end_bin)
        start_bin_to_ozone_to_tour_ids = {} # start_bin -> o_zone -> list of (tour_id, subtour_id)
        end_bin_to_dzone_to_tour_ids = {} # end_bin -> d_zone -> list of (tour_id, subtour_id)
        
        for vid, list_ps in new_vids_to_ps.items():
            tour_id_subtour_id_parameters[tour_id] = {}
            subtour_id = 0
            
            vp: VehiclePlan = veh_plans.get(vid, VehiclePlan(self.fleetctrl.sim_vehicles[0], sim_time, self.routing_engine, []))
            LOG.debug(f"target {vid} {list_ps} {vp}")
            start_time = sim_time
            if len(list_ps) > 0:
                o_z = self.zone_system.get_zone_from_pos(list_ps[0].get_pos())
            else:
                o_z = self.zone_system.get_zone_from_pos(vp.list_plan_stops[0].get_pos())
                
            centroid = self.zone_system.get_random_centroid_node(o_z)
            any_vehicle = next(iter(veh_objs.values()))
            new_veh = SimulationVehicleStruct(any_vehicle, VehiclePlan(any_vehicle, sim_time, self.routing_engine, []), sim_time, self.routing_engine, empty_init=True)
            new_veh.pos = (centroid, None, None)
            new_veh.vid = -1
            new_veh.status = VRL_STATES.IDLE

            full_vp = VehiclePlan(new_veh, start_time, self.routing_engine, list_ps + vp.list_plan_stops )
            LOG.debug(f" -- full vp {full_vp}")
            #cost_gain = self.fleetctrl.compute_VehiclePlan_utility(start_time, new_veh, full_vp)
            cost_gain = self._sampling_ctrl_function(start_time, new_veh, full_vp, rq_dict, self.routing_engine)
            arr_time = max(full_vp.list_plan_stops[0].get_planned_arrival_and_departure_time()[0], full_vp.list_plan_stops[0].get_earliest_start_time())
            arr_at_zone = arr_time - self.routing_engine.return_travel_costs_1to1(new_veh.pos, full_vp.list_plan_stops[0].get_pos())[0]
            
            end_time = full_vp.list_plan_stops[-1].get_planned_arrival_and_departure_time()[1]
            start_pos = full_vp.list_plan_stops[0].get_pos()
            end_pos = full_vp.list_plan_stops[-1].get_pos()
            end_zone = self.zone_system.get_zone_from_pos(end_pos)
            
            LOG.debug(f" -- add {(tour_id, subtour_id)} {(o_z, cost_gain, arr_at_zone)}")
            
            start_bin = int(arr_at_zone / self.zone_system.fc_temp_resolution) * self.zone_system.fc_temp_resolution
            end_bin = int(end_time / self.zone_system.fc_temp_resolution) * self.zone_system.fc_temp_resolution
            
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
            
            # create also parameters for subplans
            if len(full_vp.get_involved_request_ids()) > 1:
                #to_remove = full_vp.list_plan_stops[0].get_list_boarding_rids()
                while True:
                    for i, ps in enumerate(full_vp.list_plan_stops):
                        found = False
                        for rid in ps.get_list_boarding_rids():
                            full_vp = simple_remove(new_veh, full_vp, rid, sim_time, self.routing_engine, self._sampling_ctrl_function, rq_dict, self.fleetctrl.const_bt, self.fleetctrl.add_bt)
                            found = True
                        if found:
                            break
                    if len(full_vp.list_plan_stops) == 0 or len(full_vp.get_involved_request_ids()) == 0:
                        break
                    o_z = self.zone_system.get_zone_from_pos(full_vp.list_plan_stops[0].get_pos())
                    
                    centroid = self.zone_system.get_random_centroid_node(o_z)
                    any_vehicle = next(iter(veh_objs.values()))
                    new_veh = SimulationVehicleStruct(any_vehicle, VehiclePlan(any_vehicle, sim_time, self.routing_engine, []), sim_time, self.routing_engine, empty_init=True)
                    new_veh.pos = (centroid, None, None)
                    new_veh.vid = -1
                    new_veh.status = VRL_STATES.IDLE

                    full_vp = VehiclePlan(new_veh, start_time, self.routing_engine, full_vp.list_plan_stops )
                    LOG.debug(f" -- full vp {full_vp}")
                    #cost_gain = self.fleetctrl.compute_VehiclePlan_utility(start_time, new_veh, full_vp)
                    cost_gain = self._sampling_ctrl_function(start_time, new_veh, full_vp, rq_dict, self.routing_engine)
                    arr_time = max(full_vp.list_plan_stops[0].get_planned_arrival_and_departure_time()[0], full_vp.list_plan_stops[0].get_earliest_start_time())
                    arr_at_zone = arr_time - self.routing_engine.return_travel_costs_1to1(new_veh.pos, full_vp.list_plan_stops[0].get_pos())[0]
                    LOG.debug(f" -- add {(o_z, cost_gain, arr_at_zone)}")
                    
                    start_bin = int(arr_at_zone / self.zone_system.fc_temp_resolution) * self.zone_system.fc_temp_resolution
                    end_bin = int(end_time / self.zone_system.fc_temp_resolution) * self.zone_system.fc_temp_resolution
                    
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
                        
            tour_id += 1
        
        #exit()           
        return tour_id_subtour_id_parameters, start_bin_to_ozone_to_tour_ids, end_bin_to_dzone_to_tour_ids, timebin_to_zone_to_idle_vehs
    
    def _solve_sampling_repositioning(self, sim_time: int, avail_for_rebal_vehicles: Dict[int, List[int]], sample_tour_id_subtour_id_parameters, 
                                                                   sample_start_bin_to_ozone_to_tour_ids, sample_end_bin_to_dzone_to_tour_ids,
                                                                   sample_timebin_to_zone_to_add_idle_vehs):
        """ solves the repositioning problem with the given input data 
        :param sim_time: current simulation time
        :param avail_for_rebal_vehicles: zone -> list of vehicle ids
        :param sample_tour_id_subtour_id_parameters: list of tour_id -> subtour_id -> (o_zone, cost_gain, arr_at_zone, end_zone, start_bin, end_bin) per sample
        :param sample_start_bin_to_ozone_to_tour_ids: list of start_bin -> o_zone -> list of (tour_id, subtour_id) per sample
        :param sample_end_bin_to_dzone_to_tour_ids: list of end_bin -> d_zone -> list of (tour_id, subtour_id) per sample
        :param sample_timebin_to_zone_to_add_idle_vehs: list of time bin -> zone -> number of idle vehicles per sample
        :param lock: bool if repositioning should be locked
        :return: dict (o_zone, d_zone) -> number of rebalancing vehicles (new repositioning targets)"""
        
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
                        for time_bin in range(sim_time + self.list_horizons[0], sim_time + self.list_horizons[1], self.zone_system.fc_temp_resolution):
                            time_bin_int = int( (time_bin - sim_time + self.list_horizons[0])/self.zone_system.fc_temp_resolution )
                            factor = self._gamma ** time_bin_int
                            if time_bin == sim_time + self.list_horizons[0]:
                                iterator = avail_for_rebal_vehicles.keys()
                            else:
                                iterator = self.zone_system.get_all_zones()
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
                        for time_bin_2 in range(sim_time + self.list_horizons[0], time_bin, self.zone_system.fc_temp_resolution):
                            prev_rebalanced_vehicles = sum(vars[x] for x in T_from_o_vars.get(time_bin_2, {}).get(o_zone, [])) + sum(vars[x] for x in current_T_from_o_vars.get(time_bin_2, {}).get(o_zone, []))
                            prev_terms -= prev_rebalanced_vehicles
                            
                            new_idle_vehs = sample_timebin_to_zone_to_add_idle_vehs[sample].get(time_bin_2, {}).get(o_zone, 0)
                            
                            ending_tours = sum(vars[x] for x in T_to_sample_to_ozone_subtour_vars.get((time_bin_2, o_zone), {}).get(sample, []))
                            
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
                    m.addConstr(lhs <= 1, name = f"subtour_constr_{sample}_{tour}" )
                                                        
            if WRITE_PROBLEM:
                m.write(os.path.join(self.fleetctrl.dir_names[G_DIR_OUTPUT], f"opt_problem_matching_{sim_time}.lp"))
                
            m.optimize()

            # retrieve solution

            vals = m.getAttr('X', vars)
            #print(vals)
            od_number_rebal = {}
            future_od_number_rebal = {}
            vid_found = {}
            LOG.debug("ODs with rebalancing vehicles:")
            for x in vals:
                v = vals[x]
                LOG.debug(f"{v}, {x}")
                if not x.startswith(f"od"):
                    continue
                v = int(np.round(v))
                if v == 0:
                    continue
                x = x.split("_")
                if len(x) == 5:
                    _, _, time_bin, o_z, d_z = x
                else:
                    _, _, sample, time_bin, o_z, d_z = x
                o_z = int(o_z)
                d_z = int(d_z)
                time_bin = int(time_bin)
                if o_z == d_z:
                    continue
                if len(x) == 5:
                    od_number_rebal[(o_z, d_z)] = v
                else:
                    future_od_number_rebal[(time_bin, o_z, d_z)] = v/self.N_samples

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

        return od_number_rebal 
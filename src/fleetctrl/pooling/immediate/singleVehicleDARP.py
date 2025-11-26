from typing import Callable, List, Dict, Any, Tuple
import logging

import numpy as np
from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.planning.VehiclePlan import BoardingPlanStop, PlanStop, VehiclePlan
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.simulation.Vehicles import SimulationVehicle
from src.routing.NetworkBase import NetworkBase
from src.fleetctrl.pooling.immediate.insertion import simple_insert, simple_remove

LOG = logging.getLogger(__name__)

INIT_TEMP_SCALING = 10
MAX_NO_NEW_SOLS = 4
SIM_ANNEAL_TEMP_SCALING = 0.9

def solve_single_vehicle_DARP_exhaustive(veh_obj: SimulationVehicle, routing_engine: NetworkBase,
                                         list_prqs: List[PlanRequest], fleetctrl: FleetControlBase, sim_time, currently_assigned_vehplan: VehiclePlan):
    """ recursive implementation of an exhaustive search for the optional solution of a single vehicle DARP
    :param veh_obj: vehicle object
    :param routing_engine: routing engine
    :param list_prqs: list of plan requests to pick up and drop off (on-board requests must be included)
    :param fleetctrl: fleet control object
    :param sim_time: simulation time
    :param currently_assigned_vehplan: current vehicle plan
    :return: best plan and its utility"""

    o_pos_s = [(prq.get_o_stop_info()[0], prq, True, i) for i, prq in enumerate(list_prqs)]
    d_pos_s = [(prq.get_d_stop_info()[0], prq, False, i) for i, prq in enumerate(list_prqs)]

    def add_one_of_next_possible_stops(next_possible_stops:List[Tuple[tuple,PlanRequest,bool,int]], index_to_add:int,
                                       current_stop_list:List[PlanStop], last_ps=None):
        #LOG.debug(f"add next stop with {[(a, b.get_rid_struct(), c, d) for a, b, c, d in next_possible_stops]} and index {index_to_add}")
        new_pos = next_possible_stops[index_to_add]
        prq = new_pos[1]
        new_rid_struct = prq.get_rid_struct()
        
        new_next_possible_stops = next_possible_stops[:index_to_add] + next_possible_stops[index_to_add+1:]
        new_current_stop_list = [ps.copy() for ps in current_stop_list]
        if len(current_stop_list) > 0 and current_stop_list[-1].get_pos() == new_pos[0] \
                and not current_stop_list[-1].is_locked() and not current_stop_list[-1].is_locked_end(): # include boarding at same stop
            if new_pos[2]: # boarding node
                prq_o_stop_pos, prq_t_pu_earliest, prq_t_pu_latest = prq.get_o_stop_info()
                old_pstop = new_current_stop_list[-1]
                new_boarding_list = old_pstop.get_list_boarding_rids() + [new_rid_struct]
                new_boarding_dict = {-1:old_pstop.get_list_alighting_rids(), 1:new_boarding_list}
                ept_dict, lpt_dict, mtt_dict, lat_dict = old_pstop.get_boarding_time_constraint_dicts()
                new_earliest_pickup_time_dict = ept_dict.copy()
                new_earliest_pickup_time_dict[new_rid_struct] = prq_t_pu_earliest
                new_latest_pickup_time_dict = lpt_dict.copy()
                new_latest_pickup_time_dict[new_rid_struct] = prq_t_pu_latest
                stop_duration, _ = old_pstop.get_duration_and_earliest_departure()
                if stop_duration is None:
                    stop_duration = fleetctrl.const_bt
                else:
                    stop_duration += fleetctrl.add_bt
                change_nr_pax = old_pstop.get_change_nr_pax()
                change_nr_pax += prq.nr_pax
                
                new_current_stop_list[-1] = BoardingPlanStop(prq_o_stop_pos, boarding_dict=new_boarding_dict, max_trip_time_dict=mtt_dict.copy(),
                                                                latest_arrival_time_dict=lat_dict.copy(), earliest_pickup_time_dict=new_earliest_pickup_time_dict,
                                                                latest_pickup_time_dict=new_latest_pickup_time_dict, change_nr_pax=change_nr_pax,duration=stop_duration,
                                                                change_nr_parcels=old_pstop.get_change_nr_parcels())
            else: # alighting node
                old_pstop = new_current_stop_list[-1]
                d_stop_pos, _, prq_max_trip_time = prq.get_d_stop_info()  
                # combine with last stop if it is at the same location (combine constraints)
                new_alighting_list = old_pstop.get_list_alighting_rids() + [new_rid_struct]
                new_boarding_dict = {1:old_pstop.get_list_boarding_rids(), -1:new_alighting_list}
                ept_dict, lpt_dict, mtt_dict, lat_dict = old_pstop.get_boarding_time_constraint_dicts()
                new_max_trip_time_dict = mtt_dict.copy()
                new_max_trip_time_dict[new_rid_struct] = prq_max_trip_time
                stop_duration, _ = old_pstop.get_duration_and_earliest_departure()
                if stop_duration is None:
                    stop_duration = fleetctrl.const_bt
                else:
                    stop_duration += fleetctrl.add_bt
                change_nr_pax = old_pstop.get_change_nr_pax()
                change_nr_pax -= prq.nr_pax
                
                new_current_stop_list[-1] = BoardingPlanStop(d_stop_pos, boarding_dict=new_boarding_dict, max_trip_time_dict=new_max_trip_time_dict,
                                                                  latest_arrival_time_dict=lat_dict.copy(), earliest_pickup_time_dict=ept_dict.copy(),
                                                                  latest_pickup_time_dict=lpt_dict.copy(), change_nr_pax=change_nr_pax, change_nr_parcels=old_pstop.get_change_nr_parcels(),
                                                                  duration=stop_duration)
        else: # new stop
            if new_pos[2]: # boarding node
                prq_o_stop_pos, prq_t_pu_earliest, prq_t_pu_latest = prq.get_o_stop_info()
                new_plan_stop = BoardingPlanStop(prq_o_stop_pos, boarding_dict={1:[new_rid_struct]},
                                                 earliest_pickup_time_dict={new_rid_struct : prq_t_pu_earliest},
                                                latest_pickup_time_dict={new_rid_struct : prq_t_pu_latest},
                                                change_nr_pax=prq.nr_pax,
                                                duration=fleetctrl.const_bt)
                new_current_stop_list.append(new_plan_stop)
            else:  # alighting node
                d_stop_pos, _, prq_max_trip_time = prq.get_d_stop_info()  
                new_plan_stop = BoardingPlanStop(d_stop_pos, boarding_dict={-1: [new_rid_struct]},
                                                 max_trip_time_dict={new_rid_struct : prq_max_trip_time},
                                                 change_nr_pax=-prq.nr_pax, duration=fleetctrl.const_bt)
                new_current_stop_list.append(new_plan_stop)
        veh_p = VehiclePlan(veh_obj, sim_time, routing_engine, new_current_stop_list)
        if veh_p.update_tt_and_check_plan(veh_obj, sim_time, routing_engine):
            if new_pos[2]:
                new_next_possible_stops.append(d_pos_s[new_pos[3]])
            if len(new_next_possible_stops) == 0:
                if last_ps is not None:
                    veh_p = VehiclePlan(veh_obj, sim_time, routing_engine, new_current_stop_list + [last_ps])
                    if veh_p.update_tt_and_check_plan(veh_obj, sim_time, routing_engine):
                        yield veh_p      
                else:
                    yield veh_p
            else:
                for i in range(len(new_next_possible_stops)):
                    yield from add_one_of_next_possible_stops(new_next_possible_stops, i, new_current_stop_list, last_ps=last_ps)
    
    init_list_plan_stops = []
    locked_ob_rids = [rq.get_rid_struct() for rq in veh_obj.pax]
    locked_db_rids = []
    for ps in currently_assigned_vehplan.list_plan_stops:
        if ps.is_locked():
            init_list_plan_stops.append(ps.copy())
            for rid in ps.get_list_boarding_rids():
                if not rid in locked_ob_rids:
                    locked_ob_rids.append(rid)
            for rid in ps.get_list_alighting_rids():
                if not rid in locked_db_rids:
                    locked_db_rids.append(rid)
        else:
            break
    last_ps = None
    if len(currently_assigned_vehplan.list_plan_stops) != 0 and currently_assigned_vehplan.list_plan_stops[-1].is_locked_end():
        last_ps = currently_assigned_vehplan.list_plan_stops[-1]

    LOG.debug(f"solve exhaustive DARP for veh {veh_obj.vid} with requests {[prq.get_rid_struct() for prq in list_prqs]} with last ps {last_ps}")
    LOG.debug(f"currently assigned: {currently_assigned_vehplan}")
    if len(veh_obj.pax) != 0:
        LOG.debug(f"vid {veh_obj}")
        LOG.debug(f"with ob {[p.get_rid_struct() for p in veh_obj.pax]}")
    init_positions = []
    for i, o_pos in enumerate(o_pos_s):
        if o_pos[1].get_rid() in locked_ob_rids:
            if not o_pos[1].get_rid() in locked_db_rids:
                init_positions.append(d_pos_s[o_pos[3]])
        else:
            init_positions.append(o_pos)
    #LOG.debug(f"init positions: {[(a, b.get_rid_struct(), c, d) for a, b, c, d in init_positions]}")
    best_obj = float("inf")
    best_plan = None
    for i, o_pos in enumerate(init_positions):
        for veh_plan in add_one_of_next_possible_stops(init_positions, i, init_list_plan_stops, last_ps=last_ps):
            if veh_plan is not None:
                LOG.debug(f"found {veh_plan}")
                obj = fleetctrl.compute_VehiclePlan_utility(sim_time, veh_obj, veh_plan)
                LOG.debug(f"obj {obj}")
                if obj < best_obj:
                    best_obj = obj
                    veh_plan.set_utility(obj)
                    best_plan = veh_plan
                    
    return best_plan, best_obj

def markovDecision(old_cfv, new_cfv, Temperature, minimize = True):
    """ this function returns if a solution transition is made based on a
    probabilistic markov process
        p(True) = 1 if new_cfv < old_cfv
                = exp(- (new_cfv - old_cfv)/T) else
        p(False) = 1 - p(True)
    :param old_cfv: cost_function_value of old solution
    :param new_cfv: cost function value of new solution
    :param Temperature: temperature value
    :param minimize: if False, a higher cfv is prefered -> process turned around 
    :return: True, if new_sol is accepted, False else
    """
    delta = new_cfv - old_cfv
    if not minimize:
        delta = -delta
    if delta < 0:
        return True
    elif Temperature == 0.0:
        return False
    else:
        r = np.random.random()
        try:
            p = np.math.exp(-delta/Temperature) #.exp(-delta/deltaT)
        except:
            if -delta/Temperature > 0:
                p = float("inf")
            else:
                print("exp error")
                print(delta, Temperature, new_cfv, old_cfv)
                exit()
        if r < p:
            return True
        else:
            return False
                
def solve_single_vehicle_DARP_LNS(veh_obj: SimulationVehicle, routing_engine: NetworkBase, fleetctrl: FleetControlBase, sim_time,
                                         currently_assigned_vehplan: VehiclePlan, destruction_degree = 4, max_iterations=10):
    """ Large Neighborhood Search for the single vehicle DARP (TODO not used currently)"""
    removeable_rids = [rid for rid in currently_assigned_vehplan.get_involved_request_ids()]
    for trq in veh_obj.pax:
        if trq.get_rid_struct() in removeable_rids:
            removeable_rids.remove(trq.get_rid_struct())
    for ps in currently_assigned_vehplan.list_plan_stops:
        if ps.is_locked():
            for rid in ps.get_list_boarding_rids():
                if rid in removeable_rids:
                    removeable_rids.remove(rid)
            for rid in ps.get_list_alighting_rids():
                if rid in removeable_rids:
                    removeable_rids.remove(rid)
        else:
            break
        
    current_best_plan = currently_assigned_vehplan.copy()
    current_best_obj = fleetctrl.compute_VehiclePlan_utility(sim_time, veh_obj, current_best_plan)
    overall_best_plan = currently_assigned_vehplan.copy()
    overall_best_obj = fleetctrl.compute_VehiclePlan_utility(sim_time, veh_obj, current_best_plan)
    temp = None
    no_new_sol_counter = 0
    number_iter = 0
    while no_new_sol_counter < MAX_NO_NEW_SOLS and number_iter < max_iterations:
        number_iter += 1
        new_plan = current_best_plan.copy()
        unassinged_rqs = []
        current_removeable_rids = removeable_rids[:]
        while len(unassinged_rqs) < destruction_degree and len(current_removeable_rids) != 0:
            remove_rid = np.random.choice(current_removeable_rids)
            new_plan = simple_remove(veh_obj, new_plan, remove_rid, sim_time, routing_engine, fleetctrl.vr_ctrl_f,
                                     fleetctrl.rq_dict, fleetctrl.const_bt, fleetctrl.add_bt)
            unassinged_rqs.append(fleetctrl.rq_dict[remove_rid])
            current_removeable_rids.remove(remove_rid)
        np.random.shuffle(unassinged_rqs)
        full_solution_found = True
        for prq in unassinged_rqs:
            best_insert_plan = None
            best_insert_obj = float("inf")
            for veh_p in simple_insert(routing_engine, sim_time, veh_obj, new_plan, prq, fleetctrl.const_bt, fleetctrl.add_bt):
                obj = fleetctrl.compute_VehiclePlan_utility(sim_time, veh_obj, veh_p)
                if obj < best_insert_obj:
                    best_insert_obj = obj
                    best_insert_plan = veh_p
            if best_insert_plan is not None:
                new_plan = best_insert_plan
            else:
                full_solution_found = False
                break
        if full_solution_found:
            if temp is None:
                if best_insert_obj != current_best_obj:
                    temp = abs(best_insert_obj - current_best_obj) * INIT_TEMP_SCALING
                else:
                    no_new_sol_counter += 1
                    continue
            if current_best_obj < overall_best_obj:
                overall_best_obj = best_insert_obj
                overall_best_plan = best_insert_plan
            if markovDecision(current_best_obj, best_insert_obj, temp):
                current_best_obj = best_insert_obj
                current_best_plan = best_insert_plan
            temp = temp * SIM_ANNEAL_TEMP_SCALING
        else:
            no_new_sol_counter += 1
            
    return overall_best_plan, overall_best_obj
            
            
        
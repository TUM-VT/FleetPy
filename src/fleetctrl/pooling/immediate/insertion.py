from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.planning.VehiclePlan import BoardingPlanStop, PlanStop, VehiclePlan
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.simulation.Vehicles import SimulationVehicle
from src.routing.NetworkBase import NetworkBase
from src.fleetctrl.pooling.immediate.searchVehicles import veh_search_for_immediate_request,\
                                                            veh_search_for_reservation_request
from src.fleetctrl.pooling.immediate.SelectRV import filter_directionality, filter_least_number_tasks
from src.misc.globals import *
import numpy as np
from typing import Callable, List, Dict, Any, Tuple

import logging
LOG = logging.getLogger(__name__)


def simple_insert(routing_engine : NetworkBase, sim_time : int, veh_obj : SimulationVehicle, orig_veh_plan : VehiclePlan, 
                  new_prq_obj : PlanRequest, std_bt : int, add_bt : int,
                  skip_first_position_insertion : bool=False) -> List[VehiclePlan]:
    """This method inserts the stops for the new request at all possible positions of orig_veh_plan and returns a
    generator that only yields the feasible solutions and None in the other case.

    :param routing_engine: Network
    :param sim_time: current simulation time
    :param veh_obj: simulation vehicle
    :param orig_veh_plan: original vehicle plan
    :param new_prq_obj: new request
    :param std_bt: standard boarding time
    :param add_bt: additional boarding time for an extra request
    :param skip_first_position_insertion: if true, an insertion at the first position of the list_plan_stops is not tried
    :return: generator with feasible new routes
    """
    #LOG.debug("simple_insert: sim_time {} veh {}".format(sim_time, veh_obj))

    # do not consider inactive vehicles
    if veh_obj.status == VRL_STATES.OUT_OF_SERVICE:
        return

    number_stops = len(orig_veh_plan.list_plan_stops)
    # add o_stop
    o_prq_feasible = True   # once max wait time of new_prq_obj is reached, no insertion at later index will be feasible
    tmp_plans : Dict[int, VehiclePlan] = {}  # insertion-index of o_stop -> tmp_VehiclePlan
    prq_o_stop_pos, prq_t_pu_earliest, prq_t_pu_latest = new_prq_obj.get_o_stop_info()
    new_rid_struct = new_prq_obj.get_rid_struct()

    skip_next = -1
    if skip_first_position_insertion:
        skip_next = 0
    first_iterator = range(number_stops)
    for i in first_iterator:
        if not o_prq_feasible:
            break
        if orig_veh_plan.list_plan_stops[i].is_locked() or orig_veh_plan.list_plan_stops[i].is_infeasible_locked():
            continue
        next_o_plan = orig_veh_plan.copy()
        # only allow combination of boarding tasks if the existing one is not locked (has not started)
        if not next_o_plan.list_plan_stops[i].is_locked() and not next_o_plan.list_plan_stops[i].is_locked_end() and prq_o_stop_pos == next_o_plan.list_plan_stops[i].get_pos():
            old_pstop = next_o_plan.list_plan_stops[i]
            new_boarding_list = old_pstop.get_list_boarding_rids() + [new_rid_struct]
            new_boarding_dict = {-1:old_pstop.get_list_alighting_rids(), 1:new_boarding_list}
            ept_dict, lpt_dict, mtt_dict, lat_dict = old_pstop.get_boarding_time_constraint_dicts()
            new_earliest_pickup_time_dict = ept_dict.copy()
            new_earliest_pickup_time_dict[new_rid_struct] = prq_t_pu_earliest
            new_latest_pickup_time_dict = lpt_dict.copy()
            new_latest_pickup_time_dict[new_rid_struct] = prq_t_pu_latest
            stop_duration, _ = old_pstop.get_duration_and_earliest_departure()
            if stop_duration is None:
                stop_duration = std_bt
            else:
                stop_duration += add_bt
            change_nr_pax = old_pstop.get_change_nr_pax()
            change_nr_pax += new_prq_obj.nr_pax
            
            next_o_plan.list_plan_stops[i] = BoardingPlanStop(prq_o_stop_pos, boarding_dict=new_boarding_dict, max_trip_time_dict=mtt_dict.copy(),
                                                              latest_arrival_time_dict=lat_dict.copy(), earliest_pickup_time_dict=new_earliest_pickup_time_dict,
                                                              latest_pickup_time_dict=new_latest_pickup_time_dict, change_nr_pax=change_nr_pax,duration=stop_duration, change_nr_parcels=old_pstop.get_change_nr_parcels())
            #LOG.debug(f"test first if boarding: {next_o_plan}")
            is_feasible = next_o_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine)
            if is_feasible:
                tmp_plans[i] = next_o_plan
                skip_next = i+1
        else:
            # add it before this stop else > planned departure after boarding time
            if i == skip_next:
                continue
            #new_earliest_departure = max(prq_t_pu_earliest+std_bt, sim_time + std_bt)
            new_plan_stop = BoardingPlanStop(prq_o_stop_pos, boarding_dict={1:[new_rid_struct]}, earliest_pickup_time_dict={new_rid_struct : prq_t_pu_earliest},
                                             latest_pickup_time_dict={new_rid_struct : prq_t_pu_latest}, change_nr_pax=new_prq_obj.nr_pax,
                                             duration=std_bt)
            next_o_plan.list_plan_stops[i:i] = [new_plan_stop]
            #LOG.debug(f"test else boarding: {next_o_plan}")
            is_feasible = next_o_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine)
            if is_feasible:
                tmp_plans[i] = next_o_plan
            else:
                # if infeasible: check if o_feasible can be changed to False
                check_info = next_o_plan.get_pax_info(new_rid_struct)
                if check_info:
                    planned_pu = check_info[0]
                    if planned_pu > prq_t_pu_latest:
                        o_prq_feasible = False
                        continue
    # add stop after last stop (waiting at this stop is also possible!)
    if o_prq_feasible and skip_next != number_stops and (number_stops == 0 or not orig_veh_plan.list_plan_stops[-1].is_locked_end()):
        i = number_stops
        #new_earliest_departure = max(prq_t_pu_earliest + std_bt, sim_time + std_bt)
        new_plan_stop = BoardingPlanStop(prq_o_stop_pos, boarding_dict={1:[new_rid_struct]}, earliest_pickup_time_dict={new_rid_struct : prq_t_pu_earliest},
                                            latest_pickup_time_dict={new_rid_struct : prq_t_pu_latest}, change_nr_pax=new_prq_obj.nr_pax,
                                            duration=std_bt)
        next_o_plan = orig_veh_plan.copy()
        next_o_plan.list_plan_stops[i:i] = [new_plan_stop]
        #LOG.debug(f"test at end: {next_o_plan}")
        is_feasible = next_o_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine)
        if is_feasible:
            tmp_plans[i] = next_o_plan

    # add d_stop for all tmp_plans
    d_stop_pos, prq_t_do_latest, prq_max_trip_time = new_prq_obj.get_d_stop_info()  # TODO # the checks with t_do_latest and max_trip_time can be confusing!
    skip_next = -1
    for o_index, tmp_next_plan in tmp_plans.items():
        d_feasible = True  # once latest arrival is reached, no insertion at later index is feasible for current pick-up
        number_stops = len(tmp_next_plan.list_plan_stops)
        # always start checking plans after pick-up of new_prq_obj -> everything before is feasible and stay the same
        next_d_plan = tmp_next_plan.copy()
        init_plan_state = next_d_plan.return_intermediary_plan_state(veh_obj, sim_time, routing_engine, o_index)
        second_iterator = range(o_index + 1, number_stops)
        for j in second_iterator:
            if not d_feasible:
                break
            # reload the plan without d-insertion
            next_d_plan = tmp_next_plan.copy()
            if d_stop_pos == next_d_plan.list_plan_stops[j].get_pos() and not next_d_plan.list_plan_stops[j].is_locked_end():
                old_pstop = next_d_plan.list_plan_stops[j]
                # combine with last stop if it is at the same location (combine constraints)
                new_alighting_list = old_pstop.get_list_alighting_rids() + [new_rid_struct]
                new_boarding_dict = {1:old_pstop.get_list_boarding_rids(), -1:new_alighting_list}
                ept_dict, lpt_dict, mtt_dict, lat_dict = old_pstop.get_boarding_time_constraint_dicts()
                new_max_trip_time_dict = mtt_dict.copy()
                new_max_trip_time_dict[new_rid_struct] = prq_max_trip_time
                stop_duration, _ = old_pstop.get_duration_and_earliest_departure()
                if stop_duration is None:
                    stop_duration = std_bt
                else:
                    stop_duration += add_bt
                change_nr_pax = old_pstop.get_change_nr_pax()
                change_nr_pax -= new_prq_obj.nr_pax
                
                next_d_plan.list_plan_stops[j] = BoardingPlanStop(d_stop_pos, boarding_dict=new_boarding_dict, max_trip_time_dict=new_max_trip_time_dict,
                                                                  latest_arrival_time_dict=lat_dict.copy(), earliest_pickup_time_dict=ept_dict.copy(),
                                                                  latest_pickup_time_dict=lpt_dict.copy(), change_nr_pax=change_nr_pax, change_nr_parcels=old_pstop.get_change_nr_parcels(),
                                                                  duration=stop_duration)

                is_feasible = next_d_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine, init_plan_state)
                if is_feasible:
                    skip_next = j + 1
                    yield next_d_plan
                else:
                    # if infeasible: check if d_feasible can be changed to False
                    planned_pu_do = next_d_plan.get_pax_info(new_rid_struct)
                    if len(planned_pu_do) > 1 and planned_pu_do[1] > prq_t_do_latest:
                        d_feasible = False
            else:
                if j == skip_next:
                    continue
                # add it after this stop else
                new_plan_stop = BoardingPlanStop(d_stop_pos, boarding_dict={-1: [new_rid_struct]}, max_trip_time_dict={new_rid_struct : prq_max_trip_time},
                                                 change_nr_pax=-new_prq_obj.nr_pax, duration=std_bt)
                next_d_plan.list_plan_stops[j:j] = [new_plan_stop]
                # check constraints > yield plan if feasible
                #LOG.debug(f"test with deboarding: {next_d_plan}")
                is_feasible = next_d_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine, init_plan_state)
                if is_feasible:
                    yield next_d_plan
                else:
                    # if infeasible: check if d_feasible can be changed to False
                    planned_pu_do = next_d_plan.get_pax_info(new_rid_struct)
                    if len(planned_pu_do) > 1 and planned_pu_do[1] > prq_t_do_latest:
                        d_feasible = False

        if skip_next != number_stops and not tmp_next_plan.list_plan_stops[-1].is_locked_end():
            next_d_plan = tmp_next_plan.copy()
            j = number_stops
            new_plan_stop = BoardingPlanStop(d_stop_pos, boarding_dict={-1: [new_rid_struct]}, max_trip_time_dict={new_rid_struct : prq_max_trip_time},
                                                change_nr_pax=-new_prq_obj.nr_pax, duration=std_bt)
            next_d_plan.list_plan_stops[j:j] = [new_plan_stop]
            # check constraints > yield plan if feasible
            #LOG.debug(f"test with deboarding: {next_d_plan}")
            is_feasible = next_d_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine, init_plan_state)
            if is_feasible:
                yield next_d_plan
            else:
                # if infeasible: check if d_feasible can be changed to False
                planned_pu_do = next_d_plan.get_pax_info(new_rid_struct)
                if len(planned_pu_do) > 1 and planned_pu_do[1] > prq_t_do_latest:
                    d_feasible = False


def simple_remove(veh_obj : SimulationVehicle, veh_plan : VehiclePlan, remove_rid, sim_time : int, 
                  routing_engine : NetworkBase, obj_function : Callable, rq_dict : Dict[Any, PlanRequest], std_bt : int, add_bt : int) -> VehiclePlan:
    """This function removes the rid 'remove_rid' from the veh_plan.

    :param veh_obj: SimulationVehicle
    :param veh_plan: VehiclePlan
    :param remove_rid: rid to remove
    :param sim_time: current simulation time
    :param routing_engine: RoutingEngine
    :param obj_function: objective function
    :param rq_dict: rid > PlanRequest dictionary
    :param std_bt: standard boarding time
    :param add_bt: additional boarding time from the second boarding request at a stop
    :return: vehicle plan after removal of 'remove_rid'
    :rtype: VehiclePlan
    """
    new_plan_list = []
    rid_found_in_plan_flag = False
    for ps in veh_plan.list_plan_stops:
        new_boarding_dict = {}
        new_max_trip_time_dict = {}
        new_earliest_pickup_time_dict = {}
        new_latest_pickup_time_dict = {}
        change_nr_pax = ps.get_change_nr_pax()
        ept_dict, lpt_dict, mtt_dict, lat_dict = ps.get_boarding_time_constraint_dicts()
        for rid in ps.get_list_boarding_rids():
            if rid != remove_rid:
                try:
                    new_boarding_dict[1].append(rid)
                except:
                    new_boarding_dict[1] = [rid]
                new_earliest_pickup_time_dict[rid] = ept_dict[rid]
                new_latest_pickup_time_dict[rid] = lpt_dict[rid]
            else:
                rid_found_in_plan_flag = True
                change_nr_pax -= rq_dict[rid].nr_pax
        for rid in ps.get_list_alighting_rids():
            if rid != remove_rid:
                try:
                    new_boarding_dict[-1].append(rid)
                except:
                    new_boarding_dict[-1] = [rid]
                new_max_trip_time_dict[rid] = mtt_dict[rid]
            else:
                rid_found_in_plan_flag = True
                change_nr_pax += rq_dict[rid].nr_pax
        if len(new_boarding_dict.keys()) > 0 or ps.is_locked() or ps.is_locked_end():
            dur, _ = ps.get_duration_and_earliest_departure()
            # new_ps = BoardingPlanStop(ps.get_pos(), boarding_dict=new_boarding_dict, max_trip_time_dict=new_max_trip_time_dict,
            #                           earliest_pickup_time_dict=new_earliest_pickup_time_dict, latest_pickup_time_dict=new_latest_pickup_time_dict,
            #                           change_nr_pax=change_nr_pax, duration=dur, locked=ps.is_locked())
            new_ps = PlanStop(ps.get_pos(), boarding_dict=new_boarding_dict, max_trip_time_dict=new_max_trip_time_dict,
                                      earliest_pickup_time_dict=new_earliest_pickup_time_dict, latest_pickup_time_dict=new_latest_pickup_time_dict,
                                      change_nr_pax=change_nr_pax, duration=dur, locked=ps.is_locked(), locked_end=ps.is_locked_end())
            new_plan_list.append(new_ps)
    #LOG.info("simple remove: {}".format([str(x) for x in new_plan_list]))
    external_pax_info = veh_plan.pax_info.copy()
    try:
        del external_pax_info[remove_rid]
    except:
        pass
    new_veh_plan = VehiclePlan(veh_obj, sim_time, routing_engine, new_plan_list, external_pax_info=external_pax_info)
    if not rid_found_in_plan_flag:
        LOG.warning(f"trying to remove rid {remove_rid} from plan which is not part of it!")
        LOG.warning(f"old plan {veh_plan}")
        LOG.warning(f"new plan {new_veh_plan}")
        LOG.warning("")
    return new_veh_plan

def single_insertion(veh_obj_list : List[SimulationVehicle], current_vid_to_vehplan_assignments : Dict[Any, VehiclePlan],
                     prq_to_insert : PlanRequest, obj_function : Callable, routing_engine : NetworkBase,
                     rq_dict : Dict[Any, PlanRequest], sim_time : int, std_bt : int, add_bt : int, check_rv : bool=True,
                     skip_first_position_insertion : bool=False) -> Tuple[Any, VehiclePlan, float]:   
    """ this function inserts a new prq into the current vehicle assignments
    a new solution is only created if
        a) insertion is feasible within time constraints
        b) the global solution cost function can be DECREASED
    :param veh_obj_list: complete id-sorted list of veh_objs
    :param current_vid_to_vehplan_assignments: dict vid -> vehplan of current assignments
    :param prq_to_insert: plan request obj of customer to insert into solution
    :param obj_function: objective function to rate veh_plans (to be MINIMIZED)
    :param routing_engine: network object
    :param rq_dict: rid -> plan request dictionary for all currently active plan requests
    :param sim_time: current simulation time
    :param std_bt: constant boarding time parameter
    :param add_bt: additional boarding time at boarding stop with multiple customers
    :param check_rv: if set to True, the list of possible insertion vehicles is first filtered by routing query for reachability
    :param skip_first_position_insertion: if set to True, the insertion algorithm will not try to insert the request in the first position of the current route (except no route is assigned)
    :return: tuple: (assigned vid, assigned vehplan, change in objective function) | (None, None, 0) of no assignment solution found
    """
    veh_pos_dict = {}
    veh_obj_dict = {}
    for veh_obj in veh_obj_list:
        # do not consider inactive vehicles
        if veh_obj.status == 5:
            continue
        vid = veh_obj.vid
        veh_obj_dict[vid] = veh_obj
        try:
            veh_pos_dict[veh_obj.pos].append(vid)
        except:
            veh_pos_dict[veh_obj.pos] = [vid]

    prq_o_stop_pos, prq_t_pu_earliest, prq_t_pu_latest = prq_to_insert.get_o_stop_info()
    d_stop_pos, prq_t_do_latest, prq_max_trip_time = prq_to_insert.get_d_stop_info()

    max_time_range = prq_t_pu_latest - sim_time
    # LOG.info("insertion:")
    # LOG.info(f"prq {prq_to_insert}")
    # LOG.info(f"at time {sim_time} with max range {max_time_range}")

    #check for vids in time_range
    if len(veh_pos_dict.keys()) > 1 and check_rv:
        rv_routing_results = routing_engine.return_travel_costs_Xto1(veh_pos_dict.keys(), prq_o_stop_pos, max_routes=None,
                                    max_cost_value=max_time_range)
        rv_vids = []
        for vid_pos, _,_,_ in rv_routing_results:
            for vid in veh_pos_dict[vid_pos]:
                rv_vids.append(vid)
    else:
        rv_vids = [veh_obj.vid for veh_obj in veh_obj_list if veh_obj.status != 5]

    # TODO # discuss: 0 only works for objective functions with assignment rewards, not for priority (match if feasible)
    current_best_obj_delta = 0
    current_best_plan = None
    current_best_vid = None

    for vid in rv_vids:
        veh_obj = veh_obj_dict[vid]
        veh_plan_to_insert = current_vid_to_vehplan_assignments.get(vid, None)
        if veh_plan_to_insert is not None:
            skip_first_position_insertion_here = skip_first_position_insertion
            if len(veh_plan_to_insert.list_plan_stops) == 0:
                skip_first_position_insertion_here = False
            old_obj_value = obj_function(sim_time, veh_obj, veh_plan_to_insert, rq_dict, routing_engine) # TODO # dont need to recompute always!
            for new_veh_plan in simple_insert(routing_engine, sim_time, veh_obj, veh_plan_to_insert, prq_to_insert, std_bt, add_bt, skip_first_position_insertion=skip_first_position_insertion_here):
                new_obj_value = obj_function(sim_time, veh_obj, new_veh_plan, rq_dict, routing_engine)
                delta_obj = new_obj_value - old_obj_value
                if delta_obj < current_best_obj_delta:
                    current_best_plan = new_veh_plan
                    current_best_vid = vid
                    current_best_obj_delta = delta_obj
        else:
            rid = prq_to_insert.get_rid_struct()
            pu_stop = BoardingPlanStop(prq_o_stop_pos, boarding_dict={1 : [rid]}, earliest_pickup_time_dict={rid: prq_t_pu_earliest},
                                       latest_pickup_time_dict={rid : prq_t_pu_latest}, duration=std_bt)
            do_stop = BoardingPlanStop(d_stop_pos, boarding_dict={-1 : [rid]}, max_trip_time_dict={rid : prq_max_trip_time}, duration=std_bt)
            new_veh_plan = VehiclePlan(veh_obj, sim_time, routing_engine, [pu_stop, do_stop])
            if not new_veh_plan.is_feasible():
                continue
            new_obj_value = obj_function(sim_time, veh_obj, new_veh_plan, rq_dict, routing_engine)
            #LOG.debug(f"inheu: new obj value: {new_obj_value} | current_best {current_best_obj_delta}")
            if new_obj_value < 0:
                if new_obj_value < current_best_obj_delta:
                    current_best_plan = new_veh_plan
                    current_best_vid = vid
                    current_best_obj_delta = new_obj_value
    #(assigned vid, assigned vehplan, change in objective function)
    return current_best_vid, current_best_plan, current_best_obj_delta


def insertion_with_heuristics(sim_time : int, prq : PlanRequest, fleetctrl : FleetControlBase, force_feasible_assignment : bool=True) -> List[Tuple[Any, VehiclePlan, float]]:
    """This function searches for suitable vehicles and return vehicle plans with insertions. Different heuristics
    depending on whether it is an immediate or reservation request can be triggered. See the respective functions
    for more details.

    :param prq: PlanRequest to insert
    :param fleetctrl: FleetControl instance
    :param force_feasible_assignment: if True, a feasible solution is assigned even with positive control function value
    :return: list of (vid, vehplan, delta_cfv) tuples
    :rtype: list
    """
    if prq.get_reservation_flag():
        return reservation_insertion_with_heuristics(sim_time, prq, fleetctrl, force_feasible_assignment)
    else:
        return immediate_insertion_with_heuristics(sim_time, prq, fleetctrl, force_feasible_assignment)


def immediate_insertion_with_heuristics(sim_time : int, prq : PlanRequest, fleetctrl : FleetControlBase, force_feasible_assignment : bool=True) -> List[Tuple[Any, VehiclePlan, float]]:
    """This function has access to all FleetControl attributes and therefore can trigger different heuristics and
    is easily extendable if new ideas for heuristics are developed.

    Here is a list of currently named heuristics that can be triggered
    1) pre vehicle-search processes
    2) vehicle-search process
        a) G_RH_I_NWS: maximum number of network positions in Dijkstra from which vehicles are considered
    3) pre-insertion vehicle-selection processes
        a) G_RVH_DIR: directionality of currently assigned route compared to vector of prq origin-destination
        b) G_RVH_LWL: selection of vehicles with least workload
    4) insertion processes
        a) G_VPI_KEEP: only return a limited number of vehicle plans per vehicle [default: 1]
        b) trigger other than simple_insert functions to check limited possibilities based on current plan
    5) post insertion vehicle selection processes
        a) G_RA_MAX_RP: return at most one (vid, vehplan, delta_cfv) tuple
        b) G_RA_MAX_VR: only return (vid, vehplan, delta_cfv) tuples for certain number of vehicles
        c) only consider vehicle plans in V2RB satisfying certain criterion
    If any of these scenario input parameters are not set, no vehicles are discarded due to the respective heuristic;
    therefore consistency between old and new versions is guaranteed.

    The function returns a list of (vid, vehplan, delta_cfv) tuples of vehicles with feasible plans and an empty list
    if no vehicle can produce a feasible solution.

    :param prq: PlanRequest to insert
    :param fleetctrl: FleetControl instance
    :param force_feasible_assignment: if True, a feasible solution is assigned even with positive control function value
    :return: list of (vid, vehplan, delta_cfv) tuples
    :rtype: list
    """
    # TODO # think about generalization to use of Parallelization_Manager
    # -> separation into multiple parts or if-clause for Parallelization_Manager in between?

    # 1) pre vehicle-search processes
    excluded_vid = []

    # 2) vehicle-search process
    rv_vehicles, rv_results_dict = veh_search_for_immediate_request(sim_time, prq, fleetctrl, excluded_vid)

    # 3) pre-insertion vehicle-selection processes
    selected_veh = set([])
    #   a) directionality of currently assigned route compared to vector of prq origin-destination
    number_directionality = fleetctrl.rv_heuristics.get(G_RVH_DIR, 0)
    if number_directionality > 0:
        veh_dir = filter_directionality(prq, rv_vehicles, number_directionality, fleetctrl.routing_engine, selected_veh)
        for veh_obj in veh_dir:
            selected_veh.add(veh_obj)
    #   b) selection of vehicles with least workload
    number_least_load = fleetctrl.rv_heuristics.get(G_RVH_LWL, 0)
    if number_least_load > 0:
        veh_ll = filter_least_number_tasks(rv_vehicles, number_least_load, selected_veh)
        for veh_obj in veh_ll:
            selected_veh.add(veh_obj)

    sum_rvh_selection = number_directionality + number_least_load
    if sum_rvh_selection > 0:
        rv_vehicles = list(selected_veh)

    # 4) insertion processes
    insertion_return_list = insert_prq_in_selected_veh_list(rv_vehicles, fleetctrl.veh_plans, prq, fleetctrl.vr_ctrl_f,
                                                            fleetctrl.routing_engine, fleetctrl.rq_dict, sim_time,
                                                            fleetctrl.const_bt, fleetctrl.add_bt,
                                                            force_feasible_assignment, fleetctrl.rv_heuristics)

    # 5) post insertion vehicle selection processes
    max_rq_plans = fleetctrl.rv_heuristics.get(G_RA_MAX_RP, None)
    max_rv_con = fleetctrl.rv_heuristics.get(G_RA_MAX_VR, None)
    return_rv_tuples = []
    rv_con = set([])
    sorted_insertion_return_list = sorted(insertion_return_list, key=lambda x: x[2])
    if max_rq_plans is not None:
        return_rv_tuples = sorted_insertion_return_list[:max_rq_plans]
    elif max_rv_con is not None:
        for rv_tuple in sorted_insertion_return_list:
            vid = rv_tuple[0]
            # add all entries of a vehicle
            if len(rv_con) < max_rv_con or vid in rv_con:
                return_rv_tuples.append(rv_tuple)
                rv_con.add(rv_tuple[0])
    else:
        return_rv_tuples = sorted_insertion_return_list

    return return_rv_tuples


def reservation_insertion_with_heuristics(sim_time : int, prq : PlanRequest, fleetctrl : FleetControlBase, force_feasible_assignment : bool=True, veh_plans : Dict[int, VehiclePlan] = None) -> List[Tuple[Any, VehiclePlan, float]]:
    """This function has access to all FleetControl attributes and therefore can trigger different heuristics and
    is easily extendable if new ideas for heuristics are developed.

    There are heuristics
    1) pre vehicle-search processes
    2) vehicle-search process
        a) G_RH_R_NWS: maximum number of network positions in Dijkstra from which vehicles are considered
        b) G_RH_R_ZSM: activate zone search, which only returns vehicles passing/ending in the zone of prq's origin
        c) G_RH_R_MPS: maximum number of plan stops that are considered (in reversed order)
                            [consider that search process automatically stops when the time difference to prq's
                             earliest pick-up time increases]
    3) pre-insertion vehicle-selection processes
    4) insertion processes
        a) G_VPI_KEEP: only return a limited number of vehicle plans per vehicle [default: 1]
        b) trigger other than simple_insert functions to check limited possibilities based on current plan
    5) post insertion vehicle selection processes
        a) G_RA_MAX_RP: return at most one (vid, vehplan, delta_cfv) tuple
        b) G_RA_MAX_VR: only return (vid, vehplan, delta_cfv) tuples for certain number of vehicles

    If any of these scenario input parameters are not set, no vehicles are discarded due to the respective heuristic;
    therefore consistency between old and new versions is guaranteed.

    It returns a list of (vid, vehplan, delta_cfv) tuples of vehicles with feasible plans and an empty list
    if no vehicle can produce a feasible solution.

    :param prq: PlanRequest (reservation) to insert
    :param fleetctrl: FleetControl instance
    :param force_feasible_assignment: if True, a feasible solution is assigned even with positive control function value
    :param veh_plans: dict vehicle id -> vehicle plan to insert to; if non fleetctrl.veh_plans is used
    :return: list of (vid, vehplan, delta_cfv) tuples
    :rtype: list
    """

    # TODO # think about generalization to use of Parallelization_Manager
    # -> separation into multiple parts or if-clause for Parallelization_Manager in between?
    
    veh_plans_to_insert_to = veh_plans
    if veh_plans_to_insert_to is None:
        veh_plans_to_insert_to = fleetctrl.veh_plans

    # 1) pre vehicle-search processes
    excluded_vid = []

    # 2) vehicle-search process
    dict_veh_to_av_infos = veh_search_for_reservation_request(sim_time, prq, fleetctrl, list_excluded_vid=excluded_vid, veh_plans=veh_plans_to_insert_to)

    # 3) pre-insertion vehicle-selection processes
    rv_vehicles = []
    # for vid, list_search_info_tuple in dict_veh_to_av_infos.items():
    #     for av_pos, av_delta_t, ps_id, later_stops_flag in list_search_info_tuple:
    #         pass
    for vid in dict_veh_to_av_infos.keys():
        rv_vehicles.append(fleetctrl.sim_vehicles[vid])

    # 4) insertion processes
    insertion_return_list = insert_prq_in_selected_veh_list(rv_vehicles, veh_plans_to_insert_to, prq, fleetctrl.vr_ctrl_f,
                                                            fleetctrl.routing_engine, fleetctrl.rq_dict, sim_time,
                                                            fleetctrl.const_bt, fleetctrl.add_bt,
                                                            force_feasible_assignment, fleetctrl.rv_heuristics)

    # 5) post insertion vehicle selection processes
    max_rq_plans = fleetctrl.rv_heuristics.get(G_RA_MAX_RP, None)
    max_rv_con = fleetctrl.rv_heuristics.get(G_RA_MAX_VR, None)
    return_rv_tuples = []
    rv_con = set([])
    sorted_insertion_return_list = sorted(insertion_return_list, key=lambda x: x[2])
    if max_rq_plans is not None:
        return_rv_tuples = sorted_insertion_return_list[:1]
    elif max_rv_con is not None:
        for rv_tuple in sorted_insertion_return_list:
            vid = rv_tuple[0]
            # add all entries of a vehicle
            if len(rv_con) < max_rv_con or vid in rv_con:
                return_rv_tuples.append(rv_tuple)
                rv_con.add(rv_tuple[0])
    else:
        return_rv_tuples = sorted_insertion_return_list

    return return_rv_tuples


def insert_prq_in_selected_veh_list(selected_veh_obj_list : List[SimulationVehicle], vid_to_vehplan_assignments : Dict[Any, VehiclePlan],
                                    prq : PlanRequest, obj_function : Callable, routing_engine : NetworkBase, rq_dict : Dict[Any, PlanRequest],
                                    sim_time : int, const_bt : int, add_bt : int, force_feasible_assignment : bool=True,
                                    insert_heuristic_dict : Dict={}):
    """This method can be used to return a list of RV entries (vid, vehplan, delta_cfv) from a list of selected
    vehicles, whereas only the currently assigned vehicle plan is assumed and different VPI (vehicle plan insertion)
    heuristics can be triggered to limit the insertions.

    :param selected_veh_obj_list: filtered vehicle list for which insertions are performed
    :param vid_to_vehplan_assignments: fleetctrl.veh_plans
    :param prq: PlanRequest to be inserted
    :param obj_function: fleetctrl.vr_ctrl_f
    :param routing_engine: fleetctrl.routing_engine
    :param rq_dict: fleetctrl.rq_dict
    :param sim_time: current simulation time
    :param const_bt: fleetctrl.const_bt
    :param add_bt: fleetctrl.add_bt
    :param force_feasible_assignment: if True, a feasible solution is assigned even with positive control function value
    :param insert_heuristic_dict: dictionary which enables the use of heuristic insertion methods (instead of exhaustive
                search, which is default)
    :return: list of (vid, vehplan, delta_cfv) tuples
    :rtype: list
    """
    # TODO # plans from consistent V2RB formulation, for now assuming only currently assigned vehicle plan
    # set heuristic attributes
    nr_plans_per_vehicle = insert_heuristic_dict.get(G_VPI_KEEP, 1)
    if insert_heuristic_dict.get(G_VPI_SF):
        skip_first_pos = True
    else:
        skip_first_pos = False
    #
    insertion_return_list = []
    for veh_obj in selected_veh_obj_list:
        veh_plan = vid_to_vehplan_assignments[veh_obj.vid]
        current_vehplan_utility = veh_plan.get_utility()
        if current_vehplan_utility is None:
            current_vehplan_utility = obj_function(sim_time, veh_obj, veh_plan, rq_dict, routing_engine)
            veh_plan.set_utility(current_vehplan_utility)
        # use (vid, vehplan, delta_cfv) tuple format from here on
        keep_plans = []
        if force_feasible_assignment:
            threshold = None
        else:
            threshold = 0
        # TODO choice of insert function/heuristics per trigger
        for next_insertion_veh_plan in simple_insert(routing_engine, sim_time, veh_obj, veh_plan, prq,
                                                     const_bt, add_bt, skip_first_position_insertion=skip_first_pos):
            next_insertion_utility = obj_function(sim_time, veh_obj, next_insertion_veh_plan, rq_dict, routing_engine)
            delta_cfv = next_insertion_utility - current_vehplan_utility
            if threshold is None or delta_cfv < threshold:
                keep_plans.append((veh_obj.vid, next_insertion_veh_plan, delta_cfv))
                keep_plans = sorted(keep_plans, key=lambda x: x[2])[:nr_plans_per_vehicle]
                threshold = keep_plans[-1][2]
        insertion_return_list.extend(keep_plans)
    return insertion_return_list

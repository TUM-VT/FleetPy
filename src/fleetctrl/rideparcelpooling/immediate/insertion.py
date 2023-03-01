from __future__ import annotations

from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.planning.VehiclePlan import BoardingPlanStop, VehiclePlan
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.simulation.Vehicles import SimulationVehicle
from src.routing.NetworkBase import NetworkBase
from src.fleetctrl.pooling.immediate.insertion import simple_insert
from src.misc.globals import *
import numpy as np
from typing import Callable, List, Dict, Any, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from src.fleetctrl.RPPFleetControl import ParcelPlanRequest

import logging
LOG = logging.getLogger(__name__)


def simple_insert_parcel(routing_engine : NetworkBase, sim_time : int, veh_obj : SimulationVehicle, orig_veh_plan : VehiclePlan, 
                  new_prq_obj : ParcelPlanRequest, std_bt : int, add_bt : int,
                  skip_first_position_insertion : bool=False, allow_parcel_pu_with_ob_cust = True) -> List[VehiclePlan]:
    """This method inserts the stops for the new parcel at all possible positions of orig_veh_plan and returns a
    generator that only yields the feasible solutions and None in the other case.

    :param routing_engine: Network
    :param sim_time: current simulation time
    :param veh_obj: simulation vehicle
    :param orig_veh_plan: original vehicle plan
    :param new_prq_obj: new parcel request
    :param std_bt: standard boarding time
    :param add_bt: additional boarding time for an extra request
    :param skip_first_position_insertion: if true, an insertion at the first position of the list_plan_stops is not tried
    :param allow_parcel_pu_with_ob_cust: if false, an insertion of a parcel pickup or dropoff is only tried, if currently no person is on board of the vehicle
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
    
    N_persons_ob = veh_obj.get_nr_pax_without_currently_boarding()

    skip_next = -1
    if skip_first_position_insertion:
        skip_next = 0
    first_iterator = range(number_stops)
    for i in first_iterator:
        if not o_prq_feasible:
            break
        if orig_veh_plan.list_plan_stops[i].is_locked() or orig_veh_plan.list_plan_stops[i].is_infeasible_locked():
            N_persons_ob += orig_veh_plan.list_plan_stops[i].get_change_nr_pax()
            continue
        if allow_parcel_pu_with_ob_cust or N_persons_ob == 0 or (not orig_veh_plan.list_plan_stops[i].is_locked() and not orig_veh_plan.list_plan_stops[i].is_locked_end() and prq_o_stop_pos == orig_veh_plan.list_plan_stops[i].get_pos()):
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
                change_nr_parcels = old_pstop.get_change_nr_parcels()
                change_nr_parcels += new_prq_obj.parcel_size
                
                next_o_plan.list_plan_stops[i] = BoardingPlanStop(prq_o_stop_pos, boarding_dict=new_boarding_dict, max_trip_time_dict=mtt_dict.copy(),
                                                                latest_arrival_time_dict=lat_dict.copy(), earliest_pickup_time_dict=new_earliest_pickup_time_dict,
                                                                latest_pickup_time_dict=new_latest_pickup_time_dict, change_nr_pax=change_nr_pax, change_nr_parcels=change_nr_parcels ,duration=stop_duration)
                #LOG.debug(f"test first if boarding: {next_o_plan}")
                is_feasible = next_o_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine)
                if is_feasible:
                    tmp_plans[i] = next_o_plan
                    skip_next = i+1
            else:
                # add it before this stop else > planned departure after boarding time
                if i == skip_next:
                    N_persons_ob += orig_veh_plan.list_plan_stops[i].get_change_nr_pax()
                    continue
                #new_earliest_departure = max(prq_t_pu_earliest+std_bt, sim_time + std_bt)
                new_plan_stop = BoardingPlanStop(prq_o_stop_pos, boarding_dict={1:[new_rid_struct]}, earliest_pickup_time_dict={new_rid_struct : prq_t_pu_earliest},
                                                latest_pickup_time_dict={new_rid_struct : prq_t_pu_latest}, change_nr_parcels=new_prq_obj.parcel_size,
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
                            N_persons_ob += orig_veh_plan.list_plan_stops[i].get_change_nr_pax()
                            continue
        N_persons_ob += orig_veh_plan.list_plan_stops[i].get_change_nr_pax()
    # add stop after last stop (waiting at this stop is also possible!)
    if o_prq_feasible and skip_next != number_stops and (number_stops == 0 or not orig_veh_plan.list_plan_stops[-1].is_locked_end()):
        if allow_parcel_pu_with_ob_cust or N_persons_ob == 0:
            i = number_stops
            #new_earliest_departure = max(prq_t_pu_earliest + std_bt, sim_time + std_bt)
            new_plan_stop = BoardingPlanStop(prq_o_stop_pos, boarding_dict={1:[new_rid_struct]}, earliest_pickup_time_dict={new_rid_struct : prq_t_pu_earliest},
                                                latest_pickup_time_dict={new_rid_struct : prq_t_pu_latest}, change_nr_parcels=new_prq_obj.parcel_size,
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
        N_persons_ob = init_plan_state["c_nr_pax"]
        second_iterator = range(o_index + 1, number_stops)
        for j in second_iterator:
            if not d_feasible:
                break
            if allow_parcel_pu_with_ob_cust or N_persons_ob == 0 or (d_stop_pos == tmp_next_plan.list_plan_stops[j].get_pos() and not tmp_next_plan.list_plan_stops[j].is_locked_end()):
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
                    change_nr_parcels = old_pstop.get_change_nr_parcels()
                    change_nr_parcels -= new_prq_obj.parcel_size
                    
                    next_d_plan.list_plan_stops[j] = BoardingPlanStop(d_stop_pos, boarding_dict=new_boarding_dict, max_trip_time_dict=new_max_trip_time_dict,
                                                                    latest_arrival_time_dict=lat_dict.copy(), earliest_pickup_time_dict=ept_dict.copy(),
                                                                    latest_pickup_time_dict=lpt_dict.copy(), change_nr_pax=change_nr_pax, change_nr_parcels=change_nr_parcels,
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
                        N_persons_ob += tmp_next_plan.list_plan_stops[j].get_change_nr_pax()
                        continue
                    # add it after this stop else
                    new_plan_stop = BoardingPlanStop(d_stop_pos, boarding_dict={-1: [new_rid_struct]}, max_trip_time_dict={new_rid_struct : prq_max_trip_time},
                                                    change_nr_parcels=-new_prq_obj.parcel_size, duration=std_bt)
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
            N_persons_ob += tmp_next_plan.list_plan_stops[j].get_change_nr_pax()

        if skip_next != number_stops and not tmp_next_plan.list_plan_stops[-1].is_locked_end():
            if allow_parcel_pu_with_ob_cust or N_persons_ob == 0:
                next_d_plan = tmp_next_plan.copy()
                j = number_stops
                new_plan_stop = BoardingPlanStop(d_stop_pos, boarding_dict={-1: [new_rid_struct]}, max_trip_time_dict={new_rid_struct : prq_max_trip_time},
                                                    change_nr_parcels=-new_prq_obj.parcel_size, duration=std_bt)
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


def insert_parcel_prq_in_selected_veh_list(selected_veh_obj_list : List[SimulationVehicle], vid_to_vehplan_assignments : Dict[Any, VehiclePlan],
                                    prq : PlanRequest, obj_function : Callable, routing_engine : NetworkBase, rq_dict : Dict[Any, PlanRequest],
                                    sim_time : int, const_bt : int, add_bt : int, force_feasible_assignment : bool=True,
                                    insert_heuristic_dict : Dict={}, allow_parcel_pu_with_ob_cust = True):
    """This method can be used to return a list of RV entries (vid, vehplan, delta_cfv) from a list of selected
    vehicles, whereas only the currently assigned vehicle plan is assumed and different VPI (vehicle plan insertion)
    heuristics can be triggered to limit the insertions.

    :param selected_veh_obj_list: filtered vehicle list for which insertions are performed
    :param vid_to_vehplan_assignments: fleetctrl.veh_plans
    :param prq: ParcelPlanRequest to be inserted
    :param obj_function: fleetctrl.vr_ctrl_f
    :param routing_engine: fleetctrl.routing_engine
    :param rq_dict: fleetctrl.rq_dict
    :param sim_time: current simulation time
    :param const_bt: fleetctrl.const_bt
    :param add_bt: fleetctrl.add_bt
    :param force_feasible_assignment: if True, a feasible solution is assigned even with positive control function value
    :param insert_heuristic_dict: dictionary which enables the use of heuristic insertion methods (instead of exhaustive
                search, which is default)
    :param allow_parcel_pu_with_ob_cust: if false, an insertion of a parcel pickup or dropoff is only tried, if currently no person is on board of the vehicle
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
        current_vehplan_utility = veh_plan.utility
        # use (vid, vehplan, delta_cfv) tuple format from here on
        keep_plans = []
        if force_feasible_assignment:
            threshold = None
        else:
            threshold = 0
        # TODO choice of insert function/heuristics per trigger
        for next_insertion_veh_plan in simple_insert_parcel(routing_engine, sim_time, veh_obj, veh_plan, prq,
                                                     const_bt, add_bt, skip_first_position_insertion=skip_first_pos, 
                                                     allow_parcel_pu_with_ob_cust=allow_parcel_pu_with_ob_cust):
            next_insertion_utility = obj_function(sim_time, veh_obj, next_insertion_veh_plan, rq_dict, routing_engine)
            delta_cfv = next_insertion_utility - current_vehplan_utility
            if threshold is None or delta_cfv < threshold:
                keep_plans.append((veh_obj.vid, next_insertion_veh_plan, delta_cfv))
                keep_plans = sorted(keep_plans, key=lambda x: x[2])[:nr_plans_per_vehicle]
                threshold = keep_plans[-1][2]
        insertion_return_list.extend(keep_plans)
    return insertion_return_list
        
def simple_insert_into_route_with_parcels(routing_engine : NetworkBase, sim_time : int, veh_obj : SimulationVehicle, orig_veh_plan : VehiclePlan, 
                  new_prq_obj : PlanRequest, std_bt : int, add_bt : int,
                  skip_first_position_insertion : bool=False, allow_parcel_pu_with_ob_cust = True) -> List[VehiclePlan]:
    """This method inserts the stops for the new parcel at all possible positions of orig_veh_plan and returns a
    generator that only yields the feasible solutions and None in the other case.

    :param routing_engine: Network
    :param sim_time: current simulation time
    :param veh_obj: simulation vehicle
    :param orig_veh_plan: original vehicle plan
    :param new_prq_obj: new parcel request
    :param std_bt: standard boarding time
    :param add_bt: additional boarding time for an extra request
    :param skip_first_position_insertion: if true, an insertion at the first position of the list_plan_stops is not tried
    :param allow_parcel_pu_with_ob_cust: if false, an insertion of a parcel pickup or dropoff is only tried, if currently no person is on board of the vehicle
    :return: generator with feasible new routes
    """
    LOG.debug("simple_insert_into_route_with_parcels: sim_time {} veh {}".format(sim_time, veh_obj))
    if allow_parcel_pu_with_ob_cust:
        raise EnvironmentError("this parameter is an artefact -> youse simple_insert directly!")
        LOG.debug("here 2")
        return simple_insert(routing_engine, sim_time, veh_obj, orig_veh_plan, new_prq_obj, std_bt, add_bt, skip_first_position_insertion=skip_first_position_insertion)

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
        next_stop_index_with_parcel = o_index + 1
        for j in range(o_index + 1, number_stops):
            ps = next_d_plan.list_plan_stops[j]
            parcel_found = False
            if not (d_stop_pos == ps.get_pos() and not ps.is_locked_end()):
                for rid in ps.get_list_boarding_rids() + ps.get_list_alighting_rids():
                    if type(rid) == str and rid.startswith("p"):
                        parcel_found = True
                        break
            next_stop_index_with_parcel = j
            if parcel_found:
                break
        second_iterator = range(o_index + 1, next_stop_index_with_parcel)
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
            parcel_at_end = False
            for rid in tmp_next_plan.list_plan_stops[-1].get_list_boarding_rids() + tmp_next_plan.list_plan_stops[-1].get_list_alighting_rids():
                if type(rid) == str and rid.startswith("p"):
                    parcel_at_end = True
                    break
            if not parcel_at_end:
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

def insert_prq_in_selected_veh_list_route_with_parcels(selected_veh_obj_list : List[SimulationVehicle], vid_to_vehplan_assignments : Dict[Any, VehiclePlan],
                                    prq : PlanRequest, obj_function : Callable, routing_engine : NetworkBase, rq_dict : Dict[Any, PlanRequest],
                                    sim_time : int, const_bt : int, add_bt : int, force_feasible_assignment : bool=True,
                                    insert_heuristic_dict : Dict={}, allow_parcel_pu_with_ob_cust = True):
    """ this method is used to insert a person prq into a route which might contain parcel prqs
    This method can be used to return a list of RV entries (vid, vehplan, delta_cfv) from a list of selected
    vehicles, whereas only the currently assigned vehicle plan is assumed and different VPI (vehicle plan insertion)
    heuristics can be triggered to limit the insertions.

    :param selected_veh_obj_list: filtered vehicle list for which insertions are performed
    :param vid_to_vehplan_assignments: fleetctrl.veh_plans
    :param prq: ParcelPlanRequest to be inserted
    :param obj_function: fleetctrl.vr_ctrl_f
    :param routing_engine: fleetctrl.routing_engine
    :param rq_dict: fleetctrl.rq_dict
    :param sim_time: current simulation time
    :param const_bt: fleetctrl.const_bt
    :param add_bt: fleetctrl.add_bt
    :param force_feasible_assignment: if True, a feasible solution is assigned even with positive control function value
    :param insert_heuristic_dict: dictionary which enables the use of heuristic insertion methods (instead of exhaustive
                search, which is default)
    :param allow_parcel_pu_with_ob_cust: if false, an insertion of a parcel pickup or dropoff is only tried, if currently no person is on board of the vehicle
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
        if veh_plan.utility is None:
            veh_plan.utility = obj_function(sim_time, veh_obj, veh_plan, rq_dict, routing_engine)
        current_vehplan_utility = veh_plan.utility
        # use (vid, vehplan, delta_cfv) tuple format from here on
        keep_plans = []
        if force_feasible_assignment:
            threshold = None
        else:
            threshold = 0
        # TODO choice of insert function/heuristics per trigger
        if not allow_parcel_pu_with_ob_cust:
            for next_insertion_veh_plan in simple_insert_into_route_with_parcels(routing_engine, sim_time, veh_obj, veh_plan, prq,
                                                        const_bt, add_bt, skip_first_position_insertion=skip_first_pos, 
                                                        allow_parcel_pu_with_ob_cust=allow_parcel_pu_with_ob_cust):
                next_insertion_utility = obj_function(sim_time, veh_obj, next_insertion_veh_plan, rq_dict, routing_engine)
                delta_cfv = next_insertion_utility - current_vehplan_utility
                if threshold is None or delta_cfv < threshold:
                    keep_plans.append((veh_obj.vid, next_insertion_veh_plan, delta_cfv))
                    keep_plans = sorted(keep_plans, key=lambda x: x[2])[:nr_plans_per_vehicle]
                    threshold = keep_plans[-1][2]
            insertion_return_list.extend(keep_plans)
        else:
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
        
        
def insert_parcel_o_in_selected_veh_list_route_with_parcels(selected_veh_obj_list : List[SimulationVehicle], vid_to_vehplan_assignments : Dict[Any, VehiclePlan],
                                    prq : PlanRequest, obj_function : Callable, routing_engine : NetworkBase, rq_dict : Dict[Any, PlanRequest],
                                    sim_time : int, const_bt : int, add_bt : int, force_feasible_assignment : bool=True,
                                    insert_heuristic_dict : Dict={}, allow_parcel_pu_with_ob_cust = True):
    """ This method only inserts the origin of a parcel into the vehicle routes
    This method can be used to return a list of RV entries (vid, vehplan, delta_cfv) from a list of selected
    vehicles, whereas only the currently assigned vehicle plan is assumed and different VPI (vehicle plan insertion)
    heuristics can be triggered to limit the insertions.

    :param selected_veh_obj_list: filtered vehicle list for which insertions are performed
    :param vid_to_vehplan_assignments: fleetctrl.veh_plans
    :param prq: ParcelPlanRequest to be inserted
    :param obj_function: fleetctrl.vr_ctrl_f
    :param routing_engine: fleetctrl.routing_engine
    :param rq_dict: fleetctrl.rq_dict
    :param sim_time: current simulation time
    :param const_bt: fleetctrl.const_bt
    :param add_bt: fleetctrl.add_bt
    :param force_feasible_assignment: if True, a feasible solution is assigned even with positive control function value
    :param insert_heuristic_dict: dictionary which enables the use of heuristic insertion methods (instead of exhaustive
                search, which is default)
    :param allow_parcel_pu_with_ob_cust: if false, an insertion of a parcel pickup or dropoff is only tried, if currently no person is on board of the vehicle
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
        current_vehplan_utility = veh_plan.utility
        # use (vid, vehplan, delta_cfv) tuple format from here on
        keep_plans = []
        if force_feasible_assignment:
            threshold = None
        else:
            threshold = 0
        # TODO choice of insert function/heuristics per trigger
        for next_insertion_veh_plan in simple_insert_parcel_o_into_route(routing_engine, sim_time, veh_obj, veh_plan, prq,
                                                     const_bt, add_bt, skip_first_position_insertion=skip_first_pos, 
                                                     allow_parcel_pu_with_ob_cust=allow_parcel_pu_with_ob_cust):
            next_insertion_utility = obj_function(sim_time, veh_obj, next_insertion_veh_plan, rq_dict, routing_engine)
            delta_cfv = next_insertion_utility - current_vehplan_utility
            if threshold is None or delta_cfv < threshold:
                keep_plans.append((veh_obj.vid, next_insertion_veh_plan, delta_cfv))
                keep_plans = sorted(keep_plans, key=lambda x: x[2])[:nr_plans_per_vehicle]
                threshold = keep_plans[-1][2]
        insertion_return_list.extend(keep_plans)
    return insertion_return_list

def insert_parcel_d_in_selected_veh_list_route_with_parcels(selected_veh_obj_list : List[SimulationVehicle], vid_to_vehplan_assignments : Dict[Any, VehiclePlan],
                                    prq : PlanRequest, obj_function : Callable, routing_engine : NetworkBase, rq_dict : Dict[Any, PlanRequest],
                                    sim_time : int, const_bt : int, add_bt : int, force_feasible_assignment : bool=True,
                                    insert_heuristic_dict : Dict={}, allow_parcel_pu_with_ob_cust = True):
    """ This method only inserts the destination of a parcel into the vehicle routes
    it is assumed that the parcel is allready on board or the pickup is part of the route
    This method can be used to return a list of RV entries (vid, vehplan, delta_cfv) from a list of selected
    vehicles, whereas only the currently assigned vehicle plan is assumed and different VPI (vehicle plan insertion)
    heuristics can be triggered to limit the insertions.

    :param selected_veh_obj_list: filtered vehicle list for which insertions are performed
    :param vid_to_vehplan_assignments: fleetctrl.veh_plans
    :param prq: ParcelPlanRequest to be inserted
    :param obj_function: fleetctrl.vr_ctrl_f
    :param routing_engine: fleetctrl.routing_engine
    :param rq_dict: fleetctrl.rq_dict
    :param sim_time: current simulation time
    :param const_bt: fleetctrl.const_bt
    :param add_bt: fleetctrl.add_bt
    :param force_feasible_assignment: if True, a feasible solution is assigned even with positive control function value
    :param insert_heuristic_dict: dictionary which enables the use of heuristic insertion methods (instead of exhaustive
                search, which is default)
    :param allow_parcel_pu_with_ob_cust: if false, an insertion of a parcel pickup or dropoff is only tried, if currently no person is on board of the vehicle
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
        current_vehplan_utility = veh_plan.utility
        # use (vid, vehplan, delta_cfv) tuple format from here on
        keep_plans = []
        if force_feasible_assignment:
            threshold = None
        else:
            threshold = 0
        # TODO choice of insert function/heuristics per trigger
        for next_insertion_veh_plan in simple_insert_parcel_d_into_route(routing_engine, sim_time, veh_obj, veh_plan, prq,
                                                     const_bt, add_bt, skip_first_position_insertion=skip_first_pos, 
                                                     allow_parcel_pu_with_ob_cust=allow_parcel_pu_with_ob_cust):
            next_insertion_utility = obj_function(sim_time, veh_obj, next_insertion_veh_plan, rq_dict, routing_engine)
            delta_cfv = next_insertion_utility - current_vehplan_utility
            if threshold is None or delta_cfv < threshold:
                keep_plans.append((veh_obj.vid, next_insertion_veh_plan, delta_cfv))
                keep_plans = sorted(keep_plans, key=lambda x: x[2])[:nr_plans_per_vehicle]
                threshold = keep_plans[-1][2]
        insertion_return_list.extend(keep_plans)
    return insertion_return_list

def simple_insert_parcel_o_into_route(routing_engine : NetworkBase, sim_time : int, veh_obj : SimulationVehicle, orig_veh_plan : VehiclePlan, 
                  new_prq_obj : ParcelPlanRequest, std_bt : int, add_bt : int,
                  skip_first_position_insertion : bool=False, allow_parcel_pu_with_ob_cust = True) -> List[VehiclePlan]:
    """This method inserts only the origin stop for the new parcel at all possible positions of orig_veh_plan and returns a
    generator that only yields the feasible solutions and None in the other case.
    it is assumed that the destination stop will be inserted a some time later in the simulation

    :param routing_engine: Network
    :param sim_time: current simulation time
    :param veh_obj: simulation vehicle
    :param orig_veh_plan: original vehicle plan
    :param new_prq_obj: new parcel request
    :param std_bt: standard boarding time
    :param add_bt: additional boarding time for an extra request
    :param skip_first_position_insertion: if true, an insertion at the first position of the list_plan_stops is not tried
    :param allow_parcel_pu_with_ob_cust: if false, an insertion of a parcel pickup or dropoff is only tried, if currently no person is on board of the vehicle
    :return: generator with feasible new routes
    """
    #LOG.debug("simple_insert: sim_time {} veh {}".format(sim_time, veh_obj))

    # do not consider inactive vehicles
    if veh_obj.status == VRL_STATES.OUT_OF_SERVICE:
        return

    number_stops = len(orig_veh_plan.list_plan_stops)
    # add o_stop
    o_prq_feasible = True   # once max wait time of new_prq_obj is reached, no insertion at later index will be feasible
    prq_o_stop_pos, prq_t_pu_earliest, prq_t_pu_latest = new_prq_obj.get_o_stop_info()
    new_rid_struct = new_prq_obj.get_rid_struct()
    
    N_persons_ob = veh_obj.get_nr_pax_without_currently_boarding()

    skip_next = -1
    if skip_first_position_insertion:
        skip_next = 0
    first_iterator = range(number_stops)
    for i in first_iterator:
        if not o_prq_feasible:
            break
        if orig_veh_plan.list_plan_stops[i].is_locked() or orig_veh_plan.list_plan_stops[i].is_infeasible_locked():
            N_persons_ob += orig_veh_plan.list_plan_stops[i].get_change_nr_pax()
            continue
        if allow_parcel_pu_with_ob_cust or N_persons_ob == 0 or (not orig_veh_plan.list_plan_stops[i].is_locked() and not orig_veh_plan.list_plan_stops[i].is_locked_end() and prq_o_stop_pos == orig_veh_plan.list_plan_stops[i].get_pos()):
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
                change_nr_parcels = old_pstop.get_change_nr_parcels()
                change_nr_parcels += new_prq_obj.parcel_size
                
                next_o_plan.list_plan_stops[i] = BoardingPlanStop(prq_o_stop_pos, boarding_dict=new_boarding_dict, max_trip_time_dict=mtt_dict.copy(),
                                                                latest_arrival_time_dict=lat_dict.copy(), earliest_pickup_time_dict=new_earliest_pickup_time_dict,
                                                                latest_pickup_time_dict=new_latest_pickup_time_dict, change_nr_pax=change_nr_pax,duration=stop_duration, change_nr_parcels=change_nr_parcels)
                #LOG.debug(f"test first if boarding: {next_o_plan}")
                is_feasible = next_o_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine)
                if is_feasible:
                    yield next_o_plan
                    skip_next = i+1
            else:
                # add it before this stop else > planned departure after boarding time
                if i == skip_next:
                    N_persons_ob += orig_veh_plan.list_plan_stops[i].get_change_nr_pax()
                    continue
                #new_earliest_departure = max(prq_t_pu_earliest+std_bt, sim_time + std_bt)
                new_plan_stop = BoardingPlanStop(prq_o_stop_pos, boarding_dict={1:[new_rid_struct]}, earliest_pickup_time_dict={new_rid_struct : prq_t_pu_earliest},
                                                latest_pickup_time_dict={new_rid_struct : prq_t_pu_latest}, change_nr_parcels=new_prq_obj.parcel_size,
                                                duration=std_bt)
                next_o_plan.list_plan_stops[i:i] = [new_plan_stop]
                #LOG.debug(f"test else boarding: {next_o_plan}")
                is_feasible = next_o_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine)
                if is_feasible:
                    yield next_o_plan
                else:
                    # if infeasible: check if o_feasible can be changed to False
                    check_info = next_o_plan.get_pax_info(new_rid_struct)
                    if check_info:
                        planned_pu = check_info[0]
                        if planned_pu > prq_t_pu_latest:
                            o_prq_feasible = False
                            N_persons_ob += orig_veh_plan.list_plan_stops[i].get_change_nr_pax()
                            continue
        N_persons_ob += orig_veh_plan.list_plan_stops[i].get_change_nr_pax()
    # add stop after last stop (waiting at this stop is also possible!)
    if o_prq_feasible and skip_next != number_stops and (number_stops == 0 or not orig_veh_plan.list_plan_stops[-1].is_locked_end()):
        if allow_parcel_pu_with_ob_cust or N_persons_ob == 0:
            i = number_stops
            #new_earliest_departure = max(prq_t_pu_earliest + std_bt, sim_time + std_bt)
            new_plan_stop = BoardingPlanStop(prq_o_stop_pos, boarding_dict={1:[new_rid_struct]}, earliest_pickup_time_dict={new_rid_struct : prq_t_pu_earliest},
                                                latest_pickup_time_dict={new_rid_struct : prq_t_pu_latest}, change_nr_parcels=new_prq_obj.parcel_size,
                                                duration=std_bt)
            next_o_plan = orig_veh_plan.copy()
            next_o_plan.list_plan_stops[i:i] = [new_plan_stop]
            #LOG.debug(f"test at end: {next_o_plan}")
            is_feasible = next_o_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine)
            if is_feasible:
                yield next_o_plan
                
def simple_insert_parcel_d_into_route(routing_engine : NetworkBase, sim_time : int, veh_obj : SimulationVehicle, orig_veh_plan : VehiclePlan, 
                  new_prq_obj : ParcelPlanRequest, std_bt : int, add_bt : int,
                  skip_first_position_insertion : bool=False, allow_parcel_pu_with_ob_cust = True) -> List[VehiclePlan]:
    """This method only inserts the destination stop for the new parcel at all possible positions of orig_veh_plan and returns a
    generator that only yields the feasible solutions and None in the other case.
    it is assumed that the parcel is allready on board or will be picked up within the assigned route.

    :param routing_engine: Network
    :param sim_time: current simulation time
    :param veh_obj: simulation vehicle
    :param orig_veh_plan: original vehicle plan
    :param new_prq_obj: new parcel request
    :param std_bt: standard boarding time
    :param add_bt: additional boarding time for an extra request
    :param skip_first_position_insertion: if true, an insertion at the first position of the list_plan_stops is not tried
    :param allow_parcel_pu_with_ob_cust: if false, an insertion of a parcel pickup or dropoff is only tried, if currently no person is on board of the vehicle
    :return: generator with feasible new routes
    """
    #LOG.debug("simple_insert: sim_time {} veh {}".format(sim_time, veh_obj))

    # do not consider inactive vehicles
    if veh_obj.status == VRL_STATES.OUT_OF_SERVICE:
        return

    number_stops = len(orig_veh_plan.list_plan_stops)
    # add o_stop
    o_prq_feasible = True   # once max wait time of new_prq_obj is reached, no insertion at later index will be feasible
    d_stop_pos, prq_t_do_latest, prq_max_trip_time = new_prq_obj.get_d_stop_info()
    new_rid_struct = new_prq_obj.get_rid_struct()
    
    N_persons_ob = veh_obj.get_nr_pax_without_currently_boarding()

    skip_next = -1
    if skip_first_position_insertion:
        skip_next = 0
    #test for ob
    ob = False
    for rq in veh_obj.pax:
        if rq.get_rid_struct() == new_rid_struct:
            ob = True
            break
    start_iter = 0
    if not ob:
        found = False
        for i, ps in enumerate(orig_veh_plan.list_plan_stops):
            for rid in ps.get_list_boarding_rids():
                if rid == new_rid_struct:
                    found = True
                    start_iter = i + 1
                    break
            if found:
                break
    LOG.debug(f"insert parcel d: allready ob {ob} boarding index {start_iter}")             
    first_iterator = range(start_iter, number_stops)
    d_feasible = True  # once latest arrival is reached, no insertion at later index is feasible for current pick-up
    for j in first_iterator:
        if not d_feasible:
            break
        if orig_veh_plan.list_plan_stops[j].is_locked() or orig_veh_plan.list_plan_stops[j].is_infeasible_locked():
            continue
        if allow_parcel_pu_with_ob_cust or N_persons_ob == 0 or (d_stop_pos == orig_veh_plan.list_plan_stops[j].get_pos() and not orig_veh_plan.list_plan_stops[j].is_locked_end()):
            # reload the plan without d-insertion
            next_d_plan = orig_veh_plan.copy()
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
                change_nr_parcels = old_pstop.get_change_nr_parcels()
                change_nr_parcels -= new_prq_obj.parcel_size
                
                next_d_plan.list_plan_stops[j] = BoardingPlanStop(d_stop_pos, boarding_dict=new_boarding_dict, max_trip_time_dict=new_max_trip_time_dict,
                                                                latest_arrival_time_dict=lat_dict.copy(), earliest_pickup_time_dict=ept_dict.copy(),
                                                                latest_pickup_time_dict=lpt_dict.copy(), change_nr_pax=change_nr_pax, change_nr_parcels=change_nr_parcels,
                                                                duration=stop_duration)

                is_feasible = next_d_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine)
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
                    N_persons_ob += orig_veh_plan.list_plan_stops[j].get_change_nr_pax()
                    continue
                # add it after this stop else
                new_plan_stop = BoardingPlanStop(d_stop_pos, boarding_dict={-1: [new_rid_struct]}, max_trip_time_dict={new_rid_struct : prq_max_trip_time},
                                                change_nr_parcels=-new_prq_obj.parcel_size, duration=std_bt)
                next_d_plan.list_plan_stops[j:j] = [new_plan_stop]
                # check constraints > yield plan if feasible
                LOG.debug(f"test with deboarding 1: {next_d_plan}")
                is_feasible = next_d_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine)
                if is_feasible:
                    yield next_d_plan
                else:
                    # if infeasible: check if d_feasible can be changed to False
                    planned_pu_do = next_d_plan.get_pax_info(new_rid_struct)
                    if len(planned_pu_do) > 1 and planned_pu_do[1] > prq_t_do_latest:
                        d_feasible = False
        N_persons_ob += orig_veh_plan.list_plan_stops[j].get_change_nr_pax()

    if len(orig_veh_plan.list_plan_stops) == 0 or (skip_next != number_stops and not orig_veh_plan.list_plan_stops[-1].is_locked_end()):
        if allow_parcel_pu_with_ob_cust or N_persons_ob == 0:
            next_d_plan = orig_veh_plan.copy()
            j = number_stops
            new_plan_stop = BoardingPlanStop(d_stop_pos, boarding_dict={-1: [new_rid_struct]}, max_trip_time_dict={new_rid_struct : prq_max_trip_time},
                                                change_nr_parcels=-new_prq_obj.parcel_size, duration=std_bt)
            next_d_plan.list_plan_stops[j:j] = [new_plan_stop]
            # check constraints > yield plan if feasible
            LOG.debug(f"test with deboarding 2: {next_d_plan}")
            is_feasible = next_d_plan.update_tt_and_check_plan(veh_obj, sim_time, routing_engine)
            if is_feasible:
                yield next_d_plan
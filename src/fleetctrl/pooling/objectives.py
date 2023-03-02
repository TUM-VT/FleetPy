from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Callable
if TYPE_CHECKING:
    from src.fleetctrl.planning.VehiclePlan import VehiclePlan
    from src.fleetctrl.planning.PlanRequest import PlanRequest
    from src.simulation.Vehicles import SimulationVehicle
    from src.routing.NetworkBase import NetworkBase

from src.misc.globals import *

LARGE_INT = 1000000


# -------------------------------------------------------------------------------------------------------------------- #
# main function
# -------------
def return_pooling_objective_function(vr_control_func_dict:dict)->Callable[[int,SimulationVehicle,VehiclePlan,Dict[Any,PlanRequest],NetworkBase],float]:
    """This function generates the control objective functions for vehicle-request assignment in pooling operation.
    The control objective functions contain an assignment reward of LARGE_INT and are to be
    ---------------
    -> minimized <-
    ---------------

    :param vr_control_func_dict: dictionary which has to contain "func_key" as switch between possible functions;
            additional parameters of a function can have additional keys.
    :type vr_control_func_dict: dict
    :return: objective function
    :rtype: function
    """
    func_key = vr_control_func_dict["func_key"]

    # ---------------------------------------------------------------------------------------------------------------- #
    # control objective function definitions
    # --------------------------------------
    if func_key == "total_distance":
        def control_f(simulation_time:float, veh_obj:SimulationVehicle, veh_plan:VehiclePlan, rq_dict:Dict[Any,PlanRequest], routing_engine:NetworkBase)->float:
            """This function evaluates the driven distance according to a vehicle plan.

            :param simulation_time: current simulation time
            :param veh_obj: simulation vehicle object
            :param veh_plan: vehicle plan in question
            :param rq_dict: rq -> Plan request dictionary
            :param routing_engine: for routing queries
            :return: objective function value
            """
            assignment_reward = len(veh_plan.pax_info) * LARGE_INT
            sum_dist = 0
            last_pos = veh_obj.pos
            for ps in veh_plan.list_plan_stops:
                pos = ps.get_pos()
                if pos != last_pos:
                    sum_dist += routing_engine.return_travel_costs_1to1(last_pos, pos)[2]
                    last_pos = pos
            return sum_dist - assignment_reward

    elif func_key == "total_system_time":
        ignore_repo_stop_wt = vr_control_func_dict.get("irswt", False)
        if not ignore_repo_stop_wt:
            def control_f(simulation_time:float, veh_obj:SimulationVehicle, veh_plan:VehiclePlan, rq_dict:Dict[Any,PlanRequest], routing_engine:NetworkBase)->float:
                """This function evaluates the total spent time of a vehicle according to a vehicle plan.

                :param simulation_time: current simulation time
                :param veh_obj: simulation vehicle object
                :param veh_plan: vehicle plan in question
                :param rq_dict: rq -> Plan request dictionary
                :param routing_engine: for routing queries
                :return: objective function value
                """
                assignment_reward = len(veh_plan.pax_info) * LARGE_INT
                # end time (for request assignment purposes) defined by arrival at last stop
                if veh_plan.list_plan_stops:
                    end_time = veh_plan.list_plan_stops[-1].get_planned_arrival_and_departure_time()[0]
                else:
                    end_time = simulation_time
                # utility is negative value of end_time - simulation_time
                return end_time - simulation_time - assignment_reward
        else:
            def control_f(simulation_time:float, veh_obj:SimulationVehicle, veh_plan:VehiclePlan, rq_dict:Dict[Any,PlanRequest], routing_engine:NetworkBase)->float:
                """This function evaluates the total spent time of a vehicle according to a vehicle plan.
                If the last stop is an empty plan stop (i.e. repositioning or reservation) it only evaluates travel time, not waiting time
                (in case of reservation, there can be a huge waiting time which does not reflect the efficiancy of the plan)

                :param simulation_time: current simulation time
                :param veh_obj: simulation vehicle object
                :param veh_plan: vehicle plan in question
                :param rq_dict: rq -> Plan request dictionary
                :param routing_engine: for routing queries
                :return: objective function value
                """
                assignment_reward = len(veh_plan.pax_info) * LARGE_INT
                # end time (for request assignment purposes) defined by arrival at last stop
                if veh_plan.list_plan_stops:
                    if veh_plan.list_plan_stops[-1].is_locked_end():
                        if len(veh_plan.list_plan_stops) > 1:
                            prev_end_time = veh_plan.list_plan_stops[-2].get_planned_arrival_and_departure_time()[0]
                            end_time = prev_end_time + routing_engine.return_travel_costs_1to1(veh_plan.list_plan_stops[-2].get_pos(), veh_plan.list_plan_stops[-1].get_pos())[1]
                        else:
                            end_time = simulation_time + routing_engine.return_travel_costs_1to1(veh_obj.pos, veh_plan.list_plan_stops[-1].get_pos())[1]   
                    else:
                        end_time = veh_plan.list_plan_stops[-1].get_planned_arrival_and_departure_time()[0]
                else:
                    end_time = simulation_time
                # utility is negative value of end_time - simulation_time
                return end_time - simulation_time - assignment_reward

    elif func_key == "user_times":
        def control_f(simulation_time:float, veh_obj:SimulationVehicle, veh_plan:VehiclePlan, rq_dict:Dict[Any,PlanRequest], routing_engine:NetworkBase)->float:
            """This function evaluates the total spent time of a vehicle according to a vehicle plan.

            :param simulation_time: current simulation time
            :param veh_obj: simulation vehicle object
            :param veh_plan: vehicle plan in question
            :param rq_dict: rq -> Plan request dictionary
            :param routing_engine: for routing queries
            :return: objective function value
            """
            assignment_reward = len(veh_plan.pax_info) * LARGE_INT
            # value of time term (treat waiting and in-vehicle time the same)
            sum_user_times = 0
            for rid, boarding_info_list in veh_plan.pax_info.items():
                rq_time = rq_dict[rid].rq_time
                drop_off_time = boarding_info_list[1]
                sum_user_times += (drop_off_time - rq_time)
            # utility is negative value of end_time - simulation_time
            return sum_user_times - simulation_time - assignment_reward
        
    elif func_key == "total_travel_times":
        def control_f(simulation_time:float, veh_obj:SimulationVehicle, veh_plan:VehiclePlan, rq_dict:Dict[Any,PlanRequest], routing_engine:NetworkBase)->float:
            """This function evaluates the total travel time of the vehicle (no waiting/boarding, ...).

            :param simulation_time: current simulation time
            :param veh_obj: simulation vehicle object
            :param veh_plan: vehicle plan in question
            :param rq_dict: rq -> Plan request dictionary
            :param routing_engine: for routing queries
            :return: objective function value
            """
            assignment_reward = len(veh_plan.pax_info) * LARGE_INT
            sum_tt = 0
            last_pos = veh_obj.pos
            for ps in veh_plan.list_plan_stops:
                pos = ps.get_pos()
                if pos != last_pos:
                    sum_tt += routing_engine.return_travel_costs_1to1(last_pos, pos)[1]
                    last_pos = pos
            return sum_tt - assignment_reward

    elif func_key == "system_and_user_time":
        user_weight = vr_control_func_dict["uw"]
        def control_f(simulation_time:float, veh_obj:SimulationVehicle, veh_plan:VehiclePlan, rq_dict:Dict[Any,PlanRequest], routing_engine:NetworkBase)->float:
            """This function calculates the total system time (time the plan takes to be completed) and the total user times
            (time from request till drop off). user times are weighted by the factor given from "uw".

            :param simulation_time: current simulation time
            :param veh_obj: simulation vehicle object
            :param veh_plan: vehicle plan in question
            :param rq_dict: rq -> Plan request dictionary
            :param routing_engine: for routing queries
            :return: objective function value
            """
            assignment_reward = len(veh_plan.pax_info) * LARGE_INT
            # value of time term (treat waiting and in-vehicle time the same)
            sum_user_times = 0
            for rid, boarding_info_list in veh_plan.pax_info.items():
                rq_time = rq_dict[rid].rq_time
                drop_off_time = boarding_info_list[1]
                sum_user_times += (drop_off_time - rq_time)

            if veh_plan.list_plan_stops:
                end_time = veh_plan.list_plan_stops[-1].get_planned_arrival_and_departure_time()[0]
            else:
                end_time = simulation_time
            system_time = end_time - simulation_time
            #print("vid {}-> vids {} | simulation time {} : ctrf: sys time {} | user time {} | both {} | all {}".format(veh_obj.vid, veh_plan.get_dedicated_rid_list(), simulation_time, system_time, sum_user_times, system_time + user_weight*sum_user_times, system_time + user_weight*sum_user_times - assignment_reward))
            return system_time + user_weight*sum_user_times - assignment_reward

    elif func_key == "distance_and_user_times":
        traveler_vot = vr_control_func_dict["vot"]

        def control_f(simulation_time:float, veh_obj:SimulationVehicle, veh_plan:VehiclePlan, rq_dict:Dict[Any,PlanRequest], routing_engine:NetworkBase)->float:
            """This function combines the total driving costs and the value of customer time.

            :param simulation_time: current simulation time
            :param veh_obj: simulation vehicle object
            :param veh_plan: vehicle plan in question
            :param rq_dict: rq -> Plan request dictionary
            :param routing_engine: for routing queries
            :return: objective function value
            """
            assignment_reward = len(veh_plan.pax_info) * LARGE_INT
            # distance term
            sum_dist = 0
            last_pos = veh_obj.pos
            for ps in veh_plan.list_plan_stops:
                pos = ps.get_pos()
                if pos != last_pos:
                    sum_dist += routing_engine.return_travel_costs_1to1(last_pos, pos)[2]
                    last_pos = pos
            # value of time term (treat waiting and in-vehicle time the same)
            sum_user_times = 0
            for rid, boarding_info_list in veh_plan.pax_info.items():
                rq_time = rq_dict[rid].rq_time
                drop_off_time = boarding_info_list[1]
                sum_user_times += (drop_off_time - rq_time)
            # vehicle costs are taken from simulation vehicle (cent per meter)
            # value of travel time is scenario input (cent per second)
            return sum_dist * veh_obj.distance_cost + sum_user_times * traveler_vot - assignment_reward

    elif func_key == "distance_and_user_times_man":
        traveler_vot = vr_control_func_dict["vot"]
        distance_cost = vr_control_func_dict["dc"]

        def control_f(simulation_time:float, veh_obj:SimulationVehicle, veh_plan:VehiclePlan, rq_dict:Dict[Any,PlanRequest], routing_engine:NetworkBase)->float:
            """This function combines the total driving costs and the value of customer time.

            :param simulation_time: current simulation time
            :param veh_obj: simulation vehicle object
            :param veh_plan: vehicle plan in question
            :param rq_dict: rq -> Plan request dictionary
            :param routing_engine: for routing queries
            :return: objective function value
            """
            assignment_reward = len(veh_plan.pax_info) * LARGE_INT
            # distance term
            sum_dist = 0
            last_pos = veh_obj.pos
            for ps in veh_plan.list_plan_stops:
                pos = ps.get_pos()
                if pos != last_pos:
                    sum_dist += routing_engine.return_travel_costs_1to1(last_pos, pos)[2]
                    last_pos = pos
            # value of time term (treat waiting and in-vehicle time the same)
            sum_user_times = 0
            for rid, boarding_info_list in veh_plan.pax_info.items():
                rq_time = rq_dict[rid].rq_time
                drop_off_time = boarding_info_list[1]
                sum_user_times += (drop_off_time - rq_time)
            # vehicle costs are taken from simulation vehicle (cent per meter)
            # value of travel time is scenario input (cent per second)
            return sum_dist * distance_cost + sum_user_times * traveler_vot - assignment_reward

    elif func_key == "distance_and_user_times_with_walk":
        traveler_vot = vr_control_func_dict["vot"]

        def control_f(simulation_time:float, veh_obj:SimulationVehicle, veh_plan:VehiclePlan, rq_dict:Dict[Any,PlanRequest], routing_engine:NetworkBase)->float:
            """This function combines the total driving costs and the value of customer time.

            :param simulation_time: current simulation time
            :param veh_obj: simulation vehicle object
            :param veh_plan: vehicle plan in question
            :param rq_dict: rq -> Plan request dictionary
            :param routing_engine: for routing queries
            :return: objective function value
            """
            assignment_reward = len(veh_plan.pax_info) * LARGE_INT
            # distance term
            sum_dist = 0
            last_pos = veh_obj.pos
            for ps in veh_plan.list_plan_stops:
                pos = ps.get_pos()
                if pos != last_pos:
                    sum_dist += routing_engine.return_travel_costs_1to1(last_pos, pos)[2]
                    last_pos = pos
            # value of time term (treat waiting and in-vehicle time the same)
            sum_user_times = 0
            for rid, boarding_info_list in veh_plan.pax_info.items():
                rq_time = rq_dict[rid].rq_time
                walking_time_end = rq_dict[rid].walking_time_end    #walking time start allready included in interval rq-time -> drop_off_time
                drop_off_time = boarding_info_list[1]
                sum_user_times += (drop_off_time - rq_time) + walking_time_end
            # vehicle costs are taken from simulation vehicle (cent per meter)
            # value of travel time is scenario input (cent per second)
            return sum_dist * veh_obj.distance_cost + sum_user_times * traveler_vot - assignment_reward

    elif func_key == "distance_and_user_vehicle_times":
        traveler_vot = vr_control_func_dict["vot"]

        def control_f(simulation_time:float, veh_obj:SimulationVehicle, veh_plan:VehiclePlan, rq_dict:Dict[Any,PlanRequest], routing_engine:NetworkBase)->float:
            """This function combines the total driving costs, the value of customer time and vehicle waiting time.

            :param simulation_time: current simulation time
            :param veh_obj: simulation vehicle object
            :param veh_plan: vehicle plan in question
            :param rq_dict: rq -> Plan request dictionary
            :param routing_engine: for routing queries
            :return: objective function value
            """
            assignment_reward = len(veh_plan.pax_info) * LARGE_INT
            # distance term
            sum_dist = 0
            sum_veh_wait = 0
            last_pos = veh_obj.pos
            for ps in veh_plan.list_plan_stops:
                # penalize VehiclePlan if it is not planned through and raise warning
                arrival_time, departure_time = ps.get_planned_arrival_and_departure_time()
                pos = ps.get_pos()
                if arrival_time is None:
                    assignment_reward = -len(veh_plan.pax_info) * LARGE_INT
                else:
                    # compute vehicle stop time if departure is already planned
                    if departure_time is not None:
                        veh_wait_time = departure_time - arrival_time
                        if veh_wait_time > 0:
                            sum_veh_wait += veh_wait_time
                if pos != last_pos:
                    sum_dist += routing_engine.return_travel_costs_1to1(last_pos, pos)[2]
                    last_pos = pos
            # value of time term (treat waiting and in-vehicle time the same)
            sum_user_times = 0
            for rid, boarding_info_list in veh_plan.pax_info.items():
                rq_time = rq_dict[rid].rq_time
                drop_off_time = boarding_info_list[1]
                sum_user_times += (drop_off_time - rq_time)
            # vehicle costs are taken from simulation vehicle (cent per meter)
            # value of travel time is scenario input (cent per second)
            return sum_dist * veh_obj.distance_cost + (sum_user_times + sum_veh_wait) * traveler_vot - assignment_reward

    elif func_key == "sys_time_and_detour_time":
        detour_weight = vr_control_func_dict["dtw"]

        def control_f(simulation_time:float, veh_obj:SimulationVehicle, veh_plan:VehiclePlan, rq_dict:Dict[Any,PlanRequest], routing_engine:NetworkBase)->float:
            """This function is used to minimize system time and detour time
            the parameter dtw can be used to weight the detour time part dtw == 0 means pure minimizaion of system time
            dtw == 1 means sytem_time + detour time/customer
    
            :param simulation_time: current simulation time
            :param veh_obj: simulation vehicle object
            :param veh_plan: vehicle plan in question
            :param rq_dict: rq -> Plan request dictionary
            :param routing_engine: for routing queries
            :return: objective function value
            """
            assignment_reward = len(veh_plan.pax_info) * LARGE_INT
            # end time (for request assignment purposes) defined by arrival at last stop
            if veh_plan.list_plan_stops:
                end_time = veh_plan.list_plan_stops[-1].get_planned_arrival_and_departure_time()[0]
            else:
                end_time = simulation_time
            sys_time = end_time - simulation_time
            s_det = 0
            for rid, boarding_info_list in veh_plan.pax_info.items():
                det = boarding_info_list[1] - boarding_info_list[0] - rq_dict[rid].init_direct_tt
                s_det += det
            if len(veh_plan.pax_info) > 0:
                return sys_time + s_det*detour_weight - assignment_reward
            else:
                return sys_time - assignment_reward
    
    elif func_key == "IRS_study_standard":
        def control_f(simulation_time:float, veh_obj:SimulationVehicle, veh_plan:VehiclePlan, rq_dict:Dict[Any,PlanRequest], routing_engine:NetworkBase)->float:
            """This function tries to minimize the waiting time of unlocked users.

            :param simulation_time: current simulation time
            :param veh_obj: simulation vehicle object
            :param veh_plan: vehicle plan in question
            :param rq_dict: rq -> Plan request dictionary
            :param routing_engine: for routing queries
            :return: objective function value
            """
            sum_user_wait_times = 0
            assignment_reward = 0
            sum_dist = 0
            last_pos = veh_obj.pos
            for ps in veh_plan.list_plan_stops:
                pos = ps.get_pos()
                if pos != last_pos and len(ps.get_list_boarding_rids()):
                    sum_dist += routing_engine.return_travel_costs_1to1(last_pos, pos)[2]
                    last_pos = pos
            for rid, boarding_info_list in veh_plan.pax_info.items():
                prq = rq_dict[rid]
                if prq.pu_time is None:
                    rq_time = rq_dict[rid].rq_time
                    pick_up_time = boarding_info_list[0]
                    sum_user_wait_times += (pick_up_time - rq_time)
                    if prq.is_locked():
                        assignment_reward += LARGE_INT*10000
                    elif prq.status < G_PRQS_LOCKED:
                        assignment_reward += float(LARGE_INT)
                        # TODO # Cplex only allows floats. Therefore this workaround.
                        #  For some reason LARGE_INT*10000 does not seem to be a problem though...
                    else:
                        assignment_reward += LARGE_INT*100
            # 4 is the empirically found parameter to weigh saved dist against saved waiting time
            return sum_dist + sum_user_wait_times - assignment_reward

    elif func_key == "soft_time_windows":
        soft_tw_rewards = {"locked": LARGE_INT * 1000, "in time window": LARGE_INT + 100}
        def control_f(simulation_time:float, veh_obj:SimulationVehicle, veh_plan:VehiclePlan, rq_dict:Dict[Any,PlanRequest], routing_engine:NetworkBase)->float:
            """This function tries to minimize the waiting time of unlocked users. It penalizes assignments that imply
            pickups outside of the respective requests' time windows.

            :param simulation_time: current simulation time
            :param veh_obj: simulation vehicle object
            :param veh_plan: vehicle plan in question
            :param rq_dict: rq -> Plan request dictionary
            :param routing_engine: for routing queries
            :return: objective function value
            """
            sum_user_wait_times = 0
            assignment_reward = 0
            sum_dist = 0
            last_pos = veh_obj.pos
            for ps in veh_plan.list_plan_stops:
                pos = ps.get_pos()
                if pos != last_pos:
                    sum_dist += routing_engine.return_travel_costs_1to1(last_pos, pos)[2]
                    last_pos = pos
            for rid, boarding_info_list in veh_plan.pax_info.items():
                prq = rq_dict[rid]
                if prq.pu_time is None:
                    rq_time = rq_dict[rid].rq_time
                    pick_up_time = boarding_info_list[0]
                    _, t_pu_earliest, t_pu_latest = rq_dict[rid].get_soft_o_stop_info()
                    sum_user_wait_times += (pick_up_time - rq_time)
                    if prq.is_locked():
                        assignment_reward += soft_tw_rewards["locked"]
                    elif t_pu_earliest <= pick_up_time <= t_pu_latest:
                        assignment_reward += soft_tw_rewards["in time window"]
                    else:
                        assignment_reward += LARGE_INT
            return sum_dist + sum_user_wait_times - assignment_reward

    else:
        raise IOError(f"Did not find valid request assignment control objective string."
                      f" Please check the input parameter {G_OP_VR_CTRL_F}!")

    return control_f

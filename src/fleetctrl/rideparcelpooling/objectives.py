from src.misc.globals import *

LARGE_INT = 1000000


# -------------------------------------------------------------------------------------------------------------------- #
# main function
# -------------
def return_parcel_pooling_objective_function(vr_control_func_dict):
    """This function generates the control objective functions for vehicle-request assignment in ride-parcel-pooling operation.
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
        def control_f(simulation_time, veh_obj, veh_plan, rq_dict, routing_engine):
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
        def control_f(simulation_time, veh_obj, veh_plan, rq_dict, routing_engine):
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
        
    elif func_key == "user_times":
        def control_f(simulation_time, veh_obj, veh_plan, rq_dict, routing_engine):
            """This function evaluates the total of customers from request to drop off according to a vehicle plan.

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
            return sum_user_times - assignment_reward
        
    else:
        raise EnvironmentError("this function is not defined for ride parcel pooling! {}".format(vr_control_func_dict))
    
    return control_f
from src.misc.globals import *

LARGE_INT = 1000000


# -------------------------------------------------------------------------------------------------------------------- #
# main function
# -------------
def return_reservation_driving_leg_objective_function(vr_control_func_dict):
    """This function generates a method to compute the objective function between two driving legs with no customer on-board 
    for the reservation use case. as input the same control function as for the objective function to rate vehicle plans is used.
    it only applies terms of this function two compute the objective function to travel between two locations
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
        def control_f(simulation_time, routing_engine, start_pos, end_pos, start_time, end_time):
            """This function evaluates the driven distance according to a vehicle plan.

            :param simulation_time: current simulation time
            :param routing_engine: for routing queries
            :param start_pos: start position
            :param end_pos: end position
            :param start_time: start time of the leg
            :param end_time: end time of the leg
            :return: objective function value
            """
            return routing_engine.return_travel_costs_1to1(start_pos, end_pos)[2]
        
    elif func_key == "total_system_time":
        def control_f(simulation_time, routing_engine, start_pos, end_pos, start_time, end_time):
            """This function evaluates the driven distance according to a vehicle plan.

            :param simulation_time: current simulation time
            :param routing_engine: for routing queries
            :param start_pos: start position
            :param end_pos: end position
            :param start_time: start time of the leg
            :param end_time: end time of the leg
            :return: objective function value
            """
            return (end_time - start_time)
        
    elif func_key == "user_times":
        def control_f(simulation_time, routing_engine, start_pos, end_pos, start_time, end_time):
            """This function evaluates the driven distance according to a vehicle plan.

            :param simulation_time: current simulation time
            :param routing_engine: for routing queries
            :param start_pos: start position
            :param end_pos: end position
            :param start_time: start time of the leg
            :param end_time: end time of the leg
            :param veh_obj: vehicle object
            :return: objective function value
            """
            return 0
        
    elif func_key == "system_and_user_time":
        user_weight = vr_control_func_dict["uw"]
        def control_f(simulation_time, routing_engine, start_pos, end_pos, start_time, end_time):
            """This function evaluates the driven distance according to a vehicle plan.

            :param simulation_time: current simulation time
            :param routing_engine: for routing queries
            :param start_pos: start position
            :param end_pos: end position
            :param start_time: start time of the leg
            :param end_time: end time of the leg
            :return: objective function value
            """

            #print("vid {}-> vids {} | simulation time {} : ctrf: sys time {} | user time {} | both {} | all {}".format(veh_obj.vid, veh_plan.get_dedicated_rid_list(), simulation_time, system_time, sum_user_times, system_time + user_weight*sum_user_times, system_time + user_weight*sum_user_times - assignment_reward))
            return (end_time - start_time)

    elif func_key == "distance_and_user_times":
        raise EnvironmentError(f"This objective is not useable for reservation because vehicle objects are not accessible -> use 'distance_and_user_times_man' instead!")

    elif func_key == "distance_and_user_times_man" or func_key == "distance_and_user_times_man_with_reservation":
        traveler_vot = vr_control_func_dict["vot"]
        distance_cost = vr_control_func_dict["dc"]

        def control_f(simulation_time, routing_engine, start_pos, end_pos, start_time, end_time):
            """This function evaluates the driven distance according to a vehicle plan.

            :param simulation_time: current simulation time
            :param routing_engine: for routing queries
            :param start_pos: start position
            :param end_pos: end position
            :param start_time: start time of the leg
            :param end_time: end time of the leg
            :return: objective function value
            """
            return routing_engine.return_travel_costs_1to1(start_pos, end_pos)[2] * distance_cost

    elif func_key == "distance_and_user_times_with_walk":
        raise EnvironmentError(f"This objective is not useable for reservation because vehicle objects are not accessible -> use 'distance_and_user_times_man' instead!")

    elif func_key == "distance_and_user_vehicle_times":
        raise EnvironmentError(f"This objective is not useable for reservation because vehicle objects are not accessible -> use 'distance_and_user_times_man' instead!")

    elif func_key == "sys_time_and_detour_time":
        detour_weight = vr_control_func_dict["dtw"]
        def control_f(simulation_time, routing_engine, start_pos, end_pos, start_time, end_time):
            """This function evaluates the driven distance according to a vehicle plan.

            :param simulation_time: current simulation time
            :param routing_engine: for routing queries
            :param start_pos: start position
            :param end_pos: end position
            :param start_time: start time of the leg
            :param end_time: end time of the leg
            :return: objective function value
            """

            #print("vid {}-> vids {} | simulation time {} : ctrf: sys time {} | user time {} | both {} | all {}".format(veh_obj.vid, veh_plan.get_dedicated_rid_list(), simulation_time, system_time, sum_user_times, system_time + user_weight*sum_user_times, system_time + user_weight*sum_user_times - assignment_reward))
            return (end_time - start_time)
        
    elif func_key == "total_travel_times":
        def control_f(simulation_time, routing_engine, start_pos, end_pos, start_time, end_time):
            """This function evaluates the driven distance according to a vehicle plan.

            :param simulation_time: current simulation time
            :param routing_engine: for routing queries
            :param start_pos: start position
            :param end_pos: end position
            :param start_time: start time of the leg
            :param end_time: end time of the leg
            :return: objective function value
            """
            return routing_engine.return_travel_costs_1to1(start_pos, end_pos)[1]
        
    else:
        raise EnvironmentError("this function is not defined for reservations! {}".format(vr_control_func_dict))
    
    return control_f

def checkRRcomptibility(plan_rq_1, plan_rq_2, routing_engine, constant_boarding_time, dynamic_boarding_time = 0):
    """This method checks the compatibility of the origins and destinations of two requests. Independent of any vehicle availability, many
    combinations can be excluded with this method right away.
    
    :param plan_rq_1: PlanRequest-obj of first request
    :param plan_rq_2: PlanRequest-obj of second request
    :param routing_engine: reference to routing engine
    :param constant_boarding_time: constant boarding duration
    :param dynamic_boarding_time: duration needed for boarding for each person at stop
    :return: True, if RR-compatible; False else
    """

    if checkRRcomptibilityInOrder(plan_rq_1, plan_rq_2, routing_engine, constant_boarding_time, dynamic_boarding_time = dynamic_boarding_time):
        return True
    elif checkRRcomptibilityInOrder(plan_rq_2, plan_rq_1, routing_engine, constant_boarding_time, dynamic_boarding_time = dynamic_boarding_time):
        return True
    else:
        return False

def checkRRcomptibilityInOrder(plan_rq_1, plan_rq_2, routing_engine, constant_boarding_time, dynamic_boarding_time = 0):
    """This method checks the compatibility of the origins and destinations of two requests. Independent of any vehicle availability, many
    combinations can be excluded with this method right away.

    In this function the order or the two requests is important! plan_rq_1 is always picked up first!
    
    :param plan_rq_1: PlanRequest-obj of first request
    :param plan_rq_2: PlanRequest-obj of second request
    :param routing_engine: reference to routing engine
    :param constant_boarding_time: constant boarding duration
    :param dynamic_boarding_time: duration needed for boarding for each person at stop
    :return: True, if RR-compatible; False else
    """

    # TODO: implement dynamic boarding time!
    if dynamic_boarding_time != 0:
        print("dynamic boarding time not implemented in RR!")
        raise NotImplementedError

    # TODO # waring: not max trip time considered!

    o_pos_1, earliest_pu_1, latest_pu_1 = plan_rq_1.get_o_stop_info()
    d_pos_1, latest_do_1, max_trip_time_1 = plan_rq_1.get_d_stop_info() #self.d_pos, self.t_do_latest, self.max_trip_time
    o_pos_2, earliest_pu_2, latest_pu_2 = plan_rq_2.get_o_stop_info()
    d_pos_2, latest_do_2, max_trip_time_2 = plan_rq_2.get_d_stop_info()
    #start with plan_rq_1
    #schedule
    if d_pos_1 != o_pos_2:
        if earliest_pu_1 + constant_boarding_time + routing_engine.return_travel_costs_1to1(o_pos_1, d_pos_1)[1] + constant_boarding_time + routing_engine.return_travel_costs_1to1(d_pos_1, o_pos_2)[1] < latest_pu_2:
            return True
    else:
        if earliest_pu_1 + constant_boarding_time + routing_engine.return_travel_costs_1to1(o_pos_1, d_pos_1)[1]:
            return True
    #o_pos_1 -> o_pos_2
    t_next = -1
    if o_pos_1 == o_pos_2:
        e_pu = max(earliest_pu_1, earliest_pu_2)
        l_pu =  min(latest_pu_1, latest_pu_2)
        if e_pu < l_pu:
            t_next = e_pu
    else:
        e_pu = max(earliest_pu_1 + constant_boarding_time + routing_engine.return_travel_costs_1to1(o_pos_1, o_pos_2)[1], earliest_pu_2)
        if e_pu < latest_pu_2:
            t_next = e_pu
    if t_next > 0:
        t_next_2 = -1
        # o_pos_2 -> d_pos_1
        if o_pos_2 == d_pos_1 and t_next < latest_do_1:
            t_next_2 = t_next
        else:
            t_next_2 = t_next + constant_boarding_time + routing_engine.return_travel_costs_1to1(o_pos_2, d_pos_1)[1]
            if t_next_2 > latest_do_1:
                t_next_2 = -1
        if t_next_2 > 0:
            #d_pos_1 -> d_pos_2
            if d_pos_1 == d_pos_2 and t_next_2 < latest_do_2:
                return True
            elif t_next_2 + constant_boarding_time + routing_engine.return_travel_costs_1to1(d_pos_1, d_pos_2)[1] < latest_do_2:
                return True

        # o_pos_2 -> d_pos_2
        t_next_2 = t_next + constant_boarding_time + routing_engine.return_travel_costs_1to1(o_pos_2, d_pos_2)[1]
        if t_next_2 > latest_do_2:
            t_next_2 = -1
        if t_next_2 > 0:
            #d_pos_2 -> d_pos_1
            if d_pos_1 == d_pos_2 and t_next_2 < latest_do_2:
                return True
            elif t_next_2 + constant_boarding_time + routing_engine.return_travel_costs_1to1(d_pos_2, d_pos_1)[1] < latest_do_1:
                return True

    return False

def get_assigned_rids_from_vehplan(vehicle_plan):
    """ this function returns a list of assigned request ids from the corresponding vehicle plan
    :param vehicle_plan: corresponding vehicle plan object
    :return: list of request ids that are part of the vehicle plan
    """
    if vehicle_plan is None:
        return []
    return list(vehicle_plan.pax_info.keys())

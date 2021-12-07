import numpy as np


def filter_directionality(prq, list_veh_obj, nr_best_veh, routing_engine, selected_veh):
    """This function filters the nr_best_veh from list_veh_obj according to the difference in directionality between
    request origin and destination and planned vehicle route. Vehicles with final position equal to current position
    are treated like driving perpendicular to the request direction.

    :param prq: plan request in question
    :param list_veh_obj: list of simulation vehicle objects in question
    :param nr_best_veh: number of vehicles that should be returned
    :param routing_engine: required to get coordinates from network positions
    :param selected_veh: set of vehicles that were already selected by another heuristic
    :return: list of simulation vehicle objects
    """
    if nr_best_veh >= len(list_veh_obj):
        return list_veh_obj
    prq_o_coord = np.array(routing_engine.return_position_coordinates(prq.o_pos))
    prq_d_coord = np.array(routing_engine.return_position_coordinates(prq.d_pos))
    tmp_diff = prq_d_coord - prq_o_coord
    prq_norm_vec = tmp_diff / np.sqrt(np.dot(tmp_diff, tmp_diff))
    tmp_list_veh_val = []
    for veh_obj in list_veh_obj:
        # vehicle already selected by other heuristic
        if veh_obj in selected_veh:
            continue
        if veh_obj.assigned_route:
            veh_coord = np.array(routing_engine.return_position_coordinates(veh_obj.pos))
            last_position = veh_obj.assigned_route[-1].destination_pos
            veh_final_coord = np.array(routing_engine.return_position_coordinates(last_position))
            if not np.array_equal(veh_coord, veh_final_coord):
                tmp_diff = veh_final_coord - veh_coord
                veh_norm_vec = tmp_diff / np.sqrt(np.dot(tmp_diff, tmp_diff))
            else:
                veh_norm_vec = np.array([0, 0])
        else:
            veh_norm_vec = np.array([0, 0])
        val = np.dot(prq_norm_vec, veh_norm_vec)
        tmp_list_veh_val.append((val, veh_obj.vid, veh_obj))
    # sort and return
    tmp_list_veh_val.sort(reverse=True)
    return_list = [x[2] for x in tmp_list_veh_val[:nr_best_veh]]
    return return_list


def filter_least_number_tasks(list_veh_obj, nr_best_veh, selected_veh):
    """This function filters the vehicles according to the number of assigned tasks.

    :param list_veh_obj: list of simulation vehicle objects in question (sorted by distance from destination)
    :param nr_best_veh: number of vehicles that should be returned
    :param selected_veh: set of vehicles that were already selected by another heuristic
    :return: list of simulation vehicle objects
    """
    if len(list_veh_obj) <= nr_best_veh:
        return list_veh_obj
    return_list = []
    remaining_dict = {}
    for veh_obj in list_veh_obj:
        if veh_obj in selected_veh:
            continue
        if not veh_obj.assigned_route:
            return_list.append(veh_obj)
        else:
            nr_vrl = len(veh_obj.assigned_route)
            try:
                remaining_dict[nr_vrl].append(veh_obj)
            except KeyError:
                remaining_dict[nr_vrl] = [veh_obj]
        if len(return_list) == nr_best_veh:
            break
    if len(return_list) < nr_best_veh:
        break_outer_loop = False
        for nr_vrl in sorted(remaining_dict.keys()):
            for veh_obj in remaining_dict[nr_vrl]:
                return_list.append(veh_obj)
                if len(return_list) == nr_best_veh:
                    break_outer_loop = True
                    break
            if break_outer_loop:
                break
    return return_list


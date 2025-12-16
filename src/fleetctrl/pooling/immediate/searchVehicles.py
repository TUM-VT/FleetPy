import logging
from src.misc.globals import *
LOG = logging.getLogger(__name__)
WARNING_FOR_SEARCH_RADIUS = 900


def veh_search_for_immediate_request(sim_time, prq, fleetctrl, list_excluded_vids=[]):
    """This function can be used to find pooling vehicles for an immediate service of a request.

    :param sim_time: current simulation time
    :param prq: PlanRequest to be considered
    :param fleetctrl: FleeControl instance
    :param list_excluded_vids: possible list of vehicles that are excluded from prior heuristic
    :return: list of vehicle objects considered for assignment, routing_results_dict ( (o_pos, d_pos) -> (cfv, tt, dis))
    :rtype: tuple of list of SimulationVehicle, dict
    """
    if sim_time != fleetctrl.pos_veh_dict_time or not fleetctrl.pos_veh_dict:
        veh_locations_to_vid = {}
        for vid, veh_obj in enumerate(fleetctrl.sim_vehicles):
            # do not consider inactive vehicles
            if veh_obj.status == 5 or vid in list_excluded_vids:
                continue
            try:
                veh_locations_to_vid[veh_obj.pos].append(vid)
            except:
                veh_locations_to_vid[veh_obj.pos] = [vid]
        fleetctrl.pos_veh_dict_time = sim_time
        fleetctrl.pos_veh_dict = veh_locations_to_vid

    # stop criteria: search radius and possibly max_routes
    prq_o_stop_pos, prq_t_pu_earliest, prq_t_pu_latest = prq.get_o_stop_info()
    sr = prq_t_pu_latest - sim_time
    max_routes = fleetctrl.rv_heuristics.get(G_RH_I_NWS)

    # backwards Dijkstra
    rv_routing = fleetctrl.routing_engine.return_travel_costs_Xto1(fleetctrl.pos_veh_dict.keys(), prq_o_stop_pos,
                                                                   max_routes=max_routes, max_cost_value=sr)
    rv_vehicles = []
    rv_results_dict = {}
    for vid_pos, cfv,tt,dis in rv_routing:
        for vid in fleetctrl.pos_veh_dict[vid_pos]:
            rv_vehicles.append(fleetctrl.sim_vehicles[vid])
            rv_results_dict[(prq_o_stop_pos, vid_pos)] = (cfv, tt, dis)
    return rv_vehicles, rv_results_dict


def veh_search_for_reservation_request(sim_time, prq, fleetctrl, list_excluded_vid=[], veh_plans = None):
    """This function returns a list of vehicles that should be considered for insertion of a plan request
    whose pick up is far in the future.

    Approach: search process on node level (not position)
    1) go through vehicles, create an availability dictionary based on current assignment, where a vehicle can create
          multiple entries based on the stops planned_arrival_time
          a) with last stop node before prq_t_pu_earliest: av_delta_t = prq_ept - planned_arrival_time
          b) all stop nodes during prq_t_pu_earliest and prq_t_pu_latest: av_delta_t = 0
          c) with first stop node after prq_t_pu_latest: av_delta_t = planned_arrival_time - prq_lpt
    2) find nodes to be considered
          P1) create Dijkstra to find 'max_routes' nearby positions
          P2) use the zone as selection criterion
    3) go through the availability dictionary and create a dictionary with a list of triples per vehicle:
          key: vehicle_id
          a) av_pos: built from first entry of current or plan_stop position [used for spatial filtering]
          b) av_delta_t: see step 1) for definition
          c) ps_id: list-index of plan_stop
          d) later_stops_flag: is True if there are later stops in the VehiclePlan

    :param sim_time: current simulation time
    :param prq: corresponding plan request
    :param list_excluded_vid: possible list of vehicles that are excluded from prior heuristic
    :param veh_plans: dict vehicle id -> assigned plan (if None -> fleetctrl.veh_plans is used)
    :return: dict: vid -> list of (av_pos, av_delta_t, ps_id, later_stops_flag) tuples
    :rtype: dict
    """
    # 0) request information and heuristics
    prq_o_stop_pos, prq_t_pu_earliest, prq_t_pu_latest = prq.get_o_stop_info()
    if fleetctrl.rv_heuristics.get(G_RH_R_ZSM):
        # use zone search method to find positions within zone of prq origin
        search_method = "P2"
        max_routes = None
    else:
        # use Dijkstra search method to find positions near prq origin; number max routes is necessary input then!
        search_method = "P1"
        max_routes = fleetctrl.rv_heuristics[G_RH_R_NWS]
    max_ps = fleetctrl.rv_heuristics.get(G_RH_R_MPS)

    # 1) vehicle availability
    pos_to_vid_time = {}  # pos -> {}: vid -> (delta_t, later_stops_flag)
    for vid, veh_obj in enumerate(fleetctrl.sim_vehicles):
        # do not consider inactive vehicles
        if veh_obj.status == 5 or vid in list_excluded_vid:
            continue
        # only use currently assigned vehicle plan, create a flag whether the stop is the latest considered stop
        if veh_plans is None:
            a_vehplan = fleetctrl.veh_plans[vid]
        else:
            a_vehplan = veh_plans[vid]
        later_stops_flag = False
        # idle vehicles
        if not a_vehplan.list_plan_stops:
            c_pos = (veh_obj.pos[0], None, None)
            delta_t = prq_t_pu_earliest - sim_time
            try:
                pos_to_vid_time[c_pos][vid] = (delta_t, -1, later_stops_flag)
            except KeyError:
                pos_to_vid_time[c_pos] = {vid: (delta_t, -1, later_stops_flag)}
        # non-idle vehicles
        delta_t_max = None
        len_lps = len(a_vehplan.list_plan_stops)
        ps_counter = 0
        for ps in reversed(a_vehplan.list_plan_stops):
            # do not consider vehicles that are about to become inactive
            if ps.is_inactive():
                break
            ps_pos = (ps.get_pos()[0], None, None)
            ps_time = ps.get_planned_arrival_and_departure_time()[0]
            # cannot work with incomplete vehicle plans
            if ps_time is None:
                continue
            if ps_time < prq_t_pu_earliest:
                delta_t = prq_t_pu_earliest - ps_time
            elif ps_time > prq_t_pu_latest:
                delta_t = ps_time - prq_t_pu_latest
            else:
                delta_t = 0
            # temporal difference is increasing -> break
            if delta_t_max is not None and delta_t > delta_t_max:
                break
            delta_t_max = delta_t
            ps_id = len_lps - ps_counter - 1
            try:
                pos_to_vid_time[ps_pos][vid] = (delta_t, ps_id, later_stops_flag)
            except KeyError:
                pos_to_vid_time[ps_pos] = {vid: (delta_t, ps_id, later_stops_flag)}
            ps_counter += 1
            later_stops_flag = True
            # enough plan stops are considered -> break
            if max_ps is not None and ps_counter >= max_ps:
                break

    # 2) find nodes to be considered
    if search_method == "P2" and fleetctrl.zones is not None:
        zone_id = fleetctrl.zones.get_zone_from_pos(prq_o_stop_pos)
        considered_node_ids = fleetctrl.zones.get_all_nodes_in_zone(zone_id)
        considered_pos = [(nid, None, None) for nid in considered_node_ids]
    else:
        sr = prq_t_pu_latest - sim_time
        if sr > WARNING_FOR_SEARCH_RADIUS and max_routes is None:
            prt_str = f"Using Dijkstra without 'max_routes' limit and time search radius of {sr} for" \
                      f" reservation/last-mile requests!"
            LOG.warning(prt_str)
        routing_results = fleetctrl.routing_engine.return_travel_costs_Xto1(pos_to_vid_time.keys(), prq_o_stop_pos,
                                                                            max_routes=max_routes, max_cost_value=sr)
        considered_pos = [pos_route_info_tuple[0] for pos_route_info_tuple in routing_results]

    # 3) create output
    vid_infos = {}
    for pos in considered_pos:
        for vid, av_infos in pos_to_vid_time[pos].items():
            new_info_tuple = (pos, *av_infos)
            try:
                vid_infos[vid].append(new_info_tuple)
            except KeyError:
                vid_infos[vid] = [new_info_tuple]
    return vid_infos


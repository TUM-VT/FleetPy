import os
import sys
import glob
import numpy as np
import pandas as pd
import importlib

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(MAIN_DIR)
from src.FleetSimulationBase import load_scenario_inputs
from src.evaluation.publictransport import public_transport_evaluation
from src.evaluation.multipleboardingpoints_eval import multiple_boarding_points
from src.infra.Zoning import ZoneSystem
from src.misc.globals import *
from src.routing.NetworkBase import return_position_from_str

EURO_PER_TON_OF_CO2 = 145 # from BVWP2030 Modulhandbuch (page 113)
EMISSION_CPG = 145 * 100 / 1000**2
ENERGY_EMISSIONS = 112 # g/kWh from https://www.swm.de/dam/swm/dokumente/geschaeftskunden/broschuere-strom-erdgas-gk.pdf
PV_G_CO2_KM = 130 # g/km from https://www.ris-muenchen.de/RII/RII/DOK/ANTRAG/2337762.pdf with 60:38 benzin vs diesel
OD_MATRIX_TIME_INTERVAL = 3600


def offerEntriesToDict(offer_entry):
    #print(offer_entry)
    #return decode_config_str(offer_entry) # geht nicht durch 2x ":"
    offer_dict = {}
    try:
        offer_strs = offer_entry.split("|")
    except:
        return {}
    for offer_str in offer_strs:
        x = offer_str.split(":")
        key = x[0]
        vals = ":".join(x[1:])
        if len(vals) == 0:
            continue
        try:
            key = int(key)
        except:
            pass
        offer_dict[key] = {}
        for offer_entries in vals.split(";"):
            try:
                k2, v2 = offer_entries.split(":")
            except:
                continue
            try:
                v2 = float(v2)
            except:
                pass
            offer_dict[key][k2] = v2
    return offer_dict

def get_directory_dict(scenario_parameters):
    """
    This function provides the correct paths to certain data according to the specified data directory structure.
    :param scenario_parameters: simulation input (pandas series)
    :return: dictionary with paths to the respective data directories
    """
    #study_name = scenario_parameters[G_STUDY_NAME]
    scenario_name = scenario_parameters[G_SCENARIO_NAME]
    network_name = scenario_parameters[G_NETWORK_NAME]
    demand_name = scenario_parameters[G_DEMAND_NAME]
    zone_name = scenario_parameters.get(G_ZONE_SYSTEM_NAME, None)
    fc_type = scenario_parameters.get(G_FC_TYPE, None)
    fc_t_res = scenario_parameters.get(G_FC_TR, None)
    gtfs_name = scenario_parameters.get(G_GTFS_NAME, None)
    infra_name = scenario_parameters.get(G_INFRA_NAME, None)
    #
    dirs = {}
    dirs[G_DIR_MAIN] = MAIN_DIR
    dirs[G_DIR_DATA] = os.path.join(dirs[G_DIR_MAIN], "data")
    #dirs[G_DIR_OUTPUT] = os.path.join(dirs[G_DIR_MAIN], "studies", study_name, "results", scenario_name)
    dirs[G_DIR_NETWORK] = os.path.join(dirs[G_DIR_DATA], "networks", network_name)
    dirs[G_DIR_VEH] = os.path.join(dirs[G_DIR_DATA], "vehicles")
    dirs[G_DIR_FCTRL] = os.path.join(dirs[G_DIR_DATA], "fleetctrl")
    #dirs[G_DIR_DEMAND] = os.path.join(dirs[G_DIR_DATA], "demand", demand_name, "matched", network_name)
    if zone_name is not None:
        dirs[G_DIR_ZONES] = os.path.join(dirs[G_DIR_DATA], "zones", zone_name, network_name)
        if fc_type is not None and fc_t_res is not None:
            dirs[G_DIR_FC] = os.path.join(dirs[G_DIR_DATA], "demand", demand_name, "aggregated", zone_name, str(fc_t_res))
    if gtfs_name is not None:
        dirs[G_DIR_PT] = os.path.join(dirs[G_DIR_DATA], "pubtrans", gtfs_name)
    if infra_name is not None:
        dirs[G_DIR_INFRA] = os.path.join(dirs[G_DIR_DATA], "infra", infra_name, network_name)
    return dirs

def create_vehicle_type_db(vehicle_data_dir):
    list_veh_data_f = glob.glob(f"{vehicle_data_dir}/*csv")
    veh_type_db = {}    # veh_type -> veh_type_data
    for f in list_veh_data_f:
        veh_type_name = os.path.basename(f)[:-4]
        veh_type_data = pd.read_csv(f, index_col=0, squeeze=True)
        veh_type_db[veh_type_name] = {}
        for k, v in veh_type_data.items():
            try:
                veh_type_db[veh_type_name][k] = float(v)
            except:
                veh_type_db[veh_type_name][k] = v
        veh_type_db[veh_type_name][G_VTYPE_NAME] = veh_type_data.name
    print(veh_type_db)
    return veh_type_db


def create_zone_od_matrices(op_df, scenario_parameters, dir_names):
    """This method creates a data frame with od matrix entries from the ridepooling trips.

    :param op_df: operator data frame
    :param dir_names: directory name
    :return: od-matrix data frame
    """
    zones = ZoneSystem(dir_names[G_DIR_ZONES], scenario_parameters, dir_names)
    trips_df = op_df[op_df["driven_distance"] > 0]
    trips_df["o_zone"] = trips_df.apply(lambda x: zones.get_zone_from_pos(return_position_from_str(x[G_VR_LEG_START_POS])), axis = 1)
    trips_df["d_zone"] = trips_df.apply(lambda x: zones.get_zone_from_pos(return_position_from_str(x[G_VR_LEG_END_POS])), axis = 1)
    trips_df["time_interval"] = np.floor(trips_df[G_VR_LEG_START_TIME] / OD_MATRIX_TIME_INTERVAL)
    trips_df["nr_trips"] = 1
    #
    agg_df = trips_df.groupby(["time_interval", "o_zone", "d_zone"]).aggregate({"nr_trips": sum})
    return agg_df

def get_rides_per_vehicle_revenue_hour(op_df, op_customer_df, scenario_parameter, operator_attributes, dir_names):
    n_vehicles = len(op_df["vehicle_id"].unique())
    veh_rev_hours = 0
    t_start = scenario_parameter[G_SIM_START_TIME]
    t_end = op_df["end_time"].max()
    print(t_start)
    if dir_names.get(G_DIR_FCTRL) is not None:
        time_active_vehicles_f = operator_attributes.get(G_OP_ACT_FLEET_SIZE, None)
        if time_active_vehicles_f is not None:
            time_active_vehicles_p = os.path.join(dir_names[G_DIR_FCTRL], "elastic_fleet_size", time_active_vehicles_f)
            v_curve = pd.read_csv(time_active_vehicles_p)
            n0 = 0
            for k, entries in v_curve.iterrows():
                t = entries["time"]
                frac = entries["share_active_fleet_size"]
                n0 = n_vehicles*frac
                if t < t_start:
                    continue
                if t < t_end:
                    veh_rev_hours += n0*(t - t_start)
                    t_start = t
                else:
                    veh_rev_hours += n0*(t_end - t_start)
                    break
        else:
            veh_rev_hours = n_vehicles*(t_end - t_start)
    else:
        veh_rev_hours = n_vehicles*(t_end - t_start)
    rides = op_customer_df[G_RQ_PAX].sum()
    return rides/veh_rev_hours*3600

def standard_evaluation(output_dir, print_comments=False):
    """This function runs a standard evaluation over a scenario output directory.

    :param output_dir: scenario output directory
    :param print_comments: print some comments about status in between
    """
    scenario_parameters, list_operator_attributes, dir_names = load_scenario_inputs(output_dir)
    if not os.path.isdir(dir_names[G_DIR_MAIN]):
        dir_names = get_directory_dict(scenario_parameters)

    # vehicle type data
    veh_type_db = create_vehicle_type_db(dir_names[G_DIR_VEH])
    veh_type_stats = pd.read_csv(os.path.join(output_dir, "2_vehicle_types.csv"))

    if print_comments:
        print(f"Evaluating {scenario_parameters[G_SCENARIO_NAME]}\nReading user stats ...")
    user_stats = pd.read_csv(os.path.join(output_dir, "1_user-stats.csv"))
    if print_comments:
        print(f"\t shape of user stats: {user_stats.shape}")

    result_dict_list = []
    operator_names = []

    # add passengers columns where necessary
    rq_id_pax = {}  # rq_id -> nr_pax
    if G_RQ_ID not in user_stats.columns:
        user_stats[G_RQ_ID] = 1

    rq_id_to_offer_dict = {}    # rq_id -> op_id -> offer
    op_id_to_offer_dict = {}    # op_id -> rq_id -> offer
    active_offer_parameters = {}
    for key, entries in user_stats.iterrows():
        #rq_id = entries[G_RQ_ID]
        rq_id_pax[key] = entries[G_RQ_PAX]
        offer_entry = entries[G_RQ_OFFERS]
        offer = offerEntriesToDict(offer_entry)
        # if rq_id_to_offer_dict.get(rq_id) is not None:
        #     print("WARNING: request id {} not unique".format(rq_id))
        rq_id_to_offer_dict[key] = offer
        for op_id, op_offer in offer.items():
            try:
                op_id_to_offer_dict[op_id][key] = op_offer
            except KeyError:
                op_id_to_offer_dict[op_id] = {key : op_offer}
            for offer_param in op_offer.keys():
                active_offer_parameters[offer_param] = 1

    number_users = user_stats.shape[0]
    number_total_travelers = user_stats[G_RQ_PAX].sum()

    for op_id, op_users in user_stats.groupby(G_RQ_OP_ID):
        op_name = "?"

        op_number_users = op_users.shape[0]
        op_number_pax = op_users[G_RQ_PAX].sum()
        op_created_offers = len(op_id_to_offer_dict.get(op_id, {}).keys())
        op_modal_split_rq = float(op_number_users)/number_users
        op_modal_split = float(op_number_pax)/number_total_travelers
        op_rel_created_offers = float(op_created_offers)/number_users*100.0
        op_avg_utility = np.nan
        if G_RQ_C_UTIL in op_users.columns:
            op_avg_utility = op_users[G_RQ_C_UTIL].sum()/op_number_users
        op_mono_modal_pax = op_number_pax
        op_firstmile_pax = np.nan
        op_lastmile_pax = np.nan
        if G_RQ_MODAL_STATE in op_users.columns:
            op_mono_modal_pax = op_users[op_users[G_RQ_MODAL_STATE] == G_RQ_STATE_MONOMODAL][G_RQ_PAX].sum()
            op_firstmile_pax = op_users[op_users[G_RQ_MODAL_STATE] == G_RQ_STATE_FIRSTMILE][G_RQ_PAX].sum()
            op_lastmile_pax = op_users[op_users[G_RQ_MODAL_STATE] == G_RQ_STATE_LASTMILE][G_RQ_PAX].sum()

        result_dict = {"operator_id": op_id, 
                       "number users": op_number_users,
                       "number travelers": op_number_pax,
                       "modal split": op_modal_split,
                       "modal split rq": op_modal_split_rq,
                       r'% created offers': op_rel_created_offers,
                       "utility" : op_avg_utility,
                       "number monomodal travelers" : op_mono_modal_pax,
                       "number firstmile travelers" : op_firstmile_pax,
                       "number lastmile travelers" : op_lastmile_pax}

        # base user_values
        op_user_sum_travel_time = np.nan
        op_revenue = np.nan
        op_avg_wait_time = np.nan
        op_avg_travel_time = np.nan
        op_avg_travel_distance = np.nan
        op_sum_direct_travel_distance = np.nan
        op_sum_direct_travel_distance_pick = np.nan
        op_avg_detour_time = np.nan 
        op_avg_rel_detour = np.nan  
        op_avg_detour_time_pick = np.nan
        op_avg_rel_detour_pick = np.nan
        op_avg_direct_travel_time_pick = np.nan

        # base fleet values
        op_fleet_utilization = np.nan
        op_total_km = np.nan
        op_distance_avg_occupancy = np.nan
        op_empty_vkm = np.nan
        op_repositioning_vkm = np.nan
        op_saved_distance = np.nan

        op_toll = np.nan
        op_parking_cost = np.nan
        op_fix_costs = np.nan
        op_var_costs = np.nan
        op_co2 = np.nan
        op_ext_em_costs = np.nan

        if op_id >= 0:  #AMoD
            op_name = "MoD_{}".format(int(op_id))
            operator_attributes = list_operator_attributes[int(op_id)]
            boarding_time = operator_attributes["op_const_boarding_time"]
            if print_comments:
                print("Loading AMoD vehicle data ...")
            try:
                op_vehicle_df = pd.read_csv(os.path.join(output_dir, f"2-{int(op_id)}_op-stats.csv"))
            except FileNotFoundError:
                op_vehicle_df = pd.DataFrame([], columns=[G_V_OP_ID, G_V_VID, G_VR_STATUS, G_VR_LOCKED, G_VR_LEG_START_TIME,
                                                        G_VR_LEG_END_TIME, G_VR_LEG_START_POS, G_VR_LEG_END_POS,
                                                        G_VR_LEG_DISTANCE, G_VR_LEG_START_SOC, G_VR_LEG_END_SOC,
                                                        G_VR_TOLL, G_VR_OB_RID, G_VR_BOARDING_RID, G_VR_ALIGHTING_RID,
                                                        G_VR_NODE_LIST, G_VR_REPLAY_ROUTE])

            if print_comments:
                print(f"\t shape of vehicle stats: {op_vehicle_df.shape}\n\t ... processing AMoD vehicle data")
            
            if G_RQ_DO in op_users.columns and G_RQ_PU in op_users.columns:
                op_user_sum_travel_time = op_users[G_RQ_DO].sum() - op_users[G_RQ_PU].sum()
            if G_RQ_FARE in op_users.columns:
                op_revenue = op_users[G_RQ_FARE].sum()
            if G_RQ_PU in op_users.columns and G_RQ_TIME in op_users.columns:
                op_avg_wait_time = (op_users[G_RQ_PU].sum() - op_users[G_RQ_TIME].sum()) / op_number_users
            if not np.isnan(op_user_sum_travel_time):
                op_avg_travel_time = op_user_sum_travel_time / op_number_users

            if not np.isnan(op_user_sum_travel_time) and G_RQ_DRT in op_users.columns:
                op_avg_detour_time = (op_user_sum_travel_time - op_users[G_RQ_DRT].sum())/op_number_users - boarding_time
            if not np.isnan(op_user_sum_travel_time) and G_RQ_DRT in op_users.columns:
                rel_det_series = (op_users[G_RQ_DO] - op_users[G_RQ_PU] - boarding_time - op_users[G_RQ_DRT])/op_users[G_RQ_DRT]
                op_avg_rel_detour = rel_det_series.sum()/op_number_users * 100.0
            if not np.isnan(op_user_sum_travel_time) and active_offer_parameters.get(G_OFFER_DRIVE) and op_id_to_offer_dict.get(op_id):
                offered_travel_time = {ind : op_id_to_offer_dict[op_id][ind][G_OFFER_DRIVE] for ind in
                                                     op_users.index.to_list()}
                op_users["offered_travel_time"] = pd.Series(offered_travel_time)
                op_avg_detour_time_pick = (op_user_sum_travel_time - op_users["offered_travel_time"].sum())/op_number_users - boarding_time
                op_avg_rel_detour_pick = 100 * (op_users[G_RQ_DO] - op_users[G_RQ_PU] - op_users["offered_travel_time"])/(op_users["offered_travel_time"] - boarding_time)
                op_avg_rel_detour_pick = op_avg_rel_detour_pick.mean()
                op_avg_direct_travel_time_pick = op_users["offered_travel_time"].mean()

            if G_RQ_DRD in op_users.columns:
                op_sum_direct_travel_distance = op_users[G_RQ_DRD].sum() / 1000.0
            if op_id_to_offer_dict.get(op_id) and active_offer_parameters.get(G_OFFER_DIST):
                op_sum_direct_travel_distance_pick = sum([op_id_to_offer_dict[op_id][rq_id][G_OFFER_DIST] for rq_id in
                                                     op_users[G_RQ_ID]]) / 1000.0

            # vehicle stats
            # -------------
            n_vehicles = sum([x for x in operator_attributes[G_OP_FLEET].values()])
            sim_end_time = scenario_parameters["end_time"]
            simulation_time = scenario_parameters["end_time"] - scenario_parameters["start_time"]
            try:
                # correct utilization: do not consider tasks after simulation end time
                op_vehicle_df["VRL_end_sim_end_time"] = np.minimum(op_vehicle_df[G_VR_LEG_END_TIME], sim_end_time)
                op_vehicle_df["VRL_start_sim_end_time"] = np.minimum(op_vehicle_df[G_VR_LEG_START_TIME], sim_end_time)
                op_fleet_utilization = 100 * (op_vehicle_df["VRL_end_sim_end_time"].sum() -
                                              op_vehicle_df["VRL_start_sim_end_time"].sum()) /\
                                       (n_vehicles * simulation_time)
            except ZeroDivisionError:
                pass
            op_total_km = op_vehicle_df[G_VR_LEG_DISTANCE].sum()/1000.0

            def weight_ob_rq(entries):
                if pd.isnull(entries[G_VR_OB_RID]):
                    return 0.0
                else:
                    number_ob_rq = len(str(entries[G_VR_OB_RID]).split(";"))
                    return number_ob_rq * entries[G_VR_LEG_DISTANCE]

            def weight_ob_pax(entries):
                try:
                    return entries[G_VR_NR_PAX] * entries[G_VR_LEG_DISTANCE]
                except:
                    return 0.0

            try:
                op_vehicle_df["weighted_ob_rq"] = op_vehicle_df.apply(weight_ob_rq, axis = 1)
                op_vehicle_df["weighted_ob_pax"] = op_vehicle_df.apply(weight_ob_pax, axis=1)
                op_distance_avg_rq = op_vehicle_df["weighted_ob_rq"].sum() / op_vehicle_df[G_VR_LEG_DISTANCE].sum()
                op_distance_avg_occupancy = op_vehicle_df["weighted_ob_pax"].sum() / op_vehicle_df[G_VR_LEG_DISTANCE].sum()
                op_rq_distance = op_vehicle_df["weighted_ob_rq"].sum()
                op_per_distance = op_vehicle_df["weighted_ob_pax"].sum()
                empty_df = op_vehicle_df[op_vehicle_df[G_VR_OB_RID].isnull()]
                op_empty_vkm = empty_df[G_VR_LEG_DISTANCE].sum()/1000.0/op_total_km*100.0
                op_repositioning_vkm = empty_df[empty_df[G_VR_STATUS] == "reposition"][G_VR_LEG_DISTANCE].sum()/op_total_km*100.0/1000.0
                if G_VR_TOLL in op_vehicle_df.columns:
                    op_toll = op_vehicle_df[G_VR_TOLL].sum()
            except ZeroDivisionError:
                op_vehicle_df["weighted_ob_rq"] = 0
                op_vehicle_df["weighted_ob_pax"] = 0
                op_distance_avg_rq = 0
                op_distance_avg_occupancy = 0
                op_empty_vkm = 0
                op_repositioning_vkm = 0
                op_toll = 0
                op_rq_distance = 0
                op_per_distance = 0

            # OD trips
            # --------
            # check if zone system exists
            if dir_names.get(G_DIR_ZONES):
                agg_df = create_zone_od_matrices(op_vehicle_df, scenario_parameters, dir_names)
                agg_f = os.path.join(output_dir, "OD_matrices.csv")
                agg_df.to_csv(agg_f)

            if not np.isnan(op_total_km) and not np.isnan(op_sum_direct_travel_distance):
                op_saved_distance = (op_sum_direct_travel_distance - op_total_km)/op_sum_direct_travel_distance * 100.0

            # by vehicle stats
            # ----------------
            op_veh_types = veh_type_stats[veh_type_stats[G_V_OP_ID] == op_id]
            op_veh_types.set_index(G_V_VID, inplace=True)
            all_vid_dict = {}
            for vid, vid_vtype_row in op_veh_types.iterrows():
                vtype_data = veh_type_db[vid_vtype_row[G_V_TYPE]]
                op_vid_vehicle_df = op_vehicle_df[op_vehicle_df[G_V_VID] == vid]
                veh_km = op_vid_vehicle_df[G_VR_LEG_DISTANCE].sum() / 1000
                veh_kWh = veh_km * vtype_data[G_VTYPE_BATTERY_SIZE] / vtype_data[G_VTYPE_RANGE]
                co2_per_kWh = scenario_parameters.get(G_ENERGY_EMISSIONS, ENERGY_EMISSIONS)
                if co2_per_kWh is None:
                    co2_per_kWh = ENERGY_EMISSIONS
                veh_co2 = co2_per_kWh * veh_kWh
                veh_fix_costs = np.rint(scenario_parameters.get(G_OP_SHARE_FC, 1.0) * vtype_data[G_VTYPE_FIX_COST])
                veh_var_costs = np.rint(vtype_data[G_VTYPE_DIST_COST] * veh_km)
                # TODO # after ISTTT: idle times
                all_vid_dict[vid] = {"type":vtype_data[G_VTYPE_NAME], "total km":veh_km, "total kWh": veh_kWh,
                                    "total CO2 [g]": veh_co2, "fix costs": veh_fix_costs,
                                    "total variable costs": veh_var_costs}
            all_vid_df = pd.DataFrame.from_dict(all_vid_dict, orient="index")
            all_vid_df.to_csv(os.path.join(output_dir, f"standard_mod-{op_id}_veh_eval.csv"))

            # aggregated specific by vehicle stats
            # ------------------------------------
            try:
                op_co2 = all_vid_df["total CO2 [g]"].sum()
                op_ext_em_costs = np.rint(EMISSION_CPG * op_co2)
                op_fix_costs = all_vid_df["fix costs"].sum()
                op_var_costs = all_vid_df["total variable costs"].sum()
            except:
                op_co2 = 0
                op_ext_em_costs = 0
                op_fix_costs = 0
                op_var_costs = 0

            result_dict["rides_per_veh_rev_hours"] = get_rides_per_vehicle_revenue_hour(op_vehicle_df, op_users, scenario_parameters, operator_attributes, dir_names)

            # multiple boarding points
            result_dict.update(multiple_boarding_points(op_users, operator_attributes, scenario_parameters, dir_names, op_var_costs))

        elif op_id == G_MC_DEC_PT:
            # 2) public transportation: -> evaluation/publictransport.py
            # - revenue, number served users, modal split, travel time, utility (user_stats)
            # - costs, emissions, utilization (publictransport evaluation)
            op_name = "PT"
            if active_offer_parameters.get(G_OFFER_DRIVE):
                op_avg_travel_time = sum([op_id_to_offer_dict[op_id][rq_id][G_OFFER_DRIVE]
                                    for rq_id in op_users[G_RQ_ID]])/op_number_users
            if G_RQ_FARE in op_users.columns:
                op_revenue = op_users[G_RQ_FARE].sum()
            pt_extra_eval_dict = public_transport_evaluation(output_dir)
            result_dict.update(pt_extra_eval_dict)

        elif op_id == G_MC_DEC_PV:
            # 3) private vehicle:
            # - number PV users, modal split, travel time, parking costs, toll, utility (user_stats)
            # - costs, emissions
            op_name = "PV"
            if active_offer_parameters.get(G_OFFER_DRIVE):
                op_avg_travel_time = sum([op_id_to_offer_dict[op_id][rq_id][G_OFFER_DRIVE]
                                    for rq_id in op_users[G_RQ_ID]])/op_number_users
            if active_offer_parameters.get(G_OFFER_DIST):
                op_avg_travel_distance = sum([op_id_to_offer_dict[op_id][rq_id][G_OFFER_DRIVE]
                                    for rq_id in op_users[G_RQ_ID]])/op_number_users   
            if G_RQ_TOLL in op_users.columns:
                op_toll = op_users[G_RQ_TOLL].sum()/op_number_users
            if G_RQ_PARK in op_users.columns:
                op_parking_cost = op_users[G_RQ_PARK].sum()/op_number_users

        else:
            # 4) intermodal AMoD: AMoD fare already treated by sub-request -> only evaluate PT fare and subsidies (user_stats)
            op_name = "IM_MoD_{}".format(op_id)
            if print_comments:
                print("\t ... processing intermodal user data")

            im_pt_revenue = np.nan
            if G_RQ_IM_PT_FARE in op_users.columns:
                im_pt_revenue = op_users[G_RQ_IM_PT_FARE].sum()
            im_subsidy = np.nan
            if G_RQ_IM_PT_FARE in op_users.columns:
                im_subsidy = op_users[G_RQ_SUB].sum()

            result_dict["pt revenue"] = im_pt_revenue
            result_dict["total intermodal MoD subsidy"] = im_subsidy

        # output
        # ------"travel distance": pv_distance,  "parking cost": pv_parking_cost, "toll": pv_toll
        result_dict["travel time"] = op_avg_travel_time
        result_dict["travel distance"] = op_avg_travel_distance
        result_dict["waiting time"] = op_avg_wait_time
        result_dict["detour time"] = op_avg_detour_time
        result_dict["rel detour"] = op_avg_rel_detour
        result_dict["detour time pudo"] = op_avg_detour_time_pick
        result_dict["rel detour pudo"] = op_avg_rel_detour_pick
        result_dict["avg direct travel time pudo"] = op_avg_direct_travel_time_pick
        result_dict[r"% fleet utilization"] = op_fleet_utilization
        result_dict["total vkm"] = op_total_km
        result_dict["occupancy"] = op_distance_avg_occupancy
        result_dict["occupancy rq"] = op_distance_avg_rq
        result_dict["request driven distance"] = op_rq_distance
        result_dict["person driven distance"] = op_per_distance
        result_dict[r"% empty vkm"] = op_empty_vkm
        result_dict[r"% repositioning vkm"] = op_repositioning_vkm
        result_dict["customer direct distance [km]"] = op_sum_direct_travel_distance
        result_dict["customer direct distance pudo [km]"] = op_sum_direct_travel_distance_pick
        result_dict["saved distance [%]"] = op_saved_distance
        result_dict["total toll"] = op_toll
        result_dict["mod revenue"] = op_revenue
        result_dict["mod fix costs"] = op_fix_costs
        result_dict["mod var costs"] = op_var_costs
        result_dict["total CO2 emissions [t]"] = op_co2 / 10**6
        result_dict["total external emission costs"] = op_ext_em_costs
        result_dict["parking cost"] = op_parking_cost
        result_dict["toll"] = op_toll

        result_dict_list.append(result_dict)
        operator_names.append(op_name)

    # combine and save
    result_df = pd.DataFrame(result_dict_list, index=operator_names)
    result_df = result_df.transpose()
    result_df.to_csv(os.path.join(output_dir, "standard_eval.csv"))

if __name__ == "__main__":
    results_dir = r'C:\Users\ge37ser\Documents\Coding\TUM_VT_FleetSimulation\tum-vt-fleet-simulation\studies\MOIA_calibration\results'
    # results_dir = r'C:\Users\ge37ser\Documents\Coding\TUM_VT_FleetSimulation\tum-vt-fleet-simulation\studies\MOIA_mobitopp\results\20210203_bisDi8h-fleetsim\results\simulation'
    # sc = "fleetsim"
    # results_dir = r'C:\Users\ge37ser\Documents\Coding\TUM_VT_FleetSimulation\tum-vt-fleet-simulation\studies\MOIA_mobitopp\results\20210204_friday-fleetsim'
    # sc = "20210204_friday-fleetsim"
    # standard_evaluation(os.path.join(results_dir, sc))
    for sc in os.listdir(results_dir):
        if sc == "_archiv":
            continue
        print("eval ", sc)
        standard_evaluation(os.path.join(results_dir, sc))
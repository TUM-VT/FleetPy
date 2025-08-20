import os
import sys
import glob
import numpy as np
import pandas as pd

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(MAIN_DIR)
from src.evaluation.multipleboardingpoints_eval import multiple_boarding_points
from src.misc.globals import *

EURO_PER_TON_OF_CO2 = 145 # from BVWP2030 Modulhandbuch (page 113)
EMISSION_CPG = 145 * 100 / 1000**2
ENERGY_EMISSIONS = 112 # g/kWh from https://www.swm.de/dam/swm/dokumente/geschaeftskunden/broschuere-strom-erdgas-gk.pdf
PV_G_CO2_KM = 130 # g/km from https://www.ris-muenchen.de/RII/RII/DOK/ANTRAG/2337762.pdf with 60:38 benzin vs diesel


def read_op_output_file(output_dir, op_id, evaluation_start_time = None, evaluation_end_time = None) -> pd.DataFrame:
    """ this method reads the ouputfile for the operator and returns its dataframe
    :param output_dir: directory of the scenario results
    :param op_id: operator id to evaluate
    :param evaluation_start_time: if given all entries starting before this time are discarded
    :param evaluation_end_time: if given, all entries starting after this time are discarded
    :return: output dataframe of specific operator
    """
    op_df = pd.read_csv(os.path.join(output_dir, f"2-{int(op_id)}_op-stats.csv"))
    if evaluation_start_time is not None:
        op_df = op_df[op_df[G_VR_LEG_START_TIME] >= evaluation_start_time]
    if evaluation_end_time is not None:
        op_df = op_df[op_df[G_VR_LEG_START_TIME] < evaluation_end_time]
    for col in [G_VR_ALIGHTING_RID, G_VR_BOARDING_RID, G_VR_OB_RID]:
        op_df[col] = op_df[col].astype(str)
    return op_df

def read_user_output_file(output_dir, evaluation_start_time = None, evaluation_end_time = None) -> pd.DataFrame:
    """ this method reads the ouputfile the users and returns its dataframe
    :param output_dir: directory of the scenario results
    :param op_id: operator id to evaluate
    :param evaluation_start_time: if given all entries starting before this time are discarded
    :param evaluation_end_time: if given, all entries starting after this time are discarded
    :return: output dataframe of specific operator
    """
    user_stats = pd.read_csv(os.path.join(output_dir, "1_user-stats.csv"))
    if evaluation_start_time is not None:
        user_stats = user_stats[user_stats[G_RQ_TIME] >= evaluation_start_time]
    if evaluation_end_time is not None:
        user_stats = user_stats[user_stats[G_RQ_TIME] < evaluation_end_time]
    return user_stats

def decode_offer_str(offer_str):
    """ this method decodes the offer string from the user stats file into a dictionary """
    result = {}
    for op_entry in offer_str.split("|"):
        parts = op_entry.split(":")
        op_id = int(parts[0])
        values = ":".join(parts[1:])
        for item in values.split(";"):
            if item:
                key, value = item.split(":")
                if op_id not in result:
                    result[op_id] = {}
                result[op_id][key] = round(float(value), 2)
    return result

def create_vehicle_type_db(vehicle_data_dir):
    list_veh_data_f = glob.glob(f"{vehicle_data_dir}/*csv")
    veh_type_db = {}    # veh_type -> veh_type_data
    for f in list_veh_data_f:
        veh_type_name = os.path.basename(f)[:-4]
        veh_type_data = pd.read_csv(f, index_col=0).squeeze("columns")
        veh_type_db[veh_type_name] = {}
        for k, v in veh_type_data.items():
            try:
                veh_type_db[veh_type_name][k] = float(v)
            except:
                veh_type_db[veh_type_name][k] = v
        veh_type_db[veh_type_name][G_VTYPE_NAME] = veh_type_data.name
    return veh_type_db

def avg_in_vehicle_distance(op_df):
    N = 0
    dis_dict = {}
    sum_dis = 0
    for vid, veh_df in op_df.groupby(G_V_VID):
        for status, distance, ob_str, boarding_str, alight_str in zip(veh_df[G_VR_STATUS].values, veh_df[G_VR_LEG_DISTANCE].values, veh_df[G_VR_OB_RID].values, veh_df[G_VR_BOARDING_RID].values, veh_df[G_VR_ALIGHTING_RID].values):
            if status == VRL_STATES.BOARDING.display_name:
                if boarding_str == boarding_str:
                    for rid in boarding_str.split(";"):
                        dis_dict[rid] = 0
                if alight_str == alight_str:
                    for rid in alight_str.split(";"):
                        try:
                            sum_dis += dis_dict[rid]
                            N += 1
                            del dis_dict[rid]
                        except KeyError:
                            pass
            else:
                if ob_str == ob_str:
                    for rid in ob_str.split(";"):
                        try:
                            dis_dict[rid] += distance
                        except KeyError:
                            pass
        dis_dict = {}
    return sum_dis/N

def shared_rides(op_df):
    N = 0
    rid_shared_dict = {}
    N_shared = 0
    for vid, veh_df in op_df.groupby(G_V_VID):
        for status, distance, ob_str, boarding_str, alight_str in zip(veh_df[G_VR_STATUS].values, veh_df[G_VR_LEG_DISTANCE].values, veh_df[G_VR_OB_RID].values, veh_df[G_VR_BOARDING_RID].values, veh_df[G_VR_ALIGHTING_RID].values):
            if status == VRL_STATES.BOARDING.display_name:
                if boarding_str == boarding_str:
                    for rid in boarding_str.split(";"):
                        rid_shared_dict[rid] = 0
                if alight_str == alight_str:
                    for rid in alight_str.split(";"):
                        try:
                            N_shared += rid_shared_dict.get(rid, 0)
                            N += 1
                            del rid_shared_dict[rid]
                        except KeyError:
                            pass
            else:
                if ob_str == ob_str:
                    ob_list = ob_str.split(";")
                    if len(ob_list) > 1:
                        for rid in ob_list:
                            try:
                                rid_shared_dict[rid] = 1
                            except KeyError:
                                pass
        rid_shared_dict = {}
    return 100.0*N_shared/N


def calculate_user_stats_for_operator(op_id, select_op_users, all_users, op_offers, operator_attributes, prefix=""):
    op_reservation_horizon = operator_attributes.get(G_RA_OPT_HOR, 0)
    number_users = len(all_users)
    number_total_travelers = all_users[G_RQ_PAX].sum()
    op_number_users = len(select_op_users)
    op_number_pax = select_op_users[G_RQ_PAX].sum()

    op_modal_split_rq = float(op_number_users) / number_users
    op_modal_split = float(op_number_pax) / number_total_travelers

    op_avg_utility = np.nan
    if G_RQ_C_UTIL in select_op_users.columns:
        op_avg_utility = select_op_users[G_RQ_C_UTIL].sum() / op_number_users

    op_reservation_users = select_op_users[select_op_users[G_RQ_EPT] - select_op_users[G_RQ_TIME] > op_reservation_horizon]
    total_reservation_users = all_users[all_users[G_RQ_EPT] - all_users[G_RQ_TIME] > op_reservation_horizon]
    op_number_reservation_users = len(op_reservation_users)
    op_number_reservation_pax = op_reservation_users[G_RQ_PAX].sum()
    op_frac_served_reservation_users = 100.0
    op_frac_served_reservation_pax = 100.0
    if len(total_reservation_users) > 0:
        op_frac_served_reservation_users = op_number_reservation_users / total_reservation_users.shape[0] * 100.0
        op_frac_served_reservation_pax = op_number_reservation_pax / total_reservation_users[G_RQ_PAX].sum() * 100.0
    op_number_online_users = op_number_users - op_number_reservation_users
    op_number_online_pax = op_number_pax - op_number_reservation_pax
    op_frac_served_online_users = 100.0
    op_frac_served_online_pax = 100.0
    if number_users - len(total_reservation_users) > 0:
        op_frac_served_online_users = op_number_online_users / (number_users - total_reservation_users.shape[0]) * 100.0
        op_frac_served_online_pax = op_number_online_pax / (number_total_travelers - total_reservation_users[G_RQ_PAX].sum()) * 100.0

    all_select_rid = set(select_op_users[G_RQ_ID].to_list())
    select_offers = len([rid for rid in op_offers if rid in all_select_rid])
    print(f"For operator : {op_id}, {prefix} {select_offers} offers created")

    result_dict = {prefix + "number users": op_number_users,
                   prefix + "number travelers": op_number_pax,
                   prefix + "modal split": round(op_modal_split, 2),
                   prefix + "modal split rq": round(op_modal_split_rq, 2),
                   prefix + "reservation users": op_number_reservation_users,
                   prefix + "reservation pax": op_number_reservation_pax,
                   prefix + "served reservation users [%]": round(op_frac_served_reservation_users, 1),
                   prefix + "served reservation pax [%]": round(op_frac_served_reservation_pax, 1),
                   prefix + "online users": op_number_online_users,
                   prefix + "online pax": op_number_online_pax,
                   prefix + "served online users [%]": round(op_frac_served_online_users, 1),
                   prefix + "served online pax [%]": round(op_frac_served_online_pax, 1),
                   prefix + r'% created offers': round(select_offers / number_users * 100.0, 1),
                   prefix + "utility": round(op_avg_utility, 2)}

    boarding_time = operator_attributes.get(G_OP_CONST_BT, 0)
    op_revenue = np.nan
    op_avg_wait_time = np.nan
    op_med_wait_time = np.nan
    op_90perquant_wait_time = np.nan
    op_avg_wait_from_ept = np.nan
    op_avg_detour_time = np.nan
    op_avg_rel_detour = np.nan
    op_sum_direct_travel_distance = np.nan

    if G_RQ_DO in select_op_users.columns and G_RQ_PU in select_op_users.columns:
        # total travel time
        op_user_sum_travel_time = select_op_users[G_RQ_DO].sum() - select_op_users[G_RQ_PU].sum()

        if G_RQ_DRT in select_op_users.columns:
            # avg abs detour time
            op_avg_detour_time = (op_user_sum_travel_time - select_op_users[G_RQ_DRT].sum()) / op_number_users - \
                                 boarding_time
            rel_det_series = (select_op_users[G_RQ_DO] - select_op_users[G_RQ_PU] - boarding_time -
                              select_op_users[G_RQ_DRT]) / select_op_users[G_RQ_DRT]
            # avg rel detour time
            op_avg_rel_detour = rel_det_series.sum() / op_number_users * 100.0

    # sum fare
    if G_RQ_FARE in select_op_users.columns:
        op_revenue = select_op_users[G_RQ_FARE].sum()

    # avg waiting time
    if G_RQ_PU in select_op_users.columns and G_RQ_TIME in select_op_users.columns:
        select_op_users["wait time"] = select_op_users[G_RQ_PU] - select_op_users[G_RQ_TIME]
        op_avg_wait_time = select_op_users["wait time"].mean()
        op_med_wait_time = select_op_users["wait time"].median()
        op_90perquant_wait_time = select_op_users["wait time"].quantile(q=0.9)

    # avg waiting time from earliest pickup time
    if G_RQ_PU in select_op_users.columns and G_RQ_EPT in select_op_users.columns:
        op_avg_wait_from_ept = (select_op_users[G_RQ_PU].sum() - select_op_users[G_RQ_EPT].sum()) / op_number_users

    # direct travel time and distance
    if G_RQ_DRD in select_op_users.columns:
        op_sum_direct_travel_distance = select_op_users[G_RQ_DRD].sum() / 1000.0

    result_dict.update({
        prefix + "waiting time": op_avg_wait_time,
        prefix + "waiting time from ept": op_avg_wait_from_ept,
        prefix + "waiting time (median)": op_med_wait_time,
        prefix + "waiting time (90% quantile)": op_90perquant_wait_time,
        prefix + "detour time": op_avg_detour_time,
        prefix + "rel detour": op_avg_rel_detour,
        prefix + "customer direct distance [km]": op_sum_direct_travel_distance,
        prefix + "mod revenue": op_revenue,
    })


    return result_dict


def standard_evaluation(output_dir, evaluation_start_time = None, evaluation_end_time = None, print_comments=False, dir_names_in = {}):
    """This function runs a standard evaluation over a scenario output directory.

    :param output_dir: scenario output directory
    :param start_time: start time of evaluation interval in s (if None, then evaluation of all data from sim start)
    :param end_time: end time of evaluation interval in s   (if None, then evalation of all data until sim end)
    :param print_comments: print some comments about status in between
    """
    if not os.path.isdir(output_dir):
        raise IOError(f"Could not find result directory {output_dir}!")
    
    scenario_parameters, list_operator_attributes, _ = load_scenario_inputs(output_dir)
    dir_names = get_directory_dict(scenario_parameters, list_operator_attributes, abs_fleetpy_dir=MAIN_DIR)
    if dir_names_in:
        dir_names = dir_names_in

    # evaluation interval
    if evaluation_start_time is None and scenario_parameters.get(G_EVAL_INT_START) is not None:
        evaluation_start_time = int(scenario_parameters[G_EVAL_INT_START])
    if evaluation_end_time is None and scenario_parameters.get(G_EVAL_INT_END) is not None:
        evaluation_end_time = int(scenario_parameters[G_EVAL_INT_END])

    # vehicle type data
    veh_type_db = create_vehicle_type_db(dir_names[G_DIR_VEH])
    veh_type_stats = pd.read_csv(os.path.join(output_dir, "2_vehicle_types.csv"))

    if print_comments:
        print(f"Evaluating {scenario_parameters[G_SCENARIO_NAME]}\nReading user stats ...")
    user_stats = read_user_output_file(output_dir, evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time)
    if print_comments:
        print(f"\t shape of user stats: {user_stats.shape}")

    result_dict_list = []
    operator_names = []

    # add passengers columns where necessary
    if G_RQ_ID not in user_stats.columns:
        user_stats[G_RQ_ID] = 1

    row_id_to_offer_dict = {}    # user_stats_row_id -> op_id -> offer
    op_id_to_offer_dict = {}    # op_id -> user_stats_row_id -> offer
    active_offer_parameters = {}
    for key, entries in user_stats.iterrows():
        offer_entry = entries[G_RQ_OFFERS]
        offer = decode_offer_str(offer_entry)
        row_id_to_offer_dict[key] = offer
        rid = entries[G_RQ_ID]
        for op_id, op_offer in offer.items():
            if op_id not in op_id_to_offer_dict:
                op_id_to_offer_dict[op_id] = {}
            op_id_to_offer_dict[op_id][rid] = op_offer
            for offer_param in op_offer.keys():
                active_offer_parameters[offer_param] = 1

    for op_id, op_users in user_stats.groupby(G_RQ_OP_ID):
        op_name = "?"
        
        operator_attributes = list_operator_attributes[int(op_id)]
        operator_offers = op_id_to_offer_dict[op_id]
        rq_types = op_users[G_RQ_TYPE].unique()
        result_dict = {"operator_id": op_id}
        if len(rq_types) > 0:
            all_dict = calculate_user_stats_for_operator(op_id, op_users, user_stats, operator_offers,
                                                         operator_attributes, prefix="[ALL] ")
            result_dict.update(all_dict)
            for rq_type in rq_types:
                select_users = op_users[op_users[G_RQ_TYPE] == rq_type].copy()
                selected_all_users = user_stats[user_stats[G_RQ_TYPE] == rq_type]
                rq_dict = calculate_user_stats_for_operator(op_id, select_users, selected_all_users, operator_offers,
                                                            operator_attributes, prefix=f"[{rq_type}] ")
                result_dict.update(rq_dict)
        else:
            all_dict = calculate_user_stats_for_operator(op_id, op_users, user_stats, operator_offers,
                                                         operator_attributes)
            result_dict.update(all_dict)

        op_number_users = len(op_users)
        op_number_pax = op_users[G_RQ_PAX].sum()

        # base user_values
        op_user_sum_travel_time = np.nan
        op_avg_travel_distance = np.nan

        # base fleet values
        op_fleet_utilization = np.nan
        op_total_km = np.nan
        op_distance_avg_occupancy = np.nan
        op_empty_vkm = np.nan
        op_repositioning_vkm = np.nan
        op_saved_distance = np.nan
        op_ride_distance_per_vehicle_distance = np.nan
        op_ride_distance_per_vehicle_distance_no_rel = np.nan
        op_trip_velocity = np.nan
        op_avg_velocity = np.nan
        op_vehicle_revenue_hours = np.nan
        op_ride_per_veh_rev_hours = np.nan
        op_ride_per_veh_rev_hours_rq = np.nan

        op_toll = np.nan
        op_parking_cost = np.nan
        op_fix_costs = np.nan
        op_var_costs = np.nan
        op_co2 = np.nan
        op_ext_em_costs = np.nan

        if op_id >= 0:  #AMoD
            op_name = "MoD_{}".format(int(op_id))
            if print_comments:
                print("Loading AMoD vehicle data ...")
            try:
                op_vehicle_df = read_op_output_file(output_dir, op_id, evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time)
            except FileNotFoundError:
                op_vehicle_df = pd.DataFrame([], columns=[G_V_OP_ID, G_V_VID, G_VR_STATUS, G_VR_LOCKED, G_VR_LEG_START_TIME,
                                                        G_VR_LEG_END_TIME, G_VR_LEG_START_POS, G_VR_LEG_END_POS,
                                                        G_VR_LEG_DISTANCE, G_VR_LEG_START_SOC, G_VR_LEG_END_SOC,
                                                        G_VR_TOLL, G_VR_OB_RID, G_VR_BOARDING_RID, G_VR_ALIGHTING_RID,
                                                        G_VR_NODE_LIST, G_VR_REPLAY_ROUTE])

            if print_comments:
                print(f"\t shape of vehicle stats: {op_vehicle_df.shape}\n\t ... processing AMoD vehicle data")

            # # sum travel time
            if G_RQ_DO in op_users.columns and G_RQ_PU in op_users.columns:
                op_user_sum_travel_time = op_users[G_RQ_DO].sum() - op_users[G_RQ_PU].sum()
                # avg travel time
                op_avg_travel_time = op_user_sum_travel_time / op_number_users

            # multiple boarding points
            result_dict.update(multiple_boarding_points(op_users, operator_attributes, scenario_parameters, dir_names, op_var_costs))

            # vehicle stats
            # -------------
            n_vehicles = veh_type_stats[veh_type_stats[G_V_OP_ID]==op_id].shape[0]

            sim_end_time = scenario_parameters["end_time"]
            simulation_time = scenario_parameters["end_time"] - scenario_parameters["start_time"]
            try:
                # correct utilization: do not consider tasks after simulation end time
                op_vehicle_df["VRL_end_sim_end_time"] = np.minimum(op_vehicle_df[G_VR_LEG_END_TIME], sim_end_time)
                op_vehicle_df["VRL_start_sim_end_time"] = np.minimum(op_vehicle_df[G_VR_LEG_START_TIME], sim_end_time)
                utilized_veh_df = op_vehicle_df[(op_vehicle_df["status"] != VRL_STATES.OUT_OF_SERVICE.display_name) & (op_vehicle_df["status"] != VRL_STATES.CHARGING.display_name)]
                utilization_time = utilized_veh_df["VRL_end_sim_end_time"].sum() - utilized_veh_df["VRL_start_sim_end_time"].sum()
                unutilized_veh_df = op_vehicle_df[(op_vehicle_df["status"] == VRL_STATES.OUT_OF_SERVICE.display_name) | (op_vehicle_df["status"] == VRL_STATES.CHARGING.display_name)]
                unutilized_time = unutilized_veh_df["VRL_end_sim_end_time"].sum() - unutilized_veh_df["VRL_start_sim_end_time"].sum()
                rev_df = op_vehicle_df[op_vehicle_df["status"].isin([x.display_name for x in G_REVENUE_STATUS])]
                op_vehicle_revenue_hours = (rev_df["VRL_end_sim_end_time"].sum() - rev_df["VRL_start_sim_end_time"].sum())/3600.0
                op_ride_per_veh_rev_hours = op_number_pax/op_vehicle_revenue_hours
                op_ride_per_veh_rev_hours_rq = op_number_users/op_vehicle_revenue_hours
                op_fleet_utilization = 100 * (utilization_time/(n_vehicles * simulation_time - unutilized_time)) #TODO: to change for SoD
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
                empty_df = op_vehicle_df[op_vehicle_df[G_VR_OB_RID] == "nan"]
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

            # saved distance
            op_sum_direct_travel_distance = np.nan
            if G_RQ_DRD in op_users.columns:
                op_sum_direct_travel_distance = op_users[G_RQ_DRD].sum() / 1000.0

            trip_direct_distance = None
            if not np.isnan(op_total_km) and result_dict.get("bp_sum_direct_distance") is not None: # direct distances between pu and do
                bp_sum_direct_distance = result_dict["bp_sum_direct_distance"]
                trip_direct_distance = bp_sum_direct_distance
                op_saved_distance = (bp_sum_direct_distance - op_total_km)/bp_sum_direct_distance * 100.0
                op_ride_distance_per_vehicle_distance = bp_sum_direct_distance / op_total_km
                op_ride_distance_per_vehicle_distance_no_rel = bp_sum_direct_distance / (op_total_km * (1.0 - op_repositioning_vkm/100.0))
            elif not np.isnan(op_total_km) and not np.isnan(op_sum_direct_travel_distance):
                trip_direct_distance = op_sum_direct_travel_distance
                op_saved_distance = (op_sum_direct_travel_distance - op_total_km)/op_sum_direct_travel_distance * 100.0
                op_ride_distance_per_vehicle_distance = op_sum_direct_travel_distance / op_total_km
                op_ride_distance_per_vehicle_distance_no_rel = op_sum_direct_travel_distance / (op_total_km * (1.0 - op_repositioning_vkm/100.0))

            # speed
            driving = op_vehicle_df[op_vehicle_df["status"].isin([i.display_name for i in G_DRIVING_STATUS])]
            driving_time = driving["end_time"].sum() - driving["start_time"].sum()
            op_avg_velocity = op_total_km/driving_time*3600.0
            op_trip_velocity = trip_direct_distance/op_user_sum_travel_time*3.6

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

        elif op_id == G_MC_DEC_PT:
            # 2) public transportation: -> evaluation/publictransport.py
            # - revenue, number served users, modal split, travel time, utility (user_stats)
            # - costs, emissions, utilization (publictransport evaluation)
            op_name = "PT"
            if active_offer_parameters.get(G_OFFER_DRIVE):
                op_avg_travel_time = sum([op_id_to_offer_dict[op_id][rq_id][G_OFFER_DRIVE]
                                    for rq_id in op_users[G_RQ_ID]])/op_number_users

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
        result_dict[r"% fleet utilization"] = op_fleet_utilization
        result_dict["rides per veh rev hours"] = op_ride_per_veh_rev_hours
        result_dict["rides per veh rev hours rq"] = op_ride_per_veh_rev_hours_rq
        result_dict["total vkm"] = op_total_km
        result_dict["occupancy"] = op_distance_avg_occupancy
        result_dict["occupancy rq"] = op_distance_avg_rq
        result_dict[r"% empty vkm"] = op_empty_vkm
        result_dict[r"% repositioning vkm"] = op_repositioning_vkm
        result_dict["saved distance [%]"] = op_saved_distance
        result_dict["trip distance per fleet distance"] = op_ride_distance_per_vehicle_distance
        result_dict["trip distance per fleet distance (no reloc)"] = op_ride_distance_per_vehicle_distance_no_rel
        result_dict["avg driving velocity [km/h]"] = op_avg_velocity
        result_dict["avg trip velocity [km/h]"] = op_trip_velocity
        result_dict["vehicle revenue hours [Fzg h]"] = op_vehicle_revenue_hours
        result_dict["total toll"] = op_toll
        result_dict["mod fix costs"] = op_fix_costs
        result_dict["mod var costs"] = op_var_costs
        result_dict["total CO2 emissions [t]"] = op_co2 / 10**6
        result_dict["total external emission costs"] = op_ext_em_costs
        result_dict["parking cost"] = op_parking_cost
        result_dict["toll"] = op_toll
        result_dict["customer in vehicle distance"] = avg_in_vehicle_distance(op_vehicle_df)
        result_dict["shared rides [%]"] = shared_rides(op_vehicle_df)

        result_dict_list.append(result_dict)
        operator_names.append(op_name)

    # combine and save
    result_df = pd.DataFrame(result_dict_list, index=operator_names)
    result_df = result_df.transpose()
    result_df.to_csv(os.path.join(output_dir, "standard_eval.csv"))

    return result_df


def evaluate_folder(path, evaluation_start_time = None, evaluation_end_time = None, print_comments = False):
    """ this function calls standard_valuation on all simulation results found in the given path 
    :param evaluation_start_time: start time of evaluation interval in s (if None, then evaluation from sim start)
    :param evaluation_end_time: end time of evaluation interval in s   (if None, then evalation ountil sim end)
    :param print_comments: print comments
    """
    for f in os.listdir(path):
        sc_path = os.path.join(path, f)
        if os.path.isdir(sc_path):
            if os.path.isfile(os.path.join(sc_path, "1_user-stats.csv")):
                standard_evaluation(sc_path, evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time, print_comments=print_comments)


if __name__ == "__main__":
    import sys
    sc = sys.argv[1]
    # sc = r'C:\Users\ge37ser\Documents\Coding\TUM_VT_FleetSimulation\tum-vt-fleet-simulation\results\FabianRPPsc01\sc01_200_1'
    #evaluate_folder(sc, print_comments=True)
    standard_evaluation(sc, print_comments=True)

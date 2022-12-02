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

def get_directory_dict(scenario_parameters):
    """
    This function provides the correct paths to certain data according to the specified data directory structure.
    :param scenario_parameters: simulation input (pandas series)
    :return: dictionary with paths to the respective data directories
    """
    study_name = scenario_parameters[G_STUDY_NAME]
    scenario_name = scenario_parameters[G_SCENARIO_NAME]
    network_name = scenario_parameters[G_NETWORK_NAME]
    demand_name = scenario_parameters[G_DEMAND_NAME]
    zone_name = scenario_parameters.get(G_ZONE_SYSTEM_NAME, None)
    fc_type = scenario_parameters.get(G_FC_TYPE, None)
    fc_t_res = scenario_parameters.get(G_FC_TR, None)
    gtfs_name = scenario_parameters.get(G_GTFS_NAME, None)
    infra_name = scenario_parameters.get(G_INFRA_NAME, None)
    parcel_demand_name = scenario_parameters.get(G_PA_DEMAND_NAME, None)
    #
    dirs = {}
    dirs[G_DIR_MAIN] = MAIN_DIR # here is the difference compared to the function in FLeetsimulationBase.py
    dirs[G_DIR_DATA] = os.path.join(dirs[G_DIR_MAIN], "data")
    dirs[G_DIR_OUTPUT] = os.path.join(dirs[G_DIR_MAIN], "studies", study_name, "results", scenario_name)
    dirs[G_DIR_NETWORK] = os.path.join(dirs[G_DIR_DATA], "networks", network_name)
    dirs[G_DIR_VEH] = os.path.join(dirs[G_DIR_DATA], "vehicles")
    dirs[G_DIR_FCTRL] = os.path.join(dirs[G_DIR_DATA], "fleetctrl")
    dirs[G_DIR_DEMAND] = os.path.join(dirs[G_DIR_DATA], "demand", demand_name, "matched", network_name)
    if zone_name is not None:
        dirs[G_DIR_ZONES] = os.path.join(dirs[G_DIR_DATA], "zones", zone_name, network_name)
        if fc_type is not None and fc_t_res is not None:
            dirs[G_DIR_FC] = os.path.join(dirs[G_DIR_DATA], "demand", demand_name, "aggregated", zone_name, str(fc_t_res))
    if gtfs_name is not None:
        dirs[G_DIR_PT] = os.path.join(dirs[G_DIR_DATA], "pubtrans", gtfs_name)
    if infra_name is not None:
        dirs[G_DIR_INFRA] = os.path.join(dirs[G_DIR_DATA], "infra", infra_name, network_name)
    if parcel_demand_name is not None:
        dirs[G_DIR_PARCEL_DEMAND] = os.path.join(dirs[G_DIR_DATA], "demand", parcel_demand_name, "matched", network_name)
    return dirs

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
    # test for correct datatypes
    def convert_str(val):
        if val != val:
            return val
        if type(val) == str:
            return val
        else:
            return str(int(val))
    test_convert = [G_VR_ALIGHTING_RID, G_VR_BOARDING_RID, G_VR_OB_RID]
    for col in test_convert:
        if op_df.dtypes[col] != str:
            op_df[col] = op_df[col].apply(convert_str)
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
    """ create a dictionary from offer_str in outputfile """
    offer_dict = {}
    try:
        offer_strs = offer_str.split("|")
    except:
        return {}
    for offer_str in offer_strs:
        x = offer_str.split(":")
        op = int(x[0])
        vals = ":".join(x[1:])
        if len(vals) == 0:
            continue
        offer_dict[op] = {}
        for offer_entries in vals.split(";"):
            try:
                offer_at, v2 = offer_entries.split(":")
            except:
                continue
            try:
                v2 = int(v2)
            except:
                try:
                    v2 = float(v2)
                except:
                    pass
            offer_dict[op][offer_at] = v2
    return offer_dict


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
    dir_names = get_directory_dict(scenario_parameters)
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
        row_id_to_offer_dict[key] = entries[G_RQ_PAX]
        offer_entry = entries[G_RQ_OFFERS]
        offer = decode_offer_str(offer_entry)
        row_id_to_offer_dict[key] = offer
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
        
        op_reservation_horizon = list_operator_attributes[int(op_id)].get(G_RA_OPT_HOR,0)

        op_number_users = op_users.shape[0]
        op_number_pax = op_users[G_RQ_PAX].sum()
        op_created_offers = len(op_id_to_offer_dict.get(op_id, {}).keys())
        op_modal_split_rq = float(op_number_users)/number_users
        op_modal_split = float(op_number_pax)/number_total_travelers
        if print_comments:
            print(op_created_offers)
        op_rel_created_offers = float(op_created_offers)/number_users*100.0
        op_avg_utility = np.nan
        if G_RQ_C_UTIL in op_users.columns:
            op_avg_utility = op_users[G_RQ_C_UTIL].sum()/op_number_users  
            
        op_reservation_users = op_users[op_users[G_RQ_EPT] - op_users[G_RQ_TIME] > op_reservation_horizon]
        total_reservation_users = user_stats[user_stats[G_RQ_EPT] - user_stats[G_RQ_TIME] > op_reservation_horizon]
        op_number_reservation_users = op_reservation_users.shape[0]
        op_number_reservation_pax = op_reservation_users[G_RQ_PAX].sum()
        try:
            op_frac_served_reservation_users = op_number_reservation_users/total_reservation_users.shape[0]*100.0
            op_frac_served_reservation_pax = op_number_reservation_pax/total_reservation_users[G_RQ_PAX].sum()*100.0
        except ZeroDivisionError:
            op_frac_served_reservation_users = 100.0
            op_frac_served_reservation_pax = 100.0
        op_number_online_users = op_number_users - op_number_reservation_users
        op_number_online_pax = op_number_pax - op_number_reservation_pax
        try:
            op_frac_served_online_users = op_number_online_users/(number_users - total_reservation_users.shape[0])*100.0
            op_frac_served_online_pax = op_number_online_pax/(number_total_travelers - total_reservation_users[G_RQ_PAX].sum())*100.0
        except ZeroDivisionError:
            op_frac_served_online_users = 100.0
            op_frac_served_online_pax = 100.0

        result_dict = {"operator_id": op_id, 
                       "number users": op_number_users,
                       "number travelers": op_number_pax,
                       "modal split": op_modal_split,
                       "modal split rq": op_modal_split_rq,
                       "reservation users": op_number_reservation_users,
                       "reservation pax" : op_number_reservation_pax,
                       "served reservation users [%]": op_frac_served_reservation_users,
                       "served reservation pax [%]": op_frac_served_reservation_pax,
                       "online users" : op_number_online_users,
                       "online pax" : op_number_online_pax,
                       "served online users [%]": op_frac_served_online_users,
                       "served online pax [%]": op_frac_served_online_pax,
                       r'% created offers': op_rel_created_offers,
                       "utility" : op_avg_utility}

        # base user_values
        op_user_sum_travel_time = np.nan
        op_revenue = np.nan
        op_avg_wait_time = np.nan
        op_med_wait_time = np.nan
        op_90perquant_wait_time = np.nan
        op_avg_wait_from_ept = np.nan
        op_avg_travel_time = np.nan
        op_avg_travel_distance = np.nan
        op_sum_direct_travel_distance = np.nan
        op_avg_detour_time = np.nan 
        op_avg_rel_detour = np.nan  
        op_shared_rids = np.nan

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
            operator_attributes = list_operator_attributes[int(op_id)]
            boarding_time = operator_attributes["op_const_boarding_time"]
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
            
            # sum travel time
            if G_RQ_DO in op_users.columns and G_RQ_PU in op_users.columns:
                op_user_sum_travel_time = op_users[G_RQ_DO].sum() - op_users[G_RQ_PU].sum()
            # avg travel time
            if not np.isnan(op_user_sum_travel_time):
                op_avg_travel_time = op_user_sum_travel_time / op_number_users
            # sum fare
            if G_RQ_FARE in op_users.columns:
                op_revenue = op_users[G_RQ_FARE].sum()
            # avg waiting time
            if G_RQ_PU in op_users.columns and G_RQ_TIME in op_users.columns:
                op_users["wait time"] = op_users[G_RQ_PU] - op_users[G_RQ_TIME]
                op_avg_wait_time = op_users["wait time"].mean()
                op_med_wait_time = op_users["wait time"].median()
                op_90perquant_wait_time = op_users["wait time"].quantile(q=0.9)
            # avg waiting time from earliest pickup time
            if G_RQ_PU in op_users.columns and G_RQ_EPT in op_users.columns:
                op_avg_wait_from_ept = (op_users[G_RQ_PU].sum() - op_users[G_RQ_EPT].sum()) / op_number_users
            # avg abs detour time
            if not np.isnan(op_user_sum_travel_time) and G_RQ_DRT in op_users.columns:
                op_avg_detour_time = (op_user_sum_travel_time - op_users[G_RQ_DRT].sum())/op_number_users - \
                                     boarding_time
            # avg rel detour time
            if not np.isnan(op_user_sum_travel_time) and G_RQ_DRT in op_users.columns:
                rel_det_series = (op_users[G_RQ_DO] - op_users[G_RQ_PU] - boarding_time -
                                  op_users[G_RQ_DRT])/op_users[G_RQ_DRT]
                op_avg_rel_detour = rel_det_series.sum()/op_number_users * 100.0
            # direct travel time and distance
            if G_RQ_DRD in op_users.columns:
                op_sum_direct_travel_distance = op_users[G_RQ_DRD].sum() / 1000.0

            # multiple boarding points
            result_dict.update(multiple_boarding_points(op_users, operator_attributes, scenario_parameters, dir_names, op_var_costs))

            # vehicle stats
            # -------------
            try:
                n_vehicles = sum([x for x in operator_attributes[G_OP_FLEET].values()])
            except AttributeError:
                if print_comments:
                    print("Warning: {} not given for this scenario?".format(G_OP_FLEET))
                n_vehicles = len(op_vehicle_df[G_V_VID].unique())

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
                op_fleet_utilization = 100 * (utilization_time/(n_vehicles * simulation_time - unutilized_time))
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

            # saved distance
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
            if G_RQ_FARE in op_users.columns:
                op_revenue = op_users[G_RQ_FARE].sum()

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
        result_dict["waiting time from ept"] = op_avg_wait_from_ept
        result_dict["waiting time (median)"] = op_med_wait_time
        result_dict["waiting time (90% quantile)"] = op_90perquant_wait_time
        result_dict["detour time"] = op_avg_detour_time
        result_dict["rel detour"] = op_avg_rel_detour
        result_dict[r"% fleet utilization"] = op_fleet_utilization
        result_dict["rides per veh rev hours"] = op_ride_per_veh_rev_hours
        result_dict["rides per veh rev hours rq"] = op_ride_per_veh_rev_hours_rq
        result_dict["total vkm"] = op_total_km
        result_dict["occupancy"] = op_distance_avg_occupancy
        result_dict["occupancy rq"] = op_distance_avg_rq
        result_dict[r"% empty vkm"] = op_empty_vkm
        result_dict[r"% repositioning vkm"] = op_repositioning_vkm
        result_dict["customer direct distance [km]"] = op_sum_direct_travel_distance
        result_dict["saved distance [%]"] = op_saved_distance
        result_dict["trip distance per fleet distance"] = op_ride_distance_per_vehicle_distance
        result_dict["trip distance per fleet distance (no reloc)"] = op_ride_distance_per_vehicle_distance_no_rel
        result_dict["avg driving velocity [km/h]"] = op_avg_velocity
        result_dict["avg trip velocity [km/h]"] = op_trip_velocity
        result_dict["vehicle revenue hours [Fzg h]"] = op_vehicle_revenue_hours
        result_dict["total toll"] = op_toll
        result_dict["mod revenue"] = op_revenue
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

def current_state_eval(output_dir, dict_dirs):
    """Evaluation of the data from the current position. As the current state is not stored, in memory for the whole
    simulation, the data needs to be loaded first and then converted back to a dictionary. Contains some hardcoded
    locations for specific folders. """
    import ast

    df_user = pd.read_csv(os.path.join(output_dir, '1_user-stats.csv'))
    df_pos = pd.read_csv(os.path.join(output_dir, 'current_state.csv'))
    df_zones = pd.read_csv(os.path.join(dict_dirs[G_DIR_ZONES], 'node_zone_info.csv'))
    df_nodes = pd.read_csv(os.path.join(dict_dirs[G_DIR_NETWORK], 'base', 'nodes.csv'))
    df_node_500 = pd.read_csv(os.path.join(dict_dirs[G_DIR_DATA], 'zones', 'manhattan_cell_500',
                                           'manhattan_osm_network_fleetpy_trb_2022', 'node_zone_info.csv'))

    drop_col = ['rq_type', 'pickup_location', 'dropoff_location', 'access_time', 'egress_time',
                'decision_time', 'chosen_operator_id', 'operator_id', 'fare', 'modal_state', 'pickup_time',
                'dropoff_time', 'number_passenger']
    df_user = df_user.drop(drop_col, axis=1)
    df_user.sort_values('request_id', inplace=True, ignore_index=True)

    # extract information from offers
    df_user['start'] = df_user['start'].str.extract('(.*?)(?=;)').astype(float)
    df_user['end'] = df_user['end'].str.extract('(.*?)(?=;)').astype(float)
    df_user['wait_time'] = df_user['offers'].str.extract('(?<=t_wait:)(.*?)(?=;)').astype(float)
    df_user['travel_time'] = df_user['offers'].str.extract('(?<=t_drive:)(.*?)(?=;)').astype(float)

    # format time
    df_user['rq_time_day'] = df_user['rq_time'].apply(
        lambda x: (int(np.floor(x / 86400)), int(x % 86400)))  # request time with date
    df_user['rq_time'] = df_user['rq_time'].apply(lambda x: int(x % 86400))
    df_user['earliest_pickup_time'] = df_user['earliest_pickup_time'].apply(lambda x: int(x % 86400))  # no date

    # create categories
    df_user['result'] = 3
    df_user.loc[(df_user['wait_time'] >= 0) & (df_user['wait_time'] < 120), 'result'] = 0
    df_user.loc[(df_user['wait_time'] >= 120) & (df_user['wait_time'] < 240), 'result'] = 1
    df_user.loc[(df_user['wait_time'] >= 240) & (df_user['wait_time'] < 360), 'result'] = 2

    # binary categories and wait time adjustements
    df_user['order_given'] = 1  # True
    df_user.loc[df_user['wait_time'].isnull().values, 'order_given'] = 0  # False
    df_user.loc[df_user['wait_time'].isnull().values, 'wait_time'] = -1
    df_user.loc[df_user['wait_time'].isnull().values, 'travel_time'] = -1

    # sort start and end nodes into zones (NY taxi zones and smaller grid (500))
    df_user['start zone'] = df_user['start'].copy()
    df_user['end zone'] = df_user['end'].copy()
    df_user['start zone'] = df_user['start zone'].apply(lambda x: df_zones.iloc[int(x), 1])
    df_user['end zone'] = df_user['end zone'].apply(lambda x: df_zones.iloc[int(x), 1])
    df_user = df_user.drop('offers', axis=1)
    df_user['start zone 500'] = df_user['start'].apply(lambda x: df_node_500.iloc[int(x), 1])
    df_user['end zone 500'] = df_user['end'].apply(lambda x: df_node_500.iloc[int(x), 1])

    # coordinates for end and start position
    df_user['start_coord_x'] = df_user['start'].copy()
    df_user['start_coord_y'] = df_user['start'].copy()
    df_user['end_coord_x'] = df_user['end'].copy()
    df_user['end_coord_y'] = df_user['end'].copy()
    df_user['start_coord_x'] = df_user['start'].apply(lambda x: int(df_nodes['pos_x'].iloc[int(x)]))
    df_user['start_coord_y'] = df_user['start'].apply(lambda x: int(df_nodes['pos_y'].iloc[int(x)]))
    df_user['end_coord_x'] = df_user['end'].apply(lambda x: int(df_nodes['pos_x'].iloc[int(x)]))
    df_user['end_coord_y'] = df_user['end'].apply(lambda x: int(df_nodes['pos_y'].iloc[int(x)]))

    # normalised direction vector
    def get_norm_direction_vector(x_start, y_start, x_end, y_end):
        x = x_end - x_start
        y = y_end - y_start
        vector = np.array([x, y]).T
        norm = np.linalg.norm(vector, axis=1)
        norm_vec = np.array([np.array([0, 0]) if j == 0 else i / j for i, j in zip(vector, norm)]).T
        return norm_vec[0], norm_vec[1]

    df_user['norm_vec_x'], df_user['norm_vec_y'] = get_norm_direction_vector(df_user['start_coord_x'].to_numpy(),
                                                                   df_user['start_coord_y'].to_numpy(),
                                                                   df_user['end_coord_x'].to_numpy(),
                                                                   df_user['end_coord_y'].to_numpy())

    # definitions for the next part
    pos = 'Pos'
    last_d = 'Last Destination'
    pax = 'Nr. Pax'
    stops = 'Number of Stops'
    dis_start_start = 'Distance Current Position - Origin'
    dis_end_start = 'Distance End Position - Origin'
    dis_end_end = 'Distance End Position - Destination'
    rid = 'Request ID'
    remaining_time = 'Remaining Time CL'
    status = 'Current Vehicle Status'
    last_d_op = 'Last Pos OP'
    last_time_op = 'Last Time OP'

    def str_to_dict(x):
        if type(x) == str:
            return ast.literal_eval(x)
        else:
            return x

    def nodes_to_zone(x):
        if type(x) == dict:
            return (df_zones.iloc[x[pos][0], 1], df_zones.iloc[x[last_d][0], 1])
        else:
            return x

    def nodes_to_zone_only_current_pos(x):
        if type(x) == dict:
            return df_zones.iloc[x[pos][0], 1]
        else:
            return x

    def get_start_position_node(x):
        if type(x) == dict:
            return x[pos][0]
        else:
            return x

    def nodes_to_zone_only_end_pos(x):
        if type(x) == dict:
            return df_zones.iloc[x[last_d][0], 1]
        else:
            return x

    def get_end_position_node(x):
        if type(x) == dict:
            try:
                end_pos = x[last_d][-1][0]
            except:
                end_pos = x[last_d][0]
            return end_pos
        else:
            return x

    def get_pax_from_dict(x):
        if type(x) == dict:
            return x[pax]
        else:
            return x

    def get_stops_from_dict(x):
        if type(x) == dict:
            return x[stops]
        else:
            return x

    def get_distance_start_start(x):
        if type(x) == dict:
            return x[dis_start_start]
        else:
            return x

    def get_distance_end_start(x):
        if type(x) == dict:
            return x[dis_end_start]
        else:
            return x

    def get_distance_end_end(x):
        if type(x) == dict:
            return x[dis_end_end]
        else:
            return x

    def get_rid(x):
        if type(x) == dict:
            return x[rid]
        else:
            return x

    def get_remaining_time_CL_from_dict(x):
        if type(x) == dict:
            return x[remaining_time]
        else:
            return x

    def get_status(x):
        if type(x) == dict:
            return x[status]
        else:
            return x

    def nodes_to_zone_only_current_pos_hailing(x):
        if type(x) == dict:
            return df_node_500.iloc[x['Pos'][0], 1]
        else:
            return x

    def nodes_to_zone_only_end_pos_hailing(x):
        if type(x) == dict:
            try:
                zone_car = df_node_500.iloc[x['Last Destination'][0], 1]
                return zone_car
            except:
                zone_car = df_node_500.iloc[x['Last Destination'][-1][0], 1]
                return zone_car
        else:
            return x

    def get_last_position_op(x):
        if type(x) == dict:
            end_pos = x[last_d_op][0]
            return end_pos
        else:
            return x

    def get_last_time_op(x):
        if type(x) == dict:
            last_time = x[last_time_op]
            return last_time
        else:
            return x

    # sort the current state data set by request id
    ds_rq_id = df_pos['0'].str.extract("(?<='Request ID': )(.*?)(?=,)").astype(float)
    df_pos['rq_id'] = ds_rq_id
    df_pos.sort_values('rq_id', inplace=True, ignore_index=True)
    df_pos.drop('rq_id', inplace=True, axis=1)

    # create copies of the data frame for the individual features
    df_end_pos = df_pos[df_pos.columns[1:]].copy()
    df_start_pos = df_pos[df_pos.columns[1:]].copy()
    df_pax = df_pos[df_pos.columns[1:]].copy()
    df_stops = df_pos[df_pos.columns[1:]].copy()
    df_start_end = df_pos[df_pos.columns[1:]].copy()
    df_end_start = df_pos[df_pos.columns[1:]].copy()
    df_end_end = df_pos[df_pos.columns[1:]].copy()
    df_start_pos_coord_x = df_pos[df_pos.columns[1:]].copy()
    df_start_pos_coord_y = df_pos[df_pos.columns[1:]].copy()
    df_end_pos_coord_x = df_pos[df_pos.columns[1:]].copy()
    df_end_pos_coord_y = df_pos[df_pos.columns[1:]].copy()
    df_remaining_time_CL = df_pos[df_pos.columns[1:]].copy()
    df_norm_vec_x = df_pos[df_pos.columns[1:]].copy()
    df_norm_vec_y = df_pos[df_pos.columns[1:]].copy()
    df_status = df_pos[df_pos.columns[1:]].copy()
    df_last_time_op = df_pos[df_pos.columns[1:]].copy()
    df_end_op = df_pos[df_pos.columns[1:]].copy()
    df_current_pos_500 = df_pos[df_pos.columns[1:]].copy()
    df_end_pos_500 = df_pos[df_pos.columns[1:]].copy()

    # extract information from dictionaries and save them into data frames
    for x in df_pos.columns.values[1:]:
        df_pos[x] = df_pos[x].apply(str_to_dict)
        # df_pos[x] = df_pos[x].apply(nodes_to_zone)
        df_start_pos[x] = df_pos[x].apply(get_start_position_node)
        df_end_pos[x] = df_pos[x].apply(get_end_position_node)
        df_pax[x] = df_pos[x].apply(get_pax_from_dict)
        df_stops[x] = df_pos[x].apply(get_stops_from_dict)
        df_start_end[x] = df_pos[x].apply(get_distance_start_start)
        df_end_start[x] = df_pos[x].apply(get_distance_end_start)
        df_end_end[x] = df_pos[x].apply(get_distance_end_end)
        df_start_pos_coord_x[x] = df_start_pos[x].apply(lambda x: int(df_nodes['pos_x'].iloc[int(x)]))
        df_start_pos_coord_y[x] = df_start_pos[x].apply(lambda x: int(df_nodes['pos_y'].iloc[int(x)]))
        df_end_pos_coord_x[x] = df_end_pos[x].apply(lambda x: int(df_nodes['pos_x'].iloc[int(x)]))
        df_end_pos_coord_y[x] = df_end_pos[x].apply(lambda x: int(df_nodes['pos_y'].iloc[int(x)]))
        df_remaining_time_CL[x] = df_pos[x].apply(get_remaining_time_CL_from_dict)
        df_norm_vec_x[x], df_norm_vec_y[x] = get_norm_direction_vector(df_start_pos_coord_x[x].to_numpy(),
                                                                       df_start_pos_coord_y[x].to_numpy(),
                                                                       df_end_pos_coord_x[x].to_numpy(),
                                                                       df_end_pos_coord_y[x].to_numpy())
        df_status[x] = df_pos[x].apply(get_status)
        df_last_time_op[x] = df_pos[x].apply(get_last_time_op)
        df_end_op[x] = df_pos[x].apply(get_last_position_op)
        df_current_pos_500[x] = df_pos[x].apply(nodes_to_zone_only_current_pos_hailing)
        df_end_pos_500[x] = df_pos[x].apply(nodes_to_zone_only_end_pos_hailing)

    # save the dataframes
    path_database = os.path.join(output_dir, 'databases')
    if not os.path.isdir(path_database):
        os.makedirs(path_database)

    df_user.to_csv(os.path.join(path_database, 'database_hailing.csv'))
    df_start_pos.to_csv(os.path.join(path_database, 'database_start_pos.csv'))
    df_end_pos.to_csv(os.path.join(path_database, 'database_end_pos.csv'))
    df_pax.to_csv(os.path.join(path_database, 'database_pax.csv'))
    df_stops.to_csv(os.path.join(path_database, 'database_stops.csv'))
    df_start_end.to_csv(os.path.join(path_database, 'database_start_end.csv'))
    df_end_start.to_csv(os.path.join(path_database, 'database_end_start.csv'))
    df_end_end.to_csv(os.path.join(path_database, 'database_end_end.csv'))
    df_start_pos_coord_x.to_csv(os.path.join(path_database, 'database_start_x.csv'))
    df_start_pos_coord_y.to_csv(os.path.join(path_database, 'database_start_y.csv'))
    df_end_pos_coord_x.to_csv(os.path.join(path_database, 'database_end_x.csv'))
    df_end_pos_coord_y.to_csv(os.path.join(path_database, 'database_end_y.csv'))
    df_remaining_time_CL.to_csv(os.path.join(path_database, 'database_remaining_time.csv'))
    df_norm_vec_x.to_csv(os.path.join(path_database, 'database_norm_vec_x.csv'))
    df_norm_vec_y.to_csv(os.path.join(path_database, 'database_norm_vec_y.csv'))
    df_status.to_csv(os.path.join(path_database, 'database_vehicle_status.csv'))
    df_last_time_op.to_csv(os.path.join(path_database, 'database_last_time_op.csv'))
    df_end_op.to_csv(os.path.join(path_database, 'database_end_pos_op.csv'))
    df_current_pos_500.to_csv(os.path.join(path_database, 'database_current_pos_500.csv'))
    df_end_pos_500.to_csv(os.path.join(path_database, 'database_end_pos_500.csv'))

    del df_pos
    del df_user
    del df_start_pos
    del df_end_pos
    del df_pax
    del df_stops
    del df_start_end
    del df_end_end
    del df_start_pos_coord_x
    del df_start_pos_coord_y
    del df_end_pos_coord_x
    del df_end_pos_coord_y
    del df_remaining_time_CL
    del df_norm_vec_x
    del df_norm_vec_y
    del df_status
    del df_last_time_op
    del df_end_op
    del df_current_pos_500
    del df_end_pos_500
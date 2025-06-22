import os
import sys
import glob
import ast
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
        offer_strs = [offer_str]
    try:
        for offer_str in offer_strs:
            # Split only on the first colon to get operator ID
            parts = offer_str.split(":", 1)
            if len(parts) < 2:
                continue
                
            try:
                op = ast.literal_eval(parts[0])
                vals = parts[1]
            except:
                continue
                
            if len(vals) == 0:
                continue
                
            offer_dict[op] = {}
            for offer_entries in vals.split(";"):
                try:
                    offer_at, v2 = offer_entries.split(":", 1)
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
    except:
        print("error in offer string: ", offer_str)
        return {}
    return offer_dict


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


def wctr26_flm_eval(output_dir, evaluation_start_time = None, evaluation_end_time = None, print_comments=False, dir_names_in = {}):
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

    # get amod operator attributes
    amod_op_id = 0
    operator_attributes = list_operator_attributes[int(amod_op_id)]
    boarding_time = operator_attributes["op_const_boarding_time"]

    # get pt operator attributes
    pt_op_id = -2

    # vehicle type data
    veh_type_db = create_vehicle_type_db(dir_names[G_DIR_VEH])
    veh_type_stats = pd.read_csv(os.path.join(output_dir, "2_vehicle_types.csv"))

    if print_comments:
        print(f"Evaluating {scenario_parameters[G_SCENARIO_NAME]}\nReading user stats ...")
    user_stats = read_user_output_file(output_dir, evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time)
    if print_comments:
        print(f"\t shape of user stats: {user_stats.shape}")

    # add passengers columns where necessary
    if G_RQ_ID not in user_stats.columns:
        user_stats[G_RQ_ID] = 1

    # get all parent requests
    parent_user_stats = user_stats[user_stats[G_RQ_IS_PARENT_REQUEST] == True]
    served_parent_user_stats = parent_user_stats[parent_user_stats[G_RQ_CHOSEN_OP_ID].notna()]

    # add new columns in to served_parent_user_stats
    served_parent_user_stats["flm_amod_0_travel_time"] = np.nan
    served_parent_user_stats["flm_amod_1_travel_time"] = np.nan
    served_parent_user_stats["flm_amod_travel_time"] = np.nan
    served_parent_user_stats["flm_pt_travel_time"] = np.nan
    served_parent_user_stats["flm_amod_0_detour_time"] = np.nan
    served_parent_user_stats["flm_amod_1_detour_time"] = np.nan
    served_parent_user_stats["flm_amod_detour_time"] = np.nan
    served_parent_user_stats["flm_amod_0_waiting_time"] = np.nan
    served_parent_user_stats["flm_amod_1_waiting_time"] = np.nan
    served_parent_user_stats["flm_amod_waiting_time"] = np.nan
    served_parent_user_stats["flm_pt_waiting_time"] = np.nan
    served_parent_user_stats["flm_amod_0_rel_detour"] = np.nan
    served_parent_user_stats["flm_amod_1_rel_detour"] = np.nan
    served_parent_user_stats["flm_amod_rel_detour"] = np.nan
    served_parent_user_stats["flm_amod_0_drt"] = np.nan
    served_parent_user_stats["flm_amod_1_drt"] = np.nan
    served_parent_user_stats["flm_amod_drt"] = np.nan

    # get all sub requests
    amod_sub_user_stats = user_stats[(user_stats[G_RQ_IS_PARENT_REQUEST] == False) & (user_stats[G_RQ_SUB_TRIP_ID] != RQ_SUB_TRIP_ID.FLM_PT.value)]
    pt_sub_user_stats = user_stats[(user_stats[G_RQ_IS_PARENT_REQUEST] == False) & (user_stats[G_RQ_SUB_TRIP_ID] == RQ_SUB_TRIP_ID.FLM_PT.value)]

    # get all served sub requests
    served_parent_request_ids = served_parent_user_stats[G_RQ_ID].values
    served_amod_sub_user_stats = amod_sub_user_stats[amod_sub_user_stats[G_RQ_ID].isin(served_parent_request_ids)]
    served_pt_sub_user_stats = pt_sub_user_stats[pt_sub_user_stats[G_RQ_ID].isin(served_parent_request_ids)]

    # all users
    number_users = parent_user_stats.shape[0]
    number_total_travelers = parent_user_stats[G_RQ_PAX].sum()

    # all served users
    number_served_users = served_parent_user_stats.shape[0]
    number_served_travelers = served_parent_user_stats[G_RQ_PAX].sum()

    # service rate
    service_rate = number_served_users/number_users*100.0
    rejection_rate = 100.0 - service_rate

    for index, request in served_parent_user_stats.iterrows():
        # convert series to dict
        request = request.to_dict()
        request_id = request[G_RQ_ID]
        flm_amod_sub_request_0 = served_amod_sub_user_stats[(served_amod_sub_user_stats[G_RQ_ID] == request_id) & (served_amod_sub_user_stats[G_RQ_SUB_TRIP_ID] == RQ_SUB_TRIP_ID.FLM_AMOD_0.value)]
        flm_amod_sub_request_1 = served_amod_sub_user_stats[(served_amod_sub_user_stats[G_RQ_ID] == request_id) & (served_amod_sub_user_stats[G_RQ_SUB_TRIP_ID] == RQ_SUB_TRIP_ID.FLM_AMOD_1.value)]
        flm_pt_sub_request = served_pt_sub_user_stats[(served_pt_sub_user_stats[G_RQ_ID] == request_id) & (served_pt_sub_user_stats[G_RQ_SUB_TRIP_ID] == RQ_SUB_TRIP_ID.FLM_PT.value)]

        flm_amod_sub_request_0 = flm_amod_sub_request_0.to_dict('records')[0]
        flm_amod_sub_request_1 = flm_amod_sub_request_1.to_dict('records')[0]
        flm_pt_sub_request = flm_pt_sub_request.to_dict('records')[0]

        waiting_time_amod_0 = flm_amod_sub_request_0[G_RQ_PU] - request[G_RQ_EPT]
        travel_time_0 = flm_amod_sub_request_0[G_RQ_DO] - flm_amod_sub_request_0[G_RQ_PU]
        drt_0 = flm_amod_sub_request_0[G_RQ_DRT]
        detour_time_0 = travel_time_0 - drt_0 - boarding_time
        rel_detour_time_0 = detour_time_0/drt_0*100.0

        waiting_time_amod_1 = flm_amod_sub_request_1[G_RQ_PU] - flm_amod_sub_request_1[G_RQ_EPT]
        travel_time_1 = flm_amod_sub_request_1[G_RQ_DO] - flm_amod_sub_request_1[G_RQ_PU]
        drt_1 = flm_amod_sub_request_1[G_RQ_DRT]
        detour_time_1 = travel_time_1 - drt_1 - boarding_time
        rel_detour_time_1 = detour_time_1/drt_1*100.0

        rel_detour_time = (detour_time_0 + detour_time_1)/(drt_0 + drt_1)*100.0

        pt_offer_str = flm_pt_sub_request[G_RQ_OFFERS]
        pt_offer_dict = decode_offer_str(pt_offer_str)
        pt_waiting_time = pt_offer_dict[pt_op_id][G_OFFER_WAIT]
        pt_travel_time = pt_offer_dict[pt_op_id][G_OFFER_DRIVE]

        eval_dict = {
            "flm_amod_0_travel_time": travel_time_0,
            "flm_amod_1_travel_time": travel_time_1,
            "flm_amod_travel_time": travel_time_0 + travel_time_1,
            "flm_pt_travel_time": pt_travel_time,
            "flm_amod_0_detour_time": detour_time_0,
            "flm_amod_1_detour_time": detour_time_1,
            "flm_amod_detour_time": detour_time_0 + detour_time_1,
            "flm_amod_0_rel_detour": rel_detour_time_0,
            "flm_amod_1_rel_detour": rel_detour_time_1,
            "flm_amod_rel_detour": rel_detour_time,
            "flm_amod_0_waiting_time": waiting_time_amod_0,
            "flm_amod_1_waiting_time": waiting_time_amod_1,
            "flm_amod_waiting_time": waiting_time_amod_0 + waiting_time_amod_1,
            "flm_pt_waiting_time": pt_waiting_time,
            "flm_amod_0_drt": drt_0,
            "flm_amod_1_drt": drt_1,
            "flm_amod_drt": drt_0 + drt_1,
        }

        for key, value in eval_dict.items():
            served_parent_user_stats.loc[index, key] = value

    # avg waiting time
    flm_amod_0_avg_waiting_time = served_parent_user_stats["flm_amod_0_waiting_time"].mean()
    flm_amod_1_avg_waiting_time = served_parent_user_stats["flm_amod_1_waiting_time"].mean()
    flm_amod_avg_waiting_time = served_parent_user_stats["flm_amod_waiting_time"].mean()
    flm_pt_avg_waiting_time = served_parent_user_stats["flm_pt_waiting_time"].mean()

    # avg travel time
    flm_amod_0_avg_travel_time = served_parent_user_stats["flm_amod_0_travel_time"].mean()
    flm_amod_1_avg_travel_time = served_parent_user_stats["flm_amod_1_travel_time"].mean()
    flm_amod_avg_travel_time = served_parent_user_stats["flm_amod_travel_time"].mean()
    flm_pt_avg_travel_time = served_parent_user_stats["flm_pt_travel_time"].mean()

    # avg detour time
    flm_amod_0_avg_detour_time = served_parent_user_stats["flm_amod_0_detour_time"].mean()
    flm_amod_1_avg_detour_time = served_parent_user_stats["flm_amod_1_detour_time"].mean()
    flm_amod_avg_detour_time = served_parent_user_stats["flm_amod_detour_time"].mean()

    # avg rel detour time
    flm_amod_0_avg_rel_detour = served_parent_user_stats["flm_amod_0_detour_time"].sum() / served_parent_user_stats["flm_amod_0_drt"].sum() * 100.0
    flm_amod_1_avg_rel_detour = served_parent_user_stats["flm_amod_1_detour_time"].sum() / served_parent_user_stats["flm_amod_1_drt"].sum() * 100.0
    flm_amod_avg_rel_detour = served_parent_user_stats["flm_amod_detour_time"].sum() / served_parent_user_stats["flm_amod_drt"].sum() * 100.0

    result_dict = {
        "number users": number_users,
        "number travelers": number_total_travelers,
        "number served users": number_served_users,
        "number served travelers": number_served_travelers,
        "service rate [%]": service_rate,
        "rejection rate [%]": rejection_rate,
        "flm_amod_0_avg_waiting_time": flm_amod_0_avg_waiting_time,
        "flm_amod_1_avg_waiting_time": flm_amod_1_avg_waiting_time,
        "flm_amod_avg_waiting_time": flm_amod_avg_waiting_time,
        "flm_pt_avg_waiting_time": flm_pt_avg_waiting_time,
        "flm_amod_0_avg_travel_time": flm_amod_0_avg_travel_time,
        "flm_amod_1_avg_travel_time": flm_amod_1_avg_travel_time,
        "flm_amod_avg_travel_time": flm_amod_avg_travel_time,
        "flm_pt_avg_travel_time": flm_pt_avg_travel_time,
        "flm_amod_0_avg_detour_time": flm_amod_0_avg_detour_time,
        "flm_amod_1_avg_detour_time": flm_amod_1_avg_detour_time,
        "flm_amod_avg_detour_time": flm_amod_avg_detour_time,
        "flm_amod_0_avg_rel_detour [%]": flm_amod_0_avg_rel_detour,
        "flm_amod_1_avg_rel_detour [%]": flm_amod_1_avg_rel_detour,
        "flm_amod_avg_rel_detour [%]": flm_amod_avg_rel_detour,
    }

    # --- operator ---
    try:
        op_vehicle_df = read_op_output_file(output_dir, amod_op_id, evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time)
    except FileNotFoundError:
        op_vehicle_df = pd.DataFrame([], columns=[G_V_OP_ID, G_V_VID, G_VR_STATUS, G_VR_LOCKED, G_VR_LEG_START_TIME,
                                                G_VR_LEG_END_TIME, G_VR_LEG_START_POS, G_VR_LEG_END_POS,
                                                G_VR_LEG_DISTANCE, G_VR_LEG_START_SOC, G_VR_LEG_END_SOC,
                                                G_VR_TOLL, G_VR_OB_RID, G_VR_BOARDING_RID, G_VR_ALIGHTING_RID,
                                                G_VR_NODE_LIST, G_VR_REPLAY_ROUTE])

    n_vehicles = veh_type_stats[veh_type_stats[G_V_OP_ID]==amod_op_id].shape[0]
    sim_end_time = scenario_parameters["end_time"]
    simulation_time = scenario_parameters["end_time"] - scenario_parameters["start_time"]

    op_vehicle_df["VRL_end_sim_end_time"] = np.minimum(op_vehicle_df[G_VR_LEG_END_TIME], sim_end_time)
    op_vehicle_df["VRL_start_sim_end_time"] = np.minimum(op_vehicle_df[G_VR_LEG_START_TIME], sim_end_time)
    utilized_veh_df = op_vehicle_df[(op_vehicle_df["status"] != VRL_STATES.OUT_OF_SERVICE.display_name) & (op_vehicle_df["status"] != VRL_STATES.CHARGING.display_name)]
    utilization_time = utilized_veh_df["VRL_end_sim_end_time"].sum() - utilized_veh_df["VRL_start_sim_end_time"].sum()
    unutilized_veh_df = op_vehicle_df[(op_vehicle_df["status"] == VRL_STATES.OUT_OF_SERVICE.display_name) | (op_vehicle_df["status"] == VRL_STATES.CHARGING.display_name)]
    unutilized_time = unutilized_veh_df["VRL_end_sim_end_time"].sum() - unutilized_veh_df["VRL_start_sim_end_time"].sum()

    op_fleet_utilization = 100 * (utilization_time/(n_vehicles * simulation_time - unutilized_time))

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

    op_vehicle_df["weighted_ob_rq"] = op_vehicle_df.apply(weight_ob_rq, axis = 1)
    op_vehicle_df["weighted_ob_pax"] = op_vehicle_df.apply(weight_ob_pax, axis=1)
    op_distance_avg_rq = op_vehicle_df["weighted_ob_rq"].sum() / op_vehicle_df[G_VR_LEG_DISTANCE].sum()
    op_distance_avg_occupancy = op_vehicle_df["weighted_ob_pax"].sum() / op_vehicle_df[G_VR_LEG_DISTANCE].sum()

    result_dict["fleet utilization [%]"] = op_fleet_utilization
    result_dict["occupancy"] = op_distance_avg_occupancy
    result_dict["occupancy rq"] = op_distance_avg_rq

    # combine and save
    result_df = pd.DataFrame([result_dict])
    result_df = result_df.transpose()
    result_df.to_csv(os.path.join(output_dir, "wctr26_flm_eval.csv"))

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
                wctr26_flm_eval(sc_path, evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time, print_comments=print_comments)


if __name__ == "__main__":
    import sys
    sc = sys.argv[1]
    # sc = r'C:\Users\ge37ser\Documents\Coding\TUM_VT_FleetSimulation\tum-vt-fleet-simulation\results\FabianRPPsc01\sc01_200_1'
    #evaluate_folder(sc, print_comments=True)
    wctr26_flm_eval(sc, print_comments=True)

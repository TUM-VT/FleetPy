import pandas as pd
import numpy as np
import os
import re
import sys
import glob

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(MAIN_DIR)
from src.evaluation.standard import read_op_output_file, read_user_output_file, create_vehicle_type_db
from src.misc.globals import *

DEFAULT_AMOD_OP_ID = 0

EURO_PER_TON_OF_CO2 = 145 # from BVWP2030 Modulhandbuch (page 113)
EMISSION_CPG = 145 * 100 / 1000**2
ENERGY_EMISSIONS = 112 # g/kWh from https://www.swm.de/dam/swm/dokumente/geschaeftskunden/broschuere-strom-erdgas-gk.pdf
PV_G_CO2_KM = 130 # g/km from https://www.ris-muenchen.de/RII/RII/DOK/ANTRAG/2337762.pdf with 60:38 benzin vs diesel


# TODO: extend evaluation with more KPIs
def intermodal_evaluation(output_dir, evaluation_start_time = None, evaluation_end_time = None, print_comments=False, dir_names_in = {}):
    """This function runs a standard evaluation over a scenario output directory for intermodal scenarios.
    Currently, only few KPIs are implemented and only for scenarios with one AMoD operator.

    :param output_dir: scenario output directory
    :param start_time: start time of evaluation interval in s (if None, then evaluation of all data from sim start)
    :param end_time: end time of evaluation interval in s (if None, then evaluation of all data until sim end)
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

    # user stats
    if print_comments:
        print(f"Evaluating {scenario_parameters[G_SCENARIO_NAME]}\nReading user stats ...")
    
    user_stats = read_user_output_file(output_dir, evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time)
    parent_user_stats = user_stats[user_stats[G_RQ_IS_PARENT_REQUEST] == True].copy()

    if print_comments:
        print(f"\t shape of user stats: {user_stats.shape}")
        print(f"\t shape of parent user stats: {parent_user_stats.shape}")

    result_dict_list = []
    operator_names = []

    result_dict = {}

    # ---------------------------- User stats --------------------------------
    num_parent_requests = parent_user_stats.shape[0]
    served_parent_requests = parent_user_stats[parent_user_stats[G_RQ_CHOSEN_OP_ID].notna()].copy()
    num_served_parent_requests = served_parent_requests.shape[0]

    # Average service rate
    average_service_rate = num_served_parent_requests / num_parent_requests if num_parent_requests > 0 else 0.0
    result_dict["average_service_rate"] = average_service_rate

    # Average FLM request service rate
    flm_parent_requests = parent_user_stats[parent_user_stats[G_RQ_MODAL_STATE_VALUE] == RQ_MODAL_STATE.FIRSTLASTMILE.value].copy()
    num_flm_parent_requests = flm_parent_requests.shape[0]
    served_flm_parent_requests = flm_parent_requests[flm_parent_requests[G_RQ_CHOSEN_OP_ID].notna()].copy()
    num_served_flm_parent_requests = served_flm_parent_requests.shape[0]
    average_flm_service_rate = num_served_flm_parent_requests / num_flm_parent_requests if num_flm_parent_requests > 0 else 0.0
    result_dict["average_flm_service_rate"] = average_flm_service_rate

    # Average FM request service rate
    fm_parent_requests = parent_user_stats[parent_user_stats[G_RQ_MODAL_STATE_VALUE] == RQ_MODAL_STATE.FIRSTMILE.value].copy()
    num_fm_parent_requests = fm_parent_requests.shape[0]
    served_fm_parent_requests = fm_parent_requests[fm_parent_requests[G_RQ_CHOSEN_OP_ID].notna()].copy()
    num_served_fm_parent_requests = served_fm_parent_requests.shape[0]
    average_fm_service_rate = num_served_fm_parent_requests / num_fm_parent_requests if num_fm_parent_requests > 0 else 0.0
    result_dict["average_fm_service_rate"] = average_fm_service_rate

    # Average LM request service rate
    lm_parent_requests = parent_user_stats[parent_user_stats[G_RQ_MODAL_STATE_VALUE] == RQ_MODAL_STATE.LASTMILE.value].copy()
    num_lm_parent_requests = lm_parent_requests.shape[0]
    served_lm_parent_requests = lm_parent_requests[lm_parent_requests[G_RQ_CHOSEN_OP_ID].notna()].copy()
    num_served_lm_parent_requests = served_lm_parent_requests.shape[0]
    average_lm_service_rate = num_served_lm_parent_requests / num_lm_parent_requests if num_lm_parent_requests > 0 else 0.0
    result_dict["average_lm_service_rate"] = average_lm_service_rate

    # Average AMoD service rate
    amod_parent_requests = parent_user_stats[parent_user_stats[G_RQ_MODAL_STATE_VALUE] == RQ_MODAL_STATE.MONOMODAL.value].copy()
    num_amod_parent_requests = amod_parent_requests.shape[0]
    served_amod_parent_requests = amod_parent_requests[amod_parent_requests[G_RQ_CHOSEN_OP_ID].notna()].copy()
    num_served_amod_parent_requests = served_amod_parent_requests.shape[0]
    average_amod_service_rate = num_served_amod_parent_requests / num_amod_parent_requests if num_amod_parent_requests > 0 else 0.0
    result_dict["average_amod_service_rate"] = average_amod_service_rate

    # ---------------------------- AMoD operator --------------------------------
    amod_op_id = DEFAULT_AMOD_OP_ID

    if print_comments:
        print(f"Reading AMoD operator {amod_op_id} vehicle stats ...")
    try:
        op_vehicle_df = read_op_output_file(output_dir, op_id=amod_op_id, evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time)
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
    op_total_km = op_vehicle_df[G_VR_LEG_DISTANCE].sum()/1000.0
    
    # by vehicle stats
    # ----------------
    op_veh_types = veh_type_stats[veh_type_stats[G_V_OP_ID] == amod_op_id]
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

    result_dict["fleet_utilization_rate"] = op_fleet_utilization
    result_dict["distance_avg_occupancy"] = op_distance_avg_occupancy
    result_dict["total_km_driven"] = op_total_km
    result_dict["average_km_driven"] = op_total_km / n_vehicles
    result_dict["total_amod_cost"] = op_fix_costs + op_var_costs
    result_dict["average_amod_cost"] = (op_fix_costs + op_var_costs) / n_vehicles
    result_dict['total_amod_fix_cost'] = op_fix_costs
    result_dict['total_amod_variable_cost'] = op_var_costs
    result_dict["total_co2_emissions_g"] = op_co2
    result_dict["total_external_emission_costs"] = op_ext_em_costs

    op_name = f"MoD_{amod_op_id}"

    result_dict_list.append(result_dict)
    operator_names.append(op_name)

    result_df = pd.DataFrame(result_dict_list, index=operator_names)
    result_df = result_df.transpose()
    result_df.to_csv(os.path.join(output_dir, "standard_eval.csv"))

    return result_df
import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(MAIN_DIR)
from src.evaluation.standard import decode_offer_str, load_scenario_inputs, get_directory_dict,\
                                    read_op_output_file, read_user_output_file
from src.misc.globals import *
from src.routing.road.NetworkBase import return_position_from_str

# assuming temporal resolution of seconds in output files
MIN = 60
HOUR = 3600
DAY = 24 * 3600
DEF_TEMPORAL_RESOLUTION = 15*MIN
DEF_SMALLER_TEMPORAL_RESOLUTION = 2*MIN # for stacked plots

# styling figures
FIG_SIZE = (6,4)
LABEL_FONT_SIZE = 14
LIST_COLORS = [x for x in plt.rcParams['axes.prop_cycle'].by_key()['color']]
#LIST_COLORS = ['#DDDBD3', '#E6AA33', '#BC8820', '#4365FF', '#0532FF', '#8C9296', '#000C15'] # MOIA colors
N_COLORS = len(LIST_COLORS)

# other
SMALL_VALUE = 0.000001

def _match_pos_str_to_zone(pos_str, node_to_zone):
    """This method matches a position string to a zone id based on the node_to_zone mapping.

    :param pos_str: position string in the format 'x,y'
    :type pos_str: str
    :param node_to_zone: mapping of node ids to zone ids
    :type node_to_zone: dict
    :return: zone id corresponding to the position string
    :rtype: int or None
    """
    node = return_position_from_str(pos_str)[0]  
    return node_to_zone.get(node, -1)  # Return -1 if the node is not found in the mapping

def evaluate_zone_based_served_requests(user_df, op_id):
    """This method evaluates the number of served requests per zone based on the user dataframe.

    :param user_df: user dataframe containing user trip data
    :type user_df: pd.DataFrame
    :param op_id: operator id
    :type op_id: int
    :return: dataframe with served requests per zone
    :rtype: pd.DataFrame
    """
    zone_to_served_requests = {}
    zone_to_total_requests = {}
    zone_to_served_requests_rel = {}
    op_user_df = user_df[user_df["operator_id"] == op_id]
    for zone, group in user_df.groupby("origin_zone"):
        all_requests = group[G_RQ_PAX].sum()
        served_count = op_user_df[(op_user_df["origin_zone"] == zone) & (op_user_df["vehicle_id"].notna())][G_RQ_PAX].sum()
        zone_to_served_requests[zone] = served_count
        zone_to_total_requests[zone] = all_requests
        zone_to_served_requests_rel[zone] = served_count / all_requests * 100 if all_requests > 0 else 0
    return pd.DataFrame({
        "zone_id": list(zone_to_served_requests.keys()),
        "served_requests_op_{}".format(op_id): list(zone_to_served_requests.values()),
        "total_requests": list(zone_to_total_requests.values()),
        "served_requests_rel_op_{}".format(op_id): list(zone_to_served_requests_rel.values())
    })

def plot_zone_based_served_requests(zone_gdf, op_id, output_dir):
    """This method creates a plot of served requests per zone for a given operator and saves it to the output directory.

    :param zone_gdf: GeoDataFrame containing zone geometries and served request data
    :type zone_gdf: gpd.GeoDataFrame
    :param op_id: operator id
    :type op_id: int
    :param output_dir: output directory to save the plot
    :type output_dir: str
    """
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    zone_gdf.plot(column="served_requests_op_{}".format(op_id), ax=ax, legend=True, cmap='OrRd', edgecolor='black')
    ax.set_title("Served Requests for Operator {}".format(op_id))
    ax.set_axis_off()
    plt.savefig(os.path.join(output_dir, "served_requests_op_{}.png".format(op_id)), bbox_inches='tight')
    plt.close(fig)

def plot_zone_based_served_requests_rel(zone_gdf, op_id, output_dir):
    """This method creates a plot of relative served requests per zone for a given operator and saves it to the output directory.

    :param zone_gdf: GeoDataFrame containing zone geometries and served request data
    :type zone_gdf: gpd.GeoDataFrame
    :param op_id: operator id
    :type op_id: int
    :param output_dir: output directory to save the plot
    :type output_dir: str
    """
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    zone_gdf.plot(column="served_requests_rel_op_{}".format(op_id), ax=ax, legend=True, cmap='YlGnBu', edgecolor='black')
    ax.set_title("Relative Served Requests for Operator {}".format(op_id))
    ax.set_axis_off()
    plt.savefig(os.path.join(output_dir, "served_requests_rel_op_{}.png".format(op_id)), bbox_inches='tight')
    plt.close(fig)

# -------------------------------------------------------------------------------------------------------------------- #
# main script call
def run_complete_spatial_evaluation(output_dir, evaluation_start_time=None, evaluation_end_time=None, zone_system_name=None, print_comments=True):
    """This method creates all plots for fleet, user, network and pt KPIs for a given scenario and saves them in
    the respective output directory. Furthermore, it creates a file 'temporal_eval.csv' containing the time series data.
    These can be used for scenario comparisons.

    :param output_dir: output directory of a scenario
    :type output_dir: str
    :param evaluation_start_time: start time of evaluation
    :param evaluation_end_time: end time of evaluation
    """
    if not os.path.isfile(os.path.join(output_dir, "standard_eval.csv")):
        print(f"WARNING: No standard evaluation file found in {output_dir}. Simulation didnt seem to finish. Skipping temporal evaluation.")
        return

    scenario_parameters, list_operator_attributes, _ = load_scenario_inputs(output_dir)
    dir_names = get_directory_dict(scenario_parameters, list_operator_attributes)

    if zone_system_name is None:
        zone_system_name = scenario_parameters.get(G_ZONE_NAME)
        if zone_system_name is None:
            print(f"WARNING: No zone system name provided and no default zone system found in scenario parameters ({G_ZONE_NAME}). Skipping spatial evaluation.")
            return

    nw_name = scenario_parameters[G_NETWORK_NAME]
    if not os.path.exists(os.path.join(MAIN_DIR, "data", "zones", zone_system_name, "polygon_definition.geojson")):
        print(f"WARNING: No polygon_definition.geojson found for zone system '{zone_system_name}'. Skipping spatial evaluation.")
        return
    if not os.path.exists(os.path.join(MAIN_DIR, "data", "zones", zone_system_name, nw_name, "node_zone_info.csv")):
        print(f"WARNING: No node_zone_info.csv found for zone system '{zone_system_name}' and network '{nw_name}'. Skipping spatial evaluation.")
        return
    print(f" ... read zone system '{zone_system_name}' for network '{nw_name}' at {os.path.join(MAIN_DIR, 'data', 'zones', zone_system_name, nw_name)}")
    zone_gdf = gpd.read_file(os.path.join(MAIN_DIR, "data", "zones", zone_system_name, "polygon_definition.geojson"))
    zone_gdf.set_index("zone_id", inplace=True)
    node_to_zone = pd.read_csv(os.path.join(MAIN_DIR, "data", "zones", zone_system_name, nw_name, "node_zone_info.csv"), index_col=0)["zone_id"].to_dict()

    # evaluation interval
    if evaluation_start_time is None and scenario_parameters.get(G_EVAL_INT_START) is not None:
        evaluation_start_time = int(scenario_parameters[G_EVAL_INT_START])
    if evaluation_end_time is None and scenario_parameters.get(G_EVAL_INT_END) is not None:
        evaluation_end_time = int(scenario_parameters[G_EVAL_INT_END])

    if print_comments:
        print(f" ... read user stats")
    user_df = read_user_output_file(output_dir, evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time)
    user_df["origin_zone"] = user_df["start"].apply(lambda x: _match_pos_str_to_zone(x, node_to_zone))
    user_df["destination_zone"] = user_df["end"].apply(lambda x: _match_pos_str_to_zone(x, node_to_zone))

    for op_id, op_attributes in enumerate(list_operator_attributes):
        if print_comments:
            print(f" ... read stats for op {op_id}")
        op_df = read_op_output_file(output_dir, op_id, evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time)

        if print_comments:
            print(f" ... eval served requests for op {op_id}")
    
        zone_based_served_requests_df = evaluate_zone_based_served_requests(user_df, op_id)
        cols_to_drop = [c for c in zone_based_served_requests_df.columns 
                        if c in zone_gdf.columns and c != "zone_id"]
        zone_gdf = zone_gdf.merge(zone_based_served_requests_df.drop(columns=cols_to_drop), how="left", left_on="zone_id", right_on="zone_id")
        plot_zone_based_served_requests(zone_gdf, op_id, output_dir)
        plot_zone_based_served_requests_rel(zone_gdf, op_id, output_dir)

    zone_gdf.to_file(os.path.join(output_dir, "spatial_eval.geojson"), driver="GeoJSON")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all temporal evaluation plots for a given scenario output directory or the full study folder.")
    parser.add_argument("--scenario_results", help="Folder name under data/scenarios containing scenario results (either this or --study_folder must be provided)")
    parser.add_argument("--study_folder", help="Folder name under data/studies containing the study folder with scenario results (path to folder or folder name) (either this or --scenario_results must be provided)")
    parser.add_argument("--zone_system_name", help="Name of the zone system to use for evaluation (if not provided, the default zone system will be used from the scenario configuration)")
    args = parser.parse_args()

    if args.scenario_results:
        zone_system_name = args.zone_system_name if args.zone_system_name else None
        run_complete_spatial_evaluation(args.scenario_results, zone_system_name=zone_system_name)
    elif args.study_folder:
        study_folder = args.study_folder
        if not os.path.isabs(study_folder):
            study_folder = os.path.join(MAIN_DIR, "studies", study_folder, "results")
            if not os.path.exists(study_folder):
                print(f"Study folder '{args.study_folder}' does not exist.")
                sys.exit(1)

        # Iterate through all scenario result directories in the study folder
        for scenario_dir in os.listdir(study_folder):
            scenario_path = os.path.join(study_folder, scenario_dir)
            if os.path.isdir(scenario_path):
                print(f"Running spatial evaluation for scenario: {scenario_dir}")
                run_complete_spatial_evaluation(scenario_path, zone_system_name=args.zone_system_name)
    else:
        print("Wrong usage: Either --scenario_results or --study_folder must be provided.")
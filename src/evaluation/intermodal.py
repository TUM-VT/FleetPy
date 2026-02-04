import os
import sys
import glob
import numpy as np
import pandas as pd
import re

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(MAIN_DIR)
from src.misc.globals import *

# Import constants from standard evaluation
EURO_PER_TON_OF_CO2 = 145 # from BVWP2030 Modulhandbuch (page 113)
EMISSION_CPG = 145 * 100 / 1000**2
ENERGY_EMISSIONS = 112 # g/kWh from https://www.swm.de/dam/swm/dokumente/geschaeftskunden/broschuere-strom-erdgas-gk.pdf
PV_G_CO2_KM = 130 # g/km from https://www.ris-muenchen.de/RII/RII/DOK/ANTRAG/2337762.pdf with 60:38 benzin vs diesel

def calculate_pt_wait_time(fm_amod_row, pt_row):
    """Calculate PT wait time for a given request based on its sub-trips.
    Defines wait time as: (PT Pick-up - AMoD Drop-off) - Required Transfer Time + Station Waiting Time. (Note: PT Pick-up refers to the source station departure time).
    """
    amod_dropoff_time = fm_amod_row.get(G_RQ_DO, np.nan)
    pt_pickup_time = pt_row.get(G_RQ_PU, np.nan)
    
    pt_offer_str = pt_row.get(G_RQ_OFFERS, None)
    if pt_offer_str is not None and not pd.isna(pt_offer_str):
        pt_station_wait_time = int(re.search(r't_wait:(\d+)', pt_offer_str).group(1))
        source_walking_time = int(re.search(r'source_walking_time:(\d+)', pt_offer_str).group(1))
        pt_wait_time = (pt_pickup_time - amod_dropoff_time) - source_walking_time + pt_station_wait_time
        return max(pt_wait_time, 0)
    return np.nan

def create_parent_user_stats(output_dir, evaluation_start_time=None, evaluation_end_time=None, print_comments=False):
    """
    Pre-processing step: Aggregate sub-trips into parent requests.

    Logic:
    - Group by request_id
    - A parent request is "served" ONLY if all its mandatory legs are served
    - Sum/aggregate metrics from sub-trips to parent level
    - Save detailed leg information for analysis

    :param output_dir: scenario output directory
    :param evaluation_start_time: start time filter
    :param evaluation_end_time: end time filter
    :param print_comments: print status messages
    :return: parent_df - DataFrame with aggregated parent request data
    """
    if print_comments:
        print("Creating parent user stats from sub-trips...")

    # Read original user stats
    user_stats_file = os.path.join(output_dir, "1_user-stats.csv")
    df = pd.read_csv(user_stats_file)

    # Apply time filters
    if evaluation_start_time is not None:
        df = df[df[G_RQ_TIME] >= evaluation_start_time]
    if evaluation_end_time is not None:
        df = df[df[G_RQ_TIME] < evaluation_end_time]

    # Separate parent requests and sub-trips
    parent_df = df[df[G_RQ_IS_PARENT_REQUEST] == True].copy()
    subtrip_df = df[df[G_RQ_IS_PARENT_REQUEST] == False].copy()

    if print_comments:
        print(f"  Found {len(parent_df)} parent requests and {len(subtrip_df)} sub-trips")

    # For each parent request, aggregate sub-trip information
    parent_rows = []

    for parent_idx, parent_row in parent_df.iterrows():
        request_id = parent_row[G_RQ_ID]
        modal_state = parent_row.get(G_RQ_MODAL_STATE_VALUE, np.nan)

        # Get all sub-trips for this request
        sub_trips = subtrip_df[subtrip_df[G_RQ_ID] == request_id]

        # Create aggregated parent row
        parent_agg = parent_row.to_dict()

        # Determine if request is served
        # For MONOMODAL (0), parent itself determines if served
        # For intermodal (FM=1, LM=2, FLM=3), check if all mandatory sub-trips are served
        is_served = False
        total_fare = 0
        total_wait_time = 0
        total_travel_time = 0

        # AMoD-only metrics (excluding PT segments for comparability with standard evaluation)
        amod_fare = 0
        amod_wait_time = 0  # Only AMoD wait time (initial wait + LM wait if applicable)
        amod_travel_time = 0  # Only AMoD in-vehicle time
        amod_direct_distance = 0  # Direct distance for AMoD segments

        # Store leg-specific information
        leg_info = {}

        if modal_state == RQ_MODAL_STATE.MONOMODAL.value:
            # MONOMODAL: parent row has all information
            is_served = not pd.isna(parent_row.get(G_RQ_PU))
            if is_served:
                total_fare = parent_row.get(G_RQ_FARE, 0)
                total_wait_time = parent_row.get(G_RQ_PU, 0) - parent_row.get(G_RQ_TIME, 0)
                total_travel_time = parent_row.get(G_RQ_DO, 0) - parent_row.get(G_RQ_PU, 0)
                # AMoD-only metrics (same as total for MONOMODAL)
                amod_fare = total_fare
                amod_wait_time = total_wait_time
                amod_travel_time = total_travel_time
                amod_direct_distance = parent_row.get(G_RQ_DRD, 0)

        elif modal_state == RQ_MODAL_STATE.FIRSTMILE.value:
            # FM: Need FM_AMOD (1) and FM_PT (2)
            fm_amod = sub_trips[sub_trips[G_RQ_SUB_TRIP_ID] == RQ_SUB_TRIP_ID.FM_AMOD.value]
            fm_pt = sub_trips[sub_trips[G_RQ_SUB_TRIP_ID] == RQ_SUB_TRIP_ID.FM_PT.value]

            if len(fm_amod) > 0 and len(fm_pt) > 0:
                fm_amod_row = fm_amod.iloc[0]
                fm_pt_row = fm_pt.iloc[0]

                # Both legs must be served
                fm_amod_served = not pd.isna(fm_amod_row.get(G_RQ_PU))
                fm_pt_served = not pd.isna(fm_pt_row.get(G_RQ_PU))
                is_served = fm_amod_served and fm_pt_served

                if is_served:
                    # Aggregate metrics
                    total_fare = fm_amod_row.get(G_RQ_FARE, 0) + fm_pt_row.get(G_RQ_FARE, 0)

                    # Wait time for FM_AMOD leg
                    fm_amod_wait = fm_amod_row.get(G_RQ_PU, 0) - fm_amod_row.get(G_RQ_TIME, 0)

                    # Wait time for PT leg (waiting for PT after AMoD dropoff)
                    fm_pt_wait = calculate_pt_wait_time(fm_amod_row, fm_pt_row)

                    total_wait_time = fm_amod_wait + fm_pt_wait

                    # Total travel time from AMoD pickup to PT drop-off
                    total_travel_time = fm_pt_row.get(G_RQ_DO, 0) - fm_amod_row.get(G_RQ_PU, 0)

                    # AMoD-only metrics (FM has one AMoD leg)
                    amod_fare = fm_amod_row.get(G_RQ_FARE, 0)
                    amod_wait_time = fm_amod_wait  # Only initial AMoD wait
                    amod_travel_time = fm_amod_row.get(G_RQ_DO, 0) - fm_amod_row.get(G_RQ_PU, 0)
                    amod_direct_distance = fm_amod_row.get(G_RQ_DRD, 0)

                    # Store leg info
                    leg_info['fm_amod_pu'] = fm_amod_row.get(G_RQ_PU)
                    leg_info['fm_amod_do'] = fm_amod_row.get(G_RQ_DO)
                    leg_info['fm_amod_wait'] = fm_amod_wait
                    leg_info['fm_amod_fare'] = fm_amod_row.get(G_RQ_FARE, 0)
                    leg_info['fm_amod_travel_time'] = amod_travel_time
                    leg_info['fm_amod_direct_distance'] = amod_direct_distance
                    leg_info['fm_pt_pu'] = fm_pt_row.get(G_RQ_PU)
                    leg_info['fm_pt_do'] = fm_pt_row.get(G_RQ_DO)
                    leg_info['fm_pt_wait'] = fm_pt_wait
                    leg_info['fm_pt_fare'] = fm_pt_row.get(G_RQ_FARE, 0)

        elif modal_state == RQ_MODAL_STATE.LASTMILE.value:
            # LM: Need LM_PT (3) and LM_AMOD (4)
            lm_pt = sub_trips[sub_trips[G_RQ_SUB_TRIP_ID] == RQ_SUB_TRIP_ID.LM_PT.value]
            lm_amod = sub_trips[sub_trips[G_RQ_SUB_TRIP_ID] == RQ_SUB_TRIP_ID.LM_AMOD.value]

            if len(lm_pt) > 0 and len(lm_amod) > 0:
                lm_pt_row = lm_pt.iloc[0]
                lm_amod_row = lm_amod.iloc[0]

                lm_pt_served = not pd.isna(lm_pt_row.get(G_RQ_PU))
                lm_amod_served = not pd.isna(lm_amod_row.get(G_RQ_PU))
                is_served = lm_pt_served and lm_amod_served

                if is_served:
                    total_fare = lm_pt_row.get(G_RQ_FARE, 0) + lm_amod_row.get(G_RQ_FARE, 0)

                    # Wait time for PT leg
                    pt_offer_str = lm_pt_row.get(G_RQ_OFFERS, None)
                    lm_pt_wait = int(re.search(r't_wait:(\d+)', pt_offer_str).group(1))

                    # Wait time for LM AMoD (waiting for AMoD)
                    lm_pt_target_walking_time = int(re.search(r'target_walking_time:(\d+)', pt_offer_str).group(1))
                    lm_amod_wait = lm_amod_row.get(G_RQ_PU, 0) - lm_pt_row.get(G_RQ_DO, 0) - lm_pt_target_walking_time

                    total_wait_time = lm_pt_wait + lm_amod_wait
                    # Total travel time from PT pickup to AMoD drop-off
                    total_travel_time = lm_amod_row.get(G_RQ_DO, 0) - lm_pt_row.get(G_RQ_PU, 0)

                    # AMoD-only metrics (LM has one AMoD leg)
                    amod_fare = lm_amod_row.get(G_RQ_FARE, 0)
                    amod_wait_time = lm_amod_wait  # AMoD wait after PT
                    lm_amod_travel_time = lm_amod_row.get(G_RQ_DO, 0) - lm_amod_row.get(G_RQ_PU, 0)
                    amod_travel_time = lm_amod_travel_time
                    amod_direct_distance = lm_amod_row.get(G_RQ_DRD, 0)

                    # Store leg info
                    leg_info['lm_pt_pu'] = lm_pt_row.get(G_RQ_PU)
                    leg_info['lm_pt_do'] = lm_pt_row.get(G_RQ_DO)
                    leg_info['lm_pt_wait'] = lm_pt_wait
                    leg_info['lm_pt_fare'] = lm_pt_row.get(G_RQ_FARE, 0)
                    leg_info['lm_amod_pu'] = lm_amod_row.get(G_RQ_PU)
                    leg_info['lm_amod_do'] = lm_amod_row.get(G_RQ_DO)
                    leg_info['lm_amod_wait'] = lm_amod_wait
                    leg_info['lm_amod_fare'] = lm_amod_row.get(G_RQ_FARE, 0)
                    leg_info['lm_amod_travel_time'] = lm_amod_travel_time
                    leg_info['lm_amod_direct_distance'] = amod_direct_distance

        elif modal_state == RQ_MODAL_STATE.FIRSTLASTMILE.value:
            # FLM: Need FLM_AMOD_0 (5), FLM_PT (6), FLM_AMOD_1 (7)
            flm_amod_0 = sub_trips[sub_trips[G_RQ_SUB_TRIP_ID] == RQ_SUB_TRIP_ID.FLM_AMOD_0.value]
            flm_pt = sub_trips[sub_trips[G_RQ_SUB_TRIP_ID] == RQ_SUB_TRIP_ID.FLM_PT.value]
            flm_amod_1 = sub_trips[sub_trips[G_RQ_SUB_TRIP_ID] == RQ_SUB_TRIP_ID.FLM_AMOD_1.value]

            if len(flm_amod_0) > 0 and len(flm_pt) > 0 and len(flm_amod_1) > 0:
                flm_amod_0_row = flm_amod_0.iloc[0]
                flm_pt_row = flm_pt.iloc[0]
                flm_amod_1_row = flm_amod_1.iloc[0]

                flm_amod_0_served = not pd.isna(flm_amod_0_row.get(G_RQ_PU))
                flm_pt_served = not pd.isna(flm_pt_row.get(G_RQ_PU))
                flm_amod_1_served = not pd.isna(flm_amod_1_row.get(G_RQ_PU))
                is_served = flm_amod_0_served and flm_pt_served and flm_amod_1_served

                if is_served:
                    total_fare = (flm_amod_0_row.get(G_RQ_FARE, 0) +
                                 flm_pt_row.get(G_RQ_FARE, 0) +
                                 flm_amod_1_row.get(G_RQ_FARE, 0))

                    # Wait times
                    flm_amod_0_wait = flm_amod_0_row.get(G_RQ_PU, 0) - flm_amod_0_row.get(G_RQ_TIME, 0)
                    flm_pt_wait = calculate_pt_wait_time(flm_amod_0_row, flm_pt_row)
                    flm_pt_offer_str = flm_pt_row.get(G_RQ_OFFERS, None)
                    flm_pt_target_walking_time = int(re.search(r'target_walking_time:(\d+)', flm_pt_offer_str).group(1))
                    # Wait time for FLM_AMOD_1 (waiting for AMoD): LM AMoD pickup - PT drop-off - target walking time
                    flm_amod_1_wait = flm_amod_1_row.get(G_RQ_PU, 0) - flm_pt_row.get(G_RQ_DO, 0) - flm_pt_target_walking_time

                    total_wait_time = flm_amod_0_wait + flm_pt_wait + flm_amod_1_wait
                    # Total travel time from first AMoD pickup to last AMoD drop-off
                    total_travel_time = flm_amod_1_row.get(G_RQ_DO, 0) - flm_amod_0_row.get(G_RQ_PU, 0)

                    # AMoD-only metrics (FLM has two AMoD legs)
                    amod_fare = flm_amod_0_row.get(G_RQ_FARE, 0) + flm_amod_1_row.get(G_RQ_FARE, 0)
                    amod_wait_time = flm_amod_0_wait + flm_amod_1_wait  # Both AMoD wait times
                    flm_amod_0_travel_time = flm_amod_0_row.get(G_RQ_DO, 0) - flm_amod_0_row.get(G_RQ_PU, 0)
                    flm_amod_1_travel_time = flm_amod_1_row.get(G_RQ_DO, 0) - flm_amod_1_row.get(G_RQ_PU, 0)
                    amod_travel_time = flm_amod_0_travel_time + flm_amod_1_travel_time
                    amod_direct_distance = flm_amod_0_row.get(G_RQ_DRD, 0) + flm_amod_1_row.get(G_RQ_DRD, 0)

                    # Store leg info
                    leg_info['flm_amod_0_pu'] = flm_amod_0_row.get(G_RQ_PU)
                    leg_info['flm_amod_0_do'] = flm_amod_0_row.get(G_RQ_DO)
                    leg_info['flm_amod_0_wait'] = flm_amod_0_wait
                    leg_info['flm_amod_0_fare'] = flm_amod_0_row.get(G_RQ_FARE, 0)
                    leg_info['flm_amod_0_travel_time'] = flm_amod_0_travel_time
                    leg_info['flm_amod_0_direct_distance'] = flm_amod_0_row.get(G_RQ_DRD, 0)
                    leg_info['flm_pt_pu'] = flm_pt_row.get(G_RQ_PU)
                    leg_info['flm_pt_do'] = flm_pt_row.get(G_RQ_DO)
                    leg_info['flm_pt_wait'] = flm_pt_wait
                    leg_info['flm_pt_fare'] = flm_pt_row.get(G_RQ_FARE, 0)
                    leg_info['flm_amod_1_pu'] = flm_amod_1_row.get(G_RQ_PU)
                    leg_info['flm_amod_1_do'] = flm_amod_1_row.get(G_RQ_DO)
                    leg_info['flm_amod_1_wait'] = flm_amod_1_wait
                    leg_info['flm_amod_1_fare'] = flm_amod_1_row.get(G_RQ_FARE, 0)
                    leg_info['flm_amod_1_travel_time'] = flm_amod_1_travel_time
                    leg_info['flm_amod_1_direct_distance'] = flm_amod_1_row.get(G_RQ_DRD, 0)

        # Update parent aggregation with computed values
        parent_agg['is_served'] = is_served
        parent_agg['total_fare'] = total_fare if is_served else np.nan
        parent_agg['total_wait_time'] = total_wait_time if is_served else np.nan
        parent_agg['total_travel_time'] = total_travel_time if is_served else np.nan

        # AMoD-only metrics (excluding PT segments for comparability with standard evaluation)
        parent_agg['amod_fare'] = amod_fare if is_served else np.nan
        parent_agg['amod_wait_time'] = amod_wait_time if is_served else np.nan
        parent_agg['amod_travel_time'] = amod_travel_time if is_served else np.nan
        parent_agg['amod_direct_distance'] = amod_direct_distance if is_served else np.nan

        # Add leg info as columns
        for key, val in leg_info.items():
            parent_agg[key] = val

        parent_rows.append(parent_agg)

    # Create parent DataFrame
    parent_result_df = pd.DataFrame(parent_rows)

    # Save to file
    parent_output_file = os.path.join(output_dir, "1_user-stats_parent.csv")
    parent_result_df.to_csv(parent_output_file, index=False)

    if print_comments:
        print(f"  Saved parent user stats to: {parent_output_file}")
        print(f"  Served requests: {parent_result_df['is_served'].sum()} / {len(parent_result_df)}")

    return parent_result_df


def categorize_modal_state(modal_state_value):
    """Categorize request based on modal state value."""
    if pd.isna(modal_state_value):
        return 'Unknown'
    modal_state_value = int(modal_state_value)
    if modal_state_value == RQ_MODAL_STATE.MONOMODAL.value:
        return 'DRT_only'
    elif modal_state_value == RQ_MODAL_STATE.FIRSTMILE.value:
        return 'FM'
    elif modal_state_value == RQ_MODAL_STATE.LASTMILE.value:
        return 'LM'
    elif modal_state_value == RQ_MODAL_STATE.FIRSTLASTMILE.value:
        return 'FLM'
    elif modal_state_value == RQ_MODAL_STATE.PT.value:
        return 'PT_only'
    else:
        return 'Unknown'


def intermodal_evaluation(output_dir, evaluation_start_time=None, evaluation_end_time=None, print_comments=False, dir_names_in={}):
    """
    Main intermodal evaluation function.

    Follows the same pattern as standard_evaluation but:
    1. Pre-processes user stats to create parent request aggregations
    2. Calculates standard metrics on parent requests
    3. Adds intermodal-specific metrics

    :param output_dir: scenario output directory
    :param evaluation_start_time: start time filter
    :param evaluation_end_time: end time filter
    :param print_comments: print status messages
    :param dir_names_in: directory dictionary (optional)
    :return: result DataFrame
    """
    if not os.path.isdir(output_dir):
        raise IOError(f"Could not find result directory {output_dir}!")

    # Load scenario configuration
    scenario_parameters, list_operator_attributes, _ = load_scenario_inputs(output_dir)
    dir_names = get_directory_dict(scenario_parameters, list_operator_attributes, abs_fleetpy_dir=MAIN_DIR)
    if dir_names_in:
        dir_names = dir_names_in

    # Evaluation interval
    if evaluation_start_time is None and scenario_parameters.get(G_EVAL_INT_START) is not None:
        evaluation_start_time = int(scenario_parameters[G_EVAL_INT_START])
    if evaluation_end_time is None and scenario_parameters.get(G_EVAL_INT_END) is not None:
        evaluation_end_time = int(scenario_parameters[G_EVAL_INT_END])

    # Vehicle type data
    from src.evaluation.standard import create_vehicle_type_db, read_op_output_file, avg_in_vehicle_distance, shared_rides
    veh_type_db = create_vehicle_type_db(dir_names[G_DIR_VEH])
    veh_type_stats = pd.read_csv(os.path.join(output_dir, "2_vehicle_types.csv"))

    if print_comments:
        print(f"Evaluating {scenario_parameters[G_SCENARIO_NAME]}")
        print("="*80)

    # Step 1: Create parent user stats
    parent_user_stats = create_parent_user_stats(output_dir, evaluation_start_time, evaluation_end_time, print_comments)

    # Add passenger column if needed
    if G_RQ_PAX not in parent_user_stats.columns:
        parent_user_stats[G_RQ_PAX] = 1

    # Filter to only served requests for metrics
    served_requests = parent_user_stats[parent_user_stats['is_served'] == True].copy()

    if print_comments:
        print(f"\nCalculating metrics for {len(served_requests)} served requests...")

    # Categorize requests by modal state
    served_requests['modal_category'] = served_requests[G_RQ_MODAL_STATE_VALUE].apply(categorize_modal_state)

    # Total counts
    total_requests = len(parent_user_stats)
    total_served = len(served_requests)
    total_pax = parent_user_stats[G_RQ_PAX].sum()
    total_served_pax = served_requests[G_RQ_PAX].sum()

    # Category breakdown
    category_counts = served_requests.groupby('modal_category').size()
    category_pax = served_requests.groupby('modal_category')[G_RQ_PAX].sum()

    # Calculate service rates
    service_rate_overall = total_served / total_requests * 100 if total_requests > 0 else 0
    service_rate_pax = total_served_pax / total_pax * 100 if total_pax > 0 else 0

    # Service rates by category
    service_rates = {}
    for category in ['DRT_only', 'FM', 'LM', 'FLM', 'PT_only']:
        cat_total = len(parent_user_stats[parent_user_stats[G_RQ_MODAL_STATE_VALUE].apply(categorize_modal_state) == category])
        cat_served = category_counts.get(category, 0)
        service_rates[f'{category}_service_rate'] = cat_served / cat_total * 100 if cat_total > 0 else np.nan
        service_rates[f'{category}_count'] = cat_served

    # Calculate uncatchable PT statistics (requests that missed their PT connection after FM leg)
    uncatchable_stats = {}
    if G_RQ_UNCATCHABLE_PT in parent_user_stats.columns:
        # Count uncatchable requests by category (only FM and FLM can be uncatchable)
        fm_uncatchable = parent_user_stats[
            (parent_user_stats[G_RQ_MODAL_STATE_VALUE].apply(categorize_modal_state) == 'FM') &
            (parent_user_stats[G_RQ_UNCATCHABLE_PT] == True)
        ]
        flm_uncatchable = parent_user_stats[
            (parent_user_stats[G_RQ_MODAL_STATE_VALUE].apply(categorize_modal_state) == 'FLM') &
            (parent_user_stats[G_RQ_UNCATCHABLE_PT] == True)
        ]
        total_uncatchable = len(fm_uncatchable) + len(flm_uncatchable)

        uncatchable_stats['FM_uncatchable_count'] = len(fm_uncatchable)
        uncatchable_stats['FLM_uncatchable_count'] = len(flm_uncatchable)
        uncatchable_stats['total_uncatchable_count'] = total_uncatchable

        # Calculate uncatchable rate (as % of FM+FLM requests)
        fm_total = len(parent_user_stats[parent_user_stats[G_RQ_MODAL_STATE_VALUE].apply(categorize_modal_state) == 'FM'])
        flm_total = len(parent_user_stats[parent_user_stats[G_RQ_MODAL_STATE_VALUE].apply(categorize_modal_state) == 'FLM'])
        intermodal_with_fm_total = fm_total + flm_total

        uncatchable_stats['FM_uncatchable_rate'] = len(fm_uncatchable) / fm_total * 100 if fm_total > 0 else np.nan
        uncatchable_stats['FLM_uncatchable_rate'] = len(flm_uncatchable) / flm_total * 100 if flm_total > 0 else np.nan
        uncatchable_stats['total_uncatchable_rate'] = total_uncatchable / intermodal_with_fm_total * 100 if intermodal_with_fm_total > 0 else np.nan

        if print_comments and total_uncatchable > 0:
            print(f"  Uncatchable PT requests: {total_uncatchable} (FM: {len(fm_uncatchable)}, FLM: {len(flm_uncatchable)})")
    else:
        # No uncatchable_pt column, set defaults
        uncatchable_stats = {
            'FM_uncatchable_count': 0,
            'FLM_uncatchable_count': 0,
            'total_uncatchable_count': 0,
            'FM_uncatchable_rate': np.nan,
            'FLM_uncatchable_rate': np.nan,
            'total_uncatchable_rate': np.nan
        }

    # Calculate PT wait times for FM and FLM
    fm_requests = served_requests[served_requests['modal_category'] == 'FM']
    flm_requests = served_requests[served_requests['modal_category'] == 'FLM']

    pt_wait_time_fm = fm_requests['fm_pt_wait'].mean() if len(fm_requests) > 0 and 'fm_pt_wait' in fm_requests.columns else np.nan
    pt_wait_time_flm = flm_requests['flm_pt_wait'].mean() if len(flm_requests) > 0 and 'flm_pt_wait' in flm_requests.columns else np.nan

    # Combined PT wait time
    pt_waits = []
    if 'fm_pt_wait' in fm_requests.columns:
        pt_waits.extend(fm_requests['fm_pt_wait'].dropna().tolist())
    if 'flm_pt_wait' in flm_requests.columns:
        pt_waits.extend(flm_requests['flm_pt_wait'].dropna().tolist())
    pt_wait_time_combined = np.mean(pt_waits) if len(pt_waits) > 0 else np.nan

    # Calculate LM wait times for LM and FLM
    lm_requests = served_requests[served_requests['modal_category'] == 'LM']

    lm_wait_time_lm = lm_requests['lm_amod_wait'].mean() if len(lm_requests) > 0 and 'lm_amod_wait' in lm_requests.columns else np.nan
    lm_wait_time_flm = flm_requests['flm_amod_1_wait'].mean() if len(flm_requests) > 0 and 'flm_amod_1_wait' in flm_requests.columns else np.nan

    # Combined LM wait time
    lm_waits = []
    if 'lm_amod_wait' in lm_requests.columns:
        lm_waits.extend(lm_requests['lm_amod_wait'].dropna().tolist())
    if 'flm_amod_1_wait' in flm_requests.columns:
        lm_waits.extend(flm_requests['flm_amod_1_wait'].dropna().tolist())
    lm_wait_time_combined = np.mean(lm_waits) if len(lm_waits) > 0 else np.nan

    # Overall metrics
    avg_wait_time = served_requests['total_wait_time'].mean()
    med_wait_time = served_requests['total_wait_time'].median()
    quantile_90_wait_time = served_requests['total_wait_time'].quantile(q=0.9)
    avg_travel_time = served_requests['total_travel_time'].mean()
    total_revenue = served_requests['total_fare'].sum()

    # AMoD-only metrics (excluding PT segments for comparability with standard evaluation)
    amod_avg_wait_time = served_requests['amod_wait_time'].mean()
    amod_med_wait_time = served_requests['amod_wait_time'].median()
    amod_quantile_90_wait_time = served_requests['amod_wait_time'].quantile(q=0.9)
    amod_avg_travel_time = served_requests['amod_travel_time'].mean()
    amod_total_revenue = served_requests['amod_fare'].sum()
    amod_total_direct_distance = served_requests['amod_direct_distance'].sum() / 1000.0  # Convert to km

    # Detour time calculation (AMoD segments only, excluding PT)
    # Detour = actual_travel_time - direct_route_time - boarding_time
    # Get boarding time from operator attributes (will be set later when processing operators)
    boarding_time = scenario_parameters.get("op_const_boarding_time", 30)  # Default 30s

    # Calculate detour for each request based on AMoD segments
    # For requests with direct_route_time available
    if G_RQ_DRT in served_requests.columns:
        # MONOMODAL: use parent's direct route time
        monomodal_mask = served_requests['modal_category'] == 'DRT_only'
        served_requests.loc[monomodal_mask, 'amod_direct_route_time'] = served_requests.loc[monomodal_mask, G_RQ_DRT]

    # Calculate detour time for AMoD segments
    served_requests['amod_detour_time'] = served_requests['amod_travel_time'] - served_requests.get('amod_direct_route_time', served_requests['amod_travel_time']) - boarding_time
    # For intermodal, estimate direct route time from direct distance (assuming avg speed ~30 km/h = 8.33 m/s)
    avg_speed_ms = 8.33  # m/s, approximately 30 km/h
    served_requests.loc[served_requests['amod_detour_time'].isna(), 'amod_detour_time'] = (
        served_requests.loc[served_requests['amod_detour_time'].isna(), 'amod_travel_time'] -
        served_requests.loc[served_requests['amod_detour_time'].isna(), 'amod_direct_distance'] / avg_speed_ms - boarding_time
    )

    amod_avg_detour_time = served_requests['amod_detour_time'].mean()

    # Relative detour (percentage)
    served_requests['amod_rel_detour'] = (
        (served_requests['amod_travel_time'] - boarding_time - served_requests['amod_direct_distance'] / avg_speed_ms) /
        (served_requests['amod_direct_distance'] / avg_speed_ms)
    ) * 100.0
    amod_avg_rel_detour = served_requests['amod_rel_detour'].mean()

    # Standard metrics (matching standard_eval.csv format)
    result_dict = {
        'operator_id': -3,  # Intermodal operator
        'number users': total_served,
        'number travelers': total_served_pax,
        'modal split': service_rate_pax / 100,
        'modal split rq': service_rate_overall / 100,
        'Service_Rate [%]': service_rate_overall,
        'Service_Rate_Pax [%]': service_rate_pax,

        # Category breakdown
        'DRT_only_count': service_rates.get('DRT_only_count', 0),
        'FM_count': service_rates.get('FM_count', 0),
        'LM_count': service_rates.get('LM_count', 0),
        'FLM_count': service_rates.get('FLM_count', 0),
        'PT_only_count': service_rates.get('PT_only_count', 0),

        'DRT_only_service_rate [%]': service_rates.get('DRT_only_service_rate', np.nan),
        'FM_service_rate [%]': service_rates.get('FM_service_rate', np.nan),
        'LM_service_rate [%]': service_rates.get('LM_service_rate', np.nan),
        'FLM_service_rate [%]': service_rates.get('FLM_service_rate', np.nan),
        'PT_only_service_rate [%]': service_rates.get('PT_only_service_rate', np.nan),

        # Uncatchable PT statistics (requests that missed their PT connection after FM leg)
        'FM_uncatchable_count': uncatchable_stats.get('FM_uncatchable_count', 0),
        'FLM_uncatchable_count': uncatchable_stats.get('FLM_uncatchable_count', 0),
        'total_uncatchable_count': uncatchable_stats.get('total_uncatchable_count', 0),
        'FM_uncatchable_rate [%]': uncatchable_stats.get('FM_uncatchable_rate', np.nan),
        'FLM_uncatchable_rate [%]': uncatchable_stats.get('FLM_uncatchable_rate', np.nan),
        'total_uncatchable_rate [%]': uncatchable_stats.get('total_uncatchable_rate', np.nan),

        # Wait times (total, including PT wait)
        'waiting time': avg_wait_time,
        'waiting time (median)': med_wait_time,
        'waiting time (90% quantile)': quantile_90_wait_time,
        'PT_Wait_Time_FM [s]': pt_wait_time_fm,
        'PT_Wait_Time_FLM [s]': pt_wait_time_flm,
        'PT_Wait_Time_Combined [s]': pt_wait_time_combined,
        'LM_Wait_Time_LM [s]': lm_wait_time_lm,
        'LM_Wait_Time_FLM [s]': lm_wait_time_flm,
        'LM_Wait_Time_Combined [s]': lm_wait_time_combined,

        # AMoD-only metrics (excluding PT, for comparability with standard evaluation)
        'amod_waiting_time': amod_avg_wait_time,
        'amod_waiting_time (median)': amod_med_wait_time,
        'amod_waiting_time (90% quantile)': amod_quantile_90_wait_time,
        'amod_travel_time': amod_avg_travel_time,
        'amod_detour_time': amod_avg_detour_time,
        'amod_rel_detour [%]': amod_avg_rel_detour,
        'amod_revenue': amod_total_revenue,
        'amod_customer_direct_distance [km]': amod_total_direct_distance,

        # Travel metrics
        'travel time': avg_travel_time,
        'mod revenue': total_revenue,
    }

    # Vehicle-level analysis for AMoD operators
    if print_comments:
        print("\nAnalyzing vehicle operations...")

    # Process each AMoD operator
    for op_id in range(scenario_parameters.get(G_NR_OPERATORS, 0)):
        try:
            op_vehicle_df = read_op_output_file(output_dir, op_id, evaluation_start_time, evaluation_end_time)
            operator_attributes = list_operator_attributes[int(op_id)]

            if print_comments:
                print(f"  Processing operator {op_id}: {op_vehicle_df.shape[0]} vehicle route legs")

            # Fleet metrics
            n_vehicles = veh_type_stats[veh_type_stats[G_V_OP_ID] == op_id].shape[0]
            sim_end_time = scenario_parameters["end_time"]
            simulation_time = scenario_parameters["end_time"] - scenario_parameters["start_time"]

            # Utilization
            op_vehicle_df["VRL_end_sim_end_time"] = np.minimum(op_vehicle_df[G_VR_LEG_END_TIME], sim_end_time)
            op_vehicle_df["VRL_start_sim_end_time"] = np.minimum(op_vehicle_df[G_VR_LEG_START_TIME], sim_end_time)
            utilized_veh_df = op_vehicle_df[(op_vehicle_df["status"] != VRL_STATES.OUT_OF_SERVICE.display_name) &
                                           (op_vehicle_df["status"] != VRL_STATES.CHARGING.display_name)]
            utilization_time = utilized_veh_df["VRL_end_sim_end_time"].sum() - utilized_veh_df["VRL_start_sim_end_time"].sum()
            unutilized_veh_df = op_vehicle_df[(op_vehicle_df["status"] == VRL_STATES.OUT_OF_SERVICE.display_name) |
                                              (op_vehicle_df["status"] == VRL_STATES.CHARGING.display_name)]
            unutilized_time = unutilized_veh_df["VRL_end_sim_end_time"].sum() - unutilized_veh_df["VRL_start_sim_end_time"].sum()

            fleet_utilization = 100 * (utilization_time / (n_vehicles * simulation_time - unutilized_time)) if (n_vehicles * simulation_time - unutilized_time) > 0 else 0

            # Distance metrics
            total_km = op_vehicle_df[G_VR_LEG_DISTANCE].sum() / 1000.0

            def weight_ob_pax(entries):
                try:
                    return entries[G_VR_NR_PAX] * entries[G_VR_LEG_DISTANCE]
                except:
                    return 0.0

            op_vehicle_df["weighted_ob_pax"] = op_vehicle_df.apply(weight_ob_pax, axis=1)
            distance_avg_occupancy = op_vehicle_df["weighted_ob_pax"].sum() / op_vehicle_df[G_VR_LEG_DISTANCE].sum() if op_vehicle_df[G_VR_LEG_DISTANCE].sum() > 0 else 0

            empty_df = op_vehicle_df[op_vehicle_df[G_VR_OB_RID].isnull()]
            empty_vkm = empty_df[G_VR_LEG_DISTANCE].sum() / 1000.0 / total_km * 100.0 if total_km > 0 else 0

            # Repositioning VKM (new)
            repositioning_df = empty_df[empty_df[G_VR_STATUS] == "reposition"]
            repositioning_vkm = repositioning_df[G_VR_LEG_DISTANCE].sum() / 1000.0 / total_km * 100.0 if total_km > 0 else 0

            # Revenue metrics
            rev_df = op_vehicle_df[op_vehicle_df["status"].isin([x.display_name for x in G_REVENUE_STATUS])]
            vehicle_revenue_hours = (rev_df["VRL_end_sim_end_time"].sum() - rev_df["VRL_start_sim_end_time"].sum()) / 3600.0

            # Rides per vehicle revenue hours (new)
            rides_per_veh_rev_hours = total_served_pax / vehicle_revenue_hours if vehicle_revenue_hours > 0 else 0
            rides_per_veh_rev_hours_rq = total_served / vehicle_revenue_hours if vehicle_revenue_hours > 0 else 0

            # Shared rides and customer in-vehicle distance (new)
            op_shared_rides = shared_rides(op_vehicle_df)
            op_customer_in_vehicle_distance = avg_in_vehicle_distance(op_vehicle_df)

            # By-vehicle stats
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

                all_vid_dict[vid] = {
                    "type": vtype_data[G_VTYPE_NAME],
                    "total km": veh_km,
                    "total kWh": veh_kWh,
                    "total CO2 [g]": veh_co2,
                    "fix costs": veh_fix_costs,
                    "total variable costs": veh_var_costs
                }

            # Save vehicle-level stats
            all_vid_df = pd.DataFrame.from_dict(all_vid_dict, orient="index")
            all_vid_df.to_csv(os.path.join(output_dir, f"standard_mod-{op_id}_veh_eval.csv"))

            # Aggregate vehicle metrics
            total_co2 = all_vid_df["total CO2 [g]"].sum() if len(all_vid_df) > 0 else 0
            fix_costs = all_vid_df["fix costs"].sum() if len(all_vid_df) > 0 else 0
            var_costs = all_vid_df["total variable costs"].sum() if len(all_vid_df) > 0 else 0

            # External emission costs (new)
            external_emission_costs = np.rint(EMISSION_CPG * total_co2)

            # Add to result dict with operator prefix
            result_dict[f'op{op_id}_fleet_utilization [%]'] = fleet_utilization
            result_dict[f'op{op_id}_total_vkm'] = total_km
            result_dict[f'op{op_id}_occupancy'] = distance_avg_occupancy
            result_dict[f'op{op_id}_empty_vkm [%]'] = empty_vkm
            result_dict[f'op{op_id}_repositioning_vkm [%]'] = repositioning_vkm
            result_dict[f'op{op_id}_vehicle_revenue_hours'] = vehicle_revenue_hours
            result_dict[f'op{op_id}_rides_per_veh_rev_hours'] = rides_per_veh_rev_hours
            result_dict[f'op{op_id}_rides_per_veh_rev_hours_rq'] = rides_per_veh_rev_hours_rq
            result_dict[f'op{op_id}_total_CO2_emissions [t]'] = total_co2 / 10**6
            result_dict[f'op{op_id}_external_emission_costs'] = external_emission_costs
            result_dict[f'op{op_id}_fix_costs'] = fix_costs
            result_dict[f'op{op_id}_var_costs'] = var_costs
            result_dict[f'op{op_id}_shared_rides [%]'] = op_shared_rides
            result_dict[f'op{op_id}_customer_in_vehicle_distance'] = op_customer_in_vehicle_distance

        except FileNotFoundError:
            if print_comments:
                print(f"  No vehicle data found for operator {op_id}")
            continue

    # Create result DataFrame
    result_df = pd.DataFrame([result_dict], index=['Intermodal']).T
    result_df.columns = ['Intermodal']

    # Save to standard_eval.csv for consistency
    output_file = os.path.join(output_dir, "standard_eval.csv")
    result_df.to_csv(output_file)

    if print_comments:
        print(f"\nEvaluation complete! Results saved to: {output_file}")
        print("="*80)

    return result_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sc = sys.argv[1]
        intermodal_evaluation(sc, print_comments=True)
    else:
        print("Usage: python intermodal.py <output_directory>")
        print("Example: python src/evaluation/intermodal.py studies/example_study/results/example_im_ptbroker")

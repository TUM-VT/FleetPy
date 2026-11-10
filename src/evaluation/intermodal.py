import ast
import os
import sys
import numpy as np
import pandas as pd
import re
from numbers import Integral, Real

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(MAIN_DIR)
from src.misc.globals import *
from src.evaluation.standard import (EMISSION_CPG, ENERGY_EMISSIONS, avg_in_vehicle_distance,
                                     read_op_output_file, shared_rides)

def _get_numeric_offer_parameter(offer_str, parameter):
    """Read a finite offer value, retaining decimal and scientific notation."""
    if not isinstance(offer_str, str):
        return np.nan
    match = re.search(r'(?:^|[;:|])' + re.escape(parameter) + r':([^;|]*)', offer_str)
    if match is None:
        return np.nan
    try:
        value = float(match.group(1))
    except ValueError:
        return np.nan
    return value if np.isfinite(value) else np.nan


def calculate_pt_wait_time(fm_alighting_end_time, pt_row):
    """Calculate PT wait time for a given request based on its sub-trips.
    Use the actual FM alighting end and the source station readiness and wait.
    """
    amod_dropoff_time = fm_alighting_end_time
    pt_pickup_time = pt_row.get(G_RQ_PU, np.nan)
    
    pt_offer_str = pt_row.get(G_RQ_OFFERS, None)
    pt_station_wait_time = _get_numeric_offer_parameter(pt_offer_str, G_OFFER_WAIT)
    source_walking_time = _get_numeric_offer_parameter(pt_offer_str, G_PT_OFFER_SOURCE_WALKING_TIME)
    try:
        pt_wait_time = float(pt_pickup_time) - float(amod_dropoff_time) - source_walking_time + pt_station_wait_time
    except (TypeError, ValueError):
        return np.nan
    return max(pt_wait_time, 0) if np.isfinite(pt_wait_time) else np.nan


def _parse_execution_mode(value):
    """Read a numeric selected mode; an empty decision remains unselected."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    number = float(value)
    if not np.isfinite(number) or not number.is_integer():
        return None
    return int(number) if int(number) in range(RQ_MODAL_STATE.PT.value + 1) else None


def _execution_mode(row):
    return _parse_execution_mode(row[G_RQ_SELECTED_MODE])


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _is_true(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1")
    return bool(value) if pd.notna(value) else False


def _pt_leg_failed(row):
    """Terminal PT cancellation states override any prefilled service times."""
    state = row[G_RQ_PT_LEG_STATE]
    return isinstance(state, str) and state in (
        PT_LEG_STATE.CANCELLED.value, PT_LEG_STATE.MISSED.value)


def _has_journey_failure(row):
    missed_without_rebooking = (_is_true(row[G_RQ_UNCATCHABLE_PT]) and
                                row.get(G_RQ_EI_REBOOKING_STATUS) != EI_REBOOKING_STATE.BOOKED.value)
    return _pt_leg_failed(row) or missed_without_rebooking or _is_true(row[G_RQ_PAYG_INTERRUPTED])


def _completion_horizon(output_dir, completion_end_time):
    """Use the simulation/observation horizon to exclude prefilled future PT ends."""
    limits = [completion_end_time]
    if completion_end_time is None and os.path.isfile(os.path.join(output_dir, G_SC_INP_F)):
        scenario_parameters, _, _ = load_scenario_inputs(output_dir)
        limits.append(scenario_parameters.get(G_SIM_END_TIME))
    finite_limits = [_finite_number(value) for value in limits]
    finite_limits = [value for value in finite_limits if np.isfinite(value)]
    return min(finite_limits) if finite_limits else None


def _leg_is_complete(row, completion_end_time):
    if row is None or _pt_leg_failed(row):
        return False
    pickup = _finite_number(row[G_RQ_PU])
    dropoff = _finite_number(row[G_RQ_DO])
    if not np.isfinite(pickup) or not np.isfinite(dropoff) or dropoff < pickup:
        return False
    egress = _finite_number(row[G_RQ_EGRESS])
    egress = egress if np.isfinite(egress) else 0
    return egress >= 0 and (completion_end_time is None or dropoff + egress <= completion_end_time)


def _selected_legs(parent_row, sub_trips, modal_state):
    """Resolve only the mandatory legs of the accepted itinerary."""
    required_ids = {
        RQ_MODAL_STATE.MONOMODAL.value: (None,),
        RQ_MODAL_STATE.FIRSTMILE.value: (RQ_SUB_TRIP_ID.FM_AMOD.value, RQ_SUB_TRIP_ID.FM_PT.value),
        RQ_MODAL_STATE.LASTMILE.value: (RQ_SUB_TRIP_ID.LM_PT.value, RQ_SUB_TRIP_ID.LM_AMOD.value),
        RQ_MODAL_STATE.FIRSTLASTMILE.value: (
            RQ_SUB_TRIP_ID.FLM_AMOD_0.value, RQ_SUB_TRIP_ID.FLM_PT.value, RQ_SUB_TRIP_ID.FLM_AMOD_1.value),
        RQ_MODAL_STATE.PT.value: (RQ_SUB_TRIP_ID.PT.value,),
    }.get(modal_state, ())
    selected_candidate = parent_row[G_RQ_SELECTED_CANDIDATE_ID]
    rows = {}
    for sub_trip_id in required_ids:
        if sub_trip_id is None:
            rows[sub_trip_id] = parent_row
            continue
        matches = sub_trips[sub_trips[G_RQ_SUB_TRIP_ID] == sub_trip_id]
        matches = matches[matches[G_RQ_SELECTED_CANDIDATE_ID] == selected_candidate]
        rows[sub_trip_id] = matches.iloc[0] if len(matches) == 1 else None
    return rows


def _connections_are_feasible(modal_state, legs, fm_alighting_end_time):
    """Require finite PT transfer times for the accepted connecting legs."""
    fm_ids = {
        RQ_MODAL_STATE.FIRSTMILE.value: (RQ_SUB_TRIP_ID.FM_AMOD.value, RQ_SUB_TRIP_ID.FM_PT.value),
        RQ_MODAL_STATE.FIRSTLASTMILE.value: (RQ_SUB_TRIP_ID.FLM_AMOD_0.value, RQ_SUB_TRIP_ID.FLM_PT.value),
    }.get(modal_state)
    if fm_ids:
        fm, pt = (legs.get(sub_trip_id) for sub_trip_id in fm_ids)
        if fm is None or pt is None:
            return False
        walking = _get_numeric_offer_parameter(pt[G_RQ_OFFERS], G_PT_OFFER_SOURCE_WALKING_TIME)
        waiting = _get_numeric_offer_parameter(pt[G_RQ_OFFERS], G_OFFER_WAIT)
        latest_origin = _finite_number(pt[G_RQ_PU]) + waiting - walking
        fm_dropoff = _finite_number(fm_alighting_end_time)
        if not np.isfinite(latest_origin) or not np.isfinite(fm_dropoff) or fm_dropoff > latest_origin:
            return False
    lm_ids = {
        RQ_MODAL_STATE.LASTMILE.value: (RQ_SUB_TRIP_ID.LM_PT.value, RQ_SUB_TRIP_ID.LM_AMOD.value),
        RQ_MODAL_STATE.FIRSTLASTMILE.value: (RQ_SUB_TRIP_ID.FLM_PT.value, RQ_SUB_TRIP_ID.FLM_AMOD_1.value),
    }.get(modal_state)
    if lm_ids:
        pt, lm = (legs.get(sub_trip_id) for sub_trip_id in lm_ids)
        if pt is None or lm is None:
            return False
        walking = _get_numeric_offer_parameter(pt[G_RQ_OFFERS], G_PT_OFFER_TARGET_WALKING_TIME)
        ready = _finite_number(pt[G_RQ_DO]) + walking
        lm_pickup = _finite_number(lm[G_RQ_PU])
        if not np.isfinite(ready) or not np.isfinite(lm_pickup) or lm_pickup < ready:
            return False
    return True


def _fm_alighting_end_time(output_dir, parent_row, fm_row, cache):
    """Resolve the selected FM request's completed vehicle stop."""
    def integer_id(value, allow_negative=False):
        if isinstance(value, (bool, np.bool_)):
            return None
        if isinstance(value, Integral):
            result = int(value)
        elif isinstance(value, str) and value.removeprefix('-').isdecimal():
            result = int(value)
        elif isinstance(value, Real) and np.isfinite(value) and float(value).is_integer():
            result = int(value)
        else:
            return None
        return result if allow_negative or result >= 0 else None

    operator_id = integer_id(fm_row.get(G_RQ_OP_ID))
    vehicle_id = integer_id(fm_row.get(G_RQ_VID))
    request_id = integer_id(parent_row.get(G_RQ_ID), allow_negative=True)
    sub_trip_id = integer_id(fm_row.get(G_RQ_SUB_TRIP_ID))
    start_time = _finite_number(fm_row.get(G_RQ_DO))
    values = (operator_id, vehicle_id, request_id, sub_trip_id)
    if any(value is None for value in values):
        raise ValueError(f"Invalid FM operator, vehicle or request identity: {values}")
    rid = f"{request_id}_{sub_trip_id}"
    if operator_id not in cache:
        path = os.path.join(output_dir, f"2-{operator_id}_op-stats.csv")
        columns = [G_V_OP_ID, G_V_VID, G_VR_STATUS, G_VR_LEG_START_TIME,
                   G_VR_LEG_END_TIME, G_VR_ALIGHTING_RID]
        try:
            records = pd.read_csv(path, usecols=columns, dtype={G_VR_ALIGHTING_RID: str},
                                  float_precision="round_trip")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read FM alighting events from {path} for {rid}") from exc
        index = {}
        statuses = {VRL_STATES.BOARDING.display_name, VRL_STATES.BOARDING_WITH_CHARGING.display_name}
        for op, vid, status, start, end, alighting in records[columns].itertuples(index=False, name=None):
            if status not in statuses or not isinstance(alighting, str):
                continue
            op, vid, start, end = map(_finite_number, (op, vid, start, end))
            if not all(np.isfinite(value) for value in (op, vid, start, end)):
                continue
            if not op.is_integer() or not vid.is_integer() or int(op) != operator_id or end < start:
                continue
            for child_id in alighting.split(';'):
                index.setdefault((int(vid), child_id.strip()), []).append((start, end))
        cache[operator_id] = index
    matches = cache[operator_id].get((vehicle_id, rid), [])
    if len(matches) != 1 or not np.isfinite(start_time) or matches[0][0] != start_time:
        raise ValueError(f"Expected one matching FM alighting event for {rid}, operator {operator_id}, "
                         f"vehicle {vehicle_id}, start {start_time}; found {matches}")
    return matches[0][1]


def create_parent_user_stats(output_dir, evaluation_start_time=None, evaluation_end_time=None,
                             print_comments=False, completion_end_time=None, return_amod_legs=False):
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
    :param completion_end_time: observation horizon for completed legs; defaults to config end_time when available
    :param return_amod_legs: also return the selected completed AMoD leg records
    :return: parent DataFrame, or a pair of parent and AMoD leg DataFrames
    """
    if print_comments:
        print("Creating parent user stats from sub-trips...")

    # Read original user stats
    user_stats_file = os.path.join(output_dir, "1_user-stats.csv")
    df = pd.read_csv(user_stats_file, float_precision="round_trip")
    completion_end_time = _completion_horizon(output_dir, completion_end_time)

    # Select the cohort by parent request time, then retain all of its children.
    parent_df = df[df[G_RQ_IS_PARENT_REQUEST] == True].copy()
    if evaluation_start_time is not None:
        parent_df = parent_df[parent_df[G_RQ_TIME] >= evaluation_start_time]
    if evaluation_end_time is not None:
        parent_df = parent_df[parent_df[G_RQ_TIME] < evaluation_end_time]
    subtrip_df = df[df[G_RQ_IS_PARENT_REQUEST] == False].copy()

    if print_comments:
        print(f"  Found {len(parent_df)} parent requests and {len(subtrip_df)} sub-trips")

    # For each parent request, aggregate sub-trip information
    parent_rows = []
    completed_leg_rows = []
    alighting_cache = {}

    for parent_idx, parent_row in parent_df.iterrows():
        request_id = parent_row[G_RQ_ID]
        modal_state = _execution_mode(parent_row)

        # Get all sub-trips for this request
        sub_trips = subtrip_df[subtrip_df[G_RQ_ID] == request_id]

        # Create aggregated parent row
        parent_agg = parent_row.to_dict()
        parent_agg[G_RQ_SELECTED_MODE] = modal_state if modal_state is not None else np.nan
        selected_legs = _selected_legs(parent_row, sub_trips, modal_state)
        fm_id = {RQ_MODAL_STATE.FIRSTMILE.value: RQ_SUB_TRIP_ID.FM_AMOD.value,
                 RQ_MODAL_STATE.FIRSTLASTMILE.value: RQ_SUB_TRIP_ID.FLM_AMOD_0.value}.get(modal_state)
        fm_row = selected_legs.get(fm_id) if fm_id is not None else None
        fm_alighting_end_time = None
        if fm_row is not None and np.isfinite(_finite_number(fm_row.get(G_RQ_DO))):
            fm_alighting_end_time = _fm_alighting_end_time(output_dir, parent_row, fm_row, alighting_cache)

        # Determine if request is served
        # For MONOMODAL (0), parent itself determines if served
        # For intermodal (FM=1, LM=2, FLM=3), check if all mandatory sub-trips are served
        is_served = False
        total_fare = 0
        total_wait_time = 0
        total_travel_time = 0

        # Parent-level AMoD sums; standard metrics are calculated separately per leg.
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

        elif modal_state == RQ_MODAL_STATE.PT.value:
            pt_row = selected_legs.get(RQ_SUB_TRIP_ID.PT.value)
            is_served = _leg_is_complete(pt_row, completion_end_time)
            if is_served:
                total_fare = pt_row.get(G_RQ_FARE, np.nan)
                total_wait_time = _get_numeric_offer_parameter(pt_row.get(G_RQ_OFFERS), G_OFFER_WAIT)
                total_travel_time = pt_row.get(G_RQ_DO) - pt_row.get(G_RQ_PU)
                leg_info['pt_pu'] = pt_row.get(G_RQ_PU)
                leg_info['pt_do'] = pt_row.get(G_RQ_DO)
                leg_info['pt_wait'] = total_wait_time

        elif modal_state == RQ_MODAL_STATE.FIRSTMILE.value:
            # FM: Need FM_AMOD (1) and FM_PT (2)
            fm_amod_row = selected_legs.get(RQ_SUB_TRIP_ID.FM_AMOD.value)
            fm_pt_row = selected_legs.get(RQ_SUB_TRIP_ID.FM_PT.value)

            if fm_amod_row is not None and fm_pt_row is not None:

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
                    fm_pt_wait = calculate_pt_wait_time(fm_alighting_end_time, fm_pt_row)

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
            lm_pt_row = selected_legs.get(RQ_SUB_TRIP_ID.LM_PT.value)
            lm_amod_row = selected_legs.get(RQ_SUB_TRIP_ID.LM_AMOD.value)

            if lm_pt_row is not None and lm_amod_row is not None:

                lm_pt_served = not pd.isna(lm_pt_row.get(G_RQ_PU))
                lm_amod_served = not pd.isna(lm_amod_row.get(G_RQ_PU))
                is_served = lm_pt_served and lm_amod_served

                if is_served:
                    total_fare = lm_pt_row.get(G_RQ_FARE, 0) + lm_amod_row.get(G_RQ_FARE, 0)

                    # Wait time for PT leg
                    pt_offer_str = lm_pt_row.get(G_RQ_OFFERS, None)
                    lm_pt_wait = _get_numeric_offer_parameter(pt_offer_str, G_OFFER_WAIT)

                    # Wait time for LM AMoD (waiting for AMoD)
                    lm_pt_target_walking_time = _get_numeric_offer_parameter(pt_offer_str, G_PT_OFFER_TARGET_WALKING_TIME)
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
            flm_amod_0_row = selected_legs.get(RQ_SUB_TRIP_ID.FLM_AMOD_0.value)
            flm_pt_row = selected_legs.get(RQ_SUB_TRIP_ID.FLM_PT.value)
            flm_amod_1_row = selected_legs.get(RQ_SUB_TRIP_ID.FLM_AMOD_1.value)

            if flm_amod_0_row is not None and flm_pt_row is not None and flm_amod_1_row is not None:

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
                    flm_pt_wait = calculate_pt_wait_time(fm_alighting_end_time, flm_pt_row)
                    flm_pt_offer_str = flm_pt_row.get(G_RQ_OFFERS, None)
                    flm_pt_target_walking_time = _get_numeric_offer_parameter(flm_pt_offer_str, G_PT_OFFER_TARGET_WALKING_TIME)
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

        amod_ids = {
            RQ_MODAL_STATE.MONOMODAL.value: (None,),
            RQ_MODAL_STATE.FIRSTMILE.value: (RQ_SUB_TRIP_ID.FM_AMOD.value,),
            RQ_MODAL_STATE.LASTMILE.value: (RQ_SUB_TRIP_ID.LM_AMOD.value,),
            RQ_MODAL_STATE.FIRSTLASTMILE.value: (RQ_SUB_TRIP_ID.FLM_AMOD_0.value, RQ_SUB_TRIP_ID.FLM_AMOD_1.value),
        }.get(modal_state, ())
        completed_amod_legs = [selected_legs.get(sub_trip_id) for sub_trip_id in amod_ids
                               if _leg_is_complete(selected_legs.get(sub_trip_id), completion_end_time)
                               and (fm_id is None or sub_trip_id != fm_id or completion_end_time is None
                                    or fm_alighting_end_time <= completion_end_time)]
        parent_agg['amod_required_legs'] = len(amod_ids)
        parent_agg['amod_completed_legs'] = len(completed_amod_legs)
        for leg in completed_amod_legs:
            operator_id = _finite_number(leg.get(G_RQ_OP_ID))
            if np.isfinite(operator_id) and operator_id >= 0 and operator_id.is_integer():
                field = f'amod_op{int(operator_id)}_completed_legs'
                parent_agg[field] = parent_agg.get(field, 0) + 1
            completed_leg_rows.append(leg.to_dict())

        # PT bookings prefill times: completion needs the selected legs, horizon and failure checks.
        is_served = bool(is_served and selected_legs and
                         all(_leg_is_complete(row, completion_end_time) for row in selected_legs.values()) and
                         not _has_journey_failure(parent_row) and
                         not any(_has_journey_failure(row) for row in selected_legs.values() if row is not None) and
                         (fm_id is None or completion_end_time is None
                          or fm_alighting_end_time <= completion_end_time) and
                         _connections_are_feasible(modal_state, selected_legs, fm_alighting_end_time))

        # Update parent aggregation with computed values
        parent_agg['is_served'] = is_served
        parent_agg['total_fare'] = total_fare if is_served else np.nan
        parent_agg['total_wait_time'] = total_wait_time if is_served else np.nan
        parent_agg['total_travel_time'] = total_travel_time if is_served else np.nan

        # Parent-level AMoD sums; standard metrics are calculated separately per leg.
        has_completed_amod_journey = is_served and bool(amod_ids)
        parent_agg['amod_fare'] = amod_fare if has_completed_amod_journey else np.nan
        parent_agg['amod_wait_time'] = amod_wait_time if has_completed_amod_journey else np.nan
        parent_agg['amod_travel_time'] = amod_travel_time if has_completed_amod_journey else np.nan
        parent_agg['amod_direct_distance'] = amod_direct_distance if has_completed_amod_journey else np.nan

        # Add leg info as columns
        for key, val in leg_info.items():
            parent_agg[key] = val

        parent_rows.append(parent_agg)

    # Create parent DataFrame
    parent_result_df = pd.DataFrame(parent_rows) if parent_rows else parent_df.assign(
        is_served=pd.Series(dtype=bool), amod_required_legs=pd.Series(dtype=int),
        amod_completed_legs=pd.Series(dtype=int))

    # Save to file
    parent_output_file = os.path.join(output_dir, "1_user-stats_parent.csv")
    parent_result_df.to_csv(parent_output_file, index=False)

    if print_comments:
        print(f"  Saved parent user stats to: {parent_output_file}")
        print(f"  Served requests: {parent_result_df['is_served'].sum()} / {len(parent_result_df)}")

    if return_amod_legs:
        return parent_result_df, pd.DataFrame(completed_leg_rows, columns=df.columns)
    return parent_result_df


def categorize_modal_state(selected_mode):
    """Categorize the numeric mode selected for execution."""
    selected_mode = _parse_execution_mode(selected_mode)
    if selected_mode is None:
        return 'Unknown'
    if selected_mode == RQ_MODAL_STATE.MONOMODAL.value:
        return 'DRT_only'
    elif selected_mode == RQ_MODAL_STATE.FIRSTMILE.value:
        return 'FM'
    elif selected_mode == RQ_MODAL_STATE.LASTMILE.value:
        return 'LM'
    elif selected_mode == RQ_MODAL_STATE.FIRSTLASTMILE.value:
        return 'FLM'
    elif selected_mode == RQ_MODAL_STATE.PT.value:
        return 'PT_only'
    else:
        return 'Unknown'


def calculate_payg_metrics(parent_user_stats, is_payg_scenario=False, print_comments=False):
    """
    Calculate PAYG-specific metrics from parent user stats.

    PAYG (Plan-As-You-Go) trips may be interrupted if:
    - No PT available after FM alighting (INTERRUPTED_NO_PT)
    - No LM AMoD available after PT alighting (INTERRUPTED_NO_LM_AMOD)

    :param parent_user_stats: DataFrame with parent request data
    :param is_payg_scenario: whether this is a PAYG scenario (determined by broker_type in config)
    :param print_comments: whether to print status messages
    :return: dict with PAYG-specific metrics, empty dict if not a PAYG scenario
    """
    payg_stats = {}

    # Check if this is a PAYG scenario based on broker type from config
    if not is_payg_scenario:
        return {}  # Not a PAYG scenario

    # Filter to intermodal requests only (FM, LM, FLM) - these are the ones that can be interrupted
    execution_modes = parent_user_stats[G_RQ_SELECTED_MODE].map(_parse_execution_mode)
    intermodal_mask = execution_modes.isin([
        RQ_MODAL_STATE.FIRSTMILE.value,
        RQ_MODAL_STATE.LASTMILE.value,
        RQ_MODAL_STATE.FIRSTLASTMILE.value
    ])
    intermodal_df = parent_user_stats[intermodal_mask].copy()
    intermodal_df[G_RQ_SELECTED_MODE] = execution_modes[intermodal_mask]

    total_intermodal = len(intermodal_df)
    if total_intermodal == 0:
        return {}

    # Count interrupted requests
    interrupted_mask = intermodal_df[G_RQ_PAYG_INTERRUPTED].map(_is_true)
    interrupted_df = intermodal_df[interrupted_mask]
    interrupted_count = len(interrupted_df)
    completed_mask = intermodal_df['is_served'].map(_is_true)
    completed_count = int((completed_mask & ~interrupted_mask).sum())
    incomplete_count = total_intermodal - interrupted_count - completed_count

    # Overall rates
    payg_stats['payg_total_intermodal_requests'] = total_intermodal
    payg_stats['payg_completed_count'] = completed_count
    payg_stats['payg_interrupted_count'] = interrupted_count
    payg_stats['payg_incomplete_count'] = incomplete_count
    payg_stats['payg_completion_rate [%]'] = completed_count / total_intermodal * 100
    payg_stats['payg_interrupt_rate [%]'] = interrupted_count / total_intermodal * 100
    payg_stats['payg_incomplete_rate [%]'] = incomplete_count / total_intermodal * 100

    # Breakdown by interrupt state
    no_pt_count = len(interrupted_df[
        interrupted_df[G_RQ_PAYG_INTERRUPT_STATE] == PAYG_TRIP_STATE.INTERRUPTED_NO_PT.value
    ])
    no_amod_count = len(interrupted_df[
        interrupted_df[G_RQ_PAYG_INTERRUPT_STATE] == PAYG_TRIP_STATE.INTERRUPTED_NO_LM_AMOD.value
    ])

    payg_stats['payg_interrupt_no_pt_count'] = no_pt_count
    payg_stats['payg_interrupt_no_pt_rate [%]'] = no_pt_count / total_intermodal * 100 if total_intermodal > 0 else 0
    payg_stats['payg_interrupt_no_amod_count'] = no_amod_count
    payg_stats['payg_interrupt_no_amod_rate [%]'] = no_amod_count / total_intermodal * 100 if total_intermodal > 0 else 0

    # Breakdown by modal state
    for category, modal_value in [('FM', RQ_MODAL_STATE.FIRSTMILE.value),
                                   ('LM', RQ_MODAL_STATE.LASTMILE.value),
                                   ('FLM', RQ_MODAL_STATE.FIRSTLASTMILE.value)]:
        cat_df = intermodal_df[intermodal_df[G_RQ_SELECTED_MODE] == modal_value]
        cat_total = len(cat_df)
        if cat_total > 0:
            cat_interrupted = len(cat_df[cat_df[G_RQ_PAYG_INTERRUPTED] == True])
            payg_stats[f'payg_{category}_total'] = cat_total
            payg_stats[f'payg_{category}_interrupted_count'] = cat_interrupted
            payg_stats[f'payg_{category}_interrupt_rate [%]'] = cat_interrupted / cat_total * 100
        else:
            payg_stats[f'payg_{category}_total'] = 0
            payg_stats[f'payg_{category}_interrupted_count'] = 0
            payg_stats[f'payg_{category}_interrupt_rate [%]'] = np.nan

    if print_comments and interrupted_count > 0:
        print(f"  PAYG Interruptions: {interrupted_count}/{total_intermodal} ({payg_stats['payg_interrupt_rate [%]']:.1f}%)")
        print(f"    - No PT available: {no_pt_count}")
        print(f"    - No LM AMoD available: {no_amod_count}")

    return payg_stats


def _fleet_ratio(numerator, denominator):
    """Return NaN for an undefined ratio."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def _sum_known(series):
    return series.sum(min_count=max(len(series), 1))


def evaluate_operator_fleet(output_dir, op_id, scenario_parameters, veh_type_stats,
                            veh_type_db, evaluation_start_time, evaluation_end_time,
                            number_users, number_pax, direct_distance_km,
                            total_travel_time):
    """Calculate standard fleet metrics for one AMoD operator.

    Counts and direct distances refer to completed AMoD legs; time is their
    aggregate in-vehicle time in seconds. Distances in vehicle records are in
    metres, and costs in vehicle type data are in cents, as in standard.py.
    Vehicle records are filtered by leg start, matching standard evaluation.
    """
    try:
        op_df = read_op_output_file(output_dir, op_id, evaluation_start_time, evaluation_end_time)
        has_vehicle_output = True
    except FileNotFoundError:
        has_vehicle_output = False
        op_df = pd.DataFrame(columns=[G_V_VID, G_V_TYPE, G_VR_STATUS,
                                    G_VR_LEG_START_TIME, G_VR_LEG_END_TIME,
                                    G_VR_LEG_DISTANCE, G_VR_NR_PAX, G_VR_OB_RID,
                                    G_VR_BOARDING_RID, G_VR_ALIGHTING_RID, G_VR_TOLL])

    if veh_type_stats is not None:
        op_types = veh_type_stats.loc[veh_type_stats[G_V_OP_ID] == op_id, [G_V_VID, G_V_TYPE]].set_index(G_V_VID)
    elif G_V_TYPE in op_df.columns:
        op_types = op_df[[G_V_VID, G_V_TYPE]].drop_duplicates().set_index(G_V_VID)
    else:
        op_types = pd.DataFrame(columns=[G_V_TYPE])
    n_vehicles = len(op_types)

    distance = pd.to_numeric(op_df[G_VR_LEG_DISTANCE], errors='coerce')
    total_km = distance.sum() / 1000 if has_vehicle_output else np.nan
    status = op_df[G_VR_STATUS]
    start = pd.to_numeric(op_df[G_VR_LEG_START_TIME], errors='coerce')
    end = pd.to_numeric(op_df[G_VR_LEG_END_TIME], errors='coerce')
    sim_end = scenario_parameters[G_SIM_END_TIME]
    sim_duration = sim_end - scenario_parameters[G_SIM_START_TIME]
    duration = end.clip(upper=sim_end) - start.clip(upper=sim_end)
    inactive = status.isin([VRL_STATES.OUT_OF_SERVICE.display_name, VRL_STATES.CHARGING.display_name])
    utilization = _fleet_ratio(duration[~inactive].sum(), n_vehicles * sim_duration - duration[inactive].sum()) * 100
    revenue_mask = status.isin([state.display_name for state in G_REVENUE_STATUS])
    revenue_hours = duration[revenue_mask].sum() / 3600 if has_vehicle_output else np.nan

    on_board = op_df[G_VR_OB_RID]
    occupancy = pd.to_numeric(op_df.get(G_VR_NR_PAX, pd.Series(np.nan, index=op_df.index)), errors='coerce')
    request_count = on_board.fillna('').astype(str).map(lambda x: len(x.split(';')) if x else 0)
    total_distance = distance.sum()
    empty_distance = distance[on_board.isna()].sum()
    reposition_distance = distance[on_board.isna() & (status == VRL_STATES.REPOSITION.display_name)].sum()
    occupancy_pax = _fleet_ratio((occupancy * distance).sum(min_count=1), total_distance)
    occupancy_rq = _fleet_ratio((request_count * distance).sum(min_count=1), total_distance)
    empty_pct = _fleet_ratio(empty_distance, total_distance) * 100
    reposition_pct = _fleet_ratio(reposition_distance, total_distance) * 100
    driving = status.isin([state.display_name for state in G_DRIVING_STATUS])
    driving_time = (end[driving] - start[driving]).sum()

    try:
        customer_distance = avg_in_vehicle_distance(op_df)
        shared = shared_rides(op_df)
    except ZeroDivisionError:
        customer_distance = shared = np.nan

    vehicle_rows = {}
    energy_emissions = scenario_parameters.get(G_ENERGY_EMISSIONS)
    if energy_emissions is None:
        energy_emissions = ENERGY_EMISSIONS
    fixed_share = scenario_parameters.get(G_OP_SHARE_FC, 1.0)
    if fixed_share is None:
        fixed_share = 1.0
    for vid, row in op_types.iterrows():
        type_name = row[G_V_TYPE]
        vtype = veh_type_db.get(type_name, {})
        veh_km = distance[op_df[G_V_VID] == vid].sum() / 1000 if has_vehicle_output else np.nan
        kwh = veh_km * _fleet_ratio(vtype.get(G_VTYPE_BATTERY_SIZE, np.nan), vtype.get(G_VTYPE_RANGE, np.nan))
        vehicle_rows[vid] = {
            'type': vtype.get(G_VTYPE_NAME, type_name),
            'total km': veh_km,
            'total kWh': kwh,
            'total CO2 [g]': energy_emissions * kwh,
            'fix costs': np.rint(fixed_share * vtype.get(G_VTYPE_FIX_COST, np.nan)),
            'total variable costs': np.rint(vtype.get(G_VTYPE_DIST_COST, np.nan) * veh_km),
        }
    vehicle_columns = ['type', 'total km', 'total kWh', 'total CO2 [g]', 'fix costs', 'total variable costs']
    vehicle_df = pd.DataFrame.from_dict(vehicle_rows, orient='index', columns=vehicle_columns)
    vehicle_df.to_csv(os.path.join(output_dir, f'standard_mod-{op_id}_veh_eval.csv'))
    co2 = _sum_known(vehicle_df['total CO2 [g]'])
    fixed_cost = _sum_known(vehicle_df['fix costs'])
    variable_cost = _sum_known(vehicle_df['total variable costs'])
    if not has_vehicle_output:
        utilization = np.nan
    toll = op_df[G_VR_TOLL].sum(min_count=1) if G_VR_TOLL in op_df else np.nan
    return {
        '% fleet utilization': utilization,
        'rides per veh rev hours': _fleet_ratio(number_pax, revenue_hours),
        'rides per veh rev hours rq': _fleet_ratio(number_users, revenue_hours),
        'total vkm': total_km,
        'occupancy': occupancy_pax,
        'occupancy rq': occupancy_rq,
        '% empty vkm': empty_pct,
        '% repositioning vkm': reposition_pct,
        'customer direct distance [km]': direct_distance_km,
        'saved distance [%]': _fleet_ratio(direct_distance_km - total_km, direct_distance_km) * 100,
        'trip distance per fleet distance': _fleet_ratio(direct_distance_km, total_km),
        'trip distance per fleet distance (no reloc)': _fleet_ratio(direct_distance_km, total_km - reposition_distance / 1000),
        'avg driving velocity [km/h]': _fleet_ratio(total_km * 3600, driving_time),
        # Keep the legacy standard formula for this field; also expose the corrected value.
        'avg trip velocity [km/h]': _fleet_ratio(direct_distance_km * 3.6, total_travel_time),
        'avg trip velocity corrected [km/h]': _fleet_ratio(direct_distance_km * 3600, total_travel_time),
        'vehicle revenue hours [Fzg h]': revenue_hours,
        'total toll': toll,
        'parking cost': np.nan,
        'toll': toll,
        'mod fix costs': fixed_cost,
        'mod var costs': variable_cost,
        'total CO2 emissions [t]': co2 / 10**6,
        'total external emission costs': np.rint(EMISSION_CPG * co2),
        'customer in vehicle distance': customer_distance,
        'shared rides [%]': shared,
    }


def _operator_user_metrics(legs, parents, operator_attributes, op_id):
    """Calculate standard user metrics per completed, selected AMoD leg.

    Counts and averages use legs. Modal split uses unique parents in the cohort,
    allowing a parent to be counted once for each participating operator.
    """
    count = len(legs)
    pax = pd.to_numeric(legs[G_RQ_PAX], errors='coerce').sum()
    total_pax = pd.to_numeric(parents[G_RQ_PAX], errors='coerce').sum()
    def numeric(name):
        return pd.to_numeric(legs[name], errors='coerce') if name in legs else pd.Series(np.nan, index=legs.index)
    pickup, dropoff = numeric(G_RQ_PU), numeric(G_RQ_DO)
    request_time, ept = numeric(G_RQ_TIME), numeric(G_RQ_EPT)
    direct_time, direct_distance = numeric(G_RQ_DRT), numeric(G_RQ_DRD)
    wait, travel = pickup - request_time, dropoff - pickup
    boarding = operator_attributes.get('op_const_boarding_time', 0)
    boarding = _finite_number(boarding)
    detour = travel - direct_time - boarding
    rel_detour = detour / direct_time.where(direct_time > 0) * 100
    horizon = operator_attributes.get(G_RA_OPT_HOR, 0)
    reservation = ept - request_time > horizon
    parent_reservation = (pd.to_numeric(parents[G_RQ_EPT], errors='coerce') -
                          pd.to_numeric(parents[G_RQ_TIME], errors='coerce') > horizon)
    reservation_pax = pd.to_numeric(legs.loc[reservation, G_RQ_PAX], errors='coerce').sum()
    parent_reservation_pax = pd.to_numeric(parents.loc[parent_reservation, G_RQ_PAX], errors='coerce').sum()
    def ratio(numerator, denominator, factor=1):
        return numerator / denominator * factor if denominator else np.nan
    def operator_offered(value):
        if not isinstance(value, str):
            return False
        for entry in value.split('|'):
            try:
                mode, key = ast.literal_eval(entry.split(':', 1)[0])
            except (ValueError, SyntaxError, TypeError):
                continue
            if mode in (RQ_MODAL_STATE.MONOMODAL.value, RQ_MODAL_STATE.FIRSTMILE.value,
                        RQ_MODAL_STATE.FIRSTLASTMILE.value) and key == op_id:
                return True
            if isinstance(key, tuple) and any(isinstance(pair, tuple) and len(pair) == 2
                                              and pair[0] == op_id for pair in key):
                return True
        return False

    offers = sum(operator_offered(value) for value in parents[G_RQ_OFFERS]) if G_RQ_OFFERS in parents else 0
    served_parents = parents[parents[G_RQ_ID].isin(legs[G_RQ_ID])]
    served_parent_pax = pd.to_numeric(served_parents[G_RQ_PAX], errors='coerce').sum()
    served_parent_reservation = parent_reservation.loc[served_parents.index]
    served_reservation_pax = pd.to_numeric(served_parents.loc[served_parent_reservation, G_RQ_PAX], errors='coerce').sum()
    result = {
        'operator_id': op_id,
        'number users': count,
        'number travelers': pax,
        'modal split': ratio(served_parent_pax, total_pax),
        'modal split rq': ratio(len(served_parents), len(parents)),
        'reservation users': int(reservation.sum()),
        'reservation pax': reservation_pax,
        'served reservation users [%]': ratio(int(served_parent_reservation.sum()), int(parent_reservation.sum()), 100),
        'served reservation pax [%]': ratio(served_reservation_pax, parent_reservation_pax, 100),
        'online users': count - int(reservation.sum()),
        'online pax': pax - reservation_pax,
        'served online users [%]': ratio(len(served_parents) - int(served_parent_reservation.sum()), len(parents) - int(parent_reservation.sum()), 100),
        'served online pax [%]': ratio(served_parent_pax - served_reservation_pax, total_pax - parent_reservation_pax, 100),
        # Rejected offers are not retained by the broker in the parent record.
        '% created offers': np.nan,
        'candidate offers [%]': ratio(offers, len(parents), 100),
        'utility': numeric(G_RQ_C_UTIL).mean(),
        'travel time': travel.mean(),
        'travel distance': np.nan,
        'waiting time': wait.mean(),
        'waiting time from ept': (pickup - ept).mean(),
        'waiting time (median)': wait.median(),
        'waiting time (90% quantile)': wait.quantile(0.9),
        'detour time': detour.mean(),
        'rel detour': rel_detour.mean(),
        'customer direct distance [km]': direct_distance.sum(min_count=1) / 1000,
        'mod revenue': numeric(G_RQ_FARE).sum(min_count=1),
    }
    return result, travel.sum(min_count=1)


def intermodal_evaluation(output_dir, evaluation_start_time=None, evaluation_end_time=None, print_comments=False, dir_names_in=None):
    """Evaluate the parent cohort and selected, completed AMoD legs.

    MoD columns use standard evaluation names and per-leg units; the Intermodal
    column describes parent journeys. Vehicle records use the standard interval
    filter. Completion is observed through the simulation end, independently of
    the request cohort interval. PT completion uses recorded booking times and
    the available failure and connection state.
    """
    from src.evaluation.standard import create_vehicle_type_db

    if not os.path.isdir(output_dir):
        raise IOError(f"Could not find result directory {output_dir}!")
    scenario_parameters, list_operator_attributes, _ = load_scenario_inputs(output_dir)
    dir_names = dir_names_in or get_directory_dict(scenario_parameters, list_operator_attributes, abs_fleetpy_dir=MAIN_DIR)
    if evaluation_start_time is None and scenario_parameters.get(G_EVAL_INT_START) is not None:
        evaluation_start_time = int(scenario_parameters[G_EVAL_INT_START])
    if evaluation_end_time is None and scenario_parameters.get(G_EVAL_INT_END) is not None:
        evaluation_end_time = int(scenario_parameters[G_EVAL_INT_END])

    parents, legs = create_parent_user_stats(
        output_dir, evaluation_start_time, evaluation_end_time, print_comments,
        completion_end_time=scenario_parameters.get(G_SIM_END_TIME), return_amod_legs=True)
    legs.to_csv(os.path.join(output_dir, '1_user-stats_amod_legs.csv'), index=False)
    served = parents[parents['is_served'].map(_is_true)].copy()
    categories = parents[G_RQ_SELECTED_MODE].apply(categorize_modal_state)
    served_categories = served[G_RQ_SELECTED_MODE].apply(categorize_modal_state)
    total_pax = pd.to_numeric(parents[G_RQ_PAX], errors='coerce').sum()
    served_pax = pd.to_numeric(served[G_RQ_PAX], errors='coerce').sum()
    required = int(parents['amod_required_legs'].sum())
    completed = int(parents['amod_completed_legs'].sum())
    def pct(numerator, denominator):
        return 100 * numerator / denominator if denominator else np.nan
    journey = {
        'journey requests': len(parents),
        'journey completed': len(served),
        'journey completed travelers': served_pax,
        'Service_Rate [%]': pct(len(served), len(parents)),
        'Service_Rate_Pax [%]': pct(served_pax, total_pax),
        'DRT_required_legs': required,
        'DRT_completed_legs': completed,
        'DRT_leg_service_rate [%]': pct(completed, required),
        'journey waiting time [s]': served['total_wait_time'].mean() if len(served) else np.nan,
        'journey first pickup to last dropoff [s]': served['total_travel_time'].mean() if len(served) else np.nan,
        'journey fare': served['total_fare'].sum(min_count=1) if len(served) else np.nan,
    }
    for category in ('DRT_only', 'FM', 'LM', 'FLM', 'PT_only'):
        count = int((served_categories == category).sum())
        journey[f'{category}_count'] = count
        journey[f'{category}_service_rate [%]'] = pct(count, int((categories == category).sum()))
    uncatchable = parents[G_RQ_UNCATCHABLE_PT].map(_is_true)
    for category in ('FM', 'FLM'):
        mask = categories == category
        journey[f'{category}_uncatchable_count'] = int((uncatchable & mask).sum())
        journey[f'{category}_uncatchable_rate [%]'] = pct(int((uncatchable & mask).sum()), int(mask.sum()))
    fm_mask = categories.isin(['FM', 'FLM'])
    journey['total_uncatchable_count'] = int((uncatchable & fm_mask).sum())
    journey['total_uncatchable_rate [%]'] = pct(int((uncatchable & fm_mask).sum()), int(fm_mask.sum()))
    rebooking = parents.get(G_RQ_EI_REBOOKING_STATUS, pd.Series(index=parents.index, dtype=object))
    booked = rebooking == EI_REBOOKING_STATE.BOOKED.value
    failed = rebooking.isin((EI_REBOOKING_STATE.FAILED_NO_PT.value, EI_REBOOKING_STATE.FAILED_NO_LM.value,
                             EI_REBOOKING_STATE.FAILED_PT_CANCEL.value, EI_REBOOKING_STATE.FAILED_HORIZON.value))
    journey['EI_rebooking_booked_count'] = int(booked.sum())
    journey['EI_rebooking_failed_count'] = int(failed.sum())
    journey['EI_rebooking_completed_count'] = int((booked & parents['is_served'].map(_is_true)).sum())
    for prefix, fields in [('PT_Wait_Time', ('fm_pt_wait', 'flm_pt_wait')),
                           ('LM_Wait_Time', ('lm_amod_wait', 'flm_amod_1_wait'))]:
        values = []
        names = ('FM', 'FLM') if prefix.startswith('PT') else ('LM', 'FLM')
        for name, field in zip(names, fields):
            series = pd.to_numeric(served[field], errors='coerce').dropna() if field in served else pd.Series(dtype=float)
            journey[f'{prefix}_{name} [s]'] = series.mean()
            values.extend(series.tolist())
        journey[f'{prefix}_Combined [s]'] = np.mean(values) if values else np.nan
    journey.update(calculate_payg_metrics(parents, scenario_parameters.get(G_BROKER_TYPE) == 'PTBrokerPAYG', print_comments))

    veh_type_db = create_vehicle_type_db(dir_names[G_DIR_VEH])
    try:
        veh_type_stats = pd.read_csv(os.path.join(output_dir, '2_vehicle_types.csv'))
    except FileNotFoundError:
        veh_type_stats = None
    results = {}
    n_operators = scenario_parameters.get(G_NR_OPERATORS, 0)
    if not legs.empty:
        operator_ids = pd.to_numeric(legs[G_RQ_OP_ID], errors='coerce')
        if (operator_ids.isna() | (operator_ids < 0) | (operator_ids >= n_operators) |
                (operator_ids % 1 != 0)).any():
            raise ValueError('Completed AMoD leg has an invalid or unconfigured operator ID')
    for op_id in range(n_operators):
        op_legs = legs[pd.to_numeric(legs[G_RQ_OP_ID], errors='coerce') == op_id]
        metrics, total_travel = _operator_user_metrics(op_legs, parents, list_operator_attributes[op_id], op_id)
        metrics.update(evaluate_operator_fleet(output_dir, op_id, scenario_parameters, veh_type_stats,
                                               veh_type_db, evaluation_start_time, evaluation_end_time,
                                               metrics['number users'], metrics['number travelers'],
                                               metrics['customer direct distance [km]'], total_travel))
        results[f'MoD_{op_id}'] = metrics
    results['Intermodal'] = journey
    result_df = pd.DataFrame(results)
    standard_rows = [
        'operator_id', 'number users', 'number travelers', 'modal split', 'modal split rq',
        'reservation users', 'reservation pax', 'served reservation users [%]',
        'served reservation pax [%]', 'online users', 'online pax', 'served online users [%]',
        'served online pax [%]', '% created offers', 'utility', 'travel time', 'travel distance',
        'waiting time', 'waiting time from ept', 'waiting time (median)',
        'waiting time (90% quantile)', 'detour time', 'rel detour', '% fleet utilization',
        'rides per veh rev hours', 'rides per veh rev hours rq', 'total vkm', 'occupancy',
        'occupancy rq', '% empty vkm', '% repositioning vkm', 'customer direct distance [km]',
        'saved distance [%]', 'trip distance per fleet distance',
        'trip distance per fleet distance (no reloc)', 'avg driving velocity [km/h]',
        'avg trip velocity [km/h]', 'vehicle revenue hours [Fzg h]', 'total toll', 'mod revenue',
        'mod fix costs', 'mod var costs', 'total CO2 emissions [t]',
        'total external emission costs', 'parking cost', 'toll', 'customer in vehicle distance',
        'shared rides [%]',
    ]
    ordered_rows = [row for row in standard_rows if row in result_df.index]
    ordered_rows.extend(row for row in result_df.index if row not in standard_rows)
    result_df = result_df.reindex(ordered_rows)
    result_df.to_csv(os.path.join(output_dir, 'standard_eval.csv'))
    if print_comments:
        print(f"Evaluation complete: {len(served)}/{len(parents)} journeys and {completed}/{required} DRT legs completed.")
    return result_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sc = sys.argv[1]
        intermodal_evaluation(sc, print_comments=True)
    else:
        print("Usage: python intermodal.py <output_directory>")
        print("Example: python src/evaluation/intermodal.py studies/example_study/results/example_im_ptbroker")

"""Fleet statistics for intermodal evaluation using standard evaluation units."""

import os

import numpy as np
import pandas as pd

from src.evaluation.standard import (
    EMISSION_CPG, ENERGY_EMISSIONS, avg_in_vehicle_distance,
    read_op_output_file, shared_rides,
)
from src.misc.globals import (
    G_DRIVING_STATUS, G_ENERGY_EMISSIONS, G_OP_SHARE_FC, G_REVENUE_STATUS,
    G_SIM_END_TIME, G_SIM_START_TIME, G_V_OP_ID, G_V_TYPE, G_V_VID,
    G_VR_ALIGHTING_RID, G_VR_BOARDING_RID, G_VR_LEG_DISTANCE,
    G_VR_LEG_END_TIME, G_VR_LEG_START_TIME, G_VR_NR_PAX, G_VR_OB_RID,
    G_VR_STATUS, G_VR_TOLL, G_VTYPE_BATTERY_SIZE, G_VTYPE_DIST_COST,
    G_VTYPE_FIX_COST, G_VTYPE_NAME, G_VTYPE_RANGE, VRL_STATES,
)


def _ratio(numerator, denominator):
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
    utilization = _ratio(duration[~inactive].sum(), n_vehicles * sim_duration - duration[inactive].sum()) * 100
    revenue_mask = status.isin([state.display_name for state in G_REVENUE_STATUS])
    revenue_hours = duration[revenue_mask].sum() / 3600 if has_vehicle_output else np.nan

    on_board = op_df[G_VR_OB_RID]
    occupancy = pd.to_numeric(op_df.get(G_VR_NR_PAX, pd.Series(np.nan, index=op_df.index)), errors='coerce')
    request_count = on_board.fillna('').astype(str).map(lambda x: len(x.split(';')) if x else 0)
    total_distance = distance.sum()
    empty_distance = distance[on_board.isna()].sum()
    reposition_distance = distance[on_board.isna() & (status == VRL_STATES.REPOSITION.display_name)].sum()
    occupancy_pax = _ratio((occupancy * distance).sum(min_count=1), total_distance)
    occupancy_rq = _ratio((request_count * distance).sum(min_count=1), total_distance)
    empty_pct = _ratio(empty_distance, total_distance) * 100
    reposition_pct = _ratio(reposition_distance, total_distance) * 100
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
        kwh = veh_km * _ratio(vtype.get(G_VTYPE_BATTERY_SIZE, np.nan), vtype.get(G_VTYPE_RANGE, np.nan))
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
        'rides per veh rev hours': _ratio(number_pax, revenue_hours),
        'rides per veh rev hours rq': _ratio(number_users, revenue_hours),
        'total vkm': total_km,
        'occupancy': occupancy_pax,
        'occupancy rq': occupancy_rq,
        '% empty vkm': empty_pct,
        '% repositioning vkm': reposition_pct,
        'customer direct distance [km]': direct_distance_km,
        'saved distance [%]': _ratio(direct_distance_km - total_km, direct_distance_km) * 100,
        'trip distance per fleet distance': _ratio(direct_distance_km, total_km),
        'trip distance per fleet distance (no reloc)': _ratio(direct_distance_km, total_km - reposition_distance / 1000),
        'avg driving velocity [km/h]': _ratio(total_km * 3600, driving_time),
        'avg trip velocity [km/h]': _ratio(direct_distance_km * 3600, total_travel_time),
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

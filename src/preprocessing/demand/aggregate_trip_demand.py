import os
import sys
import glob
import numpy as np
import pandas as pd
BASEPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASEPATH)
from src.routing.NetworkBasicWithStoreCpp import NetworkBasicWithStoreCpp
from src.misc.safe_pathname import slugify
from src.misc.globals import *


def return_od_zones(row, node_zone_dict, temporal_agg):
    return node_zone_dict.get(row[G_RQ_ORIGIN], None), node_zone_dict.get(row[G_RQ_DESTINATION], None), \
           np.ceil(row[G_RQ_TIME]/temporal_agg)*temporal_agg,\
           np.ceil(row["estimated_arrival_time"]/temporal_agg)*temporal_agg


def aggregate_demand_for_perfect_forecast(demand_f, node_zone_f, temporal_agg=15*60, use_nw=None):
    """This method creates a forecast of demand for a given zone system and temporal aggregation by counting the actual
    trips in the demand file, thereby creating a perfect forecast given the chosen resolution. Since trip data only
    has time of request, the arrival time is estimated (based on free-flow speed). This underestimates the travel time,
    but the waiting time of actual trips is not included either; hence, the estimate should be acceptable, especially
    considering typically aggregation on lower temporal resolution.
    The forecast is computed in time steps of the temporal aggregation. If fleet controls require other resolutions,
    the algorithm has to be run again or the forecasts should be interpolated.
    The result columns will be called:
    - out perfect_trips
    - in perfect_trips
    - out perfect_pax
    - in perfect_pax

    :param demand_f: trip file
    :param node_zone_f: node-zone relation file
    :param temporal_agg: temporal aggregation in seconds
    :oaram use_nw: use already initiated network
    :return: None
    """
    demand_f_name = os.path.basename(demand_f)
    network_name = os.path.basename(os.path.dirname(demand_f))
    network_name_dir = os.path.join(BASEPATH, "data", "networks", network_name)
    if use_nw is not None:
        nw = use_nw
    else:
        nw = NetworkBasicWithStoreCpp(network_name_dir)
    zone_system_name = os.path.basename(os.path.dirname(os.path.dirname(node_zone_f)))
    print(f"Aggregating trips from {demand_f_name} for zone-system {zone_system_name}")
    # prepare aggregation
    demand_df = pd.read_csv(demand_f)
    node_zone__df = pd.read_csv(node_zone_f, index_col=0)
    node_zone_dict = node_zone__df["zone_id"].to_dict()
    demand_df["nr_trips"] = 1
    print("\t ... computing arrival times")
    arrival_times = []
    counter = 0
    counter_end = len(demand_df)
    for _, row in demand_df.iterrows():
        counter += 1
        if counter % 10000 == 0:
            print(f"\t\t ... {counter}/{counter_end}")
        o_pos = nw.return_node_position(int(row[G_RQ_ORIGIN]))
        d_pos = nw.return_node_position(int(row[G_RQ_DESTINATION]))
        _, tt, _ = nw.return_travel_costs_1to1(o_pos, d_pos)
        ept = row.get(G_RQ_EPT, row[G_RQ_TIME])
        estimated_arrival_time = ept + tt
        arrival_times.append(estimated_arrival_time)
    demand_df["estimated_arrival_time"] = arrival_times
    if G_RQ_PAX not in demand_df.columns:
        demand_df[G_RQ_PAX] = 1
    demand_df[["o_zone", "d_zone", "o_time", "d_time"]] = demand_df.apply(return_od_zones,
                                                                          args=(node_zone_dict, temporal_agg),
                                                                          axis=1, result_type="expand")
    demand_df = demand_df.notna()
    for col in ["o_zone", "d_zone", "o_time", "d_time"]:
        demand_df[col] = demand_df[col].astype(np.int64)
    # aggregate origin
    o_agg_df = demand_df.groupby(["o_time", "o_zone"]).aggregate({"nr_trips": "sum", G_RQ_PAX: "sum"})
    o_agg_df.index.names = ["time", "zone_id"]
    o_agg_df.rename({"nr_trips": "out perfect_trips", G_RQ_PAX: "out perfect_pax"}, axis=1, inplace=True)
    # aggregate destination
    d_agg_df = demand_df.groupby(["d_time", "d_zone"]).aggregate({"nr_trips": "sum", G_RQ_PAX: "sum"})
    d_agg_df.index.names = ["time", "zone_id"]
    d_agg_df.rename({"nr_trips": "in perfect_trips", G_RQ_PAX: "in perfect_pax"}, axis=1, inplace=True)
    # merge
    forecast_df = o_agg_df.merge(d_agg_df, left_index=True, right_index=True)
    forecast_df.fillna(0, inplace=True)
    # output
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(demand_f))), "aggregated",
                              zone_system_name, str(temporal_agg))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_f = os.path.join(output_dir, f"agg_{demand_f_name}")
    print(output_f)
    forecast_df.to_csv(output_f)
    return forecast_df


def create_demand_dir_perfect_and_file_mean_forecasts(demand_network_dir, node_zone_f, temp_agg=15*60, glob_str=None, use_nw=None):
    """This function calls aggregate_demand_for_perfect_forecast() to create perfect forecasts (i.e. perfect accuracy
    for given precision) for a set of request files and creates an additional forecast file for the mean, which can be
    used as an imperfect forecast.

    :param demand_network_dir: matched demand containing request files
    :param node_zone_f: node-zone relation file
    :param temp_agg: temporal aggregation in seconds
    :param glob_str: string for globbing if subset of all request files should be used
    :param use_nw: routing engine instance that is used for creating the files
    :return: None
    """
    if glob_str is not None:
        list_f = glob.glob(f"{demand_network_dir}/{glob_str}")
    else:
        list_f = glob.glob(f"{demand_network_dir}/*.csv")
    if not list_f:
        raise IOError(f"No requests file in {demand_network_dir} with glob-string {glob_str}")
    print(f"Found {len(list_f)} demand files: {list_f}")
    zone_system_name = os.path.basename(os.path.dirname(os.path.dirname(node_zone_f)))
    network_name = os.path.basename(demand_network_dir)
    network_name_dir = os.path.join(BASEPATH, "data", "networks", network_name)
    if not use_nw:
        keep_nw = NetworkBasicWithStoreCpp(network_name_dir)
    else:
        keep_nw = use_nw
    #
    fc_type_list_series = {}
    fc_types = ["in perfect_trips", "in perfect_pax", "out perfect_trips", "out perfect_pax"]
    for fc_type in fc_types:
        fc_type_list_series[fc_type] = []
    counter = 0
    for f in list_f:
        rename_dict = {}
        for fc_type in fc_types:
            rename_dict[fc_type] = f"{fc_type} {counter}"
        tmp_df = aggregate_demand_for_perfect_forecast(f, node_zone_f, temp_agg, use_nw=keep_nw)
        tmp_df.rename(rename_dict, axis=1, inplace=True)
        for fc_type in fc_types:
            fc_type_list_series[fc_type].append(tmp_df[f"{fc_type} {counter}"])
        counter += 1
    #
    print(f"Creating 'mean_forecast' from {list_f}")
    combine_series = []
    for fc_type in fc_types:
        tmp_df = pd.concat(fc_type_list_series[fc_type], axis=1)
        tmp_df.fillna(0, inplace=True)
        tmp_series = tmp_df.mean(axis=1)
        tmp_series.name = fc_type.replace("perfect", "mean")
        combine_series.append(tmp_series)
    #
    mean_df = pd.concat(combine_series, axis=1)
    if glob is not None:
        f_name = f"mean_{slugify(glob_str)}".rstrip(".csv")
        if not f_name.endswith(".csv"):
            f_name += ".csv"
    else:
        f_name = "mean_forecast.csv"
    # output
    output_dir = os.path.join(os.path.dirname(os.path.dirname(demand_network_dir)), "aggregated", zone_system_name,
                              str(temp_agg))
    out_f = os.path.join(output_dir, f_name)
    mean_df.to_csv(out_f)


if __name__ == "__main__":
    if len(sys.argv) in [2, 3]:
        aggregate_demand_for_perfect_forecast(*sys.argv[1:])
    else:
        demand_path = os.path.join(BASEPATH, "data", "demand")
        demand_name = "MUC_miv_oev_concat"
        nw_name = "MUNbene_withBPs_300_1_LHMArea_OVstations_reduced_Flo"
        rq_files_dir = os.path.join(demand_path, demand_name, "matched", nw_name)

        zones_path = os.path.join(BASEPATH, "data", "zones")
        zone_name = "MUC_A99_max2km_4lvl_mivoev"
        node_zone_info_f = os.path.join(zones_path, zone_name, nw_name, "node_zone_info.csv")

        temporal_aggregation = 15*60

        for demand_level in [1, 5, 10, 15]:
            use_glob_str = f"sampled_{demand_level}_pcDemand_*csv"

            create_demand_dir_perfect_and_file_mean_forecasts(rq_files_dir, node_zone_info_f, temporal_aggregation,
                                                              use_glob_str)

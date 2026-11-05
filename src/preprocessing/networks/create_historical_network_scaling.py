"""
File: create_historical_network_scaling.py
Author: Arslan Ali Syed
Email:  arslan.syed@tum.de
Date: 04-03-2026
Description: This script scales the network edges to match the historical travel times from trips.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from pathlib import Path
import contextily as ctx
import igraph as ig
from tqdm import tqdm
import gurobipy as grbpy

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_zones_edges(cell_df, edges_df):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    ax = axes[0]
    cell_df.boundary.plot(alpha=0.5, figsize=(15, 15), ax=ax)
    ctx.add_basemap(ax, crs=cell_df.crs, source=ctx.providers.CartoDB.Positron, attribution_size=0)
    ax.set_title("Zones to be used for scaling")
    for idx, row in cell_df.iterrows():
        ax.text(s=str(row["zone_id"]), x=row['geometry'].centroid.x, y=row['geometry'].centroid.y, horizontalalignment='center')

    ax = axes[1]
    cell_df.boundary.plot(alpha=0.5, figsize=(15, 15), ax=ax)
    ctx.add_basemap(ax, crs=cell_df.crs, source=ctx.providers.CartoDB.Positron, attribution_size=0)
    edges_df.plot(ax=ax, edgecolor='green', linewidth=0.5)
    edges_df[edges_df.is_multiple_zones == True].plot(ax=ax, edgecolor='red', linewidth=0.5)
    ax.set_title("Edges intersecting with multiple zones are marked red")

    plt.show()

def calculate_subzone(latitudes, longitudes, geo_df, attrb_name="zone_id"):
    """ Calculates the subzones for the given coordinates

    :param latitudes: list of latitudes or y (for other projections)
    :param longitudes: list of longitudes or x (for other projections)
    :return: numpy array of subzone id numbers
    """
    from shapely.vectorized import contains as shapely_contains

    # assert self.subzone_df is not None, "subzone file not provided"
    subzone_array = np.empty(len(longitudes), dtype="object")
    subzone_array[:] = np.nan

    for _, row in geo_df.iterrows():
        mask = shapely_contains(row.geometry, longitudes, latitudes)
        subzone_array[np.argwhere(mask)] = row[attrb_name]
    return subzone_array

def create_igraph(nodes_df, edges_df):
    """ Create igraph graph for quick calculation of shortest paths """
    graph_ig = ig.Graph(directed=True)
    graph_ig.add_vertices(nodes_df.index.values)
    graph_ig.add_edges(edges_df[["from_node", "to_node"]].values.tolist())
    graph_ig.es["length"] = edges_df.length.values.tolist()
    return graph_ig

def calculate_zone_intersect_edges(edges_df, zones_df, zone_col_name="zone_id"):
    """ Calculates the list of zones intersecting with each edge especially. It is used to identify the edges that
    are intersecting with multiple zones """

    edges_df_with_zone_list = edges_df.copy()
    zone_list_dict = {inx: [] for inx in edges_df_with_zone_list.index.values}
    for _, row in zones_df.iterrows():
        zone_geom = row["geometry"]
        zone_id = row[zone_col_name]
        for inx in edges_df[edges_df.intersects(zone_geom)].index.values:
            zone_list_dict[inx].append(zone_id)
    edges_df_with_zone_list["zones"] = [zone_list_dict[inx] for inx in edges_df_with_zone_list.index.values]
    edges_df_with_zone_list["is_multiple_zones"] = edges_df_with_zone_list.zones.apply(lambda x: True if len(x) > 1 else False)
    return edges_df_with_zone_list


def prepare_trip_data_for_time_groups(day_df, nodes_df, edges_df, freq_min):
    freq = freq_min * 60
    time_group_trips = day_df.copy()
    time_group_trips["time_group"] = (time_group_trips.rq_time / freq).astype(int)
    time_group_trips["time"] = time_group_trips["time_group"] * freq_min
    time_group_trips = time_group_trips[time_group_trips["start"] != time_group_trips["end"]]
    if "trip_distance" in time_group_trips.columns:
        origin_dest_trips = time_group_trips.groupby(["time_group", "start", "end"]).mean()[["trip_duration", "trip_distance"]]
        origin_dest_trips["trip_distance_min"] = time_group_trips.groupby(["time_group", "start", "end"]).min()["trip_distance"]
        origin_dest_trips["trip_distance_max"] = time_group_trips.groupby(["time_group", "start", "end"]).max()["trip_distance"]
    else:
        origin_dest_trips = time_group_trips.groupby(["time_group", "start", "end"]).mean()[["trip_duration"]]
    origin_dest_trips["trips_counts"] = time_group_trips.groupby(["time_group", "start", "end"]).count()["trip_duration"]
    origin_dest_trips["trip_duration_min"] = time_group_trips.groupby(["time_group", "start", "end"]).min()["trip_duration"]
    origin_dest_trips["trip_duration_max"] = time_group_trips.groupby(["time_group", "start", "end"]).max()["trip_duration"]
    origin_dest_trips[origin_dest_trips.trips_counts != 1].head()

    unique_edges_dict = {}
    edges_travel_time = edges_df.set_index(["from_node", "to_node"])["travel_time"].to_dict()
    edges_travel_distance = edges_df.set_index(["from_node", "to_node"]).length.to_dict()
    graph_ig = create_igraph(nodes_df, edges_df)
    list_df = []
    for time_group, df in tqdm(origin_dest_trips.groupby(level=0)):
        path_nodes = []
        edges_list = []
        travel_times = []
        travel_distances = []
        unique_edges = set()

        for inx, row in df.iterrows():
            nodes = graph_ig.get_shortest_paths(v=inx[1], to=inx[2], weights="length")[0]
            path_nodes.append(nodes)
            edges = list(zip(nodes, nodes[1:]))
            edges_list.append(edges)
            unique_edges = unique_edges.union(edges)
            travel_times.append(sum([edges_travel_time[x] for x in edges]))
            travel_distances.append(sum([edges_travel_distance[x] for x in edges]))
        unique_edges = list(unique_edges)
        df["path_nodes"] = path_nodes
        df["path_edges"] = edges_list
        df["free_flow_time"] = travel_times
        df["osm_network_distance"] = travel_distances

        mean_travel_ratio = df["trip_duration"].sum() / df["free_flow_time"].sum()
        df["scaled_time (mean factored)"] = [sum([mean_travel_ratio * edges_travel_time[x] for x in edges]) for edges in
                                             df["path_edges"].values]
        df["error (mean factored)"] = df["trip_duration"] - df["scaled_time (mean factored)"]

        unique_edges_dict[time_group] = unique_edges
        list_df.append(df)

    return pd.concat(list_df), unique_edges_dict


def solve_network_optimization(edges_df, trips_df, objectives, unique_edges, zone_edges_dict, multi_zone_edges):
    free_flow_times = edges_df.set_index(["from_node", "to_node"])["travel_time"].to_dict()
    edges_distances = edges_df.set_index(["from_node", "to_node"])["distance"].to_dict()
    all_edges = edges_df.set_index(["from_node", "to_node"]).index.tolist()
    # Minimum speed for calculating maximum travel time on an edge
    minimum_speed = 1.609 / 3.6
    edges_df_with_factor = edges_df.copy()
    trips_df = trips_df.set_index(["start", "end"]).copy()

    for objective in objectives:
        m = grbpy.Model("network edges scaling")
        m.setParam("OutputFlag", 0)

        # The variables provide the scaling factor for each of the edges
        var_dict = {tuple(x): None for x in unique_edges}

        # Add variables to the model with the constraints for the least travel time (or maximum speed limits) on the edges
        for edge in var_dict:
            free_flow = free_flow_times[edge]
            max_travel_time = edges_distances[edge] / minimum_speed
            var_dict[edge] = m.addVar(name=f"x_{edge[0]}_{edge[1]}", lb=1, vtype=grbpy.GRB.CONTINUOUS)
            m.addConstr(var_dict[edge] <= max_travel_time / free_flow)

        m.update()
        objective_terms = []
        scaled_travel_times = []
        scalled_error = []

        for (origin_node, dest_node), row in trips_df.iterrows():
            scaled_time = sum([free_flow_times[edge] * var_dict[edge] for edge in row.path_edges])
            error = row["trip_duration"] - scaled_time
            scaled_travel_times.append(scaled_time)
            scalled_error.append(error)

            if objective == "abs_factored":
                z1 = m.addVar(name=f"z1_{origin_node}_{dest_node}", lb=0, ub=grbpy.GRB.INFINITY,
                              vtype=grbpy.GRB.CONTINUOUS)
                z2 = m.addVar(name=f"z2_{origin_node}_{dest_node}", lb=0, ub=grbpy.GRB.INFINITY,
                              vtype=grbpy.GRB.CONTINUOUS)
                m.addConstr(z1 - z2 == error)
                objective_terms.append(z1 + z2)
            elif objective == "squ_factored":
                z = m.addVar(name=f"z_{origin_node}_{dest_node}", lb=-grbpy.GRB.INFINITY, ub=grbpy.GRB.INFINITY,
                             vtype=grbpy.GRB.CONTINUOUS)
                scaled_time = sum([free_flow_times[edge] * var_dict[edge] for edge in row.path_edges])
                error = scaled_time - row["trip_duration"]
                m.addConstr(z == error)
                objective_terms.append(z * z)
            else:
                raise TypeError("wrong objective")

        m.setObjective(sum(objective_terms), grbpy.GRB.MINIMIZE)
        m.update()
        m.optimize()

        trips_df[f"network_time ({objective})"] = [round(x.getValue(), 2) for x in scaled_travel_times]
        trips_df[f"error ({objective})"] = [round(x.getValue(), 2) for x in scalled_error]

        edges_df_with_factor[f"scale_factor {objective}"] = np.nan
        edges_df_with_factor[f"scaled_speed_kph {objective}"] = np.nan
        for edge in var_dict:
            mask = (edges_df_with_factor.from_node == edge[0]) & (edges_df_with_factor.to_node == edge[1])
            edges_df_with_factor.loc[mask, f"scale_factor {objective}"] = var_dict[edge].X
        mask = ~edges_df_with_factor[f"scale_factor {objective}"].isnull()
        masked_df = edges_df_with_factor[mask]
        scaled_speed = 3.6 * masked_df["distance"] / (masked_df[f"scale_factor {objective}"] * masked_df["travel_time"])
        edges_df_with_factor.loc[mask, f"scaled_speed_kph {objective}"] = scaled_speed

    # Set the values for the non-unique edges
    edges_factor = {}
    e_df = edges_df_with_factor.set_index(["from_node", "to_node"]).copy()
    max_edge_factors = {e: (edges_distances[e] / minimum_speed) / free_flow_times[e] for e in all_edges}
    zones_with_mean_factor_scaling = {}
    multizone_with_mean_factor_scaling = []

    for objective in objectives:
        mean_factor = edges_df_with_factor[f"scale_factor {objective}"].mean()
        zone_mean = {}
        for zone_id, zone_edges in zone_edges_dict.items():
            unique_edges_zone = set(zone_edges).intersection(unique_edges)
            unique_edges_zone = list(unique_edges_zone.difference(multi_zone_edges.keys()))
            if len(unique_edges_zone) > 0:
                factor = e_df.loc[unique_edges_zone][f"scale_factor {objective}"].mean()
                zone_mean[zone_id] = factor
            else:
                factor = mean_factor
                zones_with_mean_factor_scaling[zone_id] = list(
                    zone_edges.difference(multi_zone_edges.keys(), unique_edges))
            for e in zone_edges.difference(multi_zone_edges.keys(), unique_edges):
                edges_factor[e] = min(factor, max_edge_factors[e])

        for e, e_zones in multi_zone_edges.items():
            neigh_zone_with_unique_edge = [z for z in e_zones if z in zone_mean]
            if len(neigh_zone_with_unique_edge) > 0:
                factor = sum([zone_mean[z] for z in neigh_zone_with_unique_edge]) / len(neigh_zone_with_unique_edge)
            else:
                factor = mean_factor
                multizone_with_mean_factor_scaling.append(e)
            edges_factor[e] = min(factor, max_edge_factors[e])

        for e, factor in edges_factor.items():
            e_df.loc[e, f"scale_factor {objective}"] = factor

        scaled_speed = 3.6 * e_df["distance"] / (e_df[f"scale_factor {objective}"] * e_df["travel_time"])
        e_df[f"scaled_speed_kph {objective}"] = scaled_speed

    return e_df.reset_index(), trips_df, zones_with_mean_factor_scaling, multizone_with_mean_factor_scaling


def generate_scaled_dyn_edges(network_name, demand, demand_file_name, zones_name, aggregation_period_min, objective,
                              crs=None):
    """ Generates the scaled edges for the given network and demand data. The edges are scaled based on the historical
    travel times from the demand data. The scaled edges are saved in the individual folders for each time group
    according to the aggregation period.
    The code assumes WGS84 (EPSG:4326) as the default CRS for the input data. If the input data is in a different CRS,
    it should be provided as an argument to the function.
    """

    # Set the folders
    data_folder = Path(__file__).resolve().absolute().parents[3].joinpath("data")
    zone_main_folder = data_folder.joinpath(f"zones/{zones_name}")
    zone_folder = data_folder.joinpath(f"zones/{zones_name}/{network_name}")
    network_folder = data_folder.joinpath(f"networks//{network_name}")

    # ***********************************************************************************************

    if crs is None:
        print("No CRS provided. Assuming the input data is in WGS84 (EPSG:4326). If the input data is in a different CRS, please provide the correct CRS as an argument to the function.")
        crs = 4326

    # Load zones and calculate valid neighbours
    zones = zone_main_folder.joinpath("polygon_definition.geojson")
    zones = gpd.read_file(str(zones))
    if zones.crs != crs:
        print(f"Zone crs will be overwritten {zones.crs} with the provided crs: {crs}. "
              f"Please double-check the consistency of the input data!")
        zones.set_crs(crs, inplace=True, allow_override=True)

    # Read edges and nodes
    nodes_df = gpd.read_file(network_folder.joinpath("base/nodes_all_infos.geojson"))
    if nodes_df.crs != crs:
        print(f"Node crs will be overwritten {nodes_df.crs} with the provided crs: {crs}. "
              f"Please double-check the consistency of the input data!")
        nodes_df.set_crs(crs, inplace=True, allow_override=True)
    nodes_df["zone_id"] = calculate_subzone(nodes_df.geometry.y, nodes_df.geometry.x, zones, "zone_id")
    edges_df = gpd.read_file(network_folder.joinpath("base/edges_all_infos.geojson"))
    if edges_df.crs != crs:
        print(f"Edge crs will be overwritten {edges_df.crs} with the provided crs: {crs}. "
              f"Please double-check the consistency of the input data!")
        edges_df.set_crs(crs, inplace=True, allow_override=True)

    edges_df["from_zone"] = nodes_df.loc[edges_df.from_node.values]["zone_id"].values
    edges_df["to_zone"] = nodes_df.loc[edges_df.to_node.values]["zone_id"].values
    edges_df = calculate_zone_intersect_edges(edges_df, zones)

    zone_edges_dict = defaultdict(set)
    for _, row in edges_df.iterrows():
        for z in row.zones:
            zone_edges_dict[z].add((row.from_node, row.to_node))
    zone_edges_dict = dict(zone_edges_dict)
    multi_zone_edges = edges_df[edges_df.is_multiple_zones == True].set_index(["from_node", "to_node"]).zones.to_dict()

    # Plot zones and edges to check the data
    # TODO: check why this is not working
    plot_zones_edges(zones, edges_df)

    # Read Demand and make groups based on the aggregation period
    day_df = pd.read_csv(data_folder.joinpath(f"demand/{demand}/matched/{network_name}/{demand_file_name}.csv"))
    day_df["trip_duration"] = day_df["dropoff_time"] - day_df["rq_time"]
    day_df.loc[day_df.trip_duration < 0, "trip_duration"] = day_df["dropoff_time"] + 24*60*60 - day_df["rq_time"]
    origin_dest_trips, unique_edge_dict = prepare_trip_data_for_time_groups(day_df, nodes_df, edges_df, aggregation_period_min)


    # Solve the optimization problem
    scaled_folder_rel_path = f"scaled_edges/{objective}_{int(aggregation_period_min)}/{demand_file_name}"
    scale_edges_main_folder = network_folder.joinpath(scaled_folder_rel_path)
    scale_demand_folder = network_folder.joinpath(f"scaled_demand/period_{int(aggregation_period_min)}/{demand_file_name}")

    for f in [scale_demand_folder, scale_edges_main_folder]:
        if f.exists() is False:
            f.mkdir(parents=True)


    max_time_group = origin_dest_trips.index[-1][0] + 1
    dyn_factor_files = {"simulation_time": [], "travel_time_folder": []}

    for time_group in tqdm(range(0, max_time_group)):
        time = int(time_group * aggregation_period_min * 60)
        try:
            tmp_od = origin_dest_trips.loc[time_group, :, :].reset_index()
            edges_df_with_factor, trip_df, zone_edge_with_mean, multi_zone_edge_with_mean = solve_network_optimization(
                edges_df, tmp_od,[objective], unique_edge_dict[time_group],
                zone_edges_dict, multi_zone_edges
            )

            edges_df_with_factor["edge_tt"] = edges_df_with_factor["travel_time"] * edges_df_with_factor[f"scale_factor {objective}"]
            edges_df_with_factor = edges_df_with_factor[["from_node", "to_node", "edge_tt"]]

            edge_folder = scale_edges_main_folder.joinpath(f"{time}")
            if edge_folder.exists() is False:
                edge_folder.mkdir(parents=True)

            edges_df_with_factor.to_csv(edge_folder.joinpath("edges_td_att.csv"), index=False)
            dyn_factor_files["simulation_time"].append(time)
            dyn_factor_files["travel_time_folder"].append(f"{scaled_folder_rel_path}/{time}")

            trip_df = trip_df.drop(columns=["path_nodes", "path_edges"])
            trip_df.to_csv(scale_demand_folder.joinpath(f"{time}.csv"))
        except:
            print(f"Time {time}: Problem when reseting index of OD trips dataframe with dimension:"
                  f" {origin_dest_trips.shape}. Continueing with next time group.")

    dyn_folders_df = pd.DataFrame(dyn_factor_files)
    dyn_file = network_folder.joinpath(f"dyn_factors_{objective}_{aggregation_period_min}.csv")
    dyn_folders_df.to_csv(dyn_file, index=False)
    print("\nThe scaled edges are saved in the following folder: ", scale_edges_main_folder)
    print("\nThe travel times of individual trips calculated using the scaled edges are saved in the following folder: ", scale_demand_folder)
    print("\n*** The cumulative dynamic factors file that can be used by the FleetPy simulation: ", dyn_file, "***")


if __name__ == "__main__":
    """ This scripts calculates the travel times of each edge according to the historical travel times from the demand 
    data. It solves an optimization problem to find the scaling factors for each edge that minimize the error between 
    the scaled travel times and the historical travel times from the demand data.
    
    The details of the scaling method is described in the following paper:
    Syed, Arslan Ali, Yunfei Zhang, and Klaus Bogenberger. "Data-driven spatio-temporal scaling of travel times for AMoD
    simulations." (ITSC). IEEE, 2023.
    
    ************    IMPORTANT       ***************
    # For the scaling method to work, the demand data must have an additional column "dropoff_time"  marking the 
    # historical travel time of the trips. 
    # Moreover, a column "trip_distance" is optional.
    
    **** OUTPUT ****
    
    (1) The scaled edges for each time group are saved in the individual folders under the network folder. 
    A time group is defined by the aggregation period.
    
    (2) The script also produces an additional csv file whose name starts with "dyn_factors" in the network folder. 
    This file contains the mapping of the simulation time to the folder name of the scaled edges for that time group. 
    This file is used by the FleetPy simulation to load the correct scaled edges for each time group.
    """

    # ******  Modify the following parameters according to your network and demand data. ******
    network_name = "Manhattan_2019_corrected"
    demand = "manhattan_2018"
    demand_file_name = "2018-11-11_sample_5_1"
    zones_name = "Manhattan_Taxi_Zones"
    aggregation_period_min = 30     # The aggregation period given in minutes.
    objective = "abs_factored"    # Other possible objective is "squ_factored"
    crs_epsg = 32618

    generate_scaled_dyn_edges(network_name, demand, demand_file_name, zones_name, aggregation_period_min, objective,
                              crs_epsg)

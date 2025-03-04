import numpy as np
import pandas as pd
import os
import shapely
from shapely.geometry import LineString
import geopandas as gpd
import pyproj
from shapely.ops import nearest_points
from scipy.stats import poisson
from scipy.special import comb
import plotly.graph_objects as go
from src.routing.NetworkBasic import NetworkBasic
import rtree
import math


class PTScheduleGen:
    def __init__(self, route_no: int, gtfs_path: str,
                 to_trip_id: str, network_path: str,
                 shape_ids: list[str],
                 back_trip_id: str | None = None, dwell_time=30, ):
        """
        PT Route Schedule Generator
        :param route_no: the route number
        :param gtfs_path: the path to the GTFS directory with files (routes, trips, shapes, stops, stop_times)
        :param to_trip_id: the trip ID for the forward direction
        :param network_path: folder path of network (consistency with FleetPy)
        :param shape_ids: the shape IDs for the route
        :param back_trip_id: the trip ID for the backward direction
            (optional: if None, assumed to be the reverse of to_trip_id)
        :param dwell_time: the dwell time at each stop (s)
        """
        # and migrate to general FleetPy demand files

        self.hourly_demand = None
        self.route_with_coords_origin_gdf = None
        self.route_with_coords = None

        print("Loading data for route {}".format(route_no))
        self.route_no = route_no

        print("Loading GTFS data for route {}".format(route_no))
        # Read GTFS
        routes = pd.read_csv(os.path.join(gtfs_path, 'routes.txt'))
        trips = pd.read_csv(os.path.join(gtfs_path, 'trips.txt'))
        shapes = pd.read_csv(os.path.join(gtfs_path, 'shapes.txt'))
        stops = pd.read_csv(os.path.join(gtfs_path, 'stops.txt'))
        stop_times = pd.read_csv(os.path.join(gtfs_path, 'stop_times.txt'))

        # Identify the route ID for the route
        route_id = routes[routes['route_short_name'] == str(route_no)]['route_id'].iloc[0]

        # Filter shapes
        filtered_shapes = shapes[shapes['shape_id'].isin(shape_ids)]

        # sort filtered_shapes by shape_pt_sequence
        filtered_shapes = filtered_shapes.sort_values(by=['shape_pt_sequence'])

        print("Creating alignment for route {}".format(route_no))
        # Create alignment
        linestrings = []
        for shape_id in shape_ids:
            shape_points = filtered_shapes[filtered_shapes['shape_id'] == shape_id]
            points = shape_points[['shape_pt_lon', 'shape_pt_lat']].values
            linestring = LineString(points)
            linestrings.append(linestring)
        all_points = []
        for linestring in linestrings:
            # Extend the list with points from each LineString
            all_points.extend(list(linestring.coords))
        self.alignment = LineString(all_points)

        # convert crs of line
        self.dest_crs = "EPSG:32632"  # WGS 84 / UTM zone 32N, for measurement in m
        self.network_crs = "EPSG:4326"
        self.meter_project = pyproj.Transformer.from_crs(pyproj.CRS(self.network_crs), pyproj.CRS(self.dest_crs),
                                                         always_xy=True).transform

        self.line_alignment_meter = shapely.ops.transform(self.meter_project, self.alignment)
        self.route_length = self.line_alignment_meter.length / 1000  # convert to km

        # Filter trips for Route
        trips_for_route = trips[trips['route_id'] == route_id]

        # Get stop IDs from the trips
        stop_ids = stop_times[stop_times['trip_id'].isin(trips_for_route['trip_id'])]['stop_id'].unique()

        # Get stop coordinates
        stop_coordinates = stops[stops['stop_id'].isin(stop_ids)][['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

        network_nodes_file = os.path.join(network_path, "base", "nodes_all_infos.geojson")
        network_nodes = gpd.read_file(network_nodes_file)
        self.network_nodes = network_nodes
        # set original crs of self.network_nodes to be self.network_crs
        self.network_nodes.crs = self.network_crs

        self.network_nodes_crs = self.network_nodes.to_crs(self.dest_crs)
        stop_coordinates = gpd.GeoDataFrame(stop_coordinates,
                                            geometry=gpd.points_from_xy(stop_coordinates['stop_lon'],
                                                                        stop_coordinates['stop_lat']))

        # Create a spatial index for the network nodes
        spatial_index = rtree.index.Index()
        for idx, geom in self.network_nodes_crs.geometry.items():
            spatial_index.insert(idx, geom.bounds)
        self.spatial_index = spatial_index

        # Prepare columns for the nearest node and distance
        stop_coordinates['nearest_node'] = None
        stop_coordinates['distance'] = None

        # Iterate over each stop coordinate
        for index, stop in stop_coordinates.iterrows():
            # Find the nearest point and distance
            nearest_node, distance = self.find_nearest_and_distance(stop.geometry, self.network_nodes_crs,
                                                                    self.spatial_index)

            # Store the information in the stop_coordinates GeoDataFrame
            stop_coordinates.at[index, 'nearest_node'] = nearest_node['node_index']
            # stop_coordinates.at[index, 'distance'] = distance * 111139
            stop_coordinates.at[index, 'distance'] = distance

        print("Creating schedule for route {}".format(route_no))
        route_stop_seq = stop_times[stop_times['trip_id'] == to_trip_id]["stop_id"].to_list()
        if back_trip_id is not None:
            route_stop_seq.extend(stop_times[stop_times['trip_id'] == back_trip_id]["stop_id"].iloc[1:].to_list())
        # create a dataframe for the route stop sequence
        route_stop_seq_df = pd.DataFrame(route_stop_seq, columns=["stop_id"])
        # match with stop_coordinates["nearest_node"]
        route_stop_seq_df = route_stop_seq_df.merge(
            stop_coordinates.reset_index()[["index", "stop_id", "nearest_node"]], on="stop_id")
        self.stop_coordinates = stop_coordinates

        # if back_trip_id is None, create a copy of the original DataFrame in reverse order
        if back_trip_id is None:
            # Create a copy of the original DataFrame in reverse order
            route_stop_seq_reversed = route_stop_seq_df.iloc[::-1].reset_index(drop=True)
            # Concatenate the original and reversed DataFrames
            route_stop_seq_df = pd.concat([route_stop_seq_df, route_stop_seq_reversed], ignore_index=True)

        print("Calculating travel time for route {}".format(route_no))
        # Create a network object through FleetPy
        fleetpy_network = NetworkBasic(network_path)

        # along route_stop_seq_df["nearest_node"], find the shortest path length between every two nodes
        for i in range(len(route_stop_seq_df) - 1):
            route_stop_seq_df.loc[i, "travel_time"] = fleetpy_network.return_travel_costs_1to1(
                (route_stop_seq_df.loc[i, "nearest_node"], None, None),
                (route_stop_seq_df.loc[i + 1, "nearest_node"], None, None)
            )[1]

        # calculate the arrival time for each stop, including the dwell time, round up to the nearest 30 s
        for i in range(len(route_stop_seq_df)):
            if i == 0:
                route_stop_seq_df.at[i, "arrival_time"] = 0
            else:
                route_stop_seq_df.at[i, "arrival_time"] = route_stop_seq_df.at[i - 1, "arrival_time"] + \
                                                          route_stop_seq_df.at[i - 1, "travel_time"] + dwell_time
                # round up to the nearest 30s
                route_stop_seq_df.at[i, "arrival_time"] = (int(route_stop_seq_df.at[i, "arrival_time"] / 30) + 1) * 30
        self.route_stop_seq_df = route_stop_seq_df
        print("Initialization finished for route {}".format(route_no))

    def load_demand(self, demand_csv: str):
        """
        Load the demand file and generate self.route_with_coords
        :param demand_csv: the path to the demand file
        (with columns route_ID, route_departure, origin_stop, destination_stop)
        """
        demand_df = pd.read_csv(demand_csv)
        route_df = demand_df[demand_df['route_ID'] == route_no]

        route_with_coords = route_df[['route_departure', 'origin_stop', 'destination_stop']].copy()
        # add the index of route_df as well
        route_with_coords["index"] = route_df.index
        print(route_with_coords.head(5))

        stop_coordinates_oneway = self.stop_coordinates.drop_duplicates(subset=["stop_name"])
        print(stop_coordinates_oneway.head(5))
        # Merge the dataframes to get coordinates for each stop
        route_with_coords = route_with_coords.merge(stop_coordinates_oneway[['stop_name', 'stop_lat', 'stop_lon']],
                                                    left_on='origin_stop', right_on='stop_name', how='left')
        route_with_coords.rename(columns={'stop_lat': 'origin_lat', 'stop_lon': 'origin_lon'}, inplace=True)
        route_with_coords = route_with_coords.merge(stop_coordinates_oneway[['stop_name', 'stop_lat', 'stop_lon']],
                                                    left_on='destination_stop', right_on='stop_name', how='left')
        route_with_coords.rename(columns={'stop_lat': 'destination_lat', 'stop_lon': 'destination_lon'}, inplace=True)

        # convert route_departure from time format to seconds
        for i in route_with_coords.index:
            sep_colon_part = route_with_coords.at[i, "route_departure"].split(":")
            if len(sep_colon_part[0]) > 2:
                # print(i)
                sep_colon_part = route_with_coords.at[i, "route_departure"].split(" ")[1].split(":")
            route_with_coords.at[i, "route_departure"] = int(sep_colon_part[0]) * 3600 + int(
                sep_colon_part[1]) * 60 + int(sep_colon_part[2])

        self.route_with_coords: pd.DataFrame = route_with_coords

        print("Demand file loaded for route {}".format(route_no))
        print(route_with_coords.head(5))

    def check_demand_loaded(self):
        """
        Check if the demand file is loaded; raise exception if self.route_with_coords is None
        """
        if self.route_with_coords is None:
            raise ValueError("route_with_coords is None, please load demand file first with load_demand(demand_csv)")

    def return_hourly_demand(self, time_range: tuple[int, int], demand_factor=1.0):
        """
        Return the hourly demand for the route
        """
        self.check_demand_loaded()

        # filter by time_range
        route_with_coords = self.route_with_coords.copy()

        if time_range is not None:
            route_with_coords = route_with_coords[(route_with_coords["route_departure"] >= time_range[0]) &
                                                  (route_with_coords["route_departure"] <= time_range[1])]

        self.hourly_demand = route_with_coords.shape[0] / (time_range[1] - time_range[0]) * 3600 * demand_factor
        return self.hourly_demand

    def return_terminus_demand_proportion(self, terminus_stop: str, time_range: tuple[int, int]):
        """
        Return the proportion of demand at the terminus stop
        """
        self.check_demand_loaded()

        route_with_coords = self.route_with_coords.copy()

        if time_range is not None:
            route_with_coords = route_with_coords[(route_with_coords["route_departure"] >= time_range[0]) &
                                                  (route_with_coords["route_departure"] <= time_range[1])]

        terminus_demand = route_with_coords[route_with_coords["origin_stop"] == terminus_stop].shape[0] + \
                          route_with_coords[route_with_coords["destination_stop"] == terminus_stop].shape[0]
        total_demand = route_with_coords.shape[0]

        return terminus_demand / total_demand

    def return_route_length(self):
        """
        Return the route length
        """
        return self.route_length

    def return_demand_walking_dist(self, max_len: float, x: float):
        """
        Return the demand distribution based on walking distance
        """
        return (abs(max_len) - abs(x)) / abs(max_len)

    def output_demand(self, terminus_stop: str, demand_file=None, max_distance_km=0.5, seed=0, time_range=None,
                      export_fixed_route=False, demand_factor=1.0, save_complete=False, offset_time=1800,
                      ) -> None:
        """
        Generate passenger demand for the route
        :param terminus_stop: the terminus stop name of the route
        :param demand_file: the file to save the demand
        :param max_distance_km: the maximum distance of the random offset
        :param seed: the random seed
        :param time_range: the time range for the demand
        :param export_fixed_route: whether to export the fixed route as well
        :param demand_factor: the demand factor
        :param save_complete: whether to save the complete demand file
        :param offset_time: the offset time for the demand
        :return: None
        """
        self.check_demand_loaded()

        route_with_coords = self.route_with_coords.copy()

        # Set the seed
        np.random.seed(seed)

        if time_range is not None:
            route_with_coords = route_with_coords[(route_with_coords["route_departure"] >= time_range[0]) &
                                                  (route_with_coords["route_departure"] <= time_range[1])]

        # Apply demand factor to draw new rows from the existing rows
        route_with_coords = route_with_coords.sample(frac=demand_factor, replace=True)

        # check if there is duplicated rid, if so, add suffix starting from 00
        if route_with_coords["index"].duplicated().any():
            route_with_coords["index"] = route_with_coords["index"].astype(str)
            route_with_coords["index"] = route_with_coords["index"] + route_with_coords.groupby(
                "index").cumcount().astype(str).str.zfill(2)

        route_with_coords.reset_index(drop=True, inplace=True)
        # sort by route_departure
        route_with_coords.sort_values(by=["route_departure"], inplace=True)

        # Apply random offsets
        route_with_coords[['offset_origin_lat', 'offset_origin_lon']] = route_with_coords.apply(
            lambda row: pd.Series(self.random_offset(max_distance_km=max_distance_km)), axis=1)
        route_with_coords[['offset_destination_lat', 'offset_destination_lon']] = route_with_coords.apply(
            lambda row: pd.Series(self.random_offset(max_distance_km=max_distance_km)), axis=1)

        # No adjustment for Bf.
        route_with_coords.loc[route_with_coords["origin_stop"] == terminus_stop, "offset_origin_lat"] = 0
        route_with_coords.loc[route_with_coords["origin_stop"] == terminus_stop, "offset_origin_lon"] = 0
        route_with_coords.loc[route_with_coords["destination_stop"] == terminus_stop, "offset_destination_lat"] = 0
        route_with_coords.loc[route_with_coords["destination_stop"] == terminus_stop, "offset_destination_lon"] = 0

        # Add the offsets to the original coordinates
        route_with_coords['final_origin_lat'] = route_with_coords['origin_lat'] + route_with_coords['offset_origin_lat']
        route_with_coords['final_origin_lon'] = route_with_coords['origin_lon'] + route_with_coords['offset_origin_lon']
        route_with_coords['final_destination_lat'] = route_with_coords['destination_lat'] + route_with_coords[
            'offset_destination_lat']
        route_with_coords['final_destination_lon'] = route_with_coords['destination_lon'] + route_with_coords[
            'offset_destination_lon']
        route_with_coords = route_with_coords.dropna()

        # convert route_with_coords to GeoDataFrame
        # Prepare columns for the nearest node and distance
        route_with_coords['origin_node'] = None
        route_with_coords['origin_distance'] = None
        route_with_coords['destination_node'] = None
        route_with_coords['destination_distance'] = None

        # Iterate over each stop coordinate
        for index, stop in route_with_coords.iterrows():
            # Find the nearest point and distance
            origin_point = shapely.Point(stop['final_origin_lon'], stop['final_origin_lat'])
            nearest_node, distance = self.find_nearest_and_distance(origin_point, self.network_nodes_crs,
                                                                    self.spatial_index)
            route_with_coords.at[index, 'origin_node'] = nearest_node['node_index']
            route_with_coords.at[index, 'origin_distance'] = distance

            destination_point = shapely.Point(stop['final_destination_lon'], stop['final_destination_lat'])
            nearest_node, distance = self.find_nearest_and_distance(destination_point, self.network_nodes_crs,
                                                                    self.spatial_index)
            route_with_coords.at[index, 'destination_node'] = nearest_node['node_index']
            route_with_coords.at[index, 'destination_distance'] = distance

        route_with_coords_origin_gdf = route_with_coords
        self.route_with_coords_origin_gdf = route_with_coords_origin_gdf
        if demand_file is None:
            return

        # randomly offset route_departure time
        route_with_coords_origin_gdf["route_departure"] = \
            route_with_coords_origin_gdf["route_departure"] + \
            np.random.randint(-offset_time, offset_time, route_with_coords_origin_gdf.shape[0])
        route_with_coords_origin_gdf["route_departure"] = route_with_coords_origin_gdf["route_departure"] % 86400
        route_with_coords_origin_gdf.sort_values(by=["route_departure"], inplace=True)

        passenger_demand = route_with_coords_origin_gdf[
            ["route_departure", "origin_node", "destination_node", "index"]].rename(
            columns={"route_departure": "rq_time", "origin_node": "start", "destination_node": "end",
                     "index": "request_id"})

        passenger_demand.to_csv(demand_file, index=False)
        print("Demand file saved to {}".format(demand_file))
        if save_complete:
            route_with_coords_origin_gdf.to_csv(demand_file.replace(".csv", "_complete.csv"), index=False)

        if export_fixed_route:
            max_lat_lon = max_distance_km / 111.139
            # for fixed route, keep each row by probability based on the mean of return_demand_walking_dist()
            # of "offset_destination_lat" and "offset_destination_lon"
            route_with_coords_origin_gdf_fixed = route_with_coords_origin_gdf.copy()
            index_to_drop = []
            for index, stop in route_with_coords_origin_gdf_fixed.iterrows():
                # calculate the prob
                prob_keep = (self.return_demand_walking_dist(max_lat_lon, stop["offset_destination_lat"]) +
                             self.return_demand_walking_dist(max_lat_lon, stop["offset_destination_lon"])) / 2
                # randomly keep the row based on the walking distance
                if np.random.rand() > prob_keep:
                    index_to_drop.append(index)
            route_with_coords_origin_gdf_fixed = route_with_coords_origin_gdf_fixed.drop(index_to_drop)
            passenger_demand_fixed = route_with_coords_origin_gdf_fixed[
                ["route_departure", "origin_node", "destination_node", "index"]].rename(
                columns={"route_departure": "rq_time", "origin_node": "start", "destination_node": "end",
                         "index": "request_id"})
            passenger_demand_fixed.to_csv(demand_file.replace(".csv", "_fixed.csv"), index=False)
            print("Fixed demand file saved to {}".format(demand_file.replace(".csv", "_fixed.csv")))
            if save_complete:
                route_with_coords_origin_gdf_fixed.to_csv(demand_file.replace(".csv", "_complete_fixed.csv"),
                                                          index=False)

    def save_alignment_geojson(self, output_path, file_name=None):
        """
        Save the alignment to a GeoJSON file
        :param output_path: the path to save the alignment file
        :param file_name: the file name of the alignment file
        """
        if file_name is None:
            file_name = str(self.route_no) + '_line_alignment.geojson'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        gdf = gpd.GeoDataFrame(geometry=[self.alignment])
        gdf.to_file(os.path.join(output_path, file_name), driver='GeoJSON')
        print("Alignment saved to {}".format(os.path.join(output_path, file_name)))

    def find_nearest_and_distance(self, input_point: shapely.Point,
                                  gdf: gpd.GeoDataFrame,
                                  spatial_index: rtree.index,
                                  input_crs="EPSG:4326"):
        """
        Find the nearest node and distance
        :param input_point: the input point
        :param gdf: the GeoDataFrame
        :param spatial_index: the spatial index (rtree)
        :param input_crs: the input CRS
        :return: the nearest node and distance
        """
        # Define the input CRS and the target CRS from the GeoDataFrame
        # WGS84 Latitude/Longitude
        target_crs = gdf.crs  # CRS of the GeoDataFrame

        input_lon, input_lat = input_point.x, input_point.y

        # Create a transformer function to convert from input_crs to target_crs
        transformer = pyproj.Transformer.from_crs(input_crs, target_crs, always_xy=True)
        transformed_point = transformer.transform(input_lon, input_lat)
        input_point = shapely.Point(transformed_point)

        # Find the nearest point using the spatial index
        nearest_idx = list(spatial_index.nearest(input_point.bounds, 1))[0]
        nearest_geom = gdf.geometry.iloc[nearest_idx]
        nearest_point = gdf.iloc[nearest_idx]

        # Calculate distance in the transformed CRS
        distance = input_point.distance(nearest_geom)

        return nearest_point, distance

    def output_station(self, output_path, station_file="stations.csv", only_in_schedule=True):
        """
        Output the station file
        :param output_path: the path to save the station file
        :param station_file: the file name of the station file
        :param only_in_schedule: whether to output only the stations in the schedule
        """
        output_schedule = self.stop_coordinates.nearest_node.reset_index()
        output_schedule.columns = ['station_id', 'network_node_index']
        if only_in_schedule:
            route_stop_seq_df = self.route_stop_seq_df
            output_schedule = output_schedule[output_schedule["station_id"].isin(route_stop_seq_df["index"])]

        output_schedule.to_csv(os.path.join(output_path, station_file), index=False)
        print("Station file saved to {}".format(os.path.join(output_path, station_file)))

    def return_schedule_req(self, turnaround_time=10 / 60,
                            headway=10 / 60, n_veh=5):
        """
        Return the schedule vehicle requirements
        """
        route_stop_seq_df = self.route_stop_seq_df
        # calculate the number of vehicles required
        journey_time = max(route_stop_seq_df["arrival_time"])
        n_veh_req = journey_time / headway / 3600

        flex_buffer = headway * 3600 * n_veh - journey_time - turnaround_time
        return journey_time, n_veh_req, flex_buffer

    def output_schedule(self, output_path,
                        schedules_file="schedules.csv",
                        veh_type="default_vehtype"
                        ) -> None:
        """
        Output the schedule file
        :param output_path: the path to save the schedule file
        :param schedules_file: the file name of the schedule file
        :param veh_type: the vehicle type
        """

        route_stop_seq_df = self.route_stop_seq_df.copy()

        schedules_df = pd.DataFrame()

        new_rows = route_stop_seq_df[["arrival_time", "index"]].rename(
            columns={"arrival_time": "departure", "index": "station_id"}).assign(
            departure=route_stop_seq_df["arrival_time"])
        new_rows["trip_id"] = 0
        new_rows["line_vehicle_id"] = 0
        schedules_df = pd.concat([schedules_df, new_rows])

        schedules_df["LINE"] = self.route_no
        schedules_df["vehicle_type"] = veh_type

        schedules_df.to_csv(os.path.join(output_path, schedules_file), index=False)
        print("Schedule file saved to {}".format(os.path.join(output_path, schedules_file)))

    def return_terminus_id(self) -> int:
        """
        Return the terminus id as the first station_id in the schedule
        """
        return self.route_stop_seq_df.iloc[0]["index"]

    # Function to generate random offset
    def random_offset(self, max_distance_km=1.) -> tuple[float, float]:
        """
        Generate random offset
        :param max_distance_km: the maximum distance of the random offset
        :return: the random offset (lat, lon) in degrees
        """
        # Convert max distance to degrees (approximation)
        max_distance_deg = max_distance_km / 111.139  # rough approximation: 1 degree = 111 km

        # Generate random offsets in degrees
        offset_lat = np.random.uniform(-max_distance_deg, max_distance_deg)
        offset_lon = np.random.uniform(-max_distance_deg, max_distance_deg)
        return offset_lat, offset_lon

    def output_all_demand(
            self,
            n: int,
            terminus_stop: str,
            demand_folder: str,
            max_distance_km=0.5,
            # seed=0,
            time_range=None,
            start_i=0,
            export_fixed_route=False,
            demand_factor=1.0,
            save_complete=False
    ) -> None:
        """
        Generate passenger demand for the route
        :param n: the number of demand files to generate
        :param terminus_stop: the terminus stop name of the route
        :param demand_folder: the folder to save the demand files
        :param max_distance_km: the maximum distance of the random offset
        :param time_range: the time range for the demand
        :param start_i: the starting index for the demand files
        :param export_fixed_route: whether to export the fixed route as well
        :param demand_factor: the factor to multiply the demand
        :param save_complete: whether to save the complete demand file
        :return: None
        """
        for i in range(start_i, n):
            print("Generating demand for seed {}".format(i))
            demand_file = os.path.join(demand_folder, "passenger_demand{}.csv".format(i))
            self.output_demand(terminus_stop, demand_file, max_distance_km=max_distance_km, seed=i,
                               time_range=time_range, export_fixed_route=export_fixed_route,
                               demand_factor=demand_factor,
                               save_complete=save_complete
                               )

    def return_lambda_(self, total_demand: float, x_flex: float, route_len: float, h: float) -> float:
        """
        Return the lambda parameter for the Poisson distribution from demand
        """
        return total_demand * (x_flex / route_len) * h

    def irvin_hall_cdf(self, n: int, x: float) -> float:
        """
        Compute the CDF of the Irvin-Hall distribution
        """
        floor_x = math.floor(x)  # Compute floor of x
        total_sum = 0  # Initialize sum

        for k in range(floor_x + 1):  # Sum from k = 0 to floor(x)
            # Calculate each term of the sum
            term = ((-1) ** k) * comb(n, k) * ((x - k) ** n)
            total_sum += term  # Add current term to the sum

        # Divide the sum by n! and return
        return total_sum / math.factorial(n)

    # Define the probability calculation function
    # def calculate_probability(t, lambda_, mu, sigma):
    def calculate_probability(self, t: float, lambda_: float, width: float, v_d=40 / 3.6, t_s=30) -> float:
        """
        Calculate the probability of detour time exceeding time t
        :param t: the time threshold
        :param lambda_: the Poisson parameter
        :param width: catchment area width (m)
        :param v_d: vehicle speed (m/s)
        :param t_s: dwell time (s)
        """
        total_probability = 0
        # Summing up to a large number, assuming Poisson probabilities are negligible beyond this
        max_n = 10 * lambda_ if lambda_ > 10 else 100
        for n in range(int(max_n) + 1):
            # Conditional probability P(T ≤ t | N = n)
            x = v_d / 2 / width * (t - n * t_s)

            if n == 0:
                cond_prob = 1
            else:
                cond_prob = self.irvin_hall_cdf(n, x)

            # Weight by the Poisson probability of n
            poisson_prob = poisson.pmf(n, lambda_)
            total_probability += cond_prob * poisson_prob
        return total_probability

    def interpolate_index(self, p_t_le_t_values: np.ndarray, target_value=0.95) -> int:
        # Ensure the array is a numpy array
        p_t_le_t_values = np.asarray(p_t_le_t_values)

        # Check if target_value is within the range of the array
        # if target_value < p_t_le_t_values[0] or target_value > p_t_le_t_values[-1]:
        if target_value < p_t_le_t_values[0]:
            return 0
        if target_value > p_t_le_t_values[-1]:
            # raise ValueError("Target value is out of the range of the array")
            print("Target value is out of the range of the array")
            return len(p_t_le_t_values) - 1

        # Find the indices of the two array elements between which target_value lies
        idx_below = np.searchsorted(p_t_le_t_values, target_value) - 1
        idx_above = idx_below + 1

        # Get the values at these indices
        x0, x1 = p_t_le_t_values[idx_below], p_t_le_t_values[idx_above]
        y0, y1 = idx_below, idx_above

        # Interpolate the index
        interpolated_index = y0 + (target_value - x0) * (y1 - y0) / (x1 - x0)

        return int(interpolated_index)

    def interpolate_value_at_index(self, t_values: np.ndarray, interpolated_index: int) -> float:
        # Ensure the array is a numpy array
        t_values = np.asarray(t_values)

        # Get the integer parts of the interpolated index
        idx_below = int(np.floor(interpolated_index))
        idx_above = int(np.ceil(interpolated_index))

        # If the interpolated index is an integer, return the corresponding value
        if idx_below == idx_above:
            return t_values[idx_below]

        # Get the values at these indices
        t0, t1 = t_values[idx_below], t_values[idx_above]

        # Interpolate the value
        interpolated_value = t0 + (interpolated_index - idx_below) * (t1 - t0) / (idx_above - idx_below)

        return interpolated_value

    def return_total_flex_time(self,
                               x_flex: float,
                               route_len: float,
                               h: float,
                               width: int,
                               v_d=30 / 3.6,
                               target_value=0.95,
                               t_values=np.linspace(0, 2000, 51),
                               max_t_c=None
                               ) -> float:
        """
        Return the total flex time (include detour and x-dir travel) for scheduling
        :param x_flex: flexible route portion (km)
        :param route_len: route length (km)
        :param h: headway (hour)
        :param width: catchment area width (m)
        :param v_d: vehicle speed (m/s)
        :param target_value: the target value for the probability
        :param t_values: the array of t to search
        :param max_t_c: maximum flexible route detour time
        :return: the total flex time for schedule
        """
        total_demand = self.hourly_demand

        assert total_demand > 0

        lambda_ = self.return_lambda_(total_demand, x_flex, route_len, h)
        p_t_le_t_values = np.array([self.calculate_probability(t, lambda_, width, v_d=v_d) for t in t_values])
        index_at_095 = self.interpolate_index(p_t_le_t_values, target_value=target_value)
        interpolated_value = self.interpolate_value_at_index(t_values, index_at_095)
        if max_t_c is not None:
            interpolated_value = min(interpolated_value, max_t_c)
        flex_route_fixed_time = x_flex * 1000 / v_d * 2

        return interpolated_value + flex_route_fixed_time

    def plot_route_with_nodes(self, html_name=None) -> go.Figure:
        """
        Plot stops and network nodes
        """

        # Creating a figure with Plotly Express
        fig = go.Figure()

        # Extract coordinates from LineString and convert to list
        line_x, line_y = map(list, self.alignment.xy)

        # Add the LineString as a layer
        fig.add_trace(go.Scattermapbox(
            lat=line_y,
            lon=line_x,
            mode='lines',
            line=dict(width=3, color='red'),
            name='Route'
        ))

        stop_coordinates = self.stop_coordinates
        # plot stop_coordinates[["stop_lon", "stop_lat"]]
        fig.add_trace(
            go.Scattermapbox(
                lat=stop_coordinates["stop_lat"],
                lon=stop_coordinates["stop_lon"],
                mode='markers',
                marker=dict(size=10, color='green'),
                name='Stops',
                # hovertext=stop_coordinates['stop_name']
                hovertext=stop_coordinates.index.astype(str) + "," +
                          stop_coordinates["stop_name"] + "," +
                          stop_coordinates["nearest_node"].astype(str)
            ),
        )

        network_nodes = self.network_nodes
        # plot network nodes
        fig.add_trace(go.Scattermapbox(
            lat=network_nodes.geometry.y,
            lon=network_nodes.geometry.x,
            mode='markers',
            marker=dict(size=3, color='blue'),
            name='Network Nodes',
            hovertext=network_nodes['node_index']
        ),
        )

        # Set up the map layout
        fig.update_layout(
            mapbox=dict(
                style='carto-positron',
                zoom=12,
                center=dict(lat=48.14, lon=11.58)
            ),
            showlegend=False,
            title=f'Route {self.route_no} with Network Nodes'
        )

        # Show the figure
        if html_name:
            fig.write_html(html_name)
        return fig

    def plot_route_with_demand(self, terminus_stop: str, time_range=None,
                               demand_factor=1.0, html_name=None) -> go.Figure:
        """
        Plot the route with the demand
        :param terminus_stop: the terminus stop name of the route
        :param time_range: the time range
        :param demand_factor: the demand factor
        :param html_name: the name of the HTML file to save
        """
        if self.route_with_coords_origin_gdf is None:
            self.output_demand(terminus_stop=terminus_stop, time_range=time_range,
                               demand_factor=demand_factor)
        fig = self.plot_route_with_nodes()

        route_with_coords_origin_gdf = self.route_with_coords_origin_gdf

        route_with_coords_origin_gdf["origin_text"] = ("Origin " + route_with_coords_origin_gdf["index"].astype(str) +
                                                       " -> " + route_with_coords_origin_gdf["origin_node"].astype(str))

        # Add origin and destination points to the same figure with labels
        fig.add_scattermapbox(lat=route_with_coords_origin_gdf['final_origin_lat'],
                              lon=route_with_coords_origin_gdf['final_origin_lon'],
                              marker=dict(size=3, color='black'),
                              hovertext=route_with_coords_origin_gdf["origin_text"]
                              )

        route_with_coords_origin_gdf["destination_text"] = "Destination " + route_with_coords_origin_gdf[
            "index"].astype(str) + " -> " + route_with_coords_origin_gdf["destination_node"].astype(str)
        # Add origin and destination points to the same figure with labels
        fig.add_scattermapbox(lat=route_with_coords_origin_gdf['final_destination_lat'],
                              lon=route_with_coords_origin_gdf['final_destination_lon'],
                              marker=dict(size=3, color='black'),
                              hovertext=route_with_coords_origin_gdf["destination_text"]
                              )

        # Show the figure
        if html_name:
            fig.write_html(html_name)
        return fig


if __name__ == "__main__":
    print(os.getcwd())
    data_p = r"C:\Users\ge37ser\Documents\Coding\FleetPy\data"
    demand_csv =  os.path.join(data_p, "demand", "SoD_demand", "sample.csv") #'data/demand/SoD_demand/sample.csv'
    GTFS_folder = os.path.join(data_p, "pubtrans", "MVG_GTFS") # "data/pubtrans/MVG_GTFS"
    network_path = os.path.join(data_p, "networks", "osm_route_MVG_road") # "data/networks/osm_route_MVG_road"

    route_no = 193
    to_trip_id = "134.T0.3-193-G-015-3.2.H" # "100.T2.3-193-G-013-1.4.H"
    shape_ids = ["3-193-G-015-3.2.H"] #["3-193-G-013-1.4.H"]
    terminus_stop = "Trudering Bf."

    start_time = 75600
    end_time = 86400

    n_seed = 2
    start_seed = 0

    # Generate demand
    pt_gen = PTScheduleGen(route_no, GTFS_folder, to_trip_id, network_path,
                           shape_ids=shape_ids)
    pt_gen.load_demand(demand_csv)

    output_demand_folder = os.path.join(data_p, "demand", f"route_{route_no}_demand", "matched", "osm_route_MVG_road") # f"data/demand/route_{route_no}_demand/matched/osm_route_MVG_road"
    if not os.path.exists(output_demand_folder):
        #os.mkdir(output_demand_folder)
        os.makedirs(output_demand_folder, exist_ok=True)

    hourly_demand = pt_gen.return_hourly_demand(time_range=(start_time, end_time))
    print(f"Route {route_no} hourly demand: {hourly_demand}")

    pt_gen.output_all_demand(n_seed, terminus_stop, output_demand_folder, time_range=(start_time, end_time),
                             start_i=start_seed, max_distance_km=0.5,
                             save_complete=True
                             )

    # Generate alignment, schedule
    scenario_name_base = "route_{}_flex_{:.1f}_demand_{}"

    h_sod = 5  # SoD headway (min)
    s_opt = 13  # fleet size (veh)
    t_c = 40  # cycle time (min)
    max_t_c = (s_opt * h_sod - t_c) * 60
    veh_size = 20  # veh size (passenger)

    pt_gen.load_demand(demand_csv)
    pt_gen.save_alignment_geojson(os.path.join(data_p, "pubtrans", f"route_{route_no}"))

    hourly_demand = pt_gen.return_hourly_demand(time_range=(start_time, end_time))
    print(f"Route {route_no} hourly demand: {hourly_demand}")

    terminus_demand_prop = pt_gen.return_terminus_demand_proportion(terminus_stop,
                                                                    time_range=(start_time, end_time))

    route_len = pt_gen.return_route_length()
    print(f"Route {route_no} length: {route_len}")

    pt_gen.output_station(os.path.join(data_p, "pubtrans", f"route_{route_no}"))

    # schedule is now standard instead of dependent on headway and n_veh
    schedule_file_name = f"{route_no}_schedules.csv"
    veh_type = f"veh_{veh_size}"
    pt_gen.output_schedule(os.path.join(data_p, "pubtrans", f"route_{route_no}"),
                           schedules_file=schedule_file_name, veh_type=veh_type)
    gtfs_name = f"route_{route_no}"
    demand_name = f"route_{route_no}_demand"

    pt_gen.plot_route_with_demand(
        terminus_stop,
        time_range=(start_time, end_time),
        html_name=os.path.join(data_p, "pubtrans", f"route_{route_no}", f"route_{route_no}_demand_sample.html")
    )

    terminus_id = pt_gen.return_terminus_id()

    x_flex = 1.0  # km
    total_flex_time = pt_gen.return_total_flex_time(
        x_flex=x_flex, route_len=route_len, h=h_sod / 60, width=500, target_value=0.95, max_t_c=max_t_c)

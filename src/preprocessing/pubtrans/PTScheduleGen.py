import numpy as np
import pandas as pd
import os
import shapely
from shapely.geometry import LineString
import geopandas as gpd
import pyproj
from shapely.ops import nearest_points
import networkx as nx


class PTScheduleGen:
    def __init__(self, route_no: int, demand_csv: str, GTFS_path: str, network_nodes_file: str,
                 to_trip_id: str, back_trip_id: str, nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                 shape_ids=None, dwell_time=30, ):
        """
        PT Route Schedule Generator
        :param route_no: the route number
        :param demand_csv: the path to the demand file
        :param GTFS_path: the path to the GTFS files
        :param network_nodes_file: the path to the network nodes file
        :param to_trip_id: the trip ID for the forward direction
        :param back_trip_id: the trip ID for the backward direction
        :param nodes_df: the nodes dataframe
        :param edges_df: the edges dataframe
        :param shape_ids: the shape IDs for the route
        :param dwell_time: the dwell time at each stop
        """
        print("Loading data for route {}".format(route_no))
        self.route_no = route_no
        demand_df = pd.read_csv(demand_csv)
        route_df = demand_df[demand_df['route_ID'] == route_no]

        print("Loading GTFS data for route {}".format(route_no))
        # Read GTFS
        routes = pd.read_csv(os.path.join(GTFS_path, 'routes.txt'))
        trips = pd.read_csv(os.path.join(GTFS_path, 'trips.txt'))
        shapes = pd.read_csv(os.path.join(GTFS_path, 'shapes.txt'))
        stops = pd.read_csv(os.path.join(GTFS_path, 'stops.txt'))
        stop_times = pd.read_csv(os.path.join(GTFS_path, 'stop_times.txt'))

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
            # points = shape_points[['shape_pt_lat', 'shape_pt_lon']].values
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

        network_nodes = gpd.read_file(network_nodes_file)
        self.network_nodes = network_nodes
        # set original crs of self.network_nodes to be self.network_crs
        self.network_nodes.crs = self.network_crs

        self.network_nodes_crs = self.network_nodes.to_crs(self.dest_crs)
        stop_coordinates = gpd.GeoDataFrame(stop_coordinates,
                                            geometry=gpd.points_from_xy(stop_coordinates['stop_lon'],
                                                                        stop_coordinates['stop_lat']))

        # Prepare columns for the nearest node and distance
        stop_coordinates['nearest_node'] = None
        stop_coordinates['distance'] = None

        # Iterate over each stop coordinate
        for index, stop in stop_coordinates.iterrows():
            # Find the nearest point and distance
            nearest_node, distance = self.find_nearest_and_distance(stop.geometry, self.network_nodes_crs)

            # Store the information in the stop_coordinates GeoDataFrame
            stop_coordinates.at[index, 'nearest_node'] = nearest_node['node_index']
            # stop_coordinates.at[index, 'distance'] = distance * 111139
            stop_coordinates.at[index, 'distance'] = distance

        self.stop_coordinates = stop_coordinates
        print("Creating schedule for route {}".format(route_no))

        route_stop_seq = stop_times[stop_times['trip_id'] == to_trip_id]["stop_id"].to_list()
        route_stop_seq.extend(stop_times[stop_times['trip_id'] == back_trip_id]["stop_id"].iloc[1:].to_list())

        # create a dataframe for the route stop sequence
        route_stop_seq_df = pd.DataFrame(route_stop_seq, columns=["stop_id"])
        # match with stop_coordinates["nearest_node"]
        route_stop_seq_df = route_stop_seq_df.merge(
            stop_coordinates.reset_index()[["index", "stop_id", "nearest_node"]], on="stop_id")

        print("Calculating travel time for route {}".format(route_no))
        # use edge file to estimate the travel time between stops
        # Create a NetworkX graph
        G = nx.Graph()  # Use nx.DiGraph() for a directed graph

        # Add nodes to the graph
        for idx, row in nodes_df.iterrows():
            # Assuming 'node_id' is the column for node identifiers
            G.add_node(row['node_index'], **row.to_dict())

        # Add edges to the graph
        for idx, row in edges_df.iterrows():
            # Assuming 'start_node' and 'end_node' columns define the edge
            # and 'weight' is an attribute of the edge
            G.add_edge(row['from_node'], row['to_node'], weight=row['travel_time'])

        # along route_stop_seq_df["nearest_node"], find the shortest path length between every two nodes
        for i in range(len(route_stop_seq_df) - 1):
            route_stop_seq_df.loc[i, "travel_time"] = nx.shortest_path_length(G,
                                                                              route_stop_seq_df.loc[i, "nearest_node"],
                                                                              route_stop_seq_df.loc[
                                                                                  i + 1, "nearest_node"],
                                                                              weight='weight')

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

        route_with_coords = route_df[['route_departure', 'origin_stop', 'destination_stop']].copy()
        # add the index of route_df as well
        route_with_coords["index"] = route_df.index
        print(route_with_coords.head(5))

        stop_coordinates_oneway = stop_coordinates.drop_duplicates(subset=["stop_name"])
        print(stop_coordinates_oneway.head(5))
        # Merge the dataframes to get coordinates for each stop
        route_with_coords = route_with_coords.merge(stop_coordinates_oneway[['stop_name', 'stop_lat', 'stop_lon']],
                                                    left_on='origin_stop', right_on='stop_name', how='left')
        route_with_coords.rename(columns={'stop_lat': 'origin_lat', 'stop_lon': 'origin_lon'}, inplace=True)
        route_with_coords = route_with_coords.merge(stop_coordinates_oneway[['stop_name', 'stop_lat', 'stop_lon']],
                                                    left_on='destination_stop', right_on='stop_name', how='left')
        route_with_coords.rename(columns={'stop_lat': 'destination_lat', 'stop_lon': 'destination_lon'}, inplace=True)
        self.route_with_coords: pd.DataFrame = route_with_coords
        print(route_with_coords.head(5))

        print("Initialization finished for route {}".format(route_no))

    def return_route_length(self):
        return self.route_length

    def return_demand_walking_dist(self, max_len, x):
        """
        Return the demand distribution based on walking distance
        """
        return (abs(max_len) - abs(x)) / abs(max_len)

    def output_demand(self, terminus_stop: str, demand_file: str, max_distance_km=0.5, seed=0, time_range=None,
                      export_fixed_route=False, demand_factor=1.0, save_complete=False, offset_time=1200,
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
        route_with_coords = self.route_with_coords.copy()

        # Set the seed
        np.random.seed(seed)

        # convert route_departure from time format to seconds
        for i in route_with_coords.index:
            sep_colon_part = route_with_coords.at[i, "route_departure"].split(":")
            if len(sep_colon_part[0]) > 2:
                # print(i)
                sep_colon_part = route_with_coords.at[i, "route_departure"].split(" ")[1].split(":")
            route_with_coords.at[i, "route_departure"] = int(sep_colon_part[0]) * 3600 + int(
                sep_colon_part[1]) * 60 + int(sep_colon_part[2])

        # filter by time_range
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
            # TODO: find rtree nearest node
            origin_point = shapely.Point(stop['final_origin_lon'], stop['final_origin_lat'])
            nearest_node, distance = self.find_nearest_and_distance(origin_point, self.network_nodes_crs)
            route_with_coords.at[index, 'origin_node'] = nearest_node['node_index']
            route_with_coords.at[index, 'origin_distance'] = distance

            destination_point = shapely.Point(stop['final_destination_lon'], stop['final_destination_lat'])
            nearest_node, distance = self.find_nearest_and_distance(destination_point, self.network_nodes_crs)
            route_with_coords.at[index, 'destination_node'] = nearest_node['node_index']
            route_with_coords.at[index, 'destination_distance'] = distance

        route_with_coords_origin_gdf = route_with_coords

        # randomly offset route_departure time
        route_with_coords_origin_gdf["route_departure"] = \
            route_with_coords_origin_gdf["route_departure"] + \
            np.random.randint(-offset_time, offset_time, route_with_coords_origin_gdf.shape[0])
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
        if not os.exists(output_path):
            os.makedirs(output_path)
        gdf = gpd.GeoDataFrame(geometry=[self.alignment])
        gdf.to_file(os.path.join(output_path, file_name), driver='GeoJSON')
        print("Alignment saved to {}".format(os.path.join(output_path, file_name)))

    def find_nearest_and_distance(self, input_point: shapely.Point, gdf, input_crs="EPSG:4326"):
        """
        Find the nearest node and distance
        """
        # Define the input CRS and the target CRS from the GeoDataFrame
        # WGS84 Latitude/Longitude
        target_crs = gdf.crs  # CRS of the GeoDataFrame

        input_lon, input_lat = input_point.x, input_point.y

        # Create a transformer function to convert from input_crs to target_crs
        transformer = pyproj.Transformer.from_crs(input_crs, target_crs, always_xy=True)
        transformed_point = transformer.transform(input_lon, input_lat)
        input_point = shapely.Point(transformed_point)

        # Find the nearest point
        nearest_geom = nearest_points(input_point, gdf.unary_union)[1]
        nearest_point = gdf[gdf.geometry == nearest_geom].iloc[0]

        # Calculate distance in the transformed CRS
        distance = input_point.distance(nearest_geom)

        return nearest_point, distance

    def output_station(self, output_path, station_file="stations.csv"):
        """
        Output the station file
        :param output_path: the path to save the station file
        :param station_file: the file name of the station file
        """
        output_schedule = self.stop_coordinates.nearest_node.reset_index()
        output_schedule.columns = ['station_id', 'network_node_index']
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
                        # headway=10 / 60, n_veh=5
                        ) -> None:
        """
        Output the schedule file
        :param output_path: the path to save the schedule file
        :param schedules_file: the file name of the schedule file
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
        schedules_df["vehicle_type"] = "default_vehtype"

        schedules_df.to_csv(os.path.join(output_path, schedules_file), index=False)
        print("Schedule file saved to {}".format(os.path.join(output_path, schedules_file)))

    # Function to generate random offset
    def random_offset(self, max_distance_km=1.) -> tuple:
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

# demand_csv = '../../../data/demand/SoD_demand/raw_data/demand_full_seed_0.csv'
# to_trip_id = "100.T2.3-193-G-013-1.4.H"
# back_trip_id = "4.T0.3-193-G-013-1.2.R"
# nodes_df = pd.read_csv("../../../data/networks/osm_route_193_road/base/nodes.csv")
# edges_df = pd.read_csv("../../../data/networks/osm_route_193_road/base/edges.csv")
# pt_gen = PTScheduleGen(193, demand_csv, "../../../data/pubtrans/MVG_GTFS",
#                        "../../../data/networks/osm_route_193_road/base/nodes_all_infos.geojson",
#                        to_trip_id, back_trip_id, nodes_df, edges_df,
#                        shape_ids=["3-193-G-013-1.4.H"])
# pt_gen.save_alignment_geojson("../../../data/pubtrans/route_193", )
# pt_gen.output_station("../../../data/pubtrans/route_193", )
# pt_gen.output_schedule("../../../data/pubtrans/route_193", )
#
# terminus_stop = "Trudering Bf."
# demand_file = "../../../data/pubtrans/route_193/passenger_demand.csv"
# pt_gen.output_demand(terminus_stop, demand_file, max_distance_km=0.667, )

import os
import shutil
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

# define function for nearest nodes of start and stop points
def ckdnearest(gdA, gdB):

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ],
        axis=1)

    return gdf

n_requests = 1500 #for density level one
n_seconds = 28800
num_rep = 30
num_density = 3


# demand file paths
demand_file = "data/demand/input_mito/trips_short.csv"
test_file = "data/demand/input_mito/test.csv"
nodes = "data/networks/munich_city_network_ID3_ref_vehtype/base/nodes.csv"

# load munich polygon
munich_gdf = gpd.read_file("data/networks/munich_network/munich-polygon.geojson")
nodes_df = pd.read_csv(nodes)

input_df = pd.read_csv(demand_file)

# Create new Dateframe with only essential information
demand_df = pd.DataFrame()
demand_df["origin_x"] = input_df["originX"]
demand_df["origin_y"] = input_df["originY"]
demand_df["destination_x"] = input_df["destinationX"]
demand_df["destination_y"] = input_df["destinationY"]
demand_df["distance"] = input_df["distance"]
demand_df["time"] = input_df["time_auto"]
demand_df["mode"] = input_df["mode"]
demand_df["trip_id"] = input_df["id"]

# Filter Auto trips
auto_demand_df = demand_df.loc[demand_df["mode"] == "autoDriver"]
auto_demand_df = auto_demand_df.drop(columns="mode")

# Create geodataframe
auto_demand_gdf_start = gpd.GeoDataFrame(data=auto_demand_df.loc[:,["trip_id","distance","time"]], geometry= gpd.points_from_xy(auto_demand_df["origin_x"], auto_demand_df["origin_y"]), crs=31468)
auto_demand_gdf_stop = gpd.GeoDataFrame(data=auto_demand_df["trip_id"], geometry=gpd.points_from_xy(auto_demand_df["destination_x"], auto_demand_df["destination_y"]), crs=31468)

# Convert points from   to EPSG:4326
auto_demand_gdf_start = auto_demand_gdf_start.to_crs(4326)
auto_demand_gdf_stop = auto_demand_gdf_stop.to_crs(4326)

# check if points are in Munich
auto_demand_gdf_start["start_in_polygon"] = auto_demand_gdf_start["geometry"].within(munich_gdf.loc[0,"geometry"])
auto_demand_gdf_stop["stop_in_polygon"] = auto_demand_gdf_stop["geometry"].within(munich_gdf.loc[0,"geometry"])

# keep only points that are inside munich polygon
auto_demand_gdf_start = auto_demand_gdf_start[auto_demand_gdf_start["start_in_polygon"] == True]
auto_demand_gdf_stop = auto_demand_gdf_stop[auto_demand_gdf_stop["stop_in_polygon"] == True]

# print(auto_demand_gdf_start)

# Create nodes gdf
munich_gdf_nodes = gpd.GeoDataFrame(index=nodes_df.index, data=nodes_df.loc[:,["node_index", "is_stop_only"]], geometry=gpd.points_from_xy(nodes_df.pos_x, nodes_df.pos_y))

# Check for nearest nodes
start_nodes = ckdnearest(auto_demand_gdf_start, munich_gdf_nodes)
stop_nodes = ckdnearest(auto_demand_gdf_stop, munich_gdf_nodes)

start_nodes = start_nodes.drop(columns={"geometry", "start_in_polygon", "is_stop_only", "dist"})
stop_nodes = stop_nodes.drop(columns={"geometry", "stop_in_polygon", "is_stop_only", "dist"})
start_nodes = start_nodes.rename(columns={'node_index':'start'})
stop_nodes = stop_nodes.rename(columns={'node_index':'end'})

# Merge start and stop
OD_nodes_df = pd.merge(start_nodes, stop_nodes, on='trip_id')
# OD_nodes_df = OD_nodes_df.drop(columns="trip_id")

# print(OD_nodes_df)

distances = []
times = []

for n in range(0, num_rep):

    distance = OD_nodes_df.iloc[(n*1500):(3000+n*1500),1].sum()
    time = OD_nodes_df.iloc[(n*1500):(3000+n*1500),2].mean()
    distances.append(distance)
    times.append(time)

    # for k in range(1, num_density+1):
    #
    #     csv_path = f'data/demand/munich_rand_demand/matched/munich_city_network/munich_demand_denselevel{k}_rep{n+1}.csv'
    #     OD_nodes_df_temp = OD_nodes_df.iloc[n*n_requests:(n+k)*n_requests]
    #     requests = np.random.randint(0, n_seconds, k*n_requests)
    #     requests_sort = np.sort(requests)
    #     OD_nodes_df_temp.insert(0, "rq_time" ,requests_sort)
    #     OD_nodes_df_temp.insert(3, "request_id", range(1, 1 + len(OD_nodes_df_temp)))
    #     OD_nodes_df_temp.to_csv(csv_path, index=False)


print(np.mean(distances))
print(np.mean(times))


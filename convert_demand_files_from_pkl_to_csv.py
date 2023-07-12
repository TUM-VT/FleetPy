import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from datetime import datetime
from matplotlib import pyplot as plt
from shapely.geometry import Point

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

# demand file paths
start = "data/demand/munich_demand/matched/munich_network/gdf_pickups_01-01-2019_01-02-2019.pkl"
stop = "data/demand/munich_demand/matched/munich_network/gdf_dropoffs_01-01-2019_01-02-2019.pkl"
nodes = "data/networks/munich_city_network_ID3_ref_vehtype/base/nodes.csv"

# load them to pandas dataframes
munich_df_stop_pickle = pd.read_pickle(stop, compression='gzip')
munich_df_start_pickle = pd.read_pickle(start, compression='gzip')
nodes_df = pd.read_csv(nodes)

# load munich polygon
munich_gdf = gpd.read_file("data/networks/munich_network/munich-polygon.geojson")

#number of replications
num_rep = 30
num_density = 3

for n in range(1, num_rep+1):

    for k in range(1, num_density+1):

        csv_path = f'data/demand/munich_demand/matched/munich_city_network/munich_demand_denselevel{k}_rep{n}.csv'
        start_time = 6
        end_time = k*8+6
        extra_day = 0
        if k == 3:
            end_time = 0
            start_time = 0
            extra_day = 1
        end_date = datetime.strptime("2019" + "-" + f'{n + 1 + extra_day}' + "-" + f'{end_time}', "%Y-%j-%H").strftime("%Y/%m/%d %H:%M:%S")
        start_date = datetime.strptime("2019" + "-" + f'{n + 1}' + "-" + f'{start_time}', "%Y-%j-%H").strftime("%Y/%m/%d %H:%M:%S")
        mask_start = munich_df_start_pickle.loc[munich_df_start_pickle["timestamp_start"] < end_date]
        munich_df_start = mask_start.loc[mask_start["timestamp_start"] > start_date]
        munich_df_stop = munich_df_stop_pickle[munich_df_stop_pickle["track_id"].isin(munich_df_start["track_id"])]

        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', None)
        # munich_df_stop = munich_df_stop.head(27642)
        # munich_df_start = munich_df_start.head(27642)
        # print(munich_df_stop.head(27642))
        # print(munich_df_start.head(27642))

        # print(munich_df_stop)

        # convert them to geopandas geodataframes
        munich_gdf_stop = gpd.GeoDataFrame(index=munich_df_stop.index, data=munich_df_stop.loc[:,["track_id", "timestamp_stop", "h3_dropoff"]], geometry=munich_df_stop.location_stop)
        munich_gdf_start = gpd.GeoDataFrame(index=munich_df_start.index, data=munich_df_start.loc[:,["track_id", "timestamp_start", "h3_pickup"]], geometry=munich_df_start.location_start)
        munich_gdf_nodes = gpd.GeoDataFrame(index=nodes_df.index, data=nodes_df.loc[:,["node_index", "is_stop_only"]], geometry=gpd.points_from_xy(nodes_df.pos_x, nodes_df.pos_y))

        # run the function
        start_nodes = ckdnearest(munich_gdf_start, munich_gdf_nodes)
        stop_nodes = ckdnearest(munich_gdf_stop, munich_gdf_nodes)
        start_nodes = start_nodes.rename(columns={'geometry':'start_points'})
        stop_nodes = stop_nodes.rename(columns={'geometry': 'stop_points'})
        start_nodes["start_in_polygon"] = start_nodes["start_points"].within(munich_gdf.loc[0,"geometry"])
        stop_nodes["stop_in_polygon"] = stop_nodes["stop_points"].within(munich_gdf.loc[0,"geometry"])

        # keep only points that are inside munich polygon
        start_nodes = start_nodes[start_nodes["start_in_polygon"] == True]
        stop_nodes = stop_nodes[stop_nodes["stop_in_polygon"] == True]

        # create one geodataframe
        munich_demand_df = start_nodes.merge(stop_nodes, left_on='track_id', right_on='track_id')
        pd.set_option('display.max_columns', None)

        # munich_demand_gdf["timestamp_start"] = start_nodes["timestamp_start"]
        # munich_demand_gdf["timestamp_stop"] = stop_nodes["timestamp_stop"]
        # munich_demand_gdf["start_point"] = start_nodes["geometry"]
        # munich_demand_gdf["stop_point"] = stop_nodes["geometry"]
        # munich_demand_gdf["start_node"] = start_nodes["node_index"]
        # munich_demand_gdf["stop_node"] = stop_nodes["node_index"]
        # munich_demand_gdf["start_distance_to_node"] = start_nodes["dist"]
        # munich_demand_gdf["stop_distance_to_node"] = stop_nodes["dist"]

        # convert timestamp to seconds
        since = datetime.strptime(start_date, "%Y/%m/%d %H:%M:%S")
        munich_demand_df["seconds"] = (munich_demand_df["timestamp_start"]-since).dt.total_seconds()

        # create dataframe for csv convertion
        demand_df = pd.DataFrame()
        demand_df["rq_time"] = munich_demand_df["seconds"].astype(int)
        demand_df["start"] = munich_demand_df["node_index_x"]
        demand_df["end"] = munich_demand_df["node_index_y"]

        demand_df["rq_time"] = np.where(demand_df["rq_time"] > 57600, demand_df["rq_time"] - 28800, demand_df["rq_time"])
        demand_df["rq_time"] = np.where(demand_df["rq_time"] > 28800, demand_df["rq_time"] - 28800, demand_df["rq_time"])

        demand_df = demand_df.sort_values(by=["rq_time"])
        demand_df.insert(3, "request_id", range(1, 1 + len(demand_df)))

        demand_df.to_csv(csv_path, index=False)

        # pd.set_option('display.max_columns', None)
        # print(munich_demand_gdf.head(10))
        # print(demand_df.head(10))

        # fig, ax = plt.subplots()
        # munich_gdf.plot(ax=ax, color='blue')
        # munich_demand_insider["stop_point"].plot(ax=ax, color='red')
        # munich_demand_insider["start_point"].plot(ax=ax, color='red')
        # plt.show()

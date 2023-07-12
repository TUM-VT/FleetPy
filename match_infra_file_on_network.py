import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from datetime import datetime
from matplotlib import pyplot as plt
from shapely.geometry import Point

# infra file paths
infra_path = "data/infra/munich_infra/input/charging_stations_epsg_32632.geojson"
nodes = "data/networks/munich_city_network_ID3_ref_vehtype/base/nodes.csv"
csv_path = "data/infra/munich_infra/munich_city_network/munich_infra_stations.csv"

# load files
nodes_df = pd.read_csv(nodes)
infra_gdf = gpd.read_file(infra_path)

# convert df to gdf
munich_gdf_nodes = gpd.GeoDataFrame(index=nodes_df.index, data=nodes_df.loc[:,["node_index", "is_stop_only"]], geometry=gpd.points_from_xy(nodes_df.pos_x, nodes_df.pos_y))

# convert infra from epsg 32632 to WSG84
munich_gdf_infra = infra_gdf.to_crs(epsg=4326)

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

# run the function
munich_infra_nodes_gdf = ckdnearest(munich_gdf_infra, munich_gdf_nodes)

# load munich polygon
munich_gdf = gpd.read_file("data/networks/munich_network/munich-polygon.geojson")

# keep only points that are inside munich polygon
munich_infra_nodes_gdf["point_in_polygon"] = munich_infra_nodes_gdf["geometry"].within(munich_gdf.loc[0,"geometry"])
munich_infra_nodes_gdf = munich_infra_nodes_gdf[munich_infra_nodes_gdf["point_in_polygon"] == True]

# create dataframe for csv convertion
infra_df = pd.DataFrame()
infra_df["charging_units"] = munich_infra_nodes_gdf["charging_units"]
infra_df["node_index"] = munich_infra_nodes_gdf["node_index"].astype(int)
infra_df.insert(0, "charging_station_id", range(1, 1 + len(infra_df)))
infra_df.to_csv(csv_path, index=False)


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
print(munich_infra_nodes_gdf)
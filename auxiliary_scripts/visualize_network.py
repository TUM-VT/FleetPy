import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

# create csv files and folder for external cost calculations of networks

vehicle1 = pd.read_csv(os.path.join("data", "vehicles","ID3_ref_vehtype.csv"), index_col=0)
vehicle2 = pd.read_csv(os.path.join("data", "vehicles","e-smart_vehtype.csv"), index_col=0)
# vehicle3 = pd.read_csv(os.path.join("data", "vehicles","EQC_vehtype.csv"), index_col=0)
vehicle4 = pd.read_csv(os.path.join("data", "vehicles","id_Buzz_vehtype.csv"), index_col=0)
vehicle5 = pd.read_csv(os.path.join("data", "vehicles","eVito_vehtype.csv"), index_col=0)
# vehicle6 = pd.read_csv(os.path.join("data", "vehicles","eSprinter_small_vehtype.csv"), index_col=0)
# vehicle7 = pd.read_csv(os.path.join("data", "vehicles","eSprinter_large_vehtype.csv"), index_col=0)


vehicles_df = pd.concat([vehicle1, vehicle2, vehicle4, vehicle5], axis=1)

for n in vehicles_df.columns[0:]:

    source_path1 = os.path.join("data", "networks", f'munich_city_network_{n}', "base", "edges_all_infos.geojson")
    source_path2 = os.path.join("data", "networks", f'munich_city_network_{n}', "base", "edges.csv")
    vehicle_gdf = gpd.read_file(source_path1)
    edges_df = pd.read_csv(source_path2)

    print(min(edges_df['external_costs']))
    print(max(edges_df['external_costs']))

    vehicle_gdf['external_costs'] = edges_df['external_costs']
    vehicle_gdf['ext_per_km'] = vehicle_gdf['external_costs'] / vehicle_gdf['distance'].apply(lambda x: x/1000)
    vehicle_gdf['speed'] = vehicle_gdf['distance'] / vehicle_gdf['travel_time']

    vehicle_gdf.plot(column='ext_per_km', legend=True, vmin=10, vmax=25, cmap='RdYlGn_r')
    # vehicle_gdf.plot(column='ext_per_km', legend=True)
    plt.title(f'External costs per km in €-ct for {n}')
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.show()



#plot Munich network

# source_path1 = os.path.join("data", "networks", "munich_city_network", "base", "edges_all_infos.geojson")
# source_path2 = os.path.join("data", "networks", "munich_city_network", "base", "edges.csv")
# munich_gdf = gpd.read_file(source_path1)
# # edges_munich_df = pd.read_csv(source_path2)
#
# # munich_gdf['external_costs'] = edges_munich_df['external_costs']
# # munich_gdf['ext_per_km'] = munich_gdf['external_costs'] / munich_gdf['distance'].apply(lambda x: x/1000)
# munich_gdf['speed'] = munich_gdf['distance'] / munich_gdf['travel_time'] * 3.6
#
#
# munich_gdf.plot(column='speed', legend=True, cmap='RdYlGn_r')
# # vehicle_gdf.plot(column='ext_per_km', legend=True)
# plt.title(f'Speed')
# plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
# plt.show()


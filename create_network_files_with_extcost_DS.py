import os
import shutil
import pandas as pd
import numpy as np
import geopandas as gpd

# create csv files and folder for external cost calculations of networks

vehicle1 = pd.read_csv(os.path.join("data", "vehicles","ID3_ref_vehtype.csv"), index_col=0)
vehicle2 = pd.read_csv(os.path.join("data", "vehicles","e-smart_vehtype.csv"), index_col=0)
vehicle3 = pd.read_csv(os.path.join("data", "vehicles","EQC_vehtype.csv"), index_col=0)
vehicle4 = pd.read_csv(os.path.join("data", "vehicles","id_Buzz_vehtype.csv"), index_col=0)
vehicle5 = pd.read_csv(os.path.join("data", "vehicles","eVito_vehtype.csv"), index_col=0)
vehicle6 = pd.read_csv(os.path.join("data", "vehicles","eSprinter_small_vehtype.csv"), index_col=0)
vehicle7 = pd.read_csv(os.path.join("data", "vehicles","eSprinter_large_vehtype.csv"), index_col=0)


vehicles_df = pd.concat([vehicle1, vehicle2, vehicle3, vehicle4, vehicle5, vehicle6, vehicle7], axis=1)

eff = 0.85
f_r = 0.014
g = 9.81 #m/s^2
rho = 1.225 #kg/m^3
airpol_per_kWh = 1.29 #ct/kWh for Germany power mix
abrasion_costs = 0.11 #ct/vkm
climate_per_kWh = 9.19 #ct/kWh for Germany power mix
noise_ref = 0.29 #ct/vkm
live_streets = ["residential", "living_street", "pedestrian"]
p_live = 1.2
p_other = 0.8
noise_money_ref = 220.13 #€
land_use_ref = 1.68 #ct/vkm
A_base_ref = 7.71 #m^2
congestion_ref = 5.48 #ct/vkm
l_ref = 4.26 #m
accident_ref = 3.82 #ct/vkm
m_ref = 2270 #kg
p_fat_ref = 0.28 #%
p_ser_ref = 10.33 #%
p_min_ref = 89.39 #%
barrier_costs_rel = 2.27 #ct/vkm

for n in vehicles_df.columns[0:]:


    #create network files

    target_path_name = os.path.join("data", "networks", f'example_network_{n}', "base")
    os.makedirs(target_path_name)
    source_path_name1 = os.path.join("data", "networks", "example_network", "base", "crs.info")
    shutil.copy(source_path_name1, target_path_name)

    source_path_name2 = os.path.join("data", "networks", "example_network", "base", "edges.csv")
    shutil.copy(source_path_name2, target_path_name)

    source_path_name3 = os.path.join("data", "networks", "example_network", "base", "edges_all_infos.geojson")
    shutil.copy(source_path_name3, target_path_name)

    source_path_name4 = os.path.join("data", "networks", "example_network", "base", "nodes.csv")
    shutil.copy(source_path_name4, target_path_name)

    source_path_name5 = os.path.join("data", "networks", "example_network", "base", "nodes_all_infos.geojson")
    shutil.copy(source_path_name5, target_path_name)

    change_path = os.path.join("data", "networks", f'example_network_{n}', "base", "edges.csv")

    edges_df = pd.read_csv(change_path)

    edges_gdf = gpd.read_file(source_path_name3)

    tot_cost_array = []
    air_pol_array = []
    climate_array = []
    noise_array = []
    land_use_array = []
    congestion_array = []
    accident_array = []
    barrier_array = []

    for index, row in edges_df.iterrows():

        pc = (1/36) * (1/eff) * (f_r * g * vehicles_df.loc['vehicle_mass [kg]', n] + vehicles_df.loc['c_w', n] * vehicles_df.loc['A_f [m^2]', n] * 0.5 * rho * np.square(row["distance"]/row["travel_time"])) #kWh/100km
        air_pol_costs = pc * 0.01 * airpol_per_kWh * 0.001 * row["distance"] + abrasion_costs * 0.001 * row["distance"]
        climate_costs = pc * 0.01 * climate_per_kWh * 0.001 * row["distance"]
        noise_v = 29.631 * np.power((row["distance"] / row["travel_time"]) * 3.6, 0.2204)
        money_noise = 0.6418 * np.square(noise_v) - 56.738 * noise_v + 1289.3

        if live_streets.count(edges_gdf.loc[index, "road_type"]) > 0:

            noise_costs = noise_ref * p_live * (money_noise / noise_money_ref) * 0.001 * row["distance"]

        else:

            noise_costs = noise_ref * p_other * (money_noise / noise_money_ref) * 0.001 * row["distance"]

        land_use_costs = land_use_ref * (vehicles_df.loc['A_base [m^2]', n] / A_base_ref) * 0.001 * row["distance"]
        congestion_costs = congestion_ref * (vehicles_df.loc['l [m]', n] / l_ref) * 0.001 * row["distance"]

        p_fat_v = 0.0011 * np.square(row["distance"]/row["travel_time"]*3.6) - 0.0392 * (row["distance"]/row["travel_time"]*3.6) + 0.1852
        p_ser_v = 0.1 * (row["distance"] / row["travel_time"]*3.6) + 5.5
        p_min_v = -0.0011 * np.square(row["distance"] / row["travel_time"] * 3.6) - 0.0608 * (row["distance"] / row["travel_time"] * 3.6) + 94.315

        if live_streets.count(edges_gdf.loc[index, "road_type"]) > 0:

            accident_costs = accident_ref * p_live * (vehicles_df.loc['vehicle_mass [kg]', n] / m_ref)*(4400000*p_fat_v+620000*p_ser_v+40000*p_min_v)/(4400000*p_fat_ref+620000*p_ser_ref+40000*p_min_ref) * 0.001 * row["distance"]

        else:

            accident_costs = accident_ref * (vehicles_df.loc['vehicle_mass [kg]', n] / m_ref)*(4400000*p_fat_v+620000*p_ser_v+40000*p_min_v)/(4400000*p_fat_ref+620000*p_ser_ref+40000*p_min_ref) * 0.001 * row["distance"]

        barrier_costs = barrier_costs_rel * 0.001 * row["distance"]

        external_costs_tot = air_pol_costs + climate_costs + noise_costs + land_use_costs + congestion_costs + accident_costs + barrier_costs

        tot_cost_array.append(external_costs_tot)
        air_pol_array.append(air_pol_costs)
        climate_array.append(climate_costs)
        noise_array.append(noise_costs)
        land_use_array.append(land_use_costs)
        congestion_array.append(congestion_costs)
        accident_array.append(accident_costs)
        barrier_array.append(barrier_costs)

    print(tot_cost_array)

    edges_df['external_costs'] = tot_cost_array
    edges_df['air_pol_costs'] = air_pol_array
    edges_df['climate_costs'] = climate_array
    edges_df['noise_costs'] = noise_array
    edges_df['land_use_costs'] = land_use_array
    edges_df['congestion_costs'] = congestion_array
    edges_df['accident_costs'] = accident_array
    edges_df['barrier_costs'] = barrier_array

    edges_df.to_csv(change_path, index=False)



    # #create demand files
    #
    # target_path_name = os.path.join("data", "demand", "example_demand", "matched", f'example_network_v-{n}')
    # #os.makedirs(target_path_name)
    # source_path_name1 = os.path.join("data", "demand", "example_demand", "matched", "example_network", "example_100.csv")
    # shutil.copy(source_path_name1, target_path_name)

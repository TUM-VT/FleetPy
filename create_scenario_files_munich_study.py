import csv
import os
import shutil
import numpy as np
import pandas as pd

study_number = 4

#Request file, choose from ["munich_demand_one_day.csv", "munich_demand_twoinone_day.csv", "munich_demand_fourinone_day.csv", "munich_demand_weekinone_day.csv"]
rq_names = []
reps = 30
dens = 3

for a in range(1, reps+1):
    for b in range(1, dens +1):
        rq_names = np.append(rq_names, f'munich_demand_denselevel{b}_rep{a}.csv')

#Fleet composition: vehicle choose from ["e-smart_vehtype", "ID3_ref_vehtype", "id_Buzz_vehtype", "eVito_vehtype"]; Number of vehicles choose from ["50", "100", "150", "200"]

vehtype_names = ["e-smart_vehtype", "ID3_ref_vehtype", "id_Buzz_vehtype", "eVito_vehtype"]
vehicle_nummbers = ["50", "100", "150", "200"]

#Routing and Pooling type: choose from ["woext", "wext"]

rootpool_types = ["woext", "wext"]

#define infos for config file
config_name = f'munich_study_{len(rq_names)}rqfiles_{len(vehtype_names)}vehtypes_{len(vehicle_nummbers)}num_{len(rootpool_types)}rootpool_study{study_number}.csv'
save_path = os.path.join("studies", "munich_study", "scenarios", config_name)

num_scenarios = len(rq_names)*len(vehtype_names)*len(vehicle_nummbers)*len(rootpool_types)
print(f'Your study has {num_scenarios} scenarios')
print(config_name)

with open(save_path, 'a', newline='') as csvfile:
    fieldnames = ['op_module', 'scenario_name',	'rq_file', 'op_fleet_composition', 'network_type', 'network_name', 'op_vr_control_func_dict', 'demand_name', 'op_charging_method', 'op_min_soc_after_planstop', 'infra_structure_name', 'ch_op_public_charging_station_f', 'nr_charging_operators']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

for n in range(0, len(rq_names)):
    for k in range(0, len(vehtype_names)):
        for l in range(0, len(vehicle_nummbers)):
            for m in range(0, len(rootpool_types)):

                with open(save_path, 'a', newline='') as csvfile:
                    fieldnames = ['op_module', 'scenario_name',	'rq_file', 'op_fleet_composition', 'network_type', 'network_name', 'op_vr_control_func_dict', 'demand_name', 'op_charging_method', 'op_min_soc_after_planstop', 'infra_structure_name', 'ch_op_public_charging_station_f', 'nr_charging_operators']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    scenario_name = f'munich_study_{study_number}_rq{n+1}_{vehtype_names[k]}_{vehicle_nummbers[l]}_{rootpool_types[m]}'
                    demand_file = rq_names[n]
                    fleet_composition = f'{vehtype_names[k]}:{vehicle_nummbers[l]}'
                    if rootpool_types[m] == 'wext':
                        network_type = "NetworkBasic_Ext_Routing"
                    else:
                        network_type = "NetworkBasic_Ext"
                    network_name = f'munich_city_network_{vehtype_names[k]}'
                    if rootpool_types[m] == 'wext':
                        pooling = "func_key:total_customized_travel_costs;vot:0.45"
                    else:
                        pooling = "func_key:distance_and_user_times_with_walk;vot:0.45"

                    writer.writerow({'op_module': 'PoolingIRSOnly', 'scenario_name': scenario_name ,'rq_file': demand_file, 'op_fleet_composition': fleet_composition, 'network_type': network_type, 'network_name': network_name, 'op_vr_control_func_dict': pooling, 'demand_name': "munich_rand_demand", 'op_charging_method': "Threshold_PCI", 'op_min_soc_after_planstop': "0.2", 'infra_structure_name': "munich_infra", 'ch_op_public_charging_station_f': "public_charging_stations.csv", 'nr_charging_operators': "1"})





config_file = pd.read_csv(save_path)

number_rows = len(config_file.index)
split_number = 3

config_split1 = pd.DataFrame()
config_split2 = pd.DataFrame()
config_split3 = pd.DataFrame()


config_split1 = config_file.iloc[0:int(number_rows/split_number-1)]
config_split2 = config_file.iloc[int(number_rows/split_number):int(number_rows/split_number*2-1)]
config_split3 = config_file.iloc[int(number_rows/split_number*2):int(number_rows/split_number*3-1)]

config_name1 = f'munich_study_{len(rq_names)}rqfiles_{len(vehtype_names)}vehtypes_{len(vehicle_nummbers)}num_{len(rootpool_types)}rootpool_study{study_number}_1.csv'
config_name2 = f'munich_study_{len(rq_names)}rqfiles_{len(vehtype_names)}vehtypes_{len(vehicle_nummbers)}num_{len(rootpool_types)}rootpool_study{study_number}_2.csv'
config_name3 =f'munich_study_{len(rq_names)}rqfiles_{len(vehtype_names)}vehtypes_{len(vehicle_nummbers)}num_{len(rootpool_types)}rootpool_study{study_number}_3.csv'

save_path1 = os.path.join("studies", "munich_study", "scenarios", config_name1)
save_path2 = os.path.join("studies", "munich_study", "scenarios", config_name2)
save_path3 = os.path.join("studies", "munich_study", "scenarios", config_name3)

config_split1.to_csv(save_path1, index=False)
config_split2.to_csv(save_path2, index=False)
config_split3.to_csv(save_path3, index=False)


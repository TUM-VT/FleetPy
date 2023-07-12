import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# study number
study_num = 4
reps = 30
dens = 3

vehtype_names = ["e-smart_vehtype", "ID3_ref_vehtype", "id_Buzz_vehtype", "eVito_vehtype"]
vehicle_nummbers = ["50", "100", "150", "200"]
rootpool_types = ["woext", "wext"]

# get result file names
result_path = os.path.join("studies", "munich_study", "results", f'munich_study_{study_num}')
files = os.listdir(result_path)
files_info = []
for f in files:
    files_info = np.append(files_info, f.replace(f'munich_study_{study_num}_', ''))
file_num = len(files_info)

print(f'{file_num} result files are evaluated')

# load standard eval as DataFrame

standard_eval_df = pd.read_csv(os.path.join(result_path, files[0], 'standard_eval.csv'), usecols=[0], index_col=0)
low_demand_df = pd.read_csv(os.path.join(result_path, files[0], 'standard_eval.csv'), usecols=[0], index_col=0)
mid_demand_df = pd.read_csv(os.path.join(result_path, files[0], 'standard_eval.csv'), usecols=[0], index_col=0)
high_demand_df = pd.read_csv(os.path.join(result_path, files[0], 'standard_eval.csv'), usecols=[0], index_col=0)

demand_num_df = pd.DataFrame(columns=["low_demand", "mid_demand", "high_demand"])

count = 1

for b in range(1, reps+1):

    demand_num_df = demand_num_df.append({"low_demand":count, "mid_demand":count + 1, "high_demand":count + 2}, ignore_index=True)
    count += 3


for n in files_info:

    if os.path.exists(os.path.join(result_path, f'munich_study_{study_num}_' + n, 'standard_eval.csv')):

        temp_df = pd.read_csv(os.path.join(result_path, f'munich_study_{study_num}_' + n, 'standard_eval.csv'), index_col=0)
        temp_df = temp_df.rename(columns={'MoD_0': n})

        if int(re.findall(r'\d+', n)[0]) in demand_num_df["low_demand"].values:
            low_demand_df[n] = temp_df[n]

        if int(re.findall(r'\d+', n)[0]) in demand_num_df["mid_demand"].values:
            mid_demand_df[n] = temp_df[n]

        if int(re.findall(r'\d+', n)[0]) in demand_num_df["high_demand"].values:
            high_demand_df[n] = temp_df[n]

        standard_eval_df[n] = temp_df[n]


low_demand_df_mean = pd.DataFrame()
mid_demand_df_mean = pd.DataFrame()
high_demand_df_mean = pd.DataFrame()
vehicle_names_df = pd.DataFrame()
vehicle_numbers_df = pd.DataFrame()
rootpool_types_df = pd.DataFrame()
capacity_df = pd.DataFrame()
range_df = pd.DataFrame()

for k in range(0, len(vehtype_names)):
    for l in range(0, len(vehicle_nummbers)):
        for m in range(0, len(rootpool_types)):

            scenario_name = f'{vehtype_names[k]}_{vehicle_nummbers[l]}_{rootpool_types[m]}'

            if not low_demand_df.filter(like=scenario_name).empty:
                low_demand_df_mean[scenario_name] = low_demand_df.filter(like=scenario_name).mean(axis=1)

            if not mid_demand_df.filter(like=scenario_name).empty:
                mid_demand_df_mean[scenario_name] = mid_demand_df.filter(like=scenario_name).mean(axis=1)

            if not high_demand_df.filter(like=scenario_name).empty:
                high_demand_df_mean[scenario_name] = high_demand_df.filter(like=scenario_name).mean(axis=1)

            vehicle_names_df.loc['vehtype', scenario_name] = vehtype_names[k]
            vehicle_numbers_df.loc['veh_num', scenario_name] = float(vehicle_nummbers[l])
            rootpool_types_df.loc['rootpool', scenario_name] = rootpool_types[m]
            if vehtype_names[k] == "e-smart_vehtype":
                capacity_df.loc['capacity', scenario_name] = 1
                range_df.loc['range', scenario_name] = 135
            if vehtype_names[k] == "ID3_ref_vehtype":
                capacity_df.loc['capacity', scenario_name] = 3
                range_df.loc['range', scenario_name] = 335
            if vehtype_names[k] == "id_Buzz_vehtype":
                capacity_df.loc['capacity', scenario_name] = 5
                range_df.loc['range', scenario_name] = 432
            if vehtype_names[k] == "eVito_vehtype":
                capacity_df.loc['capacity', scenario_name] = 6
                range_df.loc['range', scenario_name] = 240


low_demand_df_mean = pd.concat([low_demand_df_mean, vehicle_names_df, vehicle_numbers_df, rootpool_types_df, capacity_df, range_df])
mid_demand_df_mean = pd.concat([mid_demand_df_mean, vehicle_names_df, vehicle_numbers_df, rootpool_types_df, capacity_df, range_df])
high_demand_df_mean = pd.concat([high_demand_df_mean, vehicle_names_df, vehicle_numbers_df, rootpool_types_df, capacity_df, range_df])

# pd.set_option('display.max_columns', None)
# print(low_demand_df_mean)

# Create csv files
low_demand_df_mean.to_csv('studies/munich_study/results/munich_study_4/low_demand_mean_df.csv')
mid_demand_df_mean.to_csv('studies/munich_study/results/munich_study_4/mid_demand_mean_df.csv')
high_demand_df_mean.to_csv('studies/munich_study/results/munich_study_4/high_demand_mean_df.csv')

# Filter out serving rates below 80%
low_demand_df_mean = low_demand_df_mean.T[low_demand_df_mean.loc["served online users [%]"] > 80].T
mid_demand_df_mean = mid_demand_df_mean.T[mid_demand_df_mean.loc["served online users [%]"] > 80].T
high_demand_df_mean = high_demand_df_mean.T[high_demand_df_mean.loc["served online users [%]"] > 80].T


# Evaluation 3D-Graph with dimensions Service, Ext. Costs and Int. Costs

# low-demand plot
# axes instance
fig = plt.figure ( figsize=(12, 6) )
ax = Axes3D ( fig, auto_add_to_figure=False )
fig.add_axes ( ax )

# find all the unique labels in the 'name' columndemand_num_df
labels = np.unique(low_demand_df_mean.loc['vehtype'].array)
# get palette from seaborn
palette = sns.color_palette ( "husl", len ( labels ) )

fleet_size = np.unique(low_demand_df_mean.loc['veh_num'].array)
markers = ["o", "v", "s", "x"]

z_min = min(np.add(low_demand_df_mean.loc["waiting time"].array, low_demand_df_mean.loc["travel time"].array) / 60)

# plot
for label, color in zip ( labels, palette ):
    df1 = low_demand_df_mean.loc[:, low_demand_df_mean.eq(label).any()]

    for size, marker in zip (fleet_size, markers):
        df2 = df1.loc[:, df1.eq(size).any()]
        x = df2.loc["total external costs"].array / 100
        y = np.add(df2.loc["mod fix costs"].array, df2.loc["mod var costs"].array) / 100
        z = np.add(df2.loc["waiting time"].array, df2.loc["travel time"].array) / 60
        ax.scatter ( x, y, z,
        s = 40, marker = marker, color = color, alpha = 1, label = str(size)+label)

        z2 = np.ones ( shape=x.shape[0] ) * z_min
        # z2 = np.zeros(shape=x.shape[0])
        for i, j, k, h in zip ( x, y, z, z2 ):
            ax.plot ( [i, i],  [j, j],  [k, h], color = color )


ax.set_xlabel('External Costs in [€]')
ax.set_ylabel('Internal Costs in [€]')
ax.set_zlabel('Service / Trip time in [min]')
# ax.set_xlim(0, 4000)
# ax.set_ylim(0, 10000)
# ax.set_zlim(0, 20)
ax.invert_xaxis()
ax.set_title('Low-Demand')

# legend
plt.legend ( bbox_to_anchor=(1.05, 1), loc=2 )
plt.show ()

# mid-demand plot
# axes instance
fig = plt.figure ( figsize=(12, 6) )
ax = Axes3D ( fig, auto_add_to_figure=False )
fig.add_axes ( ax )

# find all the unique labels in the 'name' columndemand_num_df
labels = np.unique(mid_demand_df_mean.loc['vehtype'].array)
# get palette from seaborn
palette = sns.color_palette ( "husl", len ( labels ) )

fleet_size = np.unique(mid_demand_df_mean.loc['veh_num'].array)
markers = ["o", "v", "s", "x"]

z_min = min(np.add(mid_demand_df_mean.loc["waiting time"].array, mid_demand_df_mean.loc["travel time"].array) / 60)

# plot
for label, color in zip ( labels, palette ):
    df1 = mid_demand_df_mean.loc[:, mid_demand_df_mean.eq(label).any()]
    for size, marker in zip (fleet_size, markers):
        df2 = df1.loc[:, df1.eq(size).any()]
        x = df2.loc["total external costs"].array / 100
        y = np.add(df2.loc["mod fix costs"].array, df2.loc["mod var costs"].array) / 100
        z = np.add(df2.loc["waiting time"].array, df2.loc["travel time"].array) / 60
        ax.scatter ( x, y, z,
        s = 40, marker = marker, color = color, alpha = 1, label = str(size)+label)

        z2 = np.ones ( shape=x.shape[0] ) *z_min
        # z2 = np.zeros ( shape=x.shape[0] )
        for i, j, k, h in zip ( x, y, z, z2 ):
            ax.plot ( [i, i],  [j, j],  [k, h], color = color )


ax.set_xlabel('External Costs in [€]')
ax.set_ylabel('Internal Costs in [€]')
ax.set_zlabel('Service / Trip time in [min]')
# ax.set_xlim(0, 4000)
# ax.set_ylim(0, 10000)
# ax.set_zlim(0, 20)
ax.invert_xaxis()
ax.set_title('Mid-Demand')

# legend
plt.legend ( bbox_to_anchor=(1.05, 1), loc=2 )
plt.show ()

# high-demand plot
# axes instance
fig = plt.figure ( figsize=(12, 6) )
ax = Axes3D ( fig, auto_add_to_figure=False )
fig.add_axes ( ax )

# find all the unique labels in the 'name' columndemand_num_df
labels = np.unique(high_demand_df_mean.loc['vehtype'].array)
# get palette from seaborn
palette = sns.color_palette ( "husl", len ( labels ) )

fleet_size = np.unique(high_demand_df_mean.loc['veh_num'].array)
markers = ["o", "v", "s", "x"]

z_min = min(np.add(high_demand_df_mean.loc["waiting time"].array, high_demand_df_mean.loc["travel time"].array) / 60)


# plot
for label, color in zip ( labels, palette ):
    df1 = high_demand_df_mean.loc[:, high_demand_df_mean.eq(label).any()]
    for size, marker in zip(fleet_size, markers):
        df2 = df1.loc[:, df1.eq(size).any()]
        x = df2.loc["total external costs"].array / 100
        y = np.add(df2.loc["mod fix costs"].array, df2.loc["mod var costs"].array) / 100
        z = np.add(df2.loc["waiting time"].array, df2.loc["travel time"].array) / 60
        ax.scatter ( x, y, z,
        s = 40, marker = marker, color = color, alpha = 1, label = str(size)+label)

        z2 = np.ones ( shape=x.shape[0] ) *z_min
        # z2 = np.zeros(shape=x.shape[0])
        for i, j, k, h in zip ( x, y, z, z2 ):
            ax.plot ( [i, i],  [j, j],  [k, h], color = color )


ax.set_xlabel('External Costs in [€]')
ax.set_ylabel('Internal Costs in [€]')
ax.set_zlabel('Service / Trip time in [min]')
# ax.set_xlim(0, 4000)
# ax.set_ylim(0, 10000)
# ax.set_zlim(0, 20)
ax.invert_xaxis()
ax.set_title('High-Demand')

# legend
plt.legend ( bbox_to_anchor=(1.05, 1), loc=2 )
plt.show ()

#
# # Evaluation  #vehicles vs. capacity
#
#external costs (low demand)
x = low_demand_df_mean.loc['capacity'].array
y = low_demand_df_mean.loc['veh_num'].array
z = low_demand_df_mean.loc['total external costs'].array
colors = low_demand_df_mean.loc['rootpool'].array

g = sns.scatterplot(x=x, y=y, size=z, hue=colors, sizes=(50, 5000), linewidth=0, alpha=0.5)
h,l = g.get_legend_handles_labels()

plt.legend(h[0:2],l[0:2],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Vehicle Capacity")
plt.ylabel("Vehicle Number")
plt.ylim(20,220)
plt.title("External Costs (Low Demand)")
plt.show()

#external costs (mid demand)
x = mid_demand_df_mean.loc['capacity'].array
y = mid_demand_df_mean.loc['veh_num'].array
z = mid_demand_df_mean.loc['total external costs'].array
colors = mid_demand_df_mean.loc['rootpool'].array

g = sns.scatterplot(x=x, y=y, size=z, hue=colors, sizes=(50, 5000), linewidth=0, alpha=0.5)
h,l = g.get_legend_handles_labels()

plt.legend(h[0:2],l[0:2],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Vehicle Capacity")
plt.ylabel("Vehicle Number")
plt.ylim(20,220)
plt.title("External Costs (Mid Demand)")
plt.show()

#external costs (high demand)
x = high_demand_df_mean.loc['capacity'].array
y = high_demand_df_mean.loc['veh_num'].array
z = high_demand_df_mean.loc['total external costs'].array
colors = high_demand_df_mean.loc['rootpool'].array

g = sns.scatterplot(x=x, y=y, size=z, hue=colors, sizes=(50, 5000), linewidth=0, alpha=0.5)
h,l = g.get_legend_handles_labels()

plt.legend(h[0:2],l[0:2],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Vehicle Capacity")
plt.ylabel("Vehicle Number")
plt.ylim(20,220)
plt.title("External Costs (High Demand)")
plt.show()

#service (low demand)
x = low_demand_df_mean.loc['capacity'].array
y = low_demand_df_mean.loc['veh_num'].array
z = np.add(low_demand_df_mean.loc["waiting time"].array, low_demand_df_mean.loc["travel time"].array) / 60
colors = low_demand_df_mean.loc['rootpool'].array

g = sns.scatterplot(x=x, y=y, size=z, hue=colors, sizes=(50, 5000), linewidth=0, alpha=0.5)
h,l = g.get_legend_handles_labels()

plt.legend(h[0:2],l[0:2],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Vehicle Capacity")
plt.ylabel("Vehicle Number")
plt.ylim(20,220)
plt.title("Service = Trip time (Low Demand)")
plt.show()
#
#service (mid demand)
x = mid_demand_df_mean.loc['capacity'].array
y = mid_demand_df_mean.loc['veh_num'].array
z = np.add(mid_demand_df_mean.loc["waiting time"].array, mid_demand_df_mean.loc["travel time"].array) / 60
colors = mid_demand_df_mean.loc['rootpool'].array

g = sns.scatterplot(x=x, y=y, size=z, hue=colors, sizes=(50, 5000), linewidth=0, alpha=0.5)
h,l = g.get_legend_handles_labels()

plt.legend(h[0:2],l[0:2],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Vehicle Capacity")
plt.ylabel("Vehicle Number")
plt.ylim(20,220)
plt.title("Service = Trip time (Mid Demand)")
plt.show()

#service (high demand)
x = high_demand_df_mean.loc['capacity'].array
y = high_demand_df_mean.loc['veh_num'].array
z = np.add(high_demand_df_mean.loc["waiting time"].array, high_demand_df_mean.loc["travel time"].array) / 60
colors = high_demand_df_mean.loc['rootpool'].array

g = sns.scatterplot(x=x, y=y, size=z, hue=colors, sizes=(50, 5000), linewidth=0, alpha=0.5)
h,l = g.get_legend_handles_labels()

plt.legend(h[0:2],l[0:2],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Vehicle Capacity")
plt.ylabel("Vehicle Number")
plt.ylim(20,220)
plt.title("Service = Trip time (High Demand)")
plt.show()

#Internal costs (low demand)
x = low_demand_df_mean.loc['capacity'].array
y = low_demand_df_mean.loc['veh_num'].array
z = np.add(low_demand_df_mean.loc["mod fix costs"].array, low_demand_df_mean.loc["mod var costs"].array) / 100
colors = low_demand_df_mean.loc['rootpool'].array

g = sns.scatterplot(x=x, y=y, size=z, hue=colors, sizes=(50, 5000), linewidth=0, alpha=0.5)
h,l = g.get_legend_handles_labels()

plt.legend(h[0:2],l[0:2],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Vehicle Capacity")
plt.ylabel("Vehicle Number")
plt.ylim(20,220)
plt.title("Internal Costs (Low Demand)")
plt.show()

#Internal costs (mid demand)
x = mid_demand_df_mean.loc['capacity'].array
y = mid_demand_df_mean.loc['veh_num'].array
z = np.add(mid_demand_df_mean.loc["mod fix costs"].array, mid_demand_df_mean.loc["mod var costs"].array) / 100
colors = mid_demand_df_mean.loc['rootpool'].array

g = sns.scatterplot(x=x, y=y, size=z, hue=colors, sizes=(50, 5000), linewidth=0, alpha=0.5)
h,l = g.get_legend_handles_labels()

plt.legend(h[0:2],l[0:2],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Vehicle Capacity")
plt.ylabel("Vehicle Number")
plt.ylim(20,220)
plt.title("Internal Costs (Mid Demand)")
plt.show()

#Internal costs (high demand)
x = high_demand_df_mean.loc['capacity'].array
y = high_demand_df_mean.loc['veh_num'].array
z = np.add(high_demand_df_mean.loc["mod fix costs"].array, high_demand_df_mean.loc["mod var costs"].array) / 100
colors = high_demand_df_mean.loc['rootpool'].array

g = sns.scatterplot(x=x, y=y, size=z, hue=colors, sizes=(50, 5000), linewidth=0, alpha=0.5)
h,l = g.get_legend_handles_labels()

plt.legend(h[0:2],l[0:2],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Vehicle Capacity")
plt.ylabel("Vehicle Number")
plt.ylim(20,220)
plt.title("Internal Costs (High Demand)")
plt.show()


# # With vs. Without external cost routing and pooling
# low_demand_df_mean_wo = low_demand_df_mean.T[low_demand_df_mean.loc["rootpool"] == 'woext'].T
# low_demand_df_mean_w = low_demand_df_mean.T[low_demand_df_mean.loc["rootpool"] == 'wext'].T
#
# low_demand_df_mean_wo_sort = low_demand_df_mean_wo.sort_values(by='total external costs', axis='columns')
#
# for col1 in range(0, len(low_demand_df_mean_wo_sort.columns)):
#     size = len(low_demand_df_mean_wo_sort.columns[col1])
#     low_demand_df_mean_wo_sort.rename(columns={low_demand_df_mean_wo_sort.columns[col1] : low_demand_df_mean_wo_sort.columns[col1][:size-6]})
#
# for col2 in range(0, len(low_demand_df_mean_wo_sort.columns)):
#     size = len(low_demand_df_mean_w.columns[col2])
#     low_demand_df_mean_w.rename(columns={low_demand_df_mean_w.columns[col2]:low_demand_df_mean_w.columns[col2][:size-5]})
#
# print(low_demand_df_mean_wo_sort.columns)
# print(low_demand_df_mean_w.columns)
#
# # y_wext = low_demand_df_mean_w.loc['total external costs']
# #
# # print(low_demand_df_mean_wo_sort)
#
# plt.plot(low_demand_df_mean_wo_sort.columns, low_demand_df_mean_wo_sort.loc['total external costs'])
# plt.plot(low_demand_df_mean_w.columns, low_demand_df_mean_w.loc['total external costs'])
# plt.show()


# # Operation file evaluation
# # path of result files
# path1_user = "studies/munich_study/results/munich_study_200e-Smart_woext_week/1_user-stats.csv"
# path2_user = "studies/munich_study/results/munich_study_200e-Smart_wext_week/1_user-stats.csv"
# path1_op = "studies/munich_study/results/munich_study_200e-Smart_woext_week/2-0_op-stats.csv"
# path2_op = "studies/munich_study/results/munich_study_200e-Smart_wext_week/2-0_op-stats.csv"
# path1_stand = "studies/munich_study/results/munich_study_200e-Smart_woext_week/standard_eval.csv"
# path2_stand = "studies/munich_study/results/munich_study_200e-Smart_wext_week/standard_eval.csv"
#
# # evaluate user stats
# # import user stats
# user_stat_df1 = pd.read_csv(path1_user, index_col='request_id')
# user_stat_df2 = pd.read_csv(path2_user, index_col='request_id')
#
# user_stat_df1_sort = user_stat_df1.sort_values(by=['request_id'])
# user_stat_df2_sort = user_stat_df2.sort_values(by=['request_id'])
# user_stat_df1_sort["waiting_time"] = user_stat_df1_sort["pickup_time"] - user_stat_df1_sort["rq_time"]
# user_stat_df2_sort["waiting_time"] = user_stat_df2_sort["pickup_time"] - user_stat_df2_sort["rq_time"]
# user_stat_df_diff = pd.DataFrame()
# user_stat_df_diff["diff_travel_time"] = user_stat_df1_sort['direct_route_travel_time'] - user_stat_df2_sort['direct_route_travel_time']
# user_stat_df_diff["diff_waiting_time"] = user_stat_df1_sort["waiting_time"] - user_stat_df2_sort["waiting_time"]
# user_stat_df_diff["diff_route_distance"] = user_stat_df1_sort["direct_route_distance"] - user_stat_df2_sort["direct_route_distance"]
#
# # pd.set_option('display.max_columns', None)
# # print(user_stat_df1_sort)
#
# text_travel = f'Average = {np.round(user_stat_df_diff["diff_travel_time"].replace([np.inf, -np.inf], np.nan).mean(), 2)}'
# text_wait = f'Average = {np.round(user_stat_df_diff["diff_waiting_time"].replace([np.inf, -np.inf], np.nan).mean(), 2)}'
# text_distance = f'Average = {np.round(user_stat_df_diff["diff_route_distance"].replace([np.inf, -np.inf], np.nan).mean(), 2)}'
#
# window = 50
# average_y_travel = []
# average_y_wait = []
# average_y_distance = []
#
# for i in range (len(user_stat_df_diff["diff_travel_time"]) - window):
#     average_y_travel.append(np.mean(user_stat_df_diff.loc[i:i+window, "diff_travel_time"]))
#     average_y_wait.append(np.mean(user_stat_df_diff.loc[i:i + window, "diff_waiting_time"]))
#     average_y_distance.append(np.mean(user_stat_df_diff.loc[i:i + window, "diff_route_distance"]))
#
# for k in range (window):
#     average_y_travel.append(0)
#     average_y_wait.append(0)
#     average_y_distance.append(0)
#
#
# #plot network
# # munich_gdf = gpd.read_file("data/networks/munich_network/munich-polygon.geojson")
# # plt.figure
# # munich_gdf.plot()
#
# # # fig, ax = plt.subplots()
# # figure for travel time
# plt.figure()
# plt.text(0, 0, text_travel, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
# plt.plot(user_stat_df_diff.index, user_stat_df_diff["diff_travel_time"], label="Original Data")
# plt.plot(user_stat_df_diff.index, average_y_travel, label="Running Average")
# plt.title('Travel Time (Without - With)')
# plt.ylabel('Difference in travel time in sec')
# plt.xlabel('Request ID')
# plt.legend()
#
# # figure for waiting time
# plt.figure()
# plt.text(0, 0, text_wait, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
# plt.plot(user_stat_df_diff.index, user_stat_df_diff["diff_waiting_time"], label="Original Data")
# plt.plot(user_stat_df_diff.index, average_y_wait, label="Running Average")
# plt.title('Waiting Time (Without - With)')
# plt.ylabel('Difference in waiting time in sec')
# plt.xlabel('Request ID')
# plt.legend()
#
# # figure for travel distance
# plt.figure()
# plt.text(0, 0, text_distance, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
# plt.plot(user_stat_df_diff.index, user_stat_df_diff["diff_route_distance"] , label="Original Data")
# plt.plot(user_stat_df_diff.index, average_y_distance, label="Running Average")
# plt.title('Travel Distance (Without - With)')
# plt.ylabel('Difference in travel distance in m')
# plt.xlabel('Request ID')
# plt.legend()
#
#
#
# # evaluate operator stats
# op_stat_df1 = pd.read_csv(path1_op)
# op_stat_df2 = pd.read_csv(path2_op)
#
# pd.set_option('display.max_columns', None)
# print(op_stat_df1)
#
# # text_ext1 = f'Average = {np.round(op_stat_df1["external_costs"].replace([np.inf, -np.inf], np.nan).mean(), 2)}'
#
# window = 100
# average_y_ext1 = []
# average_y_ext2 = []
#
# for i in range (len(op_stat_df1["external_costs"]) - window):
#     average_y_ext1.append(np.mean(op_stat_df1.loc[i:i+window, "external_costs"]))
#
# for l in range(len(op_stat_df2["external_costs"]) - window):
#     average_y_ext2.append(np.mean(op_stat_df2.loc[l:l + window, "external_costs"]))
#
# for k in range (window):
#     average_y_ext1.append(0)
#     average_y_ext2.append(0)
#
#
# plt.figure()
# # plt.text(0, 0, text_ext, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
# plt.plot(op_stat_df2.index, op_stat_df2["external_costs"], label="With Ext Routing")
# plt.plot(op_stat_df1.index, op_stat_df1["external_costs"], label="Without Ext Routing")
# plt.plot(op_stat_df2.index, average_y_ext2, label="Running Average With Ext Routing")
# plt.plot(op_stat_df1.index, average_y_ext1, label="Running Average without Ext Routing")
# plt.title('External Costs')
# plt.ylabel('External Costs in Euro-Cent')
# plt.xlabel('Operation ID')
# plt.legend()
#
# plt.figure()
# # plt.text(0, 0, text_ext, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
# plt.plot(op_stat_df2.index, op_stat_df2["occupancy"], label="With Ext Routing")
# plt.plot(op_stat_df1.index, op_stat_df1["occupancy"], label="Without Ext Routing")
# plt.title('Occupancy')
# plt.ylabel('Occupancy in persons')
# plt.xlabel('Operation ID')
# plt.legend()
#
# plt.show()




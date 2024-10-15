import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
plt.rcParams.update({'font.size': 22})

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(MAIN_DIR)
from src.FleetSimulationBase import load_scenario_inputs
from src.evaluation.standard import get_directory_dict
from src.misc.globals import *

# assuming temporal resolution of seconds in output files
MIN = 60
HOUR = 3600
DAY = 24 * 3600
DEF_TEMPORAL_RESOLUTION = 15*MIN

def weight_vrl_interval(veh_stat_row, interval_start, interval_end):
    st = veh_stat_row[G_VR_LEG_START_TIME]
    et = veh_stat_row[G_VR_LEG_END_TIME]
    if st > et:
        return 0
    elif st > interval_end:
        return 0
    elif et < interval_start:
        return 0
    else:
        return (min(et, interval_end) - max(st, interval_start))/(interval_end - interval_start)

def bin_vehicle_stats(veh_stat_df, interval_list):
    binned_vehicle_stats = {}
    interval_list.append(float("inf"))
    for i in range(1, len(interval_list)):
        sb = interval_list[i-1]
        eb = interval_list[i]
        #veh_stats = veh_stat_df[((veh_stat_df[G_VR_LEG_START_TIME] >= sb) & (veh_stat_df[G_VR_LEG_START_TIME] <= eb))]
        veh_stats = veh_stat_df[(( (veh_stat_df[G_VR_LEG_START_TIME] >= sb) & (veh_stat_df[G_VR_LEG_START_TIME] <= eb) )|( (veh_stat_df[G_VR_LEG_END_TIME] >= sb) & (veh_stat_df[G_VR_LEG_END_TIME] <= eb) )|( (veh_stat_df[G_VR_LEG_START_TIME] < sb) & (veh_stat_df[G_VR_LEG_END_TIME] > eb) ) )]
        if veh_stats.shape[0] > 0:
            veh_stats["interval_weight"] = veh_stats.apply(weight_vrl_interval, axis = 1, args = (sb, eb))
            binned_vehicle_stats[sb] = veh_stats.copy()
    return binned_vehicle_stats

def avg_occ_binned(binned_vehicle_stats, output_dir, op_id, show = True):
    def weight_ob_rq(entries):
        if pd.isnull(entries[G_VR_OB_RID]):
            return 0.0
        else:
            number_ob_rq = len(str(entries[G_VR_OB_RID]).split(";"))
            return number_ob_rq * entries[G_VR_LEG_DISTANCE] * entries["interval_weight"]

    def weight_ob_pax(entries):
        try:
            return entries[G_VR_NR_PAX] * entries[G_VR_LEG_DISTANCE] * entries["interval_weight"]
        except:
            return 0.0
    ts = []
    occs_rq = []
    occs = []
    last = 0
    last_rq = 0
    for t, binned_stats_df in binned_vehicle_stats.items():
        ts.append(t/3600.0)
        occs.append(last)
        occs_rq.append(last_rq)
        last_rq = binned_stats_df.apply(weight_ob_rq, axis = 1).sum() / (binned_stats_df[G_VR_LEG_DISTANCE]*binned_stats_df["interval_weight"]).sum()
        last = binned_stats_df.apply(weight_ob_pax, axis=1).sum() / (binned_stats_df[G_VR_LEG_DISTANCE]*binned_stats_df["interval_weight"]).sum()
        ts.append(t/3600.0)
        occs.append(last)
        occs_rq.append(last_rq)
    plt.figure(figsize=(7,7))
    plt.plot(ts, occs, label = "people")
    plt.plot(ts, occs_rq, label = "requests")
    plt.xlabel("Time [h]")
    plt.ylabel("Avg Occupancy")
    plt.legend()
    plt.xticks([i*24 for i in range(int(np.floor(ts[0]/24)), int(np.ceil(ts[-1]/24)) )])
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(output_dir, "temporal_occupancy_op_{}.png".format(op_id)))
    plt.close()
    return ts, occs, occs_rq

def avg_util_binned(binned_vehicle_stats, output_dir, op_id, n_vehicles, show = True):
    ts = []
    util = []
    last = 0
    bins = list(binned_vehicle_stats.keys())
    util_states = ["route", "boarding", "waiting", "reposition"]
    for t, binned_stats_df in binned_vehicle_stats.items():
        ts.append(t/3600.0)
        util.append(last)
        last = binned_stats_df[binned_stats_df["status"].isin(util_states)]["interval_weight"].sum()* 100.0 / n_vehicles 
        #last = (binned_stats_df["interval_weight"]*(binned_stats_df[G_VR_LEG_END_TIME] - binned_stats_df[G_VR_LEG_START_TIME])).sum() * 100.0 / 250 / delta / binned_stats_df.shape[0]
        ts.append(t/3600.0)
        util.append(last)
    plt.figure(figsize=(7,7))
    plt.plot(ts, util)
    plt.xlabel("Time [h]")
    plt.ylabel("Utilization [%]")
    plt.xticks([i*24 for i in range(int(np.floor(ts[0]/24)), int(np.ceil(ts[-1]/24)) )])
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(output_dir, "temporal_util_op_{}.png".format(op_id)))
    plt.close()
    return ts, util

def temporal_plots(output_dir, scenario_parameters, op_attributes, dir_names, op_id, show=False):
    op_stats = os.path.join(output_dir, "2-{}_op-stats.csv".format(int(op_id)))
    op_df = pd.read_csv(op_stats)
    start_time = scenario_parameters[G_SIM_START_TIME]
    end_time = max(op_df["end_time"].values)
    bin_intervals = []
    i = 0
    while i * DEF_TEMPORAL_RESOLUTION + start_time <= end_time:
        bin_intervals.append(i * DEF_TEMPORAL_RESOLUTION + start_time)
        i+=1
    binned_vehicle_stats = bin_vehicle_stats(op_df, bin_intervals)

    n_vehicles = len(op_df["vehicle_id"].unique())
    temporal_plots_dict = {}
    temporal_plots_dict["avg_util_binned"] = avg_util_binned(binned_vehicle_stats, output_dir, op_id, n_vehicles, show = show)
    temporal_plots_dict["avg_occ_binned"] = avg_occ_binned(binned_vehicle_stats, output_dir, op_id, show = show)
    return temporal_plots_dict

def vehicle_stats_over_time(output_dir, scenario_parameters, operator_attributes, dir_names, op_id, show = True, bin_size = None):
    """ this function creates a plot of the different vehicle states over time """
    op_stats = os.path.join(output_dir, "2-{}_op-stats.csv".format(int(op_id)))
    op_df = pd.read_csv(op_stats)
    #insert idle
    start_time = scenario_parameters[G_SIM_START_TIME]
    end_time = max(op_df["end_time"].values)
    add_df_list = []
    to_drop = []    # some jobs dont finish correctly (charging and out_of_service doubled) # TODO #
    for vid, veh_stats in op_df.groupby("vehicle_id"):
        veh_stats.sort_values(by='end_time', inplace = True)
        # if veh_stats.shape[0] > 1 and veh_stats.iloc[-1]["start_time"] < veh_stats.iloc[-2]["end_time"]:
        #     to_drop.append(veh_stats.iloc[-1].name)
        #     veh_stats.drop([veh_stats.iloc[-1].name], inplace = True)
        last_end_time = start_time
        c = 0
        last_entry = None
        for _, entry in veh_stats.iterrows():
            if c == 0:
                if entry["start_time"] > start_time:
                    d = {}
                    for key, val in entry.items():
                        if key == "start_time":
                            d["start_time"] = start_time
                        elif key == "end_time":
                            d["end_time"] = entry["start_time"]
                        elif key == "status":
                            d["status"] = "idle"
                        elif key == "end_soc":
                            d["end_soc"] = entry["start_soc"]
                        elif key == "start_soc":
                            d["start_soc"] = entry["start_soc"]
                        else:
                            d[key] = val
                    add_df_list.append(d)
                last_end_time = entry["end_time"]
                c+=1
                continue
            c+=1
            # if entry["start_time"] < last_end_time:
            #     #print(entry["vehicle_id"], entry["start_time"], last_end_time)
            #     continue
            if entry["start_time"] > last_end_time:
                d = {}
                for key, val in entry.items():
                    if key == "start_time":
                        d["start_time"] = last_end_time
                    elif key == "end_time":
                        d["end_time"] = entry["start_time"]
                    elif key == "status":
                        if entry["status"] == "out_of_service" or entry["status"] == "charging" or entry["status"] == "to_depot":
                            d["status"] = "idle"
                        else:
                            d["status"] = "idle"
                    elif key == "end_soc":
                        d["end_soc"] = entry["start_soc"]
                    else:
                        d[key] = val
                add_df_list.append(d)
            last_end_time = entry["end_time"]
            last_entry = entry
        if last_end_time < end_time and last_entry is not None:
            d = {}
            for key, val in last_entry.items():
                if key == "start_time":
                    d["start_time"] = last_end_time
                elif key == "end_time":
                    d["end_time"] = end_time
                elif key == "status":
                    d["status"] = "idle"
                elif key == "end_soc":
                    d["end_soc"] = last_entry["start_soc"]
                else:
                    d[key] = val
            # print(last_entry)
            # print(pd.Series(d))
            # print("")
            add_df_list.append(d)
    idle_df = pd.DataFrame(add_df_list)
    op_df.drop(to_drop, inplace=True)
    op_df = pd.concat([op_df, idle_df], axis = 0, ignore_index = True)
    op_df.sort_values(by='end_time', inplace = True)
    times = {}
    for t in op_df["start_time"].values:
        times[t] = 1
    for t in op_df["end_time"].values:
        times[t] = 1
    times = sorted([t for t in times.keys()])
    new_times = []
    for i, t in enumerate(times):
        if i == 0:
            new_times.append(t)
            continue
        while t - new_times[-1] > 150:
            new_times.append(new_times[-1] + 150)
        new_times.append(t)
    times = new_times
    plotvalues = {}
    for status, status_df in op_df.groupby("status"):
        starts = status_df["start_time"].values
        ends = status_df["end_time"].values
        together = [(s,1) for s in starts] + [(e,-1) for e in ends]
        together = sorted(together, key = lambda x:x[0])
        y = []
        n = 0
        together_index = 0
        for t in times:
            while together_index < len(together) and t >= together[together_index][0]:
                n += together[together_index][1]
                together_index += 1
            y.append(n)
        plotvalues[status] = y
    ts = []
    vehs = []
    sum_veh = len(op_df["vehicle_id"].unique())
    print("NUMBER VEHICLES DETECTED: ", sum_veh)
    if dir_names.get(G_DIR_FCTRL) is not None:
        time_active_vehicles_f = operator_attributes.get(G_OP_ACT_FLEET_SIZE, None)
        if time_active_vehicles_f is not None:
            time_active_vehicles_p = os.path.join(dir_names[G_DIR_FCTRL], "elastic_fleet_size", time_active_vehicles_f)
            v_curve = pd.read_csv(time_active_vehicles_p)
            n0 = 0
            for k, entries in v_curve.iterrows():
                t = entries["time"]
                frac = entries["share_active_fleet_size"]
                ts.append(t)
                vehs.append(n0)
                n0 = sum_veh*frac
                ts.append(t)
                vehs.append(n0)

    states_last = ["idle", "to_depot", "charging", "out_of_service"]
    values_last = [plotvalues.get(k, [0 for i in range(len(times))]) for k in states_last]
    states = []
    values = []
    for key, value in plotvalues.items():
        if not key in states_last:
            states.append(key)
            values.append(value)
    states += states_last
    values += values_last

    # if smooth > 1:
    #     new_times = []
    #     for i, t in enumerate(times):
    #         if i%smooth == 0:
    #             new_times.append(t)
    #     if (len(times)-1)%smooth != 0:
    #         new_times.append(times[-1])
    #     times = new_times
    #     for j, value in enumerate(values):
    #         new_value = []
    #         for i, t in enumerate(value):
    #             if i%smooth == 0:
    #                 new_value.append(t)
    #         if (len(value)-1)%smooth != 0:
    #             new_value.append(value[-1])
    #         values[j] = new_value

    if bin_size is not None:
        new_times = []
        new_indices = []
        last_time = None
        for i, t in enumerate(times):
            if i == 0:
                last_time = t
                new_times.append(t)
                new_indices.append(i)
            else:
                if t - last_time >= bin_size:
                    if t - times[i-1] >= bin_size:
                        new_times.append(times[i-1])
                        new_indices.append(i-1)
                    new_times.append(t)
                    new_indices.append(i)
                    last_time = t
        times = new_times
        for j, value in enumerate(values):
            new_value = []
            for i in new_indices:
                new_value.append(value[i])
            values[j] = new_value

    # for i, value in enumerate(values):
    #     plt.plot(times, value, label = states[i])
    # plt.legend()
    # plt.show()
    # plt.close()

    plt.figure(figsize=(10,10))
    plt.stackplot(times, values, labels=states)
    if len(vehs) > 0:
        plt.plot(ts, vehs, "k-")
    plt.legend(title='Vehicle States', bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.xlabel("Time [h]")
    plt.ylabel("Accumulated Number of Vehicles")
    plt.xticks([i*3600*12 for i in range(int(times[-1]/3600/12))], labels=[f"{i*12}" for i in range(int(times[-1]/3600/12))]) #labels=[f"2019/{i+18}/11" for i in range(int(times[-1]/3600/24))]
    plt.xlim(start_time, times[-1])
    #plt.tight_layout(rect=[0,0,0.75,1])
    if show:
        plt.show()
    plt.savefig(os.path.join(output_dir, "fleet_states_time_op_{}.png".format(op_id)), bbox_inches="tight")
    plt.close()

    # plt.figure(figsize=(10,10))
    # plt.stackplot(times, values, labels=states)
    # if len(vehs) > 0:
    #     plt.plot(ts, vehs, "k-")
    # plt.legend()
    # plt.xlabel("time [h]")
    # plt.ylabel("accumulated number of vehicles")
    # plt.xticks([i*3600*6 for i in range(int(times[-1]/3600/6))], labels=[i*6 for i in range(int(times[-1]/3600/6))])
    # plt.xlim(start_time, times[-1])
    # plt.tight_layout()
    # if show:
    #     plt.show()
    # plt.savefig(os.path.join(output_dir, "fleet_states_time_op_{}.png".format(op_id)))
    # plt.close()

def occupancy_over_time(output_dir, scenario_parameters, operator_attributes, dir_names, op_id, show = True, bin_size = None):
    """ this function creates a plot of the different vehicle occupancies over time """
    op_stats = os.path.join(output_dir, "2-{}_op-stats.csv".format(int(op_id)))
    op_df = pd.read_csv(op_stats)
    max_occupancy = op_df[op_df["status"] == "route"]["occupancy"].max()
    #insert idle
    start_time = scenario_parameters[G_SIM_START_TIME]
    end_time = max(op_df["end_time"].values)
    add_df_list = []
    to_drop = []    # some jobs dont finish correctly (charging and out_of_service doubled) # TODO #
    for vid, veh_stats in op_df.groupby("vehicle_id"):
        veh_stats.sort_values(by='end_time', inplace = True)
        if veh_stats.shape[0] > 1 and veh_stats.iloc[-1]["start_time"] < veh_stats.iloc[-2]["end_time"]:
            to_drop.append(veh_stats.iloc[-1].name)
            veh_stats.drop([veh_stats.iloc[-1].name], inplace = True)
        last_end_time = start_time
        c = 0
        last_entry = None
        for _, entry in veh_stats.iterrows():
            if c == 0:
                if entry["start_time"] > start_time:
                    d = {}
                    for key, val in entry.items():
                        if key == "start_time":
                            d["start_time"] = start_time
                        elif key == "end_time":
                            d["end_time"] = entry["start_time"]
                        elif key == "status":
                            d["status"] = "idle"
                        elif key == "end_soc":
                            d["end_soc"] = entry["start_soc"]
                        elif key == "start_soc":
                            d["start_soc"] = entry["start_soc"]
                        elif key == "occupancy":
                            d["occupancy"] = 0
                        else:
                            d[key] = val
                    add_df_list.append(d)
                last_end_time = entry["end_time"]
                c+=1
                continue
            c+=1
            # if entry["start_time"] < last_end_time:
            #     #print(entry["vehicle_id"], entry["start_time"], last_end_time)
            #     continue
            if entry["start_time"] > last_end_time:
                d = {}
                for key, val in entry.items():
                    if key == "start_time":
                        d["start_time"] = last_end_time
                    elif key == "end_time":
                        d["end_time"] = entry["start_time"]
                    elif key == "status":
                        if entry["status"] == "out_of_service" or entry["status"] == "charging" or entry["status"] == "to_depot":
                            d["status"] = "idle"
                        else:
                            d["status"] = "idle"
                    elif key == "end_soc":
                        d["end_soc"] = entry["start_soc"]
                    elif key == "occupancy":
                        d["occupancy"] = 0
                    else:
                        d[key] = val
                add_df_list.append(d)
            last_end_time = entry["end_time"]
            last_entry = entry
        if last_end_time < end_time and last_entry is not None:
            d = {}
            for key, val in last_entry.items():
                if key == "start_time":
                    d["start_time"] = last_end_time
                elif key == "end_time":
                    d["end_time"] = end_time
                elif key == "status":
                    d["status"] = "idle"
                elif key == "end_soc":
                    d["end_soc"] = last_entry["start_soc"]
                elif key == "occupancy":
                    d["occupancy"] = 0
                else:
                    d[key] = val
            # print(last_entry)
            # print(pd.Series(d))
            # print("")
            add_df_list.append(d)
    idle_df = pd.DataFrame(add_df_list)
    print(to_drop)
    op_df.drop(to_drop, inplace=True)
    op_df = pd.concat([op_df, idle_df], axis = 0, ignore_index = True)

    op_df.sort_values(by='end_time', inplace = True)
    times = {}
    for t in op_df["start_time"].values:
        times[t] = 1
    for t in op_df["end_time"].values:
        times[t] = 1
    times = sorted([t for t in times.keys()])
    new_times = []
    for i, t in enumerate(times):
        if i == 0:
            new_times.append(t)
            continue
        while t - new_times[-1] > 150:
            new_times.append(new_times[-1] + 150)
        new_times.append(t)
    times = new_times
    plotvalues = {}
    for occupancy, occ_df in op_df.groupby("occupancy"):
        if occupancy > max_occupancy:
            print("Warning {} entries with large occupancy than {} -> ignored (usually boarding processes)".format(occ_df.shape[0], max_occupancy))
            continue
        if occupancy > 0:
            starts = occ_df["start_time"].values
            ends = occ_df["end_time"].values
            together = [(s,1) for s in starts] + [(e,-1) for e in ends]
            together = sorted(together, key = lambda x:x[0])
            y = []
            n = 0
            together_index = 0
            for t in times:
                while together_index < len(together) and t >= together[together_index][0]:
                    n += together[together_index][1]
                    together_index += 1
                y.append(n)
            plotvalues[str(occupancy)] = y
        else:
            #non action_states 
            non_action_states = ["out_of_service", "charging", "idle"]
            action_occ_df = occ_df[occ_df["status"].isin(non_action_states) == False]
            non_action_occ_df = occ_df[occ_df["status"].isin(non_action_states)]
            for i, occ_0_df in enumerate([action_occ_df, non_action_occ_df]):
                starts = occ_0_df["start_time"].values
                ends = occ_0_df["end_time"].values
                together = [(s,1) for s in starts] + [(e,-1) for e in ends]
                together = sorted(together, key = lambda x:x[0])
                y = []
                n = 0
                together_index = 0
                for t in times:
                    while together_index < len(together) and t >= together[together_index][0]:
                        n += together[together_index][1]
                        together_index += 1
                    y.append(n)
                if i == 0:
                    plotvalues["0"] = y
                else:
                    plotvalues["inactive"] = y

    occs_last = ["inactive"]
    values_last = [plotvalues.get(k, [0 for i in range(len(times))]) for k in occs_last]
    occupancy = []
    values = []
    for key, value in sorted(plotvalues.items(), key = lambda x:x[0]):
        if not key in occs_last:
            occupancy.append(key)
            values.append(value)
    # occupancy += occs_last
    # values += values_last

    if bin_size is not None:
        new_times = []
        new_indices = []
        last_time = None
        for i, t in enumerate(times):
            if i == 0:
                last_time = t
                new_times.append(t)
                new_indices.append(i)
            else:
                if t - last_time >= bin_size:
                    if t - times[i-1] >= bin_size:
                        new_times.append(times[i-1])
                        new_indices.append(i-1)
                    new_times.append(t)
                    new_indices.append(i)
                    last_time = t
        times = new_times
        for j, value in enumerate(values):
            new_value = []
            for i in new_indices:
                new_value.append(value[i])
            values[j] = new_value


    plt.figure(figsize=(10,10))
    plt.stackplot(times, values, labels=occupancy)
    plt.legend(title='Occupancy', bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.xlabel("Time [h]")
    plt.ylabel("Accumulated Number of Vehicles")
    plt.xticks([i for i in range(start_time, int(times[-1]), 3600*12)], labels=[f"{int(i/3600)}" for i in range(start_time, int(times[-1]), 3600*12)]) #labels=[f"2019/{i+18}/11" for i in range(int(times[-1]/3600/24))]
    plt.xlim(start_time, times[-1])
    #plt.ylim(0, 3000)
    #plt.tight_layout(rect=[0,0,0.75,1])
    if show:
        plt.show()
    plt.savefig(os.path.join(output_dir, "fleet_occupancy_time_op_{}.png".format(op_id)), bbox_inches="tight")
    plt.close()

def occupancy_over_time_rel(output_dir, scenario_parameters, operator_attributes, dir_names, op_id, bin_size = None, show = True):
    """ this function creates a plot of the different vehicle occupancies over time """
    op_stats = os.path.join(output_dir, "2-{}_op-stats.csv".format(int(op_id)))
    op_df = pd.read_csv(op_stats)
    max_occupancy = op_df[op_df["status"] == "route"]["occupancy"].max()
    #insert idle
    start_time = scenario_parameters[G_SIM_START_TIME]
    end_time = max(op_df["end_time"].values)
    add_df_list = []
    to_drop = []    # some jobs dont finish correctly (charging and out_of_service doubled) # TODO #
    for vid, veh_stats in op_df.groupby("vehicle_id"):
        veh_stats.sort_values(by='end_time', inplace = True)
        if veh_stats.shape[0] > 1 and veh_stats.iloc[-1]["start_time"] < veh_stats.iloc[-2]["end_time"]:
            to_drop.append(veh_stats.iloc[-1].name)
            veh_stats.drop([veh_stats.iloc[-1].name], inplace = True)
        last_end_time = start_time
        c = 0
        last_entry = None
        for _, entry in veh_stats.iterrows():
            if c == 0:
                if entry["start_time"] > start_time:
                    d = {}
                    for key, val in entry.items():
                        if key == "start_time":
                            d["start_time"] = start_time
                        elif key == "end_time":
                            d["end_time"] = entry["start_time"]
                        elif key == "status":
                            d["status"] = "idle"
                        elif key == "end_soc":
                            d["end_soc"] = entry["start_soc"]
                        elif key == "start_soc":
                            d["start_soc"] = entry["start_soc"]
                        elif key == "occupancy":
                            d["occupancy"] = 0
                        else:
                            d[key] = val
                    add_df_list.append(d)
                last_end_time = entry["end_time"]
                c+=1
                continue
            c+=1
            # if entry["start_time"] < last_end_time:
            #     #print(entry["vehicle_id"], entry["start_time"], last_end_time)
            #     continue
            if entry["start_time"] > last_end_time:
                d = {}
                for key, val in entry.items():
                    if key == "start_time":
                        d["start_time"] = last_end_time
                    elif key == "end_time":
                        d["end_time"] = entry["start_time"]
                    elif key == "status":
                        if entry["status"] == "out_of_service" or entry["status"] == "charging" or entry["status"] == "to_depot":
                            d["status"] = "idle"
                        else:
                            d["status"] = "idle"
                    elif key == "end_soc":
                        d["end_soc"] = entry["start_soc"]
                    elif key == "occupancy":
                        d["occupancy"] = 0
                    else:
                        d[key] = val
                add_df_list.append(d)
            last_end_time = entry["end_time"]
            last_entry = entry
        if last_end_time < end_time and last_entry is not None:
            d = {}
            for key, val in last_entry.items():
                if key == "start_time":
                    d["start_time"] = last_end_time
                elif key == "end_time":
                    d["end_time"] = end_time
                elif key == "status":
                    d["status"] = "idle"
                elif key == "end_soc":
                    d["end_soc"] = last_entry["start_soc"]
                elif key == "occupancy":
                    d["occupancy"] = 0
                else:
                    d[key] = val
            # print(last_entry)
            # print(pd.Series(d))
            # print("")
            add_df_list.append(d)
    idle_df = pd.DataFrame(add_df_list)
    print(to_drop)
    op_df.drop(to_drop, inplace=True)
    op_df = pd.concat([op_df, idle_df], axis = 0, ignore_index = True)

    op_df.sort_values(by='end_time', inplace = True)
    times = {}
    for t in op_df["start_time"].values:
        times[t] = 1
    for t in op_df["end_time"].values:
        times[t] = 1
    times = sorted([t for t in times.keys()])
    new_times = []
    for i, t in enumerate(times):
        if i == 0:
            new_times.append(t)
            continue
        while t - new_times[-1] > 150:
            new_times.append(new_times[-1] + 150)
        new_times.append(t)
    times = new_times
    plotvalues = {}
    for occupancy, occ_df in op_df.groupby("occupancy"):
        if occupancy > max_occupancy:
            print("Warning {} entries with large occupancy than {} -> ignored (usually boarding processes)".format(occ_df.shape[0], max_occupancy))
            continue
        if occupancy > 0:
            starts = occ_df["start_time"].values
            ends = occ_df["end_time"].values
            together = [(s,1) for s in starts] + [(e,-1) for e in ends]
            together = sorted(together, key = lambda x:x[0])
            y = []
            n = 0
            together_index = 0
            for t in times:
                while together_index < len(together) and t >= together[together_index][0]:
                    n += together[together_index][1]
                    together_index += 1
                y.append(n)
            plotvalues[str(occupancy)] = y
        else:
            #non action_states 
            non_action_states = ["out_of_service", "charging", "idle"]
            action_occ_df = occ_df[occ_df["status"].isin(non_action_states) == False]
            non_action_occ_df = occ_df[occ_df["status"].isin(non_action_states)]
            for i, occ_0_df in enumerate([action_occ_df, non_action_occ_df]):
                starts = occ_0_df["start_time"].values
                ends = occ_0_df["end_time"].values
                together = [(s,1) for s in starts] + [(e,-1) for e in ends]
                together = sorted(together, key = lambda x:x[0])
                y = []
                n = 0
                together_index = 0
                for t in times:
                    while together_index < len(together) and t >= together[together_index][0]:
                        n += together[together_index][1]
                        together_index += 1
                    y.append(n)
                if i == 0:
                    plotvalues["0"] = y
                else:
                    plotvalues["inactive"] = y

    occs_last = ["inactive"]
    values_last = [plotvalues.get(k, [0 for i in range(len(times))]) for k in occs_last]
    occupancy = []
    values = []
    for key, value in sorted(plotvalues.items(), key = lambda x:x[0]):
        if not key in occs_last:
            occupancy.append(key)
            values.append(value)
    # occupancy += occs_last
    # values += values_last
    for t_index in range(len(values[0])):
        s = 0
        for value in values:
            s += value[t_index]
        if s > 0:
            for i in range(len(values)):
                values[i][t_index] = values[i][t_index]/s*100.0
    if bin_size is not None:
        new_times = []
        new_indices = []
        last_time = None
        for i, t in enumerate(times):
            if i == 0:
                last_time = t
                new_times.append(t)
                new_indices.append(i)
            else:
                if t - last_time >= bin_size:
                    if t - times[i-1] >= bin_size:
                        new_times.append(times[i-1])
                        new_indices.append(i-1)
                    new_times.append(t)
                    new_indices.append(i)
                    last_time = t
        times = new_times
        for j, value in enumerate(values):
            new_value = []
            for i in new_indices:
                new_value.append(value[i])
            values[j] = new_value

    plt.figure(figsize=(10,10))
    plt.stackplot(times, values, labels=occupancy)
    plt.legend(title='Occupancy', bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.xlabel("Time [h]")
    plt.ylabel("Fraction of Vehicles [%]")
    plt.xticks([i*3600*12 for i in range(int(times[-1]/3600/12))], labels=[f"{12 * i}" for i in range(int(times[-1]/3600/12))]) #labels=[f"2019/{i+18}/11" for i in range(int(times[-1]/3600/24))]
    plt.xlim(start_time, times[-1])
    #plt.tight_layout(rect=[0,0,0.75,1])
    if show:
        plt.show()
    plt.savefig(os.path.join(output_dir, "fleet_occupancy_frac_time_op_{}.png".format(op_id)), bbox_inches="tight")
    plt.close()

### rq stats
def set_sc_rq_df_departure_delay_and_wt(sc_rq_df):
    from src.evaluation.standard import decode_offer_str
    def get_departure_delay(rq_row):
        offers = rq_row["offers"]
        #0:t_wait:585.8999999999614;t_drive:1896.6000000000022;fare:762.0;t_access:110;t_egress:242
        offer_dict = decode_offer_str(offers)
        if len(offer_dict) == 0:
            return np.nan
        else:
            return offer_dict[0]["t_wait"] - offer_dict[0]["t_access"]
    sc_rq_df["departure_delay"] = sc_rq_df.apply(get_departure_delay, axis = 1)
    def get_moia_wait_time(rq_row):
        if pd.isna(rq_row["pickup_time"]):
            return rq_row["pickup_time"]
        real_departure_dely = rq_row["pickup_time"] - rq_row["rq_time"] - rq_row["access_time"]
        return real_departure_dely - rq_row["departure_delay"]
    sc_rq_df["moia_wait_time"] = sc_rq_df.apply(get_moia_wait_time, axis = 1)
    sc_rq_df["travel time"] = sc_rq_df["dropoff_time"] - sc_rq_df["pickup_time"]
    walking_speed = 1.0
    def get_walking_distance(rq_row):
        offers = rq_row["offers"]
        #0:t_wait:585.8999999999614;t_drive:1896.6000000000022;fare:762.0;t_access:110;t_egress:242
        offer_dict = decode_offer_str(offers)
        if len(offer_dict) == 0:
            return 0
        else:
            return (offer_dict[0]["t_access"] + offer_dict[0]["t_egress"])/walking_speed
    sc_rq_df["walking_distance"] = sc_rq_df.apply(get_walking_distance, axis = 1)
    return sc_rq_df

def bin_sc_data(sc_rq_df, binsize):
    print(".... bin customer data")
    max_t = max(sc_rq_df["rq_time"].values)
    min_t = min(sc_rq_df["rq_time"].values)
    max_bin = np.math.ceil((max_t-min_t)/binsize)
    time_intervals = [(i*binsize + min_t) for i in range(max_bin)]

    bined_sc_rqs = {}
    for i, t in enumerate(time_intervals):
        #print("{}/{}".format(i, len(time_intervals)))
        if i==0:
            continue
        part_served = sc_rq_df[(sc_rq_df["rq_time"] >= time_intervals[i-1]) & (sc_rq_df["rq_time"] < t)]
        bined_sc_rqs[t] = part_served
    return bined_sc_rqs

def plot_served_rq_stats(bined_sc_rqs, results_dir, show = False):

    number_rqs = [x.shape[0] for x in bined_sc_rqs.values()]
    time = [x/3600.0 for x in bined_sc_rqs.keys()]
    served_sc = [x[x["vehicle_id"].notna()].shape[0] for x in bined_sc_rqs.values()]
    offered_sc = [x[(x["offers"].notna()) & (x["offers"] != "0:")].shape[0] for x in bined_sc_rqs.values()]
    no_offer_sc = [x[(x["offers"].isna()) | (x["offers"] == "0:")].shape[0] for x in bined_sc_rqs.values()]


    plt.figure(figsize=(7,7))
    #plt.plot(time, number_rqs, 'k-', label = "Number Rqs")
    plt.plot(time, served_sc, label = "served", color = "tab:blue")
    plt.plot(time, offered_sc, '--', label = "offered", color = "tab:orange")
    plt.plot(time, no_offer_sc, '--', label = "no offer", color = "tab:green")
    plt.legend()
    plt.xlabel("time [h]")
    plt.ylabel("Number Rqs")
    plt.xticks([i*24 for i in range(int(np.floor(time[0]/24)), int(np.ceil(time[-1]/24)) )])
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(results_dir, "served_stats_time.png"))
    plt.close()

    plt.figure(figsize=(7,7))
    #plt.plot(time, number_rqs, 'k-', label = "Number Rqs")
    plt.plot(time, served_sc, label = "served", color = "tab:blue")
    plt.plot(time, no_offer_sc, '--', label = "no offer", color = "tab:green")
    plt.legend()
    plt.xlabel("time [h]")
    plt.ylabel("Number Rqs")
    plt.xticks([i*24 for i in range(int(np.floor(time[0]/24)), int(np.ceil(time[-1]/24)) )])
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(results_dir, "served_stats_time_2.png"))
    plt.close()
    return time, served_sc, offered_sc, no_offer_sc

def plot_departure_delay_rq_stats(bined_sc_rqs,  results_dir,  show = False):

    number_rqs = [x.shape[0] for x in bined_sc_rqs.values()]
    time = [x/3600.0 for x in bined_sc_rqs.keys()]
    served_sc_dep = [x[x["vehicle_id"].notna()]["departure_delay"].mean() for x in bined_sc_rqs.values()]
    offered_sc_dep = [x[(x["offers"].notna()) | (x["offers"] != "0:")]["departure_delay"].mean() for x in bined_sc_rqs.values()]

    plt.figure(figsize=(7,7))
    #plt.plot(time, number_rqs, 'k-', label = "Number Rqs")
    plt.plot(time, served_sc_dep, label = "served", color = "tab:blue")
    plt.plot(time, offered_sc_dep, '--', label = "offered", color = "tab:orange")
    plt.legend()
    plt.xlabel("time [h]")
    plt.ylabel("departure delay [s]")
    plt.xticks([i*24 for i in range(int(np.floor(time[0]/24)), int(np.ceil(time[-1]/24)) )])
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(results_dir,  "departure_delay_stats.png"))
    plt.close()
    return time, served_sc_dep, offered_sc_dep

def plot_wait_rq_stats(bined_sc_rqs, results_dir, show = False):

    number_rqs = [x.shape[0] for x in bined_sc_rqs.values()]
    time = [x/3600.0 for x in bined_sc_rqs.keys()]

    served_sc_wait = [x[x["vehicle_id"].notna()]["moia_wait_time"].mean() for x in bined_sc_rqs.values()]

    plt.figure(figsize=(7,7))
    #plt.plot(time, number_rqs, 'k-', label = "Number Rqs")
    plt.plot(time, served_sc_wait, color = "tab:blue")
    plt.xlabel("time [h]")
    plt.ylabel("wait time [s]")
    plt.xticks([i*24 for i in range(int(np.floor(time[0]/24)), int(np.ceil(time[-1]/24)) )])
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(results_dir, "moia_wait_time_stats.png"))
    plt.close()
    return time, served_sc_wait

def plot_trip_duration_rq_stats(bined_sc_rqs, results_dir, show = False):

    number_rqs = [x.shape[0] for x in bined_sc_rqs.values()]
    time = [x/3600.0 for x in bined_sc_rqs.keys()]

    served_sc_wait = [x[x["vehicle_id"].notna()]["travel time"].mean() for x in bined_sc_rqs.values()]

    plt.figure(figsize=(7,7))
    #plt.plot(time, number_rqs, 'k-', label = "Number Rqs")
    plt.plot(time, served_sc_wait,  color = "tab:blue")
    plt.xlabel("time [h]")
    plt.ylabel("trip duration [s]")
    plt.xticks([i*24 for i in range(int(np.floor(time[0]/24)), int(np.ceil(time[-1]/24)) )])
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(results_dir, "moia_trip_duration_stats.png"))
    plt.close()
    return time, served_sc_wait

def plot_walking_rq_stats(bined_sc_rqs, results_dir,show = False):

    number_rqs = [x.shape[0] for x in bined_sc_rqs.values()]
    time = [x/3600.0 for x in bined_sc_rqs.keys()]

    served_sc_wait = [x[x["vehicle_id"].notna()]["walking_distance"].mean() for x in bined_sc_rqs.values()]

    plt.figure(figsize=(7,7))
    #plt.plot(time, number_rqs, 'k-', label = "Number Rqs")
    plt.plot(time, served_sc_wait, color = "tab:blue")
    plt.xlabel("time [h]")
    plt.ylabel("walking distance [m]")
    plt.xticks([i*24 for i in range(int(np.floor(time[0]/24)), int(np.ceil(time[-1]/24)) )])
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(results_dir, "moia_walk_time_stats.png"))
    plt.close()
    return time, served_sc_wait

def rq_stats_temporal(output_dir, bin_size = 900):
    sc_rq_df = pd.read_csv(os.path.join(output_dir, "1_user-stats.csv"))
    sc_rq_df = set_sc_rq_df_departure_delay_and_wt(sc_rq_df)
    binned_sc_rq_df = bin_sc_data(sc_rq_df, bin_size)
    rq_stats_temporal_dict = {}
    rq_stats_temporal_dict["plot_served_rq_stats"] = plot_served_rq_stats(binned_sc_rq_df, output_dir)
    rq_stats_temporal_dict["plot_departure_delay_rq_stats"] = plot_departure_delay_rq_stats(binned_sc_rq_df, output_dir)
    rq_stats_temporal_dict["plot_wait_rq_stats"] = plot_wait_rq_stats(binned_sc_rq_df, output_dir)
    rq_stats_temporal_dict["plot_trip_duration_rq_stats"] = plot_trip_duration_rq_stats(binned_sc_rq_df, output_dir)
    rq_stats_temporal_dict["plot_walking_rq_stats"] = plot_walking_rq_stats(binned_sc_rq_df, output_dir)
    return rq_stats_temporal_dict

### day values ### =================================================================

def get_veh_revenue_hours_driving(op_stats):
    active_veh_rev = {}
    for key in range(0,7):
        t_start = key*86400
        t_end = (key+1)*86400
        part_op_stats = op_stats[(op_stats["start_time"] >= t_start) & (op_stats["start_time"] < t_end)]
        def is_active_status(row):
            status = row["status"]
            if status in ["boarding", "route", "reposition", "waiting"]:
                return True
            else:
                return False
        part_active_op_stats = part_op_stats[part_op_stats.apply(is_active_status, axis = 1)]
        active_veh_rev[key] = (part_active_op_stats["end_time"].sum() - part_active_op_stats["start_time"].sum())/3600.0
    return pd.Series(active_veh_rev)

def get_rides(user_stats):
    rides = {}
    user_served = user_stats[user_stats["operator_id"] == 0]
    for key in range(0,7):
        t_start = key*86400
        t_end = (key+1)*86400
        user_int = user_served[(user_served["rq_time"] >= t_start) & (user_served["rq_time"] < t_end)]
        rides[key] = user_int["number_passenger"].sum()
    return pd.Series(rides)

def get_trip_pickups(user_stats):
    trip_pickups = {}
    user_served = user_stats[user_stats["operator_id"] == 0]
    for key in range(0,7):
        t_start = key*86400
        t_end = (key+1)*86400
        user_int = user_served[(user_served["rq_time"] >= t_start) & (user_served["rq_time"] < t_end)]
        trip_pickups[key] = user_int["number_passenger"].shape[0]
    return pd.Series(trip_pickups)

def get_vkm(op_stats):
    vkm = {}
    for key in range(0,7):
        t_start = key*86400
        t_end = (key+1)*86400
        op_int = op_stats[(op_stats["start_time"] >= t_start) & (op_stats["start_time"] < t_end)]
        vkm[key] = op_int["driven_distance"].sum()/1000.0
    return pd.Series(vkm)

def get_vehicle_active_trip_duration_h(op_stats):
    vehicle_active_trip_duration_h = {}
    #op_active_stats = op_stats[(op_stats["status"] != "out_of_service") & (op_stats["status"] != "charging")]
    def is_active_trip_status(row):
        status = row["status"]
        if status in ["boarding", "route", "waiting"]:
            if row["occupancy"] > 0:
                return True
            else:
                return False
        else:
            return False
    op_active_stats = op_stats[op_stats.apply(is_active_trip_status, axis = 1)]
    for key in range(0,7):
        t_start = key*86400
        t_end = (key+1)*86400
        op_int = op_active_stats[(op_active_stats["start_time"] >= t_start) & (op_active_stats["start_time"] < t_end)]
        vehicle_active_trip_duration_h[key] = (op_int["end_time"].sum() - op_int["start_time"].sum())/3600.0
    return pd.Series(vehicle_active_trip_duration_h)

def get_empty_km(op_stats):
    empty_km = {}
    op_empy_stats = op_stats[(op_stats["occupancy"] == 0)]
    for key in range(0,7):
        t_start = key*86400
        t_end = (key+1)*86400
        op_int = op_empy_stats[(op_empy_stats["start_time"] >= t_start) & (op_empy_stats["start_time"] < t_end)]
        empty_km[key] = op_int["driven_distance"].sum()/1000.0
    return pd.Series(empty_km)

def get_occupied_km(op_stats):
    occupied_km = {}
    op_occ_stats = op_stats[(op_stats["occupancy"] != 0)]
    for key in range(0,7):
        t_start = key*86400
        t_end = (key+1)*86400
        op_int = op_occ_stats[(op_occ_stats["start_time"] >= t_start) & (op_occ_stats["start_time"] < t_end)]
        occupied_km[key] = op_int["driven_distance"].sum()/1000.0
    return pd.Series(occupied_km)

def get_pkm(op_stats):
    pkm = {}
    op_occ_stats = op_stats[(op_stats["occupancy"] != 0)]
    for key in range(0,7):
        t_start = key*86400
        t_end = (key+1)*86400
        op_int = op_occ_stats[(op_occ_stats["start_time"] >= t_start) & (op_occ_stats["start_time"] < t_end)]
        pkm[key] = (op_int["driven_distance"]*op_int["occupancy"]).sum()/1000.0
    return pd.Series(pkm)

def get_departure_delays(sc_rq_df, results_dir):
    sc_departure_delays = {}
    sc_user_served = sc_rq_df[sc_rq_df["operator_id"] == 0]
    bp_data = []
    x_tick_labels = []
    for key in range(0,7):
        t_start = key*86400
        t_end = (key+1)*86400
        sc_user_int = sc_user_served[(sc_user_served["rq_time"] >= t_start) & (sc_user_served["rq_time"] < t_end)]
        sc_departure_delays[key] = sc_user_int["departure_delay"].mean()
        bp_data.append(sc_user_int[sc_user_int["departure_delay"].notnull()]["departure_delay"].values)
        x_tick_labels.append(f"{key}")

    fig, ax = plt.subplots()
    ax.set_title('Departure Delays per Day')
    ax.boxplot(bp_data)
    plt.xticks([i+1 for i in range(len(bp_data))], labels=x_tick_labels)
    plt.ylabel("Departure Delay [s]")
    plt.xlabel("Day")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "box_departure_delay.png"))
    plt.close()

    return pd.Series(sc_departure_delays)

def get_wait_times(sc_rq_df,  results_dir):
    sc_wait_time = {}
    sc_user_served = sc_rq_df[sc_rq_df["operator_id"] == 0]
    bp_data = []
    x_tick_labels = []
    for key in range(0,7):
        t_start = key*86400
        t_end = (key+1)*86400
        sc_user_int = sc_user_served[(sc_user_served["rq_time"] >= t_start) & (sc_user_served["rq_time"] < t_end)]
        sc_wait_time[key] = sc_user_int["moia_wait_time"].mean()
        bp_data.append(sc_user_int[sc_user_int["moia_wait_time"].notnull()]["moia_wait_time"].values)
        x_tick_labels.append(f"{key}")

    fig, ax = plt.subplots()
    ax.set_title('Wait Time Per Day')
    ax.boxplot(bp_data)
    plt.xticks([i+1 for i in range(len(bp_data))], labels=x_tick_labels)
    plt.ylabel("Moia Wait Time [s]")
    plt.xlabel("Day")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "box_moia_wait_time.png"))
    plt.close()

    return pd.Series(sc_wait_time)

def get_travel_times(sc_rq_df, results_dir):
    sc_travel_time = {}
    sc_user_served = sc_rq_df[sc_rq_df["operator_id"] == 0]
    bp_data = []
    x_tick_labels = []
    for key in range(0,7):
        t_start = key*86400
        t_end = (key+1)*86400
        sc_user_int = sc_user_served[(sc_user_served["rq_time"] >= t_start) & (sc_user_served["rq_time"] < t_end)]
        sc_travel_time[key] = sc_user_int["travel time"].mean()
        bp_data.append(sc_user_int[sc_user_int["travel time"].notnull()]["travel time"].values)
        x_tick_labels.append(f"{key}")

    fig, ax = plt.subplots()
    ax.set_title('Trip Duration Per Day')
    ax.boxplot(bp_data)
    plt.xticks([i+1 for i in range(len(bp_data))], labels=x_tick_labels)
    plt.ylabel("Moia Trip Duration [s]")
    plt.xlabel("Day")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "box_moia_trip_duration.png"))
    plt.close()

    return pd.Series(sc_travel_time)

def get_walking_distance(sc_rq_df, results_dir):
    sc_walking_distance = {}
    sc_user_served = sc_rq_df[sc_rq_df["operator_id"] == 0]
    bp_data = []
    x_tick_labels = []
    for key in range(0,7):
        t_start = key*86400
        t_end = (key+1)*86400
        sc_user_int = sc_user_served[(sc_user_served["rq_time"] >= t_start) & (sc_user_served["rq_time"] < t_end)]
        sc_walking_distance[key] = sc_user_int["walking_distance"].mean()
        bp_data.append(sc_user_int[sc_user_int["walking_distance"].notnull()]["walking_distance"].values)
        x_tick_labels.append(f"{key}")

    fig, ax = plt.subplots()
    ax.set_title('Walking Distance Per Day')
    ax.boxplot(bp_data)
    plt.xticks([i+1 for i in range(len(bp_data))], labels=x_tick_labels)
    plt.ylabel("Moia Walking Distance [m]")
    plt.xlabel("Day")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,  "box_moia_walking_distance.png"))
    plt.close()

    return pd.Series(sc_walking_distance)

def plot_day_data(moia_data_agg, results_dir):
    columns = "vehicle_revenue_hours;vehicle_active_trip_duration_h;rides;trip_pickups;pkm;vkm;empty_km;occupied_km;rides / vrh".split(";")
    fig, axs = plt.subplots(3,3, figsize=(18,15))
    for i, column in enumerate(columns):
        col_data = moia_data_agg[column]
        if not f"{column}" in moia_data_agg.keys():
            print(f"coudlnt find {column}! in {moia_data_agg.keys()}")
            continue
        x = [j+1 for j in range(len(col_data))]
        v = i%3
        w = int(i/3)
        axs[v][w].plot(x, col_data.values)
        axs[v][w].set(xlabel='Time [Days]', ylabel=column)
        # axs[i].xlabel("Time [Days]")
        # axs[i].ylabel(column)
        # axs[i].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "day_values_comparison.png"))
    plt.close()

def eval_day_data(output_dir, scenario_parameters, op_attributes, dir_names, op_id):
    sc_user_df = pd.read_csv(os.path.join(output_dir, "1_user-stats.csv"))
    sc_user_df = sc_user_df[sc_user_df["vehicle_id"].notna()]
    sc_user_df = set_sc_rq_df_departure_delay_and_wt(sc_user_df)
    sc_op_df = pd.read_csv(os.path.join(output_dir, "2-0_op-stats.csv"))
    results_dir = output_dir
    eval_day_data_dict = {}
    sc_walking_distance = get_walking_distance(sc_user_df, results_dir)
    eval_day_data_dict["walking_distance"] = sc_walking_distance
    sc_trip_duration = get_travel_times(sc_user_df, results_dir)
    eval_day_data_dict["trip_duration"] = sc_trip_duration
    sc_departure_delays = get_departure_delays(sc_user_df, results_dir)
    eval_day_data_dict["departure_delay"] = sc_departure_delays
    sc_wait_times = get_wait_times(sc_user_df, results_dir)
    eval_day_data_dict["wait_time"] = sc_wait_times
    eval_day_data_dict["vkm"] = get_vkm(sc_op_df)
    eval_day_data_dict["vehicle_active_trip_duration_h"] = get_vehicle_active_trip_duration_h(sc_op_df)
    eval_day_data_dict["empty_km"] = get_empty_km(sc_op_df)
    eval_day_data_dict["occupied_km"] = get_occupied_km(sc_op_df)
    eval_day_data_dict["pkm"] = get_pkm(sc_op_df)
    eval_day_data_dict["rides"] = get_rides(sc_user_df)
    eval_day_data_dict["trip_pickups"] = get_trip_pickups(sc_user_df)
    eval_day_data_dict["vehicle_revenue_hours"] = get_veh_revenue_hours_driving(sc_op_df)
    eval_day_data_dict["rides / vrh"] = eval_day_data_dict["rides"]/eval_day_data_dict["vehicle_revenue_hours"]
    plot_day_data(eval_day_data_dict, results_dir)
    return eval_day_data_dict

### Whole Week Data
def evaluate_whole_week_data(output_dir, scenario_parameters, op_attributes, dir_names, op_id):   
    standard_eval_df = pd.read_csv(os.path.join(output_dir, "standard_eval.csv"), index_col=0)
    standard_eval_dict = standard_eval_df["MoD_0"].to_dict()
    sc_rq_df = pd.read_csv(os.path.join(output_dir, "1_user-stats.csv"))
    sc_rq_df = sc_rq_df[sc_rq_df["vehicle_id"].notna()]
    sc_rq_df = set_sc_rq_df_departure_delay_and_wt(sc_rq_df)

    sc_op_df = pd.read_csv(os.path.join(output_dir, "2-0_op-stats.csv"))
    pkm = (sc_op_df["driven_distance"]*sc_op_df["occupancy"]).sum()/1000.0


    sim_week_data = {
        "abs_served_customers" : standard_eval_dict["number travelers"],
        "abs_served_requests" : standard_eval_dict["number users"],
        "walking_distance" : standard_eval_dict["avg_walk_dist_start"] + standard_eval_dict["avg_walk_dist_end"]
    }

    sim_week_data["avg_rel_detour"] = standard_eval_dict["bp_rel_detour"]
    sim_week_data["avg_abs_detour"] = standard_eval_dict["bp_abs_detour"]
    sim_week_data["avg_occ_rq"] = standard_eval_dict["occupancy rq"]
    sim_week_data["avg_occ_per"] = standard_eval_dict["occupancy"]
    sim_week_data["frac_empty_vkm"] = standard_eval_dict[r"% empty vkm"]
    sim_week_data["vkm"] = standard_eval_dict[r"total vkm"]
    sim_week_data["empty_km"] = standard_eval_dict[r"% empty vkm"]/100.0 * standard_eval_dict[r"total vkm"]
    sim_week_data["pkm"] = pkm
    sim_week_data["occupied_km"] = (1.0 - standard_eval_dict[r"% empty vkm"]/100.0) * standard_eval_dict[r"total vkm"]
    sim_week_data["trip_duration"] = standard_eval_dict["travel time"]
    sim_week_data["departure_delay_s"] = sc_rq_df["departure_delay"].mean()
    sim_week_data["waiting_time_s"] = sc_rq_df["moia_wait_time"].mean()
    sim_week_data["whole_wait_time_s"] = sim_week_data["departure_delay_s"] + sim_week_data["waiting_time_s"]

    sim_series = pd.Series(sim_week_data, name="sim")
    df = pd.DataFrame([sim_series])
    df.transpose().to_csv(os.path.join(output_dir, "additional_whole_week_results.csv"))

### dynamic fleet ctrl data
def active_fleet_size_over_time(output_dir, scenario_parameters, op_attributes, dir_names, op_id):
    if op_attributes.get(G_OP_ACT_FLEET_SIZE):
        act_veh_df = pd.read_csv(os.path.join(dir_names[G_DIR_FCTRL], "elastic_fleet_size", op_attributes[G_OP_ACT_FLEET_SIZE]), index_col=0)
        act_veh_series = act_veh_df["share_active_fleet_size"]
        ts = [i/3600.0 for i in act_veh_series.keys()]
        N_vehicles = sum(op_attributes[G_OP_FLEET].values())
        vehs = [k * N_vehicles for k in act_veh_series.values]
    elif op_attributes.get(G_OP_DYFS_TARGET_UTIL):
        dyn_op_file = pd.read_csv(os.path.join(output_dir, "3-0_op-dyn_atts.csv"), index_col=0)
        act_veh_series = dyn_op_file["active vehicles"]
        ts = [i/3600.0 for i in act_veh_series.keys()]
        vehs = act_veh_series.values
    else:
        print("constant fleet size!")
        return [], []
    plt.figure(figsize=(7,7))
    plt.plot(ts, vehs)
    plt.xlabel("Time [h]")
    plt.ylabel("Number Active Vehicles")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "active_vehicles.png"))
    plt.close()
    return ts, vehs

def dynamic_pricing_factor_over_time(output_dir, scenario_parameters, op_attributes, dir_names, op_id):
    if op_attributes.get(G_OP_ELA_PRICE):
        ela_price_df = pd.read_csv(os.path.join(dir_names[G_DIR_FCTRL], "elastic_pricing", op_attributes[G_OP_ELA_PRICE]), index_col=0)
        ela_price_series = ela_price_df["distance_fare_factor"]
        ts = [i/3600.0 for i in ela_price_series.keys()]
        price_factor = [k for k in ela_price_series.values]
    elif op_attributes.get(G_OP_UTIL_SURGE):
        dyn_op_file = pd.read_csv(os.path.join(output_dir, "3-0_op-dyn_atts.csv"), index_col=0)
        ela_price_series = dyn_op_file["distance fare factor"]
        ts = [i/3600.0 for i in ela_price_series.keys()]
        price_factor = ela_price_series.values
    else:
        print("constant fare!")
        return [], []
    plt.figure(figsize=(7,7))
    plt.plot(ts, price_factor)
    plt.xlabel("Time [h]")
    plt.ylabel("Distance Fare Factor")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distance_fare_factor.png"))
    plt.close()
    return ts, price_factor

def evaluate_dyn_fleetctrl_data(output_dir, scenario_parameters, op_attributes, dir_names, op_id):
    dyn_fleetctrl_dict = {}
    dyn_fleetctrl_dict["active_fleet_size_over_time"] = active_fleet_size_over_time(output_dir, scenario_parameters, op_attributes, dir_names, op_id)
    dyn_fleetctrl_dict["dynamic_pricing_factor_over_time"] = dynamic_pricing_factor_over_time(output_dir, scenario_parameters, op_attributes, dir_names, op_id)
    return dyn_fleetctrl_dict
#======

def temporal_evaluation(output_dir, print_comments=False, dir_names_in = {}):
    """This function runs a standard evaluation over a scenario output directory.

    :param output_dir: scenario output directory
    :param print_comments: print some comments about status in between
    """
    scenario_parameters, list_operator_attributes, dir_names = load_scenario_inputs(output_dir)
    if dir_names_in:
        dir_names = dir_names_in
    plot_dict = {}
    op_id, op_attributes = 0, list_operator_attributes[0]
    print(" ... eval dyn fleetctrl data")
    plot_dict["dyn_fleetctrl_data"] = evaluate_dyn_fleetctrl_data(output_dir, scenario_parameters, op_attributes, dir_names, op_id)
    print(" ... eval week data")
    evaluate_whole_week_data(output_dir, scenario_parameters, op_attributes, dir_names, op_id)
    print(" ... eval day data")
    plot_dict["eval_day_data"] = eval_day_data(output_dir, scenario_parameters, op_attributes, dir_names, op_id)
    print(" ... create fleet stats temporal")
    occupancy_over_time_rel(output_dir, scenario_parameters, op_attributes, dir_names, op_id, bin_size=DEF_TEMPORAL_RESOLUTION, show=False)
    plot_dict["temporal_plots"] = temporal_plots(output_dir, scenario_parameters, op_attributes, dir_names, op_id, show=False)
    occupancy_over_time(output_dir, scenario_parameters, op_attributes, dir_names, op_id, bin_size=DEF_TEMPORAL_RESOLUTION, show=False)
    vehicle_stats_over_time(output_dir, scenario_parameters, op_attributes, dir_names, op_id, bin_size=DEF_TEMPORAL_RESOLUTION, show=False)
    print(" ... create rq stats temporal")
    plot_dict["rq_stats_temporal"] = rq_stats_temporal(output_dir, bin_size=DEF_TEMPORAL_RESOLUTION)
    with open(os.path.join(output_dir, "plot_dict.pickle"), "wb") as f:
        print("... write plot dict file")
        pickle.dump(plot_dict, f)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        if os.path.isfile(os.path.join(input_dir, "00_config.json")):
            print(" ... eval ", input_dir)
            temporal_evaluation(input_dir)
        else:
            results_dir = os.path.join(input_dir, "results")
            print("... eval study ", input_dir)
            for sc in os.listdir(results_dir):
                if sc == "_archiv" or not os.path.isfile(os.path.join(results_dir, sc, "standard_eval.csv")):
                    continue
                print(" ... eval ", sc)
                temporal_evaluation(os.path.join(results_dir, sc))
    



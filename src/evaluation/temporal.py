import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(MAIN_DIR)
from src.evaluation.standard import decode_offer_str, load_scenario_inputs, get_directory_dict,\
                                    read_op_output_file, read_user_output_file
from src.misc.globals import *

# plt.style.use("seaborn-whitegrid")
# matplotlib.rcParams['ps.useafm'] = True
# matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True

# assuming temporal resolution of seconds in output files
MIN = 60
HOUR = 3600
DAY = 24 * 3600
DEF_TEMPORAL_RESOLUTION = 15*MIN
DEF_SMALLER_TEMPORAL_RESOLUTION = 2*MIN # for stacked plots

# styling figures
FIG_SIZE = (6,4)
LABEL_FONT_SIZE = 14
#LIST_COLORS = [x for x in plt.rcParams['axes.prop_cycle'].by_key()['color']]
LIST_COLORS = ['#DDDBD3', '#E6AA33', '#BC8820', '#4365FF', '#0532FF', '#8C9296', '#000C15'] # MOIA colors
N_COLORS = len(LIST_COLORS)

# other
SMALL_VALUE = 0.000001


def _bin_operator_stats(op_df, interval_list):
    """ this function is used to bin the operator stats into time bins
    :param op_df: operator dataframe
    :param interval_list: list of time steps to bin into
    :return: dict {time_bin} -> sub_df of op_df within this time bin (with weight of fraction of task is within this bin)
    """

    def _weight_vrl_interval(veh_stat_row, interval_start, interval_end):
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

    binned_vehicle_stats = {}
    interval_list.append(float("inf"))
    for i in range(1, len(interval_list)):
        sb = interval_list[i-1]
        eb = interval_list[i]
        veh_stats = op_df[(( (op_df[G_VR_LEG_START_TIME] >= sb) & (op_df[G_VR_LEG_START_TIME] <= eb) )|( (op_df[G_VR_LEG_END_TIME] >= sb) & (op_df[G_VR_LEG_END_TIME] <= eb) )|( (op_df[G_VR_LEG_START_TIME] < sb) & (op_df[G_VR_LEG_END_TIME] > eb) ) )]
        if veh_stats.shape[0] > 0:
            veh_stats["interval_weight"] = veh_stats.apply(_weight_vrl_interval, axis = 1, args = (sb, eb))
            binned_vehicle_stats[sb] = veh_stats.copy()
    return binned_vehicle_stats


def _bin_served_user_stats(user_df, interval_list):
    """ this function is used to bin the (served!) users stats into time bins; users are collect to bins corresponding to their request time and drop off time and are weighted according to the fraction of time they are active in this bin
    :param user_df: user dataframe
    :param interval_list: list of time steps to bin into
    :return: dict {time_bin} -> user_df of user_df within this time bin
    """

    def _weight_user_interval(user_row, interval_start, interval_end):
        st = user_row[G_RQ_TIME]
        et = user_row[G_RQ_DO]
        if st > et:
            return 0
        elif st > interval_end:
            return 0
        elif et < interval_start:
            return 0
        else:
            return (min(et, interval_end) - max(st, interval_start))/(interval_end - interval_start)

    binned_user_stats = {}
    interval_list.append(float("inf"))
    for i in range(1, len(interval_list)):
        sb = interval_list[i-1]
        eb = interval_list[i]
        user_part_stats = user_df[(( (user_df[G_RQ_TIME] >= sb) & (user_df[G_RQ_DO] <= eb) )|( (user_df[G_RQ_DO] >= sb) & (user_df[G_RQ_DO] <= eb) )|( (user_df[G_RQ_TIME] < sb) & (user_df[G_RQ_DO] > eb) ) )]
        if user_part_stats.shape[0] > 0:
            user_part_stats["interval_weight"] = user_part_stats.apply(_weight_user_interval, axis = 1, args = (sb, eb))
            binned_user_stats[sb] = user_part_stats.copy()
    return binned_user_stats


def avg_occ_binned(binned_operator_stats, output_dir, op_id, show=True):
    """ this function creats plots for the average vehicle occupancy over time
    :param binned_operator_stats: return of method "_bin_operator_stats"
    :param output_dir: output directory
    :param op_id: operator id
    :param show: if True, plot is directly shown but not saved
    :return: tuple of time_values, occ_values (of customers), (occ_values (for requests) if a difference between these values is found)
    """
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
    differs = False
    for t, binned_stats_df in binned_operator_stats.items():
        ts.append(t/3600.0)
        occs.append(last)
        occs_rq.append(last_rq)
        last_rq = binned_stats_df.apply(weight_ob_rq, axis=1).sum() / (binned_stats_df[G_VR_LEG_DISTANCE]*binned_stats_df["interval_weight"]).sum()
        last = binned_stats_df.apply(weight_ob_pax, axis=1).sum() / (binned_stats_df[G_VR_LEG_DISTANCE]*binned_stats_df["interval_weight"]).sum()
        if abs(last_rq - last) > SMALL_VALUE:
            differs = True
        ts.append(t/3600.0)
        occs.append(last)
        occs_rq.append(last_rq)
    plt.figure(figsize=(7,7))
    plt.plot(ts, occs, label = "people", color = LIST_COLORS[0])
    if differs:
        plt.plot(ts, occs_rq, label = "requests", color = LIST_COLORS[1])
    plt.xlabel("Time [h]")
    plt.ylabel("Avg Occupancy")
    if differs:
        plt.legend()
    plt.xticks([i*24 for i in range(int(np.floor(ts[0]/24)), int(np.ceil(ts[-1]/24)) )])
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(output_dir, "temporal_occupancy_op_{}.png".format(op_id)))
    plt.close()
    if differs:
        return ts, occs, occs_rq
    else:
        return ts, occs, None


def avg_util_binned(binned_operator_stats, output_dir, op_id, n_vehicles, show=True):
    """ this function creats plots for the average vehicle utilization over time (states that count for utilization
    defined in globals)
    :param binned_operator_stats: return of method "_bin_operator_stats"
    :param output_dir: output directory
    :param op_id: operator id
    :param n_vehicles: number of vehicles of operator
    :param show: if True, plot is directly shown but not saved
    :return: tuple of time_values, utilization values
    """
    ts = []
    util = []
    last = 0
    bins = list(binned_operator_stats.keys())
    util_states = [x.display_name for x in G_REVENUE_STATUS]
    for t, binned_stats_df in binned_operator_stats.items():
        ts.append(t/3600.0)
        util.append(last)
        last = binned_stats_df[binned_stats_df[G_VR_STATUS].isin(util_states)]["interval_weight"].sum() * 100.0 / n_vehicles
        # last = (binned_stats_df["interval_weight"]*(binned_stats_df[G_VR_LEG_END_TIME] - binned_stats_df[G_VR_LEG_START_TIME])).sum() * 100.0 / 250 / delta / binned_stats_df.shape[0]
        ts.append(t/3600.0)
        util.append(last)

    plt.figure(figsize=(7,7))
    plt.plot(ts, util, color = LIST_COLORS[0])
    plt.xlabel("Time [h]")
    plt.ylabel("Utilization [%]")
    plt.xticks([i*24 for i in range(int(np.floor(ts[0]/24)), int(np.ceil(ts[-1]/24)) )])
    plt.ylim(0, 100)
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(output_dir, "temporal_util_op_{}.png".format(op_id)))
    plt.close()
    return ts, util

def avg_fleet_km_binned(binned_operator_stats, output_dir, op_id, show = True):
    """ this function creats plots for the average fleet km per bin over time
    :param binned_operator_stats: return of method "_bin_operator_stats"
    :param output_dir: output directory
    :param op_id: operator id
    :param show: if True, plot is directly shown but not saved
    :return: tuple of time_values, utilization values
    """
    ts = []
    driven_distances = []
    last = 0
    bins = list(binned_operator_stats.keys())
    bin_size = bins[1] - bins[0]
    driving_states = [x.display_name for x in G_DRIVING_STATUS]
    for t, binned_stats_df in binned_operator_stats.items():
        ts.append(t/3600.0)
        driven_distances.append(last)
        dr = binned_stats_df[binned_stats_df[G_VR_STATUS].isin(driving_states)]
        last = (dr["interval_weight"]*dr[G_VR_LEG_DISTANCE]).sum()/1000.0
        ts.append(t/3600.0)
        driven_distances.append(last)

    plt.figure(figsize=(7,7))
    plt.plot(ts, driven_distances, color = LIST_COLORS[0])
    plt.xlabel("Time [h]")
    plt.ylabel(f"Fleet KM per {int(bin_size/MIN)} min")
    plt.xticks([i*24 for i in range(int(np.floor(ts[0]/24)), int(np.ceil(ts[-1]/24)) )])
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(output_dir, "temporal_fleet_km_op_{}.png".format(op_id)))
    plt.close()
    return ts, driven_distances


def avg_fleet_driving_speeds_binned(binned_operator_stats, output_dir, op_id, show=True):
    """ this function creats plots for the average fleet spped per bin over time
    it differs between the speed of vehicles actively driving
    and the effective speed including other active revenue tasks (i.e boarding and waiting)
    :param binned_operator_stats: return of method "_bin_operator_stats"
    :param output_dir: output directory
    :param op_id: operator id
    :param show: if True, plot is directly shown but not saved
    :return: tuple of time_values, driven speed, revenue speed
    """
    ts = []
    driven_speed = []
    revenue_speed = []
    last_dr = 0
    last_rev = 0
    bins = list(binned_operator_stats.keys())
    bin_size = bins[1] - bins[0]
    driving_states = [x.display_name for x in G_DRIVING_STATUS]
    util_states = [x.display_name for x in G_REVENUE_STATUS]
    for t, binned_stats_df in binned_operator_stats.items():
        ts.append(t/3600.0)
        driven_speed.append(last_dr)
        revenue_speed.append(last_rev)

        dr = binned_stats_df[binned_stats_df[G_VR_STATUS].isin(driving_states)]
        dr_driven = (dr[G_VR_LEG_DISTANCE]*dr["interval_weight"]).sum()
        dr_time = ((dr[G_VR_LEG_END_TIME] - dr[G_VR_LEG_START_TIME])*dr["interval_weight"]).sum()
        if dr_time > 0:
            last_dr = dr_driven/dr_time*3.6
        else:
            last_dr = 0
        rev = binned_stats_df[binned_stats_df[G_VR_STATUS].isin(util_states)]
        rev_driven = (rev[G_VR_LEG_DISTANCE]*rev["interval_weight"]).sum()
        rev_time = ((rev[G_VR_LEG_END_TIME] - rev[G_VR_LEG_START_TIME])*rev["interval_weight"]).sum()
        if rev_time > 0:
            last_rev = rev_driven/rev_time*3.6
        else:
            last_rev = 0

        ts.append(t/3600.0)
        driven_speed.append(last_dr)
        revenue_speed.append(last_rev)

    min_y = min([x for x in revenue_speed if x != 0])

    plt.figure(figsize=(7,7))
    plt.plot(ts, driven_speed, color = LIST_COLORS[0], label = "Driven Speed")
    plt.plot(ts, revenue_speed, color = LIST_COLORS[1], label = "Revenue Speed")
    plt.legend()
    plt.ylim(bottom=min_y)
    plt.xlabel("Time [h]")
    plt.ylabel(f"Fleet Speeds [km/h]")
    plt.xticks([i*24 for i in range(int(np.floor(ts[0]/24)), int(np.ceil(ts[-1]/24)) )])
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(output_dir, "temporal_fleet_speeds_op_{}.png".format(op_id)))
    plt.close()
    return ts, driven_speed, revenue_speed


def avg_revenue_hours_binned(binned_operator_stats, output_dir, op_id, n_vehicles, show=True):
    """ this function creats plots for the average vehicle revenue hours over time (states that count for vehicle
    revenue hours defined in globals).
    revenue hours is defined as the sum of times of all vehicles which are in this states
    :param binned_operator_stats: return of method "_bin_operator_stats"
    :param output_dir: output directory
    :param op_id: operator id
    :param n_vehicles: number of vehicles of operator
    :param show: if True, plot is directly shown but not saved
    :return: tuple of time_values, vehicle revenue time values
    """
    ts = []
    vrhs = []
    last = 0
    bins = list(binned_operator_stats.keys())
    bin_size = bins[1] - bins[0]
    util_states = [x.display_name for x in G_REVENUE_STATUS]
    for t, binned_stats_df in binned_operator_stats.items():
        ts.append(t/3600.0)
        vrhs.append(last)
        revenue_entries = binned_stats_df[binned_stats_df[G_VR_STATUS].isin(util_states)]
        last = revenue_entries["interval_weight"].sum() * bin_size / 3600.0
        ts.append(t/3600.0)
        vrhs.append(last)

    plt.figure(figsize=(7,7))
    plt.plot(ts, vrhs, color = LIST_COLORS[0])
    plt.xlabel("Time [h]")
    plt.ylabel("Vehicle Revenue Hours [h]")
    plt.xticks([i*24 for i in range(int(np.floor(ts[0]/24)), int(np.ceil(ts[-1]/24)) )])
    plt.ylim(0, n_vehicles * bin_size/3600.0)
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(output_dir, "temporal_VRH_op_{}.png".format(op_id)))
    plt.close()
    return ts, vrhs

def avg_active_customers_binned(binned_served_customer_stats, output_dir, op_id, show = True):
    """ this function creats plots for the average number of currently active requests
    revenue hours is defined as the sum of times of all vehicles which are in this states
    :param binned_served_customer_stats: return of method "_bin_served_user_stats"
    :param output_dir: output directory
    :param op_id: operator id
    :param show: if True, plot is directly shown but not saved
    :return: tuple of time_values, active customers, (active requests; or None if no difference is found between customers and requests)
    """
    ts = []
    act_cust = []
    act_req = []
    last_cust = 0
    last_rq = 0
    bins = list(binned_served_customer_stats.keys())
    bin_size = bins[1] - bins[0]
    differs = False
    for t, binned_customer_df in binned_served_customer_stats.items():
        ts.append(t/3600.0)
        act_cust.append(last_cust)
        act_req.append(last_rq)
        avg_active_requests = binned_customer_df["interval_weight"].sum()
        avg_active_customers = (binned_customer_df["number_passenger"]*binned_customer_df["interval_weight"]).sum()
        last_cust = avg_active_customers
        last_rq = avg_active_requests
        if abs(last_cust - last_rq) > SMALL_VALUE:
            differs = True
        ts.append(t/3600.0)
        act_cust.append(last_cust)
        act_req.append(last_rq)

    plt.figure(figsize=(7,7))
    plt.plot(ts, act_cust, label = "people", color = LIST_COLORS[0])
    if differs:
        plt.plot(ts, act_req, label= "requests", color = LIST_COLORS[1])
    plt.xlabel("Time [h]")
    plt.ylabel("Number of Active Customers")
    plt.xticks([i*24 for i in range(int(np.floor(ts[0]/24)), int(np.ceil(ts[-1]/24)) )])
    if differs:
        plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(output_dir, "temporal_active_customers_op_{}.png".format(op_id)))
    plt.close()
    if differs:
        return ts, act_cust, act_req
    else:
        return ts, act_cust, None


def avg_customers_per_vehicle_revenue_hous_binned(binned_served_customer_stats, binned_operator_stats, output_dir,
                                                  op_id, show=True):
    """ this function creats plots for the average customers per vehicle revenue hours over time
    (states that count for vehicle revenue hours defined in globals)
    revenue hours is defined as the sum of times of all vehicles which are in this states
    :param binned_served_customer_stats: return of method "_bin_served_user_stats"
    :param binned_operator_stats: return of method "_bin_operator_stats"
    :param output_dir: output directory
    :param op_id: operator id
    :param show: if True, plot is directly shown but not saved
    :return: tuple of time_values, customers per vehicle revenue hours, (active requests per vehicle revenue hours;
             or None if no difference is found between customers and requests)
    """
    ts = []
    act_cust_pvrh = []
    act_req_pvrh = []
    last_cust_pvrh = 0
    last_rq_pvrh = 0
    bins = list(binned_served_customer_stats.keys())
    bin_size = bins[1] - bins[0]
    differs = False
    util_states = [x.display_name for x in G_REVENUE_STATUS]
    def get_frac_active_cust_time(row):
        rq_time = row[G_RQ_DO] - row[G_RQ_TIME]
        return row["interval_weight"] * bin_size/rq_time
    for t, binned_customer_df in binned_served_customer_stats.items():
        ts.append(t/3600.0)
        act_cust_pvrh.append(last_cust_pvrh)
        act_req_pvrh.append(last_rq_pvrh)
        binned_stats_df = binned_operator_stats.get(t)
        if binned_stats_df is not None and binned_stats_df.shape[0] > 0:
            binned_customer_df["frac_active_time"] = binned_customer_df.apply(get_frac_active_cust_time, axis=1)
            avg_active_requests = binned_customer_df["frac_active_time"].sum()
            avg_active_customers = (binned_customer_df["number_passenger"]*binned_customer_df["frac_active_time"]).sum()
            revenue_entries = binned_stats_df[binned_stats_df[G_VR_STATUS].isin(util_states)]
            rev_hours = revenue_entries["interval_weight"].sum() * bin_size / 3600.0
            if rev_hours == 0:
                last_cust_pvrh = 0
                last_rq_pvrh = 0
            else:
                last_cust_pvrh = avg_active_customers/rev_hours
                last_rq_pvrh = avg_active_requests/rev_hours
        else:
            last_cust_pvrh = 0
            last_rq_pvrh = 0
        if abs(last_cust_pvrh - last_rq_pvrh) > SMALL_VALUE:
            differs = True
        ts.append(t/3600.0)
        act_cust_pvrh.append(last_cust_pvrh)
        act_req_pvrh.append(last_rq_pvrh)

    plt.figure(figsize=(7,7))
    plt.plot(ts, act_cust_pvrh, label = "people", color = LIST_COLORS[0])
    if differs:
        plt.plot(ts, act_req_pvrh, label= "requests", color = LIST_COLORS[1])
    plt.xlabel("Time [h]")
    plt.ylabel("Customers per Vehicle Revenue Hours")
    plt.xticks([i*24 for i in range(int(np.floor(ts[0]/24)), int(np.ceil(ts[-1]/24)) )])
    if differs:
        plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(output_dir, "temporal_customers_per_vehicle_revenue_hours_op_{}.png".format(op_id)))
    plt.close()
    if differs:
        return ts, last_cust_pvrh, act_req_pvrh
    else:
        return ts, last_cust_pvrh, None


def temporal_operator_plots(output_dir, op_id, show=False, evaluation_start_time=None, evaluation_end_time=None,
                            print_comments=False):
    """ this method creates all time line plots and returns dictionary of x-y values to use them to create plots for
    direct scenario comparisons
    :param output_dir: directory of scenario result files
    :param op_id: corresponding operator id to evaluate
    :param show: if True plots are directly shown, if false they are stored in the output dir
    :param evaluation_start_time: if given all data entries before this time [s] are discarded
    :param evaluation_end_time: if geven, all data entries after this time [s] are discarded
    :param print_comments: if True some prints are given to show what the evaluation is currently doing
    :return: dictionary {function_name} -> tuple of all plot values
    """
    if print_comments:
        print(f" ... start evaluating temporal stats for op {op_id}")
    # scenario_parameters, op_attributes, dir_names, 
    scenario_parameters, list_operator_attributes, dir_names = load_scenario_inputs(output_dir)
    if not os.path.isdir(dir_names[G_DIR_MAIN]):
        dir_names = get_directory_dict(scenario_parameters)
    op_attributes = list_operator_attributes[op_id]

    # evaluation interval
    if evaluation_start_time is None and scenario_parameters.get(G_EVAL_INT_START) is not None:
        evaluation_start_time = int(scenario_parameters[G_EVAL_INT_START])
    if evaluation_end_time is None and scenario_parameters.get(G_EVAL_INT_END) is not None:
        evaluation_end_time = int(scenario_parameters[G_EVAL_INT_END])
    if print_comments:
        print(f" ... read stats for op {op_id}")
    op_df = read_op_output_file(output_dir, op_id, evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time)
    if print_comments:
        print(f" ... read user stats for op {op_id}")
    user_df = read_user_output_file(output_dir, evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time)
    op_user_df = user_df[user_df["operator_id"] == op_id]

    # create bins and bin vehicle stats
    if print_comments:
        print(f" ... create bins for op {op_id}")
    start_time = scenario_parameters[G_SIM_START_TIME]
    if evaluation_start_time is not None:
        start_time = evaluation_start_time
    end_time = max(op_df[G_VR_LEG_END_TIME].values)
    bin_intervals = []
    i = 0
    while i * DEF_TEMPORAL_RESOLUTION + start_time <= end_time:
        bin_intervals.append(i * DEF_TEMPORAL_RESOLUTION + start_time)
        i+=1
    binned_operator_stats = _bin_operator_stats(op_df, bin_intervals)
    #  bin users stats
    binned_users_stats = _bin_served_user_stats(op_user_df, bin_intervals)

    # create all plots
    n_vehicles = len(op_df[G_V_VID].unique())
    temporal_plots_dict = {}
    if print_comments:
        print(f" ... evaluate utilization for op {op_id}")
    temporal_plots_dict["avg_util_binned"] = avg_util_binned(binned_operator_stats, output_dir, op_id, n_vehicles, show = show)
    if print_comments:
        print(f" ... evaluate fleet speeds for op {op_id}")
    temporal_plots_dict["avg_fleet_driving_speeds_binned"] = avg_fleet_driving_speeds_binned(binned_operator_stats, output_dir, op_id, show = show)
    if print_comments:
        print(f" ... evaluate fleet km for op {op_id}")
    temporal_plots_dict["avg_fleet_km_binned"] = avg_fleet_km_binned(binned_operator_stats, output_dir, op_id, show = show)
    if print_comments:
        print(f" ... evaluate occupancy for op {op_id}")
    temporal_plots_dict["avg_occ_binned"] = avg_occ_binned(binned_operator_stats, output_dir, op_id, show = show)
    if print_comments:
        print(f" ... evaluate revenue hours for op {op_id}")
    temporal_plots_dict["avg_revenue_hours_binned"] = avg_revenue_hours_binned(binned_operator_stats, output_dir, op_id, n_vehicles, show = show)
    if print_comments:
        print(f" ... evaluate active customers for op {op_id}")
    temporal_plots_dict["avg_active_customers_binned"] = avg_active_customers_binned(binned_users_stats, output_dir, op_id, show = show)
    if print_comments:
        print(f" ... evaluate customers per vehicle revenue hours for op {op_id}")
    temporal_plots_dict["avg_customers_per_vehicle_revenue_hous_binned"] = avg_customers_per_vehicle_revenue_hous_binned(binned_users_stats, binned_operator_stats, output_dir, op_id, show = show)
    return temporal_plots_dict


# =====================================================================================================================#
# ================ STACKED VEHICLE STATE PLOTS ========================================================================#
# =====================================================================================================================#

def _load_op_stats_and_infer_idle_states(output_dir, scenario_parameters, op_id,
                                         evaluation_start_time=None, evaluation_end_time=None):
    """ this function loads the operator infos and include idle states within (only with information needed to plot the
    vehicle states over time). jobs are cut if the exceed evaluation start or end time if given
    :param ouput_dir: directory of scenario results
    :param scenario_parameters: scenario parameter dictionary
    :param evaluation_start_time: start time of the evaluation time interval
    :param evaluation_end_time: end time of the evaluation time interval
    :return: operator dataframe with vehicle idle states
    """
    op_stats = os.path.join(output_dir, "2-{}_op-stats.csv".format(int(op_id)))
    op_df = pd.read_csv(op_stats)
    #insert idle
    start_time = scenario_parameters[G_SIM_START_TIME]
    end_time = max(op_df[G_VR_LEG_END_TIME].values)
    add_df_list = []
    to_drop = []    # some jobs dont finish correctly (charging and out_of_service doubled) # TODO #
    for vid, veh_stats in op_df.groupby(G_V_VID):
        veh_stats.sort_values(by=G_VR_LEG_END_TIME, inplace = True)
        last_end_time = start_time
        c = 0
        last_entry = None
        for _, entry in veh_stats.iterrows():
            if c == 0:
                if entry[G_VR_LEG_START_TIME] > start_time:
                    d = {}
                    for key, val in entry.items():
                        if key == G_VR_LEG_START_TIME:
                            d[G_VR_LEG_START_TIME] = start_time
                        elif key == G_VR_LEG_END_TIME:
                            d[G_VR_LEG_END_TIME] = entry[G_VR_LEG_START_TIME]
                        elif key == G_VR_STATUS:
                            d[G_VR_STATUS] = "idle"
                        elif key == G_VR_LEG_END_SOC:
                            d[G_VR_LEG_END_SOC] = entry[G_VR_LEG_START_SOC]
                        elif key == G_VR_LEG_START_SOC:
                            d[G_VR_LEG_START_SOC] = entry[G_VR_LEG_START_SOC]
                        else:
                            d[key] = val
                    add_df_list.append(d)
                last_end_time = entry[G_VR_LEG_END_TIME]
                c+=1
                continue
            c+=1
            if entry[G_VR_LEG_START_TIME] > last_end_time:
                d = {}
                for key, val in entry.items():
                    if key == G_VR_LEG_START_TIME:
                        d[G_VR_LEG_START_TIME] = last_end_time
                    elif key == G_VR_LEG_END_TIME:
                        d[G_VR_LEG_END_TIME] = entry[G_VR_LEG_START_TIME]
                    elif key == G_VR_STATUS:
                        if entry[G_VR_STATUS] == "out_of_service" or entry[G_VR_STATUS] == "charging" or entry[G_VR_STATUS] == "to_depot":
                            d[G_VR_STATUS] = "idle"
                        else:
                            d[G_VR_STATUS] = "idle"
                    elif key == G_VR_LEG_END_SOC:
                        d[G_VR_LEG_END_SOC] = entry[G_VR_LEG_START_SOC]
                    else:
                        d[key] = val
                add_df_list.append(d)
            last_end_time = entry[G_VR_LEG_END_TIME]
            last_entry = entry
        if last_end_time < end_time and last_entry is not None:
            d = {}
            for key, val in last_entry.items():
                if key == G_VR_LEG_START_TIME:
                    d[G_VR_LEG_START_TIME] = last_end_time
                elif key == G_VR_LEG_END_TIME:
                    d[G_VR_LEG_END_TIME] = end_time
                elif key == G_VR_STATUS:
                    d[G_VR_STATUS] = "idle"
                elif key == G_VR_LEG_END_SOC:
                    d[G_VR_LEG_END_SOC] = last_entry[G_VR_LEG_START_SOC]
                else:
                    d[key] = val
            add_df_list.append(d)
    idle_df = pd.DataFrame(add_df_list)
    op_df.drop(to_drop, inplace=True)
    op_df = pd.concat([op_df, idle_df], axis = 0, ignore_index = True)
    op_df.sort_values(by=G_VR_LEG_END_TIME, inplace = True)
    # cut times
    if evaluation_start_time is not None:
        op_df = op_df[op_df[G_VR_LEG_END_TIME] > evaluation_start_time]
        def set_start_time(row):
            return max(evaluation_start_time, row[G_VR_LEG_START_TIME])
        op_df[G_VR_LEG_START_TIME] = op_df.apply(set_start_time, axis=1)
    if evaluation_end_time is not None:
        op_df = op_df[op_df[G_VR_LEG_START_TIME] < evaluation_end_time]
        def set_end_time(row):
            return min(evaluation_end_time, row[G_VR_LEG_END_TIME])
        op_df[G_VR_LEG_END_TIME] = op_df.apply(set_end_time, axis=1)
    return op_df


def vehicle_stats_over_time(output_dir, scenario_parameters, operator_attributes, dir_names, op_id, show=False,
                            bin_size=DEF_TEMPORAL_RESOLUTION, evaluation_start_time=None, evaluation_end_time=None):
    """ this function creates a plot of the different vehicle states over time 
    :param ouput_dir: directory of scenario results
    :param scenario_parameters: scenario parameter dictionary
    :param operator_attributes: op atts dictionary
    :param dir_names: dir name dictionary
    :param op_id: operator id
    :param show: if True, plot is shown but not stored; if false otherwise
    :param bin_size: bin size of plot
    :param evaluation_start_time: start time of the evaluation time interval
    :param evaluation_end_time: end time of the evaluation time interval
    """
    op_df = _load_op_stats_and_infer_idle_states(output_dir, scenario_parameters, op_id,
                                                 evaluation_start_time=evaluation_start_time,
                                                 evaluation_end_time=evaluation_end_time)
    # get times with state changes
    times = {}
    for t in op_df[G_VR_LEG_START_TIME].values:
        times[t] = 1
    for t in op_df[G_VR_LEG_END_TIME].values:
        times[t] = 1
    times = sorted([t for t in times.keys()])
    # bin time stamps
    new_times = []
    for i, t in enumerate(times):
        if i == 0:
            new_times.append(t)
            continue
        while t - new_times[-1] > DEF_SMALLER_TEMPORAL_RESOLUTION:
            new_times.append(new_times[-1] + DEF_SMALLER_TEMPORAL_RESOLUTION)
        new_times.append(t)
    times = new_times
    # aggregate number of vehicle in state at time stamps
    plotvalues = {}
    for status, status_df in op_df.groupby(G_VR_STATUS):
        starts = status_df[G_VR_LEG_START_TIME].values
        ends = status_df[G_VR_LEG_END_TIME].values
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
    # add active vehicle curve if given
    ts = []
    vehs = []
    sum_veh = len(op_df[G_V_VID].unique())
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

    # sort states
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

    # additional binning
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

    # do the plot
    plt.figure(figsize=(10,10))
    plt.stackplot(times, values, labels=states, colors=LIST_COLORS)
    if len(vehs) > 0:
        plt.plot(ts, vehs, "k-")
    plt.legend(title='Vehicle States', bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.xlabel("Time [h]")
    plt.ylabel("Accumulated Number of Vehicles")
    # labels=[f"2019/{i+18}/11" for i in range(int(times[-1]/3600/24))]
    plt.xticks([i*3600*12 for i in range(int(times[-1]/3600/12))],
               labels=[f"{i*12}" for i in range(int(times[-1]/3600/12))])
    start_time = scenario_parameters[G_SIM_START_TIME]
    if evaluation_start_time is None and evaluation_end_time is None:
        plt.xlim(start_time, times[-1])
    else:
        s = start_time
        if evaluation_start_time is not None:
            s = evaluation_start_time
        e = times[-1]
        if evaluation_end_time is not None:
            e = evaluation_end_time
        plt.xlim(s, e)
    plt.ylim(0, sum_veh)
    # plt.tight_layout(rect=[0,0,0.75,1])
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "fleet_states_time_op_{}.png".format(op_id)), bbox_inches="tight")
        plt.close()


def _create_occ_plot_lists(op_df, max_occupancy):
    times = {}
    for t in op_df[G_VR_LEG_START_TIME].values:
        times[t] = 1
    for t in op_df[G_VR_LEG_END_TIME].values:
        times[t] = 1
    times = sorted([t for t in times.keys()])
    new_times = []
    for i, t in enumerate(times):
        if i == 0:
            new_times.append(t)
            continue
        while t - new_times[-1] > DEF_SMALLER_TEMPORAL_RESOLUTION:
            new_times.append(new_times[-1] + DEF_SMALLER_TEMPORAL_RESOLUTION)
        new_times.append(t)
    times = new_times
    plotvalues = {}
    for occupancy, occ_df in op_df.groupby(G_VR_NR_PAX):
        if occupancy > max_occupancy:
            print("Warning {} entries with large occupancy than {} -> ignored (usually boarding processes)".format(occ_df.shape[0], max_occupancy))
            continue
        if occupancy > 0:
            starts = occ_df[G_VR_LEG_START_TIME].values
            ends = occ_df[G_VR_LEG_END_TIME].values
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
            # non action_states
            non_action_states = ["out_of_service", "charging", "idle"]
            action_occ_df = occ_df[occ_df[G_VR_STATUS].isin(non_action_states) == False]
            non_action_occ_df = occ_df[occ_df[G_VR_STATUS].isin(non_action_states)]
            for i, occ_0_df in enumerate([action_occ_df, non_action_occ_df]):
                starts = occ_0_df[G_VR_LEG_START_TIME].values
                ends = occ_0_df[G_VR_LEG_END_TIME].values
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
    return times, values, occupancy


def occupancy_over_time(output_dir, scenario_parameters, operator_attributes, dir_names, op_id, show=False,
                        bin_size=DEF_TEMPORAL_RESOLUTION, evaluation_start_time=None, evaluation_end_time=None):
    """ this function creates a plot of the different vehicle occupancies over time
    :param ouput_dir: directory of scenario results
    :param scenario_parameters: scenario parameter dictionary
    :param operator_attributes: op atts dictionary
    :param dir_names: dir name dictionary
    :param op_id: operator id
    :param show: if True, plot is shown but not stored; if false otherwise
    :param bin_size: bin size of plot
    :param evaluation_start_time: start time of the evaluation time interval
    :param evaluation_end_time: end time of the evaluation time interval
    """
    # load operator stats and infor idle times
    op_df = _load_op_stats_and_infer_idle_states(output_dir, scenario_parameters, op_id,
                                                 evaluation_start_time=evaluation_start_time,
                                                 evaluation_end_time=evaluation_end_time)
    # create number of occupancy stats at different time stamps
    max_occupancy = op_df[op_df[G_VR_STATUS] == "route"][G_VR_NR_PAX].max()
    times, values, occupancy = _create_occ_plot_lists(op_df, max_occupancy)
    # additional binning
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

    # create plots
    number_vehicles = len(op_df[G_V_VID].unique())
    start_time = scenario_parameters[G_SIM_START_TIME]
    plt.figure(figsize=(10,10))
    plt.stackplot(times, values, labels=occupancy, colors=LIST_COLORS)
    plt.legend(title='Occupancy', bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.xlabel("Time [h]")
    plt.ylabel("Accumulated Number of Vehicles")
    # labels=[f"2019/{i+18}/11" for i in range(int(times[-1]/3600/24))]
    plt.xticks([i for i in range(start_time, int(times[-1]), 3600*12)],
               labels=[f"{int(i/3600)}" for i in range(start_time, int(times[-1]), 3600*12)])
    if evaluation_start_time is None and evaluation_end_time is None:
        plt.xlim(start_time, times[-1])
    else:
        s = start_time
        if evaluation_start_time is not None:
            s = evaluation_start_time
        e = times[-1]
        if evaluation_end_time is not None:
            e = evaluation_end_time
        plt.xlim(s, e)
    plt.ylim(0, number_vehicles)
    # plt.tight_layout(rect=[0,0,0.75,1])
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "fleet_occupancy_time_op_{}.png".format(op_id)), bbox_inches="tight")
        plt.close()
    return times, values
    

def occupancy_over_time_rel(output_dir, scenario_parameters, operator_attributes, dir_names, op_id,
                            bin_size=DEF_TEMPORAL_RESOLUTION, show=False,
                            evaluation_start_time=None, evaluation_end_time=None):
    """ this function creates a plot of the different vehicle occupancies over time
    :param ouput_dir: directory of scenario results
    :param scenario_parameters: scenario parameter dictionary
    :param operator_attributes: op atts dictionary
    :param dir_names: dir name dictionary
    :param op_id: operator id
    :param show: if True, plot is shown but not stored; if false otherwise
    :param bin_size: bin size of plot
    :param evaluation_start_time: start time of the evaluation time interval
    :param evaluation_end_time: end time of the evaluation time interval
    """
    # load operator stats and infor idle times
    op_df = _load_op_stats_and_infer_idle_states(output_dir, scenario_parameters, op_id,
                                                 evaluation_start_time=evaluation_start_time,
                                                 evaluation_end_time=evaluation_end_time)
    # create number of occupancy stats at different time stamps
    max_occupancy = op_df[op_df[G_VR_STATUS] == "route"][G_VR_NR_PAX].max()
    times, values, occupancy = _create_occ_plot_lists(op_df, max_occupancy)
    # rescale values to sum up to 100
    for t_index in range(len(values[0])):
        s = 0
        for value in values:
            s += value[t_index]
        if s > 0:
            for i in range(len(values)):
                values[i][t_index] = values[i][t_index]/s*100.0
    # additional binning
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
    # do the plots
    start_time = scenario_parameters[G_SIM_START_TIME]
    plt.figure(figsize=(10,10))
    plt.stackplot(times, values, labels=occupancy, colors=LIST_COLORS)
    plt.legend(title='Occupancy', bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.xlabel("Time [h]")
    plt.ylabel("Fraction of Vehicles [%]")
    # labels=[f"2019/{i+18}/11" for i in range(int(times[-1]/3600/24))]
    plt.xticks([i*3600*12 for i in range(int(times[-1]/3600/12))],
               labels=[f"{12 * i}" for i in range(int(times[-1]/3600/12))])
    if evaluation_start_time is None and evaluation_end_time is None:
        plt.xlim(start_time, times[-1])
    else:
        s = start_time
        if evaluation_start_time is not None:
            s = evaluation_start_time
        e = times[-1]
        if evaluation_end_time is not None:
            e = evaluation_end_time
        plt.xlim(s, e)
    plt.ylim(0, 100)
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "fleet_occupancy_relative_time_op_{}.png".format(op_id)), bbox_inches="tight")
        plt.close()


# -------------------------------------------------------------------------------------------------------------------- #
# main script call
def run_complete_temporal_evaluation(output_dir, evaluation_start_time=None, evaluation_end_time=None):
    """This method creates all plots for fleet, user, network and pt KPIs for a given scenario and saves them in
    the respective output directory. Furthermore, it creates a file 'temporal_eval.csv' containing the time series data.
    These can be used for scenario comparisons.

    :param output_dir: output directory of a scenario
    :type output_dir: str
    :param evaluation_start_time: start time of evaluation
    :param evaluation_end_time: end time of evaluation
    """
    scenario_parameters, list_operator_attributes, _ = load_scenario_inputs(output_dir)
    dir_names = get_directory_dict(scenario_parameters)
    eval_dict = {}
    for op_id, op_attributes in enumerate(list_operator_attributes):
        eval_dict[op_id] = temporal_operator_plots(output_dir, op_id, print_comments=True)
        occupancy_over_time(output_dir, scenario_parameters, op_attributes, dir_names, op_id,
                            evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time)
        occupancy_over_time_rel(output_dir, scenario_parameters, op_attributes, dir_names, op_id,
                                evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time)
        vehicle_stats_over_time(output_dir, scenario_parameters, op_attributes, dir_names, op_id,
                                evaluation_start_time=evaluation_start_time, evaluation_end_time=evaluation_end_time)


if __name__ == "__main__":
    # bugfix
    # -> set up a bugfix configuration with output dir!

    # normal script call
    # ------------------
    if len(sys.argv) == 2:
        run_complete_temporal_evaluation(sys.argv[1])
    else:
        print("Please provide the scenario output directory as additional input parameter.")
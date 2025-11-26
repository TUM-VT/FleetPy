import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


class ScenarioResultEval:
    """
    Evaluate the results of the scenarios
    """

    def __init__(self, params_df: pd.DataFrame):
        self.result_path = params_df.loc['result_path', 'value']
        self.type_name = params_df.loc['type_name', 'value']
        self.start_res_time = params_df.loc['start_res_time', 'value']
        self.end_res_time = params_df.loc['end_res_time', 'value']
        self.n_seed = params_df.loc['n_seed', 'value']
        self.CI_col_list = params_df.loc['CI_col_list', 'value']
        # self.route_len = params_df.loc['route_len', 'value']
        self.walking_speed = params_df.loc['walking_speed', 'value']
        self.gamma_walk = params_df.loc['gamma_walk', 'value']
        self.gamma_wait = params_df.loc['gamma_wait', 'value']
        self.gamma_time = params_df.loc['gamma_time', 'value']
        self.gamma_op_dist = params_df.loc['gamma_op_dist', 'value']
        self.gamma_op_time = params_df.loc['gamma_op_time', 'value']
        self.CI = [x / 100 for x in range(101)]

    # Calculate costs and metrics
    def analyze_rl_result(self, scenario_full_df: pd.DataFrame):
        scenario_full_res_df = pd.DataFrame()
        ci_dict_x_f = {}

        for i in range(len(scenario_full_df)):
            scenario = scenario_full_df.iloc[i]
            # if scenario["demand_seed"] >= n_seed:
            #     continue

            print(scenario["scenario_name"])
            scenario_folder = os.path.join(self.result_path, scenario["scenario_name"])
            # print(scenario_folder)

            # user's costs
            try:
                user_stat_df = pd.read_csv(os.path.join(scenario_folder, "1_user-stats.csv"))
            except FileNotFoundError:
                print("No user stats for {}".format(scenario["scenario_name"]))
                continue

            # filter out requests outside of start_res_time and end_res_time
            no_request_unsatisfied = len(user_stat_df[(user_stat_df["decision_time"] >= self.start_res_time) & (
                    user_stat_df["decision_time"] <= self.end_res_time)])

            user_stat_df = user_stat_df[
                (user_stat_df["rq_time"] >= self.start_res_time) & (user_stat_df["rq_time"] <= self.end_res_time)]

            # drop users with no offers user_stat_df["offers"]=="0:"
            user_stat_df = user_stat_df[user_stat_df["offers"].str.startswith("0:", na=False) &
                                        (user_stat_df["offers"].str.len() > 2) &
                                        user_stat_df["operator_id"].notnull()]
            no_request_satisfied = len(user_stat_df)
            # print(no_request_satisfied)
            # count user_stat_df["offers"] is blank
            no_request = no_request_satisfied + no_request_unsatisfied

            user_stat_df["riding_time"] = user_stat_df["dropoff_time"] - user_stat_df["pickup_time"]
            user_stat_df["waiting_time"] = user_stat_df["pickup_time"] - user_stat_df["rq_time"]

            user_stat_df['walking_distance_origin'] = \
                user_stat_df['offers'].str[2:].str.split(';', expand=True)[3].str.split('walking_distance_origin:',
                                                                                        expand=True)[1].astype(float)
            user_stat_df['walking_distance_destination'] = \
                user_stat_df['offers'].str[2:].str.split(';', expand=True)[4].str.split('walking_distance_destination:',
                                                                                        expand=True)[1].astype(float)
            user_stat_df["walking_time"] = (user_stat_df['walking_distance_origin'] + user_stat_df[
                'walking_distance_destination']) / self.walking_speed
            user_stat_df["walking_cost"] = self.gamma_walk * user_stat_df['walking_time'] * self.gamma_time / 3600
            user_stat_df["waiting_cost"] = self.gamma_wait * user_stat_df["waiting_time"] * self.gamma_time / 3600
            user_stat_df["riding_cost"] = user_stat_df["riding_time"] * self.gamma_time / 3600
            user_stat_df["user_cost"] = user_stat_df["walking_cost"] + user_stat_df["waiting_cost"] + user_stat_df[
                "riding_cost"]
            user_stat_df["journey_time"] = user_stat_df["walking_time"] + user_stat_df["waiting_time"] + user_stat_df[
                "riding_time"]

            avg_riding_time = user_stat_df["riding_time"].mean()
            avg_waiting_time = user_stat_df["waiting_time"].mean()
            avg_walking_distance_origin = user_stat_df["walking_distance_origin"].mean()
            avg_walking_distance_destination = user_stat_df["walking_distance_destination"].mean()
            avg_walking_time = user_stat_df["walking_time"].mean()
            avg_walking_cost = user_stat_df["walking_cost"].mean()
            avg_waiting_cost = user_stat_df["waiting_cost"].mean()
            avg_riding_cost = user_stat_df["riding_cost"].mean()
            avg_user_cost = user_stat_df["user_cost"].mean()
            avg_journey_time = user_stat_df["journey_time"].mean()

            for CI_col in self.CI_col_list:
                if CI_col not in ci_dict_x_f.keys():
                    ci_dict_x_f[CI_col] = {}
                if scenario['pt_fixed_length'] not in ci_dict_x_f[CI_col].keys():
                    ci_dict_x_f[CI_col][scenario['pt_fixed_length']] = user_stat_df[CI_col].quantile(self.CI)
                else:
                    ci_dict_x_f[CI_col][scenario['pt_fixed_length']] += user_stat_df[CI_col].quantile(self.CI)

                    # overall_stat = pd.read_csv(os.path.join(scneario_folder, "standard_eval.csv"), index_col=0)
            # veh_dist = overall_stat.loc["total vkm"].iloc[0]
            # veh_time = (end_time - start_time) * scenario["n_veh"]
            # veh_cost = veh_dist * gamma_op_dist * 1000 + veh_time * gamma_op_time
            op_stat_df = pd.read_csv(os.path.join(scenario_folder, "2-0_op-stats.csv"))

            # filter out requests outside of start_res_time and end_res_time
            op_stat_df = op_stat_df[
                (op_stat_df["start_time"] >= self.start_res_time) & (op_stat_df["end_time"] <= self.end_res_time)]
            veh_time = -(self.start_res_time - self.end_res_time) * scenario["pt_n_veh"]
            veh_dist = op_stat_df["driven_distance"].sum()
            veh_cost = veh_dist * self.gamma_op_dist + veh_time * self.gamma_op_time
            veh_occ = (op_stat_df["occupancy"] * op_stat_df["driven_distance"]).sum() / veh_dist
            veh_cost_per_user = veh_cost / no_request_satisfied

            total_cost = veh_cost + avg_user_cost * no_request_satisfied

            # combine the results to scenario to form scenario_res
            # no_request, no_request_satisfied, avg_riding_time, avg_waiting_time, avg_walking_distance_origin,
            # avg_walking_distance_destination, avg_walking_time, avg_user_cost, veh_dist, veh_time, veh_cost
            scenario_res = scenario.copy()
            scenario_res["no_request"] = no_request
            scenario_res["no_request_satisfied"] = no_request_satisfied
            scenario_res["no_request_unsatisfied"] = no_request_unsatisfied
            scenario_res["avg_riding_time"] = avg_riding_time
            scenario_res["avg_waiting_time"] = avg_waiting_time
            scenario_res["avg_walking_distance_origin"] = avg_walking_distance_origin
            scenario_res["avg_walking_distance_destination"] = avg_walking_distance_destination
            scenario_res["avg_walking_time"] = avg_walking_time
            scenario_res["avg_journey_time"] = avg_journey_time
            scenario_res["avg_walking_cost"] = avg_walking_cost
            scenario_res["avg_waiting_cost"] = avg_waiting_cost
            scenario_res["avg_riding_cost"] = avg_riding_cost
            scenario_res["avg_user_cost"] = avg_user_cost
            scenario_res["veh_dist"] = veh_dist
            scenario_res["veh_time"] = veh_time
            scenario_res["veh_cost"] = veh_cost
            scenario_res["veh_cost_per_user"] = veh_cost_per_user
            scenario_res["veh_occ"] = veh_occ
            scenario_res["total_cost"] = total_cost
            scenario_res["avg_cost_per_satisfied_request"] = total_cost / no_request_satisfied

            # rotate scenario_res to be a column
            scenario_res = scenario_res.to_frame().T

            scenario_full_res_df = pd.concat([scenario_full_res_df, scenario_res])

        scenario_full_res_df.set_index("scenario_name", inplace=True)

        return scenario_full_res_df, ci_dict_x_f

    def percentile(self, n):
        """
        Function to calculate the percentile of a series
        """

        def percentile_(x):
            return x.quantile(n)

        percentile_.__name__ = 'percentile_{:02.0f}'.format(n * 100)
        return percentile_

    def calculate_percentile(self, scenario_full_res_df, group_by_index):
        """
        Calculate the percentiles of the results
        """
        # scenario_full_res_df['pt_flex_length'] = self.route_len - scenario_full_res_df['pt_fixed_length']
        # scenario_full_res_df.loc[scenario_full_res_df['pt_flex_length'] < 0, 'pt_flex_length'] = 0

        # only include no_request, no_request_satisfied, avg_riding_time, avg_waiting_time, avg_walking_distance_origin,
        # avg_walking_distance_destination, avg_walking_time, avg_user_cost, veh_dist, veh_time, veh_cost, total_cost,
        # avg_cost_per_satisfied_request
        # only keep first n_seed of each scenario by group_by_index

        # drop alphanumeric columns
        grouped_df = scenario_full_res_df.reset_index().drop(columns=[
            'scenario_name', 'schedule_file', 'rq_file', 'gtfs_name', 'demand_name'
        ])
        # grouped_df = scenario_full_res_df[["n_veh", "pt_flex_length", "no_request", "no_request_unsatisfied", "no_request_satisfied", "avg_riding_time", "avg_waiting_time", "avg_walking_distance_origin", "avg_walking_distance_destination", "avg_walking_time", "avg_journey_time", "avg_user_cost", "veh_dist", "veh_time", "veh_occ", "veh_cost", "total_cost", "avg_cost_per_satisfied_request"]].astype(float).groupby(group_by_index)
        grouped_df = grouped_df.astype(float).groupby(group_by_index)

        # Applying multiple aggregation functions
        statistics_df = grouped_df.agg(['mean', 'std', 'min', 'max', 'median'])

        # For percentiles, you can use the quantile function
        percentiles_df = grouped_df.agg(
            [self.percentile(0.025), self.percentile(0.25), self.percentile(0.50),
             self.percentile(0.75), self.percentile(0.975), min, max])

        # Now, you can concatenate these DataFrames
        final_stats_df = pd.concat([statistics_df, percentiles_df], axis=1)

        return percentiles_df, final_stats_df

    def calculate_action_density(self, scenario_rl_action_np, actions, ignore_actions=[]):
        """
        Calculate the density of each action over time for RL
        """
        time_periods, simulations = scenario_rl_action_np.shape

        action_densities = {action: np.zeros(time_periods) for action in actions}

        for time_period in range(time_periods):
            ignore_action_count = 0
            for action in ignore_actions:
                ignore_action_count += np.count_nonzero(scenario_rl_action_np[time_period] == action)

            for action in actions:
                if action in ignore_actions:
                    action_densities[action][time_period] = 0
                    continue
                action_count = np.count_nonzero(scenario_rl_action_np[time_period] == action)
                if simulations - ignore_action_count == 0:
                    action_densities[action][time_period] = 0
                else:
                    action_densities[action][time_period] = action_count / (simulations - ignore_action_count)

        # replace keys of actions dict by {-1:"Regular", 0:"Zone 0", 1:"Zone 1", 2:"Hold"}
        key_map = {-1: "Regular", 0: "Zone 0", 1: "Zone 1", 2: "Hold"}
        action_densities = {key_map[old_key]: value for old_key, value in action_densities.items() if
                            old_key in key_map}

        return action_densities

    def plot_action_densities(self, action_densities, scenario_rl_action_np, title='Action Densities Over Time'):
        """
        Plot the action densities over time for RL
        """
        time_periods = np.arange(scenario_rl_action_np.shape[0])
        # Stacking the action densities vertically requires calculating the cumulative sum of densities for plotting
        densities_stack = np.vstack([action_densities[action] for action in action_densities])
        cumulative_densities = np.cumsum(densities_stack, axis=0)

        plt.figure(figsize=(10, 6))

        # Plotting the stacked areas
        for i, action in enumerate(action_densities.keys()):
            if i == 0:
                plt.fill_between(time_periods, 0, cumulative_densities[i, :], label=f'{action}')
            else:
                plt.fill_between(time_periods, cumulative_densities[i - 1, :], cumulative_densities[i, :],
                                 label=f'{action}')

        plt.xlabel('Time Period')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # print working directory
    print(os.getcwd())

    result_path = "../../studies/SoDMultiRoute/results/"
    type_name = "baseline"
    folder_path = os.path.join(result_path, type_name)
    CI_col_list = ["waiting_time", "riding_time", "walking_time", "journey_time", "user_cost"]
    n_seed = 1

    params_df = pd.DataFrame({
        'param': ['result_path', 'type_name', 'start_res_time', 'end_res_time', 'n_seed', 'CI_col_list',
                  'route_len', 'walking_speed', 'gamma_walk', 'gamma_wait', 'gamma_time', 'gamma_op_dist',
                  'gamma_op_time'],
        'value': [result_path, type_name, 79200, 86400, 100, CI_col_list,
                  5.611, 4 * 1000 / 3600, 2, 1.5, 16.5, 0.6938 / 1000,
                  7.59 / 3600]
    }).set_index('param')

    analyzer = ScenarioResultEval(params_df)

    scenario_full_df = pd.read_csv("../../studies/SoDMultiRoute/scenarios/full_case_study.csv")
    scenario_full_res_df, CI_dict_x_f = analyzer.analyze_rl_result(scenario_full_df)

    group_by_index = ["pt_route_id", "pt_flex_length"]
    percentiles_df, final_stats_df = analyzer.calculate_percentile(scenario_full_res_df.reset_index(), group_by_index)

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    scenario_full_res_df.to_csv(os.path.join(folder_path, "full_res.csv"))
    final_stats_df.to_csv(os.path.join(folder_path, 'full_res_stat.csv'))

    for CI_col in CI_col_list:
        for x_f, CI_dict in CI_dict_x_f[CI_col].items():
            CI_dict /= n_seed
    with open(os.path.join(folder_path, 'CI_dict_x_f.pickle'), 'wb') as handle:
        pickle.dump(CI_dict_x_f, handle, protocol=pickle.HIGHEST_PROTOCOL)

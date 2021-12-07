import os
import sys
import numpy as np
import pandas as pd

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(MAIN_DIR)
from src.misc.globals import *
from src.misc.init_modules import load_routing_engine
from src.infra.BoardingPointInfrastructure import BoardingPointInfrastructure
from src.fleetctrl.pooling.objectives import LARGE_INT

def multiple_boarding_points(mod_user_stats, operator_attributes, scenario_parameters, dir_names, op_var_costs):
    infrastructure_dir = dir_names.get(G_DIR_INFRA)
    if dir_names.get(G_DIR_INFRA, None) is not None and os.path.isfile(os.path.join(infrastructure_dir, "boarding_points.csv")):
        routing_engine = load_routing_engine(scenario_parameters[G_NETWORK_TYPE], dir_names[G_DIR_NETWORK], network_dynamics_file_name=scenario_parameters.get(G_NW_DYNAMIC_F, None))
        max_walking_distance = scenario_parameters[G_BP_MAX_DIS]
        boarding_time = operator_attributes[G_OP_CONST_BT]

        walking_speed = scenario_parameters[G_WALKING_SPEED]
        boardingpoints = BoardingPointInfrastructure(dir_names[G_DIR_INFRA], routing_engine)
        sorted_rqs = mod_user_stats.sort_values(by = [G_RQ_TIME])

        start_time = int(scenario_parameters[G_SIM_START_TIME])
        routing_engine.update_network(start_time)
        res_dict = {}
        for key, user_stat in sorted_rqs.iterrows():
            #print(start_time, user_stat)
            for t in range(start_time + 1, int(user_stat[G_RQ_TIME]) + 1):
                routing_engine.update_network(t)
            start_time = int(user_stat[G_RQ_TIME])

            org_pos = routing_engine.return_position_from_str(user_stat[G_RQ_ORIGIN])
            dest_pos = routing_engine.return_position_from_str(user_stat[G_RQ_DESTINATION])
            try:
                pu_pos = routing_engine.return_position_from_str(user_stat[G_RQ_PUL])
            except:
                pu_pos = org_pos
            try:
                do_pos = routing_engine.return_position_from_str(user_stat[G_RQ_DOL])
            except:
                do_pos = dest_pos
            walking_distance_start = boardingpoints.return_walking_distance(org_pos, pu_pos)
            walking_distance_end = boardingpoints.return_walking_distance(dest_pos, do_pos)

            _, tt, dis = routing_engine.return_travel_costs_1to1(pu_pos, do_pos)

            bp_abs_detour = user_stat[G_RQ_DO] - user_stat[G_RQ_PU] - boarding_time - tt
            bp_rel_detour = bp_abs_detour/(tt + boarding_time) * 100.0

            res_dict[key] = {"walk_start_dis" : walking_distance_start, "walk_end_dis" : walking_distance_end,
                "walk_start_time" : walking_distance_start / walking_speed, "walk_end_time" : walking_distance_end / walking_speed,
                "bp_direct_travel_time" : tt, "bp_direct_travel_dis" : dis,
                "bp_abs_detour" : bp_abs_detour, "bp_rel_detour" : bp_rel_detour}
        new_df = pd.DataFrame(res_dict.values(), index= res_dict.keys())
        res = pd.concat([sorted_rqs, new_df], axis=1)

        
        N_r = new_df.shape[0]
        return_dict = {}
        return_dict["avg_walk_dist_start"] = new_df["walk_start_dis"].sum()/N_r 
        return_dict["avg_walk_dist_end"] = new_df["walk_end_dis"].sum()/N_r
        return_dict["sum_walk_dis"] = new_df["walk_start_dis"].sum() + new_df["walk_end_dis"].sum()
        return_dict["sum_walk_time"] = new_df["walk_start_time"].sum() + new_df["walk_end_time"].sum()
        return_dict["avg_walk_dis"] = return_dict["sum_walk_dis"]/N_r
        return_dict["avg_walk_time"] = return_dict["sum_walk_time"]/N_r
        return_dict["bp_sum_direct_distance"] = new_df["bp_direct_travel_dis"].sum()
        return_dict["bp_sum_direct_distance_per"] = (res["number_passenger"]*res["bp_direct_travel_dis"]).sum()
        return_dict["bp_sum_direct_tt"] = new_df["bp_direct_travel_time"].sum()
        return_dict["bp_abs_detour"]= new_df["bp_abs_detour"].sum()/N_r
        return_dict["bp_rel_detour"]= new_df["bp_rel_detour"].sum()/N_r

        return return_dict
    return {}
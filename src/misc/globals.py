"""
Specification of global variables.
These variables should be referenced for Data Input, Output and Evaluation to guarantee consistency.
"""
import os
import json
from enum import Enum
from types import DynamicClassAttribute



# -------------------------------------------------------------------------------------------------------------------- #
# Scenario Definition
# -------------------
G_SC_INP_F = "00_config.json"

# general required input
G_SIM_ENV = "sim_env"
G_STUDY_NAME = "study_name"
G_SCENARIO_NAME = "scenario_name"
G_NETWORK_NAME = "network_name"
G_DEMAND_NAME = "demand_name"
G_RANDOM_SEED = "random_seed"
G_SIM_START_TIME = "start_time"
G_SIM_END_TIME = "end_time"
G_SIM_TIME_STEP = "time_step"
G_SIM_ROUTE_OUT_FLAG = "route_output_flag"
G_SIM_REPLAY_FLAG = "replay_flag"
G_SIM_REALTIME_PLOT_FLAG = "realtime_plot"
G_SIM_REALTIME_PLOT_VEHICLE_STATUS = "realtime_plot_veh_states"
G_SIM_REALTIME_PLOT_EXTENTS = "realtime_plot_extents"
G_NR_OPERATORS = "nr_mod_operators"
G_NR_CH_OPERATORS = "nr_charging_operators"
G_LOG_GUROBI = "log_gurobi" # optional; if True gurobi output file written -> default False

# general optional input
G_ZONE_SYSTEM_NAME = "zone_system_name"
G_INIT_STATE_SCENARIO = "initial_state_scenario"
G_ENERGY_EMISSIONS = "co2_per_kWh"
G_OP_SHARE_FC = "eval_share_daily_fix_cost"
G_PV_G_CO2_KM = "pV_emissions"
G_BP_INFRA_NAME = "boardingpoint_infra_name"
G_INFRA_NAME = "infra_structure_name"
G_INFRA_DEP = "infra_depot_file"
G_INFRA_PBCS = "infra_public_cs_file"
G_INFRA_ALLOW_SP = "infra_allow_street_parking"

# slave simulation specific input
G_MAIN_DATA_PATH = "data_main_path"
G_OUTPUT_PATH = "output_path"
G_NW_TRAFO_F = "nw_trafo_f"
G_COM_SOCKET_PORT = "socket_port"
G_SLAVE_CPU = "n_cpu_per_sim"
G_INIT_STATE_F = "init_state_file"
G_UNPROCESSED_NETWORK_NAME = "unprocessed_network"  # network path with original exported data and source_ids
G_MAX_IM_OF_PROC = "max_im_offer_process"


# network general attributes
G_NETWORK_TYPE = "network_type"
# network specific attributes
# -> dynamic networks
G_NW_DENSITY_T_BIN_SIZE = "nw_density_temporal_bin_size"
G_NW_DENSITY_AVG_DURATION = "nw_density_avg_duration"
# network dynamic file
G_NW_DYNAMIC_F = "nw_dynamic_f"

# zone specific attributes
G_PARK_COST_SCALE = "park_cost_scale"
G_TOLL_COST_SCALE = "toll_cost_scale"
G_ZONE_CORR_M_F = "zone_correlation_file"

# forecast attributes
G_FC_TYPE = "fc_type"
G_FC_TR = "temporal_resolution"
G_FC_FNAME = "forecast_f"

# public transport specific attributes
G_PT_TYPE = "pt_type"
G_GTFS_NAME = "gtfs_name"
G_PT_SCHEDULE_F = "schedule_file"
G_PT_FRQ_SCALE = "pt_freq_scale"
G_PT_FRQ_HOURS = "pt_freq_scale_hours"
G_PT_FARE_B = "pt_base_fare"

# traveler general attributes
G_RQ_FILE = "rq_file"
G_RQ_TYP1 = "rq_type"
G_RQ_TYP2 = "rq_type_distribution"
G_RQ_TYP3 = "rq_type_od_distribution"

# parcel traveler input
G_PA_DEMAND_NAME = "parcel_demand_name"
G_PA_RQ_FILE = "parcel_rq_file"
G_PA_RQ_TYP1 = "parcel_rq_type"
G_PA_RQ_TYP2 = "parcel_rq_type_distribution"
G_PA_RQ_TYP3 = "parcel_rq_type_od_distribution"

# traveler MoD accept/reject attributes
G_AR_MIN_WT = "user_min_wait_time"
G_AR_MAX_WT = "user_max_wait_time"
G_AR_MAX_WT_2 = "user_max_wait_time_2"  # for linear decline in acceptence probability (BMW RP study)
G_AR_MAX_CDT = "user_max_constant_detour_time"
G_AR_MAX_DTF = "user_max_detour_time_factor"
G_AR_MAX_DEC_T = "user_max_decision_time"
G_AR_PRICE_F = "user_price_sensitivity_func_dict"

# traveler mode choice attributes
G_MC_VOT = "value_of_time"
G_MC_U0_PV = "private_vehicle_mode_choice_intercept"
G_MC_TRANSFER_P = "pt_transfer_penalty"
G_MC_C_D_PV = "private_vehicle_full_costs_per_m"

# traveler specific attributes
G_WALKING_SPEED = "walking_speed"

# -> inter-modal travellers
G_IM_MIN_MOD_DISTANCE = "min_IM_MOD_distance"
G_IM_PER_KM_SUBSIDY = "subsidy_IM_MOD_per_km"
G_IM_TRANSFER_TIME = "im_transfer_time"


# operator general attributes
G_OP_MODULE = "op_module"
G_OP_FLEET = "op_fleet_composition"
G_OP_ACT_FLEET_SIZE = "op_act_fs_file"
G_OP_INIT_VEH_DIST = "op_init_veh_distribution"
# -> plan requests
G_OP_VR_CTRL_F = "op_vr_control_func_dict"
G_OP_MIN_WT = "op_min_wait_time"
G_OP_MAX_WT = "op_max_wait_time"
G_OP_MAX_WT_2 = "op_max_wait_time_2"  # for linear decline in acceptence probability (BMW RP study)
G_OP_OFF_TW = "op_offer_time_window"  # for fixing time window after accepting the amod offer (BMW RP study)
G_OP_MAX_DTF = "op_max_detour_time_factor"
G_OP_ADD_CDT = "op_add_constant_detour_time"
G_OP_MAX_CDT = "op_max_constant_detour_time"
G_OP_MIN_DTW = "op_min_detour_time_window"
G_OP_CONST_BT = "op_const_boarding_time"
G_OP_ADD_BT = "op_add_boarding_time"
G_OP_FARE_B = "op_base_fare"
G_OP_FARE_D = "op_distance_fare"
G_OP_FARE_T = "op_time_fare"
G_OP_FARE_MIN = "op_min_standard_fare"
G_OP_UTIL_SURGE = "op_util_surge_price_file"
G_OP_UTIL_EVAL_INT = "op_util_eval_interval"    # time interval utilization is evaluated for before prices are adopted
G_OP_ZONE_PRICE = "op_zone_price_scale_dict"
G_OP_ELA_PRICE = "op_elastic_price_file"
G_OP_FC_SUPPLY = "op_supply_fc_type"
# charging
G_OP_DEPOT_F = "op_depot_file"
G_OP_CH_N_STATION_QUERY = "op_n_charge_station_query"   # max number of stations to query charge offers from
G_OP_CH_N_OFFER_P_ST_QUERY = "op_n_charge_offer_per_station"    # max number of offers per station
#parcel constraints
G_OP_PA_EPT = "op_parcel_earliest_pickup_time"
G_OP_PA_LPT = "op_parcel_latest_pickup_time"
G_OP_PA_EDT = "op_parcel_earliest_dropoff_time"
G_OP_PA_LDT = "op_parcel_latest_dropoff_time"
G_OP_PA_CONST_BT = "op_parcel_const_boarding_time"
G_OP_PA_ADD_BT = "op_parcel_add_boarding_time"

# operator specific attributes
G_RA_SOLVER = "op_solver"   # currently "Gurobi" or "CPLEX"
G_RA_RP_BATCH_OPT = "op_rp_batch_optimizer"
G_RA_LOCK_TIME = "op_lock_time"
G_RA_REOPT_TS = "op_reoptimisation_timestep"
G_RA_TB_TO_PER_VEH = "op_treebuild_timeout_per_veh"
G_RA_OPT_TO = "op_optimisation_timeout"
G_RA_HEU = "op_applied_heuristic"
G_RA_TW_HARD = "op_time_window_hardness"    # 1 -> soft | 2 -> hard # TODO # think about renaming to update_time_window_hardness
G_RA_TW_LENGTH = "op_time_window_length"
G_RA_LOCK_RID_VID = "op_lock_rid_vid_assignment" # no re-assignment if false

G_RA_OP_NW_TYPE = "op_network_type"    # if given, operator loads a different network for its usage (currently only for reservation)
G_RA_OP_NW_NAME = "op_network_name"     # if given, operator loads a different network for its usage (currently only for reservation)
G_RA_OP_NW_DYN_F = "op_network_dynamics_file" # if given, operator loads a different network for its usage (currently only for reservation)

# reservation
G_RA_RES_MOD = "op_reservation_module"
G_RA_OPT_HOR = "op_short_term_horizon"  # time ahead when requests will be treated as reservation requests
G_RA_ASS_HOR = "op_res_assignment_horizon"  # time ahead when reservation plans will be assigned to vehicles (must exceed op_short_term_horizon)
G_RA_RES_REASSIGN_SP = "op_res_reassign_sp" # True, if supproting points should be reassign after each rp optimization
G_RA_MAX_BATCH_SIZE = "op_res_batch_size"   # size of of batches for ForwardBatchOptimization
G_RA_MAX_BATCH_CONCAT = "op_res_batch_concat"   # how many batches are schedule in ForwardBatchOptimization
G_RA_RES_BOPT_TS = "op_res_opt_timestep"    # time interval of reservation module
G_RA_RES_LG_MAX_DEPTH = "op_res_loc_graph_max_depth"    # for GraphContractionTSP -> depth of local graph for evalutating new sol
G_RA_RES_LG_MAX_CUT = "op_res_loc_graph_max_cut_time"   # for GraphContractionTSP -> only add edges wtih this max time horizon and cut rest
G_RA_RES_APP_BUF_TIME = "op_res_approach_buffer_time"   # time buffer before vehicle starts driving to the position of an assigned VRL with reservation in future

# RV heuristics
G_RA_MAX_VR = "op_max_VR_con"
G_RA_MAX_RP = "op_max_request_plans"
G_RH_I_NWS = "op_rh_immediate_max_routes"
G_RH_R_NWS = "op_rh_reservation_max_routes"
G_RH_R_ZSM = "op_rh_reservation_zone_search"
G_RH_R_MPS = "op_rh_reservation_consider_max_ps"
G_RVH_DIR = "op_rvh_nr_direction"
G_RVH_B_DIR = "op_batch_rvh_nr_direction"
G_RVH_LWL = "op_rvh_nr_least_load"
G_RVH_B_LWL = "op_batch_rvh_nr_least_load"
G_VPI_KEEP = "op_vpi_nr_plans"
G_VPI_SF = "op_vpi_skip_first"
G_RVH_AM_RR = "op_rvh_AM_nr_check_assigned_rrs"
G_RVH_AM_TI = "op_rvh_AM_nr_test_best_insertions"

#PT Optimizer
G_RA_PT_N_HB = "op_pt_n_heatbaths"
G_RA_PT_HB_ITS = "op_pt_n_heatbath_iterations"
G_RA_PT_BREAK = "op_pt_break_condition"

# Repositioning
G_OP_REPO_M = "op_repo_method"
G_OP_REPO_TS = "op_repo_timestep"
G_OP_REPO_LOCK = "op_repo_lock"
G_OP_REPO_TH_DEF = "op_repo_horizons"
G_OP_REPO_SI = "op_repo_add_infos"
G_OP_REPO_REL_FRAC = "op_repo_rel_frac_veh" # MOIARepoPavoneRelative (0, 1)
G_OP_REPO_GAMMA = "op_repo_discount_factor"
G_OP_REPO_EXP_TP = "op_repo_exp_trip_profit"
G_OP_REPO_SR_F = "op_repo_sharing_rate_file"
G_OP_REPO_QBT = "op_repo_quadratic_benefit_threshold"
G_OP_REPO_FRONTIERS_M = "op_repo_frontiers_method"          # Method name for Frontier's approaches
G_OP_REPO_RES_PUF = "op_repo_res_buffer_time"       # if reservation leg, how long in future to be considered for repo

# Dynamic Pricing
G_OP_DYN_P_M = "op_dyn_pricing_method"
G_OP_DYN_P_FUNC = "op_dyn_pricing_func_dict"
G_OP_DYN_MAX_PF = "op_dyn_pricing_max_factor"
G_OP_DYN_P_LOG_A = "op_dyn_pricing_log_par_a"
G_OP_DYN_P_LOG_B = "op_dyn_pricing_log_par_b"

# Dynamic Fleet Sizing
G_OP_DYN_FS_M = "op_dyn_fs_method"
G_OP_DYFS_TARGET_UTIL = "op_dyfs_target_utilization"    # control target value
G_OP_DYFS_TARGET_UTIL_INT = "op_dyfs_target_utilization_interval"   # reactive only if intervall around target util exceeded
G_OP_DYFS_MIN_ACT_FS = "op_dyfs_minimun_active_fleetsize"   # cant get lower then that [vehicles]
G_OP_DYFS_UNDER_UTIL_DUR = "op_dyfs_underutilization_duration"  # only remove vehicles if underutilization for this period

# Charging
G_OP_CH_M = "op_charging_method"
G_OP_CHARGE_PUBLIC_ONLY = "op_charging_public_only"
G_OP_APS_SOC = "op_min_soc_after_planstop"
G_PUBLIC_CHARGING_FILE = "op_charging_stations_file"
G_OP_MIN_SOC_CHARGE_PUBLIC = "op_min_soc_public_station"    # Minimum soc limit after which a EV must visit charging station
G_OP_MAX_DURATION_HOURS = "op_max_charging_duration_hours"           # List of pairs (start_time, end_time) when the max durations are applied
G_OP_STATIONS_MAX_DURATIONS = "op_max_charging_durations"            # max allowed charging durations at stations corresponding to applicable hours in G_OP_MAX_DURATION_HOURS

# Broker / Multi-Operator parameters
G_MULTIOP_PREF_OP_RSEED = "multiop_preferred_operator_random_seed"
G_MULTIOP_PREF_OP_PROB = "op_multiop_preferred_operator_probabilities"
G_MULTIOP_EVAL_METHOD = "op_multiop_eval_will_method"
G_MULTIOP_EVAL_LOOKAHEAD = "op_mulitop_eval_lookahead_t"
G_MULTIOP_EXCH_AC_OBS_TIME = "op_multiop_exchange_asscost_observation_time"
G_MULTIOP_EXCH_AC_STD_WEIGHT = "op_multiop_exchange_asscost_std_weight"

#multiple boarding points
G_BP_MAX_DIS = "bp_max_walking_distance"
G_BP_MAX_BPS = "bp_max_bp_to_consider"

# aimsun api
G_AIMSUN_STAT_INT = "aimsun_statistics_interval"
G_AIMSUN_VEH_TYPE_NAME = "aimsun_vehicle_type_name"

# sumo api
G_SUMO_STAT_INT = "sumo_statistics_interval"    # interval in which new network statistics are gathered and sent to FleetPy to updated network (if not given, no statistics are gathered)
G_SUMO_SIM_TIME_OFFSET = "sumo_sim_time_offset" # offset between fleetpy and sumo simulation time (fleetpy simtime = sumo simtim + offset; if not given, 0)

# RPP fleetcontrol
G_OP_PA_ASSTH = "op_parcel_assignment_threshold"
G_OP_PA_OBASS = "op_parcel_passenger_ob_assignment"
G_OP_PA_REDEL = "op_parcel_remaining_delivery_time"

# -------------------------------------------------------------------------------------------------------------------- #
# Charging Stations/Depots
# ------------------------
# scenario input parameters
G_INFRA_OP_OWNER = "op_depot_ownership"
G_INFRA_DEPOT_FOR_ST = "depot_keep_free_for_short_term"

# data input
G_INFRA_CS_ID = "charging_station_id"
G_INFRA_CU_DEF = "charging_units"
G_INFRA_MAX_PARK = "max_nr_parking"
G_INFRA_PUB_UTIL = "public_util"

# active vehicle
G_ACT_VEH_SHARE = "share_active_fleet_size"

# charging operator
G_CH_OP_F = "ch_op_public_charging_station_f"
G_CH_DISCONNECT_ON_FULL_SOC = "ch_op_disconnect_charging_on_full_soc"                  # Should a charging vehicle be disconnected immediately after full soc
G_CH_OP_INIT_CH_EVENTS_F = "ch_op_init_charge_events_f"
# parameter for all charging operators
G_CH_OP_MAX_STATION_SEARCH_RADIUS = "ch_max_station_search_radius"  # radius in seconds travel time (TODO ?)
G_CH_OP_MAX_CHARGING_SEARCH = "ch_max_station_search"    # max number of charging stations to be considered

#private vehicles
G_PRIVATE_TRIPS_FILE = "op_private_trips_file"  # Private vehicle trips file
G_PRIVATE_PRIME_CUSTOMER_RATIO = "op_private_prime_vehicle_ratio"      # Percentage of private vehicles of each type to be marked as prime members for booking charging station

# -------------------------------------------------------------------------------------------------------------------- #
# Directories
# -----------
G_DIR_MAIN = "main"
G_DIR_DATA = "data"
G_DIR_OUTPUT = "output"
G_DIR_NETWORK = "network"
G_DIR_DEMAND = "demand"
G_DIR_ZONES = "zones"
G_DIR_FC = "forecasts"
G_DIR_PT = "pubtrans"
G_DIR_VEH = "vehicles"
G_DIR_FCTRL = "fleetctrl"
G_DIR_BP = "boardingpoints"
G_DIR_INFRA = "infra"
G_DIR_PARCEL_DEMAND = "parceldemand"

# -------------------------------------------------------------------------------------------------------------------- #
# General
# -------
G_SIM_TIME = "simulation_time"

# -------------------------------------------------------------------------------------------------------------------- #
# Network Model
# -------------

# nodes.csv
# ---------
G_NODE_ID = "node_index"
G_NODE_STOP_ONLY = "is_stop_only"
G_NODE_X = "pos_x"
G_NODE_Y = "pos_y"
G_NODE_CH_ORDER = "node_order"

# edges.csv
# ---------
G_EDGE_FROM = "from_node"
G_EDGE_TO = "to_node"
G_EDGE_DIST = "distance"
G_EDGE_TT = "travel_time"
G_EDGE_SC = "shortcut_def"
G_EDGE_SOURCE = "source_edge_id"

# -------------------------------------------------------------------------------------------------------------------- #
# Public Transport Model
# ----------------------

# traveltime_matrix.csv
# ---------------------
G_PT_TT_W = "walk_distance"
G_PT_TT_TT = "travel_time"
G_PT_TT_NT = "number_transfers"

# add_route_information.csv
# -------------------------
G_PT_R_L = "route_length"
G_PT_R_CAP = "capacity"
G_PT_R_NR = "nr_trips"
G_PT_R_EPT = "energy_per_trip"
G_PT_AVG_DUR = "tt"
G_PT_COST_PER_KM = "pt_cost_per_km"

# output
# ------
G_PT_ROUTE = "route_id"
G_PT_TRAVELERS = "number_travelers"
G_PT_MOV_AVG = "moving_average_travelers"
G_PT_BG_T = "background_travelers"
G_PT_CAP = "total_capacity"
G_PT_CROWD = "crowding_factor"

# -------------------------------------------------------------------------------------------------------------------- #
# Zone-System Model
# -----------------
G_ZONE_ZID = "zone_id"
G_ZONE_NAME = "zone_name"
G_ZONE_NID = "node_index"
G_ZONE_CEN = "is_centroid"
G_ZONE_FLM = "offer_first_last_mile"
G_ZONE_PC = "park_cost_scale_factor"
G_ZONE_TC = "toll_cost_scale_factor"
G_ZONE_FC_T = "time"

# TODO # add zone/nw specific attributes

# output
# ------
G_ZONE_MOVE_VEH = "moving_vehicles (sim_only)"
G_ZONE_DENSITY = "density"
G_ZONE_FLOW = "flow"
G_ZONE_CTT = "cluster_tt_factor"
G_ZONE_CTC = "cluster_toll_coefficient"
G_ZONE_CPC = "cluster_park_coefficient"

# -------------------------------------------------------------------------------------------------------------------- #
# Traveler Models
# ---------------

# input data
# ----------
G_RQ_ORIGIN = "start"
G_RQ_DESTINATION = "end"
G_RQ_TIME = "rq_time"
G_RQ_ID = "request_id"

# optional input data (and output)
# --------------------------------
G_RQ_PAX = "number_passenger"
G_RQ_EPT = "earliest_pickup_time"
G_RQ_LPT = "latest_pickup_time"
G_RQ_LDT = "latest_decision_time"
G_RQ_MRD = "max_rel_detour"
G_RQ_MAX_FARE = "max_fare"
# parcel
G_RQ_PA_SIZE = "parcel_size"
G_RQ_PA_EPT = "parcel_earliest_pickup_time"
G_RQ_PA_LPT = "parcel_latest_pickup_time"
G_RQ_PA_EDT = "parcel_earliest_dropoff_time"
G_RQ_PA_LDT = "parcel_latest_dropoff_time"

# output general
# --------------
G_RQ_TYPE = "rq_type"
G_RQ_OFFERS = "offers"
G_RQ_LEAVE_TIME = "decision_time"
G_RQ_CHOSEN_OP_ID = "chosen_operator_id"
G_RQ_OP_ID = "operator_id"
G_RQ_VID = "vehicle_id"
G_RQ_PU = "pickup_time"
G_RQ_PUL = "pickup_location"
G_RQ_DO = "dropoff_time"
G_RQ_DOL = "dropoff_location"
G_RQ_FARE = "fare"
G_RQ_ACCESS = "access_time"
G_RQ_EGRESS = "egress_time"
G_RQ_MODAL_STATE = "modal_state" # (see traveler modal state -> indicates monomodal/intermodal)

# output environment specific
# ---------------------------
G_RQ_DRT = "direct_route_travel_time"
G_RQ_DRD = "direct_route_distance"
G_RQ_TRD = "traveled_route_distance"
G_RQ_SRD = "shared_route_distances"
G_RQ_SUB = "included_subsidy"
G_RQ_TOLL = "included_toll"
G_RQ_PARK = "included_park_costs"
G_RQ_IM_PT_FARE = "intermodal_included_pt_fare"
G_RQ_C_UTIL = "utility_chosen_mode"
G_RQ_DEC_THRESH = "decision_threshold"
G_RQ_DEC_SPP = "decision_steps_per_process"
G_RQ_DEC_FEEDBACK = "decision_feedback_rate"
G_RQ_DEC_DIFF_RATE = "decision_diffusion_rate"
G_RQ_DEC_MAX = "max_decisions"
G_RQ_DEC_GROUP = "decision_group"
G_RQ_DEC_WT_FAC = "waiting_time_factor"
G_RQ_DEC_REAC = "reaction_time"

# traveler modal state
G_RQ_STATE_MONOMODAL = 0
G_RQ_STATE_FIRSTMILE = 1
G_RQ_STATE_LASTMILE = 2
G_RQ_STATE_FIRSTLASTMILE = 3

# -------------------------------------------------------------------------------------------------------------------- #
# Mode Choice Model
# -----------------

# non-MoD choices
# ---------------
G_MC_DEC_PT = -1
G_MC_DEC_PV = -2
G_MC_DEC_IM = -3

# offer parameters
# ----------------
G_OFFER_ID = "offer_id"
G_OFFER_ACCESS_W = "t_access"
G_OFFER_WAIT = "t_wait"
G_OFFER_DRIVE = "t_drive"
G_OFFER_EGRESS_W = "t_egress"
G_OFFER_FARE = "fare"
G_OFFER_PU_DELAY = "t_pickup_delay" # time [s] from earliest start time to expected pickup time
G_OFFER_TRANSFERS = "nr_transfers"
G_OFFER_TOLL = "toll_costs"
G_OFFER_PARK = "park_costs"
G_OFFER_CROWD = "pt_crowding_factor"
G_OFFER_DIST = "d_drive"
G_OFFER_PU_INT_START = "t_pickup_interval_start"
G_OFFER_PU_INT_END = "t_pickup_interval_end"
G_OFFER_PU_POS = "pickup_position"
G_OFFER_DO_POS = "dropoff_position"
G_OFFER_WILLING_FLAG = "willing"
G_OFFER_PREFERRED_OP = "is_preferred_op"
G_OFFER_IS_VALID = "is_valid"   # indicates if offer is still valid at time of choosing (to get the output)
G_OFFER_ADD_VMT = "add_fleet_vmt"   # for easyride broker
G_OFFER_BROKER_FLAG = "chosen_by_broker"    # for easyride broker

# additional parameters for intermodal solutions
# ----------------------------------------------
G_IM_OFFER_PT_START = "im_pt_t_start"
G_IM_OFFER_PT_END = "im_pt_t_end"
G_IM_OFFER_PT_COST = "im_pt_fare"
G_IM_OFFER_MOD_DRIVE = "im_mod_t_drive"
G_IM_OFFER_MOD_COST = "im_mod_fare"
G_IM_OFFER_MOD_SUB = "im_mod_subsidy"

# -------------------------------------------------------------------------------------------------------------------- #
# Fleet Simulation Pattern
# ########################

# Vehicle General
# ---------------
G_V_OP_ID = "operator_id"
G_V_VID = "vehicle_id"
G_V_TYPE = "vehicle_type"

# Vehicle Final/Init Status
# -------------------------
G_V_INIT_NODE = "final_node_index"
G_V_INIT_TIME = "final_time"
G_V_INIT_SOC = "final_soc"

# Vehicle Status
# --------------
class VRL_STATES(Enum):
    BLOCKED_INIT = (-1, "blocked_init_status")
    IDLE = (0, "idle")
    BOARDING = (1, "boarding")
    CHARGING = (2, "charging")
    BOARDING_WITH_CHARGING = (3, "boarding_with_charging")
    WAITING = (4, "waiting")
    OUT_OF_SERVICE = (5, "out_of_service")
    PLANNED_STOP = (6, "planned_stop")    # TODO whats that for?
    REPO_TARGET = (7, "repositioning_target")
    ROUTE = (10, "route")
    REPOSITION = (11, "reposition")
    TO_CHARGE = (12, "to_charge")
    TO_DEPOT = (13, "to_depot")
    TO_RESERVATION = (14, "to_reservation")

    @DynamicClassAttribute
    def value(self):
        return self._value_[0]

    @DynamicClassAttribute
    def display_name(self):
        return self._value_[1]

    @staticmethod
    def G_VEHICLE_STATUS_DICT() -> dict:
        # print("WARNING: G_VEHICLE_STATUS_DICT is still accessed! (misc.globals)")
        return {status.value: status.display_name for status in VRL_STATES}

G_DRIVING_STATUS = [VRL_STATES.ROUTE, VRL_STATES.REPOSITION, VRL_STATES.TO_CHARGE, VRL_STATES.TO_DEPOT, VRL_STATES.TO_RESERVATION] # [10,11,12,13]
G_REVENUE_STATUS = [VRL_STATES.BOARDING, VRL_STATES.WAITING, VRL_STATES.ROUTE, VRL_STATES.REPOSITION] # [1, 4, 10, 11]
G_LAZY_STATUS = [VRL_STATES.WAITING] # [4]     # VRLs not actively planned and dont do anything (i.e. waiting)
G_LOCK_DURATION_STATUS = [VRL_STATES.BLOCKED_INIT, VRL_STATES.BOARDING, VRL_STATES.BOARDING_WITH_CHARGING] # [-1, 1, 3]
G_INACTIVE_STATUS = [VRL_STATES.OUT_OF_SERVICE, VRL_STATES.TO_DEPOT, VRL_STATES.CHARGING, VRL_STATES.TO_CHARGE]

# TODO # after ISTTT: define all vehicle states

# Vehicle Record
# --------------
# TODO # define output format
G_VR_STATUS = "status"
G_VR_LOCKED = "locked"
G_VR_LEG_START_TIME = "start_time"
G_VR_LEG_END_TIME = "end_time"
G_VR_LEG_START_POS = "start_pos"
G_VR_LEG_END_POS = "end_pos"
G_VR_LEG_DISTANCE = "driven_distance"
G_VR_LEG_START_SOC = "start_soc"
G_VR_LEG_END_SOC = "end_soc"
G_VR_NR_PAX = "occupancy"
G_VR_OB_RID = "rq_on_board"
G_VR_BOARDING_RID = "rq_boarding"
G_VR_ALIGHTING_RID = "rq_alighting"
G_VR_NODE_LIST = "route"
G_VR_REPLAY_ROUTE = "trajectory"
G_VR_TOLL = "toll"
G_VR_CHARGING_UNIT = "charging_unit"
G_VR_CHARGING_POWER = "charging_power"

# Vehicle Type
# ------------
G_VTYPE_NAME = "vtype_name_full"
G_VTYPE_MAX_PAX = "maximum_passengers"
G_VTYPE_FIX_COST = "daily_fix_cost [cent]"
G_VTYPE_DIST_COST = "per_km_cost [cent]"
G_VTYPE_BATTERY_SIZE = "battery_size [kWh]"
G_VTYPE_RANGE = "range [km]"
G_VTYPE_HOME_CHARGING_POWER = "home charging power [kW]"
G_VTYPE_MAX_PARCELS = "maximum_parcels"


# -------------------------------------------------------------------------------------------------------------------- #
# Fleet Control
# #############

# Departure plan
G_FCTRL_PRQ = "PlanRequest"

class G_PLANSTOP_STATES(Enum):
    """ this enum is used to identify different planstop states 
    MIXED : multiple tasks at the plan stop (i.e. boarding + charging)
    BOARDING : only boarding processes at plan stop
    REPO_TARGET: only routing target for repositioning
    CHARGING: only charging processes at target
    INACTIVE: vehicle supposed to be inactive at planstop
    RESERVATION: routing target for future reservation
    """
    MIXED : int = 0
    BOARDING : int = 1
    REPO_TARGET : int = 2
    CHARGING : int = 3
    INACTIVE : int = 4
    RESERVATION : int = 5

# Plan Request States
G_PRQS_NO_OFFER = 1
G_PRQS_INIT_OFFER = 2
G_PRQS_ACC_OFFER = 3
G_PRQS_LOCKED = 4
G_PRQS_IN_VEH = 5

# Computation Time Record
G_FCTRL_CT_RQU = "user_requests"
G_FCTRL_CT_RQB = "request_batch"
G_FCTRL_CT_CH = "charging_strategy"
G_FCTRL_CT_REPO = "repositioning_strategy"
G_FCTRL_CT_DP = "dyn_pricing_strategy"
G_FCTRL_CT_DFS = "dyn_fleetsizing_strategy"
G_FCTRL_CT_RES = "reservation_time_trigger"

#--------------------------------------------------------------------------------------------------------------#
# Evaluation specific params
# ####################

# only evaluate data within specific interval
G_EVAL_INT_START = "evaluation_int_start"
G_EVAL_INT_END = "evaluation_int_end"

# -------------------------------------------------------------------------------------------------------------------- #
# load functions


def load_scenario_inputs(output_dir):
    """This function reads the scenario config file, which enables access to all input parameters for evaluations etc.

    :param output_dir: scenario output directory
    :return: scenario_parameter dictionary, list_op_attributes, directory dictionary
    """
    config_f = os.path.join(output_dir, G_SC_INP_F)
    fhin = open(config_f)
    tmp = json.load(fhin)
    fhin.close()
    scenario_parameters = tmp["scenario_parameters"]
    list_operator_attributes = tmp["list_operator_attributes"]
    dir_names = tmp["directories"]
    return scenario_parameters, list_operator_attributes, dir_names


def get_directory_dict(scenario_parameters):
    """
    This function provides the correct paths to certain data according to the specified data directory structure.
    :param scenario_parameters: simulation input (pandas series)
    :return: dictionary with paths to the respective data directories
    """
    study_name = scenario_parameters[G_STUDY_NAME]
    scenario_name = scenario_parameters[G_SCENARIO_NAME]
    network_name = scenario_parameters[G_NETWORK_NAME]
    demand_name = scenario_parameters[G_DEMAND_NAME]
    zone_name = scenario_parameters.get(G_ZONE_SYSTEM_NAME, None)
    fc_type = scenario_parameters.get(G_FC_TYPE, None)
    fc_t_res = scenario_parameters.get(G_FC_TR, None)
    gtfs_name = scenario_parameters.get(G_GTFS_NAME, None)
    infra_name = scenario_parameters.get(G_INFRA_NAME, None)
    parcel_demand_name = scenario_parameters.get(G_PA_DEMAND_NAME, None)
    #
    dirs = {}
    dirs[G_DIR_MAIN] = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
    dirs[G_DIR_DATA] = os.path.join(dirs[G_DIR_MAIN], "data")
    dirs[G_DIR_OUTPUT] = os.path.join(dirs[G_DIR_MAIN], "studies", study_name, "results", scenario_name)
    dirs[G_DIR_NETWORK] = os.path.join(dirs[G_DIR_DATA], "networks", network_name)
    dirs[G_DIR_VEH] = os.path.join(dirs[G_DIR_DATA], "vehicles")
    dirs[G_DIR_FCTRL] = os.path.join(dirs[G_DIR_DATA], "fleetctrl")
    dirs[G_DIR_DEMAND] = os.path.join(dirs[G_DIR_DATA], "demand", demand_name, "matched", network_name)
    if zone_name is not None:
        dirs[G_DIR_ZONES] = os.path.join(dirs[G_DIR_DATA], "zones", zone_name, network_name)
        if fc_type is not None and fc_t_res is not None:
            dirs[G_DIR_FC] = os.path.join(dirs[G_DIR_DATA], "demand", demand_name, "aggregated", zone_name, str(fc_t_res))
    if gtfs_name is not None:
        dirs[G_DIR_PT] = os.path.join(dirs[G_DIR_DATA], "pubtrans", gtfs_name)
    if infra_name is not None:
        dirs[G_DIR_INFRA] = os.path.join(dirs[G_DIR_DATA], "infra", infra_name, network_name)
    if parcel_demand_name is not None:
        dirs[G_DIR_PARCEL_DEMAND] = os.path.join(dirs[G_DIR_DATA], "demand", parcel_demand_name, "matched", network_name)
    return dirs

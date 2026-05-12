# ----------------
import sys
import os
import traceback
import pandas as pd
import multiprocessing as mp

to_del = []
for p in os.sys.path:
    if p.endswith("FleetPy"):
        to_del.append(p)
for p in to_del:
    os.sys.path.remove(p)
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# src imports
# -----------
from src.misc.globals import *
from run_scenarios import run_scenarios

# ====================================================================================================================== #
# --------------------------------------> READ AND CREATE BENCHMARKS <----------------------------------------------------------- #

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

def get_complete_results_df():
    """This function reads all standard_eval.csv files from the results folder of the studies and combines them into a
    single data frame.

    :return: complete data frame of all standard_eval.csv files
    :rtype: DataFrame
    """
    scs = [d for d in os.listdir(os.path.join(STUDY_DIR, "results"))]
    list_temp_dfs = []
    for sc in scs:
        results_dir = os.path.join(STUDY_DIR, "results", sc)
        standard_eval_f = os.path.join(results_dir, "standard_eval.csv")
        if os.path.exists(standard_eval_f):
            tmp_df = pd.read_csv(standard_eval_f, index_col=0)
            new_cols = [f"{sc}_{c}" for c in tmp_df.columns]
            tmp_df.columns = new_cols
            list_temp_dfs.append(tmp_df)
        
    complete_results_df = pd.concat(list_temp_dfs, axis=1)    
    return complete_results_df

def _read_benchmark_file():
    """This function reads the benchmark file from the results folder of the studies.

    :return: benchmark data frame
    :rtype: DataFrame
    """
    return pd.read_csv(os.path.join(STUDY_DIR, "results", "benchmark.csv"), index_col=0)

def _overwrite_benchmark_file():
    """This function creates a benchmark file from the standard_eval.csv files in the results folder of the studies.

    """
    print("WARNING: Overwriting benchmark file!")
    complete_results_df = get_complete_results_df()
    complete_results_df.to_csv(os.path.join(STUDY_DIR, "results", "benchmark.csv"))
    print("Benchmark file created! -> ", os.path.join(STUDY_DIR, "results", "benchmark.csv"))
    
def compare_results_to_benchmark():
    """This function compares the results of the standard_eval.csv files in the results folder of the studies to the
    benchmark file to evaluate if code updates change the results of previous modules.

    """
    benchmark_df = _read_benchmark_file()
    complete_results_df = get_complete_results_df()
    for c in complete_results_df.columns:
        if c in benchmark_df.columns:
            complete_results_df[c] = complete_results_df[c] - benchmark_df[c]
    complete_results_df.to_csv(os.path.join(STUDY_DIR, "benchmark_comparison.csv"))
    print("created benchmark comparison file -> ", os.path.join(STUDY_DIR, "benchmark_comparison.csv"))

# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------- module tests ----------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

def run_module_test_simulations(N_PARALLEL_SIM=1):
    """This function runs the module test simulations for the studies.

    """
    scs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")
    cc = os.path.join(scs_path, "constant_config.csv")
    log_level = "info"
    
    # Test Simulation Environment Modules
    print("Test Simulation Environment Modules ...")
    sc = os.path.join(scs_path, "sc_config_sim_envs.csv")
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    print(" => Test Simulation Environment Modules completed!")
    
    # Test Multi-Operator Modules
    print("Test Multi-Operator Modules ...")
    sc = os.path.join(scs_path, "sc_config_multi_op.csv")
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    print(" => Test Multi-Operator Modules completed!")
    
    # Test Routing Modules
    print("Test Routing Modules ...")
    sc = os.path.join(scs_path, "sc_config_routing.csv")
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    print(" => Test Routing Modules completed!")
    
    # Test Fleetcontrol Parameters
    print("Test Fleetcontrol Parameters ...")
    sc = os.path.join(scs_path, "sc_config_fc_params.csv")
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    print(" => Test Fleetcontrol Parameters completed!")
    
    # Test Fleetcontrol Modules
    print("Test RPP Fleetcontrol Modules ...")
    sc = os.path.join(scs_path, "sc_config_rpp.csv")
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    print(" => Test RPP Fleetcontrol Modules completed!")
    
    # Test Repositioning Modules
    print("Test Repositioning Modules ...")
    sc = os.path.join(scs_path, "sc_config_repo.csv")
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    # TODO frontiers ist not working anymore
    print(" => Test Repositioning Modules completed!")
    
    # Test Charging Modules
    print("Test Charging Modules ...")
    sc = os.path.join(scs_path, "sc_config_charging.csv")
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    print(" => Test Charging Modules completed!")
    
    # Test dynamic pricing Modules
    print("Test Dynamic Pricing Modules ...")
    sc = os.path.join(scs_path, "sc_config_dyn_price.csv")
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    # TODO time based not implemented
    print(" => Test Dynamic Pricing Modules completed!")
    
    # Test Dynamic Fleet Sizng Modules
    print("Test Dynamic Fleet Sizing Modules ...")
    sc = os.path.join(scs_path, "sc_config_dyn_fleet_size.csv")
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    print(" => Test Dynamic Fleet Sizing Modules completed!")

    # Test Reservation Modules
    print("Test Reservation Modules ...")
    sc = os.path.join(scs_path, "sc_config_reservation.csv")
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    print(" => Test Reservation Modules completed!")
    
    # Test Ridepooling Batch Optimization Modules
    print("Test Ridepooling Batch Optimization Modules ...")
    sc = os.path.join(scs_path, "sc_config_rp_batch_opt.csv")
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    print(" => Test Ridepooling Batch Optimization Modules completed!")
    
    # Test Forecasting Modules
    print("Test Forecasting Modules ...")
    sc = os.path.join(scs_path, "sc_config_forecasting.csv")
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    print(" => Test Forecasting Modules completed!")
    
    # Test SoD Max Modules
    # TODO test after upgrade
    # print("Test SoD Max Modules ...")
    # cc_sod = os.path.join(scs_path, "constant_config_sod.csv")
    # sc = os.path.join(scs_path, "sc_config_sod_max.csv")
    # run_scenarios(cc_sod, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    # print(" => Test SoD Max Modules completed!")
    

    


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    mp.freeze_support()
    N_PARALLEL_SIM = 1

    print("Run module tests ...")
    run_module_test_simulations(N_PARALLEL_SIM=N_PARALLEL_SIM)
    
    print(" => all module tests completed!")

    print("... compare results to benchmark ...")

    compare_results_to_benchmark()
    
    print("... done!")
    print(" => check benchmark_comparison.csv for changes in results!")
    

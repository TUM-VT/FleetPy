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
import src.misc.config as config
from src.misc.init_modules import load_simulation_environment
from src.misc.globals import *

# main functions
# --------------
def run_single_simulation(scenario_parameters):
    SF = load_simulation_environment(scenario_parameters)
    if scenario_parameters.get("bugfix", False):
        try:
            SF.run()
        except:
            traceback.print_exc()
    else:
        SF.run()


def run_scenarios(constant_config_file, scenario_file, n_parallel_sim=1, n_cpu_per_sim=1, evaluate=1, log_level="info",
                  keep_old=False, continue_next_after_error=False):
    """
    This function combines constant study parameters and scenario parameters.
    Then it sets up a pool of workers and starts a simulation for each scenario.
    The required parameters are stated in the documentation.

    :param constant_config_file: this file contains all input parameters that remain constant for a study
    :type constant_config_file: str
    :param scenario_file: this file contain all input parameters that are varied for a study
    :type scenario_file: str
    :param n_parallel_sim: number of parallel simulation processes
    :type n_parallel_sim: int
    :param n_cpu_per_sim: number of cpus for a single simulation
    :type n_cpu_per_sim: int
    :param evaluate: 0: no automatic evaluation / != 0 automatic simulation after each simulation
    :type evaluate: int
    :param log_level: hierarchical output to the logging file. Possible inputs with hierarchy from low to high:
            - "verbose": lowest level -> logs everything; even code which could scale exponentially
            - "debug": standard debugging logger. code which scales exponentially should not be logged here
            - "info": basic information during simulations (default)
            - "warning": only logs warnings
    :type log_level: str
    :param keep_old: does not start new simulation if result files are already available in scenario output directory
    :type keep_old: bool
    :param continue_next_after_error: continue with next simulation if one the simulations threw an error (only SP)
    :type continue_next_after_error: bool
    """
    assert type(n_parallel_sim) == int, "n_parallel_sim must be of type int"
    # read constant and scenario config files
    constant_cfg = config.ConstantConfig(constant_config_file)
    scenario_cfgs = config.ScenarioConfig(scenario_file)

    # set constant parameters from function arguments
    # TODO # get study name and check if its a studyname
    const_abs = os.path.abspath(constant_config_file)
    study_name = os.path.basename(os.path.dirname(os.path.dirname(const_abs)))

    if study_name == "scenarios":
        print("ERROR! The path of the config files is not longer up to date!")
        print("See documentation/Data_Directory_Structure.md for the updated directory structure needed as input!")
        exit()
    if constant_cfg.get(G_STUDY_NAME) is not None and study_name != constant_cfg.get(G_STUDY_NAME):
        print("ERROR! {} from constant config is not consistent with study directory: {}".format(constant_cfg[G_STUDY_NAME], study_name))
        print("{} is now given directly by the folder name !".format(G_STUDY_NAME))
        exit()
    constant_cfg[G_STUDY_NAME] = study_name

    constant_cfg["n_cpu_per_sim"] = n_cpu_per_sim
    constant_cfg["evaluate"] = evaluate
    constant_cfg["log_level"] = log_level
    constant_cfg["keep_old"] = keep_old

    # combine constant and scenario parameters into verbose scenario parameters
    for i, scenario_cfg in enumerate(scenario_cfgs):
        scenario_cfgs[i] = constant_cfg + scenario_cfg

    # perform simulation(s)
    print(f"Simulation of {len(scenario_cfgs)} scenarios on {n_parallel_sim} processes with {n_cpu_per_sim} cpus per simulation ...")
    if n_parallel_sim == 1:
        for scenario_cfg in scenario_cfgs:
            if continue_next_after_error:
                try:
                    run_single_simulation(scenario_cfg)
                except:
                    traceback.print_exc()
            else:
                run_single_simulation(scenario_cfg)
    else:
        if n_cpu_per_sim == 1:
            mp_pool = mp.Pool(n_parallel_sim)
            mp_pool.map(run_single_simulation, scenario_cfgs)
        else:
            n_scenarios = len(scenario_cfgs)
            rest_scenarios = n_scenarios
            current_scenario = 0
            while rest_scenarios != 0:
                if rest_scenarios >= n_parallel_sim:
                    par_processes = [None for i in range(n_parallel_sim)]
                    for i in range(n_parallel_sim):
                        par_processes[i] = mp.Process(target=run_single_simulation, args=(scenario_cfgs[current_scenario],))
                        current_scenario += 1
                        par_processes[i].start()
                    for i in range(n_parallel_sim):
                        par_processes[i].join()
                        rest_scenarios -= 1
                else:
                    par_processes = [None for i in range(rest_scenarios)]
                    for i in range(rest_scenarios):
                        par_processes[i] = mp.Process(target=run_single_simulation, args=(scenario_cfgs[current_scenario],))
                        current_scenario += 1
                        par_processes[i].start()
                    for i in range(rest_scenarios):
                        par_processes[i].join()
                        rest_scenarios -= 1

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
    
    # Test Rq Starting on edges
    print("Test Simulation RQ Edge start/end ...")
    sc = os.path.join(scs_path, "sc_config_rq_pos.csv")
    log_level="debug"
    run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=N_PARALLEL_SIM)
    print(" => Test Simulation RQ Edge start/end completed!")
    
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
    

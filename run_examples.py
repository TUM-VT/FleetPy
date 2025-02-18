# -------------------------------------------------------------------------------------------------------------------- #
# external imports
# ----------------
import sys
import traceback
import pandas as pd
import multiprocessing as mp

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

# -------------------------------------------------------------------------------------------------------------------- #
# ----> you can replace the following part by your respective if __name__ == '__main__' part for run_private*.py <---- #
# -------------------------------------------------------------------------------------------------------------------- #

# global variables for testing
# ----------------------------
MAIN_DIR = os.path.dirname(__file__)
MOD_STR = "MoD_0"
MM_STR = "Assertion"
LOG_F = "standard_bugfix.log"


# testing results of examples
# ---------------------------
def read_outputs_for_comparison(constant_csv, scenario_csv):
    """This function reads some output parameters for a test of meaningful results of the test cases.

    :param constant_csv: constant parameter definition
    :param scenario_csv: scenario definition
    :return: list of standard_eval data frames
    :rtype: list[DataFrame]
    """
    constant_cfg = config.ConstantConfig(constant_csv)
    scenario_cfgs = config.ScenarioConfig(scenario_csv)
    const_abs = os.path.abspath(constant_csv)
    study_name = os.path.basename(os.path.dirname(os.path.dirname(const_abs)))
    return_list = []
    for scenario_cfg in scenario_cfgs:
        complete_scenario_cfg = constant_cfg + scenario_cfg
        scenario_name = complete_scenario_cfg[G_SCENARIO_NAME]
        output_dir = os.path.join(MAIN_DIR, "studies", study_name, "results", scenario_name)
        standard_eval_f = os.path.join(output_dir, "standard_eval.csv")
        tmp_df = pd.read_csv(standard_eval_f, index_col=0)
        tmp_df.loc[G_SCENARIO_NAME, MOD_STR] = scenario_name
        return_list.append((tmp_df))
    return return_list


def check_assertions(list_eval_df, all_scenario_assertion_dict):
    """This function checks assertions of scenarios to give a quick impression if results are fitting.

    :param list_eval_df: list of evaluation data frames
    :param all_scenario_assertion_dict: dictionary of scenario id to assertion dictionaries
    :return: list of (scenario_name, mismatch_flag, tmp_df) tuples
    """
    list_result_tuples = []
    for sc_id, assertion_dict in all_scenario_assertion_dict.items():
        tmp_df = list_eval_df[sc_id]
        scenario_name = tmp_df.loc[G_SCENARIO_NAME, MOD_STR]
        print("-"*80)
        mismatch = False
        for k, v in assertion_dict.items():
            if tmp_df.loc[k, MOD_STR] != v:
                tmp_df.loc[k, MM_STR] = v
                mismatch = True
        if mismatch:
            prt_str = f"Scenario {scenario_name} has mismatch with assertions:/n{tmp_df}/n" + "-"*80 + "/n"
        else:
            prt_str = f"Scenario {scenario_name} results match assertions/n" + "-"*80 + "/n"
        print(prt_str)
        with open(LOG_F, "a") as fh:
            fh.write(prt_str)
        list_result_tuples.append((scenario_name, mismatch, tmp_df))
    return list_result_tuples


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    mp.freeze_support()

    if len(sys.argv) > 1:
        run_scenarios(*sys.argv)
    else:
        import time
        # touch log file
        with open(LOG_F, "w") as _:
            pass

        scs_path = os.path.join(os.path.dirname(__file__), "studies", "example_study", "scenarios")
        # Base Examples IRS only
        # ----------------------
        # a) Pooling in ImmediateOfferEnvironment
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_ir.csv")
        sc = os.path.join(scs_path, "example_ir_only.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        list_results = read_outputs_for_comparison(cc, sc)
        all_scenario_assert_dict = {0: {"number users": 88}}
        check_assertions(list_results, all_scenario_assert_dict)

        # Base Examples with Optimization (requires gurobi license!)
        # ----------------------------------------------------------
        # b) Pooling in BatchOffer environment
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_pool.csv")
        sc = os.path.join(scs_path, "example_pool.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        list_results = read_outputs_for_comparison(cc, sc)
        all_scenario_assert_dict = {0: {"number users": 91}}
        check_assertions(list_results, all_scenario_assert_dict)

        # c) Pooling in ImmediateOfferEnvironment
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_ir.csv")
        sc = os.path.join(scs_path, "example_ir_batch.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        list_results = read_outputs_for_comparison(cc, sc)
        all_scenario_assert_dict = {0: {"number users": 90}}
        check_assertions(list_results, all_scenario_assert_dict)

        # d) Pooling with RV heuristics in ImmediateOfferEnvironment (with doubled demand)
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_ir.csv")
        t0 = time.perf_counter()
        # no heuristic scenario
        sc = os.path.join(scs_path, "example_pool_noheuristics.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        list_results = read_outputs_for_comparison(cc, sc)
        all_scenario_assert_dict = {0: {"number users": 199}}
        check_assertions(list_results, all_scenario_assert_dict)
        # with heuristic scenarios
        t1 = time.perf_counter()
        sc = os.path.join(scs_path, "example_pool_heuristics.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        list_results = read_outputs_for_comparison(cc, sc)
        t2 = time.perf_counter()
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=2)
        t3 = time.perf_counter()
        print(f"Computation time without heuristics: {round(t1-t0, 1)} | with heuristics 1 CPU: {round(t2-t1,1)}"
              f"| with heuristics 2 CPU: {round(t3-t2,1)}")
        all_scenario_assert_dict = {0: {"number users": 191}}
        check_assertions(list_results, all_scenario_assert_dict)

        # g) Pooling with RV heuristic and Repositioning in ImmediateOfferEnvironment (with doubled demand and
        #       bad initial vehicle distribution)
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_ir_repo.csv")
        sc = os.path.join(scs_path, "example_ir_heuristics_repositioning.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        list_results = read_outputs_for_comparison(cc, sc)
        all_scenario_assert_dict = {0: {"number users": 198}}
        check_assertions(list_results, all_scenario_assert_dict)
        
        # h) Pooling with public charging infrastructure (low range vehicles)
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_charge.csv")
        sc = os.path.join(scs_path, "example_charge.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        
        # i) Pooling and active vehicle fleet size is controlled externally (time and utilization based)
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_depot.csv")
        sc = os.path.join(scs_path, "example_depot.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        
        # j) Pooling with public charging and fleet size control (low range vehicles)
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_depot_charge.csv")
        sc = os.path.join(scs_path, "example_depot_charge.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        
        # h) Pooling with multiprocessing
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_depot_charge.csv")
        # no heuristic scenario single core
        t0 = time.perf_counter()
        sc = os.path.join(scs_path, "example_depot_charge.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        list_results = read_outputs_for_comparison(cc, sc)
        all_scenario_assert_dict = {0: {"number users": 199}}
        check_assertions(list_results, all_scenario_assert_dict)
        print("Computation without multiprocessing took {}s".format(time.perf_counter() - t0))
        # no heuristic scenario multiple cores
        cores = 2
        t0 = time.perf_counter()
        sc = os.path.join(scs_path, "example_depot_charge.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=cores, n_parallel_sim=1)
        list_results = read_outputs_for_comparison(cc, sc)
        all_scenario_assert_dict = {0: {"number users": 199}}
        check_assertions(list_results, all_scenario_assert_dict)
        print("Computation with multiprocessing took {}s".format(time.perf_counter() - t0))
        print(" -> multiprocessing only usefull for large vehicle fleets")
        
        # j) Pooling - multiple operators and broker
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_broker.csv")
        sc = os.path.join(scs_path, "example_broker.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        
        # h) Ride-Parcel-Pooling example
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_rpp.csv")
        sc = os.path.join(scs_path, "example_rpp.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)

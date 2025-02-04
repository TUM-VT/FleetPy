import os
import traceback
import multiprocessing as mp

# the following two lines are only necessary if you have your run file within your study directory
# import sys
# sys.path.append("../..")

from run_examples import run_scenarios


if __name__ == "__main__":
    mp.freeze_support()

    # ------- #
    # study 1 #
    # ------- #



    # the following lines are necessary if you run from the main directory, where 

    study_name = "fleetpy_sumo_coupling_in"
    const_fn = "constant_config.csv"
    sc_fn = "300_sumo_in_0.02_SUMOcontrolledSim.csv"
    constant_config_file = f"studies/{study_name}/scenarios/{const_fn}"
    scenario_file = f"studies/{study_name}/scenarios/{sc_fn}"
    

    # otherwise, you can use these lines
    # constant_config_file = f"scenarios/{const_fn}"
    ## scenario_file = f"scenarios/{sc_fn}"
    
    if not os.path.isfile(constant_config_file) or not os.path.isfile(scenario_file):
        raise IOError("Invalid paths to config files! Script has to be placed in main directory!")
    

    # -------------- #
    # other settings #
    # -------------- #
    evaluate=1
    log_level="info"
    n_parallel_sim = 1
    n_cpu_per_sim = 1

    try:
        run_scenarios(constant_config_file, scenario_file, n_parallel_sim, n_cpu_per_sim, evaluate, log_level,
                      continue_next_after_error=True)
    except:
        traceback.print_exc()



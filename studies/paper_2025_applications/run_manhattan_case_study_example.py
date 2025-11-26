import multiprocessing as mp
import os
import sys
import traceback

fleetpy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(fleetpy_path)

from run_examples import run_scenarios
from src.misc.globals import *

SC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")

if __name__ == "__main__":
    mp.freeze_support()

    try:
        # Example 1 - Batch assignment
        log_level = "info"
        cc = os.path.join(SC_PATH, "const_cfg_ex1_batch_assignment_manhattan.yaml")
        sc = os.path.join(SC_PATH, "scenario_cfg_ex1_batch_assignment_manhattan.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)

        # Example 2 - Multiple operators
        log_level = "info"
        cc = os.path.join(SC_PATH, "const_cfg_ex2_n_operator_manhattan.csv")
        sc = os.path.join(SC_PATH, "scenario_cfg_ex2_n_operator_manhattan.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
    except:
        traceback.print_exc()

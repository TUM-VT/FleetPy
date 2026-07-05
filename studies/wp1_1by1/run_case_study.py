import os
import multiprocessing as mp
import sys
from time import time
import traceback

fleetpy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(fleetpy_path)

from run_scenarios import run_scenarios

SC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")

if __name__ == "__main__":
    mp.freeze_support()

    try:
        # calculate computation time

        start_time = time()
        cc = os.path.join(SC_PATH, "const_cfg.yaml")
        sc = os.path.join(SC_PATH, "scenario_cfg.csv")
        run_scenarios(cc, sc, log_level="info", n_cpu_per_sim=1, n_parallel_sim=1)
        end_time = time()
        print(f"Computation time: {end_time - start_time} seconds")
    except:
        traceback.print_exc()

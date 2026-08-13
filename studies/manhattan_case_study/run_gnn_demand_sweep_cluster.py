import os
import sys
import multiprocessing as mp
import traceback

fleetpy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(fleetpy_path)

from run_examples import run_scenarios

SC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")

if __name__ == "__main__":
    mp.freeze_support()

    try:
        cc = os.path.join(SC_PATH, "const_cfg_manhattan_case_study.yaml")
        sc = os.path.join(SC_PATH, "sc_cfg_gnn_demand_sweep_cluster.csv")
        run_scenarios(cc, sc, log_level="info", n_cpu_per_sim=1, n_parallel_sim=2,
                      continue_next_after_error=True)
    except:
        traceback.print_exc()

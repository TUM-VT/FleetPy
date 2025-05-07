import os
import pandas as pd
import multiprocessing as mp
import sys
import traceback

fleetpy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(fleetpy_path)

from run_examples import run_scenarios
import src.misc.config as config
from src.misc.globals import *

SC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")

if __name__ == "__main__":
    mp.freeze_support()

    try:
        log_level = "info"
        cc = os.path.join(SC_PATH, r"const_cfg_chicago_case_study.yaml")
        sc = os.path.join(SC_PATH, r"scenario_cfg_chicago_case_study.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
    except:
        traceback.print_exc()

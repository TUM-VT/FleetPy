import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))) # add fleetpy path

from src.ml_gym.MLEnv import MLEnv
from src.ml_gym.HookManager import HookManager

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
def run_fleetpy_sim(scenario_parameters, fleetpy_in_queue, fleetpy_out_queue):
    hook_manager = HookManager(fleetpy_in_queue, fleetpy_out_queue)
    SF = load_simulation_environment(scenario_parameters,  hook_manager=hook_manager)
    try:
        SF.run()
    except:
        traceback.print_exc()


def run_fleetpy_gym(scenario_parameters):
    ### set up multiprocessing queues and hook manager for ML integration
    fleetpy_in_queue, fleetpy_out_queue = mp.Queue(), mp.Queue()
    
    ### start process for fleetpy simulation
    fleetpy_process = mp.Process(target=run_fleetpy_sim, args=(scenario_parameters, fleetpy_in_queue, fleetpy_out_queue))
    fleetpy_process.start()

    ### set up ML environment
    ml_env = MLEnv(fleetpy_in_queue, fleetpy_out_queue)
    ml_env.run()
    
    fleetpy_process.join()
    
    
if __name__ == "__main__":
    sc_config = r"C:\Users\ge37ser\Documents\Coding\FleetPy\studies\MLtest\scenarios\sc_config_repo.csv"
    const_config = r"C:\Users\ge37ser\Documents\Coding\FleetPy\studies\MLtest\scenarios\constant_config.csv"
    
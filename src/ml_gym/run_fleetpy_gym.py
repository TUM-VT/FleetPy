import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))) # add fleetpy path

from src.ml_gym.MLEnvs.MLEnv import MLEnv
from src.ml_gym.Hooks.HookManager import HookManager

# -------------------------------------------------------------------------------------------------------------------- #
# external imports
# ----------------
import traceback
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
    SF = load_simulation_environment(scenario_parameters, hook_manager=hook_manager)
    try:
        SF.run()
    except:
        traceback.print_exc()
    finally:
        # sentinel to signal MLEnv that simulation has ended
        fleetpy_out_queue.put('SIMULATION_ENDED')


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
    mp.freeze_support()

    MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if len(sys.argv) >= 3:
        const_config = sys.argv[1]
        sc_config = sys.argv[2]
    else:
        # default: ml_test study
        scs_path = os.path.join(MAIN_DIR, "studies", "ml_test", "scenarios")
        const_config = os.path.join(scs_path, "constant_config_ir.csv")
        sc_config = os.path.join(scs_path, "example_sl_ir_only.csv")

    constant_cfg = config.ConstantConfig(const_config)
    scenario_cfgs = config.ScenarioConfig(sc_config)

    study_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(const_config))))
    constant_cfg[G_STUDY_NAME] = study_name
    constant_cfg["n_cpu_per_sim"] = 1
    constant_cfg["evaluate"] = 1
    constant_cfg["log_level"] = "info"

    for scenario_cfg in scenario_cfgs:
        scenario_parameters = constant_cfg + scenario_cfg
        run_fleetpy_gym(scenario_parameters)

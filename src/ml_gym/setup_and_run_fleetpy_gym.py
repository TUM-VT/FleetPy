import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))) # add fleetpy path

from src.ml_gym.FleetPyMLInterface import FleetPyMLInterface 
from src.misc.globals import *

from src.misc.init_modules import load_simulation_environment
from src.ml_gym.MLEnvs.RandomRepositioningEnv import RandomRepositioningEnv
from src.ml_gym.Hooks.HookManager import HookManager, MLHook, Events
import src.misc.config as config

import traceback
import multiprocessing as mp

if __name__ == "__main__":

    MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if len(sys.argv) >= 3:
        const_config = sys.argv[1]
        sc_config = sys.argv[2]
    else:
        # default: ml_test study
        scs_path = os.path.join(MAIN_DIR, "studies", "MLtest", "scenarios")
        const_config = os.path.join(scs_path, "constant_config.csv")
        sc_config = os.path.join(scs_path, "sc_config_repo.csv")

    constant_cfg = config.ConstantConfig(const_config)
    scenario_cfgs = config.ScenarioConfig(sc_config)

    study_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(const_config))))
    constant_cfg[G_STUDY_NAME] = study_name
    constant_cfg["n_cpu_per_sim"] = 1
    constant_cfg["evaluate"] = 1
    constant_cfg["log_level"] = "info"

    fleetpy_config = constant_cfg + scenario_cfgs[0]
    
    # init ML environment
    ml_environment = RandomRepositioningEnv()
    
    # init FleetPyMLInterface
    fp_ml_interface = FleetPyMLInterface(fleetpy_config, ml_environment, multiprocessing=False)
    
    # define event for interaction between FleetPy and ML environment
    event = Events.OBSERVE_BEFORE_REPOSITIONING
    # register observers
    from src.ml_gym.Observers.repositioning_observers import observe_sim_time, observe_demand_forecast, observe_zonebase_vehicle_states
    fp_ml_interface.register_observer(event, observe_sim_time)
    fp_ml_interface.register_observer(event, observe_demand_forecast)
    fp_ml_interface.register_observer(event, observe_zonebase_vehicle_states)
    # register actor
    from src.ml_gym.Actors.repositioning_actors import apply_od_assignment
    fp_ml_interface.register_actor(event, apply_od_assignment)
    
    # run interface
    try:
        fp_ml_interface.run()
    except Exception as e:
        print("Error in FleetPyMLInterface:")
        traceback.print_exc()
import sys
import os 
from src.misc.globals import *

from src.misc.init_modules import load_simulation_environment
from src.ml_gym.MLEnv import MLEnv, GreedyRepositioningEnv
from src.ml_gym.HookManager import HookManager, MLHook, Events
import src.misc.config as config

import traceback
import multiprocessing as mp


class FleetPyMLInterface:
    def __init__(self, scenario_parameters, ml_environment: MLEnv, multiprocessing=False):
        if multiprocessing:
            self.fleetpy_in_queue, self.fleetpy_out_queue = mp.Queue(), mp.Queue()
        else:
            self.fleetpy_in_queue, self.fleetpy_out_queue = None, None
        self.hook_manager = HookManager(self.fleetpy_in_queue, self.fleetpy_out_queue)
        self.ml_environment = ml_environment
        self.multiprocessing = multiprocessing
        self.scenario_parameters = scenario_parameters
                
    def register_observer(self, event, observer_method):
        ml_hooks = self.hook_manager.get_ml_hooks(event)
        if len(ml_hooks) == 0:
            ml_hook = MLHook(event, ml_interface=self)
            ml_hook.register_observer(observer_method)
            self.hook_manager.register(event, ml_hook)
        else:
            ml_hooks[0].register_observer(observer_method)
            
    def register_actor(self, event, actor_method):
        ml_hooks = self.hook_manager.get_ml_hooks(event)
        if len(ml_hooks) == 0:
            ml_hook = MLHook(event, ml_interface=self)
            ml_hook.register_actor(actor_method)
            self.hook_manager.register(event, ml_hook)
        else:
            ml_hooks[0].register_actor(actor_method)
            
    def communicate(self, event, observation):
        # send observation to ML environment and get action
        if self.multiprocessing:
            self.fleetpy_out_queue.put(observation)
            action = self.fleetpy_in_queue.get()
        else:
            self.ml_environment.receive_observation(event, observation)
            action = self.ml_environment.get_action()
        return action
    
    def run(self):
        # start ML environment (in separate process if multiprocessing is enabled)
        if self.multiprocessing:
            ml_process = mp.Process(target=self.ml_environment.run)
            ml_process.start()
        
        # start FleetPy simulation with registered hooks
        SF = load_simulation_environment(self.scenario_parameters, self.hook_manager)
        SF.run()
        
        # wait for ML process to finish
        if self.multiprocessing:
            ml_process.join()
        
    
        
        
if __name__ == "__main__":

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

    fleetpy_config = const_config + scenario_cfgs[0]
    
    # init ML environment
    ml_environment = GreedyRepositioningEnv()
    
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
    
    
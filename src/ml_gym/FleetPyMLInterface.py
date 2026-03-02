import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))) # add fleetpy path
from src.misc.globals import *

from src.misc.init_modules import load_simulation_environment
from src.ml_gym.MLEnvs.MLEnv import MLEnv
from src.ml_gym.Hooks.HookManager import HookManager, MLHook, Events
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
    
    
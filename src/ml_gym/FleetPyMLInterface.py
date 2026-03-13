import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))) # add fleetpy path

from src.misc.init_modules import load_simulation_environment
from src.ml_gym.hooks_manager import HookManager, Events
from src.ml_gym.observers import AbstractObserver
from src.ml_gym.actors import AbstractActor
import multiprocessing as mp


class FleetPyMLInterface:
    def __init__(self, scenario_parameters, multiprocessing=False):
        if multiprocessing:
            self.fleetpy_in_queue, self.fleetpy_out_queue = mp.Queue(), mp.Queue()
        else:
            self.fleetpy_in_queue, self.fleetpy_out_queue = None, None
        self.hook_manager = HookManager(self.fleetpy_in_queue, self.fleetpy_out_queue)
        self.multiprocessing = multiprocessing
        self.scenario_parameters = scenario_parameters
                
    def register_observer(self, event: Events, observer: AbstractObserver):
        self.hook_manager.add_observer(event, observer)
            
    def register_actor(self, event: Events, actor: AbstractActor):
        self.hook_manager.add_actor(event, actor)

    def couple_actors_to_observers(self, event: Events, actors: list[AbstractActor], observers: list[AbstractObserver]):
        self.hook_manager.couple_actors_to_observers(event, actors, observers)
    
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
    
    
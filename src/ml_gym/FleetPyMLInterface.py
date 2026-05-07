import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add fleetpy path

from src.misc.init_modules import load_simulation_environment
from src.ml_gym.hooks_manager import HookManager, Events
from src.ml_gym.Observers import AbstractObserver
from src.ml_gym.Actors import AbstractActor
import multiprocessing as mp

from typing import List


def run_single_simulation(scenario_parameters, hooks_manager, process_id):
    SF = load_simulation_environment(scenario_parameters, hooks_manager, process_id)
    SF.run(process_id)


class FleetPyMLInterface:
    def __init__(self, scenario_parameters, nr_parallel=1):
        in_out_queue = {}
        if nr_parallel > 1:
            for process_id in range(nr_parallel):
                in_out_queue[process_id] = mp.Queue(), mp.Queue()
        self.hook_manager = HookManager(in_out_queue)
        self.scenario_parameters = scenario_parameters
        self.nr_parallel = nr_parallel
        self.fleetpy_process = []

    def register_observer(self, event: Events, observer: AbstractObserver):
        self.hook_manager.add_observer(event, observer)

    def register_actor(self, event: Events, actor: AbstractActor):
        self.hook_manager.add_actor(event, actor)

    def couple_actors_to_observers(self, event: Events, actors: List[AbstractActor], observers: List[AbstractObserver]):
        self.hook_manager.couple_actors_to_observers(event, actors, observers)

    def reply_to_slave_processes(self):
        self.hook_manager.reply_to_slave_processes()

    def run(self):
        # start ML environment (in separate process if multiprocessing is enabled)
        if self.nr_parallel > 1:
            for i in range(self.nr_parallel):
                ml_process = mp.Process(target=run_single_simulation,
                                        args=(self.scenario_parameters, self.hook_manager, i))
                self.fleetpy_process.append(ml_process)
                ml_process.start()
            alive_processes = self.fleetpy_process
            while len(alive_processes) > 0:
                self.reply_to_slave_processes()
                for process in alive_processes:
                    if process.is_alive() is False:
                        alive_processes.remove(process)

        else:
            # start FleetPy simulation with registered hooks
            SF = load_simulation_environment(self.scenario_parameters, self.hook_manager)
            SF.run()


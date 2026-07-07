import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add fleetpy path

from src.misc.init_modules import load_simulation_environment
from src.ml_gym.hooks_manager import HookManager, Events
from src.ml_gym.Observers import AbstractObserver
from src.ml_gym.Actors import AbstractActor
from threading import Thread
from queue import Queue, Empty
import gymnasium as gym
from abc import ABC, abstractmethod


def run_single_simulation(scenario_parameters, hooks_manager: HookManager, process_id: int):
    SF = load_simulation_environment(scenario_parameters, hooks_manager, process_id)
    SF.run(process_id)


class FleetPyGym(gym.Env, ABC):

    def __init__(self, scenario_parameters):
        in_out_queue = {0: (Queue(), Queue())}
        self._hook_manager = HookManager(in_out_queue)
        self.scenario_parameters = scenario_parameters
        self.last_observation = None
        self.last_reward = None
        self.last_action = None
        self._fleetpy_thread: Thread = None

    def register_observer(self, event: Events, observer: AbstractObserver):
        self._hook_manager.add_observer(event, observer)

    def register_actor(self, event: Events, actor: AbstractActor):
        self._hook_manager.add_actor(event, actor)

    def reset(self, *, seed=None, options=None):
        self._fleetpy_thread = Thread(target=run_single_simulation,
                                        args=(self.scenario_parameters, self._hook_manager, 0))
        self._fleetpy_thread.start()
        observation, actor_type = self._hook_manager.get_observations(process_id=0)

        self.last_observation = observation
        translated_observation = self.translate_observation(observation)
        return translated_observation, {}

    @abstractmethod
    def translate_observation(self, observation):
        """ Implement this method to translate FleetPy's raw observation into the desired format for the RL algorithm (e.g. a vector or dict of vectors) """
        pass

    @abstractmethod
    def reward(self, observation, action, actor_type):
        """ Implement this method to calculate the reward based on the received observation and the action taken by the agent """
        pass

    def step(self, action):
        self._hook_manager.send_actor_response(0, action)
        observation, reward = self.last_observation, self.last_reward
        done = False
        while True:
            try:
                observation, actor_type = self._hook_manager.get_observations(process_id=0, timeout=1)
                self.last_observation = observation
                reward = self.reward(observation, action, actor_type)
                self.last_reward = reward
                break
            except Empty:
                if not self._fleetpy_thread.is_alive():
                    print(f"FleetPy thread is dead")
                    done = True
                    break
        translated_observation = self.translate_observation(observation)
        return translated_observation, reward, done, False, {}



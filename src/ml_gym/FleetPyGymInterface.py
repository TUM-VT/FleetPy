import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add fleetpy path

from src.misc.init_modules import load_simulation_environment
from src.ml_gym.hooks_manager_gym import HookManager, Events
from src.ml_gym.Observers import AbstractObserver
from src.ml_gym.Actors import AbstractActor
from threading import Thread
from queue import Queue, Empty
import gymnasium as gym
from abc import ABC, abstractmethod


def run_single_simulation(scenario_parameters, hooks_manager, process_id):
    SF = load_simulation_environment(scenario_parameters, hooks_manager, process_id)
    SF.run(process_id)


class FleetPyGym(gym.Env, ABC):

    def __init__(self, scenario_parameters):
        self._comm_queue = Queue()
        self._hook_manager = HookManager(self._comm_queue)
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
        self.fleetpy_thread = Thread(target=run_single_simulation,
                                        args=(self.scenario_parameters, self._hook_manager, None))
        self.fleetpy_thread.start()
        observation = self._hook_manager.get_observations()

        self.last_observation = observation
        return observation, {}

    @abstractmethod
    def translate_observation(self, observation):
        """ Implement this method to translate FleetPy's raw observation into the desired format for the RL algorithm (e.g. a vector or dict of vectors) """
        pass

    @abstractmethod
    def reward(self, observation, action):
        """ Implement this method to calculate the reward based on the received observation and the action taken by the agent """
        pass

    def step(self, action):
        self._hook_manager.send_actor_response(action)
        observation, reward = self.last_observation, self.last_reward
        done = False
        while True:
            try:
                observation = self._hook_manager.get_observations(block=False, timeout=1)
                self.last_observation = observation
                reward = self.reward(observation, action)
            except Empty:
                if not self.fleetpy_thread.is_alive():
                    done = True
                    break
        self.translate_observation(observation)
        return observation, reward, done, False, {}



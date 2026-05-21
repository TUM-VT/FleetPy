import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add fleetpy path

from src.misc.init_modules import load_simulation_environment
from src.ml_gym.hooks_manager import HookManager, Events
from src.ml_gym.Observers import AbstractObserver
from src.ml_gym.Actors import AbstractActor
import threading
from queue import Queue, Empty
import gymnasium as gym
from abc import ABC, abstractmethod


def run_single_simulation(scenario_parameters, hooks_manager: HookManager, process_id: int):
    SF = load_simulation_environment(scenario_parameters, hooks_manager, process_id)
    SF.run(process_id)


class FleetPyGym(gym.Env, ABC):

    def __init__(self, scenario_parameters, process_id=0):
        in_out_queue = {process_id: (Queue(), Queue())}
        self._hook_manager = HookManager(in_out_queue)
        self.scenario_parameters = scenario_parameters
        self.last_observation = None
        self.last_reward = None
        self.last_action = None
        self.last_actor_type = None
        self.last_event: Events = None
        self._fleetpy_thread: threading.Thread = None
        self._process_id = process_id

    def register_observer(self, event: Events, observer: AbstractObserver):
        self._hook_manager.add_observer(event, observer)

    def register_actor(self, event: Events, actor: AbstractActor):
        self._hook_manager.add_actor(event, actor)

    def reset(self, *, seed=None, options=None):
        self._fleetpy_thread = threading.Thread(target=run_single_simulation,
                                        args=(self.scenario_parameters, self._hook_manager, self._process_id))
        self._fleetpy_thread.start()
        self.last_observation, self.last_actor_type, self.last_event = self._hook_manager.get_observations(process_id=self._process_id)
        translated_observation = self.translate_observation(self.last_observation, self.last_actor_type, self.last_event)
        return translated_observation, {}

    @abstractmethod
    def translate_observation(self, observation, actor_type, event: Events):
        """ Implement this method to translate FleetPy's raw observation into the desired format for the RL algorithm (e.g. a vector or dict of vectors) """
        pass

    @abstractmethod
    def translate_action(self, observation, action, actor_type, event: Events):
        """ Implement this method to translate the RL algorithm's output into the desired format for FleetPy.
        IMPORTANT:  The output of this method must be in the same format as expected by the FleetPy actor for which the
        action is taken. Refer to the output typing of the compute_action method of the FleetPy actor expected output.

        :param observation: the raw observation received from FleetPy
        :param action: the action output by the RL algorithm for the current step
        :param actor_type: the type of the actor for which the action is taken (e.g. repositioning, pricing, etc.)
        :param event: the event for which the action is taken
        """
        pass

    @abstractmethod
    def reward(self, observation, action, actor_type, event: Events):
        """ Implement this method to calculate the reward based on the received observation and the action taken by the agent """
        pass

    def step(self, action):
        translated_action = self.translate_action(self.last_observation, action, self.last_actor_type, self.last_event)
        self._hook_manager.send_actor_response(self._process_id, translated_action)
        done = False
        while True:
            try:
                observation, actor_type, event = self._hook_manager.get_observations(process_id=self._process_id, timeout=1)
                reward = self.reward(observation, action, actor_type, event)
                translated_observation = self.translate_observation(observation, actor_type, event)
                self.last_observation = observation
                self.last_actor_type = actor_type
                self.last_event = event
                self.last_action = action
                break
            except Empty:
                if not self._fleetpy_thread.is_alive():
                    print(f"Ending FleetPy thread {threading.get_native_id()} of pid {os.getpid()} with worker index {self._process_id}.")
                    done = True
                    translated_observation = self.translate_observation(self.last_observation, self.last_actor_type, self.last_event)
                    reward = self.last_reward
                    break
        return translated_observation, reward, done, False, {}



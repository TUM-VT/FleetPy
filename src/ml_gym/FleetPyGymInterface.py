import sys
import os
import numpy as np
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add fleetpy path

from src.misc.init_modules import load_simulation_environment
from src.ml_gym.hooks_manager import HookManager, Events
from src.ml_gym.Observers import AbstractObserver
from src.ml_gym.Actors import AbstractActor
from threading import Thread
from queue import Queue, Empty
import gymnasium as gym
from abc import ABC, abstractmethod


def run_single_simulation(scenario_parameters, hooks_manager: HookManager, process_id: int, error_slot: list):
    try:
        SF = load_simulation_environment(scenario_parameters, hooks_manager, process_id)
        SF.run(process_id)
    except BaseException as e:
        # Threads swallow exceptions silently (they only get printed by the default
        # excepthook); stash it so the main thread can notice the crash and re-raise
        # instead of hanging forever on a queue that will never receive data.
        error_slot.append(e)
        traceback.print_exc()


class FleetPyGym(gym.Env, ABC):

    def __init__(self, scenario_parameters):
        in_out_queue = {0: (Queue(), Queue())}
        self._hook_manager = HookManager(in_out_queue)
        self.scenario_parameters = scenario_parameters
        self.last_observation = None
        self.last_reward = None
        self.last_action = None
        self._fleetpy_thread: Thread = None
        self._fleetpy_thread_error: list = []

    def register_observer(self, event: Events, observer: AbstractObserver):
        self._hook_manager.add_observer(event, observer)

    def register_actor(self, event: Events, actor: AbstractActor):
        self._hook_manager.add_actor(event, actor)

    def reset(self, *, seed=None, options=None):
        self._fleetpy_thread_error = []
        # daemon=True: only the main thread receives Ctrl+C/SIGINT in CPython, so on
        # interrupt this simulation thread would otherwise keep running (typically stuck
        # forever waiting on in_queue.get() for an action the dead main thread will never
        # send) and, being non-daemon, block interpreter shutdown so the process never exits.
        self._fleetpy_thread = Thread(target=run_single_simulation,
                                        args=(self.scenario_parameters, self._hook_manager, 0, self._fleetpy_thread_error),
                                        daemon=True)
        self._fleetpy_thread.start()
        observation, actor_type, done = self._poll_for_observation(process_id=0)
        if done:
            raise RuntimeError(
                "FleetPy simulation thread ended before producing an observation for reset()"
            )

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

    def _poll_for_observation(self, process_id, poll_timeout=1):
        """ Poll for the next observation, checking after every timeout whether the FleetPy
        simulation thread is still alive. Returns (observation, actor_type, thread_ended)
        instead of blocking forever on a queue that a dead thread will never fill again. If
        the thread died because of an unhandled exception, that exception is re-raised here
        so a crash actually surfaces instead of hanging or being silently treated as a
        normal episode end. """
        while True:
            try:
                observation, actor_type = self._hook_manager.get_observations(process_id=process_id, timeout=poll_timeout)
                return observation, actor_type, False
            except Empty:
                if not self._fleetpy_thread.is_alive():
                    if self._fleetpy_thread_error:
                        raise RuntimeError(
                            "FleetPy simulation thread crashed"
                        ) from self._fleetpy_thread_error[0]
                    return None, None, True

    def step(self, action):
        self._hook_manager.send_actor_response(0, action)
        observation, reward = self.last_observation, self.last_reward
        new_observation, actor_type, done = self._poll_for_observation(process_id=0)
        if done:
            print("FleetPy thread is dead")

            # Check values of terms in reward function 
            for name, values in [
                ("Unserved", self.cost_unserved_history),
                ("Travel", self.cost_travel_history),
                ("Deviation", self.cost_deviation_history),
                ]:
                if len(values) == 0:
                    continue

                print(f"\n{name}")
                print(f"Mean   : {np.mean(values):.3f}")
                print(f"Median : {np.median(values):.3f}")
                print(f"Max    : {np.max(values):.3f}")
                print(f"Min    : {np.min(values):.3f}")

            self.cost_unserved_history.clear()
            self.cost_travel_history.clear()
            self.cost_deviation_history.clear()
            
        else:
            observation = new_observation
            self.last_observation = observation
            reward = self.reward(observation, action, actor_type)
            self.last_reward = reward
        translated_observation = self.translate_observation(observation)
        return translated_observation, reward, done, False, {}



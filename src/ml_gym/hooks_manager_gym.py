import os
import sys
from enum import Enum
from collections import defaultdict
from multiprocessing.connection import PipeConnection
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))) # add fleetpy path

from src.ml_gym.Observers import AbstractObserver
from src.ml_gym.Actors import AbstractActor
from queue import Queue

LOG = logging.getLogger(__name__)
    
class Events(Enum):
    ML_OBSERVE = "ml_observe",
    ML_ACTION = "ml_action",
    OBSERVE_FLEET_STATE_AFTER_RECEIVING_STATUS_UPDATE = "observe_fleet_state_after_receiving_status_update", # this event is triggered after the fleetcontrol received a new status update of its vehicle and is about to trigger its optimization
    OBSERVE_BEFORE_REPOSITIONING = "observe_bevore_repositioning" # this event is triggered directly before the repositioning algorithm would calculate new repositioning trips

class HookManager:
    
    def __init__(self, comm_queue: Queue):
        self._hooks: dict[Events, list[Hook]] = {} # event_name -> list of hooks
        self._hooks_by_id: dict[int, Hook] = {}
        self._hook_id_count = 0
        self._comm_queue: Queue = comm_queue

    def __create_new_hook(self, event: Events):
        new_hook = Hook(event, self._hook_id_count)
        self._hooks[event] = [new_hook]
        self._hooks_by_id[self._hook_id_count] = new_hook
        self._hook_id_count += 1
        return new_hook

    def __remove_hook(self, event, hook):
        del self._hooks_by_id[hook.get_hook_id()]
        self._hooks[event].remove(hook)

    def add_observer(self, event: Events, observer: AbstractObserver):
        if event not in self._hooks:
            self.__create_new_hook(event)
        for hook in self._hooks[event]:
            hook.add_observer(observer)

    def add_actor(self, event: Events, actor: AbstractActor):
        if event not in self._hooks:
            self.__create_new_hook(event)
        for hook in self._hooks[event]:
            hook.add_actor(actor)

    def trigger(self, event: Events, fleetpy_module, **kwargs):
        if event in self._hooks:
            print(f"\ntrigger {event}")
            for h in self._hooks[event]:
                h.on_event(event, fleetpy_module, None, self._comm_queue)

    def get_observations(self, block=True, timeout=None):
        process_id, hook_id, actor_type, observations = self._comm_queue.get(block, timeout)
        return observations, hook_id, actor_type, process_id

    def send_actor_response(self, action):
        self._comm_queue.put(action)

class Hook:
    def __init__(self, event: Events, hook_id: int):
        self._observers: list[AbstractObserver] = []
        self._actors: list[AbstractActor] = []
        self._event = event
        self._hook_id = hook_id

    def get_observes_actors(self):
        return self._observers, self._actors

    def get_hook_id(self):
        return self._hook_id
    
    def on_event(self, event, fleetpy_module, process_id = None, conn: Queue = None):
        if event != self._event:
            return

        observation = {}
        for observer in self._observers:
            print(f"\nHook: trigger observer - {observer}")
            observation.update(observer.observe(fleetpy_module))

        for actor in self._actors:
            actor._act(observation, fleetpy_module, self._hook_id, process_id, conn)
    
    def add_observer(self, observer: AbstractObserver):
        if observer not in self._observers:
            self._observers.append(observer)
            
    def add_actor(self, actor: AbstractActor):
        if actor is not self._actors:
            self._actors.append(actor)

    def remove_actor(self, actor: AbstractActor):
        if actor in self._actors:
            self._actors.remove(actor)

        

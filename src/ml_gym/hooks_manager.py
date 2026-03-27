import os
import sys
from enum import Enum
from collections import defaultdict
from multiprocessing.connection import PipeConnection
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))) # add fleetpy path

from src.ml_gym.observers import AbstractObserver
from src.ml_gym.actors import AbstractActor

LOG = logging.getLogger(__name__)
    
class Events(Enum):
    ML_OBSERVE = "ml_observe",
    ML_ACTION = "ml_action",
    OBSERVE_FLEET_STATE_AFTER_RECEIVING_STATUS_UPDATE = "observe_fleet_state_after_receiving_status_update", # this event is triggered after the fleetcontrol received a new status update of its vehicle and is about to trigger its optimization
    OBSERVE_BEFORE_REPOSITIONING = "observe_bevore_repositioning" # this event is triggered directly before the repositioning algorithm would calculate new repositioning trips

class HookManager:
    
    def __init__(self, conn_dict_master=None, conn_dict_child=None):
        self._hooks: dict[Events, list[Hook]] = {} # event_name -> list of hooks
        self._hooks_by_id: dict[int, Hook] = {}
        self._hook_id_count = 0
        # Process id is only used in case of multiple process
        self._process_id = None
        self._conn_dict_master: dict[int, PipeConnection] = conn_dict_master
        self._conn_dict_child: dict[int, PipeConnection] = conn_dict_child

    def set_process_id(self, process_id):
        self._process_id = process_id

    def __create_new_hook(self, event: Events):
        new_hook = Hook(event, self._hook_id_count)
        self._hooks[event] = [new_hook]
        self._hooks_by_id[self._hook_id_count] = new_hook
        self._hook_id_count += 1
        return new_hook

    def __remove_hook(self, event, hook):
        del self._hooks_by_id[hook.get_hook_id()]
        self._hooks[event].remove(hook)

    def _get_all_hooks_details(self, event: Events = None):
        observers_dict = defaultdict(list)
        actors_dict = defaultdict(list)
        hook_list = self._hooks.values() if event is None else self._hooks[event]
        for hook in hook_list:
            observers_list, actors_list = hook.get_observes_actors()
            for observer in observers_list:
                observers_dict[observer].append(hook)
            for actor in actors_list:
                actors_dict[actor].append(hook)
        return dict(observers_dict), dict(actors_dict)

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

    def couple_actors_to_observers(self, event: Events, actors: list[AbstractActor], observers: list[AbstractObserver]):
        """ Limits the given actors to the given observers. The rest of the actors will continue to recieve
        observations from all observers, unless they are previously limited to specific observers. """

        if len(actors) == 0 or len(observers) == 0:
            LOG.info("Empty actors or observers list provided for coupling.")
            return False

        current_observers_dict, current_actors_dict = self._get_all_hooks_details(event)
        hooks_for_removal = set()
        marked_observers = set()
        for actor in actors:
            if actor in current_actors_dict:
                for associated_hook in current_actors_dict[actor]:
                    hook_observers, hook_actors = associated_hook.get_observes_actors()
                    # Remove the hook and mark the lost observers if the hook only contains the coupling actors
                    if len(set(actors).difference(hook_actors)) == 0:
                        hooks_for_removal.add(associated_hook)
                        marked_observers.update(associated_hook.get_observes_actors()[0])

        # TODO: some scenarios with "lone" observes should also be taken into account before removing the hook
        for hook in hooks_for_removal:
            self.__remove_hook(event, hook)

        new_hook = self.__create_new_hook(event)
        for observer in observers:
            new_hook.add_observer(observer)
        for actor in actors:
            new_hook.add_actor(actor)

    def trigger(self, event: Events, fleetpy_module, **kwargs):
        conn = self._conn_dict_child[self._process_id] if self._process_id is not None else None
        if event in self._hooks:
            print(f"\ntrigger {event}")
            for h in self._hooks[event]:
                h.on_event(event, fleetpy_module, self._process_id, conn)

    def listen_to_slave_processes(self):
        for process_id, master_conn in self._conn_dict_master.items():
            if master_conn.poll() is True:
                recv_process_id, hook_id, actor_type, observations = master_conn.recv()
                assert recv_process_id == process_id, "Message recieved from a different process id than expected."
                response = self._hooks_by_id[hook_id].get_actor_response(actor_type, observations, process_id)
                master_conn.send(response)


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

    def get_actor_response(self, actor_type: str, observations: dict, process_id: int):
        """ This method is specific to get the actor responses from master process in case of multiprocessing """
        for actor in self._actors:
            if type(actor) == actor_type:
                return actor.compute_action(observations, process_id)
        raise AssertionError(f"The actor {actor_type} for hook id {self._hook_id} was not found. Make sure you are "
                             f"using multiprocessing, otherwise this method should not have been called.")
    
    def on_event(self, event, fleetpy_module, process_id = None, conn: PipeConnection = None):
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

        

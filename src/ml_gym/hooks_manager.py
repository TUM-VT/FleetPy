import os
import sys
from enum import Enum
from collections import defaultdict
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
    
    def __init__(self, fleetpy_q_in, fleetpy_q_out):
        self._fleetpy_q_in = fleetpy_q_in
        self._fleetpy_q_out = fleetpy_q_out
        self._hooks: dict[Events, list[Hook]] = {} # event_name -> list of hooks

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
            self._hooks[event] = [Hook(event)]
        for hook in self._hooks[event]:
            hook.add_observer(observer)

    def add_actor(self, event: Events, actor: AbstractActor):
        if event not in self._hooks:
            self._hooks[event] = [Hook(event)]
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
            self._hooks[event].remove(hook)

        new_hook = Hook(event)
        for observer in observers:
            new_hook.add_observer(observer)
        for actor in actors:
            new_hook.add_actor(actor)
        self._hooks[event].append(new_hook)


    def trigger(self, event: Events, sim, **kwargs):
        # TODO: remove print statements after debugging
        #print(f"\ntrigger {event}")
        #print(f"\nhooks: {self._hooks}")
        if event in self._hooks:
            print(f"\ntrigger {event}")
            for h in self._hooks[event]:
                h.on_event(event, sim)
                
    def get_queues(self, **kwargs):
        # to be implemented in specific use cases
        return self._fleetpy_q_in, self._fleetpy_q_out
    
    def has_hooks(self, event):
        return event in self._hooks and len(self._hooks[event]) > 0
    
    def _get_hooks(self, event):
        if self.has_hooks(event):
            return [h for h in self._hooks[event] if isinstance(h, Hook)]
        else:
            return []


class Hook:
    def __init__(self, event: Events):
        self._observers: list[AbstractObserver] = []
        self._actors: list[AbstractActor] = []
        self._event = event

    def get_observes_actors(self):
        return self._observers, self._actors
    
    def on_event(self, event, fleetpy_module):
        if event != self._event:
            return

        observation = {}
        for observer in self._observers:
            print(f"\nHook: trigger observer - {observer}")
            observation.update(observer.observe(fleetpy_module))

        for actor in self._actors:
            actor._act(observation, fleetpy_module)
    
    def add_observer(self, observer: AbstractObserver):
        if observer not in self._observers:
            self._observers.append(observer)
            
    def add_actor(self, actor: AbstractActor):
        if actor is not self._actors:
            self._actors.append(actor)

    def remove_actor(self, actor: AbstractActor):
        if actor in self._actors:
            self._actors.remove(actor)

        

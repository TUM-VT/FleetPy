import os
import sys
from enum import Enum
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))) # add fleetpy path

from types import DynamicClassAttribute
import typing as tp
if tp.TYPE_CHECKING:
    from src.FleetSimulationBase import FleetSimulationBase
    
class Events(Enum):
    ML_OBSERVE = "ml_observe",
    ML_ACTION = "ml_action",
    OBSERVE_FLEET_STATE_AFTER_RECEIVING_STATUS_UPDATE = "observe_fleet_state_after_receiving_status_update", # this event is triggered after the fleetcontrol received a new status update of its vehicle and is about to trigger its optimization
    OBSERVE_BEFORE_REPOSITIONING = "observe_bevore_repositioning" # this event is triggered directly before the repositioning algorithm would calculate new repositioning trips


class HookManager:
    
    def __init__(self, fleetpy_q_in, fleetpy_q_out):
        self._fleetpy_q_in = fleetpy_q_in
        self._fleetpy_q_out = fleetpy_q_out
        self._hooks: tp.Dict[str, tp.List['Hook']] = {} # event_name -> list of hooks

    def register(self, event: Events, hook):
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(hook)

    def trigger(self, event, sim, **kwargs):
        # TODO: remove print statements after debugging
        print(f"\ntrigger {event}")
        print(f"\nhooks: {self._hooks}")
        if event in self._hooks:
            for h in self._hooks[event]:
                h.on_event(event, sim, **kwargs)
                
    def get_queues(self, **kwargs):
        # to be implemented in specific use cases
        return self._fleetpy_q_in, self._fleetpy_q_out
    
    def has_hooks(self, event):
        return event in self._hooks and len(self._hooks[event]) > 0
    
    def get_ml_hooks(self, event):
        if self.has_hooks(event):
            return [h for h in self._hooks[event] if isinstance(h, MLHook)]
        else:
            return []

class Hook:
    def on_event(self, event, fleetpy_module, **kwargs):
        pass            

class MLHook(Hook):
    def __init__(self, event: Events, ml_interface):
        self._registered_observers = []
        self._actor = None
        self._event = event
        self._ml_interface = ml_interface
    
    def on_event(self, event, fleetpy_module, **kwargs):
        if event != self._event:
            return
        observation = self._observe(fleetpy_module)
        action = self._ml_interface.communicate(event, observation)
        if self._actor is not None and action is not None:
            self._act(action, fleetpy_module)
        elif action is None and self._actor is None:
            return
        else:
            raise AttributeError("Either ML communicates and actor and no actor is registered, or the other way around")
    
    def register_observer(self, observer_method):
        if observer_method not in self._registered_observers:
            self._registered_observers.append(observer_method)
            
    def register_actor(self, actor_method):
        if self._actor is not None:
            raise ValueError("Only one actor can be registered per hook!")
        self._actor = actor_method
        
    def _observe(self, fleetpy_module):
        observation = {}
        for observer in self._registered_observers:
            print(f"\nMLHook: trigger observer - {observer}")
            print(f"observer(fleetpy_module): {observer(fleetpy_module)}")
            observation.update(observer(fleetpy_module))
        return observation
    
    def _act(self, action, fleetpy_module):
        self._actor(fleetpy_module, action)
        

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


class HookManager:
    
    def __init__(self, fleetpy_q_in, fleetpy_q_out):
        self._fleetpy_q_in = fleetpy_q_in
        self._fleetpy_q_out = fleetpy_q_out
        self._hooks: tp.Dict[str, tp.List['Hook']] = {} # event_name -> list of hooks

    def register(self, event: Events, hook):
        if event not in self._hooks:
            self._hooks[event.value] = []
        self._hooks[event.value].append(hook)

    def trigger(self, event, sim, **kwargs):
        if event in self._hooks:
            for h in self._hooks[event]:
                h.on_event(event, sim, **kwargs)
                
    def get_queues(self, **kwargs):
        # to be implemented in specific use cases
        return self._fleetpy_q_in, self._fleetpy_q_out
            
class Hook:
    def on_event(self, event, sim: 'FleetSimulationBase', **kwargs):
        pass

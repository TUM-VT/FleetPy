from abc import ABC, abstractmethod
from queue import Queue

class AbstractActor(ABC):

    @abstractmethod
    def compute_action(self, observation, process_id):
        pass

    def _compute_action_via_master_process(self, observation, hook_id, process_id=None, in_queue: Queue = None,
                                           out_queue: Queue = None,):
        if out_queue is not None:
            out_queue.put((process_id, hook_id, type(self), observation))
            action = in_queue.get()
        else:
            action = self.compute_action(observation, process_id)
        return action

    def _act(self, observation, fleetpy_module, hook_id, process_id: int = None, in_queue: Queue = None,
             out_queue: Queue = None):
        self.compute_action(observation, process_id)

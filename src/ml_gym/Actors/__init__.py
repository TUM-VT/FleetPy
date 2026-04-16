from abc import ABC, abstractmethod
from multiprocessing.connection import PipeConnection

class AbstractActor(ABC):

    @abstractmethod
    def compute_action(self, observation, process_id):
        pass

    def _compute_action_via_master_process(self, observation, hook_id, process_id=None, conn: PipeConnection=None):
        if process_id is None:
            return self.compute_action(observation, process_id)
        else:
            assert conn is not None, f"No connection object provided for {type(self)} to communicate with master"
            conn.send((process_id, hook_id, type(self), observation))
            response = conn.recv()
            return response

    def _act(self, observation, fleetpy_module, hook_id, process_id: int = None, conn: PipeConnection = None):
        self.compute_action(observation, process_id)

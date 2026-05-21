from abc import ABC, abstractmethod
from queue import Queue
from typeguard import check_type, TypeCheckError, CollectionCheckStrategy
import inspect
import logging

LOG = logging.getLogger(__name__)

class AbstractActor(ABC):

    @abstractmethod
    def compute_action(self, observation, process_id):
        """ This method computes the action required by specific FleetPy actor.

        Note for FleetPy Developers: This method is used to manually calculate the FleetPy action. If the FleetPy actor
        is expected to only get the RL action via FleetPyGym environment, then this method is not needed. Instead, use
        the method '_compute_action_via_master_process' to directly recieve the RL action via Queue from FleetPyGym.
        In this case, the FleetPy actor subclass can simply override the compute_action with a simple 'pass' and provide
        the expected return type hints of the compute_method. Describing the return type is important to make sure that
        the RL method is sending the action in the correct format. """
        pass

    def check_action_format(self, action):
        expected_signature = inspect.signature(self.compute_action).return_annotation
        if expected_signature == inspect.Signature.empty:
            LOG.warning(f"The method 'compute_action' of class {type(self).__name__} lacks a return type annotation. "
                          f"The format of the action recieved can not be checked.", UserWarning)
        else:
            try:
                check_type(action, expected_signature, collection_check_strategy=CollectionCheckStrategy.FIRST_ITEM)
            except TypeCheckError as e:
                raise TypeError(f"The recieved action for the FleetPy actor {type(self).__name__} is not in the expected "
                                f"format. Refer to the output typing of the method compute_action or the doc string of the "
                                f"{type(self).__name__} for details. Expected format : {expected_signature}, but got type "
                                f"{type(action)} action: {action}.")

    def _compute_action_via_master_process(self, observation, hook_id, process_id=None, in_queue: Queue = None,
                                           out_queue: Queue = None,):
        if out_queue is not None:
            out_queue.put((process_id, hook_id, type(self), observation))
            action = in_queue.get()
            self.check_action_format(action)
        else:
            action = self.compute_action(observation, process_id)
        return action

    def _act(self, observation, fleetpy_module, hook_id, process_id: int = None, in_queue: Queue = None,
             out_queue: Queue = None):
        self.compute_action(observation, process_id)

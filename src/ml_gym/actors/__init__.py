from abc import ABC, abstractmethod

class AbstractActor(ABC):

    @abstractmethod
    def compute_action(self, observation):
        pass

    def _act(self, observation, fleetpy_module, **kwargs):
        self.compute_action(observation)

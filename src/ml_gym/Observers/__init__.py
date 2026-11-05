from abc import abstractmethod, ABC


class AbstractObserver(ABC):

    @abstractmethod
    def observe(self, fleetpy_module):
        pass
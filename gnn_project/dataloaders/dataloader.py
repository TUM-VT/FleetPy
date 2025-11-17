from typing import Any, List, Dict
from torch import Tensor
from abc import ABC, abstractmethod


class DataLoader(ABC):
    """
    Abstract base class for data loaders.

    This class defines the interface for loading and processing data from simulation scenarios.
    Subclasses should implement the load_data method to provide specific data loading logic.
    """

    @abstractmethod
    def __init__(self, config):
        """
        Initialize the DataLoader with configuration.

        Args:
            config: Configuration object with data loading parameters.
        """
        self.config = config

    @abstractmethod
    def load_data(self) -> tuple[list[Any], Dict[str, List[Tensor]]]:
        """Load and process data from all scenarios.

        Returns:
            data: A list containing processed data objects.
            masks: A dictionary with keys 'train', 'val', 'test' mapping to lists of Tensors representing data masks.
        """
        pass

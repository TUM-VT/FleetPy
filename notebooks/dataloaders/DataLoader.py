# Standard library imports
from typing import Any

from torch import Tensor

# Local imports
from abc import ABC, abstractmethod


class DataLoader(ABC):
    """
    Abstract base class for data loaders.

    This class defines the interface for loading and processing data from simulation scenarios.
    Subclasses should implement the load_data method to provide specific data loading logic.
    """

    @abstractmethod
    def load_data(self) -> tuple[list[Any], list[Tensor], list[Tensor], list[Tensor]]:
        """Load and process data from all scenarios."""
        pass

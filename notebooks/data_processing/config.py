"""Configuration management for data processing."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

@dataclass
class DataProcessingConfig:
    """Configuration for data processing parameters.
    
    This class centralizes all configuration parameters used in data processing,
    making it easier to modify and track settings.
    """
    
    # Directory structure
    base_data_dir: str = 'data'
    processed_dir: str = 'processed'
    raw_dir: str = 'raw'
    train_dir: str = 'train'
    models_dir: str = 'models'

    REQUEST_FEATURES = 'request_features'
    VEHICLE_FEATURES = 'vehicle_features'
    REQUEST_REQUEST_GRAPH = 'request_request_graph'
    VEHICLE_REQUEST_GRAPH = 'vehicle_request_graph'
    LABEL = 'opt_assign'

    # Simulation parameters
    sim_duration: int = 86400  # seconds (24h)
    sim_step: int = 30  # seconds (30s)

    # Data splitting
    train_ratio: float = 1/3 # 3/7
    val_ratio: float = 1/3 # 2/7
    test_ratio: float = 1/3 # 2/7
    
    # Model parameters
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random_seed = 42
    test_mode : bool = True  # If True, use a smaller dataset for quick testing
    max_graphs_test: int = 24
    start_graph : int = 480

    SERVICE_DURATION = 30 # seconds

    # Feature processing
    categorical_features: Dict[str, List[str]] = None
    
    def __post_init__(self):
        """Initialize derived attributes after instance creation."""
        if self.categorical_features is None:
            self.categorical_features = {
                self.REQUEST_FEATURES: ['status'],
                self.VEHICLE_FEATURES: ['type', 'status']
            }
        
        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 < total_ratio < 1.01):  # Allow for small floating point errors
            raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio}")

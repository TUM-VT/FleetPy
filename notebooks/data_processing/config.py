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
    
    # Simulation parameters
    sim_duration: int = 86400  # seconds (24h)
    sim_step: int = 60  # seconds (1min)
    
    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Model parameters
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Feature processing
    categorical_features: Dict[str, List[str]] = None
    
    def __post_init__(self):
        """Initialize derived attributes after instance creation."""
        if self.categorical_features is None:
            self.categorical_features = {
                'req_features': ['status', 'o_pos', 'd_pos'],
                'veh_features': ['type', 'status', 'pos']
            }
        
        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 < total_ratio < 1.01):  # Allow for small floating point errors
            raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio}")

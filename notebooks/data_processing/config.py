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
    INIT_LABEL = 'init_assign'
    LABEL_THRESHOLD = 0.5

    INIT_ASSIGN_IDX = -1
    LOCKED_IDX = 10

    # Simulation parameters
    sim_start: int = 0  # seconds
    sim_end: int = 86400  # seconds (24h)
    sim_step: int = 30  # seconds (30s)

    # Data splitting
    train_ratio: float = 1 / 3  # 3/7
    val_ratio: float = 1 / 3  # 2/7
    test_ratio: float = 1 / 3  # 2/7

    # Model parameters
    device: torch.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    random_seed = 42

    SERVICE_DURATION = 30  # seconds

    LOG_LEVEL = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

    categorical_features = {
        REQUEST_FEATURES: ['status'],
        VEHICLE_FEATURES: ['type', 'status']
    }

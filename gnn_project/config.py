from dataclasses import dataclass, field
import os
from typing import Dict, List
import torch
from pathlib import Path


REQUEST_FEATURES_KEY = 'request_features'
VEHICLE_FEATURES_KEY = 'vehicle_features'
REQUEST_REQUEST_GRAPH_KEY = 'request_request_graph'
VEHICLE_REQUEST_GRAPH_KEY = 'vehicle_request_graph'
ASSIGNMENT_KEY = 'assignments'
INIT_ASSIGNMENT_KEY = 'init_assignments'
LABEL_KEY = 'optimal_assign'
INIT_LABEL_KEY = 'init_assign'

LOG_LEVEL_DEFAULT = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'


@dataclass
class Config:
    """Configuration class for GNN training and data processing parameters.
    """
    # ----- Directory structure -----
    project_dir: Path = Path(__file__).parent.parent
    ml_data_dir: Path = Path('data')
    experiment_name: str = 'gnn_v1'
    train_data_dir: str = 'train'
    processed_dir: Path = field(init=False)
    trained_models_dir: Path = field(init=False)
    norm_stats_dir: Path = field(init=False)
    saved_model_path: Path = field(init=False)

    # ----- Simulation parameters -----
    # Define case studies and their scenarios
    scenario_names: Dict[str, List[str]] = field(default_factory=lambda: {
        'manhattan_case_study': [
            'test_manhattan_scenario_1',
            'test_manhattan_scenario_2',
            'test_manhattan_scenario_3',
        ]
    })

    # List of full scenario paths in the format '../studies/{case_study}/results/{scenario_name}'
    scenario_paths: List[str] = field(
        init=False)  # Initialized in __post_init__
    shuffle_scenarios: bool = False  # Whether to shuffle scenarios before splitting

    sim_start: int = 0  # seconds
    sim_end: int = 86400  # seconds (24h)
    sim_step: int = 30  # seconds (30s)
    overwrite_data: bool = False
    log_level: str = LOG_LEVEL_DEFAULT

    # Data splitting
    dataloader_type: str = 'GNNDataLoader'  # Options: 'GNNDataLoader', others TBD
    train_ratio: float = 1 / 3  # 3/7
    val_ratio: float = 1 / 3  # 2/7
    test_ratio: float = 1 / 3  # 2/7

    # ----- Feature names -----
    request_features_key: str = REQUEST_FEATURES_KEY
    vehicle_features_key: str = VEHICLE_FEATURES_KEY
    request_request_graph_key: str = REQUEST_REQUEST_GRAPH_KEY
    vehicle_request_graph_key: str = VEHICLE_REQUEST_GRAPH_KEY
    assignment_key : str = ASSIGNMENT_KEY
    init_assignment_key : str = INIT_ASSIGNMENT_KEY
    label_key: str = LABEL_KEY
    init_label_key: str = INIT_LABEL_KEY
    # Features to exclude from edge attributes
    excluded_edge_features: List[str] = field(default_factory=lambda: ['source', 'target',
                                                                       LABEL_KEY, 'timestep'])
    # Features to exclude from node attributes
    excluded_node_features: List[str] = field(
        default_factory=lambda: ['id', 'timestep'])

    categorical_features: Dict[str, List[str]] = field(default_factory=lambda: {
        REQUEST_FEATURES_KEY: ['status'],
        VEHICLE_FEATURES_KEY: ['type', 'status']
    })

    # ----- Feature indices -----
    # TODO automate
    init_assign_idx: int = -1
    locked_idx: int = 10

    # ----- Model parameters -----
    model_type: str = 'HeteroGAT'  # Options: 'HeteroGAT', others TBD
    device: torch.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    random_seed: int = 42
    classification_threshold: float = 0.5  # Threshold for binary classification
    num_classes: int = 1  # Binary classification
    hidden_channels: int = 32  # GNN hidden layer size
    epochs: int = 500  # Maximum number of training epochs
    batch_size: int = 8  # Number of scenarios per batch
    learning_rate: float = 0.0001  # Learning rate for optimizer
    weight_decay: float = 1e-5  # Weight decay for optimizer
    patience: int = 20  # Early stopping patience
    dropout: float = 0.2  # Dropout rate
    num_layers: int = 2  # Number of GNN layers
    heads: int = 4  # Number of attention heads in GAT
    gamma: float = 2.0  # Focal loss gamma parameter
    alpha: float = 0.25  # Focal loss alpha parameter
    print_interval: int = 10  # Interval for printing training progress
    load_saved_model: bool = True

    def __post_init__(self):
        self.scenario_paths = [
            os.path.join(self.project_dir, 'studies', case_study, 'results', sc)
            for case_study, sc_names in self.scenario_names.items()
            for sc in sc_names
        ]

        if type(self.ml_data_dir) is str:
            self.ml_data_dir = Path(self.ml_data_dir)
        if not self.ml_data_dir.exists():
            self.ml_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir = self.ml_data_dir / 'processed'
        self.norm_stats_dir = self.ml_data_dir / 'norm_stats' / self.experiment_name
        self.trained_models_dir = self.ml_data_dir / 'models' / self.experiment_name
        self.saved_model_path = self.trained_models_dir / 'best_model.pt'

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.norm_stats_dir.mkdir(parents=True, exist_ok=True)
        self.trained_models_dir.mkdir(parents=True, exist_ok=True)

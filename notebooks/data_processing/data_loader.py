# Standard library imports
import os
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

# Local imports
from .config import DataProcessingConfig
from .graph_definitions import NODE_TYPES, EDGE_TYPES, CATEGORICAL_FEATURES
from .data_processor import DataProcessor


class DataLoader:
    """
    A class for loading and processing data from simulation scenarios.
    
    This class handles loading of raw and pre-processed data, transforming it into
    a format suitable for graph neural networks, and managing data splits.
    """
    
    def __init__(self, scenarios: List[str], config: Optional[DataProcessingConfig] = None, overwrite: bool = False):
        """Initialize the DataLoader.
        
        Args:
            scenarios: List of scenario paths to process
            config: Configuration for data processing, uses default if None
            overwrite: If True, forces reprocessing of data even if preprocessed data exists
        """
        self.config = config or DataProcessingConfig()
        self.overwrite = overwrite
        
        self.scenarios = scenarios
        self.rr_edge_feature_dim = None
        self.vr_edge_feature_dim = None

    def load_data(self) -> Tuple[List[HeteroData], List[torch.Tensor]]:
        """Load and process data from all scenarios.
        
        Attempts to load pre-processed data first, falls back to processing raw data
        if necessary. Also generates train/val/test masks for the loaded data.
        
        Returns:
            Tuple containing:
                - List of processed HeteroData objects
                - List of corresponding data masks
        """
        scenario_data = []
        scenario_masks = []
        
        for scenario_path in tqdm(self.scenarios):
            scenario_name = self._get_scenario_name(scenario_path)
            data = self._load_or_process_scenario(scenario_path, scenario_name)
            
            if data is not None:
                scenario_data.extend(data)
                scenario_masks.extend(self._create_data_masks(data))
        
        return scenario_data, scenario_masks
    
    def _get_scenario_name(self, scenario_path: str) -> str:
        """Extract scenario name from path."""
        scenario_path = os.path.normpath(scenario_path)
        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Scenario path {scenario_path} does not exist")
        return os.path.basename(scenario_path)
    
    def _load_or_process_scenario(self, scenario_path: str, scenario_name: str) -> Optional[List[HeteroData]]:
        """Load pre-processed data or process raw data for a scenario.
        
        If self.overwrite is True, skips loading preprocessed data and forces reprocessing.
        """
        # Skip loading preprocessed data if overwrite is True
        if not self.overwrite:
            data = self._try_load_preprocessed(scenario_name)
            if data is not None:
                return data
            
        # Process raw data (either because preprocessed doesn't exist or overwrite=True)
        return self._process_raw_data(scenario_path)
    
    def _try_load_preprocessed(self, scenario_name: str) -> Optional[List[HeteroData]]:
        """Try to load pre-processed data for a scenario."""
        graph_path = os.path.join(
            self.config.base_data_dir,
            scenario_name,
            self.config.processed_dir,
            'graph_data.pt'
        )
        
        if not os.path.exists(graph_path):
            return None
            
        try:
            return torch.load(graph_path)
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            return None
    
    def _process_raw_data(self, scenario_path: str) -> List[HeteroData]:
        """Process raw data into graph format."""
        # Initialize paths
        train_dir = os.path.join(scenario_path, self.config.train_dir)
        raw_dir = os.path.join(self.config.base_data_dir, self.config.raw_dir)
        
        # Process data through pipeline
        scenario_name = self._get_scenario_name(scenario_path)
        processor = DataProcessor(train_dir, raw_dir, self.config)
        data = processor.process_data(scenario_name)
        data = self._transform_features(data)
        return self._transform_and_save_data(data, scenario_name)
    
    def _transform_features(self, data: Dict) -> Dict:
        """Transform categorical features to one-hot encoded features."""
        for feature_type, categories in self.config.categorical_features.items():
            if feature_type in data and not data[feature_type].empty:
                data[feature_type] = pd.get_dummies(
                    data=data[feature_type],
                    columns=categories,
                    dtype=float
                )
        return data
    
    def _transform_and_save_data(self, data: Dict, scenario_name: str) -> List[HeteroData]:
        """Transform processed data into graph format and save it."""
        self._calculate_feature_dimensions(data)
        graphs = self._create_heterogeneous_graphs(data)
        graphs = self._apply_graph_transforms(graphs)
        self._save_processed_graphs(graphs, scenario_name)
        return graphs
    
    def _calculate_feature_dimensions(self, data: Dict) -> None:
        """Calculate edge feature dimensions."""
        if 'rr_graph' in data and not data['rr_graph'].empty:
            self.rr_edge_feature_dim = len(data['rr_graph'].columns) - 3
        if 'vr_graph' in data and not data['vr_graph'].empty:
            self.vr_edge_feature_dim = len(data['vr_graph'].columns) - 3
    
    def _create_heterogeneous_graphs(self, data: Dict) -> List[HeteroData]:
        """Create heterogeneous graphs from processed data."""
        max_timestep = data['req_features']['timestep'].max()
        graphs = [HeteroData() for _ in range(max_timestep + 1)]
        
        # Add features and edges
        self._add_node_features(data, graphs)
        self._add_edge_features(data, graphs)
        
        return graphs
    
    def _add_node_features(self, data: Dict, graphs: List[HeteroData]) -> None:
        """Add node features to graphs."""
        for name, node_type in [('req_features', 'request'), ('veh_features', 'vehicle')]:
            grouped = data[name].groupby('timestep')
            for timestep, graph in enumerate(graphs):
                try:
                    features = grouped.get_group(timestep)
                    if not features.empty:
                        graph[node_type].x = torch.tensor(features.values)
                except KeyError:
                    pass
    
    def _add_edge_features(self, data: Dict, graphs: List[HeteroData]) -> None:
        """Add edge features to graphs."""
        edge_configs = [
            ('rr_graph', ('request', 'connects', 'request'), self.rr_edge_feature_dim),
            ('vr_graph', ('vehicle', 'connects', 'request'), self.vr_edge_feature_dim)
        ]
        
        for name, edge_type, feat_dim in edge_configs:
            grouped = data[name].groupby('timestep')
            for timestep, graph in enumerate(graphs):
                try:
                    edges = grouped.get_group(timestep)
                    self._add_edge_data(edges, graph, edge_type, feat_dim)
                except KeyError:
                    self._add_empty_edge_data(graph, edge_type, feat_dim)
    
    def _add_edge_data(self, edges: pd.DataFrame, graph: HeteroData, 
                      edge_type: Tuple[str, str, str], feat_dim: int) -> None:
        """Add edge data to graph."""
        if edges.empty:
            self._add_empty_edge_data(graph, edge_type, feat_dim)
        else:
            graph[edge_type].edge_index = torch.tensor(
                np.array([edges['source'].values, edges['target'].values])
            ).int()
            graph[edge_type].edge_attr = torch.tensor(
                edges.drop(columns=['source', 'target', 'label']).values
            )
            graph[edge_type].y = torch.tensor(edges['label'].values).int()
    
    def _add_empty_edge_data(self, graph: HeteroData, 
                            edge_type: Tuple[str, str, str], 
                            feat_dim: int) -> None:
        """Add empty edge data to graph."""
        graph[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
        graph[edge_type].edge_attr = torch.zeros((0, feat_dim), dtype=torch.float)
        graph[edge_type].y = torch.zeros(0, dtype=torch.long)
    
    def _apply_graph_transforms(self, graphs: List[HeteroData]) -> List[HeteroData]:
        """Apply transforms to graphs."""
        transforms = T.Compose([
            T.NormalizeFeatures(attrs=['x', 'edge_attr']),
            T.ToUndirected()
        ])
        return [transforms(graph) for graph in graphs]
    
    def _save_processed_graphs(self, graphs: List[HeteroData], scenario_name: str) -> None:
        """Save processed graphs."""
        save_path = os.path.join(
            self.config.base_data_dir, 
            scenario_name,
            self.config.processed_dir, 
            'graph_data.pt'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(graphs, save_path)
    
    def _create_data_masks(self, data: List[HeteroData]) -> List[torch.Tensor]:
        """Create train/val/test masks for the data."""
        num_samples = len(data)
        
        # Calculate split sizes
        train_size = int(self.config.train_ratio * num_samples)
        val_size = int(self.config.val_ratio * num_samples)
        
        # Create and initialize masks
        device = self._get_data_device(data)
        masks = self._initialize_masks(num_samples, device)
        
        # Assign splits
        masks['train'][:train_size] = True
        masks['val'][train_size:train_size + val_size] = True
        masks['test'][train_size + val_size:] = True
        
        self._log_split_info(train_size, val_size, num_samples)
        return masks
    
    @staticmethod
    def _get_data_device(data: List[HeteroData]) -> torch.device:
        """Determine the device of the input data."""
        if len(data) > 0 and hasattr(data[0], 'device'):
            return data[0].device
        return torch.device('cpu')
    
    @staticmethod
    def _initialize_masks(size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize boolean masks for data splitting."""
        return {
            'train': torch.zeros(size, dtype=torch.bool, device=device),
            'val': torch.zeros(size, dtype=torch.bool, device=device),
            'test': torch.zeros(size, dtype=torch.bool, device=device)
        }
    
    @staticmethod
    def _log_split_info(train_size: int, val_size: int, total_size: int) -> None:
        """Log information about the data split."""
        test_size = total_size - train_size - val_size
        print(f"Data split: Train={train_size}, Val={val_size}, Test={test_size}")

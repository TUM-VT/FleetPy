# Standard library imports
import os
from typing import List, Tuple, Optional, Dict, Any

from torch import Tensor
from tqdm import tqdm

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

# Local imports
from .config import DataProcessingConfig
from .data_processor import DataProcessor


class DataLoader:
    """
    A class for loading and processing data from simulation scenarios.
    
    This class handles loading of raw and pre-processed data, transforming it into
    a format suitable for graph neural networks, and managing data splits.
    """
    EXCLUDED_EDGE_FEATURES = ['source', 'target', 'label']  # Features to exclude from edge attributes

    def __init__(self, scenarios: List[str], config: Optional[DataProcessingConfig] = None, overwrite: bool = False, balance_edges: bool = False, edge_balance_ratio: float = 1.0):
        """Initialize the DataLoader.
        
        Args:
            scenarios: List of scenario paths to process
            config: Configuration for data processing, uses default if None
            overwrite: If True, forces reprocessing of data even if preprocessed data exists
        """
        self.config = config or DataProcessingConfig()
        self.overwrite = overwrite
        self.balance_edges = balance_edges
        self.edge_balance_ratio = edge_balance_ratio  # Ratio of majority to minority class
        
        self.scenarios = scenarios
        self.rr_edge_feature_dim = None
        self.vr_edge_feature_dim = None

    def load_data(self) -> tuple[list[Any], list[Tensor], list[Tensor], list[Tensor]]:
        """Load and process data from all scenarios.
        
        Attempts to load pre-processed data first, falls back to processing raw data
        if necessary. Also generates train/val/test masks for the loaded data.
        
        Returns:
            Tuple containing:
                - List of processed HeteroData objects
                - List of corresponding data masks
        """
        scenario_data = []
        scenario_sizes = []  # Keep track of number of timesteps per scenario
        
        # First, load all scenario data
        for scenario_path in tqdm(self.scenarios, desc="Loading scenarios"):
            scenario_name = self._get_scenario_name(scenario_path)
            data = self._load_or_process_scenario(scenario_path, scenario_name)
            data = data[self.config.start_graph:self.config.start_graph+self.config.max_graphs_test] if self.config.test_mode else data
            
            if data is not None:
                scenario_data.extend(data)
                scenario_sizes.append(len(data))
        
        # Create masks based on scenario-level splits
        train_masks, val_masks, test_masks = self._create_scenario_based_masks(scenario_sizes)
        
        return scenario_data, train_masks, val_masks, test_masks
    
    def _get_scenario_name(self, scenario_path: str) -> str:
        """Extract scenario name from path."""
        scenario_path = os.path.normpath(scenario_path)
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
            return torch.load(graph_path, weights_only=False)
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
        processor = DataProcessor(train_dir, raw_dir, self.config, prefer_processed=True)
        data = processor.process_data(scenario_name)
        data = self._transform_features(data)
        return self._transform_and_save_data(data, scenario_name)

    def _transform_features(self, data: Dict) -> Dict:
        """Transform categorical features to one-hot encoded features."""
        for feature_type, categories in self.config.categorical_features.items():
            if feature_type in data and not data[feature_type].empty:
                if categories:
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
            self.rr_edge_feature_dim = len(data['rr_graph'].columns) - len(self.EXCLUDED_EDGE_FEATURES)
        if 'vr_graph' in data and not data['vr_graph'].empty:
            self.vr_edge_feature_dim = len(data['vr_graph'].columns) - len(self.EXCLUDED_EDGE_FEATURES)
    
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
                        # Select only numeric columns for node features
                        numeric_features = features.select_dtypes(include=[np.number])
                        graph[node_type].x = torch.tensor(numeric_features.values, dtype=torch.float32)
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
        """Add edge data to graph, with optional class balancing."""
        if edges.empty:
            self._add_empty_edge_data(graph, edge_type, feat_dim)
        else:
            if self.balance_edges and 'label' in edges.columns:
                class_counts = edges['label'].value_counts()
                if len(class_counts) > 1:
                    min_class = class_counts.idxmin()
                    max_class = class_counts.idxmax()
                    min_count = class_counts[min_class]
                    max_count = class_counts[max_class]
                    # Calculate number of majority samples to keep
                    keep_majority = int(self.edge_balance_ratio * min_count)
                    sampled_edges = []
                    for label in class_counts.index:
                        class_edges = edges[edges['label'] == label]
                        if label == min_class:
                            sampled_edges.append(class_edges)
                        else:
                            n_samples = min(keep_majority, len(class_edges))
                            sampled_edges.append(class_edges.sample(n=n_samples, random_state=self.config.random_seed))
                    edges = pd.concat(sampled_edges).sample(frac=1, random_state=self.config.random_seed)  # Shuffle
            graph[edge_type].edge_index = torch.tensor(
                np.array([edges['source'].values, edges['target'].values])
            ).int()
            graph[edge_type].edge_attr = torch.tensor(
                edges.drop(columns=self.EXCLUDED_EDGE_FEATURES).values, dtype=torch.float32
            )
            graph[edge_type].y = torch.tensor(edges['label'].values).int()

    def _add_empty_edge_data(self, graph: HeteroData, 
                            edge_type: Tuple[str, str, str], 
                            feat_dim: int) -> None:
        """Add empty edge data to graph."""
        graph[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
        graph[edge_type].edge_attr = torch.zeros((0, feat_dim), dtype=torch.float32)
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
    
    def _create_scenario_based_masks(self, scenario_sizes: List[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Create train/val/test masks at the scenario level.
        
        Args:
            scenario_sizes: List of number of timesteps in each scenario
            
        Returns:
            Tuple of (train_masks, val_masks, test_masks) where each mask is a list of boolean tensors
        """
        num_scenarios = len(scenario_sizes)
        total_timesteps = sum(scenario_sizes)
        
        # Calculate split sizes for scenarios
        train_scenarios = int(self.config.train_ratio * num_scenarios)
        val_scenarios = int(self.config.val_ratio * num_scenarios)
        
        # Create shuffled scenario indices TODO check if this is needed
        # scenario_indices = torch.randperm(num_scenarios)
        scenario_indices = list(range(num_scenarios))
        train_indices = scenario_indices[:train_scenarios]
        val_indices = scenario_indices[train_scenarios:train_scenarios + val_scenarios]
        test_indices = scenario_indices[train_scenarios + val_scenarios:]
        
        # Initialize masks for all timesteps
        device = torch.device('cpu')  # We'll keep masks on CPU initially
        train_masks = torch.zeros(total_timesteps, dtype=torch.bool, device=device)
        val_masks = torch.zeros(total_timesteps, dtype=torch.bool, device=device)
        test_masks = torch.zeros(total_timesteps, dtype=torch.bool, device=device)
        
        # Fill masks based on scenario assignments
        current_pos = 0
        for scenario_idx in range(num_scenarios):
            size = scenario_sizes[scenario_idx]
            if scenario_idx in train_indices:
                train_masks[current_pos:current_pos + size] = True
            elif scenario_idx in val_indices:
                val_masks[current_pos:current_pos + size] = True
            else:  # Test set
                test_masks[current_pos:current_pos + size] = True
            current_pos += size
        
        # Convert to list form as expected by the rest of the code
        train_masks = [train_masks[i] for i in range(total_timesteps)]
        val_masks = [val_masks[i] for i in range(total_timesteps)]
        test_masks = [test_masks[i] for i in range(total_timesteps)]
        
        # Log split information
        train_timesteps = sum(1 for m in train_masks if m)
        val_timesteps = sum(1 for m in val_masks if m)
        test_timesteps = sum(1 for m in test_masks if m)
        
        print(f"Scenario split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)} scenarios")
        print(f"Timestep split: Train={train_timesteps}, Val={val_timesteps}, Test={test_timesteps} timesteps")
        
        return train_masks, val_masks, test_masks
    
    @staticmethod
    def _get_data_device(data: List[HeteroData]) -> torch.device:
        """Determine the device of the input data."""
        if len(data) > 0 and hasattr(data[0], 'device'):
            return data[0].device
        return torch.device('cpu')
    
    @staticmethod
    def _log_split_info(train_size: int, val_size: int, total_size: int) -> None:
        """Log information about the data split."""
        test_size = total_size - train_size - val_size
        print(f"Data split: Train={train_size}, Val={val_size}, Test={test_size}")

    def get_edge_classification_data_for_rf(self, edge_type: str = 'vr_graph'):
        """
        Loads edge features and labels for all scenarios for use with scikit-learn classifiers (e.g., Random Forest).
        For each edge, concatenates edge features with source and target node features.
        Splits the data into train/val/test by scenario, using the same logic as _create_scenario_based_masks.
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test) as numpy arrays
        Args:
            edge_type: 'vr_graph' or 'rr_graph'
        """
        scenario_names = [self._get_scenario_name(s) for s in self.scenarios]
        
        scenario_sizes = []
        scenario_edge_dfs = []
        scenario_node_features = []  # List of dicts: {'veh': DataFrame, 'req': DataFrame}
        
        scenario_node_mappings = []
        scenario_reverse_node_mappings = []
        # Load edge and node CSVs for each scenario
        for scenario_path, scenario_name in zip(self.scenarios, scenario_names):
            edge_csv_path = os.path.join(
                self.config.base_data_dir,
                scenario_name,
                self.config.processed_dir,
                f'{edge_type}.csv'
            )
            req_csv_path = os.path.join(
                self.config.base_data_dir,
                scenario_name,
                self.config.processed_dir,
                'req_features.csv'
            )
            veh_csv_path = os.path.join(
                self.config.base_data_dir,
                scenario_name,
                self.config.processed_dir,
                'veh_features.csv'
            )
            if not os.path.exists(edge_csv_path) or not os.path.exists(req_csv_path):
                continue
            edge_df = pd.read_csv(edge_csv_path)
            # print(f"Loaded edge_df for scenario {scenario_name}, shape: {edge_df.shape}")
            req_df = pd.read_csv(req_csv_path)
            # print(f"Loaded req_df for scenario {scenario_name}, shape: {req_df.shape}")
            req_df = req_df.set_index('id')
            veh_df = None
            if os.path.exists(veh_csv_path):
                veh_df = pd.read_csv(veh_csv_path)
                # print(f"Loaded veh_df for scenario {scenario_name}, shape: {veh_df.shape}")
                veh_df = veh_df.set_index('id')
            # One-hot encode categorical columns
            def one_hot_df(df):
                if df is not None:
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns
                    if len(cat_cols) > 0:
                        df = pd.get_dummies(df, columns=cat_cols)
                return df
            edge_df = one_hot_df(edge_df)
            req_df = one_hot_df(req_df)
            if veh_df is not None:
                veh_df = one_hot_df(veh_df)
            # Filter by timestep if test_mode is enabled
            if hasattr(self.config, 'test_mode') and getattr(self.config, 'test_mode', False):
                start = getattr(self.config, 'start_graph', 0)
                max_graphs = getattr(self.config, 'max_graphs_test', 1)
                end = start + max_graphs
                edge_df = edge_df[(edge_df['timestep'] >= start) & (edge_df['timestep'] < end)]
                req_df = req_df[(req_df['timestep'] >= start) & (req_df['timestep'] < end)]
                if veh_df is not None:
                    veh_df = veh_df[(veh_df['timestep'] >= start) & (veh_df['timestep'] < end)]
                # print(f"After filtering by timestep: edge_df shape: {edge_df.shape}, req_df shape: {req_df.shape}, veh_df shape: {veh_df.shape if veh_df is not None else None}")
            # Build node_mapping and reverse_node_mapping for this scenario from req_df
            node_mapping = {}
            reverse_node_mapping = {}
            for timestep in req_df['timestep'].unique():
                timestep_df = req_df[req_df['timestep'] == timestep]
                mapping = {rid: idx for idx, rid in enumerate(timestep_df.index)}
                node_mapping[timestep] = mapping
                reverse_node_mapping[timestep] = {idx: rid for rid, idx in mapping.items()}
            scenario_node_mappings.append(node_mapping)
            scenario_reverse_node_mappings.append(reverse_node_mapping)
            scenario_edge_dfs.append(edge_df)
            scenario_sizes.append(len(edge_df))
            scenario_node_features.append({'req': req_df, 'veh': veh_df})
        
        if not scenario_edge_dfs:
            raise RuntimeError(f"No edge CSVs found for edge_type {edge_type}.")
        
        # Concatenate all edges, keep scenario index for splitting
        all_edges = pd.concat(scenario_edge_dfs, keys=range(len(scenario_edge_dfs)), names=['scenario_idx'])
        all_edges = all_edges.reset_index(level='scenario_idx')
        # print(f"After concatenation: all_edges shape: {all_edges.shape}")
        
        # Use the same scenario split logic as _create_scenario_based_masks
        num_scenarios = len(scenario_sizes)
        train_scenarios = int(self.config.train_ratio * num_scenarios)
        val_scenarios = int(self.config.val_ratio * num_scenarios)
        scenario_indices = list(range(num_scenarios))
        train_indices = scenario_indices[:train_scenarios]
        val_indices = scenario_indices[train_scenarios:train_scenarios + val_scenarios]
        test_indices = scenario_indices[train_scenarios + val_scenarios:]
        
        # Split by scenario
        train_df = all_edges[all_edges['scenario_idx'].isin(train_indices)]
        val_df = all_edges[all_edges['scenario_idx'].isin(val_indices)]
        test_df = all_edges[all_edges['scenario_idx'].isin(test_indices)]
        # print(f"Split sizes: train_df {train_df.shape}, val_df {val_df.shape}, test_df {test_df.shape}")
        
        # Prepare features and labels (exclude source, target, label columns from edge features)
        exclude_cols = ['source', 'target', 'label']
        edge_feature_cols = [c for c in train_df.columns if c not in exclude_cols + ['scenario_idx']]
        # print(f"Edge feature columns: {edge_feature_cols}")
        
        # Helper to get node features for a given scenario, node type, node index, and timestep
        def get_node_features(scenario_idx, node_type, node_id, timestep):
            node_df = scenario_node_features[scenario_idx][node_type]
            reverse_node_mapping = scenario_reverse_node_mappings[scenario_idx]
            # Convert node_id to int for lookup
            try:
                node_id_cast = int(node_id)
            except Exception:
                node_id_cast = node_id  # fallback, in case it's not convertible
            if node_df is not None:
                # For requests, map node_id (index from edge CSV) to original rid using reverse_node_mapping
                if node_type == 'req':
                    rid = reverse_node_mapping.get(timestep, {}).get(node_id_cast, None)
                    if rid is not None and rid in node_df.index:
                        filtered = node_df[(node_df.index == rid) & (node_df['timestep'] == timestep)]
                        if not filtered.empty:
                            arr = filtered.iloc[0].values
                            return arr
                        else:
                            return np.zeros(len(node_df.columns))
                    else:
                        return np.zeros(len(node_df.columns))
                else:
                    # For vehicles, use as before
                    filtered = node_df[(node_df.index == node_id_cast) & (node_df['timestep'] == timestep)]
                    if not filtered.empty:
                        arr = filtered.iloc[0].values
                        return arr
                    else:
                        return np.zeros(len(node_df.columns))
            else:
                return np.zeros(1)
        
        def build_features(df, name):
            features = []
            labels = []
            for idx, row in df.iterrows():
                scenario_idx = int(row['scenario_idx'])
                timestep = row['timestep']
                edge_feats = row[edge_feature_cols].values.astype(np.float32)
                if edge_type == 'vr_graph':
                    # source: vehicle, target: request
                    src_feats = get_node_features(scenario_idx, 'veh', row['source'], timestep)
                    tgt_feats = get_node_features(scenario_idx, 'req', row['target'], timestep)
                elif edge_type == 'rr_graph':
                    # source: request, target: request
                    src_feats = get_node_features(scenario_idx, 'req', row['source'], timestep)
                    tgt_feats = get_node_features(scenario_idx, 'req', row['target'], timestep)
                else:
                    raise ValueError(f"Unknown edge_type: {edge_type}")
                full_feats = np.concatenate([edge_feats, src_feats, tgt_feats]).astype(np.float32)
                features.append(full_feats)
                labels.append(int(row['label']))
            X = np.stack(features)
            y = np.array(labels, dtype=np.int64)
            # print(f"{name}: X shape {X.shape}, y shape {y.shape}")
            # print(f"{name}: X sample {X[:2] if len(X) > 1 else X}")
            # print(f"{name}: y sample {y[:10] if len(y) > 10 else y}")
            return X, y
        
        X_train, y_train = build_features(train_df, 'Train')
        X_val, y_val = build_features(val_df, 'Val')
        X_test, y_test = build_features(test_df, 'Test')
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

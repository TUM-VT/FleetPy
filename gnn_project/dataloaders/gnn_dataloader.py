# Standard library imports
import os
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
import logging

# Third-party imports
import pandas as pd
import numpy as np

from torch import Tensor
from tqdm import tqdm
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

# Local imports
from gnn_project.config import Config
from gnn_project.data_processing.data_processor import DataProcessor
from gnn_project.dataloaders.normalization import (
    load_normalization_statistics,
    normalize_features,
    clean_normalization_directory,
    get_feature_type
)

logger = logging.getLogger(__name__)


class GNNDataLoader:
    """
    A class for loading and processing data from simulation scenarios.

    This class handles loading of raw and pre-processed data, transforming it into
    a format suitable for graph neural networks, and managing data splits.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the DataLoader.

        Args:
            config: Configuration for data processing
        """
        self.config = config
        self.enable_overwrite_data = self.config.enable_overwrite_data
        self.scenario_paths = self.config.scenario_paths

        # Edge feature dimensions
        self.rr_edge_feature_dim = None
        self.vr_edge_feature_dim = None

        # Store one-hot columns for each feature type after first transform (training)
        self._onehot_columns = {}

        # Set up normalization directory
        if self.enable_overwrite_data:
            clean_normalization_directory(self.config.norm_stats_dir)

    def load_data(self) -> tuple[list[Any], Dict[str, List[Tensor]]]:
        """Load and process data from all scenarios.

        Attempts to load pre-processed data first, falls back to processing raw data
        if necessary. Also generates train/val/test masks for the loaded data.

        Returns:
            Tuple containing:
                - List of processed HeteroData objects
                - Dictionary with train/val/test masks
        """
       # First determine train/val/test split
        num_scenarios = len(self.scenario_paths)
        logger.debug(f"Total number of scenarios: {num_scenarios}")

        train_size = int(self.config.train_ratio * num_scenarios)
        val_size = int(self.config.val_ratio * num_scenarios)
        test_size = num_scenarios - train_size - val_size
        logger.debug(
            f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

        # Set up central statistics directory
        stats_exist = len(os.listdir(self.config.norm_stats_dir)) > 0
        logger.debug(f"Normalization statistics exist: {stats_exist}")

        scenario_data, scenario_sizes = self._prepare_scenario_data(
            train_size, stats_exist)

        # Create masks based on scenario-level splits
        masks = self._create_scenario_based_masks(scenario_sizes, shuffle=self.config.shuffle_scenarios)

        return scenario_data, masks

    def _prepare_scenario_data(self, train_size: int, stats_exist: bool) -> Tuple[List[HeteroData], List[int]]:
        """Prepare data from all scenarios, computing statistics if needed.

        Args:
            train_size: Number of training scenarios
            stats_exist: Whether normalization statistics already exist

        Returns:
            Tuple of (scenario_data, scenario_sizes)
        """
        scenario_data = []
        scenario_sizes = []  # Keep track of number of timesteps per scenario
        # Process all scenarios, computing statistics from training set if needed
        for idx, scenario_path in enumerate(tqdm(self.scenario_paths, desc="Processing scenarios")):
            # Compute statistics after processing training scenarios
            if not stats_exist and idx == train_size:
                training_data = [
                    (self._get_scenario_name(self.scenario_paths[i]), scenario_data[sum(
                        scenario_sizes[:i]):sum(scenario_sizes[:i+1])])
                    for i in range(train_size)
                ]
                self._compute_global_statistics(training_data)

            is_training = idx < train_size
            data = self._load_or_process_scenario(
                scenario_path, is_training=is_training)
            if data is not None:
                scenario_data.extend(data)
                scenario_sizes.append(len(data))

        return scenario_data, scenario_sizes

    def _get_scenario_name(self, scenario_path: str) -> str:
        """Extract scenario name from path."""
        scenario_path = os.path.normpath(scenario_path)
        return os.path.basename(scenario_path)

    def _get_scenario_path(self, scenario_name: str) -> str:
        """Get the full path for a scenario by its name.

        Args:
            scenario_name: Name of the scenario

        Returns:
            Full path to the scenario directory
        """
        # Search through scenarios to find the matching one
        for path in self.scenario_paths:
            if os.path.basename(os.path.normpath(path)) == scenario_name:
                return path
        raise ValueError(f"Could not find scenario path for {scenario_name}")

    def _load_or_process_scenario(self, scenario_path: str, is_training: bool = False) -> Optional[List[HeteroData]]:
        """Load pre-processed data or process raw data for a scenario.
        If self.overwrite is True, skips loading preprocessed data and forces reprocessing.

        Args:
            scenario_path: Path to the scenario directory
            is_training: Whether this is training data (to set one-hot columns)

        Returns:
            List of processed HeteroData objects, or None if processing fails
        """
        scenario_name = self._get_scenario_name(scenario_path)
        logger.debug(f"\n=== Processing scenario: {scenario_name} ===")
        logger.debug(f"Overwrite mode: {self.enable_overwrite_data}")

        # Skip loading preprocessed data if overwrite is True
        if not self.enable_overwrite_data:
            data = self._try_load_preprocessed_graph(scenario_name)
            if data is not None:
                return data

        # Process raw data (either because preprocessed doesn't exist or overwrite=True)
        return self._process_raw_data(scenario_path, scenario_name, is_training=is_training)

    def _try_load_preprocessed_graph(self, scenario_name: str) -> Optional[List[HeteroData]]:
        """Try to load pre-processed graph data for a scenario.

        Args:
            scenario_name: Name of the scenario

        Returns:
            Loaded data if available, else None
        """
        graph_path = self.config.processed_dir / scenario_name / 'graph_data.pt'

        if not graph_path.exists():
            return None

        try:
            return torch.load(graph_path, weights_only=False)
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {e}")
            return None

    def _process_raw_data(self, scenario_path: str, scenario_name: str, is_training: bool = False) -> List[HeteroData]:
        """Process raw data into graph format.

        Args:
            scenario_path: Path to the scenario directory
            scenario_name: Name of the scenario
            is_training: Whether this is training data (to set one-hot columns)

        Returns:
            List of processed HeteroData objects
        """
        train_data_dir = os.path.join(
            scenario_path, self.config.train_data_dir)
        prefer_processed = not self.enable_overwrite_data
        processor = DataProcessor(
            train_data_dir, self.config, prefer_processed=prefer_processed)
        try:
            data = processor.process_data(scenario_name)
            data = self._encode_categorical_features(
                data, is_training=is_training)
            normalized_data = self._normalize_data(data)
            graphs = self._create_and_save_graphs(
                normalized_data, scenario_name)
            return graphs
        except Exception as e:
            logger.error(f"\nError during data processing: {str(e)}")
            import traceback
            logger.error("Full traceback:")
            traceback.print_exc()
            raise

    def _normalize_data(self, data: Dict) -> Dict:
        """Normalize the data dictionary and return a new normalized dict.

        Args:
            data: Dictionary containing dataframes for different feature types

        Returns:
            A new dictionary with normalized dataframes 
        """
        normalization_stats = load_normalization_statistics(
            self.config.norm_stats_dir)
        if normalization_stats is None:
            return data
        means = normalization_stats['means']
        stds = normalization_stats['stds']
        data_mappings = [
            (self.config.request_features_key, 'req_'),
            (self.config.vehicle_features_key, 'veh_'),
            (self.config.request_request_graph_key, 'rr_'),
            (self.config.vehicle_request_graph_key, 'vr_')
        ]
        normalized = data.copy()
        for feature_key, prefix in data_mappings:
            if feature_key not in normalized or not isinstance(normalized[feature_key], pd.DataFrame):
                continue

            df = normalized[feature_key]
            feature_types = self._categorize_features(df)
            for ftype, cols in feature_types.items():
                if cols:
                    logger.debug(f"{ftype.capitalize()} features: {cols}")
            clean_prefix = prefix.replace('_', '')
            feature_means = {
                k.replace(prefix, ''): v for k, v in means.items() if k.startswith(prefix)}
            feature_stds = {
                k.replace(prefix, ''): v for k, v in stds.items() if k.startswith(prefix)}
            exclude_columns = feature_types['binary'] + \
                feature_types['categorical'] + feature_types['metadata']
            normalized[feature_key] = normalize_features(
                df,
                feature_means,
                feature_stds,
                exclude_columns=exclude_columns,
                prefix=clean_prefix
            )
        return normalized

    def _create_and_save_graphs(self, normalized_data: Dict, scenario_name: str) -> List[HeteroData]:
        """Create heterogeneous graphs and save them.

        Args:
            normalized_data: Dictionary containing normalized dataframes for different feature types
            scenario_name: Name of the scenario

        Returns:
            List of processed HeteroData objects
        """
        self._calculate_feature_dimensions(normalized_data)
        graphs = self._create_heterogeneous_graphs(normalized_data)
        self._save_processed_graphs(graphs, scenario_name)
        return graphs

    def _encode_categorical_features(self, data: Dict, is_training: bool = False) -> Dict:
        """Transform categorical features to one-hot encoded features, ensuring consistent columns.

        Args:
            data: Dictionary of dataframes for different feature types
            is_training: Whether this is training data (to set one-hot columns)

        Returns:
            Updated data dictionary with one-hot encoded categorical features   
        """
        for feature_type, categories in self.config.categorical_features.items():
            if feature_type not in data or data[feature_type].empty or not categories:
                continue

            categories = [
                cat for cat in categories if cat in data[feature_type].columns]
            temp = pd.get_dummies(
                data[feature_type], columns=categories, dtype=float)

            if is_training or feature_type not in self._onehot_columns:
                self._onehot_columns[feature_type] = temp.columns.tolist()
            else:
                # Add missing columns and reorder to match training
                for col in self._onehot_columns[feature_type]:
                    if col not in temp.columns:
                        temp[col] = 0.0
            temp = temp[self._onehot_columns[feature_type]]

            data[feature_type] = temp
        return data

    def _categorize_features(self, df: pd.DataFrame) -> Dict[str, list]:
        """Categorize features in a DataFrame by type.

        Args:
            df: DataFrame containing features to categorize

        Returns:
            Dictionary categorizing features by type
        """
        feature_types = {
            'continuous': [],
            'binary': [],
            'categorical': [],
            'metadata': []
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            feature_type = get_feature_type(df[col], col)
            feature_types[feature_type].append(col)
        return feature_types

    def _calculate_feature_dimensions(self, data: Dict) -> None:
        """Calculate edge feature dimensions.

        Args:
            data: Dictionary containing normalized dataframes for different feature types    
        """
        request_graph_key = self.config.request_request_graph_key
        if request_graph_key in data and not data[request_graph_key].empty:
            self.rr_edge_feature_dim = len(
                data[request_graph_key].columns) - len(self.config.excluded_edge_features)
        vr_graph_key = self.config.vehicle_request_graph_key
        if vr_graph_key in data and not data[vr_graph_key].empty:
            self.vr_edge_feature_dim = len(
                data[vr_graph_key].columns) - len(self.config.excluded_edge_features)

    def _create_heterogeneous_graphs(self, data: Dict) -> List[HeteroData]:
        """Create heterogeneous graphs directly as PyG HeteroData objects.

        Args:
            data: Dictionary containing normalized dataframes for different feature types

        Returns:
            List of processed HeteroData objects
        """
        # Only create graphs for timesteps present in the request features
        r_key = self.config.request_features_key
        if isinstance(data.get(r_key), pd.DataFrame):
            if 'timestep' in data[r_key].columns:
                max_timestep = data[r_key]['timestep'].max()
            else:
                max_timestep = 0
        else:
            max_timestep = 0
        logger.debug(f"Max timestep determined: {max_timestep}")
        graphs = []
        for timestep in range(max_timestep + 1):
            graph = HeteroData()
            self._add_node_features(graph, data, timestep)
            self._add_edge_features(graph, data, timestep)
            undirected_transform = T.ToUndirected(merge=True)
            graph = undirected_transform(graph)
            graph = T.NormalizeFeatures()(graph)
            graphs.append(graph)
        return graphs

    def _add_node_features(self, graph: HeteroData, data: Dict, timestep: int) -> None:
        """Add node features to the graph for a specific timestep.

        Args:
            graph: The HeteroData graph object to which node features will be added
            data: Dictionary containing normalized dataframes for different feature types
            timestep: The current timestep for which features are being added
        """
        for name, node_type in [(self.config.request_features_key, 'request'), (self.config.vehicle_features_key, 'vehicle')]:
            if name in data and isinstance(data[name], pd.DataFrame):
                features = data[name][data[name]['timestep'] == timestep]
                if not features.empty:
                    numeric_features = features.select_dtypes(include=[np.number])
                    numeric_features = numeric_features.drop(columns=self.config.excluded_node_features)
                    numeric_features = numeric_features.fillna(0.0)
                    if 'id' in features.columns:
                        node_ids = features['id'].values
                    else:
                        node_ids = numeric_features.index.values
                    graph[node_type].x = torch.tensor(numeric_features.values, dtype=torch.float32)
                    graph[node_type].node_ids = torch.tensor(node_ids, dtype=torch.long)
                else:
                    self._set_empty_node_features(graph, node_type)
            else:
                self._set_empty_node_features(graph, node_type)

    def _set_empty_node_features(self, graph: HeteroData, node_type: str) -> None:
        """Set empty node features for a given node type.
        Args:
            graph: The HeteroData graph object
            node_type: The node type string
        """
        graph[node_type].x = torch.zeros((0, 1), dtype=torch.float32)
        graph[node_type].node_ids = torch.zeros((0,), dtype=torch.long)

    def _add_edge_features(self, graph: HeteroData, data: Dict, timestep: int) -> None:
        """Add edge features to the graph for a specific timestep.
        Args:
            graph: The HeteroData graph object to which edge features will be added
            data: Dictionary containing normalized dataframes for different feature types
            timestep: The current timestep for which features are being added
        """
        edge_configs = [
            (self.config.request_request_graph_key, ('request', 'connects', 'request'), self.rr_edge_feature_dim),
            (self.config.vehicle_request_graph_key, ('vehicle', 'connects', 'request'), self.vr_edge_feature_dim)
        ]
        for name, edge_type, feat_dim in edge_configs:
            if name in data and isinstance(data[name], pd.DataFrame):
                edges = data[name][data[name]['timestep'] == timestep]
                if not edges.empty:
                    edge_index = torch.tensor(
                        [edges['source'].values, edges['target'].values], dtype=torch.long)
                    edge_features = edges.drop(columns=self.config.excluded_edge_features + ['timestep'])
                    edge_features = edge_features.fillna(0.0)
                    edge_attr = torch.tensor(edge_features.values, dtype=torch.float32)
                    y = torch.tensor(
                        edges[self.config.label_key].values, dtype=torch.long) if self.config.label_key in edges.columns else None
                    graph[edge_type].edge_index = edge_index
                    graph[edge_type].edge_attr = edge_attr
                    if y is not None:
                        graph[edge_type].y = y
                else:
                    self._set_empty_edge_features(graph, edge_type, feat_dim)
            else:
                self._set_empty_edge_features(graph, edge_type, feat_dim)

    def _set_empty_edge_features(self, graph: HeteroData, edge_type: tuple, feat_dim: int) -> None:
        """Set empty edge features for a given edge type.
        Args:
            graph: The HeteroData graph object
            edge_type: The edge type tuple
            feat_dim: Dimension of the edge features
        """
        graph[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
        graph[edge_type].edge_attr = torch.zeros((0, feat_dim), dtype=torch.float32)
        graph[edge_type].y = torch.zeros((0,), dtype=torch.long)

    def _save_processed_graphs(self, graphs: List[HeteroData], scenario_name: str) -> None:
        """Save processed graphs. Creates directories as needed.

        Args:
            graphs: List of processed HeteroData objects
            scenario_name: Name of the scenario
        """
        save_path = os.path.join(
            self.config.processed_dir,
            scenario_name,
            'graph_data.pt'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(graphs, save_path)

    def _create_scenario_based_masks(self, scenario_sizes: List[int], shuffle: bool) -> Dict[str, torch.Tensor]:
        """Create train/val/test masks at the scenario level.

        Args:
            scenario_sizes: List of number of timesteps in each scenario

        Returns:
            Dict[str, torch.Tensor]: Dictionary with keys "train_masks", "val_masks", and "test_masks", each containing a boolean tensor
        """
        num_scenarios = len(scenario_sizes)
        total_timesteps = sum(scenario_sizes)

        # Calculate split sizes for scenarios
        train_scenarios = int(self.config.train_ratio * num_scenarios)
        val_scenarios = int(self.config.val_ratio * num_scenarios)

        if shuffle:
            scenario_indices = torch.randperm(num_scenarios).tolist()
        else:
            scenario_indices = list(range(num_scenarios))
        train_indices = scenario_indices[:train_scenarios]
        val_indices = scenario_indices[train_scenarios:train_scenarios + val_scenarios]
        test_indices = scenario_indices[train_scenarios + val_scenarios:]

        # Initialize masks for all timesteps
        device = torch.device('cpu')  # We'll keep masks on CPU initially
        train_masks = torch.zeros(
            total_timesteps, dtype=torch.bool, device=device)
        val_masks = torch.zeros(
            total_timesteps, dtype=torch.bool, device=device)
        test_masks = torch.zeros(
            total_timesteps, dtype=torch.bool, device=device)

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

        # Log split information
        train_timesteps = train_masks.sum().item()
        val_timesteps = val_masks.sum().item()
        test_timesteps = test_masks.sum().item()

        logger.debug(
            f"Scenario split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)} scenarios")
        logger.debug(
            f"Timestep split: Train={train_timesteps}, Val={val_timesteps}, Test={test_timesteps} timesteps")

        return {"train_masks": train_masks, "val_masks": val_masks, "test_masks": test_masks}

    def _compute_global_statistics(self, training_data: List[Tuple[str, Dict]]) -> None:
        """Compute and save global statistics across all training scenarios.

        Args:
            training_data: List of (scenario_name, data) tuples from training scenarios
        """
        collections = defaultdict(list)
        for _, data_dict in training_data:
            for key in [self.config.request_features_key, self.config.vehicle_features_key,
                        self.config.request_request_graph_key, self.config.vehicle_request_graph_key]:
                if key in data_dict and isinstance(data_dict[key], pd.DataFrame):
                    collections[key].append(data_dict[key])

        stats = {"means": {}, "stds": {}, "mins": {}, "maxs": {}}
        for name, dfs in collections.items():
            if not dfs:
                continue
            df = pd.concat(dfs, ignore_index=True)
            numeric = df.select_dtypes(include=[np.number]).columns
            feature_types = {ftype: [] for ftype in ["continuous", "binary", "categorical", "metadata"]}
            for col in numeric:
                feature_types[get_feature_type(df[col], col)].append(col)
            cont = feature_types["continuous"]
            if not cont:
                continue
            prefix = name.replace(self.config.request_features_key, "req_") \
                .replace(self.config.vehicle_features_key, "veh_") \
                .replace(self.config.request_request_graph_key, "rr_") \
                .replace(self.config.vehicle_request_graph_key, "vr_") \
                .lower()
            cols = {col: f"{prefix}{col}" for col in cont}
            stats["means"].update(df[cont].mean().rename(cols).to_dict())
            stats["stds"].update(df[cont].std().replace(0, 1.0).rename(cols).to_dict())
            stats["mins"].update(df[cont].min().rename(cols).to_dict())
            stats["maxs"].update(df[cont].max().rename(cols).to_dict())

        for stat, values in stats.items():
            pd.DataFrame.from_dict(values, orient="index").to_parquet(
                os.path.join(self.config.norm_stats_dir, f"{stat}.parquet"))

    def load_single_timestep(self, scenario_path: str, timestep: int) -> Optional[HeteroData]:
        """Load and process a single timestep from a scenario for inference.

        Args:
            scenario_path: Path to the scenario directory
            timestep: The timestep to process

        Returns:
            Processed HeteroData object for the specified timestep, or None if processing fails
        """
        # TODO implement
        pass
# Standard library imports
import os
import json
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
import logging
import pickle

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
from gnn_project.defaults import *
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
            
        Note:
            - overwrite_data: Forces reprocessing of raw data and regeneration of graphs
            - recompute_norm_stats: Forces recomputation of normalization statistics from training data
              (normally, existing norm stats are preserved unless they don't exist)
        """
        self.config = config
        self.enable_overwrite_data = self.config.overwrite_data
        self.scenario_paths = self.config.scenario_paths

        # Edge feature dimensions
        self.rr_edge_feature_dim = None
        self.vr_edge_feature_dim = None

        # Store one-hot columns for each feature type after first transform (training)
        self._onehot_columns = {}
        self._load_onehot_columns()

        # Store feature names for each node/edge type
        self.feature_names = {}

    def log_scenario_sizes(self):
        """Log the number of scenarios and their split sizes."""
        num_scenarios = len(self.scenario_paths)
        logger.debug(f"Total number of scenarios: {num_scenarios}")

        train_size = int(self.config.train_ratio * num_scenarios)
        val_size = int(self.config.val_ratio * num_scenarios)
        test_size = num_scenarios - train_size - val_size
        logger.debug(
            f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    def load_data(self) -> tuple[list[Any], dict[str, Tensor] | None] | tuple[list[Any], dict[str, Tensor]]:
        """
        Load and process data from all scenarios in three steps:
        1. Load and process scenarios (no normalization)
        2. Compute normalization statistics from training data
        3. Normalize all scenarios using training stats
        """
        # Step 0: Try loading saved graphs
        loaded_graphs, masks = self._try_load_saved_graphs()
        if loaded_graphs is not None:
            return loaded_graphs, masks

        self.log_scenario_sizes()

        # Step 1: Load or process feature dicts
        train_size = int(self.config.train_ratio * len(self.scenario_paths))
        raw_scenario_data, scenario_sizes = self._load_or_process_feature_dicts(
            train_size)

        # Step 2: If norm stats missing or forced recomputation, compute them
        norm_stats_exist = (self.config.norm_stats_dir / MEANS_FILE).exists()
        if not norm_stats_exist or self.config.recompute_norm_stats:
            if self.config.recompute_norm_stats:
                clean_normalization_directory(self.config.norm_stats_dir)
                logger.info("Recomputing normalization statistics from training data...")
            training_data = raw_scenario_data[:train_size]
            self._compute_global_statistics(training_data)

        # Step 3: Normalize and create graphs
        normalized_scenario_data, masks = self._normalize_and_create_graphs(
            raw_scenario_data, scenario_sizes)
        return normalized_scenario_data, masks

    def _try_load_saved_graphs(self) -> tuple[Optional[List[Any]], Optional[Dict[str, torch.Tensor]]]:
        """Try to load previously saved processed graphs and create masks by block assignment."""
        if self.config.overwrite_data:
            return None, None
        processed_dir = self.config.processed_dir / self.config.experiment_name
        train_path = processed_dir / TRAIN_GRAPHS
        val_path = processed_dir / VAL_GRAPHS
        test_path = processed_dir / TEST_GRAPHS
        if train_path.exists() and val_path.exists() and test_path.exists():
            train_graphs = torch.load(train_path)
            val_graphs = torch.load(val_path)
            test_graphs = torch.load(test_path)
            normalized_scenario_data = train_graphs + val_graphs + test_graphs
            masks = self.create_masks(len(normalized_scenario_data), len(train_graphs), len(val_graphs))
            # Load feature names if available
            self.feature_names = self.load_feature_names()
            return normalized_scenario_data, masks
        return None, None

    def create_masks(self, total: int, train_len: int, val_len: int) -> Dict[str, torch.Tensor]:
        """Create train/val/test masks based on provided lengths."""
        train_masks = torch.zeros(total, dtype=torch.bool)
        val_masks = torch.zeros(total, dtype=torch.bool)
        test_masks = torch.zeros(total, dtype=torch.bool)
        train_masks[:train_len] = True
        val_masks[train_len:train_len+val_len] = True
        test_masks[train_len+val_len:] = True
        masks = {TRAIN_MASKS: train_masks, VAL_MASKS: val_masks, TEST_MASKS: test_masks}
        return masks

    def _load_or_process_feature_dicts(self, train_size: int) -> tuple[List[Tuple[str, Dict]], List[int]]:
        """Load or process feature dicts for all scenarios."""
        norm_stats_exist = (self.config.norm_stats_dir / MEANS_FILE).exists()
        raw_scenario_data = []
        scenario_sizes = []
        for idx, scenario_path in enumerate(tqdm(self.scenario_paths, desc="Loading/Processing feature dicts")):
            scenario_name = self._get_scenario_name(scenario_path)
            data = None
            if norm_stats_exist:
                data = self._try_load_feature_dict(scenario_name)
            if data is None:
                is_training = idx < train_size
                data = self._process_raw_data(
                    scenario_path, scenario_name, is_training=is_training)
                self._save_feature_dict(data, scenario_name)
            raw_scenario_data.append((scenario_name, data))
            # Count timesteps from request features
            req_key = self.config.request_features_key
            if req_key in data and isinstance(data[req_key], pd.DataFrame) and TIMESTEP in data[req_key].columns:
                num_timesteps = data[req_key][TIMESTEP].nunique()
            else:
                num_timesteps = 1
            scenario_sizes.append(num_timesteps)
        return raw_scenario_data, scenario_sizes

    def _normalize_and_create_graphs(self, raw_scenario_data, scenario_sizes) -> tuple[List[Any], Dict[str, Tensor]]:
        """Normalize data and create graphs for all scenarios."""
        normalized_scenario_data = []
        for _, data in tqdm(raw_scenario_data, desc="Normalizing scenarios"):
            normalized_data = self._normalize_data(data)
            graphs = self._create_hetero_graphs(normalized_data)
            normalized_scenario_data.extend(graphs)

        masks = self._create_scenario_based_masks(
            scenario_sizes, shuffle=self.config.shuffle_scenarios)
        # Group graphs by split and save
        train_graphs = []
        val_graphs = []
        test_graphs = []
        for i, graph in enumerate(normalized_scenario_data):
            if masks[TRAIN_MASKS][i]:
                train_graphs.append(graph)
            elif masks[VAL_MASKS][i]:
                val_graphs.append(graph)
            elif masks[TEST_MASKS][i]:
                test_graphs.append(graph)
        processed_dir = self.config.processed_dir / self.config.experiment_name
        os.makedirs(processed_dir, exist_ok=True)
        torch.save(train_graphs, processed_dir / TRAIN_GRAPHS)
        torch.save(val_graphs, processed_dir / VAL_GRAPHS)
        torch.save(test_graphs, processed_dir / TEST_GRAPHS)
        self._save_feature_names(processed_dir)
        return normalized_scenario_data, masks

    def _save_onehot_columns(self) -> None:
        """Save the one-hot columns mapping to a pickle file."""
        save_path = os.path.join(
            self.config.norm_stats_dir,
            ONEHOT_COLS
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self._onehot_columns, f)

    def _load_onehot_columns(self) -> None:
        """Load the one-hot columns mapping from a pickle file."""
        load_path = os.path.join(
            self.config.norm_stats_dir,
            ONEHOT_COLS
        )
        if not os.path.exists(load_path):
            return
        try:
            with open(load_path, 'rb') as f:
                self._onehot_columns = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading one-hot columns: {e}")

    def _save_feature_names(self, processed_dir) -> None:
        """Save feature names mapping to a JSON file.
        
        Args:
            processed_dir: Directory where processed data is saved
        """
        save_path = os.path.join(processed_dir, 'feature_names.json')
        with open(save_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        logger.debug(f"Saved feature names to {save_path}")

    def load_feature_names(self, experiment_name: Optional[str] = None) -> Dict[str, List[str]]:
        """Load feature names mapping from a JSON file.
        
        Args:
            experiment_name: Name of the experiment. If None, uses config.experiment_name
            
        Returns:
            Dictionary mapping node/edge types to their feature names
        """
        exp_name = experiment_name or self.config.experiment_name
        processed_dir = self.config.processed_dir / exp_name
        load_path = os.path.join(processed_dir, 'feature_names.json')
        
        if not os.path.exists(load_path):
            logger.warning(f"Feature names file not found at {load_path}")
            return {}
        
        try:
            with open(load_path, 'r') as f:
                feature_names = json.load(f)
            logger.debug(f"Loaded feature names from {load_path}")
            return feature_names
        except Exception as e:
            logger.error(f"Error loading feature names: {e}")
            return {}

    def _get_scenario_name(self, scenario_path: str) -> str:
        """Extract scenario name from path."""
        scenario_path = os.path.normpath(scenario_path)
        return os.path.basename(scenario_path)

    def _process_raw_data(self, scenario_path: str, scenario_name: str, is_training: bool = False) -> dict:
        """Process raw data into graph format.

        Args:
            scenario_path: Path to the scenario directory
            scenario_name: Name of the scenario
            is_training: Whether this is training data (to set one-hot columns)

        Returns:
            List of processed data dicts
        """
        train_data_dir = os.path.join(
            scenario_path, self.config.train_data_dir)
        prefer_processed = not self.enable_overwrite_data
        processor = DataProcessor(
            train_data_dir, self.config, prefer_processed=prefer_processed)
        data = processor.process_data(scenario_name)
        data = self._encode_categorical_features(data, is_training=is_training)
        if is_training:
            self._save_onehot_columns()
        return data
            

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
        means = normalization_stats[MEANS]
        stds = normalization_stats[STDS]
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

    def _create_hetero_graphs(self, normalized_data: Dict) -> List[HeteroData]:
        """Create heterogeneous graphs.

        Args:
            normalized_data: Dictionary containing normalized dataframes for different feature types

        Returns:
            List of processed HeteroData objects
        """
        self._calculate_feature_dimensions(normalized_data)
        graphs = self._create_heterogeneous_graphs(normalized_data)
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
        if request_graph_key in data and isinstance(data[request_graph_key], pd.DataFrame) and not data[request_graph_key].empty:
            self.rr_edge_feature_dim = len(
                data[request_graph_key].columns) - len(self.config.excluded_edge_features)
        elif self.rr_edge_feature_dim is None:
            # If we couldn't determine dimension, default to 0
            self.rr_edge_feature_dim = 0
            
        vr_graph_key = self.config.vehicle_request_graph_key
        if vr_graph_key in data and isinstance(data[vr_graph_key], pd.DataFrame) and not data[vr_graph_key].empty:
            self.vr_edge_feature_dim = len(
                data[vr_graph_key].columns) - len(self.config.excluded_edge_features)
        elif self.vr_edge_feature_dim is None:
            # If we couldn't determine dimension, default to 0
            self.vr_edge_feature_dim = 0

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
                timesteps = sorted(data[r_key][TIMESTEP].unique())
            else:
                timesteps = []
        else:
            timesteps = []
        graphs = []
        for timestep in timesteps:
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
        for name, node_type in [(self.config.request_features_key, REQUEST), (self.config.vehicle_features_key, VEHICLE)]:
            if name in data and isinstance(data[name], pd.DataFrame):
                features = data[name][data[name][TIMESTEP] == timestep]
                if not features.empty:
                    numeric_features = features.select_dtypes(
                        include=[np.number])
                    numeric_features = numeric_features.drop(
                        columns=self.config.excluded_node_features)
                    numeric_features = numeric_features.fillna(0.0)
                    
                    # Store feature names for this node type
                    if node_type not in self.feature_names:
                        self.feature_names[node_type] = list(numeric_features.columns)
                    
                    if ID in features.columns:
                        node_ids = features[ID].values
                    else:
                        node_ids = numeric_features.index.values
                    graph[node_type].x = torch.tensor(
                        numeric_features.values, dtype=torch.float32)
                    graph[node_type].node_ids = torch.tensor(
                        node_ids, dtype=torch.long)
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
            (self.config.request_request_graph_key, RR_EDGE_NAME, self.rr_edge_feature_dim),
            (self.config.vehicle_request_graph_key, VR_EDGE_NAME, self.vr_edge_feature_dim)
        ]
        for name, edge_type, feat_dim in edge_configs:
            if name in data and isinstance(data[name], pd.DataFrame):
                if TIMESTEP in data[name].columns:
                    edges = data[name][data[name][TIMESTEP] == timestep]
                else:
                    edges = data[name]
                if not edges.empty:
                    edge_index = torch.tensor(
                        np.array([edges[SOURCE].values, edges[TARGET].values]), dtype=torch.long)
                    edge_features = edges.drop(
                        columns=self.config.excluded_edge_features, errors='ignore')
                    edge_features = edge_features.fillna(0.0)
                    
                    # Store feature names for this edge type
                    edge_type_key = f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
                    if edge_type_key not in self.feature_names:
                        self.feature_names[edge_type_key] = list(edge_features.columns)
                    
                    edge_attr = torch.tensor(
                        edge_features.values, dtype=torch.float32)
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

    def _set_empty_edge_features(self, graph: HeteroData, edge_type: tuple[str, str, str], feat_dim: Optional[int]) -> None:
        """Set empty edge features for a given edge type.
        Args:
            graph: The HeteroData graph object
            edge_type: The edge type tuple
            feat_dim: Dimension of the edge features (None defaults to 0)
        """
        # Handle None by defaulting to 0 features
        if feat_dim is None:
            feat_dim = 0
            print("Warning: Edge feature dimension is None, defaulting to 0.")
        graph[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
        graph[edge_type].edge_attr = torch.zeros(
            (0, feat_dim), dtype=torch.float32)
        graph[edge_type].y = torch.zeros((0,), dtype=torch.long)

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
        
        # Create sets for faster lookup
        train_indices_set = set(scenario_indices[:train_scenarios])
        val_indices_set = set(scenario_indices[train_scenarios:train_scenarios + val_scenarios])
        test_indices_set = set(scenario_indices[train_scenarios + val_scenarios:])

        # Initialize masks for all timesteps
        device = torch.device('cpu')  # We'll keep masks on CPU initially
        train_masks = torch.zeros(
            total_timesteps, dtype=torch.bool, device=device)
        val_masks = torch.zeros(
            total_timesteps, dtype=torch.bool, device=device)
        test_masks = torch.zeros(
            total_timesteps, dtype=torch.bool, device=device)

        # Fill masks based on scenario assignments
        # IMPORTANT: Iterate in the ORIGINAL order since graphs are in original order
        current_pos = 0
        for scenario_idx in range(num_scenarios):
            size = scenario_sizes[scenario_idx]
            if scenario_idx in train_indices_set:
                train_masks[current_pos:current_pos + size] = True
            elif scenario_idx in val_indices_set:
                val_masks[current_pos:current_pos + size] = True
            else:  # scenario_idx in test_indices_set
                test_masks[current_pos:current_pos + size] = True
            current_pos += size

        # Log split information
        train_timesteps = train_masks.sum().item()
        val_timesteps = val_masks.sum().item()
        test_timesteps = test_masks.sum().item()

        logger.debug(
            f"Scenario split: Train={len(train_indices_set)}, Val={len(val_indices_set)}, Test={len(test_indices_set)} scenarios")
        logger.debug(
            f"Timestep split: Train={train_timesteps}, Val={val_timesteps}, Test={test_timesteps} timesteps")

        return {TRAIN_MASKS: train_masks, VAL_MASKS: val_masks, TEST_MASKS: test_masks}

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

        stats = {MEANS: {}, STDS: {}, MINS: {}, MAXS: {}}
        for name, dfs in collections.items():
            if not dfs:
                continue
            df = pd.concat(dfs, ignore_index=True)
            numeric = df.select_dtypes(include=[np.number]).columns
            feature_types = {ftype: [] for ftype in [
                "continuous", "binary", "categorical", "metadata"]}
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
            stats[MEANS].update(df[cont].mean().rename(cols).to_dict())
            stats[STDS].update(df[cont].std().replace(
                0, 1.0).rename(cols).to_dict())
            stats[MINS].update(df[cont].min().rename(cols).to_dict())
            stats[MAXS].update(df[cont].max().rename(cols).to_dict())

        for stat, values in stats.items():
            pd.DataFrame.from_dict(values, orient="index").to_parquet(
                os.path.join(self.config.norm_stats_dir, f"{stat}.parquet"))

    def _save_feature_dict(self, data: Dict, scenario_name: str) -> None:
        """Save processed feature dict as a pickle file."""
        save_path = os.path.join(
            self.config.processed_dir,
            scenario_name,
            FEATURE_DICT
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def _try_load_feature_dict(self, scenario_name: str) -> Optional[Dict]:
        """Try to load pre-processed feature dict for a scenario."""
        feature_path = os.path.join(
            self.config.processed_dir,
            scenario_name,
            FEATURE_DICT
        )
        if not os.path.exists(feature_path):
            return None
        try:
            with open(feature_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading feature dict: {e}")
            return None

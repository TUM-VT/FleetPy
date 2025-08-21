
# Standard library imports
import os
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any

# Third-party imports
import pandas as pd
import numpy as np

from torch import Tensor
from tqdm import tqdm
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

# Local imports
from notebooks.data_processing.config import DataProcessingConfig as cfg
from notebooks.data_processing.data_processor import DataProcessor
from notebooks.dataloaders.normalization import (
    load_normalization_statistics, 
    normalize_features,
    clean_normalization_directory,
    get_feature_type
)


class GNNDataLoader:
    """
    A class for loading and processing data from simulation scenarios.

    This class handles loading of raw and pre-processed data, transforming it into
    a format suitable for graph neural networks, and managing data splits.
    """
    EXCLUDED_EDGE_FEATURES = ['source', 'target',
                              cfg.LABEL]  # Features to exclude from edge attributes

    def __init__(self, scenarios: List[str], config: Optional[cfg] = None, overwrite: bool = False):
        """Initialize the DataLoader.

        Args:
            scenarios: List of scenario paths to process
            config: Configuration for data processing, uses default if None
            overwrite: If True, forces reprocessing of data even if preprocessed data exists
            balance_edges: If True, balances edge classes in the dataset
            edge_balance_ratio: Ratio of majority to minority class
        """
        self.config = config or cfg()
        self.overwrite = overwrite
        self.scenarios = scenarios
        
        # Set up normalization directory
        self.stats_dir = os.path.join(self.config.base_data_dir, 'normalization_stats')
        if self.overwrite:
            clean_normalization_directory(self.stats_dir)
        else:
            os.makedirs(self.stats_dir, exist_ok=True)
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
        print("\n=== Starting GNNDataLoader.load_data() ===")
        scenario_data = []
        scenario_sizes = []  # Keep track of number of timesteps per scenario

        # First determine train/val/test split
        num_scenarios = len(self.scenarios)
        print(f"Total number of scenarios: {num_scenarios}")
        train_size = int(self.config.train_ratio * num_scenarios)
        val_size = int(self.config.val_ratio * num_scenarios)
        print(f"Train size: {train_size}, Val size: {val_size}")

        # Set up central statistics directory
        self.stats_dir = os.path.join(
            self.config.base_data_dir, 'normalization_stats')
        os.makedirs(self.stats_dir, exist_ok=True)
        stats_exist = len(os.listdir(self.stats_dir)) > 0

        if not stats_exist:
            # First pass: Process training scenarios without normalization to collect statistics
            training_data = []
            print("\nFirst pass: Processing training scenarios without normalization...")
            for i, scenario_path in enumerate(tqdm(self.scenarios[:train_size], desc="Collecting training data")):
                scenario_name = self._get_scenario_name(scenario_path)
                data = self._load_or_process_scenario(
                    scenario_path, scenario_name
                )
                if data is not None:
                    training_data.append((scenario_name, data))

            # Compute global statistics from all training scenarios
            print("\nComputing global statistics across all training scenarios...")
            self._compute_global_statistics(training_data)
        else:
            print("\nUsing existing normalization statistics from", self.stats_dir)

        # Process all scenarios using global statistics
        print("\nProcessing all scenarios with global statistics...")

        # Process all scenarios (both training and val/test)
        for i, scenario_path in enumerate(tqdm(self.scenarios, desc="Processing scenarios")):
            scenario_name = self._get_scenario_name(scenario_path)

            data = self._load_or_process_scenario(
                scenario_path, scenario_name
            )

            data = data[self.config.start_graph:self.config.start_graph +
                        self.config.max_graphs_test] if self.config.test_mode else data

            if data is not None:
                scenario_data.extend(data)
                scenario_sizes.append(len(data))

        # Create masks based on scenario-level splits
        train_masks, val_masks, test_masks = self._create_scenario_based_masks(
            scenario_sizes)

        return scenario_data, train_masks, val_masks, test_masks

    def _get_scenario_name(self, scenario_path: str) -> str:
        """Extract scenario name from path."""
        scenario_path = os.path.normpath(scenario_path)
        return os.path.basename(scenario_path)

    def _get_scenario_path(self, scenario_name: str) -> str:
        """Get the full path for a scenario by its name."""
        # Search through scenarios to find the matching one
        for path in self.scenarios:
            if os.path.basename(os.path.normpath(path)) == scenario_name:
                return path
        raise ValueError(f"Could not find scenario path for {scenario_name}")

    def _compute_global_statistics(self, training_data: List[Tuple[str, Dict]]) -> None:
        """Compute global statistics across all training scenarios.

        Args:
            training_data: List of (scenario_name, data) tuples from training scenarios
        """
        print("\n=== Computing global statistics ===")

        # Collect feature data across all scenarios
        feature_collections = defaultdict(list)
        graph_collections = defaultdict(list)

        for scenario_name, scenario_data in training_data:
            scenario_dir = os.path.join(
                self.config.base_data_dir, scenario_name, self.config.processed_dir)

            # Collect node features
            for feature_type in [cfg.REQUEST_FEATURES, cfg.VEHICLE_FEATURES]:
                file_path = os.path.join(
                    scenario_dir, f'{feature_type}.parquet')
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    feature_collections[feature_type].append(df)

            # Collect edge features
            for graph_type in [cfg.REQUEST_REQUEST_GRAPH, cfg.VEHICLE_REQUEST_GRAPH]:
                file_path = os.path.join(scenario_dir, f'{graph_type}.parquet')
                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    graph_collections[graph_type].append(df)

        # Initialize dictionaries for all features
        all_means = {}
        all_stds = {}
        all_mins = {}
        all_maxs = {}

        # Process all feature collections and graph collections together
        for collection_name, dfs in {**feature_collections, **graph_collections}.items():
            if not dfs:
                continue

            print(f"\nProcessing {collection_name}")
            combined_df = pd.concat(dfs, ignore_index=True)

            # Get numerical columns and categorize them
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
            
            # Organize columns by feature type
            feature_types = {
                'continuous': [],
                'binary': [],
                'categorical': [],
                'metadata': []
            }
            
            # Categorize each column
            for col in numeric_cols:
                feature_type = get_feature_type(combined_df[col], col)
                feature_types[feature_type].append(col)
            
            # Print information about feature categories
            for feature_type, cols in feature_types.items():
                if cols and feature_type != 'continuous':
                    print(f"\n{feature_type.capitalize()} features (will be excluded from normalization):")
                    print(f"  {cols}")
            
            # Only normalize continuous features
            numeric_cols = feature_types['continuous']
            if numeric_cols:
                print(f"\nContinuous features to be normalized:")
                print(f"  {numeric_cols}")
            
            if not numeric_cols:
                continue
                
            print(f"Numeric columns for {collection_name}: {list(numeric_cols)}")

            # Add prefix to column names to avoid collisions
            prefix = collection_name.replace(cfg.REQUEST_FEATURES, 'req_')\
                                 .replace(cfg.VEHICLE_FEATURES, 'veh_')\
                                 .replace(cfg.REQUEST_REQUEST_GRAPH, 'rr_')\
                                 .replace(cfg.VEHICLE_REQUEST_GRAPH, 'vr_')\
                                 .lower()
            
            # Create prefixed column names
            prefixed_cols = {col: f"{prefix}{col}" for col in numeric_cols}
            
            # Compute statistics for each numeric column
            means = combined_df[numeric_cols].mean().rename(prefixed_cols)
            stds = combined_df[numeric_cols].std().fillna(1.0).rename(prefixed_cols)  # Replace 0 std with 1
            mins = combined_df[numeric_cols].min().rename(prefixed_cols)
            maxs = combined_df[numeric_cols].max().rename(prefixed_cols)

            # Update global statistics with prefixed names
            all_means.update(means.to_dict())
            all_stds.update(stds.to_dict())
            all_mins.update(mins.to_dict())
            all_maxs.update(maxs.to_dict())

        # Save all statistics
        pd.DataFrame.from_dict(all_means, orient='index').to_parquet(
            os.path.join(self.stats_dir, "means.parquet"))
        pd.DataFrame.from_dict(all_stds, orient='index').to_parquet(
            os.path.join(self.stats_dir, "stds.parquet"))
        pd.DataFrame.from_dict(all_mins, orient='index').to_parquet(
            os.path.join(self.stats_dir, "mins.parquet"))
        pd.DataFrame.from_dict(all_maxs, orient='index').to_parquet(
            os.path.join(self.stats_dir, "maxs.parquet"))

        print("\nSaved global normalization statistics to", self.stats_dir)

    def _load_or_process_scenario(self, scenario_path: str, scenario_name: str) -> Optional[List[HeteroData]]:
        """Load pre-processed data or process raw data for a scenario.

        Args:
            scenario_path: Path to the scenario directory
            scenario_name: Name of the scenario
            is_train: Whether this is a training scenario
            stats_dir: Directory containing normalization statistics (required for non-training scenarios)

        If self.overwrite is True, skips loading preprocessed data and forces reprocessing.
        """
        print(f"\n=== Processing scenario: {scenario_name} ===")
        print(f"Scenario path: {scenario_path}")
        print(f"Overwrite mode: {self.overwrite}")

        # Skip loading preprocessed data if overwrite is True
        if not self.overwrite:
            print("Attempting to load preprocessed data...")
            data = self._try_load_preprocessed(scenario_name)
            if data is not None:
                print("Successfully loaded preprocessed data")
                return data
            print("No preprocessed data found or failed to load")

        # Process raw data (either because preprocessed doesn't exist or overwrite=True)
        return self._process_raw_data(scenario_path, scenario_name)

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

    def _process_raw_data(self, scenario_path: str, scenario_name: str) -> List[HeteroData]:
        """Process raw data into graph format.

        Args:
            scenario_path: Path to the scenario directory
            scenario_name: Name of the scenario
        """
        print("\n=== Processing raw data ===")
        # Initialize paths
        train_dir = os.path.join(scenario_path, self.config.train_dir)
        raw_dir = os.path.join(self.config.base_data_dir, self.config.raw_dir)

        print(f"Train directory: {train_dir}")
        print(f"Raw directory: {raw_dir}")
        print("Checking directories exist:")
        print(f"- Train dir exists: {os.path.exists(train_dir)}")
        print(f"- Raw dir exists: {os.path.exists(raw_dir)}")

        # Process data through pipeline
        print("\nInitializing DataProcessor...")
        processor = DataProcessor(
            train_dir, raw_dir, self.config, prefer_processed=False)

        print("\nProcessing data through pipeline...")
        try:
            data = processor.process_data(scenario_name)
            print("Data processing completed successfully")
            print(
                f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")

            print("\nTransforming features...")
            data = self._transform_features(data)
            print("Feature transformation completed")

            return self._transform_and_save_data(data, scenario_name)
        except Exception as e:
            print(f"\nError during data processing: {str(e)}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            raise

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
        # Load normalization statistics if they exist
        try:
            means, stds, mins, maxs = load_normalization_statistics(self.stats_dir)
            
            # Columns to exclude from normalization
            exclude_cols = ['id', 'timestep', 'source', 'target', 'label']
            
            # Apply normalization with prefixed stats to each dataframe
            data_mappings = [
                (cfg.REQUEST_FEATURES, 'req_'),
                (cfg.VEHICLE_FEATURES, 'veh_'),
                (cfg.REQUEST_REQUEST_GRAPH, 'rr_'),
                (cfg.VEHICLE_REQUEST_GRAPH, 'vr_')
            ]
            
            for feature_key, prefix in data_mappings:
                if feature_key in data and isinstance(data[feature_key], pd.DataFrame):
                    df = data[feature_key]
                    
                    # Organize features by type
                    feature_types = {
                        'continuous': [],
                        'binary': [],
                        'categorical': [],
                        'metadata': []
                    }
                    
                    # Categorize numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        feature_type = get_feature_type(df[col], col)
                        feature_types[feature_type].append(col)
                    
                    # Print feature categorization
                    print(f"\nFeature categorization for {feature_key}:")
                    for ftype, cols in feature_types.items():
                        if cols:
                            print(f"{ftype.capitalize()} features: {cols}")
                    
                    # Create feature-specific statistics by filtering the global stats
                    clean_prefix = prefix.replace('_', '')  # Remove underscore for stats
                    feature_means = {k.replace(prefix, ''): v for k, v in means.items() if k.startswith(prefix)}
                    feature_stds = {k.replace(prefix, ''): v for k, v in stds.items() if k.startswith(prefix)}
                    
                    # Apply normalization only to continuous features
                    data[feature_key] = normalize_features(
                        df,
                        feature_means,
                        feature_stds,
                        exclude_columns=feature_types['binary'] + feature_types['categorical'] + feature_types['metadata'],
                        prefix=clean_prefix
                    )

                    print('Normalization sample:')
                    print(data[feature_key].head())
                    data[feature_key].to_csv(f"normalized_{feature_key}.csv", index=False)

        except FileNotFoundError:
            print("Warning: Normalization statistics not found. Proceeding without normalization.")
            
        self._calculate_feature_dimensions(data)
        graphs = self._create_heterogeneous_graphs(data)
        self._save_processed_graphs(graphs, scenario_name)
        return graphs

    def _calculate_feature_dimensions(self, data: Dict) -> None:
        """Calculate edge feature dimensions."""
        if cfg.REQUEST_REQUEST_GRAPH in data and not data[cfg.REQUEST_REQUEST_GRAPH].empty:
            self.rr_edge_feature_dim = len(
                data[cfg.REQUEST_REQUEST_GRAPH].columns) - len(self.EXCLUDED_EDGE_FEATURES)
        if cfg.VEHICLE_REQUEST_GRAPH in data and not data[cfg.VEHICLE_REQUEST_GRAPH].empty:
            self.vr_edge_feature_dim = len(
                data[cfg.VEHICLE_REQUEST_GRAPH].columns) - len(self.EXCLUDED_EDGE_FEATURES)

    def _create_heterogeneous_graphs(self, data: Dict) -> List[HeteroData]:
        """Create heterogeneous graphs directly as PyG HeteroData objects."""
        print("\n=== Creating heterogeneous graphs ===")
        print(f"Data keys available: {list(data.keys())}")
        print(f"Type of req_features: {type(data.get(cfg.REQUEST_FEATURES))}")

        # Handle request features as DataFrame (not dict)
        if isinstance(data.get(cfg.REQUEST_FEATURES), pd.DataFrame):
            print("Sample of req_features DataFrame:", data[cfg.REQUEST_FEATURES].head(2))
            # Get all unique timesteps from the DataFrame
            if 'timestep' in data[cfg.REQUEST_FEATURES].columns:
                max_timestep = data[cfg.REQUEST_FEATURES]['timestep'].max()
            else:
                max_timestep = 0
        else:
            max_timestep = 0  # Default value if request features are missing

        print(f"Max timestep determined: {max_timestep}")
        graphs = []

        # Process each timestep
        for timestep in range(max_timestep + 1):
            graph = HeteroData()

            # Add node features
            for name, node_type in [(cfg.REQUEST_FEATURES, 'request'), (cfg.VEHICLE_FEATURES, 'vehicle')]:
                try:
                    if name in data and isinstance(data[name], pd.DataFrame):
                        features = data[name][data[name]['timestep'] == timestep]
                        if not features.empty:
                            # Select only numeric columns for node features
                            numeric_features = features.select_dtypes(
                                include=[np.number])
                            numeric_features = numeric_features.drop(
                                columns=['timestep'])
                            
                            # Use node IDs from the DataFrame
                            if 'id' in features.columns:
                                node_ids = features['id'].values
                            else:
                                node_ids = numeric_features.index.values

                            graph[node_type].x = torch.tensor(
                                numeric_features.values, dtype=torch.float32)
                            graph[node_type].node_ids = torch.tensor(
                                node_ids, dtype=torch.long)
                        else:
                            graph[node_type].x = torch.zeros(
                                (0, 1), dtype=torch.float32)
                            graph[node_type].node_ids = torch.zeros(
                                (0,), dtype=torch.long)
                    else:
                        # Handle missing data or unexpected type
                        graph[node_type].x = torch.zeros(
                            (0, 1), dtype=torch.float32)
                        graph[node_type].node_ids = torch.zeros(
                            (0,), dtype=torch.long)
                except KeyError as e:
                    print(f"KeyError in node features for {node_type}: {e}")
                    # Handle case where no data exists for this timestep
                    graph[node_type].x = torch.zeros(
                        (0, 1), dtype=torch.float32)
                    graph[node_type].node_ids = torch.zeros(
                        (0,), dtype=torch.long)

            # Add edge features
            edge_configs = [
                (cfg.REQUEST_REQUEST_GRAPH, ('request',
                 'connects', 'request'), self.rr_edge_feature_dim),
                (cfg.VEHICLE_REQUEST_GRAPH, ('vehicle',
                 'connects', 'request'), self.vr_edge_feature_dim)
            ]

            for name, edge_type, feat_dim in edge_configs:
                try:
                    edges = data[name][data[name]['timestep'] == timestep]
                    if not edges.empty:
                        # Create edge index tensor
                        edge_index = torch.tensor(
                            [edges['source'].values, edges['target'].values],
                            dtype=torch.long
                        )

                        # Create edge attributes tensor
                        edge_features = edges.drop(
                            columns=self.EXCLUDED_EDGE_FEATURES + ['timestep'])
                        edge_attr = torch.tensor(
                            edge_features.values, dtype=torch.float32)

                        # Create label tensor if available
                        y = torch.tensor(
                            edges[cfg.LABEL].values, dtype=torch.long) if cfg.LABEL in edges.columns else None

                        # Add to graph
                        graph[edge_type].edge_index = edge_index
                        graph[edge_type].edge_attr = edge_attr
                        if y is not None:
                            graph[edge_type].y = y
                    else:
                        # Add empty edge data
                        graph[edge_type].edge_index = torch.zeros(
                            (2, 0), dtype=torch.long)
                        graph[edge_type].edge_attr = torch.zeros(
                            (0, feat_dim), dtype=torch.float32)
                        graph[edge_type].y = torch.zeros(
                            (0,), dtype=torch.long)
                except KeyError:
                    # Add empty edge data if no edges exist for this timestep
                    graph[edge_type].edge_index = torch.zeros(
                        (2, 0), dtype=torch.long)
                    graph[edge_type].edge_attr = torch.zeros(
                        (0, feat_dim), dtype=torch.float32)
                    graph[edge_type].y = torch.zeros((0,), dtype=torch.long)

            graphs.append(graph)

        return graphs

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
        print(f"Processed graphs saved to {save_path}")

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

        # Convert to list form as expected by the rest of the code
        train_masks = [train_masks[i] for i in range(total_timesteps)]
        val_masks = [val_masks[i] for i in range(total_timesteps)]
        test_masks = [test_masks[i] for i in range(total_timesteps)]

        # Log split information
        train_timesteps = sum(1 for m in train_masks if m)
        val_timesteps = sum(1 for m in val_masks if m)
        test_timesteps = sum(1 for m in test_masks if m)

        print(
            f"Scenario split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)} scenarios")
        print(
            f"Timestep split: Train={train_timesteps}, Val={val_timesteps}, Test={test_timesteps} timesteps")

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
        print(
            f"Data split: Train={train_size}, Val={val_size}, Test={test_size}")

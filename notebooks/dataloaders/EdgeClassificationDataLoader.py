# Standard library imports
import os
from typing import List, Optional
import datetime


# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from data_processing.config import DataProcessingConfig
from data_processing.data_processor import DataProcessor
from .DataLoader import DataLoader


class EdgeClassificationDataLoader(DataLoader):
    """
    DataLoader for edge classification tasks.
    This class extends the base DataLoader to specifically handle edge classification
    tasks by loading edge features and labels, and preparing them for use with classifiers.
    """

    def __init__(self, scenarios: List[str], edge_type='vr_graph', config: Optional[DataProcessingConfig] = None, overwrite: bool = False,  version: str = None, load_dir: str = None):
        """Initialize the EdgeClassificationDataLoader.

        Args:
            scenarios: List of scenario paths to process
            edge_type: Type of edge graph (e.g., 'vr_graph', 'rr_graph')
            version: Data version string to enforce versioned directory
            config: Configuration for data processing, uses default if None
            overwrite: If True, forces reprocessing of data even if preprocessed data exists
            load_dir: Optional directory to manually load data from (overrides version search)
        """
        self.scenarios = scenarios
        self.config = config if config else DataProcessingConfig()
        self.overwrite = overwrite
        self.edge_type = edge_type
        self.load_dir = load_dir
        self.version = version
        # Add timestamp to the save_dir to avoid overwriting
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_suffix = f"_v{version}" if version else ""
        self.save_dir = os.path.join(
            self.config.base_data_dir, self.config.train_dir, f"{folder_suffix}_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        print('Save directory for edge classification data:', self.save_dir)

    def _find_latest_version_dir(self, version: str):
        """Find the latest directory for the given version, if any."""
        base_dir = os.path.join(self.config.base_data_dir, self.config.train_dir)
        folder_suffix = f"_v{version}" if version else ""
        candidates = []
        if os.path.exists(base_dir):
            for d in os.listdir(base_dir):
                if d.startswith(folder_suffix + "_"):
                    candidates.append(d)
        if not candidates:
            return None
        # Sort by timestamp descending
        candidates.sort(reverse=True)
        return os.path.join(base_dir, candidates[0])

    def load_data(self):
        suffix = self.edge_type.split('_')[0]
        # Use manual load_dir if provided, else latest versioned dir for loading if not overwriting
        load_dir = self.save_dir
        if not self.overwrite:
            if self.load_dir is not None:
                load_dir = self.load_dir
            else:
                latest_dir = self._find_latest_version_dir(self.version)
                if latest_dir:
                    load_dir = latest_dir
            # Check if the data already exists
            if os.path.exists(os.path.join(load_dir, f"X_train_{suffix}.parquet")):
                try:
                    X_train = pd.read_parquet(
                        os.path.join(load_dir, f"X_train_{suffix}.parquet"))
                    y_train = pd.read_parquet(
                        os.path.join(load_dir, f"y_train_{suffix}.parquet"))
                    X_val = pd.read_parquet(
                        os.path.join(load_dir, f"X_val_{suffix}.parquet"))
                    y_val = pd.read_parquet(
                        os.path.join(load_dir, f"y_val_{suffix}.parquet"))
                    X_test = pd.read_parquet(
                        os.path.join(load_dir, f"X_test_{suffix}.parquet"))
                    y_test = pd.read_parquet(
                        os.path.join(load_dir, f"y_test_{suffix}.parquet"))
                    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
                except FileNotFoundError:
                    print(
                        f"Data files not found for edge type '{self.edge_type}'. Reprocessing data.")

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.get_split_data()

        # Save the data to Parquet files
        X_train.to_parquet(os.path.join(self.save_dir, f"X_train_{suffix}.parquet"), index=False)
        y_train.to_frame().to_parquet(os.path.join(self.save_dir, f"y_train_{suffix}.parquet"), index=False)
        X_val.to_parquet(os.path.join(self.save_dir, f"X_val_{suffix}.parquet"), index=False)
        y_val.to_frame().to_parquet(os.path.join(self.save_dir, f"y_val_{suffix}.parquet"), index=False)
        X_test.to_parquet(os.path.join(self.save_dir, f"X_test_{suffix}.parquet"), index=False)
        y_test.to_frame().to_parquet(os.path.join(self.save_dir, f"y_test_{suffix}.parquet"), index=False)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_split_data(self):
        """
        Loads edge features and labels for all scenarios for use with scikit-learn classifiers (e.g., Random Forest).
        For each edge, concatenates edge features with source and target node features.
        Splits the data into train/val/test by scenario, using the same logic as _create_scenario_based_masks.
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test) as tuples of DataFrames and Series.
        """
        print(f"Loading edge features and labels for {self.edge_type}...")
        scenario_names = [self._get_scenario_name(s) for s in self.scenarios]

        scenario_X = []
        scenario_y = []
        scenario_indices = []

        for scenario_idx, (scenario_path, scenario_name) in enumerate(zip(self.scenarios, scenario_names)):
            edge_parquet_path = os.path.join(
                self.config.base_data_dir,
                scenario_name,
                self.config.processed_dir,
                f'{self.edge_type}.parquet'
            )
            req_parquet_path = os.path.join(
                self.config.base_data_dir,
                scenario_name,
                self.config.processed_dir,
                'req_features.parquet'
            )
            veh_parquet_path = os.path.join(
                self.config.base_data_dir,
                scenario_name,
                self.config.processed_dir,
                'veh_features.parquet'
            )
            if not os.path.exists(edge_parquet_path) or not os.path.exists(req_parquet_path):
                try:
                    train_dir = os.path.join(
                        scenario_path, self.config.train_dir)
                    raw_dir = os.path.join(
                        self.config.base_data_dir, self.config.raw_dir)
                    processor = DataProcessor(
                        train_dir, raw_dir, self.config, prefer_processed=True)
                    processor.process_data(scenario_name)
                except Exception as e:
                    print(f"Error processing scenario {scenario_name}: {e}")
                    continue
            edge_df = pd.read_parquet(edge_parquet_path)
            req_df = pd.read_parquet(req_parquet_path)
            veh_df = pd.read_parquet(veh_parquet_path)

            # Filter by timestep if test_mode is enabled
            if hasattr(self.config, 'test_mode') and getattr(self.config, 'test_mode', False):
                start = getattr(self.config, 'start_graph', 0)
                max_graphs = getattr(self.config, 'max_graphs_test', 1)
                end = start + max_graphs
                edge_df = edge_df[(edge_df['timestep'] >= start)
                                  & (edge_df['timestep'] < end)]
                req_df = req_df[(req_df['timestep'] >= start)
                                & (req_df['timestep'] < end)]
                veh_df = veh_df[(veh_df['timestep'] >= start)
                                & (veh_df['timestep'] < end)]
                print(
                    f"After filtering by timestep: edge_df shape: {edge_df.shape}, req_df shape: {req_df.shape}, veh_df shape: {veh_df.shape}")

            # --- Restore mapping logic for request/vehicle ids ---
            # Build node_mapping and reverse_node_mapping for this scenario from req_df
            node_mapping = {}
            reverse_node_mapping = {}
            for timestep in req_df['timestep'].unique():
                timestep_df = req_df[req_df['timestep'] == timestep]
                mapping = {rid: idx for idx,
                           rid in enumerate(timestep_df.index)}
                node_mapping[timestep] = mapping
                reverse_node_mapping[timestep] = {
                    idx: rid for rid, idx in mapping.items()}

            # Add columns to edge_df for true request/vehicle ids using reverse mapping
            def map_source(row):
                timestep = row['timestep']
                idx = row['source']
                if self.edge_type == 'vr_graph':
                    return idx  # vehicle id is already correct
                return reverse_node_mapping.get(timestep, {}).get(idx, idx)

            def map_target(row):
                timestep = row['timestep']
                idx = row['target']
                return reverse_node_mapping.get(timestep, {}).get(idx, idx)
            edge_df['source_id'] = edge_df.apply(map_source, axis=1)
            edge_df['target_id'] = edge_df.apply(map_target, axis=1)

            # Prepare features and labels (exclude source, target, label columns from edge features)
            exclude_cols = ['source', 'target', 'label']
            edge_feature_cols = [
                c for c in edge_df.columns if c not in exclude_cols]
            edge_features = edge_df[edge_feature_cols].add_prefix('edge_')

            # Merge source node features
            source_type = 'veh' if self.edge_type == 'vr_graph' else 'req'
            src_df = veh_df if source_type == 'veh' else req_df
            tgt_df = req_df

            src_feat_cols = [
                col for col in src_df.columns if col not in ['timestep', 'id']]
            tgt_feat_cols = [
                col for col in tgt_df.columns if col not in ['timestep', 'id']]
            src_df_renamed = src_df[src_feat_cols + ['timestep', 'id']
                                    ].rename(columns={col: f'src_{col}' for col in src_feat_cols})
            tgt_df_renamed = tgt_df[tgt_feat_cols + ['timestep', 'id']
                                    ].rename(columns={col: f'tgt_{col}' for col in tgt_feat_cols})

            # Merge source node features
            merged = edge_features.merge(
                src_df_renamed,
                left_on=['edge_source_id', 'edge_timestep'],
                right_on=['id', 'timestep'],
                how='left',
            ).drop(columns=['id', 'edge_source_id'])

            # Merge target node features
            merged = merged.merge(
                tgt_df_renamed,
                left_on=['edge_target_id', 'edge_timestep'],
                right_on=['id', 'timestep'],
                how='left',
            ).drop(columns=['id', 'edge_target_id'])

            merged = merged.loc[:, ~merged.columns.duplicated()]
            merged = merged.fillna(0)

            X = merged[[col for col in merged.columns if col.startswith(
                'edge_') or col.startswith('src_') or col.startswith('tgt_')]]
            y = edge_df['label'].astype(np.int64)

            scenario_X.append(X)
            scenario_y.append(y)
            scenario_indices.append(scenario_idx)

        # Split scenarios into train/val/test
        num_scenarios = len(scenario_X)
        train_scenarios = int(self.config.train_ratio * num_scenarios)
        val_scenarios = int(self.config.val_ratio * num_scenarios)
        indices = list(range(num_scenarios))
        train_indices = indices[:train_scenarios]
        val_indices = indices[train_scenarios:train_scenarios + val_scenarios]
        test_indices = indices[train_scenarios + val_scenarios:]

        X_train = pd.concat([scenario_X[i]
                            for i in train_indices], ignore_index=True)
        y_train = pd.concat([scenario_y[i]
                            for i in train_indices], ignore_index=True)
        X_val = pd.concat([scenario_X[i]
                          for i in val_indices], ignore_index=True)
        y_val = pd.concat([scenario_y[i]
                          for i in val_indices], ignore_index=True)
        X_test = pd.concat([scenario_X[i]
                           for i in test_indices], ignore_index=True)
        y_test = pd.concat([scenario_y[i]
                           for i in test_indices], ignore_index=True)

        print(
            f"Split sizes: train_df {X_train.shape}, val_df {X_val.shape}, test_df {X_test.shape}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _get_scenario_name(self, scenario_path: str) -> str:
        """Extract scenario name from path."""
        scenario_path = os.path.normpath(scenario_path)
        return os.path.basename(scenario_path)

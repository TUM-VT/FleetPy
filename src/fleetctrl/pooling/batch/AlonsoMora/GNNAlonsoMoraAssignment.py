import logging
import pickle
import os
import time
from typing import Dict, List, Callable
from collections import defaultdict
import pandas as pd
import torch
import numpy as np
import torch_geometric.transforms as T

from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraAssignmentOriginal import AlonsoMoraAssignmentOriginal
from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraAssignment import AlonsoMoraAssignment
from src.misc.globals import *
from src.routing.NetworkBase import NetworkBase
from src.simulation.Legs import VehicleRouteLeg
from src.simulation.Vehicles import SimulationVehicle

from torch_geometric.data import HeteroData
from gnn_project.config import Config
from gnn_project.defaults import VR_EDGE_NAME, RR_EDGE_NAME, RV_EDGE_NAME, SOURCE, TARGET, REQUEST, VEHICLE, TRAIN_GRAPHS
from gnn_project.data_processing.data_processor import DataProcessor
from gnn_project.dataloaders.gnn_dataloader import GNNDataLoader
from gnn_project.training.train_utils import get_edge_predictions, load_saved_model
from src.fleetctrl.planning.PlanRequest import PlanRequest


LOG = logging.getLogger(__name__)


EDGE_TYPE = 'edge_type'
PRED_SCORE = 'pred_score'
GNN = 'gnn'
TOP_K = 'top_k'
PREDICTION_THRESHOLD = 'threshold'
    

class GNNAlonsoMoraAssignment(AlonsoMoraAssignmentOriginal):
    """Extension of Alonso-Mora Assignment Class that can optionally use ML for assignment predictions.

    This class extends the original Alonso-Mora implementation to optionally use machine learning
    models (XGBoost or GNN) for predicting feasible vehicle-request connections. By default,
    it uses the original implementation unless ML is explicitly enabled.

    Training data is stored in compressed pickle format with bz2 compression for optimal storage efficiency.

    Configuration via operator_attributes:
        enable_ml_training (bool): Whether to enable ML training. Default: False
        enable_ml_inference (bool): Whether to enable ML inference. Default: False
        model_type (str): Which model to use ('xgboost' or 'gnn'). Default: 'gnn'
        ml_selection_method (str): 'threshold' or 'top_k_vehicles'. Default: 'top_k_vehicles'
        prediction_threshold (float): Probability threshold for predictions. Default: 0.5
        top_k_vehicles (int): Number of top predictions to consider if using 'top_k_vehicles' method
    """
    EXPERIMENT_NAME = 'gnn_v1'
    ENABLE_ML_DEFAULT = False
    MODEL_TYPE_DEFAULT = GNN  # only 'gnn' for now
    ML_SELECTION_METHOD_DEFAULT = TOP_K  # 'threshold' or 'top_k_vehicles'
    TOP_K_VR_DEFAULT = 5  # Used if selection method is 'top_k_vehicles'
    TOP_K_RR_DEFAULT = 5  # Used if selection method is 'top_k_rr'
    PREDICTION_THRESHOLD_DEFAULT = 0.5  # Used if selection method is 'threshold'


    def __init__(self, fleetcontrol: FleetControlBase, routing_engine: NetworkBase, sim_time: int,
                 obj_function: Callable, operator_attributes: dict, optimisation_cores: int = 1, seed: int = 6061992,
                 veh_objs_to_build: Dict[int, SimulationVehicle] = {}):
        """Initializes the GNNAlonsoMoraAssignment with optional ML settings."""
        super().__init__(fleetcontrol, routing_engine, sim_time, obj_function, operator_attributes, optimisation_cores,
                         seed, veh_objs_to_build)
        
        # Configure ML settings from operator attributes
        self.enable_ml_training = operator_attributes.get(
            G_OP_ENABLE_ML_TRAINING, self.ENABLE_ML_DEFAULT)  # ML is disabled by default
        self.enable_ml_inference = operator_attributes.get(
            G_OP_ENABLE_ML_INFERENCE, self.ENABLE_ML_DEFAULT)
        if self.enable_ml_inference:
            self.model_type = operator_attributes.get(
                G_OP_MODEL_TYPE, self.MODEL_TYPE_DEFAULT)  # 'xgboost' or 'gnn'
            self.ml_selection_method = operator_attributes.get(
                G_OP_ML_SELECTION_METHOD, self.ML_SELECTION_METHOD_DEFAULT)  # 'threshold' or 'top_k_vehicles'
            # int(): scenario CSV columns with blank cells in other rows get parsed as
            # float64 by pandas (e.g. 999 -> 999.0), which nlargest() rejects as n
            self.top_k_vr = int(operator_attributes.get(G_OP_TOP_K_VR, self.TOP_K_VR_DEFAULT))
            self.top_k_rr = int(operator_attributes.get(G_OP_TOP_K_RR, self.TOP_K_RR_DEFAULT))
            self.prediction_threshold = operator_attributes.get(
                G_OP_PREDICTION_THRESHOLD, self.PREDICTION_THRESHOLD_DEFAULT)
        
        # Allow overriding the training-data directory via operator attributes.
        # If the operator provides `G_OP_TRAIN_DATA_DIR`, prefer it; otherwise use the existing default.
        self.train_data_path = operator_attributes.get(
            G_OP_TRAIN_DATA_DIR,
            os.path.join(self.fleetcontrol.dir_names[G_DIR_OUTPUT], G_DIR_TRAIN),
        )
        
        # Initialize Config
        self.config = Config(
            ml_data_dir=os.path.join(self.fleetcontrol.dir_names[G_DIR_MAIN], 'gnn_project/data'),
            # blank scenario-CSV cells show up as a present key with value None/NaN, which
            # defeats dict.get()'s default - "or" catches that as well as a missing key
            experiment_name=operator_attributes.get(G_OP_ML_EXPERIMENT_NAME) or self.EXPERIMENT_NAME,
            sim_start=0,
            sim_end=2*60*60,
            load_saved_model=True,
            overwrite_data=False,
        )

        # Initialize model attributes as None (will be loaded on first use)
        self._gnn_classifier = None
        self._data_processor = None
        self._gnn_dataloader = None

        # Initialize prediction storage
        # TODO clear after each time step
        self.rv_predictions = {}
        self.rr_predictions = {}
        
        # Cache for travel time calculations to avoid recomputation
        self._travel_time_cache = {}
        self._coord_cache = {}  # Cache for coordinate transformations

    def compute_new_vehicle_assignments(self, sim_time: int, vid_to_list_passed_VRLs: Dict[int, List[VehicleRouteLeg]],
                                        veh_objs_to_build: Dict[int, SimulationVehicle] = {
    },
            new_travel_times: bool = False, build_from_scratch: bool = False):
        """Compute new vehicle assignments, optionally using ML predictions. Writes training data after computation."""
        super().compute_new_vehicle_assignments(sim_time, vid_to_list_passed_VRLs, veh_objs_to_build, new_travel_times,
                                                build_from_scratch)
        self.write_train_data(sim_time)

    def write_train_data(self, sim_time: int):
        """Writes training data for the current timestep to disk in compressed pickle format."""
        if not self.enable_ml_training:
            return
        dir_path = os.path.join(self.train_data_path, str(sim_time))
        os.makedirs(dir_path, exist_ok=True)
        train_data = self.get_train_data()
        for name, data in train_data.items():
            path = os.path.join(dir_path, f'{name}_compressed.pkl')
            self.write_compressed_pickle(path, data)

    def get_train_data(self) -> dict[str, dict]:
        """Collects training data for the current timestep."""
        train_data = {
            self.config.request_features_key: self.get_req_features(),
            self.config.vehicle_features_key: self.get_veh_features(),
            self.config.vehicle_request_graph_key: self.get_v2r_graph_with_features(),
            self.config.request_request_graph_key: self.get_rr_graph_with_features(),
            self.config.init_assignment_key: self.current_assignments,
            self.config.assignment_key: self.optimisation_solutions,
        }
        return train_data

    def get_veh_features(self):
        """Collects vehicle features for training data with optimized coordinate batching."""
        # Batch coordinate transformations for efficiency
        all_positions = [vehicle.pos for vehicle in self.veh_objs.values()]
        if all_positions:
            coords = self.routing_engine.return_positions_lon_lat(all_positions)
            coord_dict = {vid: coords[i] for i, vid in enumerate(self.veh_objs.keys())}
        else:
            coord_dict = {}
            
        veh_features = {vid: {
            G_TRAIN_FEATURE_TYPE: vehicle.veh_type,  # Keep as string
            G_TRAIN_FEATURE_STATUS: int(vehicle.status.value),  # Convert enum value to int
            G_TRAIN_FEATURE_SOC: round(float(vehicle.soc), 4),  # Reduce precision
            G_TRAIN_FEATURE_V_POS_LAT: round(coord_dict[vid][0], 6) if vid in coord_dict else 0.0,  # 6 decimal places ~0.1m precision
            G_TRAIN_FEATURE_V_POS_LON: round(coord_dict[vid][1], 6) if vid in coord_dict else 0.0,  # 6 decimal places
        }
            for vid, vehicle in self.veh_objs.items()}
        return veh_features

    def get_req_features(self):
        """Collects request features for training data with optimized coordinate batching."""
        # Batch coordinate transformations for all origin and destination positions
        all_positions = []
        req_pos_mapping = {}
        for rid, req in self.active_requests.items():
            o_idx = len(all_positions)
            all_positions.append(req.o_pos)
            d_idx = len(all_positions)
            all_positions.append(req.d_pos)
            req_pos_mapping[rid] = (o_idx, d_idx)
            
        if all_positions:
            coords = self.routing_engine.return_positions_lon_lat(all_positions)
        else:
            coords = []
            
        req_features = {
            rid: {G_TRAIN_FEATURE_O_POS_LAT: round(coords[req_pos_mapping[rid][0]][0], 6) if coords else 0.0,
                  G_TRAIN_FEATURE_O_POS_LON: round(coords[req_pos_mapping[rid][0]][1], 6) if coords else 0.0,
                  G_TRAIN_FEATURE_D_POS_LAT: round(coords[req_pos_mapping[rid][1]][0], 6) if coords else 0.0,
                  G_TRAIN_FEATURE_D_POS_LON: round(coords[req_pos_mapping[rid][1]][1], 6) if coords else 0.0,
                  G_TRAIN_FEATURE_RQ_TIME: int(req.rq_time),  # Convert to int (seconds)
                  G_TRAIN_FEATURE_TW_PE: int(req.t_pu_earliest),  # Convert to int
                  G_TRAIN_FEATURE_TW_PL: int(req.t_pu_latest),  # Convert to int
                  G_TRAIN_FEATURE_DIRECT_TT: round(req.init_direct_tt, 1),  # 0.1s precision
                  G_TRAIN_FEATURE_DIRECT_TD: round(req.init_direct_td, 0),  # 1m precision
                  G_TRAIN_FEATURE_MAX_TRIP_TIME: int(req.max_trip_time),  # Convert to int
                  G_TRAIN_FEATURE_STATUS: req.status,  # Keep original type
                  G_TRAIN_FEATURE_LOCKED: 1 if self.r2v_locked.get(
                      rid, None) else 0
                  }
            for rid, req in self.active_requests.items()}
        return req_features

    def get_travel_time_v2r(self, vid: int, rid: int) -> dict:
        """get travel times between vehicle vid and request rid origin position with caching"""
        # Create cache key
        v_pos = self.veh_objs[vid].pos
        r_pos = self.active_requests[rid].get_o_stop_info()[0]
        cache_key = (v_pos, r_pos)
        
        # Check cache first
        if cache_key in self._travel_time_cache:
            cost, time, dist = self._travel_time_cache[cache_key]
        else:
            cost, time, dist = self.routing_engine.return_travel_costs_1to1(v_pos, r_pos)
            self._travel_time_cache[cache_key] = (cost, time, dist)
            
        return {
            G_TRAIN_FEATURE_TRAVEL_COST: round(cost, 2),  # 2 decimal places for cost
            G_TRAIN_FEATURE_TRAVEL_TIME: round(time, 1),  # 0.1s precision
            G_TRAIN_FEATURE_TRAVEL_DIST: round(dist, 0)   # 1m precision
        }

    def get_travel_time_r2r(self, rid1: int, rid2: int) -> dict:
        """get all travel times between the 6 combinations of rid1 and rid2 origins and destination positions"""
        req1, req2 = self.active_requests[rid1], self.active_requests[rid2]
        travel_times = {}
        
        for name, pos1, pos2 in self.get_od_pool_pairs(req1, req2):
            cost, time, dist = self.routing_engine.return_travel_costs_1to1(pos1, pos2)
            travel_times[f"{name}_{G_TRAIN_FEATURE_TRAVEL_COST}"] = round(cost, 2)
            travel_times[f"{name}_{G_TRAIN_FEATURE_TRAVEL_TIME}"] = round(time, 1)
            travel_times[f"{name}_{G_TRAIN_FEATURE_TRAVEL_DIST}"] = round(dist, 0)
            
        return travel_times

    @staticmethod
    def get_od_pool_pairs(req1: PlanRequest, req2: PlanRequest) -> List[tuple]:
        """Returns all combinations of origin and destination positions for two requests."""
        return [('o1_o2', req1.o_pos, req2.o_pos), ('o2_o1', req2.o_pos, req1.o_pos),
                ('o2_d1', req2.o_pos, req1.d_pos), ('o1_d2', req1.o_pos, req2.d_pos),
                ('d1_d2', req1.d_pos, req2.d_pos), ('d2_d1', req2.d_pos, req1.d_pos),
                ('d1_o2', req1.d_pos, req2.o_pos), ('d2_o1', req2.d_pos, req1.o_pos),
                ('o1_d1', req1.o_pos, req1.d_pos), ('o2_d2', req2.o_pos, req2.d_pos)]

    def get_rr_graph_with_features(self):
        """Return rr graph with travel-time features as flattened list for direct DataFrame conversion."""
        rr_edges = []
        for rid1, rid2 in self.rr:
            edge_features = {'source': rid1, 'target': rid2}
            edge_features.update(self.get_travel_time_r2r(rid1, rid2))
            rr_edges.append(edge_features)
        return rr_edges

    def get_v2r_graph_with_features(self):
        """Return v2r graph with travel-time features as flattened list for direct DataFrame conversion.

        This now includes:
        - Current v2r connections from self.v2r
        - Locked v2r connections from self.v2r_locked
        - Existing assignments from previous timestamps via self.fleetcontrol.veh_plans
        
        Returns flattened list of edges with source/target/features for direct DataFrame conversion.
        """
        v2r_edges = []
        v2r_locked = getattr(self, 'v2r_locked', {})

        # Union of vehicle ids present in v2r, v2r_locked, or with existing plans
        all_vids = set(self.v2r.keys()) | set(v2r_locked.keys())
        # Also include vehicles with existing assignments
        if hasattr(self.fleetcontrol, 'veh_plans'):
            all_vids.update(self.fleetcontrol.veh_plans.keys())

        for vid in all_vids:
            # Skip if vehicle object isn't available
            if vid not in self.veh_objs:
                continue

            # Collect rids from all sources
            rids = set()
            
            # 1. From v2r (current connections)
            v2r_entry = self.v2r.get(vid, {})
            v2r_rids = set()
            if isinstance(v2r_entry, dict):
                v2r_rids.update(v2r_entry.keys())
            else:
                try:
                    v2r_rids.update(v2r_entry)
                except Exception:
                    pass
            rids.update(v2r_rids)

            # 2. From v2r_locked (locked connections)
            locked_entry = v2r_locked.get(vid, {})
            locked_rids = set()
            if isinstance(locked_entry, dict):
                locked_rids.update(locked_entry.keys())
            else:
                try:
                    locked_rids.update(locked_entry)
                except Exception:
                    pass
            rids.update(locked_rids)

            # 3. From existing assignments (previous timestamps)
            existing_rids = set()
            if hasattr(self.fleetcontrol, 'veh_plans') and vid in self.fleetcontrol.veh_plans:
                try:
                    existing_rids = set(self.fleetcontrol.veh_plans[vid].get_involved_request_ids())
                    rids.update(existing_rids)
                except Exception as e:
                    LOG.debug(f"Could not get existing assignments for vehicle {vid}: {e}")

            # Debugging: Log sources of requests for this vehicle
            if len(rids) > 0:
                LOG.debug(f"Vehicle {vid} at sim_time {self.sim_time}: "
                         f"v2r={len(v2r_rids)}, locked={len(locked_rids)}, "
                         f"existing={len(existing_rids)}, total={len(rids)}")

            # Build flattened edge list for this vehicle, skipping missing requests
            missing_requests = []
            for rid in rids:
                if rid not in self.active_requests:
                    missing_requests.append(rid)
                    continue
                try:
                    edge_features = {'source': vid, 'target': rid}
                    edge_features.update(self.get_travel_time_v2r(vid, rid))
                    v2r_edges.append(edge_features)
                except Exception as e:
                    # If travel time computation fails for this pair, skip it
                    LOG.debug(f"Failed to compute travel time for v{vid}-r{rid}: {e}")
                    continue

            # Log if we had to skip any requests
            if missing_requests:
                LOG.debug(f"Vehicle {vid}: Skipped {len(missing_requests)} missing requests: {missing_requests[:5]}...")

        return v2r_edges

    @staticmethod
    def write_compressed_pickle(path: str, data) -> None:
        """Write data to a compressed pickle file using bz2 compression.
        
        Args:
            path: File path for the compressed pickle file
            data: Data to write
        """
        try:
            import bz2
            import pickle
            
            with bz2.BZ2File(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            LOG.debug(f"Successfully wrote compressed pickle {path}")
            
        except Exception as e:
            LOG.error(f"Failed to write compressed pickle file {path}: {e}")
            raise



    def _score_RV_RR_connections(self):
        """Predicts the rv-connections using either GNN or XGBoost model and stores them in dictionaries. Updates self.rv_predictions and self.rr_predictions."""
        # If ML is not enabled, skip prediction and use original implementation
        if not self.enable_ml_inference:
            return

        # Skip if no requests to consider
        if not self.rid_to_consider_for_global_optimisation:
            return

        # Initialize data processor if needed
        if not self._data_processor:
            self._data_processor = DataProcessor(
                self.train_data_path, self.config, prefer_processed=False)

        # Get current timestep data using existing data collection methods
        t0 = time.time()
        data = self._get_current_timestep_data()
        t_collect = time.time()

        try:
            # Get predictions and update connections
            edges_df = self._get_predictions(data)
            t_predict = time.time()
            self._save_predictions(edges_df)
            t_save = time.time()
            LOG.info(
                "ML TIMING {}: collect_features {:.3f}s | predict (incl. graph_features) {:.3f}s | "
                "save_predictions {:.3f}s | total {:.3f}s".format(
                    self.sim_time, t_collect - t0, t_predict - t_collect,
                    t_save - t_predict, t_save - t0)
            )

        except Exception as e:
            LOG.error(f"Error during prediction with {self.model_type}: {e}")
            import traceback
            traceback.print_exc()
            self.rv_predictions = {}
            self.rr_predictions = {}

    def _get_current_timestep_data(self):
        """Collects current timestep data for prediction."""
        return {
            self.config.request_features_key: self.get_req_features(),
            self.config.vehicle_features_key: self.get_veh_features(),
            # get_v2r/rr_graph_with_features() return flat lists; _add_graph_features needs nested dicts
            self.config.vehicle_request_graph_key: self._nest_edges(self.get_v2r_graph_with_features()),
            self.config.request_request_graph_key: self._nest_edges(self.get_rr_graph_with_features()),
            self.config.init_assignment_key: self.current_assignments
        }

    @staticmethod
    def _nest_edges(flat_edges: List[dict]) -> dict:
        """Convert a flat list of {'source', 'target', ...features} dicts into a nested
        {source: {target: features}} dict, matching data_processor.py's _load_timestep_data."""
        nested = defaultdict(dict)
        for edge in flat_edges:
            features = {k: v for k, v in edge.items() if k not in ('source', 'target')}
            nested[edge['source']][edge['target']] = features
        return nested

    def _get_predictions(self, data: dict):
        """Get predictions from the appropriate model"""
        # Load appropriate model if not loaded
        if not self._load_model():
            return None

        if self.model_type == GNN:
             # Initialize dataloader if needed
            if not self._gnn_dataloader:
                self._gnn_dataloader = GNNDataLoader(self.config)
            return self._get_gnn_predictions(data)
        else:
            LOG.error(f"Unsupported model type: {self.model_type}")
            return None
        
    def _get_gnn_predictions(self, data: dict):
        """Get predictions using GNN model.
        
        This method ensures data format consistency between training and inference through:
        1. Deterministic ID-to-index mapping (sorted by ID)
        2. Alphabetical column sorting for DataFrames
        3. Consistent one-hot encoding (using saved column names from training)
        4. Same normalization statistics applied to features
        5. Validation checks at each step
        
        Args:
            data: Dictionary containing raw timestep data with keys:
                - request_features_key: dict of request features
                - vehicle_features_key: dict of vehicle features
                - request_request_graph_key: dict of RR edges with features
                - vehicle_request_graph_key: dict of VR edges with features
        
        Returns:
            DataFrame with predictions (source, target, pred_score, edge_type) or None if error
        """
        stage_times = {}
        t = time.time()

        # 1. Add graph features (in-place modification of data dicts)
        self._data_processor._add_graph_features(self.sim_time, data)
        stage_times['add_graph_features'] = time.time() - t; t = time.time()

        # 2. Convert to DataFrames and map IDs to 0-based indices
        data_dfs, req_id_to_idx, veh_id_to_idx = self._convert_to_dataframes(
            data)
        stage_times['convert_to_dataframes'] = time.time() - t; t = time.time()

        # Validation: Check if we have data to predict on
        if not data_dfs:
            LOG.warning("No data available after DataFrame conversion.")
            return None

        # Log data statistics for debugging
        self._log_data_statistics(data_dfs, "After DataFrame conversion")

        # 3. Encode categorical features (Must be done BEFORE normalization), using the
        # one-hot columns fitted during training (loaded via GNNDataLoader.__init__)
        encoded_data = self._gnn_dataloader._encode_categorical_features(data_dfs)
        stage_times['encode_categorical'] = time.time() - t; t = time.time()

        # Validation: Check that one-hot encoding was applied correctly
        for key, expected_cols in self._gnn_dataloader._onehot_columns.items():
            if key in encoded_data and isinstance(encoded_data[key], pd.DataFrame):
                actual_cols = encoded_data[key].columns.tolist()
                if set(expected_cols) != set(actual_cols):
                    LOG.warning(f"Column mismatch for {key}. Expected {len(expected_cols)} columns, got {len(actual_cols)}.")

        # Log data statistics after encoding
        self._log_data_statistics(encoded_data, "After one-hot encoding")

        # 4. Normalize features
        normalized_data = self._gnn_dataloader._normalize_data(
            encoded_data)
        stage_times['normalize'] = time.time() - t; t = time.time()

        # 5. Create HeteroData Graph
        graph = self._create_hetero_graph(normalized_data)
        stage_times['create_hetero_graph'] = time.time() - t; t = time.time()

        # Validation: Check graph structure and dimensions
        self._validate_graph_structure(graph)

        # 6. Predict
        result = self._predict_with_gnn(graph, req_id_to_idx, veh_id_to_idx)
        stage_times['gnn_predict'] = time.time() - t

        LOG.info("ML STAGE TIMING {}: {}".format(
            self.sim_time,
            " | ".join(f"{k} {v:.3f}s" for k, v in stage_times.items())
        ))
        return result

    def _validate_graph_structure(self, graph: HeteroData) -> None:
        """Validate that the graph structure matches expectations.
        
        Args:
            graph: HeteroData graph to validate
        """
        for node_type in graph.node_types:
            if node_type in graph and hasattr(graph[node_type], 'x'):
                LOG.debug(f"{node_type} features shape: {graph[node_type].x.shape}")
        
        for edge_type in graph.edge_types:
            if edge_type in graph:
                if hasattr(graph[edge_type], 'edge_index'):
                    LOG.debug(f"{edge_type} edge_index shape: {graph[edge_type].edge_index.shape}")
                if hasattr(graph[edge_type], 'edge_attr'):
                    LOG.debug(f"{edge_type} edge_attr shape: {graph[edge_type].edge_attr.shape}")
    
    def _log_data_statistics(self, data_dfs: dict, stage: str) -> None:
        """Log data statistics for debugging.
        
        Args:
            data_dfs: Dictionary of DataFrames
            stage: Description of current processing stage
        """
        LOG.debug(f"=== Data Statistics at {stage} ===")
        for key, df in data_dfs.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                LOG.debug(f"{key}: shape={df.shape}, columns={len(df.columns)}, first_cols={list(df.columns[:5])}")

    def _convert_to_dataframes(self, data: dict):
        """Convert data dicts to DataFrames and create ID mappings.
        
        Ensures consistency with training data by:
        1. Sorting node IDs to create deterministic index mappings
        2. Sorting DataFrame columns alphabetically for consistent ordering
        3. Filling missing values with 0.0 (implicitly done by DataFrame.from_dict)
        
        Args:
            data: Dictionary with request_features_key, vehicle_features_key, 
                  request_request_graph_key, vehicle_request_graph_key
        
        Returns:
            Tuple of (data_dfs, req_id_to_idx, veh_id_to_idx)
            - data_dfs: Dict of DataFrames with sorted columns
            - req_id_to_idx: Mapping from request ID to 0-based index
            - veh_id_to_idx: Mapping from vehicle ID to 0-based index
        """
        dfs = {}

        # Requests
        req_data = data[self.config.request_features_key]
        if not req_data:
            return {}, {}, {}

        req_df = pd.DataFrame.from_dict(req_data, orient='index')
        req_df['timestep'] = self.sim_time
        # Create mapping: rid -> 0..N index
        # We sort index to ensure deterministic order
        sorted_rids = sorted(req_df.index)
        req_id_to_idx = {rid: i for i, rid in enumerate(sorted_rids)}
        # Reindex dataframe to match mapping order
        req_df = req_df.reindex(sorted_rids).reset_index(drop=True)
        # Ensure consistent column order (sort columns alphabetically for determinism)
        req_df = req_df[sorted(req_df.columns)]
        dfs[self.config.request_features_key] = req_df

        # Vehicles
        veh_data = data[self.config.vehicle_features_key]
        veh_df = pd.DataFrame.from_dict(veh_data, orient='index')
        veh_df['timestep'] = self.sim_time
        sorted_vids = sorted(veh_df.index)
        veh_id_to_idx = {vid: i for i, vid in enumerate(sorted_vids)}
        veh_df = veh_df.reindex(sorted_vids).reset_index(drop=True)
        # Ensure consistent column order (sort columns alphabetically for determinism)
        veh_df = veh_df[sorted(veh_df.columns)]
        dfs[self.config.vehicle_features_key] = veh_df

        # Graphs
        for key in [self.config.request_request_graph_key, self.config.vehicle_request_graph_key]:
            edge_rows = []
            graph_data = data.get(key, {})
            for source, targets in graph_data.items():
                for target, features in targets.items():
                    # Map source and target IDs to indices
                    if key == self.config.vehicle_request_graph_key:
                        src_idx = veh_id_to_idx.get(source)
                        tgt_idx = req_id_to_idx.get(target)
                    else:  # rr
                        src_idx = req_id_to_idx.get(source)
                        tgt_idx = req_id_to_idx.get(target)

                    if src_idx is not None and tgt_idx is not None:
                        row = {SOURCE: src_idx, TARGET: tgt_idx}
                        row.update(features)
                        edge_rows.append(row)

            if edge_rows:
                df = pd.DataFrame(edge_rows)
                df['timestep'] = self.sim_time
                # Ensure consistent column order (sort columns alphabetically for determinism)
                df = df[sorted(df.columns)]
                dfs[key] = df
            else:
                dfs[key] = pd.DataFrame()

        return dfs, req_id_to_idx, veh_id_to_idx

    def _load_model(self):
        """Load the appropriate model if not already loaded"""
        try:
            if self.model_type == GNN and not self._gnn_classifier:
                self._gnn_classifier = load_saved_model(self.config)
            return True
        except Exception as e:
            LOG.error(f"Error loading {self.model_type} model: {e}")
            return False

    def _predict_with_gnn(self, graph: HeteroData, req_id_to_idx: dict, veh_id_to_idx: dict):
        """Make predictions using GNN model"""
        # Inverse mappings
        idx_to_req_id = {v: k for k, v in req_id_to_idx.items()}
        idx_to_veh_id = {v: k for k, v in veh_id_to_idx.items()}

        edges_dict = {SOURCE: [], TARGET: [], PRED_SCORE: [], EDGE_TYPE: []}
        # get_edge_predictions now returns a dict by edge type
        pred_scores_by_type = get_edge_predictions(
            graph, self._gnn_classifier, device='cpu')

        for edge_type, edges in graph.edge_index_dict.items():
            if len(edges[0]) == 0:
                continue  # Skip empty edge types

            if edge_type == RV_EDGE_NAME:
                continue
            
            # Get predictions for this specific edge type
            edge_type_predictions = pred_scores_by_type[edge_type]
            
            for i, edge in enumerate(zip(edges[0], edges[1])):
                src_idx, tgt_idx = edge
                src_id = idx_to_veh_id[src_idx.item(
                )] if edge_type[0] == VEHICLE else idx_to_req_id[src_idx.item()]
                tgt_id = idx_to_req_id[tgt_idx.item(
                )] if edge_type[2] == REQUEST else idx_to_veh_id[tgt_idx.item()]
                score = float(edge_type_predictions[i])
                edges_dict[SOURCE].append(src_id)
                edges_dict[TARGET].append(tgt_id)
                edges_dict[PRED_SCORE].append(score)
                edges_dict[EDGE_TYPE].append(edge_type)
        return pd.DataFrame(edges_dict)

    def _create_hetero_graph(self, data: dict):
        """Creates a PyG HeteroData object from the current timestep data"""
        graph = HeteroData()

        # Calculate feature dimensions (needed for edge features)
        self._gnn_dataloader._calculate_feature_dimensions(data)

        # Use GNNDataLoader methods which expect data dict and timestep
        self._gnn_dataloader._add_node_features(graph, data, self.sim_time)
        self._gnn_dataloader._add_edge_features(graph, data, self.sim_time)

        # NormalizeFeatures() deliberately not applied here - see gnn_dataloader.py
        undirected_transform = T.ToUndirected(merge=True)
        graph = undirected_transform(graph)

        # Validate feature names if available from training
        self._validate_feature_names(graph)

        return graph
    
    def _validate_feature_names(self, graph: HeteroData) -> None:
        """Validate that inference graph has the same feature names as training.
        
        Loads the first training graph to get expected feature names and compares
        with the current inference graph's feature names.
        
        Args:
            graph: HeteroData graph from inference
        """
        try:
            # Try to load a training graph to get expected feature names
            train_graphs_path = self.config.processed_dir / self.config.experiment_name / TRAIN_GRAPHS
            if not train_graphs_path.exists():
                LOG.debug("No training graphs found for feature name validation")
                return
            
            # Load just the first graph to check feature names
            train_graphs = torch.load(train_graphs_path, weights_only=True)
            if not train_graphs:
                LOG.debug("Training graphs empty")
                return
                
            reference_graph = train_graphs[0]
            
            # Check node feature names
            for node_type in [REQUEST, VEHICLE]:
                if hasattr(reference_graph[node_type], 'feature_names') and hasattr(graph[node_type], 'feature_names'):
                    expected = reference_graph[node_type].feature_names
                    actual = graph[node_type].feature_names
                    if expected != actual:
                        LOG.warning(f"{node_type} feature mismatch!")
                        LOG.warning(f"  Expected ({len(expected)}): {expected[:5]}...")
                        LOG.warning(f"  Actual ({len(actual)}): {actual[:5]}...")
                        # Find differences
                        missing = set(expected) - set(actual)
                        extra = set(actual) - set(expected)
                        if missing:
                            LOG.warning(f"  Missing features: {missing}")
                        if extra:
                            LOG.warning(f"  Extra features: {extra}")
            
            # Check edge feature names
            for edge_type in [VR_EDGE_NAME, RR_EDGE_NAME]:
                if hasattr(reference_graph[edge_type], 'feature_names') and hasattr(graph[edge_type], 'feature_names'):
                    expected = reference_graph[edge_type].feature_names
                    actual = graph[edge_type].feature_names
                    if expected != actual:
                        LOG.warning(f"{edge_type} feature mismatch!")
                        LOG.warning(f"  Expected ({len(expected)}): {expected[:5]}...")
                        LOG.warning(f"  Actual ({len(actual)}): {actual[:5]}...")
                        # Find differences
                        missing = set(expected) - set(actual)
                        extra = set(actual) - set(expected)
                        if missing:
                            LOG.warning(f"  Missing features: {missing}")
                        if extra:
                            LOG.warning(f"  Extra features: {extra}")
                            
        except Exception as e:
            LOG.debug(f"Could not validate feature names: {e}")

    def _is_RR_pred_compatible(self, rid1: int, rid2: int) -> bool:
        """This method checks if the predicted RR connection between rid1 and rid2 is compatible.

        :param rid1: plan_request_id 1
        :param rid2: plan_request_id 2
        :return: True if compatible, False otherwise
        """
        if not self.enable_ml_inference:
            # If ML is disabled, assume all connections are compatible
            return True

        # Simple lookup - filtering was already done in _save_predictions
        return (rid1, rid2) in self.rr_predictions

    def _filter_RV_with_scores(self, vid: int, r_dict: Dict[int, float]) -> Dict[int, float]:
        """Filters the RV connections for a given vehicle based on predicted scores."""
        if not self.enable_ml_inference:
            return r_dict

        logging.debug('Number of Connections before filtering: ' + str(len(r_dict)))
        # Simple filtering - only keep requests that passed prediction filtering
        filtered_r_dict = {rid: tt for rid, tt in r_dict.items()
                          if (vid, rid) in self.rv_predictions}
        logging.debug('Number of Connections after filtering: ' + str(len(filtered_r_dict)))
        return filtered_r_dict

    def _save_predictions(self, edges_df):
        """Save predictions after applying filtering logic based on ml_selection_method.
        
        This method filters predictions once and stores only the connections that pass
        the threshold or top-k criteria. Subsequent lookups can then simply check for
        existence in these dictionaries.
        """
        if edges_df is None or edges_df.empty:
            self.rv_predictions = {}
            self.rr_predictions = {}
            return

        # Separate predictions by edge type
        vr_edges = edges_df[edges_df[EDGE_TYPE] == VR_EDGE_NAME]
        rr_edges = edges_df[edges_df[EDGE_TYPE] == RR_EDGE_NAME]

        # Apply filtering based on selection method
        if self.ml_selection_method == PREDICTION_THRESHOLD:
            # Filter by probability threshold
            vr_edges = vr_edges[vr_edges[PRED_SCORE] >= self.prediction_threshold]
            rr_edges = rr_edges[rr_edges[PRED_SCORE] >= self.prediction_threshold]
            
            # Store all that passed threshold
            self.rv_predictions = {(getattr(row, SOURCE), getattr(row, TARGET)): row.pred_score 
                                  for row in vr_edges.itertuples()}
            self.rr_predictions = {(getattr(row, SOURCE), getattr(row, TARGET)): row.pred_score 
                                  for row in rr_edges.itertuples()}
            
        elif self.ml_selection_method == TOP_K:
            # For VR edges: top-k vehicles per request
            self.rv_predictions = {}
            for rid in vr_edges[TARGET].unique():
                rid_edges = vr_edges[vr_edges[TARGET] == rid].nlargest(self.top_k_vr, PRED_SCORE)
                for row in rid_edges.itertuples():
                    self.rv_predictions[(getattr(row, SOURCE), getattr(row, TARGET))] = getattr(row, PRED_SCORE)
            
            # For RR edges: top-k per request
            self.rr_predictions = {}
            for rid in rr_edges[SOURCE].unique():
                rid_edges = rr_edges[rr_edges[SOURCE] == rid].nlargest(self.top_k_rr, PRED_SCORE)
                for row in rid_edges.itertuples():
                    self.rr_predictions[(row.source, row.target)] = row.pred_score
        
        else:
            LOG.warning(f"Unknown ml_selection_method: {self.ml_selection_method}. Storing all predictions.")
            self.rv_predictions = {(getattr(row, SOURCE), getattr(row, TARGET)): getattr(row, PRED_SCORE) 
                                  for row in vr_edges.itertuples()}
            self.rr_predictions = {(getattr(row, SOURCE), getattr(row, TARGET)): getattr(row, PRED_SCORE) 
                                  for row in rr_edges.itertuples()}

    def clear_databases(self):
        """Clears the stored prediction databases and caches."""
        self.rv_predictions = {}
        self.rr_predictions = {}
        # Clear travel time and coordinate caches to prevent memory buildup
        self._travel_time_cache.clear()
        self._coord_cache.clear()
        return super().clear_databases()
import logging
import pickle
import os
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
from gnn_project.data_processing.data_processor import DataProcessor
from gnn_project.config import Config
from gnn_project.dataloaders.gnn_dataloader import GNNDataLoader

from torch_geometric.data import HeteroData
from gnn_project.models.hetero_gat import HeteroGAT
from gnn_project.training.train_utils import get_edge_predictions


LOG = logging.getLogger(__name__)

class GNNAlonsoMoraAssignment(AlonsoMoraAssignmentOriginal):
    """Extension of Alonso-Mora Assignment Class that can optionally use ML for assignment predictions.

    This class extends the original Alonso-Mora implementation to optionally use machine learning
    models (XGBoost or GNN) for predicting feasible vehicle-request connections. By default,
    it uses the original implementation unless ML is explicitly enabled.

    Configuration via operator_attributes:
        enable_ml (bool): Whether to use ML predictions. Default: False
        model_type (str): Which model to use ('xgboost' or 'gnn'). Default: 'gnn'
        prediction_threshold (float): Probability threshold for predictions. Default: 0.5

    Example configuration:
        operator_attributes = {
            'enable_ml': True,  # Enable ML predictions
            'model_type': 'gnn',  # Use GNN model
            'ml_selection_method': 'probability',  # Use probability selection method
            'prediction_threshold': 0.7,  # Higher threshold for more selective pruning
        }
    """

    GNN_MODEL_PATH = 'gnn_project/data/models/gnn_v1/best_model.pt'
    ENABLE_ML_DEFAULT = False
    MODEL_TYPE_DEFAULT = 'gnn'  # only 'gnn' for now
    ML_SELECTION_METHOD_DEFAULT = 'top_k'  # 'probability' or 'top_k'
    TOP_K_DEFAULT = 10  # Used if selection method is 'top_k'
    PREDICTION_THRESHOLD_DEFAULT = 0.5  # Used if selection method is 'probability'

    def __init__(self, fleetcontrol: FleetControlBase, routing_engine: NetworkBase, sim_time: int,
                 obj_function: Callable, operator_attributes: dict, optimisation_cores: int = 1, seed: int = 6061992,
                 veh_objs_to_build: Dict[int, SimulationVehicle] = {}):
        """Initializes the GNNAlonsoMoraAssignment with optional ML settings."""
        super().__init__(fleetcontrol, routing_engine, sim_time, obj_function, operator_attributes, optimisation_cores,
                         seed, veh_objs_to_build)
        self.train_data_path = os.path.join(
            self.fleetcontrol.dir_names[G_DIR_OUTPUT], G_DIR_TRAIN)

        # Configure ML settings from operator attributes
        # TODO add to operator attributes
        self.enable_ml = operator_attributes.get(
            'enable_ml', self.ENABLE_ML_DEFAULT)  # ML is disabled by default
        if self.enable_ml:
            self.model_type = operator_attributes.get(
                'model_type', self.MODEL_TYPE_DEFAULT)  # 'xgboost' or 'gnn'
            self.ml_selection_method = operator_attributes.get(
                'ml_selection_method', self.ML_SELECTION_METHOD_DEFAULT)  # 'probability' or 'top_k'
            self.top_k = operator_attributes.get('top_k', self.TOP_K_DEFAULT)
            self.prediction_threshold = operator_attributes.get(
                'prediction_threshold', self.PREDICTION_THRESHOLD_DEFAULT)

        # Initialize Config
        self.config = Config()

        # Initialize model attributes as None (will be loaded on first use)
        self._gnn_classifier = None
        self._data_processor = None
        self._gnn_dataloader = None

        # Initialize prediction storage
        self.rv_predictions = {}
        self.rr_predictions = {}

        if self.enable_ml:
            LOG.info(f"ML predictions enabled using {self.model_type} model")

    def compute_new_vehicle_assignments(self, sim_time: int, vid_to_list_passed_VRLs: Dict[int, List[VehicleRouteLeg]],
                                        veh_objs_to_build: Dict[int, SimulationVehicle] = {
    },
            new_travel_times: bool = False, build_from_scratch: bool = False):
        """Compute new vehicle assignments, optionally using ML predictions. Writes training data after computation."""
        super().compute_new_vehicle_assignments(sim_time, vid_to_list_passed_VRLs, veh_objs_to_build, new_travel_times,
                                                build_from_scratch)
        self.write_train_data(sim_time)

    def write_train_data(self, sim_time: int):
        """Writes training data for the current timestep to disk."""
        dir_path = os.path.join(self.train_data_path, str(sim_time))
        os.makedirs(dir_path, exist_ok=True)
        train_data = self.get_train_data()
        for name, data in train_data.items():
            path = os.path.join(dir_path, f'{name}.pkl')
            self.write_pickle(path, data)

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
        """Collects vehicle features for training data."""
        veh_features = {vid: {
            G_TRAIN_FEATURE_TYPE: vehicle.veh_type,
            G_TRAIN_FEATURE_STATUS: vehicle.status.value,
            G_TRAIN_FEATURE_SOC: vehicle.soc,
            G_TRAIN_FEATURE_V_POS_LAT: self.routing_engine.return_positions_lon_lat([vehicle.pos])[0][0],
            G_TRAIN_FEATURE_V_POS_LON: self.routing_engine.return_positions_lon_lat([vehicle.pos])[0][1],
        }
            for vid, vehicle in self.veh_objs.items()}
        return veh_features

    def get_req_features(self):
        """Collects request features for training data."""
        req_features = {
            rid: {G_TRAIN_FEATURE_O_POS_LAT: self.routing_engine.return_positions_lon_lat([req.o_pos])[0][0],
                  G_TRAIN_FEATURE_O_POS_LON: self.routing_engine.return_positions_lon_lat([req.o_pos])[0][1],
                  G_TRAIN_FEATURE_D_POS_LAT: self.routing_engine.return_positions_lon_lat([req.d_pos])[0][0],
                  G_TRAIN_FEATURE_D_POS_LON: self.routing_engine.return_positions_lon_lat([req.d_pos])[0][1],
                  G_TRAIN_FEATURE_RQ_TIME: req.rq_time,
                  G_TRAIN_FEATURE_TW_PE: req.t_pu_earliest,
                  G_TRAIN_FEATURE_TW_PL: req.t_pu_latest,
                  G_TRAIN_FEATURE_DIRECT_TT: req.init_direct_tt,
                  G_TRAIN_FEATURE_DIRECT_TD: req.init_direct_td,
                  G_TRAIN_FEATURE_MAX_TRIP_TIME: req.max_trip_time,
                  G_TRAIN_FEATURE_STATUS: req.status,
                  G_TRAIN_FEATURE_LOCKED: 1 if self.r2v_locked.get(
                      rid, None) else 0
                  }
            for rid, req in self.active_requests.items()}
        return req_features

    def get_travel_time_v2r(self, vid: int, rid: int) -> dict:
        """get travel times between vehicle vid and request rid origin position"""
        v_pos = self.veh_objs[vid].pos
        r_pos = self.active_requests[rid].get_o_stop_info()[0]
        return {key: val for key, val in
                zip([G_TRAIN_FEATURE_TRAVEL_COST, G_TRAIN_FEATURE_TRAVEL_TIME, G_TRAIN_FEATURE_TRAVEL_DIST],
                    self.routing_engine.return_travel_costs_1to1(v_pos, r_pos))}

    def get_travel_time_r2r(self, rid1: int, rid2: int) -> dict:
        """get all travel times between the 6 combinations of rid1 and rid2 origins and destination positions"""
        req1, req2 = self.active_requests[rid1], self.active_requests[rid2]
        return {G_TRAIN_FEATURE_TRAVEL_COST: {name: {key: val for key, val in
                       zip([G_TRAIN_FEATURE_TRAVEL_COST, G_TRAIN_FEATURE_TRAVEL_TIME,
                            G_TRAIN_FEATURE_TRAVEL_DIST],
                           self.routing_engine.return_travel_costs_1to1(pos1, pos2))} for
                name, pos1, pos2 in
                self.get_od_pool_pairs(req1, req2)}}

    @staticmethod
    def get_od_pool_pairs(req1, req2):
        """Returns all combinations of origin and destination positions for two requests."""
        return [('o1_o2', req1.o_pos, req2.o_pos), ('o2_o1', req2.o_pos, req1.o_pos),
                ('o2_d1', req2.o_pos, req1.d_pos), ('o1_d2', req1.o_pos, req2.d_pos),
                ('d1_d2', req1.d_pos, req2.d_pos), ('d2_d1', req2.d_pos, req1.d_pos),
                ('d1_o2', req1.d_pos, req2.o_pos), ('d2_o1', req2.d_pos, req1.o_pos),
                ('o1_d1', req1.o_pos, req1.d_pos), ('o2_d2', req2.o_pos, req2.d_pos)]

    def get_rr_graph_with_features(self):
        """Return rr graph with travel-time features."""
        rr_graph = defaultdict(dict)
        for rid1, rid2 in self.rr:
            rr_graph[rid1][rid2] = self.get_travel_time_r2r(rid1, rid2)
        return rr_graph

    def get_v2r_graph_with_features(self):
        """Return v2r graph with travel-time features.

        This now includes locked v2r connections provided in `self.v2r_locked`.
        We merge rids from `self.v2r` and `self.v2r_locked`, skip missing vehicles
        or requests, and only include entries for which travel-time features
        could be computed.
        """
        v2r_graph = {}
        v2r_locked = getattr(self, 'v2r_locked', {})

        # Union of vehicle ids present in either map
        all_vids = set(self.v2r.keys()) | set(v2r_locked.keys())

        for vid in all_vids:
            # Skip if vehicle object isn't available
            if vid not in self.veh_objs:
                continue

            # Collect rids from both maps; handle dict or iterable values
            rids = set()
            v2r_entry = self.v2r.get(vid, {})
            if isinstance(v2r_entry, dict):
                rids.update(v2r_entry.keys())
            else:
                try:
                    rids.update(v2r_entry)
                except Exception:
                    pass

            locked_entry = v2r_locked.get(vid, {})
            if isinstance(locked_entry, dict):
                rids.update(locked_entry.keys())
            else:
                try:
                    rids.update(locked_entry)
                except Exception:
                    pass

            # Build feature map for this vehicle, skipping missing requests
            features = {}
            for rid in rids:
                if rid not in self.active_requests:
                    continue
                try:
                    features[rid] = self.get_travel_time_v2r(vid, rid)
                except Exception:
                    # If travel time computation fails for this pair, skip it
                    continue

            if features:
                v2r_graph[vid] = features

        return v2r_graph

    @staticmethod
    def write_pickle(path, data: Dict):
        """Writes data to a pickle file at the specified path."""
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _score_RV_RR_connections(self):
        """Predicts the rv-connections using either the original method or ML models (XGBoost/GNN)."""
        # If ML is not enabled, skip prediction and use original implementation
        if not self.enable_ml:
            return

        # Skip if no requests to consider
        if not self.rid_to_consider_for_global_optimisation:
            return

        # Initialize data processor if needed
        if not self._data_processor:
            self._data_processor = DataProcessor(
                self.train_data_path, self.config, prefer_processed=False)

        # Initialize dataloader if needed
        if not self._gnn_dataloader:
            self._gnn_dataloader = GNNDataLoader(self.config)

        # Get current timestep data using existing data collection methods
        data = self._get_timestep_data()

        try:
            # Get predictions and update connections
            edges_df = self._get_predictions(data)
            if edges_df is None or edges_df.empty:
                self.rv_predictions = {}
                self.rr_predictions = {}
                return
            
            # TODO Store predictions in dict format for easy lookup
            self.rv_predictions = {
                (row['source'], row['target']): row['pred_prob'] for _, row in edges_df.iterrows()
            }
            self.rr_predictions = {}

        except Exception as e:
            LOG.error(f"Error during prediction with {self.model_type}: {e}")
            import traceback
            traceback.print_exc()
            self.rv_predictions = {}
            self.rr_predictions = {}

    def _get_timestep_data(self):
        """Collects current timestep data for prediction."""
        return {
            self.config.request_features_key: self.get_req_features(),
            self.config.vehicle_features_key: self.get_veh_features(),
            self.config.vehicle_request_graph_key: self.get_v2r_graph_with_features(),
            self.config.request_request_graph_key: self.get_rr_graph_with_features(),
            self.config.init_assignment_key: self.current_assignments
        }

    def _get_predictions(self, data):
        """Get predictions from the appropriate model"""
        # Load appropriate model if not loaded
        if not self._load_model():
            return None

        if self.model_type == 'gnn':
            # GNN pipeline
            # 1. Add graph features (in-place modification of data dicts)
            self._data_processor._add_graph_features(self.sim_time, data)

            # 2. Convert to DataFrames and map IDs to 0-based indices
            data_dfs, req_id_to_idx, veh_id_to_idx = self._convert_to_dataframes(
                data)

            # 3. Encode categorical features (Must be done BEFORE normalization)
            # TODO save _onehot_columns
            encoded_data = self._gnn_dataloader._encode_categorical_features(
                data_dfs)

            # 4. Normalize features
            normalized_data = self._gnn_dataloader._normalize_data(
                encoded_data)

            # 5. Create HeteroData Graph
            graph = self._create_hetero_graph(normalized_data)

            # 6. Predict
            return self._predict_with_gnn(graph, req_id_to_idx, veh_id_to_idx)
        else:
            LOG.error(f"Unsupported model type: {self.model_type}")
            return None

    def _convert_to_dataframes(self, data):
        """Convert data dicts to DataFrames and create ID mappings"""
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
        dfs[self.config.request_features_key] = req_df

        # Vehicles
        veh_data = data[self.config.vehicle_features_key]
        veh_df = pd.DataFrame.from_dict(veh_data, orient='index')
        veh_df['timestep'] = self.sim_time  # TODO training uses indices as timesteps. can set to 0?
        sorted_vids = sorted(veh_df.index)
        veh_id_to_idx = {vid: i for i, vid in enumerate(sorted_vids)}
        veh_df = veh_df.reindex(sorted_vids).reset_index(drop=True)
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
                        row = {'source': src_idx, 'target': tgt_idx}
                        row.update(features)
                        edge_rows.append(row)

            if edge_rows:
                df = pd.DataFrame(edge_rows)
                df['timestep'] = self.sim_time  # TODO training uses indices as timesteps. can set to 0?
                dfs[key] = df
            else:
                dfs[key] = pd.DataFrame()

        return dfs, req_id_to_idx, veh_id_to_idx

    def _load_model(self):
        """Load the appropriate model if not already loaded"""
        try:
            if self.model_type == 'gnn' and not self._gnn_classifier:
                self._gnn_classifier = HeteroGAT(self.config)
                state_dict = torch.load(self.GNN_MODEL_PATH, map_location='cpu')
                self._gnn_classifier.load_state_dict(state_dict)
                self._gnn_classifier.eval()
            return True
        except Exception as e:
            LOG.error(f"Error loading {self.model_type} model: {e}")
            return False

    def _predict_with_gnn(self, graph, req_id_to_idx, veh_id_to_idx):
        """Make predictions using GNN model"""
        # Inverse mappings
        idx_to_req_id = {v: k for k, v in req_id_to_idx.items()}
        idx_to_veh_id = {v: k for k, v in veh_id_to_idx.items()}

        edges_df = pd.DataFrame(columns=['source', 'target', 'pred_prob'])
        with torch.no_grad():
            pred_probs = get_edge_predictions(
                graph, self._gnn_classifier, device='cpu')
    
            # TODO
            # pred_probs has all edge types
        return edges_df

    def _create_hetero_graph(self, data):
        """Creates a PyG HeteroData object from the current timestep data"""
        graph = HeteroData()

        # Calculate feature dimensions (needed for edge features)
        self._gnn_dataloader._calculate_feature_dimensions(data)

        # Use GNNDataLoader methods which expect data dict and timestep
        self._gnn_dataloader._add_node_features(graph, data, self.sim_time)
        self._gnn_dataloader._add_edge_features(graph, data, self.sim_time)

        # Apply transformations to match training pipeline
        undirected_transform = T.ToUndirected(merge=True)
        graph = undirected_transform(graph)
        graph = T.NormalizeFeatures()(graph)

        return graph

    def _is_RR_pred_compatible(self, rid1: int, rid2: int) -> bool:
        """This method checks if the predicted RR connection between rid1 and rid2 is compatible.

        :param rid1: plan_request_id 1
        :param rid2: plan_request_id 2
        :return: True if compatible, False otherwise
        """
        if not self.enable_ml or not self.rr_predictions:
            # If ML is disabled or no predictions, assume all connections are compatible
            return True

        score = self.rr_predictions.get((rid1, rid2), None)
        if self.ml_selection_method == "probability":
            return score is not None and score >= self.prediction_threshold
        elif self.ml_selection_method == "top_k":
            # TODO: Implement top_k logic
            return score is not None
        return True

    def _filter_RV_with_scores(self, vid: int, r_dict: Dict[int, float]) -> Dict[int, float]:
        """Filters the RV connections for a given vehicle based on predicted scores."""
        if not self.enable_ml:
            return r_dict

        if self.ml_selection_method == 'probability':
            filtered_r_dict = {rid: tt for rid, tt in r_dict.items(
            ) if self.rv_predictions.get((vid, rid), 0) >= self.prediction_threshold}
            return filtered_r_dict
        elif self.ml_selection_method == 'top_k':
            # Select top-k requests based on scores
            top_k_rids = sorted(r_dict, key=lambda rid: self.rv_predictions.get(
                (vid, rid), 0), reverse=True)[:self.top_k]
            filtered_r_dict = {rid: r_dict[rid] for rid in top_k_rids}
            return filtered_r_dict
        return r_dict

import pickle
import os
from typing import Dict, List, Callable
from collections import defaultdict
import pandas as pd
import torch
import numpy as np

from src.fleetctrl.FleetControlBase import FleetControlBase
from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraAssignmentOriginal import AlonsoMoraAssignmentOriginal
from src.fleetctrl.pooling.batch.AlonsoMora.AlonsoMoraAssignment import AlonsoMoraAssignment
from src.misc.globals import *
from src.routing.NetworkBase import NetworkBase
from src.simulation.Legs import VehicleRouteLeg
from src.simulation.Vehicles import SimulationVehicle
from notebooks.data_processing.data_processor import DataProcessor
from notebooks.models.EdgeClassifier import XGBClassifier

import torch
from torch_geometric.data import HeteroData
from notebooks.models.HeteroGAT import HeteroGAT
from notebooks.dataloaders.GNNDataLoader import GNNDataLoader

class GNNAlonsoMoraAssignment(AlonsoMoraAssignmentOriginal):
    """Extension of Alonso-Mora Assignment Class that can optionally use ML for assignment predictions.
    
    This class extends the original Alonso-Mora implementation to optionally use machine learning
    models (XGBoost or GNN) for predicting feasible vehicle-request connections. By default,
    it uses the original implementation unless ML is explicitly enabled.
    
    Configuration via operator_attributes:
        enable_ml (bool): Whether to use ML predictions. Default: False
        model_type (str): Which model to use ('xgboost' or 'gnn'). Default: 'xgboost'
        prediction_threshold (float): Probability threshold for predictions. Default: 0.5
        
    Example configuration:
        operator_attributes = {
            'enable_ml': True,  # Enable ML predictions
            'model_type': 'gnn',  # Use GNN model
            'prediction_threshold': 0.7  # Higher threshold for more selective pruning
        }
    """

    XGBOOST_MODEL_PATH = 'notebooks/data/train/_20250717_131757/xgbclassifier_rr.pkl'
    GNN_MODEL_PATH = 'notebooks/data/train/_20250717_131757/gnn_classifier.pt'
    
    def __init__(self, fleetcontrol: FleetControlBase, routing_engine: NetworkBase, sim_time: int,
                 obj_function: Callable, operator_attributes: dict, optimisation_cores: int = 1, seed: int = 6061992,
                 veh_objs_to_build: Dict[int, SimulationVehicle] = {}):
        super().__init__(fleetcontrol, routing_engine, sim_time, obj_function, operator_attributes, optimisation_cores,
                         seed, veh_objs_to_build)
        self.train_data_path = os.path.join(self.fleetcontrol.dir_names[G_DIR_OUTPUT], G_DIR_TRAIN)
        
        # Configure ML settings from operator attributes
        self.enable_ml = operator_attributes.get('enable_ml', False)  # ML is disabled by default
        self.model_type = operator_attributes.get('model_type', 'xgboost')  # 'xgboost' or 'gnn'
        self.prediction_threshold = operator_attributes.get('prediction_threshold', 0.5)
        
        # Initialize model attributes as None (will be loaded on first use)
        self._xgb_classifier = None
        self._gnn_classifier = None
        self._data_processor = None
        
        if self.enable_ml:
            print(f"ML predictions enabled using {self.model_type} model")


    def compute_new_vehicle_assignments(self, sim_time: int, vid_to_list_passed_VRLs: Dict[int, List[VehicleRouteLeg]],
                                        veh_objs_to_build: Dict[int, SimulationVehicle] = {},
                                        new_travel_times: bool = False, build_from_scratch: bool = False):
        super().compute_new_vehicle_assignments(sim_time, vid_to_list_passed_VRLs, veh_objs_to_build, new_travel_times,
                                                build_from_scratch)
        self.write_train_data(sim_time)
        # TODO save top 10 optimized assignments

    def write_train_data(self, sim_time: int):
        dir_path = os.path.join(self.train_data_path, str(sim_time))
        os.makedirs(dir_path, exist_ok=True)
        train_data = self.get_train_data()
        for name, data in train_data.items():
            path = os.path.join(dir_path, f'{name}.pkl')
            self.write_pickle(path, data)

    def get_train_data(self) -> dict[str, dict]:
        train_data = {
            G_TRAIN_RR_FILE: self.get_rr_graph_with_features(),
            G_TRAIN_VR_FILE: self.get_v2r_graph_with_features(),
            G_TRAIN_ASSIGNMENTS_FILE: self.optimisation_solutions,
            G_TRAIN_INIT_ASSIGNMENTS_FILE: self.current_assignments,
            G_TRAIN_REQ_FEATURES_FILE: self.get_req_features(),
            G_TRAIN_VEH_FEATURES_FILE: self.get_veh_features()
        }
        return train_data

    def get_veh_features(self):
        veh_features = {vid: {
            G_TRAIN_FEATURE_TYPE: vehicle.veh_type,
            G_TRAIN_FEATURE_STATUS: vehicle.status.value,
            G_TRAIN_FEATURE_SOC: vehicle.soc,
            G_TRAIN_FEATURE_V_POS_LAT: self.routing_engine.return_positions_lon_lat([vehicle.pos])[0][0],
            G_TRAIN_FEATURE_V_POS_LON: self.routing_engine.return_positions_lon_lat([vehicle.pos])[0][1],
            # TODO add other vehicle features as needed
        }
        for vid, vehicle in self.veh_objs.items()}
        return veh_features

    def get_req_features(self):
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
                  G_TRAIN_FEATURE_LOCKED: 1 if req.locked else 0
                  # TODO add other request features as needed
                  }
            for rid, req in self.active_requests.items()}
        return req_features

    def get_travel_time_v2r(self, vid, rid):
        v_pos = self.veh_objs[vid].pos
        r_pos = self.active_requests[rid].get_o_stop_info()[0]
        return {key: val for key, val in
                zip([G_TRAIN_FEATURE_TRAVEL_COST, G_TRAIN_FEATURE_TRAVEL_TIME, G_TRAIN_FEATURE_TRAVEL_DIST],
                    self.routing_engine.return_travel_costs_1to1(v_pos, r_pos))}

    def get_travel_time_r2r(self, rid1, rid2):
        """get all travel times between the 6 combinations of rid1 and rid2 origins and destination positions"""
        req1, req2 = self.active_requests[rid1], self.active_requests[rid2]
        return {G_TRAIN_FEATURE_TRAVEL_COST: [{key: val for key, val in
                                               zip([G_TRAIN_FEATURE_TRAVEL_COST, G_TRAIN_FEATURE_TRAVEL_TIME,
                                                    G_TRAIN_FEATURE_TRAVEL_DIST],
                                                   self.routing_engine.return_travel_costs_1to1(pos1, pos2))} for
                                              pos1, pos2 in
                                              self.get_od_pool_pairs(req1, req2)]}

    @staticmethod
    def get_od_pool_pairs(req1, req2):
        # TODO double check if this is correct
        """Returns all combinations of origin and destination positions for two requests."""
        return [(req1.o_pos, req1.d_pos), (req1.o_pos, req2.o_pos), (req1.o_pos, req2.d_pos), 
                (req1.d_pos, req2.o_pos), (req2.o_pos, req2.d_pos), (req1.d_pos, req2.d_pos)]

    def get_rr_graph_with_features(self):
        rr_graph = defaultdict(dict)
        for rid1, rid2 in self.rr:
            rr_graph[rid1][rid2] = self.get_travel_time_r2r(rid1, rid2)
        return rr_graph

    def get_v2r_graph_with_features(self):
        return {vid: {rid: self.get_travel_time_v2r(vid, rid) for rid in rids} for vid, rids in self.v2r.items()}

    @staticmethod
    def write_pickle(path, data: Dict):
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _predict_rv_connections(self):
        """
        Predicts the rv-connections using either the original method or ML models (XGBoost/GNN).
        This is intended for online use at each simulation timestep.
        """
        # If ML is not enabled, skip prediction and use original implementation
        if not self.enable_ml:
            return
            
        print(f"predict rv connections (online) using {self.model_type} model")
        
        # Skip if no requests to consider
        if not self.rid_to_consider_for_global_optimisation:
            return

        # Initialize data processor if needed
        if not self._data_processor:
            self._data_processor = DataProcessor(None, None)

        # Get current timestep data using existing data collection methods
        data = self._get_timestep_data()
        
        try:
            # Process data through existing pipeline
            processed_data = self._data_processor.process_single_timestep(
                data, 
                edge_type='vr_graph',
                timestep=self.sim_time
            )

            # Get predictions and update connections
            pruned_edges = self._get_predictions(processed_data)
            if pruned_edges is not None:
                self._update_connection_maps(pruned_edges)
            
        except Exception as e:
            print(f"Error during prediction with {self.model_type}: {e}")
            import traceback
            traceback.print_exc()

    def _get_timestep_data(self):
        """Collects current timestep data using existing data collection methods"""
        return {
            'req_features': self.get_req_features(),
            'veh_features': self.get_veh_features(),
            'vr_graph': self.get_v2r_graph_with_features(),
            'rr_graph': self.get_rr_graph_with_features(),
            'init_assignments': self.current_assignments,
            'assignments': self.optimisation_solutions
        }

    def _get_predictions(self, processed_data):
        """Get predictions from the appropriate model"""
        merged_data, _ = processed_data  # Unpack processed data

        # Load appropriate model if not loaded
        if not self._load_model():
            return None

        if self.model_type == 'xgboost':
            return self._predict_with_xgboost(merged_data)
        else:  # gnn
            return self._predict_with_gnn(merged_data)

    def _load_model(self):
        """Load the appropriate model if not already loaded"""
        try:
            if self.model_type == 'xgboost' and not self._xgb_classifier:
                self._xgb_classifier = XGBClassifier(dataloader=None)
                import joblib
                self._xgb_classifier.pipeline = joblib.load(self.XGBOOST_MODEL_PATH)
            elif self.model_type == 'gnn' and not self._gnn_classifier:
                self._gnn_classifier = HeteroGAT()
                self._gnn_classifier.load_state_dict(torch.load(self.GNN_MODEL_PATH))
                self._gnn_classifier.eval()
            return True
        except Exception as e:
            print(f"Error loading {self.model_type} model: {e}")
            return False

    def _predict_with_xgboost(self, merged_data):
        """Make predictions using XGBoost model"""
        feature_cols = [col for col in merged_data.columns if col not in ['label', 'init_label']]
        X_pred = merged_data[feature_cols]
        y_pred_proba = self._xgb_classifier.pipeline.predict_proba(X_pred)[:, 1]
        merged_data['pred_prob'] = y_pred_proba
        return merged_data[merged_data['pred_prob'] >= self.prediction_threshold]

    def _predict_with_gnn(self, merged_data):
        """Make predictions using GNN model"""
        # The DataProcessor has already created the graph structure
        # We just need to convert it to PyG format
        graph = self._data_processor.create_pyg_graph(merged_data)
        
        with torch.no_grad():
            pred_probs = torch.sigmoid(self._gnn_classifier(graph))
            
            # Convert to DataFrame format matching XGBoost output
            edges_df = pd.DataFrame({
                'source': graph['vehicle', 'connects', 'request'].edge_index[0].numpy(),
                'target': graph['vehicle', 'connects', 'request'].edge_index[1].numpy(),
                'pred_prob': pred_probs.numpy()
            })
            
        return edges_df[edges_df['pred_prob'] >= self.prediction_threshold]
            
    def _predict_xgboost(self, data):
        """XGBoost-specific prediction logic"""
        if not self._data_processor:
            self._data_processor = DataProcessor(None, None)
            
        # Process features
        merged, _ = self._data_processor.process_single_timestep(data, edge_type='vr_graph', timestep=self.sim_time)
        
        # Load model if needed
        if not self._xgb_classifier:
            self._xgb_classifier = XGBClassifier(dataloader=None)
            import joblib
            try:
                self._xgb_classifier.pipeline = joblib.load(self.XGBOOST_MODEL_PATH)
            except Exception as e:
                print(f"Error loading XGBoost model: {e}")
                return None

        # Predict
        feature_cols = [col for col in merged.columns if col not in ['label', 'init_label']]
        X_pred = merged[feature_cols]
        y_pred_proba = self._xgb_classifier.pipeline.predict_proba(X_pred)[:, 1]
        merged['pred_prob'] = y_pred_proba
        
        return merged[merged['pred_prob'] >= self.prediction_threshold]
        
    def _predict_gnn(self, data):
        """GNN-specific prediction logic"""
        if not self._data_processor:
            self._data_processor = DataProcessor(None, None)
            
        # Load model if needed
        if not self._gnn_classifier:
            try:
                self._gnn_classifier = HeteroGAT()
                self._gnn_classifier.load_state_dict(torch.load(self.GNN_MODEL_PATH))
                self._gnn_classifier.eval()
            except Exception as e:
                print(f"Error loading GNN model: {e}")
                return None
                
        # Convert to heterogeneous graph
        graph = self._create_hetero_graph(data)
        
        # Predict
        with torch.no_grad():
            edge_index = graph['vehicle', 'connects', 'request'].edge_index
            edge_attr = graph['vehicle', 'connects', 'request'].edge_attr
            
            pred_probs = self._gnn_classifier(graph)
            pred_probs = torch.sigmoid(pred_probs)
            
            # Convert to DataFrame format matching XGBoost output
            edges_df = pd.DataFrame({
                'source': edge_index[0].numpy(),
                'target': edge_index[1].numpy(),
                'pred_prob': pred_probs.numpy()
            })
            
        return edges_df[edges_df['pred_prob'] >= self.prediction_threshold]
        
    def _create_hetero_graph(self, data):
        """Creates a PyG HeteroData object from the current timestep data"""
        graph = HeteroData()
        
        # Add node features
        req_features = pd.DataFrame(data['req_features'])
        veh_features = pd.DataFrame(data['veh_features'])
        
        graph['request'].x = torch.FloatTensor(req_features.drop(['timestep'], axis=1).values)
        graph['vehicle'].x = torch.FloatTensor(veh_features.drop(['timestep'], axis=1).values)
        
        # Add edge features
        vr_edges = pd.DataFrame(data['vr_graph'])
        edge_index = torch.tensor([[int(s), int(t)] for s, t in zip(vr_edges['source'], vr_edges['target'])], dtype=torch.long).t()
        edge_attr = torch.FloatTensor(vr_edges.drop(['source', 'target', 'label'], axis=1).values)
        
        graph['vehicle', 'connects', 'request'].edge_index = edge_index
        graph['vehicle', 'connects', 'request'].edge_attr = edge_attr
        
        return graph
        
    def _update_connection_maps(self, pruned_edges):
        """Updates v2r and r2v mappings based on predicted edges"""
        # Clear existing connections
        for vid in self.v2r:
            self.v2r[vid].clear()
        for rid in self.r2v:
            self.r2v[rid].clear()

        # Add predicted feasible connections
        n_total = len(pruned_edges)
        n_valid = 0
        
        for _, edge in pruned_edges.iterrows():
            vid = int(edge['source'])
            rid = int(edge['target'])
            
            # Skip if vehicle or request doesn't exist
            if vid not in self.v2r or rid not in self.active_requests:
                continue
                
            # Add bidirectional connections
            self.v2r[vid][rid] = 1
            if rid not in self.r2v:
                self.r2v[rid] = {}
            self.r2v[rid][vid] = 1
            n_valid += 1

        print(f"Kept {n_valid} valid connections out of {n_total} predicted connections")

    def _get_predicted_score(self, vid: int, rid: int, edge_type: str) -> float:
        """This method returns the predicted score for a given rid and vid.
        If no prediction is available, it returns None.
        
        :param vid: vehicle_id
        :param rid: plan_request_id
        :return: predicted score or None
        """
        # TODO hoda: implement ML model prediction logic here
        # For now, we return None as a placeholder
        return None

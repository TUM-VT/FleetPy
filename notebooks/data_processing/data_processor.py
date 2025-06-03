# Standard library imports
from collections import defaultdict
import os
import pickle
from typing import Dict, List, Set, Any, Optional
import shutil

# Third-party imports
import pandas as pd

# Local imports
from .config import DataProcessingConfig


class DataProcessor:
    """
    Process raw simulation data into a format suitable for machine learning.
    
    This class handles the processing of raw simulation data, including:
    - Feature extraction
    - Graph construction
    - Data transformation
    """
    
    def __init__(self, train_data_dir: str, save_raw_dir: str, 
                 config: Optional[DataProcessingConfig] = None):
        """Initialize the DataProcessor.
        
        Args:
            train_data_dir: Directory containing training data
            save_raw_dir: Directory to save raw processed data
            config: Configuration for data processing
        """
        self.config = config or DataProcessingConfig()
        
        self.train_data_dir = train_data_dir
        self.save_raw_dir = save_raw_dir
        self.n_requests = 0  # Will be set during processing
    
    def process_data(self, scenario_name: str) -> List[Dict]:
        """Process raw data and generate feature-rich dataset.
        
        Args:
            scenario_name: Scenario name
            
        Returns:
            List of dictionaries containing processed data for each timestep
        """
        all_data = self._process_timesteps()
        processed_dir = self._prepare_process_directory(scenario_name)
        self._save_feature_data(processed_dir, all_data)
        node_mapping = self._create_node_mapping(all_data)
        self._save_graph_data(all_data, processed_dir, node_mapping)
        all_data = self.load_processed_data(processed_dir)
        return all_data
    
    def _process_timesteps(self) -> List[Dict]:
        """Process data for each timestep."""
        all_data = []
        failed_timesteps = []
        
        for timestep in range(0, self.config.sim_duration, 
                                 self.config.sim_step):
            try:
                data = self._load_timestep_data(timestep)
                data = self._add_graph_features(data)
                # self._save_raw_data(timestep, data)
                all_data.append(data)
            except Exception as e:
                print(f'Error processing timestep {timestep}: {e}')
                failed_timesteps.append(timestep)
        
        if failed_timesteps:
            print(f'Failed timesteps: {failed_timesteps}')
        
        return all_data
    
    def _load_timestep_data(self, timestep: int) -> Dict:
        """Load data for a specific timestep."""
        timestep_dir = os.path.join(self.train_data_dir, str(timestep))
        if not os.path.exists(timestep_dir):
            raise FileNotFoundError(f"No data for timestep {timestep}")
        
        data = {}
        for file in os.scandir(timestep_dir):
            if file.name.endswith('.pkl'):
                name = file.name.split('.')[0]
                with open(file.path, 'rb') as f:
                    data[name] = pickle.load(f)
        
        if not data:
            raise FileNotFoundError(f"No valid data for timestep {timestep}")
        return data
    
    def _save_raw_data(self, timestep: int, data: Dict) -> None:
        """Save processed data for a timestep."""
        save_path = os.path.join(self.save_raw_dir, f"{timestep}.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _add_graph_features(self, data: Dict) -> Dict:
        """Add graph features to the data."""
        self.n_requests = len(data['req_features'])
        
        # Add basic node features
        in_degrees, out_degrees = self._calculate_node_degrees(data)
        self._add_degree_features(data, in_degrees, out_degrees)
        
        # Add neighborhood features
        neighborhoods = self._get_node_neighborhoods(data)
        self._add_neighborhood_features(data, neighborhoods)
        self._add_clustering_features(data, neighborhoods)
        
        # Add assignment features
        self._add_assignment_features(data)
        
        return data
    
    def _calculate_node_degrees(self, data: Dict) -> tuple[Dict[int, int], Dict[int, int]]:
        """Calculate in and out degrees for all nodes."""
        in_degrees = defaultdict(int)
        out_degrees = defaultdict(int)
        
        # Process request-request and vehicle-request graphs
        for node_type, graph in [('request', data['rr_graph']), 
                               ('vehicle', data['vr_graph'])]:
            for source_id, targets in graph.items():
                node_fn = self._get_node_index_fn(node_type)
                source_node = node_fn(source_id)
                out_degrees[source_node] += len(targets)
                
                for target_id in targets:
                    in_degrees[self._node_rid(target_id)] += 1
        
        return in_degrees, out_degrees
    
    def _add_degree_features(self, data, in_degrees, out_degrees):
        """Add degree features to node attributes."""
        for node_type, features in [('request', data['req_features']), ('vehicle', data['veh_features'])]:
            node_fn = self._get_node_index_fn(node_type)
            for node_id, feats in features.items():
                node = node_fn(node_id)
                feats['in_degree'] = in_degrees[node]
                feats['out_degree'] = out_degrees[node]

    
    def _get_node_neighborhoods(self, data: Dict) -> Dict[int, Set[int]]:
        """Get neighborhoods for all nodes."""
        neighborhoods = defaultdict(set)
        
        for node_type, graph in [('request', data['rr_graph']), 
                               ('vehicle', data['vr_graph'])]:
            node_fn = self._get_node_index_fn(node_type)
            for source_id, targets in graph.items():
                source_node = node_fn(source_id)
                neighborhoods[source_node].update(
                    self._node_rid(rid) for rid in targets.keys()
                )
        
        return neighborhoods
    
    def _add_neighborhood_features(self, data: Dict, 
                                 neighborhoods: Dict[int, Set[int]]) -> None:
        """Add neighborhood-based features to edges."""
        for node_type, graph in [('request', data['rr_graph']), 
                               ('vehicle', data['vr_graph'])]:
            node_fn = self._get_node_index_fn(node_type)
            for source_id, targets in graph.items():
                source_node = node_fn(source_id)
                for target_id, features in targets.items():
                    target_node = self._node_rid(target_id)
                    self._compute_neighborhood_features(
                        source_node, target_node, 
                        neighborhoods, features
                    )
        
    def _compute_neighborhood_features(self, source_node: int, target_node: int,
                                     neighborhoods: Dict[int, Set[int]], 
                                     features: Dict) -> None:
        """Compute neighborhood-based features for an edge."""
        source_nghb = neighborhoods[source_node]
        target_nghb = neighborhoods[target_node]
        
        common = source_nghb.intersection(target_nghb)
        union = source_nghb.union(target_nghb)
        
        features['common_nghbs'] = len(common)
        features['jaccards_coeff'] = len(common) / len(union) if union else 0.0
    
    def _add_clustering_features(self, data: Dict, 
                               neighborhoods: Dict[int, Set[int]]) -> None:
        """Add clustering coefficient features to nodes."""
        for node_type, features in [('request', data['req_features']), 
                                  ('vehicle', data['veh_features'])]:
            node_fn = self._get_node_index_fn(node_type)
            for node_id, node_features in features.items():
                node = node_fn(node_id)
                self._compute_clustering_coefficient(
                    node, node_features, neighborhoods
                )
    
    def _compute_clustering_coefficient(self, node: int, features: Dict,
                                     neighborhoods: Dict[int, Set[int]]) -> None:
        """Compute clustering coefficient for a node."""
        neighbors = neighborhoods[node]
        n_neighbors = len(neighbors)
        
        if n_neighbors < 2:
            features['clustering_coeff'] = 0.0
            return
        
        links = sum(len(neighborhoods[n].intersection(neighbors)) 
                   for n in neighbors)
        features['clustering_coeff'] = links / (n_neighbors * (n_neighbors - 1))
    
    def _add_assignment_features(self, data: Dict) -> None:
        """Add assignment labels to edges."""
        self._add_assignment_sequence(data, data['init_assignments'], 'init_assign')
        self._add_assignment_sequence(data, data['assignments'], 'opt_assign')
    
    def _add_assignment_sequence(self, data: Dict, assignments: Dict, 
                               feature_name: str) -> None:
        """Add assignment sequence labels to edges."""
        for vehicle_id, sequence in assignments.items():
            if not sequence or len(sequence) < 2:
                continue
            
            # Add vehicle-request edge label
            data['vr_graph'][vehicle_id][sequence[1]][feature_name] = 1
            
            # Add request-request edge labels
            for req1, req2 in zip(sequence[1:], sequence[2:]):
                data['rr_graph'][req1][req2][feature_name] = 1
    
    def _prepare_process_directory(self, scenario_name: str) -> str:
        """Prepare directory for processed data."""
        process_dir = os.path.join(self.config.base_data_dir, scenario_name, self.config.processed_dir)
        if os.path.exists(process_dir):
            shutil.rmtree(process_dir)
        os.makedirs(process_dir)
        return process_dir
    
    def _save_feature_data(self, process_dir: str, all_data: List[Dict]) -> None:
        """Save processed feature data."""
        for feature_type in ['req_features', 'veh_features']:
            dfs = []
            for timestep, data in enumerate(all_data):
                df = pd.DataFrame.from_dict(data[feature_type], orient='index')
                df['timestep'] = timestep
                dfs.append(df)
            pd.concat(dfs).to_csv(
                os.path.join(process_dir, f'{feature_type}.csv'),
                index_label='id'
            )
    
    def _create_node_mapping(self, all_data: List[Dict]) -> Dict[int, Dict[Any, int]]:
        """Create mapping between node IDs and indices."""
        return {
            timestep: {rid: idx for idx, rid in enumerate(data['req_features'].keys())}
            for timestep, data in enumerate(all_data)
        }
    
    def _save_graph_data(self, all_data: List[Dict], process_dir: str,
                        node_mapping: Dict[int, Dict[Any, int]]) -> None:
        """Save graph structure data."""
        for graph_type in ['rr_graph', 'vr_graph']:
            dfs = []
            for timestep, data in enumerate(all_data):
                df = [
                    self._get_edge_attributes(timestep, node_mapping, graph_type,
                                           node, neighbor, info)
                    for node, neighbors in data[graph_type].items()
                    for neighbor, info in neighbors.items()
                ]
                df = pd.DataFrame(df)
                df['timestep'] = timestep
                dfs.append(df)
            pd.concat(dfs).to_csv(
                os.path.join(process_dir, f'{graph_type}.csv'),
                index=False
            )
    
    def _get_edge_attributes(self, timestep: int, node_mapping: Dict[int, Dict[Any, int]],
                           graph_type: str, source: Any, target: Any, 
                           info: Dict) -> Dict:
        """Get attributes for a graph edge."""
        attrs = {
            'source': node_mapping[timestep][source] if graph_type == 'rr_graph' else source,
            'target': node_mapping[timestep][target]
        }
        
        # Add travel time information
        if isinstance(info['travel_cost'], list):
            for i, cost in enumerate(info['travel_cost']):
                attrs[f'travel_time{i}'] = cost['travel_time']
        else:
            attrs['travel_time'] = info['travel_time']
        
        # Add topological features
        attrs['common_nghbs'] = info['common_nghbs']
        attrs['jaccards_coeff'] = info['jaccards_coeff']
        
        # Add labels
        attrs['init_label'] = info.get('init_assign', 0)
        attrs['label'] = info.get('opt_assign', 0)
        
        return attrs
    
    def _get_node_index_fn(self, node_type: str):
        """Get the appropriate node index function for a node type."""
        return self._node_vid if node_type == 'vehicle' else self._node_rid
    
    def _node_rid(self, rid: Any) -> int:
        """Get node index for a request ID."""
        return rid
    
    def _node_vid(self, vid: Any) -> int:
        """Get node index for a vehicle ID."""
        return vid + self.n_requests

    @staticmethod
    def load_processed_data(data_dir):
        data = {}
        for file in os.scandir(data_dir):
            if not file.name.endswith('.csv'):
                continue
            with open(file, 'r') as f:
                file_name = file.name[:file.name.find('.')]
                data[file_name] = pd.read_csv(f)
        return data

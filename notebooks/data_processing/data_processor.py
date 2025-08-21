# Standard library imports
from collections import defaultdict
import os
import pickle
from typing import Dict, List, Set, Any, Optional, Tuple
import shutil
import math

# Third-party imports
import pandas as pd
import networkx as nx

# Local imports
from .config import DataProcessingConfig as cfg


class DataProcessor:
    """
    Process raw simulation data into a format suitable for machine learning.

    This class handles the processing of raw simulation data, including:
    - Feature extraction
    - Graph construction
    - Data transformation
    """

    def __init__(self, train_data_dir: str, save_raw_dir: str,
                 config: Optional[cfg] = None, prefer_processed: bool = True):
        """Initialize the DataProcessor.

        Args:
            train_data_dir: Directory containing training data
            save_raw_dir: Directory to save raw processed data
            config: Configuration for data processing
            prefer_processed: If True, load processed data if it exists (default). If False, always regenerate.
        """
        self.config = config or cfg()
        self.train_data_dir = train_data_dir
        self.save_raw_dir = save_raw_dir
        self.timestep_n_requests = 0  # Will be set during processing
        self.prefer_processed = prefer_processed

    def process_data(self, scenario_name: str) -> dict[Any, Any]:
        """Process raw data and generate feature-rich dataset.

        Args:
            scenario_name: Scenario name
        Returns:
            List of dictionaries containing processed data for each timestep
        """
        print("\n=== DataProcessor.process_data() ===")
        print(f"Processing scenario: {scenario_name}")

        processed_dir = self._prepare_process_directory(scenario_name)
        print(f"Processed directory: {processed_dir}")

        # Try to load existing processed data if preferred
        if self.prefer_processed and os.path.exists(processed_dir):
            print(f"Loading processed data from {processed_dir}")
            try:
                data = self.load_processed_data(processed_dir)
                print("Successfully loaded processed data")
                print(
                    f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")
                return data
            except Exception as e:
                print(f"Error loading processed data: {str(e)}")
                import traceback
                traceback.print_exc()
                # If loading fails, we'll recreate the directory below

        # Process new data
        print("Processing new data...")
        all_data = self._process_timesteps()

        # Clean directory before saving new data
        if os.path.exists(processed_dir):
            print("Cleaning existing processed directory...")
            shutil.rmtree(processed_dir)
        os.makedirs(processed_dir)

        node_mapping = self._create_node_mapping(all_data)
        
        # Save raw features and graph data
        self._save_feature_data(processed_dir, all_data)
        self._save_graph_data(all_data, processed_dir, node_mapping)

        # Load and return the processed data
        all_data = self.load_processed_data(processed_dir)
        return all_data

    def _process_timesteps(self) -> List[Dict]:
        """Process data for each timestep."""
        print("\n=== Processing timesteps ===")
        all_data = []
        failed_timesteps = []

        total_steps = self.config.sim_duration // self.config.sim_step
        print(f"Processing {total_steps} timesteps...")
        print(f"Simulation duration: {self.config.sim_duration} seconds")
        print(f"Simulation step: {self.config.sim_step} seconds")

        for timestep in range(0, self.config.sim_duration,
                              self.config.sim_step):
            print(f"Processing timestep {timestep}/{self.config.sim_duration}")
            try:
                data = self._load_timestep_data(timestep)

                print("Adding graph features...")
                data = self._add_graph_features(timestep, data)

                # Validate processed data
                required_features = [
                    cfg.VEHICLE_REQUEST_GRAPH, cfg.REQUEST_REQUEST_GRAPH, cfg.VEHICLE_FEATURES, cfg.REQUEST_FEATURES]
                missing_features = [
                    f for f in required_features if f not in data]
                if missing_features:
                    raise ValueError(
                        f"Missing required features: {missing_features}")

                print(f"\nProcessed data structure for timestep {timestep}:")
                for key in required_features:
                    feature_dict = data[key]
                    print(f"{key}:")
                    print(f"Number of items: {len(feature_dict)}")
                    if feature_dict:
                        sample_item = next(iter(feature_dict.items()))
                        print(
                            f"Sample features: {list(sample_item[1].keys())}")

                all_data.append(data)
                print(f"Successfully processed timestep {timestep}")
            except Exception as e:
                print(f'Error processing timestep {timestep}: {str(e)}')
                import traceback
                print("Full error traceback:")
                traceback.print_exc()
                failed_timesteps.append(timestep)

        if failed_timesteps:
            print(f'Failed timesteps: {failed_timesteps}')

        return all_data

    def _load_timestep_data(self, timestep: int) -> Dict:
        """Load data for a specific timestep."""
        print(f"\n=== Loading timestep data for timestep {timestep} ===")
        timestep_dir = os.path.join(self.train_data_dir, str(timestep))
        print(f"Looking for data in: {timestep_dir}")

        if not os.path.exists(timestep_dir):
            print(f"ERROR: Directory not found: {timestep_dir}")
            raise FileNotFoundError(f"No data for timestep {timestep}")

        print("\nAvailable files in timestep directory:")
        data = {}
        for file in os.scandir(timestep_dir):
            print(f"- {file.name}")
            if file.name.endswith('.pkl'):
                name = file.name.split('.')[0]
                try:
                    with open(file.path, 'rb') as f:
                        data[name] = pickle.load(f)
                    print(f"Successfully loaded {name}")
                except Exception as e:
                    print(f"Error loading {file.name}: {str(e)}")

        if not data:
            print("No data was loaded from any files")
            raise FileNotFoundError(f"No valid data for timestep {timestep}")

        print(f"\nLoaded data keys: {list(data.keys())}")
        return data

    def _add_graph_features(self, timestep: int, data: Dict) -> Dict:
        """Add graph features to the data, including NetworkX-based features."""
        print("\n=== Adding graph features ===")
        print(f"Initial data keys: {list(data.keys())}")

        # Check if request_features exists
        if cfg.REQUEST_FEATURES not in data:
            print(f"ERROR: '{cfg.REQUEST_FEATURES}' not found in data!")
            print("Available keys and their types:")
            for key, value in data.items():
                print(f"- {key}: {type(value)}")
            raise KeyError(
                "Required key cfg.REQUEST_FEATURES not found in data")

        self.timestep_n_requests = len(data[cfg.REQUEST_FEATURES])
        print(f"Number of requests: {self.timestep_n_requests}")

        print("\nAdding domain-specific features...")
        # Add domain-specific features first
        try:
            print("Adding temporal features...")
            self._add_temporal_features(timestep, data)
            print("Adding spatial features...")
            self._add_spatial_features(data)
            print("Adding competition features...")
            self._add_competition_features(data)
        except Exception as e:
            print(f"Error adding domain-specific features: {str(e)}")
            import traceback
            traceback.print_exc()

        # --- NetworkX graph construction (once) ---
        # Request-Request graph
        G_rr = nx.DiGraph()
        for src, targets in data[cfg.REQUEST_REQUEST_GRAPH].items():
            for tgt in targets:
                G_rr.add_edge(src, tgt)

        # Vehicle-Request graph
        G_vr = nx.DiGraph()
        for veh, targets in data[cfg.VEHICLE_REQUEST_GRAPH].items():
            for req in targets:
                G_vr.add_edge(veh, req)

        # Combined graph (created once and reused)
        G_combined = nx.DiGraph()
        G_combined.add_edges_from(G_rr.edges())
        G_combined.add_edges_from(G_vr.edges())

        # Add basic node features using the combined graph
        in_degrees, out_degrees = self._calculate_node_degrees(
            data, G_rr, G_vr, G_combined)
        self._add_degree_features(data, in_degrees, out_degrees)

        # Add topological features using NetworkX
        self._add_neighborhood_features(data, G_combined)
        self._add_clustering_features(data, G_combined)
        self._add_centrality_features(data, G_rr, G_vr, G_combined)

        # Add edge-specific features
        self._add_edge_compatibility_features(timestep, data)
        self._add_assignment_features(data)

        return data

    def _add_centrality_features(self, data: Dict, G_rr, G_vr, G_combined) -> None:
        """Add centrality features using the combined graph.

        Calculate centrality measures on the complete heterogeneous graph structure,
        as this better represents the actual graph that the GNN will process.

        Selected centrality measures:
        - Degree centrality: Direct connectivity in the complete graph
        - Betweenness centrality: Path importance considering all node types
        - PageRank: Global importance in the heterogeneous network
        - Closeness centrality: Proximity to all other nodes in the complete graph
        """
        def calculate_centralities(G):
            """Calculate centrality measures for the combined graph."""
            centralities = {
                'degree': nx.degree_centrality(G),
                'betweenness': nx.betweenness_centrality(G),
                'pagerank': nx.pagerank(G),
            }

            try:
                # Try to calculate closeness centrality
                centralities['closeness'] = nx.closeness_centrality(G)
            except Exception as e:
                print(
                    f"Warning: Could not calculate closeness centrality: {str(e)}")
                centralities['closeness'] = {node: 0.0 for node in G.nodes()}

            return centralities

        # Calculate centralities for each graph
        centralities = {
            'rr': calculate_centralities(G_rr) if len(G_rr) > 0 else None,
            'vr': calculate_centralities(G_vr) if len(G_vr) > 0 else None,
            'combined': calculate_centralities(G_combined) if len(G_combined) > 0 else None
        }

        # Add centrality measures to request features
        for req_id, feats in data[cfg.REQUEST_FEATURES].items():
            # Request-Request graph centralities
            if centralities['rr']:
                feats.update({
                    'nx_rr_degree_centrality': centralities['rr']['degree'].get(req_id, 0.0),
                    'nx_rr_betweenness_centrality': centralities['rr']['betweenness'].get(req_id, 0.0),
                    'nx_rr_closeness_centrality': centralities['rr']['closeness'].get(req_id, 0.0),
                    'nx_rr_pagerank': centralities['rr']['pagerank'].get(req_id, 0.0)
                })

            # Combined graph centralities for requests
            if centralities['combined']:
                feats.update({
                    'nx_combined_degree_centrality': centralities['combined']['degree'].get(req_id, 0.0),
                    'nx_combined_betweenness_centrality': centralities['combined']['betweenness'].get(req_id, 0.0),
                    'nx_combined_closeness_centrality': centralities['combined']['closeness'].get(req_id, 0.0),
                    'nx_combined_pagerank': centralities['combined']['pagerank'].get(req_id, 0.0)
                })

        # Add centrality measures to vehicle features
        for veh_id, feats in data[cfg.VEHICLE_FEATURES].items():
            # Vehicle-Request graph centralities
            if centralities['vr']:
                feats.update({
                    'nx_vr_degree_centrality': centralities['vr']['degree'].get(veh_id, 0.0),
                    'nx_vr_betweenness_centrality': centralities['vr']['betweenness'].get(veh_id, 0.0),
                    'nx_vr_closeness_centrality': centralities['vr']['closeness'].get(veh_id, 0.0),
                    'nx_vr_pagerank': centralities['vr']['pagerank'].get(veh_id, 0.0)
                })

            # Combined graph centralities for vehicles
            if centralities['combined']:
                feats.update({
                    'nx_combined_degree_centrality': centralities['combined']['degree'].get(veh_id, 0.0),
                    'nx_combined_betweenness_centrality': centralities['combined']['betweenness'].get(veh_id, 0.0),
                    'nx_combined_closeness_centrality': centralities['combined']['closeness'].get(veh_id, 0.0),
                    'nx_combined_pagerank': centralities['combined']['pagerank'].get(veh_id, 0.0)
                })

    def _calculate_node_degrees(self, data: Dict, G_rr, G_vr, G_combined) -> tuple[Dict[int, int], Dict[int, int]]:
        """Calculate in and out degrees for all nodes using NetworkX from different graph views."""
        print("\n=== Calculating node degrees ===")
        print(
            f"Total nodes in req_features: {len(data[cfg.REQUEST_FEATURES])}")
        print(
            f"Total nodes in veh_features: {len(data[cfg.VEHICLE_FEATURES])}")
        print(f"Total nodes in combined graph: {len(G_combined.nodes())}")

        # Initialize degree dictionaries
        combined_in_degrees = defaultdict(int)
        combined_out_degrees = defaultdict(int)
        rr_in_degrees = defaultdict(int)     # Request-to-request degrees
        rr_out_degrees = defaultdict(int)
        vr_in_degrees = defaultdict(int)     # Vehicle-to-request degrees
        vr_out_degrees = defaultdict(int)

        # Calculate degrees from request-request graph
        print("\nCalculating request-request degrees...")
        for node in G_rr.nodes():
            rr_in_degrees[node] = G_rr.in_degree(node)
            rr_out_degrees[node] = G_rr.out_degree(node)

        # Calculate degrees from vehicle-request graph
        print("Calculating vehicle-request degrees...")
        for node in G_vr.nodes():
            vr_in_degrees[node] = G_vr.in_degree(node)
            vr_out_degrees[node] = G_vr.out_degree(node)

        # Calculate degrees from combined graph
        print("Calculating combined graph degrees...")
        for node in G_combined.nodes():
            try:
                combined_in_degrees[node] = G_combined.in_degree(node)
                combined_out_degrees[node] = G_combined.out_degree(node)

                # Add graph-specific degrees
                if node in data[cfg.REQUEST_FEATURES]:
                    # For requests, combine RR and VR degrees
                    combined_in_degrees[f"{node}_rr"] = rr_in_degrees[node]
                    combined_out_degrees[f"{node}_rr"] = rr_out_degrees[node]
                    combined_in_degrees[f"{node}_vr"] = vr_in_degrees[node]
                    combined_out_degrees[f"{node}_vr"] = vr_out_degrees[node]
                elif node in data[cfg.VEHICLE_FEATURES]:
                    # For vehicles, only VR degrees are relevant
                    combined_in_degrees[f"{node}_vr"] = vr_in_degrees[node]
                    combined_out_degrees[f"{node}_vr"] = vr_out_degrees[node]

            except Exception as e:
                print(f"Error calculating degrees for node {node}: {str(e)}")
                combined_in_degrees[node] = 0
                combined_out_degrees[node] = 0

        print(f"\nProcessed degrees for {len(combined_in_degrees)} nodes")
        return combined_in_degrees, combined_out_degrees

    def _add_degree_features(self, data, in_degrees, out_degrees):
        """Add degree features to node attributes."""
        print("\n=== Adding degree features ===")

        # Process requests (have both RR and VR degrees)
        print("\nProcessing request nodes...")
        for req_id, feats in data[cfg.REQUEST_FEATURES].items():
            try:
                # Combined graph degrees
                feats['in_degree_total'] = in_degrees[req_id]
                feats['out_degree_total'] = out_degrees[req_id]

                # Request-Request graph degrees
                feats['in_degree_from_requests'] = in_degrees[f"{req_id}_rr"]
                feats['out_degree_to_requests'] = out_degrees[f"{req_id}_rr"]

                # Vehicle-Request graph degrees
                feats['in_degree_from_vehicles'] = in_degrees[f"{req_id}_vr"]
                feats['out_degree_to_vehicles'] = out_degrees[f"{req_id}_vr"]

                # Ratio features
                # Avoid division by zero
                total_in = max(1, feats['in_degree_total'])
                total_out = max(1, feats['out_degree_total'])
                feats['request_vehicle_in_ratio'] = feats['in_degree_from_vehicles'] / total_in
                feats['request_vehicle_out_ratio'] = feats['out_degree_to_vehicles'] / total_out

            except Exception as e:
                print(
                    f"Error adding degree features for request {req_id}: {str(e)}")
                # Initialize all degree features to 0 on error
                for prefix in ['', '_from_requests', '_from_vehicles', '_to_requests', '_to_vehicles']:
                    feats[f'in_degree{prefix}'] = 0
                    feats[f'out_degree{prefix}'] = 0
                feats['request_vehicle_in_ratio'] = 0
                feats['request_vehicle_out_ratio'] = 0

        # Process vehicles (only have VR degrees)
        print("\nProcessing vehicle nodes...")
        for veh_id, feats in data[cfg.VEHICLE_FEATURES].items():
            try:
                # Combined graph degrees (same as VR for vehicles)
                feats['in_degree'] = in_degrees[veh_id]
                feats['out_degree'] = out_degrees[veh_id]

                # Vehicle-Request specific degrees
                feats['in_degree_from_requests'] = in_degrees[f"{veh_id}_vr"]
                feats['out_degree_to_requests'] = out_degrees[f"{veh_id}_vr"]

                # Add total connections
                feats['total_request_connections'] = feats['in_degree_from_requests'] + \
                    feats['out_degree_to_requests']

            except Exception as e:
                print(
                    f"Error adding degree features for vehicle {veh_id}: {str(e)}")
                feats['in_degree'] = 0
                feats['out_degree'] = 0
                feats['in_degree_from_requests'] = 0
                feats['out_degree_to_requests'] = 0
                feats['total_request_connections'] = 0

    def _add_neighborhood_features(self, data: Dict, G_combined) -> None:
        """Add neighborhood-based features to edges using NetworkX for common neighbors and Jaccard coefficient.

        Calculates type-specific neighborhood metrics:
        - For request-request edges: common request neighbors and common vehicle neighbors
        - For vehicle-request edges: common request neighbors and common vehicle neighbors
        """
        def calculate_type_specific_metrics(src, tgt, G, data):
            """Calculate type-specific neighborhood metrics."""
            if not (G.has_node(src) and G.has_node(tgt)):
                return {
                    'common_request_neighbors': 0,
                    'common_vehicle_neighbors': 0,
                    'total_common_neighbors': 0,
                    'request_jaccard': 0.0,
                    'vehicle_jaccard': 0.0,
                    'combined_jaccard': 0.0
                }

            # Get all neighbors
            neighbors_src = set(G.neighbors(src))
            neighbors_tgt = set(G.neighbors(tgt))

            # Split neighbors by type
            src_req_neighbors = {
                n for n in neighbors_src if n in data[cfg.REQUEST_FEATURES]}
            src_veh_neighbors = {
                n for n in neighbors_src if n in data[cfg.VEHICLE_FEATURES]}
            tgt_req_neighbors = {
                n for n in neighbors_tgt if n in data[cfg.REQUEST_FEATURES]}
            tgt_veh_neighbors = {
                n for n in neighbors_tgt if n in data[cfg.VEHICLE_FEATURES]}

            # Calculate common neighbors by type
            common_requests = src_req_neighbors & tgt_req_neighbors
            common_vehicles = src_veh_neighbors & tgt_veh_neighbors

            # Calculate exclusive (uncommon) neighbors by type
            exclusive_src_requests = src_req_neighbors - \
                tgt_req_neighbors  # Only connected to source
            exclusive_tgt_requests = tgt_req_neighbors - \
                src_req_neighbors  # Only connected to target
            exclusive_src_vehicles = src_veh_neighbors - tgt_veh_neighbors
            exclusive_tgt_vehicles = tgt_veh_neighbors - src_veh_neighbors

            # Calculate unions by type
            union_requests = src_req_neighbors | tgt_req_neighbors
            union_vehicles = src_veh_neighbors | tgt_veh_neighbors

            # Calculate total common and union
            total_common = len(common_requests) + len(common_vehicles)
            total_union = len(union_requests) + len(union_vehicles)

            # Calculate total exclusive neighbors
            total_exclusive_src = len(
                exclusive_src_requests) + len(exclusive_src_vehicles)
            total_exclusive_tgt = len(
                exclusive_tgt_requests) + len(exclusive_tgt_vehicles)

            return {
                # Common neighbor metrics
                'common_request_neighbors': len(common_requests),
                'common_vehicle_neighbors': len(common_vehicles),
                'total_common_neighbors': total_common,

                # Exclusive neighbor metrics
                'exclusive_src_requests': len(exclusive_src_requests),
                'exclusive_tgt_requests': len(exclusive_tgt_requests),
                'exclusive_src_vehicles': len(exclusive_src_vehicles),
                'exclusive_tgt_vehicles': len(exclusive_tgt_vehicles),
                'total_exclusive_src': total_exclusive_src,
                'total_exclusive_tgt': total_exclusive_tgt,

                # Jaccard coefficients
                'request_jaccard': len(common_requests) / len(union_requests) if union_requests else 0.0,
                'vehicle_jaccard': len(common_vehicles) / len(union_vehicles) if union_vehicles else 0.0,
                'combined_jaccard': total_common / total_union if total_union else 0.0,

                # Overlap ratios
                'request_overlap_ratio': len(common_requests) / max(1, len(union_requests)),
                'vehicle_overlap_ratio': len(common_vehicles) / max(1, len(union_vehicles)),

                # Exclusivity ratios
                'src_exclusivity_ratio': total_exclusive_src / max(1, len(neighbors_src)),
                'tgt_exclusivity_ratio': total_exclusive_tgt / max(1, len(neighbors_tgt)),

                # Competition metrics
                'request_competition_index': (
                    len(exclusive_src_requests) + len(exclusive_tgt_requests)
                ) / max(1, len(union_requests)),
                'vehicle_competition_index': (
                    len(exclusive_src_vehicles) + len(exclusive_tgt_vehicles)
                ) / max(1, len(union_vehicles)),

                # Service area overlap
                'service_area_overlap': total_common / max(1, total_common + total_exclusive_src + total_exclusive_tgt)
            }

        # Calculate metrics for request-request edges
        print("\nProcessing request-request edges...")
        for src, targets in data[cfg.REQUEST_REQUEST_GRAPH].items():
            for tgt, features in targets.items():
                metrics = calculate_type_specific_metrics(
                    src, tgt, G_combined, data)
                features.update(metrics)
                # Add edge-specific ratios
                features['request_to_vehicle_neighbor_ratio'] = (
                    metrics['common_request_neighbors'] /
                    max(1, metrics['common_vehicle_neighbors'])
                )

        # Calculate metrics for vehicle-request edges
        print("\nProcessing vehicle-request edges...")
        for veh, targets in data[cfg.VEHICLE_REQUEST_GRAPH].items():
            for req, features in targets.items():
                metrics = calculate_type_specific_metrics(
                    veh, req, G_combined, data)
                features.update(metrics)
                # Add competition metrics
                features['vehicle_competition'] = metrics['common_vehicle_neighbors']
                features['request_competition'] = metrics['common_request_neighbors']
                features['competition_ratio'] = (
                    metrics['common_vehicle_neighbors'] /
                    max(0.001, metrics['common_request_neighbors'])
                )

    def _add_clustering_features(self, data: Dict, G_combined) -> None:
        """Add clustering coefficient features to nodes using the combined graph.

        Calculate clustering coefficients on the complete heterogeneous graph,
        which represents how well-connected a node's neighbors are in the context
        of the entire system (both vehicles and requests).
        """
        # Convert to undirected for clustering calculations
        G_combined_undir = G_combined.to_undirected()

        # Calculate clustering coefficients for the combined graph
        clustering_combined = nx.clustering(G_combined_undir)

        # Add to all node features
        for req_id, feats in data[cfg.REQUEST_FEATURES].items():
            feats['clustering_coeff'] = clustering_combined.get(req_id, 0.0)

        for veh_id, feats in data[cfg.VEHICLE_FEATURES].items():
            feats['clustering_coeff'] = clustering_combined.get(veh_id, 0.0)

    def _add_temporal_features(self, current_time: int, data: Dict) -> None:
        """Add temporal features that capture time-based relationships."""
        # For requests
        for req_id, feats in data[cfg.REQUEST_FEATURES].items():
            # Time urgency features
            time_until_earliest = feats['tw_pe'] - current_time
            time_until_latest = feats['tw_pl'] - current_time
            time_window_width = feats['tw_pl'] - feats['tw_pe']

            feats.update({
                # How soon can we serve
                'time_until_earliest': max(0, time_until_earliest),
                # How urgent is it
                'time_until_latest': max(0, time_until_latest),
                # How flexible is it
                'time_window_width': time_window_width,              
                # Higher for more urgent requests
                'time_window_urgency': 1.0 / max(1, time_until_latest),
                # How long has it been waiting
                'request_age': current_time - feats['rq_time']
            })

    def _add_spatial_features(self, data: Dict) -> None:
        """Add spatial features that capture geographical relationships."""

        # For requests
        for req_id, feats in data[cfg.REQUEST_FEATURES].items():
            # Calculate spatial features
            manhattan_distance = abs(
                feats['d_lat'] - feats['o_lat']) + abs(feats['d_lon'] - feats['o_lon'])
            # Ratio of actual route distance to manhattan distance indicates route complexity
            route_directness = feats['direct_td'] / \
                max(0.001, manhattan_distance)

            feats.update({
                'manhattan_distance': manhattan_distance,
                'route_directness': route_directness,
                'origin_dest_bearing': self._calculate_bearing(
                    feats['o_lat'], feats['o_lon'],
                    feats['d_lat'], feats['d_lon']
                )
            })

    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the bearing (angle) between two points."""
        # Convert to radians
        lat1, lon1 = math.radians(lat1), math.radians(lon1)
        lat2, lon2 = math.radians(lat2), math.radians(lon2)

        # Calculate bearing
        x = math.cos(lat2) * math.sin(lon2 - lon1)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * \
            math.cos(lat2) * math.cos(lon2 - lon1)
        bearing = math.atan2(x, y)

        # Convert to degrees
        return math.degrees(bearing)

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points in kilometers."""
        R = 6371  # Earth's radius in kilometers

        # Convert to radians
        lat1, lon1 = math.radians(lat1), math.radians(lon1)
        lat2, lon2 = math.radians(lat2), math.radians(lon2)

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * \
            math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def _add_competition_features(self, data: Dict) -> None:
        """Add features that capture competition between requests and vehicle availability."""
        print("\n=== Adding competition features ===")

        # Calculate request density in different regions
        request_locations = {}
        for req_id, feats in data[cfg.REQUEST_FEATURES].items():
            if 'o_lat' in feats and 'o_lon' in feats:
                request_locations[req_id] = (feats['o_lat'], feats['o_lon'])
            else:
                print(f"Warning: Missing lat/lon for request {req_id}")
                print(f"Available features: {list(feats.keys())}")
                continue

        print(f"Found {len(request_locations)} requests with valid locations")

        # For each request, count nearby requests and vehicles
        for req_id, feats in data[cfg.REQUEST_FEATURES].items():
            if req_id not in request_locations:
                continue

            nearby_requests = 0
            req_loc = request_locations[req_id]

            for other_id, other_loc in request_locations.items():
                if other_id != req_id:
                    dist = self._haversine_distance(
                        req_loc[0], req_loc[1],
                        other_loc[0], other_loc[1]
                    )
                    if dist < 1.0:  # within 1km
                        nearby_requests += 1

            # Count available vehicles nearby
            nearby_vehicles = 0
            for veh_id, veh_feats in data[cfg.VEHICLE_FEATURES].items():
                if 'lat' in veh_feats and 'lon' in veh_feats:  # if vehicle location available
                    dist = self._haversine_distance(
                        req_loc[0], req_loc[1],
                        veh_feats['lat'], veh_feats['lon']
                    )
                    if dist < 1.0:  # within 1km
                        nearby_vehicles += 1

            feats.update({
                'nearby_requests': nearby_requests,
                'nearby_vehicles': nearby_vehicles,
                'demand_supply_ratio': nearby_requests / max(1, nearby_vehicles)
            })

    def _is_peak_hour(self, time_of_day: float) -> bool:
        """Check if a given time is during peak hours."""
        # Convert time to hours (assuming time is in minutes)
        hour = (time_of_day / 60.0) % 24
        # Morning peak: 7-10 AM, Evening peak: 4-7 PM
        return (7 <= hour <= 10) or (16 <= hour <= 19)

    def _calculate_pooling_metrics(self, req1_feats: Dict, req2_feats: Dict) -> Dict:
        """Calculate metrics related to request pooling compatibility."""
        # Calculate spatial overlap
        req1_origin = (req1_feats['o_lat'], req1_feats['o_lon'])
        req1_dest = (req1_feats['d_lat'], req1_feats['d_lon'])
        req2_origin = (req2_feats['o_lat'], req2_feats['o_lon'])
        req2_dest = (req2_feats['d_lat'], req2_feats['d_lon'])

        # Distance between origins and destinations
        origin_distance = self._haversine_distance(*req1_origin, *req2_origin)
        dest_distance = self._haversine_distance(*req1_dest, *req2_dest)

        # Calculate potential detour
        direct_dist1 = req1_feats['direct_td']
        direct_dist2 = req2_feats['direct_td']
        pooled_distance = (
            # First pickup to second pickup
            self._haversine_distance(*req1_origin, *req2_origin) +
            # Second pickup to its destination
            self._haversine_distance(*req2_origin, *req2_dest) +
            # Second destination to first destination
            self._haversine_distance(*req2_dest, *req1_dest)
        )

        max_detour_ratio = max(
            pooled_distance / max(0.1, direct_dist1),
            pooled_distance / max(0.1, direct_dist2)
        )

        return {
            # Higher when origins are closer
            'origin_proximity': 1.0 / max(0.1, origin_distance),
            # Higher when destinations are closer
            'destination_proximity': 1.0 / max(0.1, dest_distance),
            'pooling_detour_ratio': max_detour_ratio,
            # Normalized spatial compatibility
            'spatial_compatibility': 1.0 / (1.0 + origin_distance + dest_distance),
        }

    def _add_edge_compatibility_features(self, current_time: int, data: Dict) -> None:
        """Add features that indicate compatibility between nodes."""

        # For vehicle-request edges
        for veh_id, targets in data[cfg.VEHICLE_REQUEST_GRAPH].items():
            veh_feats = data[cfg.VEHICLE_FEATURES][veh_id]

            for req_id, edge_feats in targets.items():
                req_feats = data[cfg.REQUEST_FEATURES][req_id]

                # Add time-of-day features
                time_of_day = current_time % (
                    24 * 60)  # Minutes within the day
                hour_of_day = time_of_day / 60.0
                is_peak = self._is_peak_hour(time_of_day)

                edge_feats.update({
                    'is_peak_hour': float(is_peak),
                    # Cyclical encoding
                    'hour_of_day_sin': math.sin(2 * math.pi * hour_of_day / 24),
                    'hour_of_day_cos': math.cos(2 * math.pi * hour_of_day / 24),
                })

                # Use travel time information from edge features
                travel_time = edge_feats.get('travel_cost', {})

                # Calculate earliest arrival as current time + travel time to request
                earliest_arrival = current_time + travel_time
                # Add to vehicle features
                veh_feats['earliest_arrival'] = earliest_arrival

                # Time compatibility and earliest arrival features
                earliest_arrival = veh_feats['earliest_arrival']
                earliest_pickup = req_feats['tw_pe']
                latest_pickup = req_feats['tw_pl']

                # Calculate various temporal margins
                arrival_slack = latest_pickup - earliest_arrival
                earliest_slack = earliest_pickup - earliest_arrival

                # Consider lock status in compatibility
                is_locked = req_feats.get('locked', False)

                edge_feats.update({
                    # Lock status feature
                    # Convert to int (0 or 1)
                    'is_locked': float(is_locked),

                    # Temporal compatibility metrics (adjusted for locks)
                    'arrival_slack': max(0, arrival_slack) if not is_locked else 0,
                    # Can be negative if vehicle arrives before earliest pickup
                    'earliest_slack': earliest_slack if not is_locked else 0,
                    'time_window_compatibility': min(1.0, max(0, arrival_slack) / max(1, latest_pickup - earliest_pickup)) if not is_locked else 0,
                    'time_feasibility_score': 1.0 if arrival_slack > 0 and not is_locked else 0.0,
                    'normalized_arrival_time': (earliest_arrival - earliest_pickup) / max(1, latest_pickup - earliest_pickup),

                    # Vehicle suitability
                    'distance_to_vehicle': self._haversine_distance(
                        veh_feats['lat'], veh_feats['lon'],
                        req_feats['o_lat'], req_feats['o_lon']
                    )
                })

        # For request-request edges
        for req1_id, targets in data[cfg.REQUEST_REQUEST_GRAPH].items():
            req1_feats = data[cfg.REQUEST_FEATURES][req1_id]

            for req2_id, edge_feats in targets.items():
                req2_feats = data[cfg.REQUEST_FEATURES][req2_id]

                # Get travel times from edge features
                travel_costs = edge_feats.get('travel_cost', [])
                if isinstance(travel_costs, list) and len(travel_costs) >= 6:
                    # Extract travel times for all combinations
                    tt_o1_d1 = travel_costs[0].get(
                        'travel_time', 0)  # origin1 -> dest1
                    tt_o1_o2 = travel_costs[1].get(
                        'travel_time', 0)  # origin1 -> origin2
                    tt_o1_d2 = travel_costs[2].get(
                        'travel_time', 0)  # origin1 -> dest2
                    tt_d1_o2 = travel_costs[3].get(
                        'travel_time', 0)  # dest1 -> origin2
                    tt_o2_d2 = travel_costs[4].get(
                        'travel_time', 0)  # origin2 -> dest2
                    tt_d1_d2 = travel_costs[5].get(
                        'travel_time', 0)  # dest1 -> dest2

                    # Calculate temporal metrics using actual travel times
                    time_gap = req2_feats['tw_pe'] - \
                        (req1_feats['tw_pl'] + tt_o1_d1)
                    total_direct_time = tt_o1_d1 + tt_o2_d2

                    # Calculate different possible shared ride sequences
                    shared_ride_time_1 = tt_o1_o2 + tt_o2_d2 + \
                        tt_d1_d2  # Pick both, drop second, drop first
                    shared_ride_time_2 = tt_o1_d1 + tt_d1_o2 + \
                        tt_o2_d2  # Pick&drop first, then second
                    # shared_ride_time_3 = tt_o1_o2 + tt_o2_d1 + tt_d1_d2
                    #       # Pick both, drop first TODO add

                    # Add detailed travel time features
                    edge_feats.update({
                        'tt_between_pickups': tt_o1_o2,      # Time between pickup locations
                        'tt_between_dropoffs': tt_d1_d2,     # Time between dropoff locations
                        'tt_pickup1_to_dropoff2': tt_o1_d2,  # Time from first pickup to second dropoff
                        'tt_dropoff1_to_pickup2': tt_d1_o2,  # Time from first dropoff to second pickup
                        'min_shared_ride_time': min(shared_ride_time_1, shared_ride_time_2),
                    })

                    # Use the better sequence
                    shared_ride_time = min(
                        shared_ride_time_1, shared_ride_time_2)
                else:
                    # Fallback to previous calculation if travel times not available
                    time_gap = req2_feats['tw_pe'] - \
                        (req1_feats['tw_pl'] + req1_feats['direct_tt'])
                    total_direct_time = req1_feats['direct_tt'] + \
                        req2_feats['direct_tt']
                    shared_ride_time = (
                        req1_feats['direct_tt'] +
                        self._haversine_distance(
                            req1_feats['d_lat'], req1_feats['d_lon'],
                            req2_feats['d_lat'], req2_feats['d_lon']
                        ) * 60/30  # Assuming 30 km/h average speed
                    )

                # Calculate pooling compatibility metrics
                pooling_metrics = self._calculate_pooling_metrics(
                    req1_feats, req2_feats)

                extra_time_ratio = max(
                    0, (shared_ride_time - total_direct_time) / max(1, total_direct_time))

                # Check if either request is locked
                is_req1_locked = req1_feats.get('locked', False)
                is_req2_locked = req2_feats.get('locked', False)
                both_locked = is_req1_locked and is_req2_locked

                edge_feats.update({
                    # Lock status features
                    'src_locked': is_req1_locked,    # Already 0 or 1
                    'tgt_locked': is_req2_locked,    # Already 0 or 1
                    'both_locked': int(both_locked),  # Convert boolean to 0/1

                    # Basic temporal compatibility (adjusted for locks)
                    'time_gap': time_gap,
                    'temporal_compatibility': 1.0 if time_gap > 0 else 0.0,

                    # Pooling compatibility metrics
                    'origin_proximity': pooling_metrics['origin_proximity'],
                    'destination_proximity': pooling_metrics['destination_proximity'],
                    'pooling_detour_ratio': pooling_metrics['pooling_detour_ratio'],
                    'spatial_compatibility': pooling_metrics['spatial_compatibility'],

                    # Ride-sharing metrics
                    # Additional waiting time for second request
                    'extra_waiting_time': max(0, time_gap),
                    'extra_time_ratio': extra_time_ratio,  # Ratio of extra time due to sharing
                    # Higher when detour is smaller
                    'ride_sharing_efficiency': 1.0 / (1.0 + extra_time_ratio),

                    # Combined compatibility score
                    'overall_pooling_score': (
                        pooling_metrics['spatial_compatibility'] *
                        (1.0 if time_gap > 0 else 0.0) *
                        (1.0 / (1.0 + extra_time_ratio))
                    )
                })

    def _add_assignment_features(self, data: Dict) -> None:
        """Add assignment labels to edges."""
        self._add_assignment_sequence(
            data, data['init_assignments'], 'init_assign', complete_graph=True)
        self._add_assignment_sequence(data, data['assignments'], 'opt_assign', complete_graph=True)

    def _add_assignment_sequence(self, data: Dict, assignments: Dict,
                                 feature_name: str, complete_graph: bool = True) -> None:
        """Add assignment sequence labels to edges.
        
        Args:
            data: Dictionary containing graph data
            assignments: Dictionary of vehicle_id -> sequence of request_ids
            feature_name: Name of the feature to set (e.g., 'init_assign' or 'opt_assign')
            complete_graph: If True, add edges between all pairs of requests in sequence (complete graph),
                          not just consecutive requests
        """
        for vehicle_id, sequence in assignments.items():
            if not sequence or len(sequence) < 2:
                continue

            # Filter out requests that don't have edges with the vehicle
            valid_sequence = []
            for req_id in sequence[1:]:  # Skip the first item (typically vehicle's initial position)
                if vehicle_id not in data[cfg.VEHICLE_REQUEST_GRAPH]:
                    print(f"Warning: Vehicle {vehicle_id} not found in graph")
                    continue
                if req_id not in data[cfg.VEHICLE_REQUEST_GRAPH][vehicle_id]:
                    print(f"Warning: Edge between vehicle {vehicle_id} and request {req_id} not found in graph")
                    continue
                valid_sequence.append(req_id)
                
            if not valid_sequence:
                continue  # Skip this vehicle if no valid requests
                
            # Add vehicle-to-request edges first
            for req_id in valid_sequence:
                try:
                    data[cfg.VEHICLE_REQUEST_GRAPH][vehicle_id][req_id][feature_name] = 1
                except KeyError as e:
                    print(f"Warning: Could not add V-R edge label for vehicle {vehicle_id} and request {req_id}: {str(e)}")
            
            # Now add request-to-request edges
            if len(valid_sequence) >= 2:
                if complete_graph:
                    # Add edges between all pairs of requests (complete graph)
                    for i, req1 in enumerate(valid_sequence):
                        for req2 in valid_sequence[i+1:]:  # All requests after req1
                            try:
                                if req1 in data[cfg.REQUEST_REQUEST_GRAPH] and req2 in data[cfg.REQUEST_REQUEST_GRAPH][req1]:
                                    data[cfg.REQUEST_REQUEST_GRAPH][req1][req2][feature_name] = 1
                            except KeyError as e:
                                print(f"Warning: Could not add R-R edge label between requests {req1} and {req2}: {str(e)}")
                else:
                    # Original behavior: Add edges only between consecutive requests
                    for req1, req2 in zip(valid_sequence[:-1], valid_sequence[1:]):
                        try:
                            if req1 in data[cfg.REQUEST_REQUEST_GRAPH] and req2 in data[cfg.REQUEST_REQUEST_GRAPH][req1]:
                                data[cfg.REQUEST_REQUEST_GRAPH][req1][req2][feature_name] = 1
                        except KeyError as e:
                            print(f"Warning: Could not add R-R edge label between requests {req1} and {req2}: {str(e)}")
                            continue

    def _prepare_process_directory(self, scenario_name: str) -> str:
        """Get the path to the processed data directory for a scenario.

        Args:
            scenario_name: Name of the scenario
        Returns:
            Path to the processed data directory
        """
        return os.path.join(
            self.config.base_data_dir, scenario_name, self.config.processed_dir)

    def _save_feature_data(self, process_dir: str, all_data: List[Dict]) -> None:
        """Save processed feature data without normalization.
        
        Args:
            process_dir: Directory to save processed data
            all_data: List of data dictionaries for a single scenario
        """
        for feature_type in [cfg.REQUEST_FEATURES, cfg.VEHICLE_FEATURES]:
            dfs = []
            total_samples = 0
            for timestep, data in enumerate(all_data):
                if data[feature_type]:
                    df = pd.DataFrame.from_dict(
                        data[feature_type], orient='index')
                    df['timestep'] = timestep
                    dfs.append(df)
                    total_samples += len(df)

            if dfs:
                print(f"\nProcessing {feature_type} with {total_samples} samples")
                combined_df = pd.concat(dfs).reset_index().rename(columns={"index": "id"})
                print(f"Combined shape: {combined_df.shape}")
                
                # Save raw features
                save_path = os.path.join(process_dir, f'{feature_type}.parquet')
                combined_df.to_parquet(save_path)
                print(f"\nSaved raw features to {save_path}")

    def _create_node_mapping(self, all_data: List[Dict]) -> Dict[int, Dict[Any, int]]:
        """Create mapping between node IDs and indices."""
        return {
            timestep: {rid: idx for idx, rid in enumerate(
                data[cfg.REQUEST_FEATURES].keys())}
            for timestep, data in enumerate(all_data)
        }

    def _save_graph_data(self, all_data: List[Dict], process_dir: str,
                         node_mapping: Dict[int, Dict[Any, int]]) -> None:
        """Save graph structure data without normalization.

        Args:
            all_data: List of data dictionaries
            process_dir: Directory to save processed data
            node_mapping: Mapping between node IDs and indices
        """
        for graph_type in [cfg.REQUEST_REQUEST_GRAPH, cfg.VEHICLE_REQUEST_GRAPH]:
            dfs = []
            for timestep, data in enumerate(all_data):
                edges = []
                for source, targets in data[graph_type].items():
                    for target, info in targets.items():
                        edge_attrs = self._get_edge_attributes(
                            timestep, node_mapping, graph_type, source, target, info)
                        edge_attrs['timestep'] = timestep
                        edges.append(edge_attrs)
                if edges:
                    dfs.append(pd.DataFrame(edges))

            if dfs:
                # Combine all timesteps and save raw data
                combined_df = pd.concat(dfs)

                # Save raw edge data
                save_path = os.path.join(process_dir, f'{graph_type}.parquet')
                combined_df.to_parquet(save_path)
                print(f"\nSaved raw graph data to {save_path}")
                
                print(f"\n=== Processing {graph_type} features ===")
                print(f"Total features found: {len(combined_df.columns)}")
                print("\nAll columns:", combined_df.columns.tolist())

    def _get_edge_attributes(self, timestep: int, node_mapping: Dict[int, Dict[Any, int]],
                             graph_type: str, source: Any, target: Any,
                             info: Dict) -> Dict:
        """Get attributes for a graph edge.
        
        Preserves all computed edge features for GNN training while handling
        special cases for nested structures.
        
        Args:
            timestep: Current timestep
            node_mapping: Mapping from node IDs to indices
            graph_type: Type of graph (REQUEST_REQUEST_GRAPH or VEHICLE_REQUEST_GRAPH)
            source: Source node ID
            target: Target node ID
            info: Dictionary of edge information/features
            
        Returns:
            Dict with all edge attributes in a flat structure
        """
        # Start with node endpoints
        attrs = {
            'source': node_mapping[timestep][source] if graph_type == cfg.REQUEST_REQUEST_GRAPH else source,
            'target': node_mapping[timestep][target]
        }

        # Handle travel_cost separately since it's a nested structure
        if 'travel_cost' in info:
            if isinstance(info['travel_cost'], list):
                # For request-request edges
                if len(info['travel_cost']) >= 6:
                    tt_names = [
                        'tt_origin1_dest1',      # origin1 -> dest1
                        'tt_between_pickups',    # origin1 -> origin2
                        'tt_pickup1_dropoff2',   # origin1 -> dest2
                        'tt_dropoff1_pickup2',   # dest1 -> origin2
                        'tt_origin2_dest2',      # origin2 -> dest2
                        'tt_between_dropoffs'    # dest1 -> dest2
                    ]
                    for name, cost in zip(tt_names, info['travel_cost']):
                        if isinstance(cost, dict):
                            attrs[name] = cost.get('travel_time', 0)
            else:
                # For vehicle-request edges
                if isinstance(info['travel_cost'], dict):
                    attrs['veh_to_req_travel_time'] = info['travel_cost'].get('travel_time', 0)

        # Copy all other features (except nested structures)
        for key, value in info.items():
            # Skip already processed nested structures
            if key == 'travel_cost':
                continue
                
            # Include only serializable primitive types
            if isinstance(value, (int, float, str, bool)) or value is None:
                attrs[key] = value
            elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in value):
                # Handle simple lists of primitives
                attrs[key] = value
        return attrs

    def _get_node_index_fn(self, node_type: str):
        """Get the appropriate node index function for a node type."""
        return self._node_vid if node_type == 'vehicle' else self._node_rid

    def _node_rid(self, rid: Any) -> int:
        """Get node index for a request ID."""
        return rid

    def _node_vid(self, vid: Any) -> int:
        """Get node index for a vehicle ID."""
        return vid + self.timestep_n_requests

    @staticmethod
    def load_processed_data(data_dir):
        data = {}
        for file in os.scandir(data_dir):
            if not file.name.endswith('.parquet'):
                continue
            file_name = file.name[:file.name.find('.')]
            data[file_name] = pd.read_parquet(file.path)
        return data

    def process_single_timestep(self, data: Dict, timestep: int, edge_type: str = cfg.VEHICLE_REQUEST_GRAPH) -> pd.DataFrame:
        """
        Process and extract features for a single timestep's data (in-memory, no disk I/O),
        and return a merged DataFrame of edge and node features ready for classifier input.
        Args:
            data: Raw data dict for the current timestep (must contain keys like cfg.REQUEST_FEATURES, cfg.VEHICLE_FEATURES, cfg.REQUEST_REQUEST_GRAPH, cfg.VEHICLE_REQUEST_GRAPH, etc.)
            timestep: The current timestep
            edge_type: Which edge graph to use (cfg.VEHICLE_REQUEST_GRAPH or cfg.REQUEST_REQUEST_GRAPH)
        Returns:
            merged: DataFrame with edge features and merged source/target node features
            y: Series of labels (if present in edge features)
        """
        # TODO update if needed later
        import pandas as pd
        self.timestep_n_requests = len(data.get(cfg.REQUEST_FEATURES, {}))
        data = self._add_graph_features(data)

        # Build edge DataFrame
        edge_graph = data[edge_type]
        edge_rows = []
        for source, targets in edge_graph.items():
            for target, features in targets.items():
                row = {'source': source, 'target': target}
                row.update(features)
                edge_rows.append(row)
        edge_df = pd.DataFrame(edge_rows)
        edge_df['timestep'] = timestep

        # Node features
        req_df = pd.DataFrame.from_dict(
            data[cfg.REQUEST_FEATURES], orient='index').reset_index().rename(columns={'index': 'id'})
        veh_df = pd.DataFrame.from_dict(
            data[cfg.VEHICLE_FEATURES], orient='index').reset_index().rename(columns={'index': 'id'})

        # Merge node features
        source_type = 'veh' if edge_type == cfg.VEHICLE_REQUEST_GRAPH else 'req'
        src_df = veh_df if source_type == 'veh' else req_df
        tgt_df = req_df

        src_feat_cols = [col for col in src_df.columns if col not in ['id']]
        tgt_feat_cols = [col for col in tgt_df.columns if col not in ['id']]
        src_df_renamed = src_df[src_feat_cols + ['id']
                                ].rename(columns={col: f'src_{col}' for col in src_feat_cols})
        tgt_df_renamed = tgt_df[tgt_feat_cols + ['id']
                                ].rename(columns={col: f'tgt_{col}' for col in tgt_feat_cols})

        merged = edge_df.merge(src_df_renamed, left_on='source',
                               right_on='id', how='left').drop(columns=['id'])
        merged = merged.merge(tgt_df_renamed, left_on='target',
                              right_on='id', how='left').drop(columns=['id'])
        merged = merged.loc[:, ~merged.columns.duplicated()]
        merged = merged.fillna(0)

        # Label column if present
        y = merged['label'] if 'label' in merged.columns else None
        return merged, y

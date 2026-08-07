# Standard library imports
import bz2
import logging
import math
import os
import pickle
import shutil
from collections import defaultdict
from typing import Dict, List, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Third-party imports
import networkx as nx
import pandas as pd

# Local imports
from gnn_project.config import Config
from gnn_project.data_processing.utils import calculate_bearing, haversine_distance, is_peak_hour
from gnn_project.defaults import *
from src.misc.globals import G_TRAIN_FEATURE_O_POS_LAT, G_TRAIN_FEATURE_O_POS_LON, \
    G_TRAIN_FEATURE_D_POS_LON, G_TRAIN_FEATURE_D_POS_LAT, \
    G_TRAIN_FEATURE_DIRECT_TD, G_TRAIN_FEATURE_DIRECT_TT, G_TRAIN_FEATURE_V_POS_LAT, G_TRAIN_FEATURE_V_POS_LON, \
    G_TRAIN_FEATURE_TW_PE, G_TRAIN_FEATURE_TW_PL, G_TRAIN_FEATURE_RQ_TIME, \
    G_TRAIN_FEATURE_TRAVEL_TIME, G_TRAIN_FEATURE_TRAVEL_DIST, G_TRAIN_FEATURE_LOCKED, G_TRAIN_FEATURE_TRAVEL_COST

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processes raw simulation data for machine learning tasks.

    Main responsibilities:
    - Extracts features from simulation data
    - Constructs graphs for GNN models
    - Transforms and saves processed data
    """

    def __init__(self, train_data_dir: str, config: Config, prefer_processed: bool = True):
        """Initialize the DataProcessor.

        Args:
            train_data_dir: Directory containing training data
            config: Configuration for data processing
            prefer_processed: If True, load processed data if it exists (default). If False, always regenerate.
        """
        self.train_data_dir = train_data_dir
        self.config = config
        self.prefer_processed = prefer_processed

        self.r_key = config.request_features_key
        self.v_key = config.vehicle_features_key
        self.rr_key = config.request_request_graph_key
        self.vr_key = config.vehicle_request_graph_key

    def process_data(self, scenario_name: str) -> dict[Any, Any]:
        """Process raw data and generate feature-rich dataset.

        Args:
            scenario_name: Scenario name

        Returns:
            List of dictionaries containing processed data for each timestep
        """
        processed_dir = self.config.processed_dir / scenario_name

        # Try to load existing processed data if preferred
        if self.prefer_processed and os.path.exists(processed_dir):
            try:
                data = self.load_processed_data(processed_dir)
                return data
            except Exception as e:
                logger.error(f"Error loading processed data: {str(e)}")
                import traceback
                traceback.print_exc()
                # If loading fails, we'll recreate the directory below

        # Process new data
        all_data = self._process_timesteps()

        # Clean directory before saving new data
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
        os.makedirs(processed_dir)

        node_mapping = self._create_node_mapping(all_data)

        # Save raw features and graph data
        self._save_node_data(processed_dir, all_data)
        self._save_graph_data(all_data, processed_dir, node_mapping)
        all_data = self.load_processed_data(processed_dir)
        return all_data

    def _process_timesteps(self) -> List[Dict]:
        """Process data for each timestep.

        Returns:
            List of processed data dictionaries for each timestep
        """
        # Use parallel processing if enabled and we have multiple timesteps
        timesteps = list(range(self.config.sim_start, self.config.sim_end, self.config.sim_step))
        
        if getattr(self.config, 'use_parallel_processing', True) and len(timesteps) > 1:
            return self._process_timesteps_parallel(timesteps)
        else:
            # Sequential processing (original behavior)
            all_data = {}
            for timestep in timesteps:
                data = self._load_timestep_data(timestep)
                data = self._add_graph_features(timestep, data)
                all_data[timestep] = data
            return all_data
    
    def _process_timesteps_parallel(self, timesteps: List[int]) -> Dict:
        """Process timesteps in parallel using ThreadPoolExecutor.
        
        Uses ThreadExecutor since timestep processing is often I/O bound (file loading).
        
        Args:
            timesteps: List of timesteps to process
            
        Returns:
            Dictionary mapping timesteps to processed data
        """
        def process_single_timestep(timestep):
            try:
                data = self._load_timestep_data(timestep)
                data = self._add_graph_features(timestep, data)
                return timestep, data
            except Exception as e:
                logger.error(f"Error processing timestep {timestep}: {e}")
                return timestep, None
        
        # Use ThreadPoolExecutor for I/O-bound timestep processing
        max_workers = min(getattr(self.config, 'max_workers', 4), len(timesteps), mp.cpu_count())
        logger.info(f"Processing {len(timesteps)} timesteps in parallel with {max_workers} workers")
        
        all_data = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_timestep, timesteps))
        
        # Collect results and filter out failed timesteps
        for timestep, data in results:
            if data is not None:
                all_data[timestep] = data
            else:
                logger.warning(f"Skipping failed timestep {timestep}")
                
        return all_data

    def _load_timestep_data(self, timestep: int) -> Dict:
        """Load data for a specific timestep from compressed pickle files.

        RR/VR files are flat edge-dict lists, converted here to nested dicts.

        Args:
            timestep: The timestep to load data for

        Returns:
            Dictionary containing data for the timestep
        """
        timestep_dir = os.path.join(self.train_data_dir, str(timestep))
        if not os.path.exists(timestep_dir):
            logger.error(f"ERROR: Directory not found: {timestep_dir}")
            raise FileNotFoundError(f"No data for timestep {timestep}")

        data = {}
        for file in os.scandir(timestep_dir):
            if not file.name.endswith('_compressed.pkl'):
                continue
            name = file.name[:-len('_compressed.pkl')]
            try:
                with bz2.BZ2File(file.path, 'rb') as f:
                    raw = pickle.load(f)

                if name in [self.rr_key, self.vr_key]:
                    # Flat list of edge dicts -> nested dict {source: {target: {features}}}
                    nested_dict = {}
                    for edge in raw:
                        edge_features = {k: v for k, v in edge.items() if k not in ('source', 'target')}
                        nested_dict.setdefault(edge['source'], {})[edge['target']] = edge_features
                    data[name] = nested_dict
                else:
                    data[name] = raw

            except Exception as e:
                logger.error(f"Error loading {file.name}: {str(e)}")

        for required_key in [self.r_key, self.v_key, self.rr_key, self.vr_key]:
            if required_key not in data:
                raise KeyError(
                    f"Required key {required_key} not found in timestep data")
        return data

    def _add_graph_features(self, timestep: int, data: Dict) -> Dict:
        """Add graph features to the data, including NetworkX-based features.

        Args:

            timestep: Current timestep
            data: Dictionary containing raw data for the timestep

        Returns:
            Updated data dictionary with added graph features
        """
        # Add domain-specific features first
        self._add_temporal_features(timestep, data)
        self._add_spatial_features(data)
        self._add_competition_features(data)

        # Create NetworkX graphs
        G_rr, G_vr, G_combined = self.create_nx_graphs(data)

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

    def create_nx_graphs(self, data: Dict) -> tuple[nx.Graph, nx.Graph, nx.Graph]:
        """Create undirected request-request/vehicle-request/combined graphs.

        Undirected since compatibility is symmetric, but the raw data only stores
        one direction per pair - directed graphs made neighbor/degree features one-sided.

        Args:
            data: Dictionary containing raw data for the timestep

        Returns:
            Tuple of three NetworkX Graphs: (G_rr, G_vr, G_combined)
        """
        # Request-Request graph
        G_rr = nx.Graph()
        for src, targets in data[self.rr_key].items():
            for tgt in targets:
                G_rr.add_edge(f'r{src}', f'r{tgt}')

        # Vehicle-Request graph
        G_vr = nx.Graph()
        for veh, targets in data[self.vr_key].items():
            for req in targets:
                G_vr.add_edge(f'v{veh}', f'r{req}')

        # Combined graph
        G_combined = nx.compose(G_rr, G_vr)

        return G_rr, G_vr, G_combined

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
            centralities = {
                'degree': nx.degree_centrality(G),
                'betweenness': nx.betweenness_centrality(G),
                'pagerank': nx.pagerank(G),
            }
            try:
                centralities['closeness'] = nx.closeness_centrality(G)
            except Exception as e:
                logger.warning(
                    f"Warning: Could not calculate closeness centrality: {str(e)}")
                centralities['closeness'] = {node: 0.0 for node in G.nodes()}
            return centralities

        centralities = {
            'rr': calculate_centralities(G_rr) if len(G_rr) > 0 else None,
            'vr': calculate_centralities(G_vr) if len(G_vr) > 0 else None,
            'combined': calculate_centralities(G_combined) if len(G_combined) > 0 else None
        }

        def update_feats(feats, prefix, cdict, node_id):
            if cdict:
                feats.update({
                    f'{prefix}_degree_centrality': cdict['degree'].get(node_id, 0.0),
                    f'{prefix}_betweenness_centrality': cdict['betweenness'].get(node_id, 0.0),
                    f'{prefix}_closeness_centrality': cdict['closeness'].get(node_id, 0.0),
                    f'{prefix}_pagerank': cdict['pagerank'].get(node_id, 0.0)
                })

        for req_id, feats in data[self.r_key].items():
            update_feats(feats, 'nx_rr', centralities['rr'], f'r{req_id}')
            update_feats(feats, 'nx_combined',
                         centralities['combined'], f'r{req_id}')

        for veh_id, feats in data[self.v_key].items():
            update_feats(feats, 'nx_vr', centralities['vr'], f'v{veh_id}')
            update_feats(feats, 'nx_combined',
                         centralities['combined'], f'v{veh_id}')

    def _calculate_node_degrees(self, data: Dict, G_rr: nx.Graph, G_vr: nx.Graph, G_combined: nx.Graph) -> tuple[Dict[int, int], Dict[int, int]]:
        """Calculate degrees for all nodes. Nodes are prefixed 'r'/'v' for requests/vehicles.

        Graphs are undirected, so in/out degree are identical; both dicts are still
        returned so `_add_degree_features` doesn't need to change.

        Args:
            data: Dictionary containing raw data for the timestep
            G_rr: NetworkX Graph for request-request edges
            G_vr: NetworkX Graph for vehicle-request edges
            G_combined: NetworkX Graph for combined edges

        Returns:
            Tuple of two dictionaries: (in_degrees, out_degrees)
        """
        # Initialize degree dictionaries
        combined_in_degrees = defaultdict(int)
        combined_out_degrees = defaultdict(int)
        rr_in_degrees = defaultdict(int)  # Request-to-request degrees
        rr_out_degrees = defaultdict(int)
        vr_in_degrees = defaultdict(int)  # Vehicle-to-request degrees
        vr_out_degrees = defaultdict(int)

        # Calculate degrees from request-request graph
        for node in G_rr.nodes():
            rr_in_degrees[node] = rr_out_degrees[node] = G_rr.degree(node)

        # Calculate degrees from vehicle-request graph
        for node in G_vr.nodes():
            vr_in_degrees[node] = vr_out_degrees[node] = G_vr.degree(node)

        # Calculate degrees from combined graph
        for node in G_combined.nodes():
            combined_in_degrees[node] = combined_out_degrees[node] = G_combined.degree(node)

            # Add graph-specific degrees
            if node.startswith('r'):
                combined_in_degrees[f"{node}_rr"] = rr_in_degrees[node]
                combined_out_degrees[f"{node}_rr"] = rr_out_degrees[node]
                combined_in_degrees[f"{node}_vr"] = vr_in_degrees[node]
                combined_out_degrees[f"{node}_vr"] = vr_out_degrees[node]
            elif node.startswith('v'):
                # For vehicles, only VR degrees are relevant
                combined_in_degrees[f"{node}_vr"] = vr_in_degrees[node]
                combined_out_degrees[f"{node}_vr"] = vr_out_degrees[node]

        return combined_in_degrees, combined_out_degrees

    def _add_degree_features(self, data, in_degrees, out_degrees) -> None:
        """Add degree features to node attributes. Degree dictionaries use prefixed node IDs.

        Args:
            data: Dictionary containing raw data for the timestep
            in_degrees: Dictionary of in-degree counts for nodes
            out_degrees: Dictionary of out-degree counts for nodes
        """
        # Process requests (have both RR and VR degrees)
        for req_id, feats in data[self.r_key].items():
            node_id = f'r{req_id}'
            # Combined graph degrees
            feats['in_degree_total'] = in_degrees[node_id]
            feats['out_degree_total'] = out_degrees[node_id]

            # Request-Request graph degrees
            feats['in_degree_from_requests'] = in_degrees[f"{node_id}_rr"]
            feats['out_degree_to_requests'] = out_degrees[f"{node_id}_rr"]

            # Vehicle-Request graph degrees
            feats['in_degree_from_vehicles'] = in_degrees[f"{node_id}_vr"]
            feats['out_degree_to_vehicles'] = out_degrees[f"{node_id}_vr"]

            # Ratio features
            # Avoid division by zero
            total_in = max(1, feats['in_degree_total'])
            total_out = max(1, feats['out_degree_total'])
            feats['request_vehicle_in_ratio'] = feats['in_degree_from_vehicles'] / total_in
            feats['request_vehicle_out_ratio'] = feats['out_degree_to_vehicles'] / total_out

        # Process vehicles (only have VR degrees)
        for veh_id, feats in data[self.v_key].items():
            node_id = f'v{veh_id}'
            # Combined graph degrees (same as VR for vehicles)
            feats['in_degree'] = in_degrees[node_id]
            feats['out_degree'] = out_degrees[node_id]

            # Vehicle-Request specific degrees
            feats['in_degree_from_requests'] = in_degrees[f"{node_id}_vr"]
            feats['out_degree_to_requests'] = out_degrees[f"{node_id}_vr"]

            # Add total connections
            feats['total_request_connections'] = feats['in_degree_from_requests'] + \
                feats['out_degree_to_requests']

    def _add_neighborhood_features(self, data: Dict, G_combined) -> None:
        """Add neighborhood-based features to edges using NetworkX for common neighbors and Jaccard coefficient.

        Calculates type-specific neighborhood metrics:
        - For request-request edges: common request neighbors and common vehicle neighbors
        - For vehicle-request edges: common request neighbors and common vehicle neighbors

        Args:
            data: Dictionary containing raw data for the timestep
            G_combined: NetworkX Graph representing the combined graph
        """
        def calculate_type_specific_metrics(src, tgt, G, data, is_vr_edge=False):
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

            # Split neighbors by type - check prefix since nodes are prefixed
            src_req_neighbors = {
                n for n in neighbors_src if n.startswith('r')}
            src_veh_neighbors = {
                n for n in neighbors_src if n.startswith('v')}
            tgt_req_neighbors = {
                n for n in neighbors_tgt if n.startswith('r')}
            tgt_veh_neighbors = {
                n for n in neighbors_tgt if n.startswith('v')}

            # Calculate common neighbors by type
            common_requests = src_req_neighbors & tgt_req_neighbors
            if is_vr_edge:
                # src is a vehicle - it has no vehicle neighbors, so intersecting is always
                # empty. Use the request's other vehicle neighbors (competing vehicles) instead.
                common_vehicles = tgt_veh_neighbors - {src}
                exclusive_src_vehicles = set()
                exclusive_tgt_vehicles = common_vehicles
            else:
                common_vehicles = src_veh_neighbors & tgt_veh_neighbors
                # Calculate exclusive (uncommon) neighbors by type
                exclusive_src_vehicles = src_veh_neighbors - tgt_veh_neighbors
                exclusive_tgt_vehicles = tgt_veh_neighbors - src_veh_neighbors

            # Calculate exclusive (uncommon) neighbors by type
            exclusive_src_requests = src_req_neighbors - \
                tgt_req_neighbors  # Only connected to source
            exclusive_tgt_requests = tgt_req_neighbors - \
                src_req_neighbors  # Only connected to target

            # Calculate unions by type
            union_requests = src_req_neighbors | tgt_req_neighbors
            union_vehicles = common_vehicles if is_vr_edge else src_veh_neighbors | tgt_veh_neighbors

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
        for src, targets in data[self.rr_key].items():
            for tgt, features in targets.items():
                metrics = calculate_type_specific_metrics(
                    f'r{src}', f'r{tgt}', G_combined, data)
                features.update(metrics)
                # Add edge-specific ratios
                features['request_to_vehicle_neighbor_ratio'] = (
                    metrics['common_request_neighbors'] /
                    max(1, metrics['common_vehicle_neighbors'])
                )

        # Calculate metrics for vehicle-request edges
        for veh, targets in data[self.vr_key].items():
            for req, features in targets.items():
                metrics = calculate_type_specific_metrics(
                    f'v{veh}', f'r{req}', G_combined, data, is_vr_edge=True)
                # always 0 for VR edges (vehicles have no vehicle neighbors), unlike on RR
                del metrics['exclusive_src_vehicles']
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

        Args:
            data: Dictionary containing raw data for the timestep
            G_combined: NetworkX Graph representing the combined graph
        """
        # Calculate clustering coefficients for the combined graph
        clustering_combined = nx.clustering(G_combined)

        # Add to all node features
        for req_id, feats in data[self.r_key].items():
            feats['clustering_coeff'] = clustering_combined.get(
                f'r{req_id}', 0.0)

        for veh_id, feats in data[self.v_key].items():
            feats['clustering_coeff'] = clustering_combined.get(
                f'v{veh_id}', 0.0)

    def _add_temporal_features(self, current_time: int, data: Dict) -> None:
        """Add temporal features that capture time-related aspects of requests.

        Args:
            current_time: Current simulation time
            data: Dictionary containing raw data for the timestep
        """
        # For requests
        for _, feats in data[self.r_key].items():
            # Time urgency features
            time_until_latest = feats[G_TRAIN_FEATURE_TW_PL] - current_time
            time_window_width = feats[G_TRAIN_FEATURE_TW_PL] - \
                feats[G_TRAIN_FEATURE_TW_PE]  # currently constant
            feats.update({
                # How urgent is it
                'time_until_latest': max(0, time_until_latest),
                # How flexible is it
                'time_window_width': time_window_width,
                # Higher for more urgent requests
                'time_window_urgency': 1.0 / max(1, time_until_latest),
                # How long has it been waiting
                'request_age': current_time - feats[G_TRAIN_FEATURE_RQ_TIME]
            })

    def _add_spatial_features(self, data: Dict) -> None:
        """Add spatial features that capture geographical relationships.

        Args:
            data: Dictionary containing raw data for the timestep
        """
        # For requests
        for _, feats in data[self.r_key].items():
            # Calculate spatial features
            manhattan_distance = abs(
                feats[G_TRAIN_FEATURE_D_POS_LAT] - feats[G_TRAIN_FEATURE_O_POS_LAT]) + abs(feats[G_TRAIN_FEATURE_D_POS_LON] - feats[G_TRAIN_FEATURE_O_POS_LON])
            # Ratio of actual route distance to manhattan distance indicates route complexity
            route_directness = feats[G_TRAIN_FEATURE_DIRECT_TD] / \
                max(0.001, manhattan_distance)

            feats.update({
                'manhattan_distance': manhattan_distance,
                'route_directness': route_directness,
                'origin_dest_bearing': calculate_bearing(
                    feats[G_TRAIN_FEATURE_O_POS_LAT], feats[G_TRAIN_FEATURE_O_POS_LON],
                    feats[G_TRAIN_FEATURE_D_POS_LAT], feats[G_TRAIN_FEATURE_D_POS_LON]
                )
            })

    def _add_competition_features(self, data: Dict) -> None:
        """Add features that capture competition between requests and vehicle availability.

        Args:
            data: Dictionary containing raw data for the timestep     
        """
        # Calculate request density in different regions
        request_locations = {}
        for req_id, feats in data[self.r_key].items():
            if G_TRAIN_FEATURE_O_POS_LAT in feats and G_TRAIN_FEATURE_O_POS_LON in feats:
                request_locations[req_id] = (
                    feats[G_TRAIN_FEATURE_O_POS_LAT], feats[G_TRAIN_FEATURE_O_POS_LON])

        # For each request, count nearby requests and vehicles
        for req_id, feats in data[self.r_key].items():
            if req_id not in request_locations:
                continue

            nearby_requests = 0
            req_loc = request_locations[req_id]

            for other_id, other_loc in request_locations.items():
                if other_id != req_id:
                    dist = haversine_distance(
                        req_loc[0], req_loc[1],
                        other_loc[0], other_loc[1]
                    )
                    if dist < 1.0:  # within 1km
                        nearby_requests += 1

            # Count available vehicles nearby
            nearby_vehicles = 0
            for _, veh_feats in data[self.v_key].items():
                if G_TRAIN_FEATURE_V_POS_LAT in veh_feats and G_TRAIN_FEATURE_V_POS_LON in veh_feats:
                    dist = haversine_distance(
                        req_loc[0], req_loc[1],
                        veh_feats[G_TRAIN_FEATURE_V_POS_LAT], veh_feats[G_TRAIN_FEATURE_V_POS_LON]
                    )
                    if dist < 1.0:  # within 1km
                        nearby_vehicles += 1

            feats.update({
                'nearby_requests': nearby_requests,
                'nearby_vehicles': nearby_vehicles,
                'demand_supply_ratio': nearby_requests / max(1, nearby_vehicles)
            })

    def _add_vehicle_request_temporal_features(self, current_time: int, veh_feats: Dict,
                                               req_feats: Dict, edge_feats: Dict) -> None:
        """Add temporal compatibility features for vehicle-request edges.

        Args:
            current_time: Current simulation time
            veh_feats: Features of the vehicle
            req_feats: Features of the request
            edge_feats: Features of the edge between vehicle and request
        """
        # Add time-of-day features
        time_of_day = current_time % (24 * 60)  # Minutes within the day
        hour_of_day = time_of_day / 60.0
        is_peak = is_peak_hour(time_of_day)

        edge_feats.update({
            'is_peak_hour': float(is_peak),
            'hour_of_day_sin': math.sin(2 * math.pi * hour_of_day / 24),
            'hour_of_day_cos': math.cos(2 * math.pi * hour_of_day / 24),
        })

        # Use travel time information from edge features
        travel_time = edge_feats.get(G_TRAIN_FEATURE_TRAVEL_TIME, 0)

        # Calculate earliest arrival as current time + travel time to request
        earliest_arrival = current_time + travel_time

        # Time compatibility features
        earliest_pickup = req_feats[G_TRAIN_FEATURE_TW_PE]
        latest_pickup = req_feats[G_TRAIN_FEATURE_TW_PL]

        # Calculate various temporal margins
        arrival_slack = latest_pickup - earliest_arrival
        earliest_slack = earliest_pickup - earliest_arrival

        # Consider lock status in compatibility
        is_locked = req_feats.get(G_TRAIN_FEATURE_LOCKED, False)

        edge_feats.update({
            'is_locked': float(is_locked),
            'arrival_slack': max(0, arrival_slack),
            'earliest_slack': earliest_slack,
            'time_window_compatibility': min(1.0, max(0, arrival_slack) / max(1,
                                                                              latest_pickup - earliest_pickup)),
            'time_feasibility_score': 1.0 if arrival_slack > 0 else 0.0,
            'normalized_arrival_time': (earliest_arrival - earliest_pickup) / max(1,
                                                                                  latest_pickup - earliest_pickup),
        })

    def _add_vehicle_assignment_context_features(self, data: Dict) -> None:
        """
        Enrich vehicle-request edges with context from the vehicle's current assignment.

        Uses:
        - data[self.config.init_assignment_key]: current (initial) assignments
        - data[self.rr_key]: request-request edges with pooling metrics
        - data[self.r_key]: request node features (for lock status)
        """
        init_assignments = data.get(self.config.init_assignment_key, {})
        rr_edges = data[self.rr_key]
        req_feats_dict = data[self.r_key]

        for veh_id, targets in data[self.vr_key].items():
            # Current assignment sequence for this vehicle, if any
            seq = init_assignments.get(veh_id, [])
            assigned_reqs = seq[1:] if seq else []  # skip vehicle_id at position 0
            n_assigned = len(assigned_reqs)

            # Locked vs unlocked requests in current plan
            locked_reqs = [
                rid for rid in assigned_reqs
                if req_feats_dict.get(rid, {}).get(G_TRAIN_FEATURE_LOCKED, False)
            ]
            n_locked = len(locked_reqs)
            n_unlocked = n_assigned - n_locked

            locked_set = set(locked_reqs)
            assigned_set = set(assigned_reqs)

            # Precompute some vehicle-level features once per vehicle
            veh_level_features = {
                "veh_n_assigned": float(n_assigned),
                "veh_n_locked_assigned": float(n_locked),
                "veh_n_unlocked_assigned": float(n_unlocked),
                "veh_locked_fraction": float(n_locked) / max(1.0, float(n_assigned)),
            }

            for req_id, edge_feats in targets.items():
                # --- Vehicle-level assignment stats on this edge -------------------
                edge_feats.update(veh_level_features)

                # --- Is this request already in the vehicle's current plan? -------
                if req_id in assigned_set:
                    pos = assigned_reqs.index(req_id)
                    edge_feats["is_in_initial_plan"] = 1.0
                    edge_feats["initial_plan_position"] = float(pos)
                    edge_feats["initial_plan_position_norm"] = (
                        float(pos) / max(1.0, float(n_assigned - 1))
                        if n_assigned > 1 else 0.0
                    )
                else:
                    edge_feats["is_in_initial_plan"] = 0.0
                    edge_feats["initial_plan_position"] = -1.0
                    edge_feats["initial_plan_position_norm"] = -1.0

                # --- Aggregated pooling metrics vs currently assigned requests ----
                pooling_scores = []
                max_detour_ratios = []

                for other_id in assigned_reqs:
                    if other_id == req_id:
                        continue

                    # Try both directions for RR edge
                    rr_edge = rr_edges.get(req_id, {}).get(other_id)
                    if rr_edge is None:
                        rr_edge = rr_edges.get(other_id, {}).get(req_id)

                    if rr_edge is None:
                        continue

                    pooling_scores.append(rr_edge.get("overall_pooling_score", 0.0))
                    if "max_detour_ratio" in rr_edge:
                        max_detour_ratios.append(rr_edge["max_detour_ratio"])

                if pooling_scores:
                    edge_feats["assigned_overlap_count"] = float(len(pooling_scores))
                    edge_feats["assigned_mean_pooling_score"] = (
                        float(sum(pooling_scores)) / len(pooling_scores)
                    )
                    edge_feats["assigned_max_pooling_score"] = float(max(pooling_scores))
                else:
                    edge_feats["assigned_overlap_count"] = 0.0
                    edge_feats["assigned_mean_pooling_score"] = 0.0
                    edge_feats["assigned_max_pooling_score"] = 0.0

                if max_detour_ratios:
                    edge_feats["assigned_min_max_detour_ratio"] = float(min(max_detour_ratios))
                else:
                    edge_feats["assigned_min_max_detour_ratio"] = 0.0
    
    def _extract_travel_costs_from_edge(self, edge_feats: Dict) -> None:
        """Rename raw RR edge travel features to the tt_/td_<pair> keys the rest of this
        module expects (raw keys are flat, e.g. 'o1_o2_travel_time'), dropping the rest.

        Args:
            edge_feats: Features of the edge
        """
        edge_feats_copy = edge_feats.copy()
        edge_feats.clear()
        for prefix, feature_type in [('tt', G_TRAIN_FEATURE_TRAVEL_TIME), ('td', G_TRAIN_FEATURE_TRAVEL_DIST)]:
            suffix = f'_{feature_type}'
            for key, value in edge_feats_copy.items():
                if key.endswith(suffix):
                    pair_name = key[:-len(suffix)]
                    edge_feats[f'{prefix}_{pair_name}'] = value

    def _calculate_spatial_proximity(self, edge_feats: Dict) -> Dict:
        """Calculate spatial proximity metrics between two requests.

        Args:
            edge_feats: Features of the edge between the two requests

        Returns:
            Dictionary of spatial proximity metrics
        """
        return {
            'origin_proximity': 1.0 / max(0.1, edge_feats['td_o1_o2']),
            'destination_proximity': 1.0 / max(0.1, edge_feats['td_d1_d2']),
            'spatial_compatibility': 1.0 / (1.0 + edge_feats['td_o1_o2'] + edge_feats['td_d1_d2']),
        }

    def _calculate_detour_metrics(self, req1_feats: Dict, req2_feats: Dict, edge_feats: Dict) -> Dict:
        """Calculate detour and ride-sharing distance metrics for two requests.

        For each of the four possible pooling sequences
        (O1->O2->D1->D2, O2->O1->D1->D2, O1->O2->D2->D1, O2->O1->D2->D1),
        we compute:
        - total vehicle distance for the sequence,
        - per-request in-vehicle distance along that sequence,
        - per-request detour ratios and extra distances.

        We then:
        - determine whether there exists a sequence where both requests stay
          within certain detour thresholds (including max detour ratio),),
        - pick a "best" sequence to summarize with:
          * primary: both detours <= max_allowed_detour,
            choose the one with minimal total distance;
          * fallback: sequence with minimal max per-request detour ratio.

        Returned features are based on this best sequence and global feasibility.
        """
        max_allowed_detour = self.config.max_detour_ratio

        # Direct OD distances for each request
        direct_dist1 = req1_feats.get(G_TRAIN_FEATURE_DIRECT_TD, 0.0)
        direct_dist2 = req2_feats.get(G_TRAIN_FEATURE_DIRECT_TD, 0.0)

        # Small epsilon to avoid division by zero
        def safe_div(num, denom):
            return num / max(0.1, denom)

        # Helper to read pairwise distances from edge_feats
        def d(a: str, b: str) -> float:
            # a, b in {"o1", "o2", "d1", "d2"}
            return edge_feats.get(f"td_{a}_{b}", 0.0)

        # Precompute commonly used pairwise distances
        o1_o2 = d("o1", "o2")
        o2_o1 = d("o2", "o1")
        o1_d1 = d("o1", "d1")
        o2_d2 = d("o2", "d2")
        o1_d2 = d("o1", "d2")
        o2_d1 = d("o2", "d1")
        d1_d2 = d("d1", "d2")
        d2_d1 = d("d2", "d1")

        sequences = []

        # Sequence 1: O1 -> O2 -> D1 -> D2
        total_1 = o1_o2 + o2_d1 + d1_d2
        r1_1 = o1_o2 + o2_d1                  # O1 -> O2 -> D1
        r2_1 = o2_d1 + d1_d2                  # O2 -> D1 -> D2
        sequences.append(("O1_O2_D1_D2", total_1, r1_1, r2_1))

        # Sequence 2: O2 -> O1 -> D1 -> D2
        total_2 = o2_o1 + o1_d1 + d1_d2
        r1_2 = o1_d1                          # O1 -> D1
        r2_2 = o2_o1 + o1_d1 + d1_d2          # O2 -> O1 -> D1 -> D2
        sequences.append(("O2_O1_D1_D2", total_2, r1_2, r2_2))

        # Sequence 3: O1 -> O2 -> D2 -> D1
        total_3 = o1_o2 + o2_d2 + d2_d1
        r1_3 = o1_o2 + o2_d2 + d2_d1          # O1 -> O2 -> D2 -> D1
        r2_3 = o2_d2                          # O2 -> D2
        sequences.append(("O1_O2_D2_D1", total_3, r1_3, r2_3))

        # Sequence 4: O2 -> O1 -> D2 -> D1
        total_4 = o2_o1 + o1_d2 + d2_d1
        r1_4 = o1_d2 + d2_d1                  # O1 -> D2 -> D1
        r2_4 = o2_o1 + o1_d2                  # O2 -> O1 -> D2
        sequences.append(("O2_O1_D2_D1", total_4, r1_4, r2_4))

        # Compute per-sequence metrics
        seq_infos = []
        for name, total_dist, r1_dist, r2_dist in sequences:
            r1_detour_ratio = safe_div(r1_dist, direct_dist1)
            r2_detour_ratio = safe_div(r2_dist, direct_dist2)

            r1_extra = r1_dist - direct_dist1
            r2_extra = r2_dist - direct_dist2

            max_detour_ratio = max(r1_detour_ratio, r2_detour_ratio)
            avg_detour_ratio = 0.5 * (r1_detour_ratio + r2_detour_ratio)
            max_extra = max(r1_extra, r2_extra)
            avg_extra = 0.5 * (r1_extra + r2_extra)

            seq_infos.append({
                "name": name,
                "total_dist": total_dist,
                "r1_dist": r1_dist,
                "r2_dist": r2_dist,
                "r1_detour_ratio": r1_detour_ratio,
                "r2_detour_ratio": r2_detour_ratio,
                "r1_extra": r1_extra,
                "r2_extra": r2_extra,
                "max_detour_ratio": max_detour_ratio,
                "avg_detour_ratio": avg_detour_ratio,
                "max_extra": max_extra,
                "avg_extra": avg_extra,
            })

        # Global metric: min pooled distance over all sequences
        min_pooled_distance = min(info["total_dist"] for info in seq_infos)

        # Detour-within-threshold flags:
        # does there exist a SEQUENCE where BOTH requests have detour <= t?
        thresholds = [1.1, 1.2, 1.4, 1.6, 2.0]
        detour_within_threshold = {}
        for t in thresholds:
            feasible_for_t = any(
                (info["r1_detour_ratio"] <= t and info["r2_detour_ratio"] <= t)
                for info in seq_infos
            )
            detour_within_threshold[f"detour_within_{t}x"] = int(feasible_for_t)
        

        # Dedicated flag for *exact* sim cap feasibility
        feasible_under_cap = any(
            (info["r1_detour_ratio"] <= self.config.max_detour_ratio and
             info["r2_detour_ratio"] <= self.config.max_detour_ratio)
            for info in seq_infos
        )

        # Choose a "best" sequence for summary metrics
        # First, prefer sequences where both requests are within a reasonable detour cap
        feasible_seqs = [
            info for info in seq_infos
            if info["r1_detour_ratio"] <= max_allowed_detour
            and info["r2_detour_ratio"] <= max_allowed_detour
        ]

        if feasible_seqs:
            # Among feasible ones, pick the one with minimal total vehicle distance
            best = min(feasible_seqs, key=lambda x: x["total_dist"])
        else:
            # Fallback: pick the sequence with the minimal worst-case detour ratio
            best = min(seq_infos, key=lambda x: x["max_detour_ratio"])

        # Symmetry: difference between per-request detour ratios in the chosen sequence
        detour_symmetry = abs(best["r1_detour_ratio"] - best["r2_detour_ratio"])

        # Margin relative to the detour cap (positive = comfortably within)
        detour_margin_to_cap = self.config.max_detour_ratio - best["max_detour_ratio"]

        return {
            # Vehicle-level pooling distance
            "min_pooled_distance": min_pooled_distance,

            # Per-sequence summaries based on the chosen "best" sequence
            "max_detour_ratio": best["max_detour_ratio"],
            "avg_detour_ratio": best["avg_detour_ratio"],
            "max_detour_distance": best["max_extra"],
            "avg_detour_distance": best["avg_extra"],
            "detour_symmetry": detour_symmetry,

            # Per-request detour metrics for the best sequence
            "r1_detour_ratio": best["r1_detour_ratio"],
            "r2_detour_ratio": best["r2_detour_ratio"],
            "r1_detour_distance": best["r1_extra"],
            "r2_detour_distance": best["r2_extra"],

            # Relation to the simulation detour cap
            "detour_ratio_cap": self.config.max_detour_ratio,
            "feasible_under_detour_cap": int(feasible_under_cap),
            "detour_margin_to_cap": detour_margin_to_cap,

            # Threshold feasibility across all sequences
            **detour_within_threshold,
        }

    def _calculate_pooling_metrics(self, req1_feats: Dict, req2_feats: Dict, edge_feats: Dict) -> Dict:
        """Calculate comprehensive metrics related to request pooling compatibility.

        Args:
            req1_feats: Features of request 1
            req2_feats: Features of request 2
            edge_feats: Features of the edge between the two requests

        Returns:
            Dictionary of pooling compatibility metrics
        """
        spatial_metrics = self._calculate_spatial_proximity(edge_feats)
        detour_metrics = self._calculate_detour_metrics(req1_feats, req2_feats, edge_feats)

        return {**spatial_metrics, **detour_metrics}

    def _calculate_ride_sharing_efficiency(self, req1_feats: Dict, req2_feats: Dict, edge_feats: Dict) -> Dict:
        """Calculate ride-sharing efficiency and time-based detour metrics.

        We consider four pooling sequences:
        - O1 -> O2 -> D1 -> D2
        - O2 -> O1 -> D1 -> D2
        - O1 -> O2 -> D2 -> D1
        - O2 -> O1 -> D2 -> D1

        For each sequence we compute:
        - total vehicle travel time,
        - per-request in-vehicle travel time along that sequence,
        - per-request time detour ratios and extra times (vs direct TT).

        We then:
        - evaluate pooling efficiency vs a single-vehicle sequential baseline,
        - pick a "best" sequence, aligned with the simulation's detour cap,
        - derive per-request and aggregate time-detour features.
        """

        # --- Detour cap (same rate as for distance) -------------------------------
        max_detour_ratio = self.config.max_detour_ratio

        # --- Direct travel times for each request --------------------------------
        direct_tt1 = req1_feats.get(G_TRAIN_FEATURE_DIRECT_TT, 0.0)
        direct_tt2 = req2_feats.get(G_TRAIN_FEATURE_DIRECT_TT, 0.0)

        def safe_div(num, denom):
            return num / max(0.1, denom)

        # Helper to read pairwise travel times from edge_feats
        def tt(a: str, b: str) -> float:
            # a, b in {"o1", "o2", "d1", "d2"}
            return edge_feats.get(f"tt_{a}_{b}", 0.0)

        # Precompute leg times
        tt_o1_d1 = tt("o1", "d1")
        tt_o1_o2 = tt("o1", "o2")
        tt_o2_o1 = tt("o2", "o1")
        tt_o1_d2 = tt("o1", "d2")
        tt_o2_d2 = tt("o2", "d2")
        tt_d1_d2 = tt("d1", "d2")
        tt_o2_d1 = tt("o2", "d1")
        tt_d2_d1 = tt("d2", "d1")
        tt_d1_o2 = tt("d1", "o2")  # used for sequential baseline

        sequences = []

        # Sequence 1: O1 -> O2 -> D1 -> D2
        total_1 = tt_o1_o2 + tt_o2_d1 + tt_d1_d2
        r1_1 = tt_o1_o2 + tt_o2_d1              # O1 -> O2 -> D1
        r2_1 = tt_o2_d1 + tt_d1_d2              # O2 -> D1 -> D2
        sequences.append(("O1_O2_D1_D2", total_1, r1_1, r2_1))

        # Sequence 2: O2 -> O1 -> D1 -> D2
        total_2 = tt_o2_o1 + tt_o1_d1 + tt_d1_d2
        r1_2 = tt_o1_d1                          # O1 -> D1
        r2_2 = tt_o2_o1 + tt_o1_d1 + tt_d1_d2    # O2 -> O1 -> D1 -> D2
        sequences.append(("O2_O1_D1_D2", total_2, r1_2, r2_2))

        # Sequence 3: O1 -> O2 -> D2 -> D1
        total_3 = tt_o1_o2 + tt_o2_d2 + tt_d2_d1
        r1_3 = tt_o1_o2 + tt_o2_d2 + tt_d2_d1    # O1 -> O2 -> D2 -> D1
        r2_3 = tt_o2_d2                          # O2 -> D2
        sequences.append(("O1_O2_D2_D1", total_3, r1_3, r2_3))

        # Sequence 4: O2 -> O1 -> D2 -> D1
        total_4 = tt_o2_o1 + tt_o1_d2 + tt_d2_d1
        r1_4 = tt_o1_d2 + tt_d2_d1              # O1 -> D2 -> D1
        r2_4 = tt_o2_o1 + tt_o1_d2              # O2 -> O1 -> D2
        sequences.append(("O2_O1_D2_D1", total_4, r1_4, r2_4))

        # --- Per-sequence metrics -------------------------------------------------
        seq_infos = []
        for name, total_t, r1_t, r2_t in sequences:
            r1_detour_ratio_t = safe_div(r1_t, direct_tt1)
            r2_detour_ratio_t = safe_div(r2_t, direct_tt2)

            r1_extra_t = r1_t - direct_tt1
            r2_extra_t = r2_t - direct_tt2

            max_detour_ratio_t = max(r1_detour_ratio_t, r2_detour_ratio_t)
            avg_detour_ratio_t = 0.5 * (r1_detour_ratio_t + r2_detour_ratio_t)
            max_extra_t = max(r1_extra_t, r2_extra_t)
            avg_extra_t = 0.5 * (r1_extra_t + r2_extra_t)

            seq_infos.append({
                "name": name,
                "total_time": total_t,
                "r1_time": r1_t,
                "r2_time": r2_t,
                "r1_detour_ratio_t": r1_detour_ratio_t,
                "r2_detour_ratio_t": r2_detour_ratio_t,
                "r1_extra_time": r1_extra_t,
                "r2_extra_time": r2_extra_t,
                "max_detour_ratio_t": max_detour_ratio_t,
                "avg_detour_ratio_t": avg_detour_ratio_t,
                "max_extra_time": max_extra_t,
                "avg_extra_time": avg_extra_t,
            })

        # Shared-ride time stats (vehicle-level)
        min_shared_ride_time = min(info["total_time"] for info in seq_infos)
        max_shared_ride_time = max(info["total_time"] for info in seq_infos)
        avg_shared_ride_time = sum(info["total_time"] for info in seq_infos) / max(1, len(seq_infos))

        # --- Baselines for efficiency --------------------------------------------
        # Single-vehicle sequential: serve R1 then reposition then R2
        total_direct_time = tt_o1_d1 + tt_d1_o2 + tt_o2_d2

        # Independent vehicles baseline (optional but useful signal)
        sum_direct_times = tt_o1_d1 + tt_o2_d2

        # System-level extra time ratio vs single-vehicle sequential baseline
        extra_time_ratio = max(
            0.0,
            (min_shared_ride_time - total_direct_time) / max(1.0, total_direct_time)
        )
        ride_sharing_efficiency = 1.0 / (1.0 + extra_time_ratio)

        # --- Cap-aware "best" sequence selection ---------------------------------
        feasible_seqs = [
            info for info in seq_infos
            if info["r1_detour_ratio_t"] <= max_detour_ratio
            and info["r2_detour_ratio_t"] <= max_detour_ratio
        ]

        if feasible_seqs:
            # Among cap-feasible sequences, pick the one with minimal total time
            best = min(feasible_seqs, key=lambda x: x["total_time"])
        else:
            # Fallback: pick sequence with the smallest worst-case time detour ratio
            best = min(seq_infos, key=lambda x: x["max_detour_ratio_t"])

        time_detour_symmetry = abs(best["r1_detour_ratio_t"] - best["r2_detour_ratio_t"])
        time_detour_margin_to_cap = max_detour_ratio - best["max_detour_ratio_t"]

        # Optional thresholds around the cap (for the model to learn gradations)
        thresholds_t = [1.1, 1.2, max_detour_ratio, max_detour_ratio + 0.2]
        time_detour_within_threshold = {}
        for t in thresholds_t:
            feasible_for_t = any(
                info["r1_detour_ratio_t"] <= t and info["r2_detour_ratio_t"] <= t
                for info in seq_infos
            )
            time_detour_within_threshold[f"time_detour_within_{t:.1f}x"] = int(feasible_for_t)

        feasible_under_time_cap = any(
            info["r1_detour_ratio_t"] <= max_detour_ratio
            and info["r2_detour_ratio_t"] <= max_detour_ratio
            for info in seq_infos
        )

        # reuses the detour-cap feasibility check computed above, since it already
        # captures whether some pooling sequence keeps both requests within the cap
        temporal_compatibility = float(feasible_under_time_cap)

        return {
            # System-level shared ride stats
            "min_shared_ride_time": min_shared_ride_time,
            "max_shared_ride_time": max_shared_ride_time,
            "avg_shared_ride_time": avg_shared_ride_time,
            "total_direct_time": total_direct_time,
            "sum_direct_times": sum_direct_times,

            # Efficiency vs sequential baseline
            "extra_time_ratio": extra_time_ratio,
            "ride_sharing_efficiency": ride_sharing_efficiency,

            # Rough time-window compatibility
            "temporal_compatibility": temporal_compatibility,

            # Time-detour metrics for the chosen "best" sequence
            "max_time_detour_ratio": best["max_detour_ratio_t"],
            "avg_time_detour_ratio": best["avg_detour_ratio_t"],
            "max_time_detour": best["max_extra_time"],
            "avg_time_detour": best["avg_extra_time"],
            "r1_time_detour_ratio": best["r1_detour_ratio_t"],
            "r2_time_detour_ratio": best["r2_detour_ratio_t"],
            "r1_time_detour": best["r1_extra_time"],
            "r2_time_detour": best["r2_extra_time"],
            "time_detour_symmetry": time_detour_symmetry,

            # Relation to detour cap (time-based view)
            "feasible_under_time_detour_cap": int(feasible_under_time_cap),
            "time_detour_margin_to_cap": time_detour_margin_to_cap,

            # Threshold feasibility across sequences
            **time_detour_within_threshold,
        }

    def _add_request_request_features(self, req1_feats: Dict, req2_feats: Dict,
                                      edge_feats: Dict) -> None:
        """Add all features for request-request edges.

        Args:
            req1_feats: Features of request 1
            req2_feats: Features of request 2
            edge_feats: Features of the edge between the two requests
        """
        # Extract travel costs. Overwrites edge_feats to flatten the nested structure. Needs to be done first.
        self._extract_travel_costs_from_edge(edge_feats)

        # Get pooling metrics
        pooling_metrics = self._calculate_pooling_metrics(
            req1_feats, req2_feats, edge_feats)

        # Get ride-sharing efficiency metrics
        efficiency_metrics = self._calculate_ride_sharing_efficiency(req1_feats, req2_feats, edge_feats)

        # Check lock status
        is_req1_locked = req1_feats.get(G_TRAIN_FEATURE_LOCKED, False)
        is_req2_locked = req2_feats.get(G_TRAIN_FEATURE_LOCKED, False)
        both_locked = is_req1_locked and is_req2_locked

        # Update edge features with all metrics
        edge_feats.update({
            # Lock status
            'src_locked': is_req1_locked,
            'tgt_locked': is_req2_locked,
            'both_locked': int(both_locked),
            
            # Pooling compatibility (spatial)
            **pooling_metrics,

            # Ride-sharing efficiency metrics
            **efficiency_metrics,

            # Combined score
            'overall_pooling_score': (
                pooling_metrics['spatial_compatibility'] *
                efficiency_metrics['temporal_compatibility'] *
                efficiency_metrics['ride_sharing_efficiency']
            )
        })

    def _add_edge_compatibility_features(self, current_time: int, data: Dict) -> None:
        """Add features that indicate compatibility between nodes.

        Args:
            current_time: Current simulation time
            data: Dictionary containing raw data for the timestep
        """
        # Process vehicle-request edges
        for veh_id, targets in data[self.vr_key].items():
            veh_feats = data[self.v_key][veh_id]

            for req_id, edge_feats in targets.items():
                req_feats = data[self.r_key][req_id]

                # Add temporal features
                self._add_vehicle_request_temporal_features(
                    current_time, veh_feats, req_feats, edge_feats)

        # Process request-request edges
        for req1_id, targets in data[self.rr_key].items():
            req1_feats = data[self.r_key][req1_id]

            for req2_id, edge_feats in targets.items():
                req2_feats = data[self.r_key][req2_id]
                self._add_request_request_features(
                    req1_feats, req2_feats, edge_feats)

        # Add assignment-context features on VR edges,
        # now that RR edges already have pooling/detour metrics
        self._add_vehicle_assignment_context_features(data)

    def _add_assignment_features(self, data: Dict) -> None:
        """Add assignment labels to edges.
        
        Initializes all edges with label=0, then sets label=1 for edges in assignments.
        This ensures the columns exist even when all values are 0 (e.g., during inference).

        Args:
            data: Dictionary containing graph data
        """
        # Initialize all edges with 0 labels first (ensures columns exist)
        for _, targets in data[self.vr_key].items():
            for _, edge_feats in targets.items():
                edge_feats[self.config.init_label_key] = 0
                if self.config.assignment_key in data:
                    edge_feats[self.config.label_key] = 0
        
        for _, targets in data[self.rr_key].items():
            for _, edge_feats in targets.items():
                edge_feats[self.config.init_label_key] = 0
                if self.config.assignment_key in data:
                    edge_feats[self.config.label_key] = 0
        
        # Now set 1 for edges that are in the assignments
        self._add_assignment_sequence(
            data, data[self.config.init_assignment_key], self.config.init_label_key, all_pairs=True)
        if self.config.assignment_key in data:
            self._add_assignment_sequence(
                data, data[self.config.assignment_key], self.config.label_key, all_pairs=True)

    def _add_assignment_sequence(self, data: Dict, assignments: Dict,
                                 feature_name: str, all_pairs: bool = True) -> None:
        """Add assignment sequence labels to edges.

        Args:
            data: Dictionary containing graph data
            assignments: Dictionary of vehicle_id -> sequence of request_ids
            feature_name: Name of the feature to set (e.g., 'init_assign' or 'optimal_assign')
            all_pairs: If True, add edges between all pairs of requests instead of only consecutive requests
        """
        for vehicle_id, sequence in assignments.items():
            if not sequence or len(sequence) < 2:
                continue

            # Add vehicle-to-request edges first
            for req_id in sequence[1:]:  # Skip the first item which is vehicle_id
                try:
                    data[self.vr_key][vehicle_id][req_id][feature_name] = 1
                except KeyError as e:
                    logger.warning(
                        f"Warning: Could not add V-R edge label for vehicle {vehicle_id} and request {req_id}: {str(e)}")

            # Now add request-to-request edges
            if all_pairs:
                # Add edges between all pairs of requests
                for i, req1 in enumerate(sequence[1:]):
                    # All requests after req1
                    for req2 in sequence[i + 2:]:
                        try:
                            data[self.rr_key][req1][req2][feature_name] = 1
                        except KeyError as e:
                            logger.warning(
                                f"Warning: Could not add R-R edge label between requests {req1} and {req2}: {str(e)}")
            else:
                # Add edges only between consecutive requests
                for req1, req2 in zip(sequence[1:-1], sequence[2:]):
                    try:
                        data[self.rr_key][req1][req2][feature_name] = 1
                    except KeyError as e:
                        logger.warning(
                            f"Warning: Could not add R-R edge label between requests {req1} and {req2}: {str(e)}")
                        continue

    def _save_node_data(self, process_dir: str, all_data: Dict[int, Dict]) -> None:
        """Save processed feature data without normalization.
        
        Important: The data is sorted by node ID to match the index mapping
        created in _create_node_mapping, ensuring consistency between node
        features and graph edge indices.

        Args:
            process_dir: Directory to save processed data
            all_data: List of data dictionaries for a single scenario
        """
        for feature_type in [self.r_key, self.v_key]:
            dfs = []
            total_samples = 0
            for timestep, data in all_data.items():
                if data[feature_type]:
                    df = pd.DataFrame.from_dict(
                        data[feature_type], orient='index')
                    df = df.fillna(0.0)
                    df[TIMESTEP] = timestep
                    # Sort by index (node IDs) to match the sorted order in _create_node_mapping
                    df = df.sort_index()
                    dfs.append(df)
                    total_samples += len(df)

            if dfs:
                combined_df = pd.concat(dfs).reset_index().rename(
                    columns={"index": ID})
                # Sort columns alphabetically for consistent ordering with inference
                combined_df = combined_df[sorted(combined_df.columns)]

                # Save raw features with snappy compression
                save_path = os.path.join(
                    process_dir, f'{feature_type}.parquet')
                combined_df.to_parquet(save_path, compression='snappy', index=False)

    def _create_node_mapping(self, all_data: List[Dict]) -> Dict[int, Dict[Any, int]]:
        """Create mapping between node IDs and indices.
        
        Sorts request IDs to ensure deterministic index assignment that matches inference.

        Args:
            all_data: List of data dictionaries

        Returns:
            Dictionary mapping timestep -> {node_id: index}
        """
        return {
            timestep: {rid: idx for idx, rid in enumerate(
                sorted(data[self.r_key].keys()))}
            for timestep, data in all_data.items()
        }

    def _save_graph_data(self, all_data: Dict[int, Dict], process_dir: str,
                         node_mapping: Dict[int, Dict[Any, int]]) -> None:
        """Save graph structure data without normalization.

        Args:
            all_data: List of data dictionaries
            process_dir: Directory to save processed data
            node_mapping: Mapping between node IDs and indices
        """
        for graph_type in [self.rr_key, self.vr_key]:
            dfs = []
            for timestep, data in all_data.items():
                edges = []
                for source, targets in data[graph_type].items():
                    for target, features in targets.items():
                        edge_attrs = {
                            SOURCE: node_mapping[timestep][source] if graph_type == self.rr_key else source,
                            TARGET: node_mapping[timestep][target]
                        }
                        edge_attrs.update(features)
                        edge_attrs[TIMESTEP] = timestep
                        edges.append(edge_attrs)
                if edges:
                    dfs.append(pd.DataFrame(edges))

            if dfs:
                # Combine all timesteps and save raw data
                combined_df = pd.concat(dfs)
                combined_df = combined_df.fillna(0.0)
                # Sort columns alphabetically for consistent ordering with inference
                combined_df = combined_df[sorted(combined_df.columns)]
                # Save raw edge data with snappy compression
                save_path = os.path.join(process_dir, f'{graph_type}.parquet')
                combined_df.to_parquet(save_path, compression='snappy', index=False)
            else:
                logger.warning(
                    f"\nNo edges found for {graph_type}, skipping save.")

    @staticmethod
    def load_processed_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
        """Load processed data from parquet files in a directory.

        Args:
            data_dir: Directory containing parquet files

        Returns:
            Dictionary with data loaded from each parquet file
        """
        data = {}
        for file in os.scandir(data_dir):
            if not file.name.endswith('.parquet'):
                continue
            file_name = file.name[:file.name.find('.')]
            data[file_name] = pd.read_parquet(file.path)
        return data

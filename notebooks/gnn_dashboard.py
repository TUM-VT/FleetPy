import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import HeteroData
import numpy as np
from utils import visualize_graph
import logging
import os
from glob import glob
from pathlib import Path
from models.HeteroGAT import HeteroGAT
from dataloaders.GNNDataLoader import GNNDataLoader
from data_processing.config import DataProcessingConfig as cfg

# Add BaseStorage class to safe globals for loading
import torch.serialization
from torch_geometric.data.storage import BaseStorage
torch.serialization.add_safe_globals([BaseStorage, 'numpy._core.multiarray.scalar'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GNNDashboard:
    def __init__(self, data=None, model=None, device=None):
        """
        Initialize the GNN Dashboard
        Args:
            data: List of HeteroData graphs
            model: Optional trained GNN model for predictions
            device: Device for model inference
        """
        self.data = data
        self.model = model
        self.device = device
        self.feature_names = {}
        
    @staticmethod
    def load_graph_data(scenario_paths: list, config: cfg = None, overwrite: bool = False) -> tuple:
        """
        Load graph data using GNNDataLoader
        Args:
            scenario_paths: List of paths to scenario directories
            config: cfg object
            overwrite: Whether to overwrite existing processed data
        Returns:
            data
        """
        try:
            if config is None:
                config = cfg(
                    sim_start=0,
                    sim_end=1800  # 30 minutes simulation
                )
            
            loader = GNNDataLoader(scenario_paths, config, overwrite=overwrite)
            data, *masks = loader.load_data()

            return data
        except Exception as e:
            st.error(f"Error loading graph data: {str(e)}")
            return None

    @staticmethod
    def load_model(model_path: str, device: torch.device = None, hidden_channels: int = 32, num_classes: int = 1) -> torch.nn.Module:
        """
        Load a trained model from a .pt file
        Args:
            model_path: Path to the .pt file containing model weights
            device: Device to load the model on
            hidden_channels: Size of hidden layers in GNN
            num_classes: Number of output classes (1 for binary classification)
        Returns:
            Loaded model
        """
        try:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize the model first
            model = HeteroGAT(hidden_channels, num_classes)
            
            # Load the checkpoint
            try:
                checkpoint = torch.load(model_path, weights_only=False)
            except Exception as e:
                print('Error loading checkpoint with weights_only=False:', e)
                checkpoint = torch.load(model_path, weights_only=True)
            
            # Load state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
        
    def run(self):
        """Run the Streamlit dashboard"""
        st.title("GNN Visualization Dashboard")
        
        if self.data is None:
            st.warning("No data loaded. Please load data first.")
            return
            
        # Sidebar controls
        with st.sidebar:
            st.header("Visualization Controls")
            
            # Graph selection
            graph_idx = st.number_input(
                "Select Graph Index", 
                min_value=0, 
                max_value=len(self.data)-1 if self.data else 0,
                value=0
            )
            
            # Node subset selection
            st.subheader("Node Subset Selection")
            graph = self.data[graph_idx]
            
            subset_nodes = {}
            for node_type in graph.node_types:
                st.write(f"{node_type.capitalize()} Nodes")
                num_nodes = graph[node_type].x.size(0)
                
                # Selection mode
                selection_mode = st.radio(
                    f"Selection Mode for {node_type}",
                    ["All", "Range", "Connected", "Status", "Manual", "Neighborhood"],
                    key=f"{node_type}_mode"
                )
                
                if selection_mode == "All":
                    subset_nodes[node_type] = list(range(num_nodes))
                
                elif selection_mode == "Range":
                    min_node, max_node = st.slider(
                        f"Select {node_type} node range",
                        0, num_nodes-1, (0, min(5, num_nodes-1)),
                        key=f"{node_type}_range"
                    )
                    subset_nodes[node_type] = list(range(min_node, max_node + 1))
                
                elif selection_mode == "Connected":
                    # Find connected nodes based on edges
                    connected_nodes = set()
                    for edge_type in graph.edge_types:
                        src_type, rel_type, dst_type = edge_type
                        if rel_type == 'rev_connects':
                            continue
                        
                        edge_index = graph[edge_type].edge_index
                        edge_y = graph[edge_type].y if hasattr(graph[edge_type], 'y') else None
                        
                        for i in range(edge_index.size(1)):
                            if edge_y is None or edge_y[i].item() == 1:  # Only consider positive edges if labels exist
                                if src_type == node_type:
                                    connected_nodes.add(edge_index[0][i].item())
                                if dst_type == node_type:
                                    connected_nodes.add(edge_index[1][i].item())
                    
                    subset_nodes[node_type] = sorted(list(connected_nodes))
                    st.write(f"Found {len(connected_nodes)} connected nodes")
                
                elif selection_mode == "Status":
                    if node_type == 'request':
                        # Allow filtering by locked status
                        show_locked = st.checkbox("Show Locked Requests", value=True)
                        show_unlocked = st.checkbox("Show Unlocked Requests", value=True)
                        
                        nodes_list = []
                        for i in range(num_nodes):
                            is_locked = graph[node_type].x[i][cfg.LOCKED_IDX].item() == 1
                            if (is_locked and show_locked) or (not is_locked and show_unlocked):
                                nodes_list.append(i)
                        subset_nodes[node_type] = nodes_list
                    else:
                        # For vehicles, you might want to add other status filters
                        subset_nodes[node_type] = list(range(num_nodes))
                
                elif selection_mode == "Manual":
                    # Allow manual selection of node IDs
                    node_ids_str = st.text_input(
                        f"Enter {node_type} IDs (comma-separated)",
                        value="0,1,2",
                        key=f"{node_type}_manual"
                    )
                    try:
                        node_ids = [int(x.strip()) for x in node_ids_str.split(',')]
                        # Filter out invalid IDs
                        node_ids = [x for x in node_ids if 0 <= x < num_nodes]
                        subset_nodes[node_type] = node_ids
                    except ValueError:
                        st.error("Please enter valid comma-separated numbers")
                        subset_nodes[node_type] = [0]

                elif selection_mode == "Neighborhood":
                    # Select seed nodes
                    seed_ids_str = st.text_input(
                        f"Enter seed {node_type} IDs (comma-separated)",
                        value="0",
                        key=f"{node_type}_seeds"
                    )
                    
                    # Number of hops
                    n_hops = st.slider(
                        "Number of hops",
                        min_value=1,
                        max_value=3,
                        value=1,
                        key=f"{node_type}_hops"
                    )
                    
                    try:
                        seed_ids = [int(x.strip()) for x in seed_ids_str.split(',')]
                        # Filter out invalid IDs
                        seed_ids = [x for x in seed_ids if 0 <= x < num_nodes]
                        
                        # Find n-hop neighborhood
                        neighborhood = set(seed_ids)
                        current_nodes = set(seed_ids)
                        
                        for _ in range(n_hops):
                            next_nodes = set()
                            # Look through all edge types
                            for edge_type in graph.edge_types:
                                src_type, rel_type, dst_type = edge_type
                                if rel_type == 'rev_connects':
                                    continue
                                    
                                edge_index = graph[edge_type].edge_index
                                
                                # If this node type is the source
                                if src_type == node_type:
                                    for i in range(edge_index.size(1)):
                                        if edge_index[0][i].item() in current_nodes:
                                            # Add destination node to next hop if it's of the same type
                                            if dst_type == node_type:
                                                next_nodes.add(edge_index[1][i].item())
                                
                                # If this node type is the destination
                                if dst_type == node_type:
                                    for i in range(edge_index.size(1)):
                                        if edge_index[1][i].item() in current_nodes:
                                            # Add source node to next hop if it's of the same type
                                            if src_type == node_type:
                                                next_nodes.add(edge_index[0][i].item())
                            
                            current_nodes = next_nodes
                            neighborhood.update(next_nodes)
                        
                        subset_nodes[node_type] = sorted(list(neighborhood))
                        st.write(f"Found {len(neighborhood)} nodes in the {n_hops}-hop neighborhood")
                        
                    except ValueError:
                        st.error("Please enter valid comma-separated numbers")
                        subset_nodes[node_type] = [0]
                
                st.write(f"Selected {len(subset_nodes[node_type])} {node_type}s")
            
            # Display options
            st.subheader("Display Options")
            show_predictions = st.checkbox("Show Model Predictions", value=True if self.model else False)
            
            # Visualization size controls
            st.subheader("Visualization Size")
            fig_width = st.slider("Figure Width", min_value=6, max_value=20, value=10, step=1)
            fig_height = st.slider("Figure Height", min_value=4, max_value=16, value=8, step=1)
            node_size = st.slider("Node Size", min_value=100, max_value=1000, value=300, step=50)
            
        # Main content area
        st.header(f"Graph Visualization (Index: {graph_idx})")
        
        # Use columns to control the width of the plot
        col1, col2, col3 = st.columns([1, 10, 1])
        
        with col2:
            # Create visualization
            plt.clf()  # Clear any existing plots
            fig = visualize_graph(
                self.data, 
                graph_idx=graph_idx,
                model=self.model if show_predictions else None,
                device=self.device,
                subset_nodes=subset_nodes,
                feature_names=self.feature_names,
                figsize=(fig_width, fig_height),
                node_size=node_size
            )
            
            # Display the plot in Streamlit
            st.pyplot(fig, use_container_width=True)
            
            # Clean up
            plt.close(fig)
        
        # Graph Statistics
        st.header("Graph Statistics")
        self.display_graph_statistics(graph, subset_nodes)
        
        # Node Features
        if st.checkbox("Show Node Features"):
            self.display_node_features(graph, subset_nodes)
            
        # Edge Features
        if st.checkbox("Show Edge Features"):
            self.display_edge_features(graph, subset_nodes)
    
    def display_graph_statistics(self, graph: HeteroData, subset_nodes: dict):
        """Display basic statistics about the graph"""
        stats_cols = st.columns(3)
        
        # Node statistics
        with stats_cols[0]:
            st.subheader("Node Counts")
            for node_type in graph.node_types:
                total_nodes = graph[node_type].x.size(0)
                selected_nodes = len(subset_nodes[node_type])
                st.write(f"{node_type.capitalize()}:")
                st.write(f"- Selected: {selected_nodes}")
                st.write(f"- Total: {total_nodes}")
        
        # Edge statistics
        with stats_cols[1]:
            st.subheader("Edge Counts")
            for edge_type in graph.edge_types:
                src, rel, dst = edge_type
                if rel == 'rev_connects':  # Skip reverse edges
                    continue
                edge_count = graph[edge_type].edge_index.size(1)
                st.write(f"{src} → {dst}:")
                st.write(f"- Total: {edge_count}")
                
        # Additional statistics
        with stats_cols[2]:
            st.subheader("Feature Dimensions")
            for node_type in graph.node_types:
                feat_dim = graph[node_type].x.size(1)
                st.write(f"{node_type.capitalize()} features: {feat_dim}")
    
    def display_node_features(self, graph: HeteroData, subset_nodes: dict):
        """Display feature values for selected nodes"""
        for node_type in graph.node_types:
            st.subheader(f"{node_type.capitalize()} Node Features")
            
            selected_indices = subset_nodes[node_type]
            if not selected_indices:
                st.write("No nodes selected")
                continue
                
            features = graph[node_type].x[selected_indices]
            
            # Create feature names if not available
            if node_type not in self.feature_names:
                self.feature_names[node_type] = [f"Feature_{i}" for i in range(features.shape[1])]
                
            # Create DataFrame for display
            import pandas as pd
            df = pd.DataFrame(
                features.detach().numpy(),
                columns=self.feature_names[node_type],
                index=[f"Node_{i}" for i in selected_indices]
            )
            st.dataframe(df)
    
    def display_edge_features(self, graph: HeteroData, subset_nodes: dict):
        """Display edge features for edges between selected nodes"""
        for edge_type in graph.edge_types:
            src, rel, dst = edge_type
            if rel == 'rev_connects':  # Skip reverse edges
                continue
                
            st.subheader(f"{src.capitalize()} → {dst.capitalize()} Edge Features")
            
            edge_index = graph[edge_type].edge_index
            edge_attr = graph[edge_type].edge_attr
            
            # Filter edges where both source and target are in selected subsets
            mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
            for i in range(edge_index.size(1)):
                src_idx = edge_index[0][i].item()
                dst_idx = edge_index[1][i].item()
                if src_idx in subset_nodes[src] and dst_idx in subset_nodes[dst]:
                    mask[i] = True
            
            filtered_edges = edge_index[:, mask]
            filtered_features = edge_attr[mask]
            
            if filtered_edges.size(1) == 0:
                st.write("No edges between selected nodes")
                continue
            
            # Create feature names if not available
            if edge_type not in self.feature_names:
                self.feature_names[edge_type] = [f"Feature_{i}" for i in range(filtered_features.shape[1])]
            
            # Create DataFrame for display
            import pandas as pd
            df = pd.DataFrame(
                filtered_features.detach().numpy(),
                columns=self.feature_names[edge_type],
                index=[f"Edge_{i}_{j}" for i, j in filtered_edges.t().tolist()]
            )
            st.dataframe(df)

def find_scenario_dirs() -> dict:
    """
    Find all available scenario directories and model files in the workspace
    Returns:
        Dictionary with paths to scenarios and models
    """
    base_studies_path = os.path.join(os.path.dirname(__file__), '..', 'studies')
    files = {
        'case_studies': [],
        'scenarios': {},
        'models': []
    }
    
    # Look for case study directories
    for case_study in os.listdir(base_studies_path):
        case_study_path = os.path.join(base_studies_path, case_study)
        if os.path.isdir(case_study_path):
            files['case_studies'].append(case_study)
            
            # Find scenarios within case study
            results_dir = os.path.join(case_study_path, 'results')
            if os.path.exists(results_dir):
                scenarios = [d for d in os.listdir(results_dir) 
                           if os.path.isdir(os.path.join(results_dir, d))]
                if scenarios:
                    files['scenarios'][case_study] = scenarios
    
    # Look for model files
    model_path = os.path.join(os.path.dirname(__file__), 'data', 'models')
    if os.path.exists(model_path):
        for path in glob(os.path.join(model_path, '**', '*.pt'), recursive=True):
            files['models'].append(path)
    
    return files

def main():
    """Main function to run the dashboard"""
    st.set_page_config(layout="wide")
    
    st.sidebar.title("GNN Dashboard")
    st.sidebar.write("""
    This dashboard helps visualize and debug GNN outputs.
    1. Select case study and scenarios
    2. Load graph data and model
    3. Visualize and analyze predictions
    """)
    
    # Data Loading Section
    st.sidebar.header("Data Selection")
    
    # Find available scenarios and models
    files = find_scenario_dirs()
    
    # Case study selection
    case_study = st.sidebar.selectbox(
        "Select Case Study",
        options=files['case_studies'],
        format_func=lambda x: x.replace('_', ' ').title()
    ) if files['case_studies'] else None
    
    if not case_study:
        st.warning("No case studies found. Please check the studies directory.")
        return
    
    # Scenario selection
    scenarios = files['scenarios'].get(case_study, [])
    selected_scenarios = st.sidebar.multiselect(
        "Select Scenarios",
        options=scenarios,
        default=scenarios[:3] if len(scenarios) > 0 else []
    ) if scenarios else []
    
    if not selected_scenarios:
        st.warning(f"No scenarios found in {case_study}. Please check the results directory.")
        return
    
    # Create scenario paths
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'studies', case_study, 'results')
    scenario_paths = [os.path.join(results_dir, sc) for sc in selected_scenarios]
    
    # Data processing config
    st.sidebar.subheader("Data Processing Config")
    sim_start = st.sidebar.number_input("Simulation Start Time", value=0)
    sim_end = st.sidebar.number_input("Simulation End Time", value=1800)
    overwrite = st.sidebar.checkbox("Overwrite Existing Data", value=False)
    
    # Load data button
    if st.sidebar.button("Load Data"):
        config = cfg(sim_start=sim_start, sim_end=sim_end)
        data = GNNDashboard.load_graph_data(scenario_paths, config, overwrite)
        
        if data is None:
            return
        
        # Store in session state
        st.session_state['data'] = data
        st.session_state['loaded'] = True
    
    # Model Loading Section
    model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if files['models']:
        st.sidebar.subheader("Model Loading (Optional)")
        load_model = st.sidebar.checkbox("Load Model for Predictions")
        
        if load_model:
            model_path = st.sidebar.selectbox(
                "Select Model File",
                options=files['models'],
                format_func=lambda x: os.path.relpath(x, os.path.dirname(__file__))
            )
            
            if model_path:
                hidden_channels = st.sidebar.number_input("Hidden Channels", value=32)
                model = GNNDashboard.load_model(model_path, device, hidden_channels)
    
    # Initialize and run dashboard if data is loaded
    dashboard = None
    if st.session_state.get('loaded', False):
        dashboard = GNNDashboard(
            data=st.session_state['data'],
            model=model,
            device=device)
    if dashboard:
        dashboard.run()

if __name__ == "__main__":
    main()

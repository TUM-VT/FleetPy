import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from data_processing.config import DataProcessingConfig as cfg

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_edge_predictions(data, graph_idx, model, device):
    """Get model predictions for a single graph"""
    model.eval()
    with torch.no_grad():
        # Prepare data dictionaries
        graph = data[graph_idx].to(device)

        # Prepare input dictionaries
        x_dict = {}
        edge_index_dict = {}
        edge_attr_dict = {}

        # Get node features
        for node_type in graph.node_types:
            x_dict[node_type] = graph[node_type].x

        # Get edge features
        for edge_type in graph.edge_types:
            edge_index = graph[edge_type].edge_index
            edge_attr = graph[edge_type].edge_attr

            edge_index_dict[edge_type] = edge_index.long()  # Ensure int64
            edge_attr_dict[edge_type] = edge_attr

        try:
            logits = model(x_dict, edge_index_dict, edge_attr_dict)
            return torch.sigmoid(logits).cpu().numpy()
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None


def visualize_graph(data, graph_idx=0, model=None, device='cpu', only_new_requests=False):
    """
    Visualize a heterogeneous graph with each request in a separate subplot,
    showing the top 5 predicted edges and true edges for each request.
    
    Args:
        data: List of HeteroData objects
        graph_idx: Index of the graph to visualize
        model: The trained model to get predictions from
        device: Device to run model predictions on
        only_new_requests: If True, only show requests that didn't appear in previous graphs
    """
    if model is None:
        raise ValueError("Model must be provided for visualization")

    # Get edge predictions
    predictions = get_edge_predictions(data, graph_idx, model, device)
    if predictions is None:
        raise ValueError("Could not get predictions from model")

    # Get the graph
    graph = data[graph_idx]
    # Get edge indices, predictions, and ground truth
    edge_type = ('vehicle', 'connects', 'request')
    edge_index = graph[edge_type].edge_index.cpu()
    true_labels = graph[edge_type].y.cpu()
    
    # Get the index where the predictions for this edge type start
    start_idx = 0
    for et in graph.edge_types:
        if et == edge_type:
            break
        start_idx += graph[et].edge_index.shape[1]
    
    # Create (source, target, prediction, is_true) tuples grouped by request
    edges_by_request = {}
    for i in range(edge_index.shape[1]):
        src = f'v{edge_index[0, i].item()}'
        dst = f'r{edge_index[1, i].item()}'
        score = float(predictions[start_idx + i])  # Offset prediction index by start_idx
        is_true = bool(true_labels[i].item())
        
        if dst not in edges_by_request:
            edges_by_request[dst] = []
        edges_by_request[dst].append((src, dst, score, is_true))
    
    # Filter out requests that are already assigned (only have one true edge)
    active_requests = []
    for request, edges in edges_by_request.items():
        true_edges = [e for e in edges if e[3]]  # Get all true edges
        predicted_edges = [(src, dst, score) for src, dst, score, is_true in edges if score > 0.01]  # Edges with non-zero predictions
        
        # Only include requests that either:
        # 1. Have no true assignments (need prediction)
        # 2. Have true assignments but also have other potential matches (interesting for analysis)
        if len(predicted_edges) > 1:  # More than just a true assignment
            active_requests.append(request)
    
    # Get the set of request node_ids from the previous graph if needed
    if only_new_requests and graph_idx > 0:
        # Get request node_ids from the previous graph
        prev_graph = data[graph_idx - 1]
        prev_request_ids = set(prev_graph['request'].node_ids.cpu().numpy())
        
        # Get current graph's request node_ids
        current_request_ids = graph['request'].node_ids.cpu().numpy()
        # Map from display name (r{idx}) to actual node_id for current requests
        request_to_nodeid = {
            f'r{i}': node_id.item() 
            for i, node_id in enumerate(current_request_ids)
        }
        
        # Filter out requests that appeared in the previous graph
        active_requests = [
            req for req in active_requests 
            if request_to_nodeid[req] not in prev_request_ids
        ]
    
    # Sort the remaining requests by their numerical index
    request_nodes = sorted(active_requests, key=lambda x: int(x[1:]))  # Extract number after 'r' and sort numerically
    if not request_nodes:
        if only_new_requests:
            print("No new requests to visualize in this graph")
        else:
            print("No active requests to visualize (all are already assigned)")
        return
        
    # Calculate grid dimensions for subplots
    num_requests = len(request_nodes)
    num_cols = int(np.ceil(np.sqrt(num_requests)))
    num_rows = int(np.ceil(num_requests / num_cols))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(5*num_cols, 4*num_rows))
    fig.suptitle('Request-Vehicle Assignment Predictions', fontsize=16, y=1.02)
    
    # Create subplots
    for idx, request in enumerate(request_nodes):
        # Create subplot
        ax = plt.subplot(num_rows, num_cols, idx + 1)
        # Get edges for this request and sort by score
        request_edges = edges_by_request[request]
        request_edges.sort(key=lambda x: x[2], reverse=True)
        top_5_edges = request_edges[:5]
        
        # Find the true assignment and best prediction for this request
        true_edge = next((edge for edge in request_edges if edge[3]), None)
        best_edge = top_5_edges[0] if top_5_edges else None
        
        # Create descriptive title based on prediction vs truth
        total_edges = len(request_edges)  # Get total number of edges before top-5 filtering
        title_parts = [f'Request {request[1:]} (Total edges: {total_edges})']
        if best_edge and true_edge:
            if best_edge == true_edge:
                title_parts.append('\nCorrect prediction! ✓')
            else:
                title_parts.append('\nIncorrect prediction ✗')
        elif best_edge:
            title_parts.append('\nPrediction only (no true assignment)')
        elif true_edge:
            title_parts.append('\nTrue assignment only (missed by model)')
        
        ax.set_title('\n'.join(title_parts), pad=10)
        
        # Get connected vehicles for this request (both top 5 and true assignment)
        connected_vehicles = {src for src, _, _, _ in top_5_edges}
        # Add vehicle from true assignment if it exists and isn't already included
        if true_edge:
            true_vehicle = true_edge[0]  # src from the true edge
            connected_vehicles.add(true_vehicle)
        
        # Create a new graph for this subplot
        G = nx.Graph()
        
        # Define positions
        pos = {}
        
        # Position request node on the right
        pos[request] = (1, 0)
        G.add_node(request)
        
        # Position vehicles on the left, sorted by their edge scores
        vehicle_scores = {}
        for src, _, score, _ in request_edges:
            if src in connected_vehicles:
                vehicle_scores[src] = score
                
        # Sort vehicles by their scores in descending order
        vehicles_list = sorted(connected_vehicles, key=lambda v: vehicle_scores[v], reverse=True)
        num_vehicles = len(vehicles_list)
        y_positions = np.linspace(0.5, -0.5, num_vehicles) if num_vehicles > 1 else [0]
        for i, vid in enumerate(vehicles_list):
            pos[vid] = (-1, y_positions[i])
            G.add_node(vid)
            
        # Track if true vehicle is in top 5
        true_vehicle_in_top5 = true_edge and true_edge[0] in {src for src, _, _, _ in top_5_edges}
            
            # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=[request], node_color='lightgreen',
                             node_size=1000, ax=ax)
                             
        # Draw vehicle nodes with different colors
        top5_vehicles = {src for src, _, _, _ in top_5_edges}
        normal_vehicles = [v for v in connected_vehicles if v in top5_vehicles]
        missed_true_vehicle = [true_edge[0]] if true_edge and not true_vehicle_in_top5 else []
        
        # Draw normal vehicles in light blue
        if normal_vehicles:
            nx.draw_networkx_nodes(G, pos, nodelist=normal_vehicles, 
                                 node_color='lightblue', node_size=1000, ax=ax)
        
        # Draw missed true vehicle in orange to highlight it
        if missed_true_vehicle:
            nx.draw_networkx_nodes(G, pos, nodelist=missed_true_vehicle,
                                 node_color='orange', node_size=1000, ax=ax)        # Draw edges with different styles based on prediction rank
        for i, (src, dst, score, is_true) in enumerate(top_5_edges):
            edge_color = 'red' if i == 0 else 'blue'
            edge_style = 'solid'
            edge_width = 3 if i == 0 else 2
            alpha = 0.8 if i == 0 else 0.6
            
            # If it's a true edge, draw it with a different style
            if is_true:
                edge_color = 'green'
                edge_style = 'dashed'
                
            nx.draw_networkx_edges(G, pos, edgelist=[(src, dst)],
                                 edge_color=edge_color, style=edge_style,
                                 width=edge_width, alpha=alpha, ax=ax)
                                 
            # Add edge labels with scores
            label = f'{score:.2f}{"✓" if is_true else ""}'
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(src, dst): label},
                                       font_size=8, ax=ax)
        
        # Add node labels
        labels = {node: node[1:] for node in G.nodes()}  # Remove v/r prefix
        nx.draw_networkx_labels(G, pos, labels, ax=ax)
        
        # Set axis properties
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1, 1])
        ax.axis('off')
        
    # Create a custom legend
    legend_elements = []
    # First add line elements
    legend_elements.extend([
        plt.Line2D([0], [0], color='red', lw=3, label='Best Match'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Other Matches'),
        plt.Line2D([0], [0], color='green', ls='--', lw=2, label='True Assignment'),
        plt.scatter([0], [0], c='lightblue', s=100, label='Vehicle (in top 5)'),
        plt.scatter([0], [0], c='orange', s=100, label='Vehicle (true but not in top 5)'),
        plt.scatter([0], [0], c='lightgreen', s=100, label='Request')
    ])
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0),
               ncol=5, borderaxespad=3)
    
    plt.tight_layout()
    plt.show()
    
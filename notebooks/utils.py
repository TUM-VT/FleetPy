import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import torch


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
            print(f"Error during prediction: {str(e)}")
            return None


def visualize_graph(data, graph_idx=0, model=None, device=None):
    """Visualize a heterogeneous graph with vehicle-request assignments"""
    G = nx.DiGraph()  # Use directed graph

    # Color scheme
    colors = {
        'vehicle': '#FF6B6B',    # Red for vehicles
        'request': '#4ECDC4',    # Turquoise for requests
    }

    # Get predictions if model is provided
    predictions = None
    if model is not None and device is not None:
        predictions = get_edge_predictions(data, graph_idx, model, device)

    # First, collect all connected node indices for each type
    connected_nodes = {'vehicle': set(), 'request': set()}
    graph = data[graph_idx]
    for edge_type in graph.edge_types:
        src_type, rel_type, dst_type = edge_type
        edge_index = graph[edge_type].edge_index
        for i in range(edge_index.size(1)):
            connected_nodes[src_type].add(edge_index[0][i].item())
            connected_nodes[dst_type].add(edge_index[1][i].item())

    # Add only connected nodes
    node_colors = []
    node_sizes = []
    node_labels = {}
    for node_type in ['vehicle', 'request']:
        if node_type in graph.node_types:
            nodes = graph[node_type]
            for i in range(nodes.x.size(0)):
                if i in connected_nodes[node_type]:
                    node_name = f'{node_type}_{i}'
                    G.add_node(node_name, type=node_type)
                    node_colors.append(colors[node_type])
                    node_sizes.append(400 if node_type == 'vehicle' else 300)
                    node_labels[node_name] = f'{node_type[0].upper()}{i}'

    # Process edges and their attributes
    edge_count = 0
    true_edges = []
    true_colors = []
    true_styles = []
    pred_edges = []
    pred_labels = {}

    for edge_type in graph.edge_types:
        src_type, rel_type, dst_type = edge_type
        if rel_type == 'rev_connects':
            continue
        edge_index = graph[edge_type].edge_index
        edge_y = graph[edge_type].y

        for i in range(edge_index.size(1)):
            src = f'{src_type}_{edge_index[0][i].item()}'
            dst = f'{dst_type}_{edge_index[1][i].item()}'
            G.add_edge(src, dst)

            # Add ground truth
            if edge_y is not None and edge_y[i].item() == 1:
                true_edges.append((src, dst))
                true_colors.append('#FF9999')  # Light red for assignments
                true_styles.append('solid')
            
            # Add predictions
            if predictions is not None and edge_count < len(predictions):
                prob = predictions[edge_count][0]
                pred_edges.append((src, dst))
                pred_labels[(src, dst)] = f'{prob:.2f}'
                edge_count += 1

    # Create plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=10)

    # Draw ground truth edges
    for (u, v), color, style in zip(true_edges, true_colors, true_styles):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color,
                               style=style, alpha=0.8, arrows=True, arrowstyle='-|>', arrowsize=20)

    # Annotate all edges with predicted probabilities (if available)
    if predictions is not None:
        edge_offset = 0.03  # Use same offset as blue line
        for edge in pred_edges:
            u, v = edge
            pos_u, pos_v = np.array(pos[u]), np.array(pos[v])
            dx = pos_v[0] - pos_u[0]
            dy = pos_v[1] - pos_u[1]
            length = np.sqrt(dx*dx + dy*dy)
            perpx, perpy = -dy/length, dx/length
            # Offset the label position
            mid_x = (pos_u[0] + pos_v[0]) / 2 + perpx * edge_offset
            mid_y = (pos_u[1] + pos_v[1]) / 2 + perpy * edge_offset
            if edge in pred_labels:
                plt.annotate(pred_labels[edge],
                             (mid_x, mid_y),
                             bbox=dict(facecolor='white',
                                       edgecolor='none', alpha=0.7),
                             fontsize=8)

    # Draw predicted edges with an offset (blue lines) if desired
    if predictions is not None:
        edge_offset = 0.03  # Adjust this value to control separation
        for edge in pred_edges:
            u, v = edge
            prob = float(pred_labels[edge]) if edge in pred_labels else 0.0
            if prob > 0.5:  # Change threshold as needed
                pos_u, pos_v = np.array(pos[u]), np.array(pos[v])
                dx = pos_v[0] - pos_u[0]
                dy = pos_v[1] - pos_u[1]
                length = np.sqrt(dx*dx + dy*dy)
                perpx, perpy = -dy/length, dx/length
                new_pos_u = pos_u + np.array([perpx, perpy]) * edge_offset
                new_pos_v = pos_v + np.array([perpx, perpy]) * edge_offset
                plt.annotate('', xy=new_pos_v, xytext=new_pos_u,
                             arrowprops=dict(arrowstyle='-|>', color='#0000FF', lw=2, alpha=0.6), zorder=1)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Vehicle',
                   markerfacecolor=colors['vehicle'], markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Request',
                   markerfacecolor=colors['request'], markersize=10),
        plt.Line2D([0], [0], color='#FF9999', label='True Assignment', linestyle='solid',
                   alpha=0.8),
    ]

    if predictions is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color='#0000FF', label='Model Prediction',
                       linestyle='--', alpha=0.6, linewidth=2)
        )

    plt.legend(handles=legend_elements,
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Vehicle-Request Assignment Graph')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

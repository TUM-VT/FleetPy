import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_graph(data, graph_idx=0, predictions=None, device='cpu', only_new_requests=False):
    """
    Visualize a heterogeneous graph with each request in a separate subplot,
    showing the top 5 predicted edges and true edges for each request.

    Args:
        data: List of HeteroData objects
        graph_idx: Index of the graph to visualize
        predictions: (Optional) Precomputed edge predictions
        device: Device to run model predictions on
        only_new_requests: If True, only show requests that didn't appear in previous graphs
    """
    # Get the graph
    graph = data[graph_idx]
    # Get edge indices, predictions, and ground truth
    # TODO make it work also for request-request edges
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
        # Offset prediction index by start_idx
        score = float(predictions[start_idx + i])
        is_true = bool(true_labels[i].item())

        if dst not in edges_by_request:
            edges_by_request[dst] = []
        edges_by_request[dst].append((src, dst, score, is_true))

    # Filter out requests that are already assigned (only have one true edge)
    active_requests = []
    for request, edges in edges_by_request.items():
        true_edges = [e for e in edges if e[3]]  # Get all true edges
        # Edges with non-zero predictions
        predicted_edges = [(src, dst, score)
                           for src, dst, score, is_true in edges if score > 0.01]

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
    # Extract number after 'r' and sort numerically
    request_nodes = sorted(active_requests, key=lambda x: int(x[1:]))
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
    fig.suptitle('Request-Vehicle Assignment Predictions', fontsize=24, y=1.02)

    # Create subplots
    for idx, request in enumerate(request_nodes):
        # Create subplot
        ax = plt.subplot(num_rows, num_cols, idx + 1)
        plt.rcParams.update({'font.size': 14})  # Increase default font size
        # Get edges for this request and sort by score
        request_edges = edges_by_request[request]
        request_edges.sort(key=lambda x: x[2], reverse=True)
        top_5_edges = request_edges[:5]

        # Find the true assignment and best prediction for this request
        true_edge = next((edge for edge in request_edges if edge[3]), None)
        best_edge = top_5_edges[0] if top_5_edges else None

        # Create descriptive title based on prediction vs truth
        # Get total number of edges before top-5 filtering
        total_edges = len(request_edges)
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

        ax.set_title('\n'.join(title_parts), pad=10, fontsize=16)

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
        vehicles_list = sorted(
            connected_vehicles, key=lambda v: vehicle_scores[v], reverse=True)
        num_vehicles = len(vehicles_list)
        y_positions = np.linspace(
            0.5, -0.5, num_vehicles) if num_vehicles > 1 else [0]
        for i, vid in enumerate(vehicles_list):
            pos[vid] = (-1, y_positions[i])
            G.add_node(vid)

        # Track if true vehicle is in top 5
        true_vehicle_in_top5 = true_edge and true_edge[0] in {
            src for src, _, _, _ in top_5_edges}

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=[request], node_color='lightgreen',
                               node_size=1000, ax=ax)

        # Draw vehicle nodes with different colors
        top5_vehicles = {src for src, _, _, _ in top_5_edges}
        normal_vehicles = [v for v in connected_vehicles if v in top5_vehicles]
        missed_true_vehicle = [
            true_edge[0]] if true_edge and not true_vehicle_in_top5 else []

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
            print(
                f'Edge from {src} to {dst}: score={score:.2f}, is_true={is_true}')
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(src, dst): label},
                                         font_size=12, ax=ax)

        # Add node labels
        labels = {node: node[1:] for node in G.nodes()}  # Remove v/r prefix
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=14)

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
        plt.Line2D([0], [0], color='green', ls='--',
                   lw=2, label='True Assignment'),
        plt.scatter([0], [0], c='lightblue', s=100,
                    label='Vehicle (in top 5)'),
        plt.scatter([0], [0], c='orange', s=100,
                    label='Vehicle (true but not in top 5)'),
        plt.scatter([0], [0], c='lightgreen', s=100, label='Request')
    ])
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0),
               ncol=5, borderaxespad=3, fontsize=14)

    plt.tight_layout()
    plt.show()


def analyze_model_performance(data, start_idx, num_graphs, predictions, device):
    """
    Analyze model performance over a range of graphs, focusing on new requests.

    Args:
        data: List of HeteroData objects
        start_idx: Starting graph index
        num_graphs: Number of graphs to analyze
        predictions: List of predictions for each graph
        device: Device to run model on

    Returns:
        fig: matplotlib figure showing performance metrics
        data_dict: (optional) Dictionary with scores and true assignments
    """
    performance_data = []
    edge_type = ('vehicle', 'connects', 'request')

    for graph_idx in range(start_idx, min(start_idx + num_graphs, len(data))):
        # Get predictions
        graph = data[graph_idx]
        predictions_for_graph = predictions[graph_idx - start_idx]

        edge_index = graph[edge_type].edge_index.cpu()
        true_labels = graph[edge_type].y.cpu()

        # Get start index for this edge type's predictions
        start_pred_idx = 0
        for et in graph.edge_types:
            if et == edge_type:
                break
            start_pred_idx += graph[et].edge_index.shape[1]

        # Get new requests by comparing with previous graph
        if graph_idx > 0:
            prev_graph = data[graph_idx - 1]
            prev_request_ids = set(
                prev_graph['request'].node_ids.cpu().numpy())
            current_request_ids = graph['request'].node_ids.cpu().numpy()
            new_request_mask = [
                id.item() not in prev_request_ids for id in current_request_ids]
        else:
            new_request_mask = [True] * len(graph['request'].node_ids)

        # Create dictionary mapping request index to its new/old status
        request_is_new = {i: new_request_mask[i]
                          for i in range(len(new_request_mask))}

        # Group edges by request
        edges_by_request = {}
        for i in range(edge_index.shape[1]):
            req_idx = edge_index[1, i].item()
            if req_idx not in edges_by_request:
                edges_by_request[req_idx] = []
            edges_by_request[req_idx].append({
                'score': float(predictions_for_graph[start_pred_idx + i]),
                'is_true': bool(true_labels[i].item())
            })

        # Analyze each new request
        for req_idx, edges in edges_by_request.items():
            if not request_is_new[req_idx]:
                continue

            # Sort edges by score
            edges.sort(key=lambda x: x['score'], reverse=True)

            # Find rank of true assignment
            true_rank = None
            is_correct = False
            for rank, edge in enumerate(edges):
                if edge['is_true']:
                    true_rank = rank + 1
                    # correct if true edge has highest score
                    is_correct = (rank == 0)
                    break

            performance_data.append({
                'graph_idx': graph_idx,
                'request_idx': req_idx,
                'correct': is_correct,
                'true_rank': true_rank
            })

    # Create visualization
    if not performance_data:
        print("No new requests found in the specified range")
        return None

    df = pd.DataFrame(performance_data)

    # Calculate different Top-K accuracies (extended range for detailed analysis)
    K_values = [1, 3, 5]  # Different thresholds for Top-K in time series
    # Extended range for detailed Top-K analysis
    K_values_detailed = list(range(1, 11))
    top_k_accuracies = {}

    # Calculate accuracies for time series plot
    for k in K_values:
        df[f'top_{k}'] = df.apply(
            lambda row: row['true_rank'] <= k if row['true_rank'] is not None else False, axis=1)

    # Calculate accuracies for detailed Top-K analysis
    detailed_accuracies = []
    for k in K_values_detailed:
        accuracy = df.apply(
            lambda row: row['true_rank'] <= k if row['true_rank'] is not None else False, axis=1).mean() * 100
        detailed_accuracies.append(accuracy)

    # Define colors for consistent visualization
    colors = ['#2ecc71', '#3498db', '#9b59b6']  # Green, Blue, Purple

    # Calculate basic statistics
    total_requests = len(df)

    # Calculate rolling averages for smoother visualization
    df['relative_idx'] = df['graph_idx'] - start_idx
    window_accuracies = {}
    window_size = 20  # 10-minute rolling window (20 graphs * 30s = 10 minutes)

    # Group by graph index and calculate rolling means
    for k in K_values:
        # Group by graph index first
        grouped = df.groupby('relative_idx')[f'top_{k}'].mean()
        # Calculate rolling average
        window_accuracies[k] = grouped.rolling(
            window=window_size, center=True, min_periods=1).mean()

    # Set larger font sizes for all plots
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # Create the combined plot for display
    fig = plt.figure(figsize=(16, 12))

    # Create subplot for accuracy over time plot
    ax1 = plt.subplot(141)

    # Plot line for each K value
    colors = ['#2ecc71', '#3498db', '#9b59b6']  # Green, Blue, Purple
    lines = []
    for k, color in zip(K_values, colors):
        # Convert graph indices to minutes (30 seconds per graph = 0.5 minutes)
        time_in_minutes = window_accuracies[k].index * 0.5

        # Plot rolling average line
        line = ax1.plot(time_in_minutes, window_accuracies[k].values * 100,
                        color=color, label=f'Top-{k}', linewidth=2)
        lines.append(line[0])

    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Overall Prediction Accuracy\n(10-minute rolling average)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Set y-axis limits with some padding
    ax1.set_ylim(0, 100)

    # Format x-axis ticks
    # Major ticks every 15 minutes
    ax1.xaxis.set_major_locator(plt.MultipleLocator(15))
    # Add minor ticks for better visual reference
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax1.grid(True, which='major', alpha=0.3)
    ax1.grid(True, which='minor', alpha=0.1)

    # Add total request count as text
    total_requests = len(df)
    ax1.text(0.02, 0.98, f'Total Requests: {total_requests}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=14)

    # Create subplot for Top-K analysis
    ax2 = plt.subplot(222)

    # Plot detailed Top-K accuracies
    ax2.plot(K_values_detailed, detailed_accuracies, marker='o', color='#3498db',
             linewidth=2, markersize=8, markerfacecolor='white')

    # Add value labels above each point
    for k, acc in zip(K_values_detailed, detailed_accuracies):
        ax2.text(k, acc + 2, f'{acc:.1f}%',
                 ha='center', va='bottom', fontsize=12)

    ax2.set_xlabel('K')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(
        'How Often the True Assignment\nAppears Among Top-K Predictions')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(K_values_detailed)
    ax2.set_ylim(0, 110)

    # Create subplot for score distribution
    ax3 = plt.subplot(223)

    # Collect scores for true and false assignments (only for new requests)
    all_scores = []
    is_true_edge_list = []  # Will store true/false status for each edge
    true_scores_hist = []  # Will store scores of true assignments for histogram
    false_scores = []  # Will store scores of false assignments for histogram

    for graph_idx in range(start_idx, min(start_idx + num_graphs, len(data))):
        graph = data[graph_idx]
        predictions_for_graph = predictions[graph_idx - start_idx]
        edge_index = graph[edge_type].edge_index.cpu()
        true_labels = graph[edge_type].y.cpu()

        # Get new requests by comparing with previous graph
        if graph_idx > 0:
            prev_graph = data[graph_idx - 1]
            prev_request_ids = set(
                prev_graph['request'].node_ids.cpu().numpy())
            current_request_ids = graph['request'].node_ids.cpu().numpy()
            new_request_mask = [
                id.item() not in prev_request_ids for id in current_request_ids]
        else:
            new_request_mask = [True] * len(graph['request'].node_ids)

        # Create set of new request indices
        new_requests = {i for i, is_new in enumerate(
            new_request_mask) if is_new}

        # Get start index for this edge type's predictions
        start_pred_idx = 0
        for et in graph.edge_types:
            if et == edge_type:
                break
            start_pred_idx += graph[et].edge_index.shape[1]

        # Collect scores and their true/false status (only for new requests)
        for i in range(edge_index.shape[1]):
            req_idx = edge_index[1, i].item()
            if req_idx in new_requests:  # Only include edges for new requests
                score = float(predictions_for_graph[start_pred_idx + i])
                is_true = bool(true_labels[i].item())
                all_scores.append(score)
                # Track true/false status directly
                is_true_edge_list.append(is_true)
                if is_true:
                    true_scores_hist.append(score)  # Store score for histogram
                else:
                    # Store score for histogram    # Create histogram (note the label parameter is moved inside hist calls)
                    false_scores.append(score)
    bins = np.linspace(0, 1, 30)
    # Plot each histogram separately to ensure proper labels
    ax3.hist(false_scores, bins=bins, label='Non-assigned',
             color='#e74c3c', alpha=0.7)
    ax3.hist(true_scores_hist, bins=bins, label='True assignments',
             color='#2ecc71', alpha=0.7)

    # Add statistical information
    true_mean = np.mean(true_scores_hist) if true_scores_hist else 0
    false_mean = np.mean(false_scores) if false_scores else 0
    stats_text = f'Mean Scores:\nTrue Assignments: {true_mean:.3f}\nNon-assigned: {false_mean:.3f}'
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round',
                                                facecolor='white', alpha=0.8), fontsize=14)

    ax3.set_xlabel('Predicted Score')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Predicted Scores\n(New Requests Only)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Adjust layout with specific spacing
    # Create subplot for cumulative coverage plot
    ax4 = plt.subplot(224)

    # Convert list to numpy array
    is_true_edge = np.array(is_true_edge_list)

    # Sort scores and maintain true/false labels
    sorted_indices = np.argsort(all_scores)[::-1]  # Sort in descending order
    sorted_scores = np.array(all_scores)[sorted_indices]
    # Sort the true/false labels in same order
    is_true_edge = is_true_edge[sorted_indices]

    # Calculate cumulative coverage
    total_true = np.sum(is_true_edge)
    true_found = np.cumsum(is_true_edge)
    edge_fractions = np.arange(1, len(all_scores) + 1) / len(all_scores) * 100
    coverage = true_found / total_true * 100

    # Plot cumulative curve
    ax4.plot(edge_fractions, coverage, 'b-', linewidth=2)

    # Add diagonal baseline
    ax4.plot([0, 100], [0, 100], '--', color='grey', alpha=0.5, label='Random')

    # Add points at interesting fractions (10%, 20%, 50%)
    interesting_fractions = [10, 20, 50]
    for frac in interesting_fractions:
        idx = np.searchsorted(edge_fractions, frac)
        if idx < len(coverage):
            ax4.plot(frac, coverage[idx], 'bo',
                     markersize=8, markerfacecolor='white')

    # Find point where we achieve 90% coverage
    target_coverage = 90
    idx_90 = np.searchsorted(coverage, target_coverage)
    if idx_90 < len(edge_fractions):
        edges_needed = edge_fractions[idx_90]
        ax4.annotate(f'Top-{edges_needed:.1f}% edges → {target_coverage}% recall',
                     xy=(edges_needed, target_coverage),
                     xytext=(edges_needed + 10, target_coverage - 20),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax4.set_xlabel('Fraction of Edges Considered (%)')
    ax4.set_ylabel('True Assignments Found (%)')
    ax4.set_title('Cumulative Edge Selection Efficiency')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.subplots_adjust(left=0.1, right=0.95, wspace=0.3)

    return fig

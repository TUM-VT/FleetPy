import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_request_first_seen_mapping(data, start_idx, end_idx):
    """
    Helper function to build a mapping of request node IDs to their first appearance index.
    
    Args:
        data: List of HeteroData objects
        start_idx: Starting graph index (inclusive)
        end_idx: Ending graph index (inclusive)
    
    Returns:
        dict: Mapping from request node_id to the graph index where it first appears
    """
    req_first_seen = {}
    for idx, g in enumerate(data[start_idx:end_idx+1], start=start_idx):
        for node_id in g['request'].node_ids.cpu().numpy():
            if node_id not in req_first_seen:
                req_first_seen[node_id] = idx
    return req_first_seen


def visualize_graph(data, graph_idx, predictions, edge_types=None, scenario_start_idx=0):
    """
    Visualize a heterogeneous graph with each request in a separate subplot,
    showing the top 5 predicted edges and true edges for each request.
    Only shows new requests (first appearance).

    Args:
        data: List of HeteroData objects
        graph_idx: Index of the graph to visualize
        predictions: (Optional) Precomputed edge predictions
        edge_types: List of edge type tuples to visualize. If None, uses all edge types in the graph.
        scenario_start_idx: Index of the first graph in the current scenario (default: 0).
                           Used to count new requests only within the current scenario.
    """
    # Build request ID to first appearance mapping (only within current scenario)
    req_first_seen = _build_request_first_seen_mapping(data, scenario_start_idx, graph_idx)
    
    # Get the graph
    graph = data[graph_idx]
    
    # Auto-detect edge types if not provided
    if edge_types is None:
        edge_types = graph.edge_types
    elif not isinstance(edge_types, list):
        edge_types = [edge_types]
    
    # Collect edges from all edge types
    all_edges_by_request = {}
    
    for edge_type_tuple in edge_types:
        edge_index = graph[edge_type_tuple].edge_index.cpu()
        true_labels = graph[edge_type_tuple].y.cpu()
        
        # Determine edge category for visualization
        if edge_type_tuple[0] == 'vehicle':
            edge_category = 'VR'
        elif edge_type_tuple[0] == 'request' and edge_type_tuple[2] == 'request':
            edge_category = 'RR'
        else:
            edge_category = 'OTHER'

        # Get predictions for this edge type from the dictionary
        edge_type_predictions = predictions[edge_type_tuple]

        # Create (source, target, prediction, is_true, edge_category) tuples grouped by target request
        for i in range(edge_index.shape[1]):
            if edge_category == 'VR':
                src = f'v{edge_index[0, i].item()}'
                dst = f'r{edge_index[1, i].item()}'
            else:  # RR or OTHER
                src = f'r{edge_index[0, i].item()}'
                dst = f'r{edge_index[1, i].item()}'
            
            # Get prediction score for this edge from the edge type's predictions
            score = float(edge_type_predictions[i])
            is_true = bool(true_labels[i].item())

            if dst not in all_edges_by_request:
                all_edges_by_request[dst] = []
            all_edges_by_request[dst].append((src, dst, score, is_true, edge_category))

    # Filter for new requests only and build mapping with global count
    new_requests = []
    new_request_node_ids = []
    for request, edges in all_edges_by_request.items():
        # Get request index and node_id
        req_idx = int(request[1:])
        req_node_id = graph['request'].node_ids[req_idx].item()
        
        # Only show new requests
        first_seen_idx = req_first_seen.get(req_node_id, graph_idx)
        if first_seen_idx == graph_idx:  # Is new
            new_requests.append(request)
            new_request_node_ids.append(req_node_id)

    # Sort by positional index (order in the graph, likely insertion order)
    request_nodes = sorted(new_requests, key=lambda x: int(x[1:]))
    if not request_nodes:
        print("No new requests to visualize in this graph")
        return

    # Calculate grid dimensions for subplots
    num_requests = len(request_nodes)
    num_cols = min(2, num_requests)  # Limit to 2 columns for better readability
    num_rows = int(np.ceil(num_requests / num_cols))

    # Create figure with subplots - larger size for clarity
    fig = plt.figure(figsize=(12*num_cols, 8*num_rows))
    # Determine if we have multiple edge types
    has_vr = any(e[4] == 'VR' for edges in all_edges_by_request.values() for e in edges)
    has_rr = any(e[4] == 'RR' for edges in all_edges_by_request.values() for e in edges)
    if has_vr and has_rr:
        title = 'Combined Vehicle-Request and Request-Request Predictions'
    elif has_vr:
        title = 'Vehicle-Request Assignment Predictions'
    else:
        title = 'Request-Request Ridesharing Predictions'
    fig.suptitle(title, fontsize=24, y=1.02)

    # Create subplots
    for idx, request in enumerate(request_nodes):
        # Create subplot
        ax = plt.subplot(num_rows, num_cols, idx + 1)
        plt.rcParams.update({'font.size': 16})  # Increase default font size
        # Get edges for this request and sort by score
        request_edges = all_edges_by_request[request]
        request_edges.sort(key=lambda x: x[2], reverse=True)
        top_5_edges = request_edges[:5]

        # Find the true assignment and best prediction for this request
        true_edge = next((edge for edge in request_edges if edge[3]), None)
        best_edge = top_5_edges[0] if top_5_edges else None

        # Create descriptive title based on prediction vs truth
        # Get total number of edges before top-5 filtering
        total_edges = len(request_edges)
        req_idx = int(request[1:])
        req_node_id = graph['request'].node_ids[req_idx].item()
        
        # Calculate global position within scenario: count how many new requests appeared before this one
        title_parts = [f'Request {req_node_id} ({total_edges} edges)']
        if true_edge is None:
            # No true assignment exists (unpaired request)
            title_parts.append('\n⚠ No true assignment (unpaired)')
            if best_edge:
                title_parts.append(f'Model prediction: {best_edge[2]:.3f}')
        elif best_edge and true_edge:
            if best_edge == true_edge:
                title_parts.append('\nCorrect prediction! ✓')
            else:
                title_parts.append('\nIncorrect prediction ✗')
        elif best_edge:
            title_parts.append('\nPrediction only (no true assignment)')
        elif true_edge:
            title_parts.append('\nTrue assignment only (missed by model)')

        ax.set_title('\n'.join(title_parts), pad=15, fontsize=18)

        # Get connected sources for this request (both top 5 and true assignment)
        connected_sources = {src for src, _, _, _, _ in top_5_edges}
        # Add source from true assignment if it exists and isn't already included
        if true_edge is not None:
            true_source = true_edge[0]  # src from the true edge
            connected_sources.add(true_source)
        
        # Handle case where there are no edges at all
        if not connected_sources:
            # Just show the target request node with a message
            ax.text(0, 0, 'No candidate edges', ha='center', va='center', fontsize=18)
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1, 1])
            ax.axis('off')
            continue

        # Create a new graph for this subplot
        G = nx.Graph()

        # Define positions
        pos = {}

        # Position target request node on the right
        pos[request] = (1, 0)
        G.add_node(request)

        # Position sources on the left, sorted by their edge scores
        source_scores = {}
        for src, _, score, _, _ in request_edges:
            if src in connected_sources:
                source_scores[src] = score

        # Sort sources by their scores in descending order
        sources_list = sorted(
            connected_sources, key=lambda v: source_scores[v], reverse=True)
        num_sources = len(sources_list)
        # Increase vertical spacing between source nodes
        y_span = min(1.0, num_sources * 0.25)  # More space per node
        y_positions = np.linspace(
            y_span, -y_span, num_sources) if num_sources > 1 else [0]
        for i, sid in enumerate(sources_list):
            pos[sid] = (-1.2, y_positions[i])  # Move further left
            G.add_node(sid)

        # Track if true source is in top 5
        true_source_in_top5 = (true_edge is not None) and (true_edge[0] in {
            src for src, _, _, _, _ in top_5_edges})

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=[request], node_color='lightgreen',
                               node_size=2000, ax=ax)

        # Draw source nodes with different colors based on edge type
        top5_sources = {src for src, _, _, _, _ in top_5_edges}
        
        # Separate sources by edge type
        vr_sources = []
        rr_sources = []
        for src in connected_sources:
            if src in top5_sources:
                # Check which edge type this source belongs to
                for edge_src, edge_dst, _, _, edge_cat in top_5_edges:
                    if edge_src == src:
                        if edge_cat == 'VR':
                            vr_sources.append(src)
                        else:
                            rr_sources.append(src)
                        break
        
        missed_true_source = [
            true_edge[0]] if (true_edge is not None) and (not true_source_in_top5) else []

        # Draw VR sources in light blue
        if vr_sources:
            nx.draw_networkx_nodes(G, pos, nodelist=vr_sources,
                                   node_color='lightblue', node_size=2000, ax=ax)
        
        # Draw RR sources in light coral
        if rr_sources:
            nx.draw_networkx_nodes(G, pos, nodelist=rr_sources,
                                   node_color='lightcoral', node_size=2000, ax=ax)

        # Draw missed true source in orange to highlight it
        if missed_true_source:
            nx.draw_networkx_nodes(G, pos, nodelist=missed_true_source,
                                   node_color='orange', node_size=2000, ax=ax)        # Draw edges with different styles based on prediction rank
        for i, (src, dst, score, is_true, edge_cat) in enumerate(top_5_edges):
            edge_color = 'red' if i == 0 else 'blue'
            edge_style = 'solid'
            edge_width = 4 if i == 0 else 3
            alpha = 0.9 if i == 0 else 0.7

            # If it's a true edge, draw it with a different style
            if is_true:
                edge_color = 'green'
                edge_style = 'dashed'

            nx.draw_networkx_edges(G, pos, edgelist=[(src, dst)],
                                   edge_color=edge_color, style=edge_style,
                                   width=edge_width, alpha=alpha, ax=ax)

            # Add edge labels with scores and edge type indicator
            label = f'{score:.2f}{"✓" if is_true else ""}'
            print(
                f'Edge from {src} to {dst}: score={score:.2f}, is_true={is_true}, type={edge_cat}')
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(src, dst): label},
                                         font_size=15, ax=ax)

        # Add node labels
        labels = {node: node[1:] for node in G.nodes()}  # Remove v/r prefix
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=18)

        # Set axis properties with more space
        ax.set_xlim([-2.0, 1.8])
        ax.set_ylim([-1.5, 1.5])
        ax.axis('off')

    # Create a custom legend
    legend_elements = []
    # Always include both types if we have mixed edges
    if has_vr and has_rr:
        legend_elements.extend([
            plt.Line2D([0], [0], color='red', lw=3, label='Best Match'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Other Matches'),
            plt.Line2D([0], [0], color='green', ls='--',
                       lw=2, label='True Assignment'),
            plt.scatter([0], [0], c='lightblue', s=100, label='Vehicle (V)'),
            plt.scatter([0], [0], c='lightcoral', s=100, label='Request (R)'),
            plt.scatter([0], [0], c='orange', s=100, label='Missed True'),
            plt.scatter([0], [0], c='lightgreen', s=100, label='Target Request')
        ])
    elif has_vr:
        legend_elements.extend([
            plt.Line2D([0], [0], color='red', lw=3, label='Best Match'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Other Matches'),
            plt.Line2D([0], [0], color='green', ls='--',
                       lw=2, label='True Assignment'),
            plt.scatter([0], [0], c='lightblue', s=100, label='Vehicle (in top 5)'),
            plt.scatter([0], [0], c='orange', s=100, label='Vehicle (missed true)'),
            plt.scatter([0], [0], c='lightgreen', s=100, label='Target Request')
        ])
    else:  # RR
        legend_elements.extend([
            plt.Line2D([0], [0], color='red', lw=3, label='Best Match'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Other Matches'),
            plt.Line2D([0], [0], color='green', ls='--',
                       lw=2, label='True Assignment'),
            plt.scatter([0], [0], c='lightcoral', s=100, label='Request (in top 5)'),
            plt.scatter([0], [0], c='orange', s=100, label='Request (missed true)'),
            plt.scatter([0], [0], c='lightgreen', s=100, label='Target Request')
        ])
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(legend_elements), 4), borderaxespad=4, fontsize=16)

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Leave space for legend and title
    plt.show()


def analyze_model_performance(data, start_idx, num_graphs, predictions, edge_types=None, scenario_start_idx=None):
    """
    Analyze model performance over a range of graphs, focusing on new requests.

    Args:
        data: List of HeteroData objects
        start_idx: Starting graph index
        num_graphs: Number of graphs to analyze
        predictions: List of predictions for each graph
        edge_types: List of edge type tuples to analyze. If None, uses all edge types in the graph.
        scenario_start_idx: Index of the first graph in the current scenario (default: None).
                           If None, uses start_idx. Used to determine new requests within scenario.

    Returns:
        fig: matplotlib figure showing performance metrics
        data_dict: (optional) Dictionary with scores and true assignments
    """
    # Default scenario_start_idx to start_idx if not provided
    if scenario_start_idx is None:
        scenario_start_idx = start_idx
    
    # Build request ID to first appearance mapping for the scenario
    end_idx = min(start_idx + num_graphs - 1, len(data) - 1)
    req_first_seen = _build_request_first_seen_mapping(data, scenario_start_idx, end_idx)
    
    # Auto-detect edge types if not provided
    if edge_types is None:
        edge_types = data[start_idx].edge_types
    elif not isinstance(edge_types, list):
        edge_types = [edge_types]
    
    # Store performance data for each edge type
    performance_data_by_type = {et: [] for et in edge_types}
    
    for edge_type_tuple in edge_types:
        performance_data = []

        for graph_idx in range(start_idx, min(start_idx + num_graphs, len(data))):
            # Get predictions
            graph = data[graph_idx]
            predictions_for_graph = predictions[graph_idx - start_idx]

            edge_index = graph[edge_type_tuple].edge_index.cpu()
            true_labels = graph[edge_type_tuple].y.cpu()

            # Get predictions for this specific edge type from the dictionary
            edge_type_predictions = predictions_for_graph[edge_type_tuple]

            # Identify new requests using the req_first_seen mapping
            new_request_mask = []
            for i, node_id in enumerate(graph['request'].node_ids.cpu().numpy()):
                first_seen_idx = req_first_seen.get(node_id, graph_idx)
                is_new = (first_seen_idx == graph_idx)
                new_request_mask.append(is_new)

            # Create dictionary mapping request index to its new/old status
            request_is_new = {i: new_request_mask[i]
                              for i in range(len(new_request_mask))}

            # Group edges by target request (destination for both VR and RR)
            edges_by_request = {}
            for i in range(edge_index.shape[1]):
                req_idx = edge_index[1, i].item()  # target request
                if req_idx not in edges_by_request:
                    edges_by_request[req_idx] = []
                edges_by_request[req_idx].append({
                    'score': float(edge_type_predictions[i]),
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
                has_true_assignment = False
                for rank, edge in enumerate(edges):
                    if edge['is_true']:
                        has_true_assignment = True
                        true_rank = rank + 1
                        # correct if true edge has highest score
                        is_correct = (rank == 0)
                        break

                # Only include in performance data if there's a true assignment
                # (unpaired requests don't have ground truth to evaluate against)
                if has_true_assignment:
                    performance_data.append({
                        'graph_idx': graph_idx,
                        'request_idx': req_idx,
                        'correct': is_correct,
                        'true_rank': true_rank
                    })
        
        performance_data_by_type[edge_type_tuple] = performance_data
    
    # Combine all performance data
    all_performance_data = []
    for perf_data in performance_data_by_type.values():
        all_performance_data.extend(perf_data)

    # Create visualization
    if not all_performance_data:
        print("No new requests found in the specified range")
        return None

    df = pd.DataFrame(all_performance_data)

    # Calculate different Top-K accuracies (extended range for detailed analysis)
    K_values = [1, 3, 5]  # Different thresholds for Top-K in time series
    # Extended range for detailed Top-K analysis
    K_values_detailed = list(range(1, 15))
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

    # Create separate figures for each plot
    figures = []
    
    # Figure 1: Accuracy over time plot
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)

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
    
    plt.tight_layout()
    figures.append(fig1)

    # Figure 2: Top-K analysis
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)

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
    
    plt.tight_layout()
    figures.append(fig2)

    # Figure 3: Score distribution
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)

    # Collect scores for true and false assignments (only for new requests)
    all_scores = []
    is_true_edge_list = []  # Will store true/false status for each edge
    true_scores_hist = []  # Will store scores of true assignments for histogram
    false_scores = []  # Will store scores of false assignments for histogram

    for edge_type_tuple in edge_types:
        for graph_idx in range(start_idx, min(start_idx + num_graphs, len(data))):
            graph = data[graph_idx]
            predictions_for_graph = predictions[graph_idx - start_idx]
            edge_index = graph[edge_type_tuple].edge_index.cpu()
            true_labels = graph[edge_type_tuple].y.cpu()

            # Identify new requests using the req_first_seen mapping
            new_request_mask = []
            for i, node_id in enumerate(graph['request'].node_ids.cpu().numpy()):
                first_seen_idx = req_first_seen.get(node_id, graph_idx)
                is_new = (first_seen_idx == graph_idx)
                new_request_mask.append(is_new)

            # Create set of new request indices
            new_requests = {i for i, is_new in enumerate(new_request_mask) if is_new}

            # Get predictions for this specific edge type from the dictionary
            edge_type_predictions = predictions_for_graph[edge_type_tuple]

            # Collect scores and their true/false status (only for new requests)
            for i in range(edge_index.shape[1]):
                req_idx = edge_index[1, i].item()  # target request
                if req_idx in new_requests:  # Only include edges for new requests
                    score = float(edge_type_predictions[i])
                    is_true = bool(true_labels[i].item())
                    all_scores.append(score)
                    # Track true/false status directly
                    is_true_edge_list.append(is_true)
                    if is_true:
                        true_scores_hist.append(score)  # Store score for histogram
                    else:
                        false_scores.append(score)  # Store score for histogram
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
    
    plt.tight_layout()
    figures.append(fig3)

    # Figure 4: Cumulative coverage plot
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = fig4.add_subplot(111)

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
    
    plt.tight_layout()
    figures.append(fig4)

    return figures

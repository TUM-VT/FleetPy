"""Utility functions for analyzing FleetPy simulation results."""
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def analyze_kpis(results_dir):
    """Analyze and summarize key performance indicators from standard evaluation."""
    eval_file = os.path.join(results_dir, 'standard_eval.csv')
    std_eval = pd.read_csv(eval_file)

    def get_eval_value(metric_name):
        row = std_eval[std_eval['Unnamed: 0'] == metric_name]
        if not row.empty:
            return row['MoD_0'].values[0]
        else:
            return float('nan')

    # Get key metrics
    served_requests = get_eval_value('number users')
    served_percentage = get_eval_value('served online users [%]')
    total_requests = served_requests / \
        (served_percentage / 100) if served_percentage else float('nan')

    kpi_summary = pd.DataFrame({
        'Metric': [
            'Total Requests',
            'Served Requests',
            'Average Wait Time (min)',
            'Average Trip Time (min)',
            'Total Distance (km)',
            'Empty Distance (km)',
            'Service Rate (%)',
            'Created Offers (%)',
            'Fleet Utilization (%)',
            'Occupancy Rate (%)'
        ],
        'Value': [
            total_requests,
            served_requests,
            get_eval_value('waiting time') / 60,
            get_eval_value('travel time') / 60,
            get_eval_value('total vkm'),
            get_eval_value('% empty vkm') * get_eval_value('total vkm') / 100,
            served_percentage,
            get_eval_value('% created offers'),
            get_eval_value('% fleet utilization'),
            get_eval_value('occupancy')
        ]
    })

    return kpi_summary


def analyze_user_stats(results_dir):
    """Analyze and visualize user statistics."""
    user_stats = pd.read_csv(os.path.join(results_dir, '1_user-stats.csv'))

    # Convert time columns from seconds to minutes
    user_stats['rq_time_min'] = user_stats['rq_time'] / 60
    user_stats['pickup_time_min'] = user_stats['pickup_time'] / 60

    # Create scatter plot of pickup vs request times
    fig = px.scatter(user_stats,
                     x='rq_time_min',
                     y='pickup_time_min',
                     title='Request vs Pickup Times',
                     labels={'rq_time_min': 'Request Time (min)',
                             'pickup_time_min': 'Pickup Time (min)'},
                     template='plotly_white')

    # Add reference line (x=y)
    fig.add_trace(go.Scatter(x=[0, user_stats['rq_time_min'].max()],
                             y=[0, user_stats['rq_time_min'].max()],
                             mode='lines',
                             name='Instant Pickup',
                             line=dict(dash='dash', color='red')))

    # Calculate wait time statistics
    wait_times = user_stats['pickup_time_min'] - user_stats['rq_time_min']
    stats = wait_times.describe() / 60

    return fig, stats


def analyze_vehicle_status(results_dir):
    """Analyze and visualize vehicle status over time."""
    op_stats = pd.read_csv(os.path.join(results_dir, '2-0_op-stats.csv'))
    veh_types = pd.read_csv(os.path.join(results_dir, '2_vehicle_types.csv'))

    # Get all vehicle IDs
    all_vehicle_ids = set(veh_types['vehicle_id'])

    # Convert times to minutes and create time bins
    op_stats['start_time_min'] = op_stats['start_time'] / 60
    op_stats['end_time_min'] = op_stats['end_time'] / 60

    # Create a time series with regular intervals (30-second intervals)
    time_range = np.arange(0, op_stats['end_time_min'].max() + 1, 0.5)
    status_counts = pd.DataFrame(index=time_range)

    # Define status categories and their corresponding real statuses in the data
    status_mapping = {
        'idle': ['idle', 'waiting', 'out_of_service'],
        'occupied': ['route'],
        'empty': ['reposition'],
        'boarding': ['boarding']
    }

    for t in time_range:
        active_vehicles = op_stats[(op_stats['start_time_min'] <= t) &
                                   (op_stats['end_time_min'] > t)]

        # Track which vehicles have a status at this time
        vehicles_with_status = set(active_vehicles['vehicle_id'])

        # Count vehicles in each status group
        for status_group, statuses in status_mapping.items():
            status_counts.loc[t, status_group] = len(active_vehicles[
                active_vehicles['status'].isin(statuses)])

        # Vehicles with no status at this time are considered idle
        idle_missing = all_vehicle_ids - vehicles_with_status
        if 'idle' in status_counts.columns:
            status_counts.loc[t, 'idle'] += len(idle_missing)
        else:
            status_counts.loc[t, 'idle'] = len(idle_missing)

    # Create multi-line plot
    fig = go.Figure()
    colors = {'empty': '#1f77b4', 'occupied': '#2ca02c',
              'boarding': '#ff7f0e', 'idle': '#d62728'}

    for status, name in [('empty', 'Repositioning'),
                         ('occupied', 'With Passengers'),
                         ('boarding', 'Boarding/Alighting'),
                         ('idle', 'Idle')]:
        if status in status_counts.columns:
            fig.add_trace(go.Scatter(x=status_counts.index,
                                     y=status_counts[status],
                                     name=name,
                                     mode='lines',
                                     line=dict(color=colors[status])))

    fig.update_layout(title='Vehicle Status Distribution Over Time',
                      xaxis_title='Time (minutes)',
                      yaxis_title='Number of Vehicles',
                      template='plotly_white',
                      hovermode='x')

    return fig

def analyze_vehicle_performance(results_dir):
    """Analyze and visualize vehicle-specific performance metrics."""
    veh_eval = pd.read_csv(os.path.join(
        results_dir, 'standard_mod-0_veh_eval.csv'))

    # Calculate total costs
    veh_eval['total costs'] = veh_eval['fix costs'] + veh_eval['total variable costs']
    
    # Define metrics for analysis
    metrics = ['total km', 'total costs', 'total CO2 [g]']
    
    # Create subplots for each metric
    fig = make_subplots(rows=1, cols=3, 
                       subplot_titles=('Distance (km)', 
                                    'Costs (currency)', 
                                    'CO2 Emissions (g)'))

    # Add box plots in separate subplots with appropriate scales
    fig.add_trace(go.Box(y=veh_eval['total km'], 
                        name='Distance', 
                        boxpoints='all'), 
                 row=1, col=1)
    
    fig.add_trace(go.Box(y=veh_eval['total costs'], 
                        name='Costs', 
                        boxpoints='all'), 
                 row=1, col=2)
    
    fig.add_trace(go.Box(y=veh_eval['total CO2 [g]'], 
                        name='CO2', 
                        boxpoints='all'), 
                 row=1, col=3)

    # Update layout with separate y-axes titles
    fig.update_yaxes(title_text="Distance (km)", row=1, col=1)
    fig.update_yaxes(title_text="Costs", row=1, col=2)
    fig.update_yaxes(title_text="CO2 (g)", row=1, col=3)

    fig.update_layout(
        title='Vehicle Performance Metrics',
        template='plotly_white',
        showlegend=False,
        height=400,  # Adjust height for better visualization
        width=1200   # Adjust width to accommodate subplots
    )

    # Calculate summary statistics
    stats = {metric: veh_eval[metric].describe().round(2)
             for metric in metrics if metric in veh_eval.columns}

    return fig, stats

def analyze_detailed_user_experience(results_dir):
    """Analyze detailed user experience metrics focusing on travel time efficiency."""
    user_stats = pd.read_csv(os.path.join(results_dir, '1_user-stats.csv'))
    
    # Create figure for travel time analysis
    fig = go.Figure()

    # Calculate travel times
    user_stats['actual_travel_time'] = (user_stats['dropoff_time'] - user_stats['pickup_time']) / 60  # to minutes
    user_stats['direct_route_travel_time'] = user_stats['direct_route_travel_time'] / 60  # to minutes
    user_stats['travel_time_ratio'] = user_stats['actual_travel_time'] / user_stats['direct_route_travel_time']
    
    # Add travel time scatter plot
    fig.add_trace(
        go.Scatter(x=user_stats['direct_route_travel_time'],
                  y=user_stats['actual_travel_time'],
                  mode='markers',
                  name='Travel Times',
                  marker=dict(color='blue', opacity=0.6))
    )

    # Add reference line (x=y)
    max_time = max(user_stats['actual_travel_time'].max(), user_stats['direct_route_travel_time'].max())
    fig.add_trace(
        go.Scatter(x=[0, max_time],
                  y=[0, max_time],
                  mode='lines',
                  name='Direct Route',
                  line=dict(dash='dash', color='red'))
    )

    # Update layout
    fig.update_layout(
        title='Travel Time Analysis',
        xaxis_title='Direct Route Time (min)',
        yaxis_title='Actual Travel Time (min)',
        template='plotly_white',
        height=500,
        width=800,
        showlegend=True
    )

    # Calculate summary statistics
    summary_stats = {
        'Travel Time Efficiency': {
            'mean_ratio': user_stats['travel_time_ratio'].mean(),
            'median_ratio': user_stats['travel_time_ratio'].median(),
            'std_ratio': user_stats['travel_time_ratio'].std()
        }
    }

    return fig, summary_stats

"""
Analysis utilities for FleetPy simulation results.
Provides functions for loading, analyzing, and visualizing simulation results,
particularly focused on vehicle status and operator metrics.
"""

import os
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from src.evaluation.standard import read_op_output_file

# Time constants
MIN = 60
HOUR = 3600
DAY = 24 * HOUR
DEF_TEMPORAL_RESOLUTION = 15 * MIN
DEF_SMALLER_TEMPORAL_RESOLUTION = 2 * MIN

# Plotting constants
VEHICLE_STATES = ["0 (Approach)", "1", "2", "3", "4", "0 (Repositioning)", "0 (Idle)"]
STATUS_NAMES = ['idle', 'pickup', 'occupied', 'reposition']


# Color scheme
ORANGE_CMAP = plt.cm.get_cmap('Oranges', 20)
COLOR_PALETTE = list(ORANGE_CMAP(np.linspace(0, 1, 5))) + ['dodgerblue', 'white']


def load_simulation_results(scenario_cfg: Dict, fleetpy_path: str) -> str:
    """
    Load results directory for a given scenario configuration.
    
    Args:
        scenario_cfg: Configuration dictionary for the scenario
        fleetpy_path: Path to FleetPy installation
        
    Returns:
        Path to results directory
    """
    study_name = scenario_cfg.get('study_name', 'paper_2025_applications')
    scenario_name = scenario_cfg.get('scenario_name')
    return os.path.join(fleetpy_path, 'studies', study_name, 'results', scenario_name)


def analyze_temporal_status(results_dir: str, fleetsize: int, with_legend: bool = True) -> plt.Figure:
    """
    Create a detailed temporal analysis of vehicle status using stacked area plots.
    
    Args:
        results_dir: Path to simulation results directory
        fleetsize: Total number of vehicles in the fleet
        with_legend: Whether to show the legend
        
    Returns:
        Matplotlib figure object
    """
    def create_timeline(stats_df: pd.DataFrame, occupancy: Optional[int] = None, is_repo: bool = False) -> List[tuple]:
        """Helper to create timeline for a specific vehicle state"""
        if is_repo:
            condition = stats_df["status"] == "reposition"
        else:
            condition = (
                (stats_df["occupancy"] == occupancy if occupancy is not None else stats_df["occupancy"] >= 4)
                & (stats_df["status"] != "reposition")
            )
            
        filtered_stats = stats_df[condition]
        start_end = (
            [(t, +1) for t in filtered_stats["start_time"].values] +
            [(t, -1) for t in filtered_stats["end_time"].values]
        )
        start_end.sort(key=lambda x: x[0])
        
        timeline = [(0, 0)]
        for t, add in start_end:
            timeline.append((t, timeline[-1][1] + add))
        return timeline
    
    def smooth_timeline(times: List[int], time_line: List[tuple]) -> List[float]:
        """Interpolate timeline values for given times"""
        return [next((v for t2, v in reversed(time_line) if t2 <= t), 0) for t in times]
    
    # Load and process data
    op_stats = read_op_output_file(results_dir, 0)
    timelines = []
    
    # Create timelines for different states
    times = list(range(0, 24 * 3600 + 1, DEF_SMALLER_TEMPORAL_RESOLUTION))
    
    # Generate and smooth timelines
    for occupancy in [0, 1, 2, 3, None]:  # None represents occupancy >= 4
        timeline = create_timeline(op_stats, occupancy)
        timelines.append(smooth_timeline(times, timeline))
    
    # Add repositioning timeline
    repo_timeline = create_timeline(op_stats, is_repo=True)
    timelines.append(smooth_timeline(times, repo_timeline))
    
    # Calculate idle vehicles
    idle_timeline = [
        fleetsize - sum(values) for values in zip(*timelines)
    ]
    timelines.append(idle_timeline)
    
    # Create the stacked plot
    plt.stackplot(
        [t/3600 for t in times],  # Convert to hours
        timelines,
        colors=COLOR_PALETTE,
        edgecolor="black",
        linewidth=0.5,
        labels=VEHICLE_STATES
    )
    
    plt.xlim(0, 24)
    plt.ylim(0, fleetsize)
    plt.xlabel("Time [h]")
    plt.ylabel("Vehicles")
    
    if with_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles[::-1],
            labels[::-1],
            title="Occupancy",
            loc="upper left"
        )
    
    return plt.gcf()


def analyze_operator_metrics(scenario_dir: str, multi_operator: bool = False) -> Optional[pd.DataFrame]:
    """
    Analyze operator-specific performance metrics from simulation results.
    
    Args:
        scenario_dir: Path to scenario results directory
        multi_operator: Whether to analyze multiple operators
        
    Returns:
        DataFrame containing metrics for each operator or None if data doesn't exist
        
    Metrics included:
        - served_users: Percentage of served users
        - fleet_util: Fleet utilization percentage
        - shared_rides: Percentage of shared rides
        - avg_wait_time: Average waiting time (minutes)
        - avg_detour: Average relative detour
        - avg_occupancy: Average vehicle occupancy
        - total_distance: Total vehicle kilometers traveled
    """
    eval_file = os.path.join(scenario_dir, 'standard_eval.csv')
    if not os.path.exists(eval_file):
        return None
        
    df = pd.read_csv(eval_file)
    metrics = {}
    
    operators = (
        ['MoD_0', 'MoD_1'] if multi_operator and 'MoD_1' in df.columns 
        else ['MoD_0']
    )
    
    metric_mappings = {
        'served_users': 'served online users [%]',
        'fleet_util': '% fleet utilization',
        'shared_rides': 'shared rides [%]',
        'avg_wait_time': ('waiting time', 60),  # (column, divisor for conversion to minutes)
        'avg_detour': 'rel detour',
        'avg_occupancy': 'occupancy',
        'total_distance': 'total vkm'
    }
    
    for op in operators:
        metrics[op] = {}
        for metric_name, source in metric_mappings.items():
            if isinstance(source, tuple):
                col_name, divisor = source
                value = df.loc[df.iloc[:, 0] == col_name, op].iat[0] / divisor
            else:
                value = df.loc[df.iloc[:, 0] == source, op].iat[0]
            metrics[op][metric_name] = value
            
    return pd.DataFrame(metrics)

def compare_batch_assignments(scenario_cfg_batch: List[Dict], fleetpy_path: str) -> Optional[pd.DataFrame]:
    """
    Compare results from different batch assignment approaches.
    
    Args:
        scenario_cfg_batch: List of scenario configurations
        fleetpy_path: Path to FleetPy installation
        
    Returns:
        DataFrame with comparison results
    """
    results = {}
    for scenario in scenario_cfg_batch:
        name = scenario['scenario_name'].split('_')[1]  # Get AM, LA, or IH
        results_dir = load_simulation_results(scenario, fleetpy_path)
        
        metrics = analyze_operator_metrics(results_dir)
        if metrics is not None:
            results[name] = metrics['MoD_0']  # Single operator results
            
    return pd.DataFrame(results) if results else None

def plot_key_metrics_comparison(df_comparison: pd.DataFrame):
    """
    Plot comparison of key metrics between different approaches.
    
    Args:
        df_comparison: DataFrame with comparison results
    """
    key_metrics = ['served_users', 'fleet_util', 'shared_rides', 'avg_wait_time', 'avg_occupancy', 'avg_detour']
    metric_labels = {
        'served_users': 'Served Users (%)',
        'fleet_util': 'Fleet Utilization (%)',
        'shared_rides': 'Shared Rides (%)',
        'avg_wait_time': 'Average Wait Time (min)',
        'avg_occupancy': 'Average Occupancy',
        'avg_detour': 'Average Detour'
    }
    
    plt.figure(figsize=(30, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18
    })
    
    for i, metric in enumerate(key_metrics):
        plt.subplot(2, 3, i+1)
        bars = plt.bar(df_comparison.columns, df_comparison.loc[metric], width=0.5)
        plt.title(metric_labels[metric], fontsize=24, pad=20)
        plt.xticks(range(len(df_comparison.columns)), df_comparison.columns)
        plt.yticks(fontsize=20)
        
        plt.ylabel(metric_labels[metric])
        plt.title(metric_labels[metric])
        plt.xticks(rotation=0)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=14)
    
    plt.tight_layout()
    return plt.gcf()

def plot_vehicle_status_comparison(scenario_cfg_batch: List[Dict], fleetpy_path: str, fleetsize: int = 180):
    """
    Plot vehicle status distribution comparison between different approaches.
    
    Args:
        scenario_cfg_batch: List of scenario configurations
        fleetpy_path: Path to FleetPy installation
        fleetsize: Number of vehicles in the fleet
    """
    plt.figure(figsize=(25, 8))
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 20,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18
    })
    plt.subplots_adjust(wspace=0.2, bottom=0.15, top=0.9)
    for i, scenario in enumerate(scenario_cfg_batch):
        name = scenario['scenario_name'].split('_')[1]
        plt.subplot(1, 3, i+1)
        results_dir = load_simulation_results(scenario, fleetpy_path)
        analyze_temporal_status(
            results_dir,
            fleetsize=fleetsize,
            with_legend=(i==0)  # Only show legend for first plot
        )
        plt.title(f'{name} Approach')
    plt.tight_layout()
    return plt.gcf()

def plot_operator_comparison(data: Dict[str, pd.DataFrame], metric_pair: tuple, ylabel: str, 
                           filename: str = None, add_intersections: bool = False) -> plt.Figure:
    """
    Plot and compare metrics between different operators under varying conditions.
    
    Args:
        data: Dictionary of DataFrames containing metrics for different value of time (VOT)
        metric_pair: Tuple of (hailing_metric, pooling_metric) column names
        ylabel: Label for y-axis
        filename: Optional filename to save the plot
        add_intersections: Whether to add intersection points to the plot
    """
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 20,
        'axes.titlesize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18
    })
    
    col_hail, col_pool = metric_pair
    
    for vot, df in data.items():
        plt.plot(df['discount_pct'], df[col_hail], marker='o', linestyle='dashed',
                 label=f'Hailing (λ={vot})', color=f'C{list(data.keys()).index(vot)}')
        plt.plot(df['discount_pct'], df[col_pool], marker='s', linestyle='-',
                 label=f'Pooling (λ={vot})', color=f'C{list(data.keys()).index(vot)}')

        # Find intersection points
        if add_intersections:
            for i in range(len(df) - 1):
                y1_hail = df[col_hail].iloc[i:i + 2].values
                y1_pool = df[col_pool].iloc[i:i + 2].values
                x1 = df['discount_pct'].iloc[i:i + 2].values

                if (y1_hail[0] - y1_pool[0]) * (y1_hail[1] - y1_pool[1]) <= 0:
                    x_cross = x1[0] + (x1[1] - x1[0]) * (y1_hail[0] - y1_pool[0]) / (
                            (y1_hail[0] - y1_pool[0]) - (y1_hail[1] - y1_pool[1]))
                    plt.axvline(x=x_cross, color=f'C{list(data.keys()).index(vot)}', 
                              linestyle='dotted', alpha=0.5)
                    plt.text(x_cross, plt.gca().get_ylim()[0] + 1 * list(data.keys()).index(vot), 
                           f'{x_cross:.0f}%', rotation=0,
                           horizontalalignment='center', verticalalignment='bottom',
                           color=f'C{list(data.keys()).index(vot)}', fontsize=12)
    
    plt.xlabel('Discount Rate [%]')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    
    return plt.gcf()

def load_multi_operator_metrics(base_dir: str, vot_values: List[float], discount_rates: List[int]) -> Dict[float, pd.DataFrame]:
    """
    Load and process metrics for multiple operators comparison.
    
    Args:
        base_dir: Base directory containing result folders
        vot_values: List of VOT values to analyze
        discount_rates: List of discount percentages to analyze
    
    Returns:
        Dictionary mapping VOT values to DataFrames with metrics
    """
    df_metrics = {}
    
    for vot in vot_values:
        records = []
        
        for pct in discount_rates:
            result_dir = f"ex2_{pct}off_{vot}vot"
            eval_file = os.path.join(base_dir, result_dir, "standard_eval.csv")
            if not os.path.exists(eval_file):
                continue
                
            df = pd.read_csv(eval_file)
            record = {'discount_pct': pct}
            
            # Extract metrics for both operators
            metrics_to_extract = {
                'users': 'number users',
                'fleet_util': '% fleet utilization',
                'shared_rides': 'shared rides [%]',
                'modal_split': 'modal split',
                'revenue': 'mod revenue',
                'fix_costs': 'mod fix costs',
                'var_costs': 'mod var costs'
            }
            
            for op in ['MoD_0', 'MoD_1']:
                prefix = 'hailing_' if op == 'MoD_0' else 'pooling_'
                
                for metric, source in metrics_to_extract.items():
                    if metric in ['revenue', 'fix_costs', 'var_costs']:
                        # Store raw values for profit calculation
                        record[f'{prefix}{metric}'] = df.loc[df.iloc[:, 0] == source, op].iat[0]
                    else:
                        # For mode share, convert to percentage
                        value = df.loc[df.iloc[:, 0] == source, op].iat[0]
                        if metric == 'modal_split':
                            value *= 100
                        record[f'{prefix}{metric}'] = value
                
                # Calculate profit in millions
                record[f'{prefix}profit'] = (record[f'{prefix}revenue'] - 
                                           record[f'{prefix}fix_costs'] - 
                                           record[f'{prefix}var_costs']) / 1_000_000
            
            records.append(record)
        
        df_metrics[vot] = pd.DataFrame(records).sort_values('discount_pct')
    
    return df_metrics

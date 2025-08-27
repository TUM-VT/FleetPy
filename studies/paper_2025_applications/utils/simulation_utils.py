"""Utility functions for running simulations in batch mode."""
import os
import multiprocessing
import pandas as pd
from typing import List, Dict, Any

def get_optimal_parallel_sims(n_scenarios: int) -> int:
    """
    Determine the optimal number of parallel simulations based on available CPUs
    and number of scenarios.
    
    Args:
        n_scenarios: Number of scenarios to run
        
    Returns:
        int: Optimal number of parallel simulations
    """
    n_cpus = multiprocessing.cpu_count()
    return min(n_cpus, n_scenarios)

def check_results_exist(scenario_cfgs: List[Dict[str, Any]], fleetpy_path: str) -> List[Dict[str, Any]]:
    """
    Check if results exist for a list of scenarios and return missing ones
    
    Args:
        scenario_cfgs: List of scenario configurations
        fleetpy_path: Path to FleetPy root directory
        
    Returns:
        list: List of scenario configurations that don't have results
    """
    from utils.analysis import load_simulation_results
    
    missing_scenarios = []
    for scenario in scenario_cfgs:
        results_dir = load_simulation_results(scenario, fleetpy_path)
        eval_file = os.path.join(results_dir, 'standard_eval.csv')
        if not os.path.exists(eval_file):
            missing_scenarios.append(scenario)
    return missing_scenarios

def run_missing_scenarios(scenario_configs: List[Dict[str, Any]], 
                         constant_cfg_path: str, 
                         scenario_cfg_path: str,
                         scenario_name: str,
                         fleetpy_path: str,
                         scenarios_path: str) -> None:
    """
    Check and run missing scenarios for a given configuration
    
    Args:
        scenario_configs: List of scenario configurations to check
        constant_cfg_path: Path to the constant configuration file
        scenario_cfg_path: Path to the scenario configuration file
        scenario_name: Name of the scenario for display purposes
        fleetpy_path: Path to FleetPy root directory
        scenarios_path: Path to scenarios directory
    """
    from run_examples import run_scenarios
    
    # Check which scenarios need to be run
    missing_scenarios = check_results_exist(scenario_configs, fleetpy_path)

    if not missing_scenarios:
        print(f"✅ All {scenario_name} simulation results already exist, skipping simulation...")
        return
    
    n_missing = len(missing_scenarios)
    n_parallel = get_optimal_parallel_sims(n_missing)
    print(f"🚀 Running {n_missing} missing {scenario_name} scenarios using {n_parallel} parallel processes...")
    
    # Create temporary config file for missing scenarios only
    temp_sc = os.path.join(scenarios_path, f"temp_scenario_cfg_{scenario_name.lower().replace(' ', '_')}.csv")
    
    try:
        # Write missing scenarios to temporary config file
        pd.DataFrame(missing_scenarios).to_csv(temp_sc, index=False)
        
        # Run only the missing scenarios with optimal parallelization
        run_scenarios(constant_cfg_path, temp_sc, log_level="info", n_cpu_per_sim=1, n_parallel_sim=n_parallel)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_sc):
            os.remove(temp_sc)

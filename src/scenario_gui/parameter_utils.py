"""Utility functions for handling parameters in the FleetPy scenario GUI."""
import os
from typing import Dict, List, Optional, Any

def get_abnormal_param_options(param: str, fleetpy_path: str) -> Optional[List[str]]:
    """Get special options for specific parameters that need to be populated from the filesystem.
    
    Args:
        param: The parameter name to get options for
        fleetpy_path: The path to the FleetPy installation
        
    Returns:
        A list of options if the parameter has special handling, None otherwise
    """
    if param == "network_name":
        path = os.path.join(fleetpy_path, "data", "networks")
        return [""] + os.listdir(path)
    elif param == "demand_name":
        path = os.path.join(fleetpy_path, "data", "demand")
        return [""] + os.listdir(path)
    elif param == "rq_file":
        # Will be populated based on network and demand selection
        return [""]
    return None

def categorize_parameters(param_names: List[str], param_dict: Dict[str, Any]) -> Dict[str, List[str]]:
    """Categorize parameters into logical groups based on their prefixes and meanings.
    
    Args:
        param_names: List of parameter names to categorize
        param_dict: Dictionary of parameter objects with metadata
        
    Returns:
        Dictionary mapping category names to lists of parameter names
    """
    categories = {
        "Basic Settings": [],
        "Time Settings": [],
        "Request Settings": [],
        "Operator Settings": [],
        "Parcel Settings": [],
        "Vehicle Settings": [],
        "Infrastructure": [],
        "Other": []
    }
    
    for param in param_names:
        param_obj = param_dict.get(param)
        if not param_obj:
            continue
            
        if param.startswith(("start_time", "end_time", "time_step", "lock_time")):
            categories["Time Settings"].append(param)
        elif param.startswith("user_") or "wait_time" in param or "detour" in param:
            categories["Request Settings"].append(param)
        elif param.startswith("op_"):
            if "parcel" in param:
                categories["Parcel Settings"].append(param)
            else:
                categories["Operator Settings"].append(param)
        elif param in ["network_name", "demand_name", "rq_file", "scenario_name", "study_name"]:
            categories["Basic Settings"].append(param)
        elif param.startswith("veh_") or "vehicle" in param or "fleet" in param:
            categories["Vehicle Settings"].append(param)
        elif param.startswith(("zone_", "infra_")):
            categories["Infrastructure"].append(param)
        else:
            categories["Other"].append(param)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

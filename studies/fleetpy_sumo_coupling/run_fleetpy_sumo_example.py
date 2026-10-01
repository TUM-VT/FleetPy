import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) ### add fleetpy src path

from src.coupling.SUMO.SUMOFleetPyServer import run_fleetpy_sumo_simulation

if __name__ == "__main__":
    study_path = os.path.dirname(os.path.abspath(__file__))
    fp_constant_config = os.path.join(study_path, "scenarios", "constant_config.csv")
    fp_scenario_config = os.path.join(study_path, "scenarios", "001_sumo_example.csv")
    sumo_config = os.path.join(study_path, "sumo_example", "sumo_example.sumocfg")
    
    sumoBinary = "sumo-gui"  # or "sumo" for command line version
    log_level = "info"
    
    run_fleetpy_sumo_simulation(fp_constant_config, fp_scenario_config, sumo_config, sumoBinary, log_level)
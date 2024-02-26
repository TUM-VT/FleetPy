import os
import pandas as pd
import multiprocessing as mp
import sys
import traceback
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.join(os.path.dirname( os.path.dirname(os.path.abspath(__file__))))))) #'c:\\Users\\ge75mum\\Documents\\Research\\Fleetpy\\tum-vt-fleet-simulation'
sys.path.append(os.path.dirname(os.path.join(os.path.dirname( os.path.dirname(os.path.abspath(__file__)))))) # c:\\Users\\ge75mum\\Documents\\Research\\Fleetpy\\tum-vt-fleet-simulation\\FleetPy
 
from run_scenarios_from_csv_and_config import run_scenarios
import src.misc.config as config
from src.misc.globals import *
 
 
SC_PATH = os.path.join( os.path.dirname(os.path.abspath(__file__)), "scenarios")
 
if __name__ == "__main__":
    mp.freeze_support()
   
    show_progress_bar = False
   
    #f_out = open(os.path.join( os.path.dirname(os.path.abspath(__file__)), "sys_out.txt"), "w")
    #sys.stdout = f_out
    #print("sys.stdout is redirected to", sys.stdout, sys.stdout.name)
    try:
        if len(sys.argv) > 1:
            run_scenarios(*sys.argv)
        else:
           
            # log_level = "info"
            # cc = os.path.join(SC_PATH, r"const_cfg_all.yaml")
            # sc = os.path.join(SC_PATH, r"calib_fs_scenarios.csv")
            # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=18, show_progress_bar=show_progress_bar)
        
            # Path to the scenarios folder
            scs_path = os.path.join(os.path.dirname(__file__), "scenarios")
            print(scs_path)
            log_level = "info" # "debug"
            cc = os.path.join(scs_path, r"constant_config_stochastic_PUDO.csv")
            sc = os.path.join(scs_path, r"scenario_config_stochastic_PUDO.csv")
            run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
           
            # log_level = "info"
            # cc = os.path.join(SC_PATH, r"const_cfg_all.yaml")
            # sc = os.path.join(SC_PATH, r"scaling.csv")
            # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=20, show_progress_bar=show_progress_bar)
       
 
    except:
        traceback.print_exc()
    # finally:
    #     f_out.close()
    #     sys.stdout = sys.__stdout__

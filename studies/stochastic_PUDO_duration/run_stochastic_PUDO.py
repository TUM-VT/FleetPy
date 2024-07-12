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
   
    compare_eval_results_in_excel = False

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
            # sc = os.path.join(scs_path, r"debugging_scenario_name_change.csv")
            sc = os.path.join(scs_path, r"scenario_config_stochastic_PUDO.csv")
            sc = os.path.join(scs_path, r"scenario_config_stochastic_PUDO_DEBUGGING.csv")

            # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=2, n_parallel_sim=3)
            run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)

            # log_level = "info"
            # cc = os.path.join(SC_PATH, r"const_cfg_all.yaml")
            # sc = os.path.join(SC_PATH, r"scaling.csv")
            # run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=20, show_progress_bar=show_progress_bar)

            if compare_eval_results_in_excel:
                # Compare the results of the scenarios -> saved in the results file (and, by default, opens the excel file automatically)
                df_scenario_comparison = pd.DataFrame()
                for scenario in pd.read_csv(sc)["scenario_name"]:
                    standard_eval_path = os.path.join(os.path.dirname(__file__), "results",scenario, "standard_eval.csv")
                    standard_eval_scenario = pd.read_csv(standard_eval_path)
                    
                    new_row = pd.DataFrame([[scenario, ""]], columns=standard_eval_scenario.columns)
                    # Concatenate the new row with the original DataFrame
                    standard_eval_scenario = pd.concat([new_row, standard_eval_scenario]).reset_index(drop=True)
                    df_scenario_comparison = pd.concat([df_scenario_comparison, standard_eval_scenario], axis=1)
                comparison_file_path = os.path.join(os.path.dirname(__file__), "results",'comparison_last_run_scenarios.csv')
                tmp_csv_file = df_scenario_comparison.to_csv(comparison_file_path, index=False)
                
                vbs_file = r"C:\Users\ge75mum\Downloads\test_vbs.vbs"
                os.system("start cmd /c cscript //B {}".format(vbs_file))


    except:
        traceback.print_exc()
    # finally:
    #     f_out.close()
    #     sys.stdout = sys.__stdout__

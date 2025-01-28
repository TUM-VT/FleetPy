import pandas as pd
import pathlib
import subprocess
import multiprocessing
import os
import re
import gzip
import shutil
import os
import math
import csv
import xml.etree.ElementTree as ET


py_path = pathlib.Path(__file__)

SELECTED_SCENARIOS = [300,301,306,307,312,313,318,319,324,325,330,331,336,337,342,343]
STUDY_NAME = "fleetpy_sumo_coupling_in"
PROCESS_COUNT = 4
SIM_NETWORK_NAME = "sumo_in"

def get_current_max_key(FP_path):
    filenames = os.listdir(FP_path/"studies"/STUDY_NAME/"scenarios")
    keys = [filename.split("_")[0] for filename in filenames]
    pattern = r'^\d+$' ## only digits
    filtered_keys = [int(key) for key in keys if re.match(pattern, key)]
    return max(filtered_keys)

class SimulationRunner:

    def __init__(self,selected_scenarios,study_name,process_count,sim_network_name):
        self.selected_scenarios = selected_scenarios
        self.study_name = study_name
        self.process_count = process_count
        self.py_path = pathlib.Path(__file__)
        sc_config = pd.read_csv(self.py_path.parent/"studies"/self.study_name/"simulation_parameters.csv")
        sc_config = sc_config.set_index('simulation_index')
        self.sc_config = sc_config
        self.sc_config_file_dict = {}
        self.sim_network_name = sim_network_name
        self.res_dir = py_path.parent / "studies" / self.study_name / "results"

    def create_sc_config_files(self):
        for sc_index, row in self.sc_config.iterrows():
            sc_df = pd.DataFrame()
            scenario_name = f'{str(sc_index).zfill(3)}_{row["network_name"]}_{row["SAV_demand_ratio"]}_{row["sim_env"]}'
            p_cstr_dt = row['p_cstr_dt'] if row['p_cstr_dt'] is not None else 0
            p_cstr_wt = row['p_cstr_wt'] if row['p_cstr_wt'] is not None else 0
            sumo_statistics_interval = (
                int(row['sumo_statistics_interval']) 
                if not pd.isna(row['sumo_statistics_interval']) 
                else 24 * 3600
)           
            sc_df["scenario_name"] = [scenario_name]
            sc_df["op_module"] = ["PoolingIRSOnly"]
            sc_df['rq_file'] = [f"{row['simulation_network']}_s_{str(row['random_seed']).zfill(2)}_{row['SAV_demand_ratio']}.csv"]
            sc_df['demand_name'] = [f"{row['simulation_network']}_s_{str(row['random_seed']).zfill(2)}_{row['SAV_demand_ratio']}"]
            sc_df['op_fleet_composition'] = [f"{row['vehtype']}:{row['fleet_size']}"]
            sc_df['op_init_veh_distribution'] = [row['op_init_veh_distribution']]
            sc_df['network_type'] = [row['network_type']]
            sc_df['op_vr_control_func_dict'] = [f"func_key:{row['objective_function']};vot:{row['vot']};vor:{row['vor']};p_cstr_dt:{p_cstr_dt};p_cstr_wt:{p_cstr_wt}"]
            sc_df['sim_env'] = [row['sim_env']]
            sc_df['network_name'] = [row['network_name']]
            sc_df['start_time'] = [int(row['start_time'])]
            sc_df['end_time'] = [row['end_time']]
            sc_df['evaluation_int_start'] = [int(row['evaluation_int_start'])]
            sc_df['evaluation_int_end'] = [int(row['evaluation_int_end'])]
            sc_df['SAV_demand_ratio'] = [row['SAV_demand_ratio']]
            sc_df['sumo_statistics_interval'] = [sumo_statistics_interval]
            sc_df['sumo_fco_vehicles'] = [row['sumo_fco_vehicles']]
            sc_df['op_routing_mode'] = [row['op_routing_mode']]
            sc_df['random_seed'] = [row['random_seed']]
            sc_df["rerouting_sc"] = [row["rerouting_sc"]]

            self.sc_config_file_dict.update({sc_index:sc_df.squeeze()})
            sc_df.to_csv(py_path.parent/"studies"/STUDY_NAME/"scenarios"/f"{scenario_name}.csv", index=False)
    def create_rerouting_xml_files(self):
        rerouting_cfg_path = self.py_path.parent.parent / "fleetpy_coupling"/"Simulation"/self.sim_network_name/"Rerouting" /"rerouting_scenarios.csv"
        with open(rerouting_cfg_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                root = ET.Element("additional")
                tree = ET.ElementTree(root)
                if row["type"] == "lane":
                    rerouter = ET.SubElement(root, 'rerouter', {'id': str(row['id']), 'edges': str(row['rerouter_edges']),"probability":str(row['probability'])})
                    param = ET.SubElement(rerouter, 'param', {'key': 'rerouting_name', 'value': str(row['rerouting_name'])})
                    interval = ET.SubElement(rerouter, 'interval', {'begin': row['begin_time'], 'end': row['end_time']})
                    ET.SubElement(interval, 'closingLaneReroute', {'id': row['closed_lane'], 'disallow': "all"})
                    self.indent_xml(root)
                    tree.write(rerouting_cfg_path.parent/f"{row['id']}_rerouting.add.xml", encoding='utf-8', xml_declaration=True)


    def run_fleetpy_sc(self,sc_index):
        SAV_demand_ratio = float(self.sc_config_file_dict[sc_index].get("SAV_demand_ratio"))
        if self.sc_config_file_dict[sc_index].get("rerouting_sc") == None or math.isnan(self.sc_config_file_dict[sc_index].get("rerouting_sc")):
            sumocfg_path = self.py_path.parent.parent/"fleetpy_coupling"/"Simulation"/self.sim_network_name/f"{self.sim_network_name}_s_{str(self.sc_config_file_dict[sc_index]['random_seed']).zfill(2)}_{round(SAV_demand_ratio,2)}.sumocfg"
        else:
            rerouting_sc = self.sc_config_file_dict[sc_index].get("rerouting_sc")

            rerouting_sc = str(int(rerouting_sc))
            #rerouting_sc = str(int(self.sc_config_file_dict[sc_index].get("rerouting_sc").round()))
            sumocfg_path = self.py_path.parent.parent/"fleetpy_coupling"/"Simulation"/self.sim_network_name/f"{self.sim_network_name}_s_{str(self.sc_config_file_dict[sc_index]['random_seed']).zfill(2)}_{round(SAV_demand_ratio,2)}_r_{rerouting_sc.zfill(3)}.sumocfg"

        
        command = [
            "python",
            str(self.py_path.parent/"SUMOFleetPyServer.py"),
            str(self.py_path.parent/"studies"/self.study_name/"scenarios"/"constant_config.csv"),
            str(self.py_path.parent/"studies"/self.study_name/"scenarios"/f"{self.sc_config_file_dict[sc_index]['scenario_name']}.csv"),
            str(sumocfg_path),
            "sumo",
            "info"
        ]
       # try:
        result = subprocess.run(command)
        #result = subprocess.run(command, capture_output=True, text=True)
            #print(f"Process ID {multiprocessing.current_process().pid} - Output:", result.stdout)
            #print(f"Process ID {multiprocessing.current_process().pid} - Errors:", result.stderr)
       # except subprocess.CalledProcessError as e:
           # print(f"Process ID {multiprocessing.current_process().pid} - Command failed with error:", e)

    def run_in_parallel(self):
        print(f"Running {sim_runner.selected_scenarios} on {sim_runner.process_count} processes in paralell.")

        # Create a pool of workers and execute the function in parallel
        with multiprocessing.Pool(self.process_count) as pool:
            pool.map(self.run_fleetpy_sc,self.selected_scenarios)  # Mapping the function to run across multiple processes


    def zip_files(self):
        zip_files = []
        for scenario in self.selected_scenarios:
            sc_res_dir = self.res_dir /self.sc_config_file_dict[scenario].get("scenario_name")
            zip_files.append(sc_res_dir / "SumoDumps" / "vehRoutes.xml")
            zip_files.append(sc_res_dir / "00_simulation.log")
        
        for file_path in zip_files:
            if os.path.isfile(file_path):
                # Create the .gz file name
                gz_file_path = str(file_path) + '.gz'
                
                # Open the original file and the gzip file
                with open(file_path, 'rb') as f_in, gzip.open(gz_file_path, 'wb') as f_out:
                    # Copy the contents to the gzip file
                    shutil.copyfileobj(f_in, f_out)
                
                # Delete the original file
                os.remove(file_path)
                print(f"Compressed and deleted: {file_path}")
            else:
                print(f"File not found: {file_path}")
    
    def indent_xml(self, elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent_xml(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

if __name__ == "__main__":
    sim_runner = SimulationRunner(selected_scenarios=SELECTED_SCENARIOS,study_name=STUDY_NAME,process_count=PROCESS_COUNT,sim_network_name=SIM_NETWORK_NAME)
    sim_runner.create_sc_config_files()
    sim_runner.create_rerouting_xml_files()
    sim_runner.run_in_parallel()
    sim_runner.zip_files()

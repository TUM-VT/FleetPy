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
import argparse
import xml.etree.ElementTree as ET


py_path = pathlib.Path(__file__)

def get_current_max_key(FP_path):
    filenames = os.listdir(FP_path/"studies"/STUDY_NAME/"scenarios")
    keys = [filename.split("_")[0] for filename in filenames]
    pattern = r'^\d+$' ## only digits
    filtered_keys = [int(key) for key in keys if re.match(pattern, key)]
    return max(filtered_keys)

class SimulationRunner:

    def __init__(self,selected_scenarios,study_name,sim_network_name,process_count):
        self.selected_scenarios = selected_scenarios
        self.sim_network_name = sim_network_name
        self.study_name = study_name
        self.process_count = process_count
        self.py_path = pathlib.Path(__file__).resolve()
        sc_config = pd.read_csv(self.py_path.parent/"studies"/self.study_name/"simulation_parameters.csv")
        sc_config = sc_config.set_index('simulation_index')
        self.sc_config = sc_config
        self.sc_config_file_dict = {}
        self.res_dir = py_path.parent / "studies" / self.study_name / "results"


    def create_sc_config_files(self):
        for sc_index, row in self.sc_config.iterrows():
            sc_df = pd.DataFrame()
            scenario_name = f'{str(sc_index).zfill(3)}_{row["network_name"]}_{row["MOD_demand_subset"]}_{row["sim_env"]}'
            p_cstr_dt = row.get('p_cstr_dt') if row.get('p_cstr_dt') is not None else 0
            p_cstr_wt = row.get('p_cstr_wt') if row.get('p_cstr_wt') is not None else 0
           
            sc_df["scenario_name"] = [scenario_name]
            sc_df["op_module"] = ["PoolingIRSOnly"]
            sc_df['rq_file'] = [f"{row['network_name']}_s_{str(row['random_seed']).zfill(2)}_{row['MOD_demand_subset']}.csv"]
            sc_df['demand_name'] = [f"{row['network_name']}_s_{str(row['random_seed']).zfill(2)}_{row['MOD_demand_subset']}"]
            sc_df['op_fleet_composition'] = [f"{row['vehtype']}:{row['fleet_size']}"]
            sc_df['network_type'] = [row['network_type']]
            sc_df['op_vr_control_func_dict'] = [f"func_key:{row['objective_function']};vot:{row.get('vot')};vor:{row.get('vor')};p_cstr_dt:{p_cstr_dt};p_cstr_wt:{p_cstr_wt}"]
            sc_df['sim_env'] = [row['sim_env']]
            sc_df['network_name'] = [row['network_name']]
            sc_df['start_time'] = [int(row['start_time'])]  
            sc_df['end_time'] = [row['end_time']]
            sc_df['evaluation_int_start'] = [int(row['evaluation_int_start'])]
            sc_df['evaluation_int_end'] = [int(row['evaluation_int_end'])]
            sc_df['MOD_demand_subset'] = [row['MOD_demand_subset']]
            sc_df['sumo_t_update'] = [int(row['sumo_t_update']) if not pd.isna(row['sumo_t_update']) else 24 * 3600]
            sc_df['sumo_fcd_vehicles'] = [row['sumo_fcd_vehicles']]
            sc_df['random_seed'] = [row['random_seed']]
            sc_df["rerouting_sc"] = [row["rerouting_sc"]]
            sc_df["hybrid_router"] = [int(row["hybrid_router"])]
            sc_df["p_opt"] = [float(row["p_opt"])]

            self.sc_config_file_dict.update({sc_index:sc_df.squeeze()})
            sc_df.to_csv(py_path.parent/"studies"/self.study_name/"scenarios"/f"{scenario_name}.csv", index=False)

    def create_rerouting_xml_files(self):
        rerouting_cfg_path = self.py_path.parent.parent / "fleetpy_coupling" / "Simulation" / self.sim_network_name / "Rerouting" / "rerouting_scenarios.csv"
        with open(rerouting_cfg_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                closed_lanes = row['closed_lane'].split('*') if row['closed_lane'] else []
                root = ET.Element("additional")
                tree = ET.ElementTree(root)
                if row["type"] == "lane":
                    rerouter = ET.SubElement(root, 'rerouter', {'id': str(row['id']), 'edges': str(row['rerouter_edges']),"probability":str(row['probability'])})
                    param = ET.SubElement(rerouter, 'param', {'key': 'rerouting_name', 'value': str(row['rerouting_name'])})
                    interval = ET.SubElement(rerouter, 'interval', {'begin': row['begin_time'], 'end': row['end_time']})
                    for closed_lane in closed_lanes:
                        ET.SubElement(interval, 'closingLaneReroute', {'id': closed_lane, 'disallow': "all"})
                    self.indent_xml(root)
                    tree.write(rerouting_cfg_path.parent/f"{row['id']}_rerouting.add.xml", encoding='utf-8', xml_declaration=True)


    def run_fleetpy_sc(self,sc_index):
        MOD_demand_subset = float(self.sc_config_file_dict[sc_index].get("MOD_demand_subset"))
        if self.sc_config_file_dict[sc_index].get("rerouting_sc") == None or math.isnan(self.sc_config_file_dict[sc_index].get("rerouting_sc")):
            sumocfg_path = self.py_path.parent.parent/"fleetpy_coupling"/"Simulation"/self.sim_network_name/f"{self.sim_network_name}_s_{str(self.sc_config_file_dict[sc_index]['random_seed']).zfill(2)}_{round(MOD_demand_subset,2)}.sumocfg"
        else:
            rerouting_sc = self.sc_config_file_dict[sc_index].get("rerouting_sc")

            rerouting_sc = str(int(rerouting_sc))
            #rerouting_sc = str(int(self.sc_config_file_dict[sc_index].get("rerouting_sc").round()))
            sumocfg_path = self.py_path.parent.parent/"fleetpy_coupling"/"Simulation"/self.sim_network_name/f"{self.sim_network_name}_s_{str(self.sc_config_file_dict[sc_index]['random_seed']).zfill(2)}_{round(MOD_demand_subset,2)}_r_{rerouting_sc.zfill(3)}.sumocfg"

        
        command = [
            "python",
            str(self.py_path.parent/"SUMOFleetPyServer.py"),
            str(self.py_path.parent/"studies"/self.study_name/"scenarios"/"constant_config.csv"),
            str(self.py_path.parent/"studies"/self.study_name/"scenarios"/f"{self.sc_config_file_dict[sc_index]['scenario_name']}.csv"),
            str(sumocfg_path),
            "sumo",
            "warning"
        ]
       # try:
        result = subprocess.run(command)

    def run_in_parallel(self):
        print(f"Running {sim_runner.selected_scenarios} on {sim_runner.process_count} processes in paralell.")
        # Create a pool of workers and execute the function in parallel
        with multiprocessing.Pool(self.process_count) as pool:
            pool.map(self.run_fleetpy_sc,self.selected_scenarios)  # Mapping the function to run across multiple processes


    def zip_files(self):
        zip_files = []
        for scenario in self.selected_scenarios:
            sc_res_dir = self.res_dir /self.sc_config_file_dict[scenario].get("scenario_name")
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
    # Get Arguments
    parser = argparse.ArgumentParser(description='Run FleetPy-SUMO Coupling.')
    parser.add_argument("--scenarios", "--sc", type=lambda s: [int(item) for item in s.split(',')], help="List of scenario IDs", default=None)
    parser.add_argument('--study_name', type=str, default=None, help='Study name')
    parser.add_argument('--processes',"--p", type=int, default=None, help='Number of Processes')
    parser.add_argument('--sim_network_name', type=str, default=None, help='Simulation Network Name')
    parser.add_argument('--sc_from', type=int, default=None, help='From Scenario...')
    parser.add_argument('--sc_to', type=int, default=None, help='To Scenario... (including)')
    parser.add_argument('--fp_path ', type=str, default=str(py_path.parent), help='Path to FleetPy repository')
    parser.add_argument('--fp_coupling_path ', type=str, default=str(py_path.parent.parent / "fleetpy_coupling"), help='Path to FleetPy Coupling repository')
    args = parser.parse_args()
    
    selected_scenarios =list(range(args.sc_from,args.sc_to+1)) if args.sc_from is not None and args.sc_to is not None else args.scenarios
    sim_runner = SimulationRunner(selected_scenarios=selected_scenarios,study_name=args.study_name,process_count=args.processes,sim_network_name=args.sim_network_name)
    sim_runner.create_sc_config_files()
    sim_runner.create_rerouting_xml_files()
    sim_runner.run_in_parallel()
    sim_runner.zip_files()

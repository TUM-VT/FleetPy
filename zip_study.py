import sys
import os
from pathlib import Path
import zipfile
import subprocess
from datetime import datetime

from src.misc.globals import *
from src.FleetSimulationBase import build_operator_attribute_dicts
from src.misc.config import ConstantConfig, ScenarioConfig


def get_git_info():
    try:
        # Branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Full commit
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Repository root path
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Repository name (folder name of root)
        repo_name = Path(repo_root).name

        # Remote URL (origin)
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        return {
            "repo_name": repo_name,
            "repo_root": repo_root,
            "branch": branch,
            "commit": commit,
            "remote_url": remote_url,
        }

    except Exception:
        return None    

def read_all_scenario_configs(study_folder):
    scenario_folder = os.path.join(study_folder, "scenarios")
    constant_configs = []
    scenario_configs = []
    for f in os.listdir(scenario_folder):
        if os.path.isfile(os.path.join(scenario_folder, f)):
            try:
                cc = ConstantConfig(os.path.join(scenario_folder, f))
                constant_configs.append(cc)
                print("read constant config: ", f)
            except:
            #print(cc)
                sc = ScenarioConfig(os.path.join(scenario_folder, f))
                scenario_configs.append(sc)
                print("read scenario config: ", f)
    return constant_configs, scenario_configs

def get_all_network_files(network_path):
    list_files = []
    for file_path in Path(network_path).rglob('*'):
        if os.path.isfile(file_path):
            if str(file_path).endswith(".npy"):
                continue
            list_files.append(file_path)
    return list_files

def create_data_path_list(constant_configs, scenario_configs, study_name):
    list_data_files = []
    for cc in constant_configs:
        for scs in scenario_configs:
            for sc in scs:
                done_params = {}
                scenario_parameters = cc + sc
                scenario_parameters[G_STUDY_NAME] = study_name
                n_op = scenario_parameters[G_NR_OPERATORS]
                try:
                    list_op_dicts = build_operator_attribute_dicts(scenario_parameters, n_op)
                except ValueError: # configs dont fit (e.g. 2 op scenario config and 1 op constant config)
                    continue
                dir_names = get_directory_dict(scenario_parameters, list_op_dicts)
                list_data_files += get_all_network_files(dir_names[G_DIR_NETWORK])
                for op_id in range(n_op):
                    operator_attributes = list_op_dicts[op_id]
                    op_dir_names = dir_names.copy()
                    op_specific_dirs = dir_names.get(f"op_{op_id}", {})
                    op_dir_names.update(op_specific_dirs)
                    
                    for param, val in operator_attributes.items():
                        done_params[param] = 1
                        if val is None:
                            continue
                        elif param == G_OP_FLEET:
                            for vehicle_type in val.keys():
                                list_data_files.append(os.path.join(op_dir_names[G_DIR_VEH], f"{vehicle_type}.csv"))
                        elif param == G_OP_ACT_FLEET_SIZE:
                            list_data_files.append(Path(os.path.join(op_dir_names[G_DIR_FCTRL], val)))
                        elif param == G_RA_FC_FNAME:
                            if not op_dir_names.get(G_DIR_FC):
                                continue
                            list_data_files.append(Path(os.path.join(op_dir_names[G_DIR_FC], val)))
                        elif param == G_RA_OP_CORR_M_F:
                            if not op_dir_names.get(G_DIR_FC):
                                continue
                            list_data_files.append(Path(os.path.join(op_dir_names[G_DIR_FC], val)))
                        elif param == G_OP_DEPOT_F:
                            if not op_dir_names.get(G_DIR_INFRA):
                                continue
                            list_data_files.append(Path(os.path.join(op_dir_names[G_DIR_INFRA], val)))
                        elif param == G_PUBLIC_CHARGING_FILE:
                            if not op_dir_names.get(G_DIR_INFRA):
                                continue
                            list_data_files.append(Path(os.path.join(op_dir_names[G_DIR_INFRA], val)))
                        elif param == G_NW_DYNAMIC_F:
                            continue # already in network
                        elif param.endswith("_file") or param.endswith("f"):
                            print(op_dir_names)
                            print(scenario_parameters)
                            print(param, val)
                            raise ValueError("didnt find file for operator parameter ", param, val)
                    
                for param, val in scenario_parameters.items():
                    if done_params.get(param):
                        continue
                    if val is None:
                        continue
                    if param == G_RQ_FILE:
                        list_data_files.append(Path(os.path.join(dir_names[G_DIR_DEMAND], val)))
                    elif param == G_PA_RQ_FILE:
                        if not dir_names.get(G_DIR_PARCEL_DEMAND):
                            continue
                        list_data_files.append(Path(os.path.join(dir_names[G_DIR_PARCEL_DEMAND], val)))
                    elif param == G_PT_SCHEDULE_F:
                        list_data_files.append(Path(os.path.join(dir_names[G_DIR_PT], val)))
                    elif param == G_PT_STATION_F:
                        list_data_files.append(Path(os.path.join(dir_names[G_DIR_PT], val)))
                    elif param == G_PT_ALIGNMENT_F:
                        print("Warning: Skip file", param, val)
                    elif param == G_PT_DEMAND_DIST:
                        print("Warning: Skip file", param, val)
                    elif param == G_CH_OP_F:
                        if not dir_names.get(G_DIR_INFRA):
                            continue
                        list_data_files.append(Path(os.path.join(dir_names[G_DIR_INFRA], val)))
                    elif param == G_NW_DYNAMIC_F:
                        continue # already in network
                    elif param.endswith("_file") or param.endswith("f"):
                        print(dir_names)
                        print(scenario_parameters)
                        print(param, val)
                        raise ValueError("didnt find file for parameter ", param, val)
    list_data_files = list(set(list_data_files))
    print(f" ... found {len(list_data_files)} data files.")
    return list_data_files

def create_study_path_list(study_folder):
    if not os.path.isfile(os.path.join(study_folder, "README.md")):
        print("====================================================================================================================================================================")
        print("WARNING!")
        print(f"Didnt find README.md for this study!\n Please create 'README.md' describing this study and put it to\n {os.path.join(study_folder, 'README.md')}")
        print("====================================================================================================================================================================")
        #raise ValueError(f"Didnt find README.md for this study!\n Please create 'README.md' describing this study and put it to\n {os.path.join(study_folder, 'README.md')}")
    list_files = []
    for file_path in Path(study_folder).rglob('*'):
        if os.path.isfile(file_path):
            if 'results' not in file_path.parts:
                list_files.append(file_path)
    print(f" ... found {len(list_files)} study files.")
    return list_files

def create_source_path_list():
    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    list_files = []
    for file_path in Path(source_dir).rglob('*'):
        if os.path.isfile(file_path):
            list_files.append(file_path)
    print(f" ... found {len(list_files)} source files.")
    return list_files

def create_results_path_list(study_folder, exhaustive=False):
    results_dir = os.path.join(study_folder, "results")
    list_files = []
    if exhaustive:
        for file_path in Path(results_dir).rglob('*'):
            if os.path.isfile(file_path):
                list_files.append(file_path)
    else:
        for file_path in Path(results_dir).rglob('*'):
            if os.path.isfile(file_path):
                if file_path.name == "standard_eval.csv" or file_path.name == "00_config.json":
                    list_files.append(file_path)
    print(f" ... found {len(list_files)} result files.")
    return list_files

def add_files_to_zip_preserve_structure(file_paths, zip_output_path, base_folder=None):
    """Preserve relative folder structure in zip"""
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_paths:
            if os.path.isfile(file_path):
                if base_folder:
                    arcname = os.path.relpath(file_path, base_folder)
                else:
                    arcname = file_path
                zipf.write(file_path, arcname=arcname)
                
def create_git_info_file(study_folder):
    git_info = get_git_info()
    if git_info is None:
        print("Warning: couldnt get git info. Skipping creation of git_info.txt")
        return
    git_info_path = os.path.join(study_folder, "git_info.txt")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(git_info_path, "w") as f:
        f.write(f"Git repository: {git_info.get('repo_name')}\n")
        f.write(f"Git remote URL: {git_info.get('remote_url')}\n")
        f.write(f"Git branch: {git_info.get('branch')}\nGit commit: {git_info.get('commit')}\nFile generated at: {current_time}")
    return git_info_path

def zip_study(study_folder, zip_results_mode="None"):
    """ 
    Zip a given study folder including all data files that are loaded in the configs, all src files and all study files.
    Results files are not included by default, but can be added with zip_results_mode = "standard" (only standard_eval.csv and 00_config.json) or "exhaustive" (all files in results folder).
    :param study_folder: path to the study folder you want to zip (e.g. FleetPy\studies\my_study)
    :param zip_results_mode: "None" (default, no results files), "standard" (only standard_eval.csv and 00_config.json) or "exhaustive" (all files in results folder)
    """
    study_name = os.path.basename(study_folder)
    base_folder = os.path.dirname(os.path.abspath(__file__))
    zip_name = f"archive_fleetpy_study_{study_name}.zip"
    zip_output_path = os.path.join(study_folder, zip_name)
    if os.path.exists(zip_output_path):
        raise IOError(f"{zip_output_path} \n Zip-file already exists. Please move or delete.")
    
    ccs, scs = read_all_scenario_configs(study_folder)
    list_files = []
    list_files += create_data_path_list(ccs, scs, study_name)
    list_files += create_study_path_list(study_folder)
    list_files += create_source_path_list()
    if zip_results_mode == "standard":
        list_files += create_results_path_list(study_folder, exhaustive=False)
    elif zip_results_mode == "exhaustive":
        list_files += create_results_path_list(study_folder, exhaustive=True)
    elif zip_results_mode == "None":
        pass
    else:
        raise ValueError(f"zip_results_mode must be 'None', 'standard' or 'exhaustive'. Got: {zip_results_mode}")
    list_files.append(create_git_info_file(study_folder))
    print(f"  -> found {len(list_files)} files overall.")
    
    add_files_to_zip_preserve_structure(list_files, zip_output_path, base_folder=base_folder)
    
    print(f"Finished zipping study! Zip file created at: {zip_output_path}")
            
if __name__ == "__main__":
    """ 
    This script can be used to archive a given fleetpy paper to ensure later reproduceability.
    It creates a zip-file in the given study folder that includes
        - all fleetpy-data that is loaded within the configs found in {study_path}\scenarios
        - all files in {study_path} except for simulation results in {study_path}\results
        - all src files in the current state
    You should add a README.md in the folder {study_path} that includes details of the study and especially the fleetpy-commit of your work.
    
    args:
    - study_path: path to the study-folder you want to archive (in FleetPy\studies\{study_name})
    - zip_results_mode [optional]: "None" (default, no results files), "standard" (only standard_eval.csv and 00_config.json) or "exhaustive" (all files in results folder)
    """
    print("Start zipping study ...")
    if len(sys.argv) == 2:
        study_folder = sys.argv[1]
        zip_study(study_folder)
    elif len(sys.argv) == 3:
        study_folder = sys.argv[1]
        zip_results_mode = sys.argv[2]
        zip_study(study_folder, zip_results_mode=zip_results_mode)
    else:
        if len(sys.argv) <= 1:
            raise IOError(f"Expected single path to study folder as input. Got none.")
        else:
            raise IOError(f"Expected single path to study folder as input. Got: {sys.argv[1:]}")
#!/usr/bin/python
# IN Traci conda env
from abc import abstractmethod
import os, sys
import pandas as pd
import csv
import logging 
from operator import itemgetter
import sys, getopt
from time import perf_counter
from typing import Tuple
from datetime import datetime
from tqdm import tqdm
import xml.etree.ElementTree as ET
import time
import numpy as np
import pathlib

from src.SUMOcontrolledSim import SUMOcontrolledSim
from src.misc.init_modules import load_simulation_environment
import src.misc.config as config
from src.misc.globals import *
import src.evaluation.standard as eval
from run_examples import run_scenarios


""" 
This script can be used to create a FleetPy-Simulation coupled to SUMO.
Customers, requests and the control of fleet vehicles are controled in FleetPy while vehicle movements are conducted in SUMO.
When a new vehicle route is available, the vehicle is created in SUMO and drives on the computed route to its destination. The vehicle is deleted once it reaches the destination in SUMO (also for boarding processes and created again after the boarding process)
The script requires as input the usual FleetPy config files (constant config and scenario config) and additionally the .sumocfg file.
The network used by FleetPy has to be synchronized to the SUMO-network. Therefore, the script preprocessing\networks\network_from_sumo.py can be used to create the corresponding FleetPy network representation.
The corresponding FleetPy-demand files have to be created manually.
Additionally, the vehicle_types (str-names) used in the FleetPy have to be defined in SUMO, too. Those will be used as fleet vehicle types
"""

LOG = logging.getLogger(__name__)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

t1_start = perf_counter()


class SUMOFleetPyServer():
    def __init__(self,constant_config_path, scenario_config_path, sumo_config, sumoBinary, log_level):
        """
        :param constant_config_file: this file contains all input parameters that remain constant for a study
        :type constant_config_file: str
        :param scenario_file: this file contain all input parameters that are varied for a study
        :type scenario_file: str
        :param n_parallel_sim: number of parallel simulation processes
        :type n_parallel_sim: int
        :param n_cpu_per_sim: number of cpus for a single simulation
        :type n_cpu_per_sim: int
        :param evaluate: 0: no automatic evaluation / != 0 automatic evalaution after each simulation
        :type evaluate: int
        :param log_level: hierarchical output to the logging file. Possible inputs with hierarchy from low to high:
                - "verbose": lowest level -> logs everything; even code which could scale exponentially
                - "debug": standard debugging logger. code which scales exponentially should not be logged here
                - "info": basic information during simulations (default)
                - "warning": only logs warnings
        :type log_level: str
        :param keep_old: does not start new simulation if result files are already available in scenario output directory
        :type keep_old: bool
        """
        
        self.fp_constant_config_path = constant_config_path
        self.fp_scenario_config_path = scenario_config_path
        self.fp_log_level = log_level
        self.fp_n_cpu_per_sim=1
        self.fp_evaluate = 1
        self.fp_keep_old = False
        self.sumo_config_path = sumo_config
        self.sumo_binary = sumoBinary
        self.sumo_edgeData_interval = 3600
        self.g_start_time = time.time()
        scenario_cfgs = config.ScenarioConfig(self.fp_scenario_config_path)
        self.sumo_sim = scenario_cfgs[0].get("sumo_sim")
        self.fp_to_sumo_veh_id_dict= {}
        self.sumo_to_fp_veh_id_dict = {}

    def _finalize_setup(self):
        self.g_end_time_setup = time.time()

    def setup_fleetsimulation(self, n_cpu_per_sim=1, evaluate=1,keep_old=False) -> SUMOcontrolledSim:
        """
        This function combines constant study parameters and scenario parameters.
        Then it sets up a pool of workers and starts a simulation for each scenario.
        The required parameters are stated in the documentation.
        """
        # read constant and scenario config files
        constant_cfg = config.ConstantConfig(self.fp_constant_config_path)

        scenario_cfgs = config.ScenarioConfig(self.fp_scenario_config_path)
        # set constant parameters from function arguments
        const_abs = os.path.abspath(self.fp_constant_config_path)
        study_name = os.path.basename(os.path.dirname(os.path.dirname(const_abs)))
        
        if study_name == "scenarios":
            print("ERROR! The path of the config files is not longer up to date!")
            print("See documentation/Data_Directory_Structure.md for the updated directory structure needed as input!")
            exit()
        
        if constant_cfg.get(G_STUDY_NAME) is not None and study_name != constant_cfg.get(G_STUDY_NAME):
            print("ERROR! {} from constant config is not consitent with study directory: {}".format(constant_cfg[G_STUDY_NAME], study_name))
            print("{} is now given directly by the folder name !".format("G_STUDY_NAME"))
            exit()
        constant_cfg[G_STUDY_NAME] = study_name
        constant_cfg["n_cpu_per_sim"] = self.fp_n_cpu_per_sim
        constant_cfg["evaluate"] = self.fp_evaluate
        constant_cfg["log_level"] = self.fp_log_level
        constant_cfg["keep_old"] = self.fp_keep_old
        
        
        # combine constant and scenario parameters into verbose scenario parameters
        for i, scenario_cfg in enumerate(scenario_cfgs):
            scenario_cfgs[i] = constant_cfg + scenario_cfg
            
        scenario_cfgs[0][G_SIM_START_TIME] += scenario_cfgs[0].get(G_SUMO_SIM_TIME_OFFSET, 0)

        SF = load_simulation_environment(scenario_cfgs[0])
        

        self.fp_scenario_config = scenario_cfgs[0]

        self.fp_sim_env = SF

        # Get interval in which new network statistics are gathered and sent to FleetPy to updated network (if not given, no statistics are gathered)
        travel_time_interval = self.fp_sim_env.scenario_parameters.get(G_SUMO_STAT_INT)

        if travel_time_interval is None or travel_time_interval == 24*3600:
            self.g_update_fleetsim_traveltimes = False
            self.g_update_travel_statistics_time_step = 10000000000000
        else:
            self.g_update_fleetsim_traveltimes = True
            self.g_update_travel_statistics_time_step = travel_time_interval

        
    def setup_hybrid_router(self):
        if self.fp_sim_env.scenario_parameters.get("hybrid_router") == 1:
            
            print("Setting up hybrid router")
            breakpoint()
            fp_path = pathlib.Path(self.fp_sim_env.dir_names.get(G_DIR_MAIN))
            sc_0_data_path = fp_path.parent / "fleetpy_coupling" / "hybrid_router" / "hourly_edge_counts_sc0.csv"
            if not os.path.isfile(sc_0_data_path):
                raise(f"Error: {sc_0_data_path} does not exist! Please run the hybrid router script to create this file.")
            self.fp_hybrid_router_sc0_df = pd.read_csv(sc_0_data_path)
            fp_path = pathlib.Path(self.fp_sim_env.dir_names.get(G_DIR_MAIN))
            self.fp_hybrid_router_sc0_df["edge_id"] = self.fp_hybrid_router_sc0_df["edge"].map(self.g_sumo_edge_id_to_fs_edge)
            self.fp_hybrid_router_sc0_df = self.fp_hybrid_router_sc0_df.rename(columns={"count":"count_sc0"})
            self.fp_hybrid_router_sc0_df["edge_id_str"] = self.fp_hybrid_router_sc0_df["edge_id"].astype(str)
            self.fp_hybrid_router_sc0_df.reset_index(inplace=True,drop=True)
            
            network_dir_path = pathlib.Path(self.fp_sim_env.dir_names[G_DIR_NETWORK])
            historic_tt_dir = network_dir_path.parent / f"sumo_in_s_{str(self.fp_sim_env.scenario_parameters.get('random_seed')).zfill(2)}"
            eval_start = self.fp_sim_env.scenario_parameters.get(G_SIM_START_TIME, self.fp_sim_env.scenario_parameters.get(G_EVAL_INT_START)) 
            eval_end = self.fp_sim_env.scenario_parameters.get(G_SIM_END_TIME, self.fp_sim_env.scenario_parameters.get(G_EVAL_INT_END))
            hourly_dirs = [
                d for d in os.listdir(historic_tt_dir)
                if d.isdigit() and eval_start <= int(d) <= eval_end and os.path.isdir(os.path.join(historic_tt_dir, d))
            ]
            hourly_tt_df_dict = {}
            for hour in hourly_dirs:
                hour_path = historic_tt_dir / hour / "edges_td_att.csv"
                edges_df = pd.read_csv(hour_path)
                edges_df["edge_id"] =  tuple(zip(edges_df['from_node'], edges_df['to_node']))
                hourly_tt_df_dict.update({hour: edges_df})
            self.fp_hybrid_router_hourly_tt_dict = hourly_tt_df_dict

        else:
            return
    def setup_traci(self):
        results_path = self.fp_sim_env.dir_names[G_DIR_OUTPUT]
        seed = self.fp_sim_env.scenario_parameters[G_RANDOM_SEED]
        SUMO_start_time = self.fp_sim_env.scenario_parameters.get(G_SIM_START_TIME)
        EdgeDataCfgPath = self._create_EdgeDataCfg_xml()       

        if not os.path.isdir(os.path.join(results_path, "SumoDumps")):
            os.mkdir(os.path.join(results_path, "SumoDumps"))

        TripInfoPath = os.path.join(results_path, "SumoDumps", "TripInfo.xml") 
        vehRoutePath = os.path.join(results_path, "SumoDumps", "vehRoutes.xml") 
        collisionPath = os.path.join(results_path, "SumoDumps", "collisionPath.xml") 
        statisticsPath = os.path.join(results_path, "SumoDumps", "statistics.xml") 
        edges_output = os.path.join(results_path, "SumoDumps", "edge-output.xml")
  

        sumoCmd = [self.sumo_binary, "-c", self.sumo_config_path ,"--collision.action","warn","--begin",str(SUMO_start_time),
                "--step-length","1","--tripinfo-output",TripInfoPath,
                "--vehroute-output",vehRoutePath,"--vehroute-output.exit-times","--vehroute-output.incomplete","--vehroute-output.write-unfinished",
                "--collision-output",collisionPath,"--statistic-output",statisticsPath,"--start", "--seed", str(seed),"--no-warnings",str(True)]   
        #"+a",EdgeDataCfgPath, Currently not yet working
        #Trajectoriespath = os.path.join(results_path, "SumoDumps", "Trajectories.xml") 
        #fullOutputPath = os.path.join(results_path, "FullOutput.xml")
        #fcdPath = os.path.join(results_path, "SumoDumps", "fcd-output.xml")
        #if sumo_fcd_output:
        #     sumoCmd += ["--fcd-output", fcdPath, "--fcd-output.geo"]
        # if sumo_lane_output:
        #     sumoCmd += ["--lanedata-output", lane_output]
      
        traci.start(sumoCmd)
        print(f"SUMO-Simulation Initialized at t={SUMO_start_time}")

    def _create_EdgeDataCfg_xml(self):
        additional = ET.Element("additional")
        edge_data = ET.SubElement(additional, "edgeData", {
            "id": str(self.fp_sim_env.scenario_parameters[G_SCENARIO_NAME]),
            "file": str(os.path.join(self.fp_sim_env.dir_names[G_DIR_OUTPUT], "SumoDumps", "EdgeData.xml")),
            "begin": str(self.fp_sim_env.scenario_parameters.get(G_EVAL_INT_START,self.fp_sim_env.scenario_parameters.get(G_SIM_START_TIME))),
            "end": str(self.fp_sim_env.scenario_parameters.get(G_EVAL_INT_END,self.fp_sim_env.scenario_parameters.get(G_SIM_END_TIME))),
            "trackVehicles": str(True),
            "period": str(self.sumo_edgeData_interval)
        }) 
        tree = ET.ElementTree(additional)
        file_name = os.path.join(self.fp_sim_env.dir_names[G_DIR_OUTPUT], "EdgeData.cfg.add.xml")
        tree.write(file_name, encoding='utf-8')
        return file_name

    def setup_network_translation(self):
        """Edge ID Dict Translator 
        :param sumo_edge_id_to_fs_edge --> SUMO_EDGE_ID : (FP_START_NODE,FP_END_NODE)
        :param fs_edge_to_sumo_edge_id -->  (FP_START_NODE,FP_END_NODE):SUMO_EDGE_ID
        :param sumo_node_list --> [J1,J1,J2...]
        :param fs_edge_to_ff_tt --> (FP_START_NODE,FP_END_NODE): TRAVEL_TIME
        :param fs_edge_to_len --> (FP_START_NODE,FP_END_NODE): DISTANCE
        :param fs_node_to_sumo_junction --> FP_Node: DUMO_JUNCTION
        """
        nw_path = self.fp_sim_env.dir_names[G_DIR_NETWORK]
        edge_df = pd.read_csv(os.path.join(nw_path, "base", "edges.csv"))
        node_df = pd.read_csv(os.path.join(nw_path, "base", "nodes.csv"))
        sumo_edge_id_to_fs_edge = {}
        fs_edge_to_sumo_edge_id = {}
        fs_edge_to_ff_tt = {}
        fs_edge_to_len ={}
        for _, row in edge_df.iterrows():
            if row["from_node"] != row["to_node"]:
                sumo_edge_id = row["source_edge_id"]
                if pd.isna(sumo_edge_id):
                    print(f"Warning: no sumo edge for {row['from_node']} -> {row['to_node']} exists!")
                    continue
                sumo_edge_id = str(sumo_edge_id)
                start_node_index = row["from_node"]
                end_node_index = row["to_node"]
                sumo_edge_id_to_fs_edge[sumo_edge_id] = (start_node_index, end_node_index)  ## SUMO_EDGE_ID : (FP_START_NODE,FP_END_NODE)
                fs_edge_to_sumo_edge_id[(start_node_index, end_node_index)] = sumo_edge_id ## (FP_START_NODE,FP_END_NODE):SUMO_EDGE_ID 
                fs_edge_to_ff_tt[(start_node_index, end_node_index)] = row["travel_time"] ## (FP_START_NODE,FP_END_NODE): TRAVEL_TIME
                fs_edge_to_len[(start_node_index, end_node_index)] = row["distance"] ## (FP_START_NODE,FP_END_NODE): DISTANCE
                
        fs_node_to_sumo_junction = node_df.set_index('node_index')['source_node_id'].to_dict()
            
        sumo_node_list = node_df.source_node_id
        sumo_node_list = sumo_node_list.values.tolist()

        self.g_sumo_edge_id_to_fs_edge = sumo_edge_id_to_fs_edge
        self.g_fs_edge_to_sumo_edge_id = fs_edge_to_sumo_edge_id
        self.sumo_node_list = sumo_node_list
        self.g_fs_edge_to_ff_tt = fs_edge_to_ff_tt
        self.g_fs_edge_to_len = fs_edge_to_len
        self.g_fs_node_to_sumo_junction =fs_node_to_sumo_junction
        self._finalize_setup()
    
    def run_fp_simulation(self):
        #run_scenarios(constant_config_file=self.fp_constant_config_path, scenario_file=self.fp_scenario_config_path, n_parallel_sim=1, n_cpu_per_sim=1, evaluate=1, log_level="info", continue_next_after_error=True)
        constant_cfg = config.ConstantConfig(self.fp_constant_config_path)
        scenario_cfgs = config.ScenarioConfig(self.fp_scenario_config_path)

        # set constant parameters from function arguments
        # TODO # get study name and check if its a studyname
        const_abs = os.path.abspath(self.fp_constant_config_path)
        study_name = os.path.basename(os.path.dirname(os.path.dirname(const_abs)))

        if study_name == "scenarios":
            print("ERROR! The path of the config files is not longer up to date!")
            print("See documentation/Data_Directory_Structure.md for the updated directory structure needed as input!")
            exit()
        if constant_cfg.get(G_STUDY_NAME) is not None and study_name != constant_cfg.get(G_STUDY_NAME):
            print("ERROR! {} from constant config is not consistent with study directory: {}".format(constant_cfg[G_STUDY_NAME], study_name))
            print("{} is now given directly by the folder name !".format(G_STUDY_NAME))
            exit()
        constant_cfg[G_STUDY_NAME] = study_name
        constant_cfg["n_cpu_per_sim"] = 1
        constant_cfg["evaluate"] = self.fp_evaluate
        constant_cfg["log_level"] = self.fp_log_level

    # combine constant and scenario parameters into verbose scenario parameters
        for i, scenario_cfg in enumerate(scenario_cfgs):
            scenario_cfgs[i] = constant_cfg + scenario_cfg
        
        print(scenario_cfgs)
        SF = load_simulation_environment(scenario_cfgs[0])
        self.fp_sim_env = SF
        resultsPath = self.fp_sim_env.dir_names[G_DIR_OUTPUT]
        sim_time_offset = self.fp_sim_env.scenario_parameters.get(G_SUMO_SIM_TIME_OFFSET, 0)
        end_time = self.fp_sim_env.scenario_parameters[G_SIM_END_TIME]
        sim_time = self.fp_sim_env.scenario_parameters.get(G_SIM_START_TIME, 0)
        while True:
            leg_status_dict = self.fp_sim_env.step(sim_time)
            if sim_time % 120 == 0:
                print("{}: current simtime: {}/{}".format(self.fp_sim_env.scenario_parameters[G_SCENARIO_NAME], sim_time, end_time)) 
            sim_time += 1
            if sim_time > end_time:
                break
        self._post_sim_evaluation()

    def run_coupled_simulation(self):
        vehicle_to_position_dict = {}
        resultsPath = self.fp_sim_env.dir_names[G_DIR_OUTPUT]
        sim_time_offset = self.fp_sim_env.scenario_parameters.get(G_SUMO_SIM_TIME_OFFSET, 0)
        end_time = self.fp_sim_env.scenario_parameters[G_SIM_END_TIME]
        fp_time_step = self.fp_sim_env.scenario_parameters.get(G_SIM_TIME_STEP, 1)

        ##tt-retrieval (old)
        veh_edge_start_time_count = {}  #{(veh_id,edge,start_time):time_counter}
        veh_start_time_dict ={} #{(veh_id,start_time_on_current_edge)}
        veh_edge_dict ={} # {(veh_id,sim_time):edge_id,..}
        
        ##tt-retrieval (new)
        sim_pos_dict = {} # {sim_time:{veh_id:(edge,start_time_on_this_edge)}}
        res_list = [] # [(edge_id, start_time, end_time, veh_id)]
        

        # get vehicle types of the simulation vehicles
        self.fp_opvid_to_veh_type = {op_vid : veh.veh_type for op_vid, veh in self.fp_sim_env.sim_vehicles.items()} # {(op_id,veh_id):"veh_type"}

        #Check for old EdgeTravelTimes and delete them if they are still there
        if os.path.isfile(os.path.join(resultsPath, "EdgeTravelTimes", "new_travel_times.csv")):
            os.remove(os.path.join(resultsPath, "EdgeTravelTimes", "new_travel_times.csv"))

        ### SIMULATION
        active_pois = {}
        step = 0
        last_time = -1
        while True:
            # 1) fleetpy time step  
            sim_time = int(traci.simulation.getTime()) # sumo time in seconds
            sim_time += sim_time_offset
            if sim_time > end_time:
                break
            sim_time_ms = traci.simulation.getCurrentTime()
            if sim_time_ms/1000 % float(fp_time_step) == 0: # sumo time in seconds
                if sim_time != last_time: # avoid same time step again due to rounding
                    LOG.info(f"---- FleetPy Step ----- {sim_time_ms}")
                    leg_status_dict = self.fp_sim_env.step(sim_time) # fleetpy timestep and computing new routes TODO: Implement Different step Sizes for SUMO and FP
                    
                    #print(f"---- FleetPy Step ----- {sim_time}")
                    last_time = sim_time
                
            if sim_time % 120 == 0:
                print("{}: current simtime: {}/{}".format(self.fp_sim_env.scenario_parameters[G_SCENARIO_NAME], sim_time, end_time))
                           
            # 2) check for new routes and finished boarding processes
            arrivedVehicles_internal = self._update_routes_and_add_vehicles(sim_time)
            LOG.info(f"t={sim_time} Vehicles in Simulation: {len(traci.vehicle.getIDList())}, Vehicles in Teleportation: {len(traci.vehicle.getTeleportingIDList())}, Pending Vehicles: {len(traci.simulation.getPendingVehicles())}")
            LOG.info(f"Arrived Vehicles: {arrivedVehicles_internal}")
            LOG.info(f"Vehicles Starting Teleportation: {traci.simulation.getStartingTeleportIDList()}")
            LOG.info(f"Vehicles Ending Teleportation: {traci.simulation.getEndingTeleportIDList()}")
            LOG.info(f"Vehicles in Teleportation: {traci.vehicle.getTeleportingIDList()}")
            LOG.info(f"Vehicle fp_0_356 in SUMO: {traci.vehicle.getRoadID('fp_0_356') if 'fp_0_356' in traci.vehicle.getIDList() else 'not in SUMO'}, {traci.vehicle.getLanePosition('fp_0_356') if 'fp_0_356' in traci.vehicle.getIDList() else 'not in SUMO'}")
            LOG.info(f"Speed of Vehicle fp_0_356 in SUMO: {traci.vehicle.getSpeed('fp_0_356') if 'fp_0_356' in traci.vehicle.getIDList() else 'not in SUMO'}")
            LOG.info(f"Vehicle fp_0_356 in FleetPy: {vehicle_to_position_dict.get((0, 356), 'not in dict')}")
            LOG.info("Vehicles on Edge -140737948#2: "+str(traci.edge.getLastStepVehicleIDs("-140737948#2")))
            LOG.info("Vehicles on Edge -140738047#2: "+str(traci.edge.getLastStepVehicleIDs("-140738047#2"))) 
            
            # 3) sumo time step
            LOG.info(f"---- Traci Step ----- {sim_time}")
            if sim_time == 21683:
                    LOG.info("Removal at simtime:", traci.simulation.getTime(),
                    "Vehicles in Simulation:", len(traci.vehicle.getIDList()),
                    "Vehicles in Teleportation:", len(traci.vehicle.getTeleportingIDList()),
                    "Pending Vehicles:", len(traci.simulation.getPendingVehicles()),
                    f"Vehicles Starting Teleportation: {traci.simulation.getStartingTeleportIDList()}")
                    traci.vehicle.remove("fp_0_356")
                    arrivedVehicles_internal.update({'fp_0_356': 1740})
            try:
                traci.simulationStep()
            except Exception as e:
                print("Crash at simtime:", traci.simulation.getTime(),
                    "Vehicles in Simulation:", len(traci.vehicle.getIDList()),
                    "Vehicles in Teleportation:", len(traci.vehicle.getTeleportingIDList()),
                    "Pending Vehicles:", len(traci.simulation.getPendingVehicles()),
                    f"Vehicles Starting Teleportation: {traci.simulation.getStartingTeleportIDList()}")

                veh_speeds = [traci.vehicle.getSpeed(veh_id) for veh_id in traci.vehicle.getIDList()]
                print("Average Speed of Vehicles in Simulation:", np.mean(veh_speeds))

                for veh_id in traci.vehicle.getTeleportingIDList():
                    print("Teleporting Vehicle:", veh_id, traci.vehicle.getRoute(veh_id))
                    if veh_id.startswith("fp_"):
                        veh_id_fp = self._sumo_v_id_to_fleetpy_v_id(veh_id)
                        print(vehicle_to_position_dict.get(veh_id_fp, "not in dict"))
                        print(traci.vehicle.getLanePosition(veh_id))
                        traci.vehicle.remove(veh_id)
                raise e
            if sim_time == 21685:
                breakpoint()

            """
            # 4) get current vehicle positions and update travel time statistics (if needed)
            if sim_time%1==0 and self.g_update_fleetsim_traveltimes==True:
                sim_pos_dict,res_list = self._get_current_edge_tt(sim_time=sim_time,sim_pos_dict=sim_pos_dict,res_list=res_list)

            # 5) send new travel times to fleetsim
            if (sim_time%self.g_update_travel_statistics_time_step==0) and self.g_update_fleetsim_traveltimes==True:
                time_df = self._process_tt_data(res_list=res_list,sim_time=sim_time)
                time_update_dict = dict(zip(zip(list(time_df["from_node"]),list(time_df["to_node"])),zip(list(time_df["edge_tt"]),list(time_df["edge_var"]))))
            """
            # 4) get current vehicle positions and update travel time statistics (if needed)
            if sim_time%1==0 and self.g_update_fleetsim_traveltimes==True:
                sim_pos_dict,res_list = self._get_current_edge_tt(sim_time=sim_time,sim_pos_dict=sim_pos_dict,res_list=res_list)
            
            # 5) send new travel times to fleetsim
            if (sim_time%self.g_update_travel_statistics_time_step==0) and self.g_update_fleetsim_traveltimes==True:
                time_df = self._process_tt_data(res_list=res_list,sim_time=sim_time)
                time_update_dict = dict(zip(zip(list(time_df["from_node"]),list(time_df["to_node"])),zip(list(time_df["edge_tt"]),list(time_df["edge_var"]))))
                self._save_tt_to_csv(time_df, sim_time)
                veh_edge_start_time_count ={}
                veh_start_time_dict ={}
                veh_edge_dict ={}         
                if self.g_update_fleetsim_traveltimes==True:
                    self.fp_sim_env.update_network_travel_times(time_update_dict, sim_time)
                    self.fp_sim_env.routing_engine.load_tt_file_SUMO(resultsPath,sim_time)  

            # 6) collect the current positions of all fleet vehicles in SUMO
            vehicle_to_position_dict = self._get_current_vehicle_positions()
            LOG.info(vehicle_to_position_dict)
            #print(vehicle_to_position_dict)
            # 7) set the new positions in FleetPy
            self.fp_sim_env.update_vehicle_positions(vehicle_to_position_dict,sim_time)

            # 8) check for vehicles that arrived at their destination
            self._update_arrived_vehicles(arrivedVehicles_internal,sim_time)

            if self.sumo_binary == "sumo-gui":
                active_pois = self._show_idle_vehicles_gui(active_pois)

            step+=1
        traci.close()
        self._post_sim_evaluation()
    
    def _post_sim_evaluation(self):
        t1_stop = perf_counter()
        time_elapsed = []
        time_elapsed.append(t1_stop)
        timefile = self.fp_sim_env.dir_names[G_DIR_OUTPUT]+"Computationaltime.csv"

        with open(timefile, 'w', newline = '') as csvfile:
            my_writer = csv.writer(csvfile, delimiter = ' ')
            my_writer.writerow(time_elapsed)

        evaluation_start_time = self.fp_sim_env.scenario_parameters.get(G_EVAL_INT_START,self.fp_sim_env.scenario_parameters.get(G_SIM_START_TIME))
        evaluation_end_time = self.fp_sim_env.scenario_parameters.get(G_EVAL_INT_END,self.fp_sim_env.scenario_parameters.get(G_SIM_END_TIME))

        eval.standard_evaluation(self.fp_sim_env.dir_names[G_DIR_OUTPUT], evaluation_start_time =evaluation_start_time, evaluation_end_time =evaluation_end_time, print_comments=True, dir_names_in = {})
        eval.evaluate_folder(self.fp_sim_env.dir_names[G_DIR_OUTPUT],evaluation_start_time = evaluation_start_time, evaluation_end_time = evaluation_end_time, print_comments = False)
        sys.stdout.flush()

    def  _update_routes_and_add_vehicles(self, sim_time):
        '''This functions reads a dict of routes with vehicleIDs as strings (key) and a list of edge (value) and sets the vehicle routes accordingly ->(vehicleID, [e1,e2,e3])
        :param vehicle_ids:  list string of sumo vehicle ids
        :return: None'''
        # Receive new routes from FleetPy
        route_dict = self.fp_sim_env.get_new_vehicle_routes(sim_time) ## Gets new Routes (Leg by Leg from FP) {(op,veh_no):[n1,n2,...],...}
        arrivedVehicles_internal = {}
        current_sumo_vehicle_ids_set = set(traci.vehicle.getIDList()) # Get all vehicles in SUMO
        current_sumo_teleporting_ids_set = set(traci.vehicle.getTeleportingIDList()) # Get all vehicles in SUMO that are teleporting
        for opid_vid_tuple in route_dict.keys():
            veh_obj = self.fp_sim_env.sim_vehicles[opid_vid_tuple]
            sumo_vid = self._fleetpy_v_id_to_sumo_v_id(opid_vid_tuple)
            route = route_dict[opid_vid_tuple]       
            sumoRoute = self._transform_route_fp_to_sumo(route)
            
            LOG.debug(f"New Route from FleetPy {opid_vid_tuple} {route} --> {sumo_vid} {sumoRoute}")
            route_name = "Route_"+str(sumo_vid)+"_"+str(sim_time) 
            # If route only consisted of internal edges, it would not be a sumoRoute

            if len(sumoRoute) > 0:                    
                
                traci.route.add(route_name, sumoRoute)  # TODO does this lead to an infinite amount of routes in long simulations? -->Yes but removal function is not yet included in traci and will probably included in SUMO 1.22.0  
                
                # A) Vehicle is already in Simulation or currently teleporting
                if (sumo_vid in current_sumo_vehicle_ids_set) or (sumo_vid in current_sumo_teleporting_ids_set):
                    edgeID = traci.vehicle.getRoadID(sumo_vid)
                    currentRoute = traci.vehicle.getRoute(sumo_vid)

                    ## SUMO-Route Update needed?
                    if sumoRoute != currentRoute: ## Route needs to be updated because of an new order of fleetpy/teleport
                        #print("Route Update in SUMO",sumo_vid,"@",edgeID,currentRoute,"-->",sumoRoute)
                        is_valid_route = True
                        try:
                            traci.vehicle.setRoute(sumo_vid,sumoRoute)
                            if traci.vehicle.isRouteValid(sumo_vid) is False:
                                LOG.warning(f'Route of {sumo_vid} is not valid') # No occurence
                                is_valid_route = False
                            else:
                                traci.vehicle.setParameter(objID=sumo_vid,param="cleg_dest",value=sumoRoute[-1])
                                traci.vehicle.setParameter(objID=sumo_vid,param="cleg",value=sumoRoute)
                        except:
                            LOG.warning(f'Route of {sumo_vid} could not be set to: {sumoRoute}')
                            #print(f'Route of {sumo_vid} could not be set to: {sumoRoute}')
                            is_valid_route = False
                        
                        if is_valid_route == False:
                                LOG.warning(f"Vehicle {sumo_vid} has an invalid route {sumoRoute}")
                                LOG.info("Use SUMO rerouter")
                                try:
                                    traci.vehicle.changeTarget(sumo_vid, sumoRoute[-1])
                                    #traci.vehicle.rerouteTraveltime(sumo_vid)
                                    LOG.info(f"Vehicle {sumo_vid} has been rerouted to {sumoRoute[-1]} on {traci.vehicle.getRoute(sumo_vid)}")
                                    #print(f"Vehicle {sumo_vid} has been rerouted to {sumoRoute[-1]} on {traci.vehicle.getRoute(sumo_vid)}")

                                except:
                                    LOG.warning(f"Vehicle {sumo_vid} could not be rerouted")
                                    breakpoint()
                    else:
                        LOG.debug(f"Vehicle {sumo_vid} is Loaded and in Network and SumoRoute {sumoRoute} is current Route {currentRoute}")
                        pass 
                        

                # B) Vehicle is not in the simulation, but already loaded and waiting to be inserted (pending) --> no new route needed
                elif (sumo_vid not in current_sumo_vehicle_ids_set) and (sumo_vid in traci.simulation.getPendingVehicles()): 
                    LOG.debug(f"{sumo_vid}/{self._sumo_v_id_to_fleetpy_v_id(sumo_vid)} has to wait to get inserted at Edge {traci.vehicle.getRoute(sumo_vid)[0]}") 
                
                # C) Vehicle not in Simualtion: Try to Load Vehicle and insert it in the simulation    
                else: 
                    if self.sumo_binary == "sumo":
                        try:
                            traci.vehicle.addFull(vehID=sumo_vid, routeID=route_name, typeID=self.fp_opvid_to_veh_type[opid_vid_tuple])   
                        except:
                                LOG.debug(f'Vehicle {sumo_vid} could not be added')#No occurence
                                print(f'Vehicle {sumo_vid} could not be added')
                                print(traci.simulation.getLoadedIDList())
                                print(traci.simulation.getEndingTeleportIDList())
                                print(traci.simulation.getStartingTeleportIDList())
                                print(traci.vehicle.getTeleportingIDList())
                                print(sumo_vid in traci.vehicle.getIDList()) 
                    elif self.sumo_binary == "sumo-gui":
                    
                        try:
                            traci.vehicle.addFull(vehID=sumo_vid, routeID=route_name, typeID=self.fp_opvid_to_veh_type[opid_vid_tuple])
                            traci.vehicle.setParameter(objID=sumo_vid,param="Num_PAX",value=len([rq.get_rid_struct() for rq in veh_obj.pax]))
                            traci.vehicle.setParameter(objID=sumo_vid,param="PAX",value=[rq.get_rid_struct() for rq in veh_obj.pax])
                            traci.vehicle.setParameter(objID=sumo_vid,param="cleg_dest",value=sumoRoute[-1])
                            traci.vehicle.setParameter(objID=sumo_vid,param="cleg",value=sumoRoute)

                            LOG.info(f"Inserted Vehicle to SUMO: {sumo_vid},{route_name},{self.fp_opvid_to_veh_type[opid_vid_tuple]}")
                            if traci.vehicle.isRouteValid(sumo_vid) is False:
                                LOG.warning(f'Route of {sumo_vid} is not valid')
                    
                        except:
                                LOG.debug(f'Vehicle {sumo_vid} could not be added')#No occurence
                                print(f'Vehicle {sumo_vid} could not be added')
                                print(traci.simulation.getLoadedIDList())
                                breakpoint()
                                pass
                if sumo_vid in traci.simulation.getEndingTeleportIDList():
                    LOG.warning(f"SUMO-vehicle  {sumo_vid} ended to teleport in this timestep")      
            
            ## If the SUMO route is len(0) it consists only of internal edges. The vehicles  are immediately considered to be "arrived" an teleported to the node
            else:  
                LOG.info(f"veh {veh_obj.vid} @ {veh_obj.pos} gets internal route: {route}")
                if veh_obj.pos[0] == route[-1] and veh_obj.pos[1] == None:
                    LOG.debug(f"Vehicle gets Route to own position:{veh_obj.vid} {veh_obj.pos}")
                elif veh_obj.status == VRL_STATES.IDLE:
                    LOG.info(f"IDLE Vehicle with new Route found that is only internal: {veh_obj}, {veh_obj.cl_remaining_route} --> No Route Change, Vehicle will be rerouted in next step")
                else:
                    arrivedVehicles_internal.update({sumo_vid:route[-1]})  

        return arrivedVehicles_internal

    def _fleetpy_v_id_to_sumo_v_id(self,op_vid_tuple : Tuple[int, int]) -> str:
        """ converts the fleetpy vehicle id tuple (op_id, vid) into the sumo string id (fp_{op_id}_{v_id})"""
        if op_vid_tuple not in self.fp_to_sumo_veh_id_dict:
            self.fp_to_sumo_veh_id_dict[op_vid_tuple] = f"fp_{op_vid_tuple[0]}_{op_vid_tuple[1]}"
        return self.fp_to_sumo_veh_id_dict[op_vid_tuple]

    def _sumo_v_id_to_fleetpy_v_id(self,sumo_v_id_str : str) -> Tuple[int, int]:
        """ converts the sumo v_id into the fleetpy vehicle id (op_id, vid)"""
        if sumo_v_id_str not in self.sumo_to_fp_veh_id_dict:
            _, op_id, vid = sumo_v_id_str.split("_")
            self.sumo_to_fp_veh_id_dict[sumo_v_id_str] = (int(op_id), int(vid))
        return self.sumo_to_fp_veh_id_dict[sumo_v_id_str]
            
    def _transform_route_fp_to_sumo(self,route):
        sumoRoute = []
        for i in range(0, len(route)-1):
            o_node = route[i]
            d_node = route[i+1]
                
            #If edge is an internal edge it does not need to be added to the sumoRoute, if route only consisted of internal edges, it would not be a sumoRoute
            if o_node != d_node:
                edgeID = self.g_fs_edge_to_sumo_edge_id.get((o_node,d_node))
            else:
                edgeID = None

            if edgeID == None:
                    LOG.warning(f'There is a KeyError in the Route which is {o_node} -> {d_node} : {route}')
                    print(f'There is a KeyError in the Route which is {o_node} -> {d_node} : {route}')
            if edgeID != None and not edgeID.startswith(":"): # internal edges start with ":"
                sumoRoute.append(edgeID)
        return sumoRoute

    def _get_current_edge_tt(self,sim_time,sim_pos_dict,res_list):
        sim_vehicle_id_list = traci.vehicle.getIDList()
        sim_pos_dict[sim_time] = {}
        # Initialise the first time step
        if self.fp_sim_env.scenario_parameters.get(G_SIM_START_TIME, 0) == sim_time:
            for veh_id in sim_vehicle_id_list:
                edge = traci.vehicle.getRoadID(veh_id)
                sim_pos_dict[sim_time].update({veh_id:(edge,sim_time)})    ##sim_pos_dict: {sim_time:{veh_id:(edge,start_time_on_this_edge)}}
            
            
            return sim_pos_dict ,res_list
        
        if sim_pos_dict.get(sim_time-1) == None:
            raise ValueError(f"Error: sim_pos_dict at time {sim_time-1} is None")
            
       ## 1 Vehicles in Simualtion
        for veh_id in sim_vehicle_id_list:
            current_edge = traci.vehicle.getRoadID(veh_id)

           # A) Vehicle was not in simulation last time step --> Initialise on new edge 
            if sim_pos_dict[sim_time-1].get(veh_id) == None:
                sim_pos_dict[sim_time].update({veh_id:(current_edge,sim_time)})
           
           
           #B) Vehicle was in simulation last time step
            elif sim_pos_dict[sim_time-1].get(veh_id) != None:
               prev_time_step_edge = sim_pos_dict[sim_time-1].get(veh_id)[0]

               
               #B-1 Vehicle was in simulation last time step, but on other edge --> Save last edge and time, update current edge and time 
               if prev_time_step_edge != current_edge:
                   last_edge = sim_pos_dict[sim_time-1].get(veh_id)[0]
                   res_list.append((veh_id,last_edge,int(sim_pos_dict[sim_time-1].get(veh_id)[1]),int(sim_time)))
                   sim_pos_dict[sim_time].update({veh_id:(current_edge,sim_time)})
                
                #B-2 Vehicle was in simulation last time step, and ist still on the same edge --> Update time and copy edge-id and start time
               elif prev_time_step_edge == current_edge:
                   sim_pos_dict[sim_time].update({veh_id:sim_pos_dict[sim_time-1].get(veh_id)})

        ## 2  Vehicles that have reached their destination
        arrived_vehicle_id_list = traci.simulation.getArrivedIDList()
        for arr_vehicle in arrived_vehicle_id_list:
            if sim_time-1 not in sim_pos_dict.keys():
                break
            if sim_pos_dict[sim_time-1].get(arr_vehicle) != None:
                last_edge = sim_pos_dict[sim_time-1].get(arr_vehicle)[0]
                res_list.append((arr_vehicle,last_edge,int(sim_pos_dict[sim_time-1].get(arr_vehicle)[1]),int(sim_time)))

        
        
        # Delete old entries in sim_pos_dict to save memory
        if sim_time-2 in sim_pos_dict.keys():
            if sim_time -2 >= self.fp_sim_env.scenario_parameters.get(G_SIM_START_TIME, 0):
                del sim_pos_dict[sim_time-2]

        return sim_pos_dict,res_list

    def _get_hybrid_router_tt(self,tt_df,sim_time):
        if self.fp_sim_env.scenario_parameters.get("hybrid_router") == 0:
            return tt_df
        
        sim_hour = int(sim_time/3600)
        #tt_df["count"] = tt_df["count"] * 3600/int(self.fp_sim_env.scenario_parameters.get("sumo_statistics_interval"))
        tt_df = tt_df.reset_index(drop=True)
        tt_df["edge_id_str"] = tt_df["edge_id"].astype(str) 
        tt_df = pd.merge(left=tt_df, right= self.fp_hybrid_router_sc0_df, on="edge_id_str", how="left")
        p_opt = self.fp_scenario_config.get("p_opt", 1) if self.fp_scenario_config.get("p_opt", 1) is not None else 1
        tt_df["p_fco"] = np.where(
                                tt_df["count_sc0"] == None,
                                p_opt,
                                tt_df["count"] / tt_df["count_sc0"])   
        tt_df["hybrid_router_alpha"] = tt_df["p_fco"] / p_opt
        tt_df["hybrid_router_alpha"] = tt_df["hybrid_router_alpha"].clip(upper=1)
        tt_df = tt_df[tt_df["hour"] == sim_hour]
    
        historic_tt_df = self.fp_hybrid_router_hourly_tt_dict.get(str(sim_hour*3600))
        historic_tt_df.rename(columns={"edge_id":"edge","edge_tt":"edge_tt_historic"}, inplace=True)      
        historic_tt_df["edge_id_str"] = historic_tt_df["edge"].astype(str)
        tt_df = tt_df.merge(historic_tt_df, on="edge_id_str", how="left")
        tt_df["edge_tt_hybrid"] = tt_df["hybrid_router_alpha"] *tt_df["edge_tt"] +  (1-tt_df["hybrid_router_alpha"])*tt_df["edge_tt_historic"]
        tt_df.drop(columns=["Unnamed: 0", "index", "from_node_y", "to_node_y"], inplace=True,errors='ignore')
        tt_df.rename(columns={"from_node_x":"from_node","to_node_x":"to_node","edge_var_x":"edge_var"}, inplace=True)
        tt_df["edge_tt"] = tt_df["edge_tt_hybrid"].round(3)
        print("Hybrid Router: ",f"alpha: {np.average(tt_df['hybrid_router_alpha'])}",f'avg p:{np.average(tt_df["p_fco"])}')
        tt_df = tt_df[["edge_tt", "edge_var", "from_node", "to_node"]]
        tt_df = tt_df.dropna(subset=['edge_tt'])
        tt_df["edge_var"] = tt_df["edge_var"].fillna(0)

        return tt_df
    


    def _process_tt_data(self,res_list,sim_time):        
        tt_df = pd.DataFrame(res_list, columns=['veh_id','edge_id', 'starting_time', 'end_time'])
        if len(tt_df) == 0:
            return pd.DataFrame(columns=['from_node', 'to_node', 'edge_tt', 'edge_var'])
        tt_df["edge_tt"] = tt_df["end_time"] - tt_df["starting_time"] ## No correction term needed
        tt_df = tt_df[tt_df['edge_tt'] > 1] 

        tt_df = self._filter_by_fco_mode(tt_df)  

        tt_df = tt_df.groupby('edge_id').agg(edge_tt=('edge_tt', 'mean'), edge_var=('edge_tt', 'var'), count=('edge_tt', 'count')).reset_index()
        tt_df["edge_id"] = tt_df["edge_id"].apply(lambda x: self.g_sumo_edge_id_to_fs_edge.get(x, None))
        if 'edge_id' in tt_df.columns:
            tt_df = tt_df.dropna(subset=['edge_id'])
        tt_df["from_node"] = tt_df["edge_id"].apply(lambda x: x[0])
        tt_df["to_node"] = tt_df["edge_id"].apply(lambda x: x[1])
    
        tt_df["edge_tt"]=tt_df['edge_tt'].round(3)
        tt_df["edge_var"] = tt_df["edge_var"].fillna(0)
        tt_df["edge_var"]=tt_df['edge_var'].round(3)
        tt_df = self._get_hybrid_router_tt(tt_df,sim_time) 
        tt_df = tt_df[["from_node", "to_node", "edge_tt", "edge_var"]]
        return tt_df 

    def _save_tt_to_csv(self,tt_df, sim_time):
        resultsPath = self.fp_sim_env.dir_names[G_DIR_OUTPUT]
        if 'count' in tt_df.columns:
            tt_df.drop(columns="count",inplace=True)
        if not os.path.isdir(os.path.join(resultsPath, "EdgeTravelTimes")):
            os.mkdir(os.path.join(resultsPath, "EdgeTravelTimes")) 
        save_path = os.path.join(resultsPath, "EdgeTravelTimes", f"SUMO_travel_times_{sim_time}.csv")
        tt_df.to_csv(save_path)
        LOG.debug(f"SUMO Traveltimes sent to FP saved at: {os.path.join(resultsPath, 'EdgeTravelTimes', f'SUMO_travel_times_{sim_time}.csv')}")


    def _get_current_vehicle_positions(self):
        """ this function reads the positions of the specified vehicles from sumo and returns a dictionary
        :param vehicle_ids: list integer of sumo vehicle ids to read positions
        :param vehicle_to_position_dict: dictionary (operator_id, fleetsim vehicle id) -> fleetsim network position (tuple (o_node, d_node, frac_position) or (o_node, None, None))
                for a single operator operator_id = 0
        :return: """
        fleetsim_vehicles = self.fp_sim_env.get_vehicle_and_op_ids()
        vehicle_to_position_dict = {}
        current_sumo_vehicle_ids_set = set(traci.vehicle.getIDList())
        for opid_vid_tuple in fleetsim_vehicles:
            sumo_vid = self._fleetpy_v_id_to_sumo_v_id(opid_vid_tuple)
            
            ## A) Vehicle Moving on the Road --> Get Update 
            if str(sumo_vid) in current_sumo_vehicle_ids_set:
                LOG.debug("Vehicle is in IDList and position should be updated")
                currentLane = traci.vehicle.getLaneID(str(sumo_vid))
                laneLength = traci.lane.getLength(currentLane)
                currentLanePosition = traci.vehicle.getLanePosition(str(sumo_vid)) #returns something like: 20.267898023103246 (Other format needed?!)
                frac_position = currentLanePosition/laneLength
                currentEdge = traci.lane.getEdgeID(currentLane)
                if self.g_sumo_edge_id_to_fs_edge.get(currentEdge) is not None:
                    o_node, d_node = self.g_sumo_edge_id_to_fs_edge[currentEdge]
                    CurrentPosition = (o_node, d_node, frac_position)
                    vehicle_to_position_dict[opid_vid_tuple] = CurrentPosition
                else:
                    LOG.debug(f"Edge {currentEdge} not known in FP, no update this timestep")

            ## B) Vehicle Pending and Waiting to get Inserted --> Update relative Position to 0.000001
            elif sumo_vid in traci.simulation.getPendingVehicles():           
                pending_veh_pos = self.g_sumo_edge_id_to_fs_edge[traci.vehicle.getRoute(sumo_vid)[0]]
                vehicle_to_position_dict[opid_vid_tuple] = (pending_veh_pos[0],pending_veh_pos[1],0.000001)
            
            ## C) Not in Network and not Moving --> No Update
            else:
                LOG.debug(f"Position of {sumo_vid} remained the same")

            ## D) TODO: Consider teleported Vehicles

        return vehicle_to_position_dict

    def _update_arrived_vehicles(self,arrivedVehicles_internal,sim_time):
        arrivedVehicleIDs = list(traci.simulation.getArrivedIDList()) #vehicles that have reached destination and have been removed in this sim timestep
        arrivedVehicleIDs = [s for s in arrivedVehicleIDs if s.startswith("fp_")] #Only consider FleetPy Vehicles

        if len(arrivedVehicleIDs) == 0 and len(arrivedVehicles_internal)== 0:
            return
  
        LOG.info(f"Internal Arrivals: {arrivedVehicles_internal}")
        LOG.info(f"Normal Arrivals: {arrivedVehicleIDs}")

        # This should not happen normally as arrived vehicles should be removed by sumo itself:
        all_arrivedVehicleIDs = arrivedVehicleIDs + list(arrivedVehicles_internal.keys())
        current_sumo_vehicle_ids_set = set(traci.vehicle.getIDList())
        for sumo_vid in all_arrivedVehicleIDs:
            if sumo_vid in current_sumo_vehicle_ids_set:
               LOG.warning(f"vehicle {sumo_vid} is being forcefully removed from SUMO")
               traci.vehicle.remove(sumo_vid)
        
        arrival_dict = {}
        for sumo_vid in arrivedVehicleIDs:
            fp_vid = self._sumo_v_id_to_fleetpy_v_id(sumo_vid)
            veh_obj = self.fp_sim_env.sim_vehicles[fp_vid]
            
            if len(veh_obj.cl_remaining_route)> 0: 
                destination_node = veh_obj.cl_remaining_route[-1]

            ## Edge Case: Vehicle arrived in SUMO but not yet in FleetPy - happens in some cases with values of relative distance of last edge > 0.95
            else:
                destination_node = veh_obj.pos[1]
                LOG.info(f"{veh_obj.vid} Vehicle arrived in SUMO but not yet in FleetPy: Teleported to {(destination_node,None,None)}")


            arrival_dict.update({fp_vid:(destination_node,None,None)})
        
        ## Handling of Internal Arrivals
        for sumo_vid,dest in arrivedVehicles_internal.items():
            fp_vid = self._sumo_v_id_to_fleetpy_v_id(sumo_vid)
            veh_obj = self.fp_sim_env.sim_vehicles[fp_vid]
            LOG.info(f"Internal Arrival: {veh_obj}, {veh_obj.cl_remaining_route},{veh_obj.status}") ##TODO:  Teleport idle vehicles without letting them arrive "reached destination"

            arrival_dict.update({fp_vid:(dest,None,None)})

        self.fp_sim_env.update_vehicle_positions(arrival_dict, sim_time)
        self.fp_sim_env.vehicles_reached_destination(sim_time,list(arrival_dict.keys()))

    def _show_idle_vehicles_gui(self,active_pois):
        for sumo_pos in active_pois.keys():
            traci.poi.remove(sumo_pos)    
        active_pois = {}
        for (op_id, veh_id),veh_obj in self.fp_sim_env.sim_vehicles.items():
            if veh_obj.pos[1] == None: #Only for vehicles which are not moving
                op_veh_id = f"fp_{op_id}_{veh_id}"
                sumo_pos = self.g_fs_node_to_sumo_junction[veh_obj.pos[0]]

                if sumo_pos in active_pois.keys():
                    active_pois[sumo_pos].append(op_veh_id)

                else:  
                    active_pois[sumo_pos]=[op_veh_id]

        for sumo_pos,op_veh_ids in active_pois.items():
            junction_pos = traci.junction.getPosition(sumo_pos)
            traci.poi.add(sumo_pos,junction_pos[0],junction_pos[1]-5,poiType=' '.join(op_veh_ids),color=(0, 0, 0, 0))
        return active_pois

    def _filter_by_fco_mode(self,tt_df):
        fco_mode = self.fp_sim_env.scenario_parameters.get(G_SUMO_FCO_VEHICLES)
        seed = self.fp_sim_env.scenario_parameters[G_RANDOM_SEED]
        tt_df["veh_id"] = tt_df["veh_id"].astype(str)
        if fco_mode == "all":
            return tt_df
        else:
            ## Filter for specified FCO SAVs
            fco_sav_operators  = fco_mode.split("-")[0].split("_")[1:]
            sav_tt_df = tt_df[tt_df['veh_id'].str.extract(r'fp_(\d+)_')[0].isin(fco_sav_operators)]

            ##Filte for specified FCO PV-Share
            fco_pv_share = float(fco_mode.split("-")[1].split("_")[1])
            pv_tt_df = tt_df[tt_df['veh_id'].str.startswith(('pv', 'dv'))]
            number_of_fco_pvs = round(int(len(pv_tt_df)) * fco_pv_share)
            pv_tt_df = pv_tt_df.sample(n=number_of_fco_pvs, random_state=seed)
            tt_df = pd.concat([sav_tt_df,pv_tt_df])
            return tt_df

## Non-Implemented Functions:

def absolute_to_relative_position(vehID):
    LanePositionInMeter = traci.vehicle.getLanePosition(vehID)
    LaneID = traci.vehicle.getLaneID(vehID)
    LaneLength = traci.lane.getLength(LaneID)
    relativePosition = LanePositionInMeter/LaneLength
    return relativePosition   


if __name__ == "__main__":
    
    try:
        constant_config_path = sys.argv[1]
        scenario_config_path = sys.argv[2]
        sumo_config = sys.argv[3]
        if len(sys.argv) > 4:
            sumoBinary = sys.argv[4]
        else:
            sumoBinary = "sumo-gui"
        if len(sys.argv) > 5:
            log_level = sys.argv[5]
        else:
            log_level = "info"
    except:
        print("something is wrong with the input given: ", sys.argv)
        exit()


    SUMOFleetPyCoupling = SUMOFleetPyServer(constant_config_path=constant_config_path, scenario_config_path=scenario_config_path, sumo_config=sumo_config, sumoBinary=sumoBinary, log_level=log_level)
    if SUMOFleetPyCoupling.sumo_sim == False:
        SUMOFleetPyCoupling.run_fp_simulation()
    else:
        if SUMOFleetPyCoupling.sumo_sim == True:
            import traci._simulation
            import traci.constants as tc
            if sumoBinary == "sumo-gui":
                import traci
            else: 
                import libsumo as traci
                print("No GUI Needed. Using libsumo instead of traci for better performance.")
        SUMOFleetPyCoupling.setup_fleetsimulation()
        SUMOFleetPyCoupling.setup_traci()
        SUMOFleetPyCoupling.setup_network_translation()
        SUMOFleetPyCoupling.setup_hybrid_router()
        SUMOFleetPyCoupling.run_coupled_simulation()
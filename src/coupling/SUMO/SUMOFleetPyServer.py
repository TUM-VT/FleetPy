#!/usr/bin/python
# IN Traci conda env
from abc import abstractmethod
import os, sys

import traci 
import pandas as pd
import csv
import logging 
from operator import itemgetter
from time import perf_counter
from typing import Tuple
from datetime import datetime
from tqdm import tqdm
import xml.etree.ElementTree as ET
import time
import numpy as np
import pathlib
from src.coupling.SUMO.SUMOcontrolledSim import SUMOcontrolledSim
from src.coupling.SUMO.sumocfg_utils import merge_additional_files
from src.coupling.SUMO.tt_source import (
    coverage_error, layer_coverage_error, requested_bin_times, resolve_source_dir,
    tt_file_for, write_source_stats)
from src.misc.init_modules import load_simulation_environment
import src.misc.config as config
from src.misc.globals import *
import src.evaluation.standard as eval
from run_scenarios import run_scenarios
import random
import time


""" 
SUMOFleetPyServer - Coupled FleetPy-SUMO Simulation Server
===========================================================

This script creates a co-simulation coupling FleetPy (mobility-on-demand simulation) with SUMO (traffic simulation).
FleetPy handles customer requests, routing decisions, and fleet control logic, while SUMO simulates the actual 
vehicle movements in a microscopic traffic simulation environment.

DESCRIPTION:
-----------
The coupling works by synchronizing two simulation environments:
- FleetPy computes routes and manages fleet operations
- SUMO executes vehicle movements and provides realistic travel times
- Vehicles are dynamically created/removed in SUMO based on FleetPy route assignments
- Travel time feedback from SUMO updates FleetPy's routing engine in real-time

INPUT ARGUMENTS (Command Line):
-------------------------------
1. constant_config_path (str, required):
   Path to the FleetPy constant configuration file containing study-wide parameters
   (e.g., network paths, vehicle types, evaluation settings)

2. scenario_config_path (str, required):
   Path to the FleetPy scenario configuration file containing scenario-specific parameters
   (e.g., demand levels, fleet sizes, operator strategies)

3. sumo_config (str, required):
   Path to the SUMO configuration file (.sumocfg) defining the SUMO simulation setup
   (network, routes, simulation time, etc.)

4. sumoBinary (str, optional, default="sumo-gui"):
   SUMO executable to use: "sumo" for command-line or "sumo-gui" for graphical interface

5. log_level (str, optional, default="info"):
   Logging verbosity level: "verbose", "debug", "info", or "warning"

PREREQUISITES:
-------------
- FleetPy network must be synchronized with SUMO network (use preprocessing/networks/network_from_sumo.py)
- FleetPy demand files must created and matched to the network (use preprocessing/networks/demand_from_sumo.py)
- Vehicle types defined in FleetPy must exist in SUMO as vehicle type definitions
- SUMO_HOME environment variable should be set 

NAMING CONVENTIONS:
------------------
- fp_*  : FleetPy-related parameters and variables
- g_*   : Global parameters shared between FleetPy and SUMO
- sumo_*: SUMO-specific parameters and variables

OUTPUT FILES:
------------
- SumoDumps/: SUMO output files (TripInfo, vehRoutes, EdgeData, collisions, statistics)
- EdgeTravelTimes/: Time-series of travel time updates sent from SUMO to FleetPy
- Standard FleetPy evaluation outputs (KPIs, statistics, plots)
- Computationaltime.csv: Total simulation execution time

"""

LOG = logging.getLogger(__name__)
t_start = time.time()


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
        if self.fp_sim_env.scenario_parameters.get(G_SUMO_STAT_INT) is None:
            self.g_update_fleetsim_traveltimes = False
            self.g_sumo_t_update = 10000000000000
        else:
            self.g_update_fleetsim_traveltimes = True
            self.g_sumo_t_update = int(self.fp_sim_env.scenario_parameters.get(G_SUMO_STAT_INT))

        # Which travel times the fleet routes on. Unset = the probe measurements
        # this server takes itself; set = a directory of per-bin CSVs prepared
        # off-line (a current-state estimate or a prediction). Both go through
        # the same load_tt_file channel, so the arms differ by one parameter.
        self.g_tt_source_dir = resolve_source_dir(
            self.fp_sim_env.scenario_parameters.get(G_SUMO_TT_SRC_DIR),
            self.fp_sim_env.dir_names.get(G_DIR_MAIN))
        self.g_tt_bins_loaded = 0
        self.g_tt_bins_missing = 0
        # Run unconditionally: a time-dependent engine with NO source directory
        # is the one misconfiguration that produces a complete, plausible cell
        # that routed statically from end to end, and the old guard skipped
        # exactly that case because it only ran when a directory was set.
        self._check_tt_source_coverage()

    def _check_tt_source_coverage(self):
        """Refuse to start if the travel-time source cannot cover the run.

        A cell costs the better part of an hour, and a source directory whose
        file names sit off the request grid raises nothing at run time: every
        edge simply keeps its previous travel time, so the arm quietly behaves
        like the baseline it is meant to beat. Checking the whole grid here
        turns that into a one-second failure that names the missing bins.
        """
        params = self.fp_sim_env.scenario_parameters
        times = requested_bin_times(params[G_SIM_START_TIME], params[G_SIM_END_TIME],
                                    self.g_sumo_t_update)
        time_dependent = type(
            self.fp_sim_env.routing_engine).__name__.startswith("NetworkTimeDependent")
        if self.g_tt_source_dir is not None:
            problem = coverage_error(self.g_tt_source_dir, times,
                                     float(params.get(G_SUMO_TT_SRC_MIN_COV, 1.0)))
            if problem:
                raise FileNotFoundError(problem)
        elif not time_dependent:
            return      # R0: the server's own probe measurements, nothing to check
        # A time-dependent engine reads five more files per bin, and an unset
        # directory leaves it nothing at all. Neither fails at run time -- both
        # leave the arm routing on a base table alone, which is its static twin.
        if time_dependent:
            layer_problem = layer_coverage_error(self.g_tt_source_dir, times)
            if layer_problem:
                raise FileNotFoundError(layer_problem)
        LOG.info(f"travel-time source {self.g_tt_source_dir}: "
                 f"all {len(times)} requested bins present")

    def setup_sumo_simulation(self):
        """ 
        This function setups the SUMO simulation environment including the traci interface. SUMO parameters are read from the scenario config file. Default Values of SUMO are used if parameters are not given.
        """
        results_path = self.fp_sim_env.dir_names[G_DIR_OUTPUT]
        EdgeDataCfgPath = self._create_EdgeDataCfg_xml()  

        ## Create output directory for sumo dumps    
        if not os.path.isdir(os.path.join(results_path, "SumoDumps")):
            os.mkdir(os.path.join(results_path, "SumoDumps"))

        TripInfoPath = os.path.join(results_path, "SumoDumps", "TripInfo.xml.gz") 
        vehRoutePath = os.path.join(results_path, "SumoDumps", "vehRoutes.xml.gz") 
        collisionPath = os.path.join(results_path, "SumoDumps", "collisionPath.xml") 
        statisticsPath = os.path.join(results_path, "SumoDumps", "statistics.xml") 
        edges_output = os.path.join(results_path, "SumoDumps", "edge-output.xml")
  

        sumoCmd = [self.sumo_binary, "-c", self.sumo_config_path ,
                "--collision.action","warn",
                "--begin",str(self.fp_sim_env.scenario_parameters.get(G_SIM_START_TIME)),
                 # SUMO's -a REPLACES the sumocfg's <additional-files> rather
                 # than extending it, so passing EdgeDataCfgPath alone would
                 # silently drop the scenario's traffic-light programs, WAUT
                 # switching and public transport. Merge them back in.
                 "-a",merge_additional_files(self.sumo_config_path, EdgeDataCfgPath),
                "--step-length","1",
                "--tripinfo-output",TripInfoPath,
                "--vehroute-output",vehRoutePath,
                "--vehroute-output.exit-times","--vehroute-output.incomplete","--vehroute-output.write-unfinished","--vehroute-output.route-length",
                "--collision-output",collisionPath,
                "--statistic-output",statisticsPath,
                "--start", 
                "--seed", str(self.fp_sim_env.scenario_parameters[G_RANDOM_SEED]),
                "--no-warnings",str(True),
                "--route-steps", str(self.fp_sim_env.scenario_parameters.get(G_SUMO_ROUTE_STEPS, 200)),
                "--no-internal-links", str(self.fp_sim_env.scenario_parameters.get(G_SUMO_NO_INTERNAL_LINKS, False)),
                "--ignore-junction-blocker", str(self.fp_sim_env.scenario_parameters.get(G_SUMO_IGNORE_JUNCTION_BLOCKER, -1)),
                "--time-to-teleport", str(self.fp_sim_env.scenario_parameters.get(G_SUMO_TIME_TO_TELEPORT, 300)),
                "--time-to-teleport.highways", str(self.fp_sim_env.scenario_parameters.get(G_SUMO_TIME_TO_TELEPORT_HIGHWAYS, 0)),
                "--eager-insert", str(self.fp_sim_env.scenario_parameters.get(G_SUMO_EAGER_INSERT, False)),
                # Event-overlay cells block lanes mid-run. Edge 29119849#1
                # carries PT routes, so a blockage can leave a bus with no path
                # to its fixed stop, and SUMO treats that as fatal: the whole
                # co-simulation aborts partway through. This downgrades it to a
                # warning and drops the un-routable vehicle. Harmless on the
                # no-event cells, which have no un-routable vehicles to drop.
                "--ignore-route-errors", str(True),
                ]
     
        traci.start(sumoCmd)
        print(f"SUMO-Simulation Initialized at t={self.fp_sim_env.scenario_parameters.get(G_SIM_START_TIME)}")

    def _create_EdgeDataCfg_xml(self):
        additional = ET.Element("additional")
        edge_data = ET.SubElement(additional, "edgeData", {
            "id": str(self.fp_sim_env.scenario_parameters[G_SCENARIO_NAME]),
            "file": str(os.path.join(self.fp_sim_env.dir_names[G_DIR_OUTPUT], "SumoDumps", "EdgeData.xml.gz")),
            "begin": str(self.fp_sim_env.scenario_parameters.get(G_EVAL_INT_START,self.fp_sim_env.scenario_parameters.get(G_SIM_START_TIME))),
            "end": str(self.fp_sim_env.scenario_parameters.get(G_EVAL_INT_END,self.fp_sim_env.scenario_parameters.get(G_SIM_END_TIME))),
            "period": str(self.fp_sim_env.scenario_parameters.get(G_SUMO_EDGE_DATA_INTERVAL, 3600)),
            "withInternal": str(self.fp_sim_env.scenario_parameters.get(G_SUMO_EDGE_DATA_WITH_INTERNAL,True)),
            "excludeEmpty": str(self.fp_sim_env.scenario_parameters.get(G_SUMO_EDGE_DATA_EXCLUDE_EMPTY, True))
        }) 
        tree = indent_xml(additional)
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
    
    def run_coupled_simulation(self):
        vehicle_to_position_dict = {}
        resultsPath = self.fp_sim_env.dir_names[G_DIR_OUTPUT]
        end_time = self.fp_sim_env.scenario_parameters[G_SIM_END_TIME]
        fp_time_step = self.fp_sim_env.scenario_parameters.get(G_SIM_TIME_STEP, 1)

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
            if sim_time > end_time:
                break
            sim_time_float = traci.simulation.getTime()
            if sim_time_float % float(fp_time_step) == 0: # sumo time in seconds
                if sim_time != last_time: # avoid same time step again due to rounding
                    LOG.info(f"---- FleetPy Step ----- {sim_time_float}")
                    leg_status_dict = self.fp_sim_env.step(sim_time) # fleetpy timestep and computing new routes TODO: Implement Different step Sizes for SUMO and FP
                    last_time = sim_time
                
            if sim_time % 120 == 0:
                print("{}: current simtime: {}/{}".format(self.fp_sim_env.scenario_parameters[G_SCENARIO_NAME], sim_time, end_time))
                           
            # 2) check for new routes and finished boarding processes
            arrivedVehicles_internal = self._update_routes_and_add_vehicles(sim_time)

            # 3) sumo time step
            try:
                traci.simulationStep()
            except Exception as e:
                LOG.info(f"Crash at simtime: {traci.simulation.getTime()}")
                LOG.info(f"Vehicles in Simulation: {len(traci.vehicle.getIDList())}")
                LOG.info(f"Vehicles in Teleportation: {traci.vehicle.getTeleportingIDList()}")
                LOG.info(f"Pending Vehicles: {traci.simulation.getPendingVehicles()}")
                LOG.info(f"Vehicles Starting Teleportation: {traci.simulation.getStartingTeleportIDList()}")
                veh_speeds = [traci.vehicle.getSpeed(veh_id) for veh_id in traci.vehicle.getIDList()]
                LOG.info(f"Average Speed of Vehicles in Simulation: {np.mean(veh_speeds)}")
                for veh_id in traci.vehicle.getTeleportingIDList():
                    LOG.info(f"Teleporting Vehicle: {veh_id} Route: {traci.vehicle.getRoute(veh_id)}")
                    if veh_id.startswith("fp_"):
                        veh_id_fp = self._sumo_v_id_to_fleetpy_v_id(veh_id)
                        LOG.info(f"FP Position: {vehicle_to_position_dict.get(veh_id_fp, 'not in dict')}")
                        LOG.info(f"Lane Position: {traci.vehicle.getLanePosition(veh_id)}")
                        traci.vehicle.remove(veh_id)
                raise e

            # 4) get current vehicle positions and update travel time statistics (if needed)
            if sim_time%1==0 and self.g_update_fleetsim_traveltimes==True:
                sim_pos_dict,res_list = self._get_current_edge_tt(sim_time=sim_time,sim_pos_dict=sim_pos_dict,res_list=res_list)
            
            # 5) send new travel times to fleetsim
            if (sim_time%self.g_sumo_t_update==0) and self.g_update_fleetsim_traveltimes==True:
                time_df = self._process_tt_data(res_list=res_list,sim_time=sim_time)
                time_update_dict = dict(zip(zip(list(time_df["from_node"]),list(time_df["to_node"])),list(time_df["edge_tt"])))
                self._save_tt_to_csv(time_df, sim_time)

                res_list = []  # Clear res_list to prevent unlimited growth
                if self.g_update_fleetsim_traveltimes==True:
                    # The measured file is written either way (above), so every
                    # cell keeps the probe-measured travel times as provenance
                    # even when the fleet routes on something else.
                    measured_path = os.path.join(resultsPath, "EdgeTravelTimes", f"SUMO_travel_times_{sim_time}.csv")
                    tt_file_path = tt_file_for(self.g_tt_source_dir, sim_time, measured_path)
                    if tt_file_path is None:
                        # Skipping rather than falling back to measured_path: a
                        # fallback would mix the baseline into the arm under
                        # test one bin at a time, invisibly to every KPI.
                        self.g_tt_bins_missing += 1
                        LOG.warning(f"no travel-time file for bin {sim_time} in "
                                    f"{self.g_tt_source_dir}; edges keep their previous values")
                    else:
                        self.g_tt_bins_loaded += 1
                        self.fp_sim_env.routing_engine.load_tt_file(sim_time, ext_path=tt_file_path)

            # 6) collect the current positions of all fleet vehicles in SUMO
            vehicle_to_position_dict = self._get_current_vehicle_positions()

            # 7) set the new positions in FleetPy
            self.fp_sim_env.update_vehicle_positions(vehicle_to_position_dict,sim_time)

            # 8) check for vehicles that arrived at their destination
            self._update_arrived_vehicles(arrivedVehicles_internal,int(sim_time))

            if self.sumo_binary == "sumo-gui":
                active_pois = self._show_idle_vehicles_gui(active_pois)

            step+=1
        traci.close()
        LOG.info(f"travel times: {self.g_tt_bins_loaded} bins loaded, "
                 f"{self.g_tt_bins_missing} missing "
                 f"(source: {self.g_tt_source_dir or 'probe measurements'})")
        write_source_stats(resultsPath, self.g_tt_source_dir,
                           self.g_tt_bins_loaded, self.g_tt_bins_missing,
                           routing_engine=self.fp_sim_env.routing_engine)
        self._post_sim_evaluation()
    
    def _post_sim_evaluation(self):
        t_stop = time.time()
        time_elapsed = t_stop - t_start
        print(f"Simulation Time: {time_elapsed} seconds; {time_elapsed/60} minutes; {time_elapsed/3600} hours")         
        timefile = self.fp_sim_env.dir_names[G_DIR_OUTPUT] + "/Computationaltime.csv"
        with open(timefile, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['simulation_time_seconds'])
                writer.writerow([time_elapsed])
   
        evaluation_start_time = int(self.fp_sim_env.scenario_parameters.get(G_EVAL_INT_START,self.fp_sim_env.scenario_parameters.get(G_SIM_START_TIME)))
        evaluation_end_time = int(self.fp_sim_env.scenario_parameters.get(G_EVAL_INT_END,self.fp_sim_env.scenario_parameters.get(G_SIM_END_TIME)))
       
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
                                traci.vehicle.setParameter(objectID=sumo_vid, key="cleg_dest", value=sumoRoute[-1])
                                traci.vehicle.setParameter(objectID=sumo_vid, key="cleg", value=sumoRoute)
                        except:
                            LOG.warning(f'Route of {sumo_vid} could not be set to: {sumoRoute}')
                            #print(f'Route of {sumo_vid} could not be set to: {sumoRoute}')
                            is_valid_route = False
                        
                        if is_valid_route == False:
                                LOG.warning(f"Vehicle {sumo_vid} has an invalid route {sumoRoute}")
                                LOG.debug("Use SUMO rerouter")
                                try:
                                    traci.vehicle.changeTarget(sumo_vid, sumoRoute[-1])
                                    #traci.vehicle.rerouteTraveltime(sumo_vid)
                                    LOG.debug(f"Vehicle {sumo_vid} has been rerouted to {sumoRoute[-1]} on {traci.vehicle.getRoute(sumo_vid)}")
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
                                LOG.debug(f'Vehicle {sumo_vid} could not be added')
                                LOG.debug(traci.simulation.getLoadedIDList())
                                LOG.debug(traci.simulation.getEndingTeleportIDList())
                                LOG.debug(traci.simulation.getStartingTeleportIDList())
                                LOG.debug(traci.vehicle.getTeleportingIDList())
                                LOG.debug(sumo_vid in traci.vehicle.getIDList()) 
                    elif self.sumo_binary == "sumo-gui":
                    
                        try:
                            traci.vehicle.addFull(vehID=sumo_vid, routeID=route_name, typeID=self.fp_opvid_to_veh_type[opid_vid_tuple])
                            traci.vehicle.setParameter(objectID=sumo_vid, key="Num_PAX", value=len([rq.get_rid_struct() for rq in veh_obj.pax]))
                            traci.vehicle.setParameter(objectID=sumo_vid, key="PAX", value=[rq.get_rid_struct() for rq in veh_obj.pax])
                            traci.vehicle.setParameter(objectID=sumo_vid, key="cleg_dest", value=sumoRoute[-1])
                            traci.vehicle.setParameter(objectID=sumo_vid, key="cleg", value=sumoRoute)

                            LOG.debug(f"Inserted Vehicle to SUMO: {sumo_vid},{route_name},{self.fp_opvid_to_veh_type[opid_vid_tuple]}")
                            if traci.vehicle.isRouteValid(sumo_vid) is False:
                                LOG.warning(f'Route of {sumo_vid} is not valid')
                    
                        except:
                                LOG.debug(f'Vehicle {sumo_vid} could not be added')
                                LOG.debug(traci.simulation.getLoadedIDList())
                                pass
                if sumo_vid in traci.simulation.getEndingTeleportIDList():
                    LOG.warning(f"SUMO-vehicle  {sumo_vid} ended to teleport in this timestep")      
            
            ## If the SUMO route is len(0) it consists only of internal edges. The vehicles  are immediately considered to be "arrived" an teleported to the node
            else:  
                LOG.debug(f"veh {veh_obj.vid} @ {veh_obj.pos} gets internal route: {route}")
                if veh_obj.pos[0] == route[-1] and veh_obj.pos[1] == None:
                    LOG.debug(f"Vehicle gets Route to own position:{veh_obj.vid} {veh_obj.pos}")
                elif veh_obj.status == VRL_STATES.IDLE:
                    LOG.debug(f"IDLE Vehicle with new Route found that is only internal: {veh_obj}, {veh_obj.cl_remaining_route} --> No Route Change, Vehicle will be rerouted in next step")
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
    
    def _process_tt_data(self,res_list,sim_time):        
        tt_df = pd.DataFrame(res_list, columns=['veh_id','edge_id', 'starting_time', 'end_time'])
        if len(tt_df) == 0:
            return pd.DataFrame(columns=['from_node', 'to_node', 'edge_tt', 'edge_var'])
        tt_df["edge_tt"] = tt_df["end_time"] - tt_df["starting_time"] ## No correction term needed
        tt_df = tt_df[tt_df['edge_tt'] > 1] 
        tt_df = self._filter_by_fcd_mode(tt_df)  
        tt_df = tt_df.groupby('edge_id').agg(edge_tt=('edge_tt', 'mean'), edge_var=('edge_tt', 'var'), count=('edge_tt', 'count')).reset_index()
        tt_df["edge_id"] = tt_df["edge_id"].apply(lambda x: self.g_sumo_edge_id_to_fs_edge.get(x, None))
        if 'edge_id' in tt_df.columns:
            tt_df = tt_df.dropna(subset=['edge_id'])
        tt_df["from_node"] = tt_df["edge_id"].apply(lambda x: x[0])
        tt_df["to_node"] = tt_df["edge_id"].apply(lambda x: x[1])
    
        tt_df["edge_tt"]=tt_df['edge_tt'].round(3)
        tt_df["edge_var"] = tt_df["edge_var"].fillna(0)
        tt_df["edge_var"]=tt_df['edge_var'].round(3)
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
  
        LOG.debug(f"Internal Arrivals: {arrivedVehicles_internal}")
        LOG.debug(f"Normal Arrivals: {arrivedVehicleIDs}")

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
                LOG.debug(f"{veh_obj.vid} Vehicle arrived in SUMO but not yet in FleetPy: Teleported to {(destination_node,None,None)}")


            arrival_dict.update({fp_vid:(destination_node,None,None)})
        
        ## Handling of Internal Arrivals
        for sumo_vid,dest in arrivedVehicles_internal.items():
            fp_vid = self._sumo_v_id_to_fleetpy_v_id(sumo_vid)
            veh_obj = self.fp_sim_env.sim_vehicles[fp_vid]
            LOG.debug(f"Internal Arrival: {veh_obj}, {veh_obj.cl_remaining_route},{veh_obj.status}") ##TODO:  Teleport idle vehicles without letting them arrive "reached destination"

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

    def _filter_by_fcd_mode(self,tt_df):
        fcd_mode = self.fp_sim_env.scenario_parameters.get(G_SUMO_FCD_VEHICLES)
        seed = self.fp_sim_env.scenario_parameters[G_RANDOM_SEED]
        tt_df["veh_id"] = tt_df["veh_id"].astype(str)
        if fcd_mode == "all":
            return tt_df
        else:
            ## Filter for specified FCO SAVs
            fco_MOD_operators  = fcd_mode.split("-")[0].split("_")[1:]
            sav_tt_df = tt_df[tt_df['veh_id'].str.extract(r'fp_(\d+)_')[0].isin(fco_MOD_operators)]

            ##Filte for specified FCO PV-Share
            fco_pv_share = float(fcd_mode.split("-")[1].split("_")[1])
            pv_tt_df = tt_df[tt_df['veh_id'].str.startswith(('pv', 'dv'))]
            number_of_fco_pvs = round(int(len(pv_tt_df)) * fco_pv_share)
            pv_tt_df = pv_tt_df.sample(n=number_of_fco_pvs, random_state=seed)
            tt_df = pd.concat([sav_tt_df,pv_tt_df])
            return tt_df


def indent_xml(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_xml(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
            
            
def run_fleetpy_sumo_simulation(constant_config_path: str, scenario_config_path: str, sumo_config: str, sumoBinary: str = "sumo-gui", log_level: str = "info"):
    """
    This method creates a co-simulation coupling FleetPy (mobility-on-demand simulation) with SUMO (traffic simulation).
    FleetPy handles customer requests, routing decisions, and fleet control logic, while SUMO simulates the actual 
    vehicle movements in a microscopic traffic simulation environment.

    DESCRIPTION:
    -----------
    The coupling works by synchronizing two simulation environments:
    - FleetPy computes routes and manages fleet operations
    - SUMO executes vehicle movements and provides realistic travel times
    - Vehicles are dynamically created/removed in SUMO based on FleetPy route assignments
    - Travel time feedback from SUMO updates FleetPy's routing engine in real-time

    INPUT ARGUMENTS (Command Line):
    -------------------------------
    :param constant_config_path: Path to the FleetPy constant configuration file containing study-wide parameters (e.g., network paths, vehicle types, evaluation settings)
    :param scenario_config_path: Path to the FleetPy scenario configuration file containing scenario-specific parameters (e.g., demand levels, fleet sizes, operator strategies)
    :param sumo_config: Path to the SUMO configuration file (.sumocfg) defining the SUMO simulation setup (network, routes, simulation time, etc.)
    :param sumoBinary: SUMO executable to use: "sumo" for command-line or "sumo-gui" for graphical interface
    :param log_level: Logging verbosity level: "verbose", "debug", "info", or "warning"
    """
    SUMOFleetPyCoupling = SUMOFleetPyServer(constant_config_path=constant_config_path, scenario_config_path=scenario_config_path, sumo_config=sumo_config, sumoBinary=sumoBinary, log_level=log_level)
    SUMOFleetPyCoupling.setup_fleetsimulation()
    SUMOFleetPyCoupling.setup_sumo_simulation()
    SUMOFleetPyCoupling.setup_network_translation()
    SUMOFleetPyCoupling.run_coupled_simulation()



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
        
    run_fleetpy_sumo_simulation(constant_config_path, scenario_config_path, sumo_config, sumoBinary, log_level)
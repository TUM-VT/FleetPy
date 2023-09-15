# Fleetpy - expansion for external cost optimized routing and external cost evaluation based on a Munich used case (network and demand)

## According Publication

Schröder, D., Mor, K., 2023. Beyond the Price Tag: Quantifying the External Costs of Autonomous On-Demand Ride Pooling Services using Agent-Based Simulation Analysis. Transportation Research: Part A. (submitted)

## Adaption in Source Code:

* Modification of Routing and Vehicle Movement: using standard bidirectional Dijkstra Algorithm with external cost as objective (external cost as edge value)
* Modification of Batch Optimization: assignment of requests to vehicles and pooling based on Alonso-Mora, their objective was also changed to external costs

## Needed Input

*Different network files for each vehicle including external cost value as 6th column in the file edges.csv
*Vehicle files including information about mass, cw, size, etc. in ___vehtype.csv
*Demand files for the simulated network


## Configuration files 

*for choosing external costs as optimization objective for routing and pooling the following configuration is needed:
	* network_type = NetworkBasic_Ext_Routing
	* op_vr_control_func_dict = func_key:total_customized_travel_costs;vot:0.45

*for choosing Scenarios without external cost routing or pooling, but still evaluation of external costs is possible:
	* network_type = NetworkBasic_Ext
	* op_vr_control_func_dict = func_key:distance_and_user_times_with_walk;vot:0.45



## Run Simulation

*run_munich_study.py is configurated for a variation of different scenarios with and without external cost routing and pooling
*the run___ file can be adapted to different configuration files that are prepared in the studies folder


## Additional auxiliary scripts for creating network files with external costs, demand files for Munich in the correct format and evaluation of results

*create network files: create_network_files_DS.py, create_network_files_with_extcost_DS.py, 
*create configuration files for different scenarios: create_scenario_files_munich_study.py
*create vehicle files: create_vehicle_files_DS.py
*create demand: convert_demand_files_from_pkl_to_csv.py, create_demand_files.py, create_demand_files_from_MITO.py, get_distance_from_demand_files.py
*evaluation of results: calculate_external_costs_for_trip.py, evaluate_munich_scenarios_DS.py, visualize_network.py
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
- FleetPy demand files must created and matched to the network (use preprocessing/demand/demand_from_sumo.py)
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

Tested with SUMO version 1.26.0
# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2025-12-18

Key update:
1. Refactored routing file structure: road-related routing modules are moved to the `road` subdirectory, and PT-related routing modules are moved to the `pt` subdirectory.
2. Introduced a C++ routing module based on the RAPTOR algorithm for querying the fastest PT travel plans between two stations.
3. Introduced the PTControl module to simulate PT operator behavior, such as recording offer information and dynamically updating GTFS files.
4. Introduced the PTBroker module based on the TPCS strategy to simulate MaaS platform planning for intermodal requests.
5. Introduced subrequest ID coding rules for intermodal scenarios, using unique integers to classify legs and `{parent_rid}_{subtrip_id}` to define new subrequest IDs.

### Added
- cpp_raptor_router: C++ implementation of the PT router based on the RAPTOR algorithm

- RaptorRouterCpp: Python entry point for the PT router, responsible for activating the cpp_raptor_router instance and regularizing PT requests

- PTControlBase & PTControlBasic: Modules to simulate PT operator behavior, such as recording offer information and dynamically updating GTFS files

- PTOffer: Class for recording PT offer information

- IntermodalOffer: Class for recording offer information for intermodal requests

- BasicIntermodalRequest: Class to simulate travel behavior of travelers with intermodal requests

- example_100_intermodal.csv: Intermodal demand based on example_100.csv, containing 25 monomodal, 25 first-mile, 25 last-mile, and 25 first-last-mile requests

- example_gtfs: Public transport design based on example_network

- PTBrokerTPCS: PTBroker module based on TPCS strategy for intermodal planning. Use `G_BROKER_TPCS_USE_DEFAULT` to toggle between full three-stage or first-stage only

- intermodal_evaluation: Evaluation methods designed for intermodal scenarios

- example_study & module_tests: Added intermodal scenario example experiments

- globals: Added subrequest ID coding rules for intermodal scenarios and global variables for PT, Intermodal, and TPCS

- init_modules: Added initialization code for PTControl and PTBroker modules

- FleetSimulationBase: Added code to load PTControl and PTBroker modules

- PlanRequest: Added `set_new_dropoff_time_constraint` method to update the passenger's latest drop-off time constraint

- Demand: Added `create_sub_requests` method to establish sub-requests for corresponding legs of intermodal requests

- RequestBase: Added `modal state` attribute (default: monomodal) and `get_modal_state` method

- PTRouterGTFSPreperation: Jupyter notebook for cleaning and formatting raw GTFS data for RaptorRouterCpp

### Changed
- BrokerBase & BrokerBasic: `collect_offers` method now accepts an input variable `sim_time` (int, default: None)

- ImmediateDecisionsSimulation: Added `sim_time` input variable when calling `self.broker.collect_offers`

- RequestBase: Updated `create_SubTripRequest` method to create intermodal sub-requests

- insertion: `insertion_with_heuristics` and `reservation_insertion_with_heuristics` methods added `excluded_vid` input (list of vehicle IDs that should not be considered for assignment)

- RollingHorizon: `return_immediate_reservation_offer` method added `excluded_vid` input

- PoolingIRSOnly: `user_request` method now adds the vehicle used for the FM leg to `excluded_vid` when processing FLM requests, ignoring this vehicle for the LM leg of the same parent request

- Vehicles: `assign_vehicle_plan` method now uses `rid_struct` to obtain request information

- FleetSimulationBase: Modified public transportation module loading code; modified `evaluate` method to use `G_EVAL_METHOD` to specify standard result evaluation (default: standard_evaluation)

- gitignore: Ignored specific C++ Router files

### Deprecated
- globals: Traveler modal state global variables are no longer used (G_RQ_STATE_MONOMODAL, G_RQ_STATE_FIRSTMILE, G_RQ_STATE_LASTMILE, G_RQ_STATE_FIRSTLASTMILE)

### Removed
- globals: Traveler modal state global variable names

### Fixed
- RollingHorizon: Correctly retrieve `vid` and `veh_obj` in `user_cancels_request` method

- FleetSimulationBase: In `update_sim_state_fleets`, ensured `rid_struct` is the actual key of the dictionary returned by `veh_obj.update_veh_state`



## [1.0.0] - 2025-04-DD

Key update:
all packages are updated for python 3.10 with pandas 2 and gurobi 12.
A corresponding install file is provided.

### Added
- module_tests study: configs to test every module in this repository -> need to be run before a pull request is accepted

- add example studies for Manhattan, Chicago, and Munich

- ForecastZoning: New base class as subclass of ZoneSystem for demand forecasts

- AggForecastZoning: implementation from former Zoning.py now for ForecastZoning -> read zone-based departure and arrival forecasts from file

- MyopicForecastZoneSystem: implementation of ForecastZoning: no data input needed; forecast of od-specific expected trips from actual requests in the past time intervall

- PerfectForecastZoneSystem: implementation of ForecastZoning: no data input needed; read aggregated (zonal, time interval) future demand directly from input demand

- PerfectOMyopicDForecast:implementation of ForecastZoning:  Mix of Myopic and perfect forecast: Origins are directly extracted from demand file, corresponding destinations from past requests

- PerfectORandomDForecast: implementation of ForecastZoning: same as PerfectOMyopicDForecast but destinations are drawn randomly for forecast

- NetworkZoning: includes all network related zone functionality after splitting zoning in forecast + network

- cpp_router_checker: small script to check if cpp compilation succeeded and python/cpp router return same results

- FullSamplingRidePoolingRebalancingMultiStage -> Sampling algorithm from thesis

- LinearHailingRebalancing/PavoneContinous -> Benchmark repo algorithms from thesis (Wallar et al. 2018 / Zhang et al. 2016)

- SimonettoAssignment: -> Linear batch assignment algorithm (Simonetto et al. 2019)

- RollingHorizonNoGuarantee -> Reservation rolling horizon method without service guarantee

- ContinousBatchRevelationReservation: two-horizon methond for reservation treatment with arbitrary reservation time

- Reinforcement learning wrapper for gymnasium

- SemiOnDemandBatchAssignmentFleetcontrol: Fleetcontrol class for semi-flexible service design


### Changed
- FleetSimulationBase: config param for showing progress bar ("show_progress_bar"), param adoption for separation of zone system and forecast zones, adoption of dir_dicts for operators (specific data to load for each operator)

- replay plotting: change state order (first route, last repo), remove wiggeling in video

- TravelerModels: same rounding of reservation time as request time

- standard.py: adopt input data paths to new operator path structure (different input paths for different operators possible)

- BrokerAndExchangeFleetControl, PoolingIRSBatchOptimization, PoolingIRSOnly, RPPFleetControl: small adoption for new repositioning functionality

- FleetControlBase: Functionality for operators to load its own network for routing (e.g. no conflict with network when tt forecasts are needed), assignment records for optional output (tracking of assigned vid, epa, edt for rids at each time step) -> param "G_OP_REC_ADD_ASS", remove self.zones -> new module "forecast strategy"

- RidePoolingBatchAssignmentFleetcontrol: updating offered pickup time interval, add "vid" to offer

- RidePoolingBatchOptimizationFleetControlBase: add parameter "G_OP_LOCK_VID, G_OP_LOCK_VID_TIME" to lock rq to current vehicle and at a specific time before scheduled pick-up

- PlanRequest: Functionality to updated time constraints after loading new tts, ArtificialPlanRequest for planning (now traveller object needed for init)

- VehiclePlan: return_intermediary_plan_state updated for speed-up (init_plan_state possible to reduce nr planstops to check)

- GeneralPoolingFunctions: include RR check for onboard requests

- objectives: parameters for dynamically adopting assignment_reward to prohibit it getting to large, add reassignment_penalty option, add function for different treatment of odm and reservation requests, embedded_control_f for better logging and debugging

- BatchAssignmentAlgorithmBase: update for registering time constraint updates of users, make it usable outside the embedding in fleetcontrol

- AlonsoMoraAssignment: move Key functions to different files, additional heuristic parameters for maximum number of schedules per v2rb, maximum number of rqs for exhaustive DARP solving, rebuilding from scratch; more output for dynamic file (eg, nr rqs, v2rbs); rr computations in methods, rr computations done only when needed, update building trees with time-outs: always starting with assigned tree (and always building that), only then building other trees; 

- misc: includes now alonsomora assignment functions

- BatchInsertionHeuristicAssignment: Update assignment process to be in line with locked or non locked repo stops

- insertion: speed up of insertion process ~factor 2 by reducing number of created PlanStops

- AlonsoMoraRepo, FrontiersDensityBasedRepositioning, PavoneHailingFC, : small update for new forecast class

- RepositioningBase: small update for new forecast class, od_assignment with reservation + repo planstops

- ReservationBase: move some reservation methods for revelation-based reservations here

- RollingHorizon: add method for treating upcoming reservation requests

- Zoning: Only keep methods defining any arbitrary zone (network or forecast)

- globals: define new input parameters, adopt directory_dict for operator distinctive input data

- init_modules: add type checking

- NetworkBasic and other Network classes: update to allow travel_time_factors or edge specific travel times

- Network.cpp: add trim functionality for data path adaption

- example configs: update zone system parameters

### Deprecated
[comment]: # Description of features that are deprecated.

### Removed
[comment]: # Description of features that have been removed.

### Fixed
- AlonsoMoraAssignment: use very small gurobi mipgap now -> could result in suboptimal assignments previously because of large assignment_reward

- V2RB: bugfixing for creatingLowerV2RB



## [0.2.0] - 2022-06-09

### Added
- Initial release.

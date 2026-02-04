# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FleetPy is an open-source fleet simulation framework for modeling and controlling vehicle fleets in ride-sharing, autonomous mobility, and on-demand transport scenarios. This branch (`chenhao/ptbroker-vibe-coding`) focuses on **PTBroker integration** - extending FleetPy for intermodal mobility-as-a-service (MaaS) combining autonomous on-demand (AMoD) services with public transportation (PT).

## Build & Run Commands

```bash
# Environment setup
conda env create -f environment.yml
conda activate fleetpy

# Build C++ routers (recommended for performance)
cd src/routing/road/cpp_router && python setup.py build_ext --inplace
cd src/routing/pt/cpp_raptor_router && python setup.py build_ext --inplace

# Run simulations
python run_examples.py                              # All example scenarios
python run_examples.py --study studies/example_study  # Specific study

# Run module tests (QA)
python studies/module_tests/run_module_tests.py

# Visualize results
python replay_pyplot.py <scenario_result_directory> [sim_seconds_per_real_second]

# Temporal analysis
python src/evaluation/temporal.py <scenario_result_directory>
```

## Architecture

### Modular Plugin System
All major components are loaded dynamically via `src/misc/init_modules.py` dictionaries:
- `get_src_simulation_environments()` - Simulation modes
- `get_src_routing_engines()` - Road/PT routing variants
- `get_src_request_modules()` - 24+ traveler models
- `get_src_fleet_control_modules()` - 20+ fleet strategies
- `get_src_broker_modules()` - Broker platforms

Custom extensions go in a `dev/` directory and get auto-discovered.

### Class Hierarchy

**Simulation** (`src/`):
- `FleetSimulationBase.py` → `BatchOfferSimulation`, `ImmediateDecisionsSimulation`, `BrokerSimulation`

**Broker/Platform** (`src/broker/`):
- `BrokerBase.py` (abstract) → `BrokerBasic` → `PTBrokerBasic` → `PTBrokerEI`
- PTBrokerEI is the latest: estimation-based integration for intermodal trips

**Fleet Control** (`src/fleetctrl/`):
- `FleetControlBase.py` (abstract) → 20+ implementations (pooling, charging, pricing, etc.)

**Demand** (`src/demand/`):
- `TravelerModels.py` - Request types including `BasicIntermodalRequest` for PT+AMoD

### Event-Driven Loop
```
FleetSimulationBase.run():
  For each time step:
    → Process user requests
    → Broker collects offers from operators
    → Users make decisions (accept/reject)
    → Update fleet status
    → Execute routing & assignment
    → Log results
```

### Key Data Models
- **Requests**: Immutable, multiple types (Basic, Intermodal, Parcel, etc.)
- **Offers**: `TravellerOffer`, `PTOffer`, `IntermodalOffer` in `src/simulation/Offers.py`
- **Vehicles**: Mutable state with route plans as `VehicleRouteLeg` objects
- **Sub-requests**: Decompose intermodal trips into modal legs

### Intermodal Trip Decomposition
For `BasicIntermodalRequest`, trips are categorized by `modal_state_value`:
- `0` (DRT_only): Direct AMoD trip, no PT
- `1` (FM - First Mile): AMoD → PT station → PT to destination
- `2` (LM - Last Mile): PT from origin → PT station → AMoD
- `3` (FLM - First & Last Mile): AMoD → PT → AMoD

Sub-requests (`sub_trip_id` field) are created for each leg:
- Parent request: `is_parent_request=True`, aggregated stats in `1_user-stats_parent.csv`
- Sub-requests: `is_parent_request=False`, individual leg details in `1_user-stats.csv`

## Configuration

Scenarios are configured via CSV files in `studies/*/scenarios/`:
- `constant_config.csv` - Fixed parameters
- `scenario.csv` - Variable parameters for scenario variations

Key parameters (150+ documented in `Input_Parameters.md`):
- `sim_env` - Simulation class
- `rq_type` - Request model (BasicRequest, BasicIntermodalRequest, etc.)
- `op_module` - Fleet control strategy
- `broker_module` - Broker class (BasicBroker, PTBrokerBasic, PTBrokerEI)
- `network_type` - Routing engine
- `gtfs_name`, `pt_type` - Public transport configuration

## Important Files

| Path | Purpose |
|------|---------|
| `src/misc/globals.py` | 400+ constants (G_*), enums (RQ_MODAL_STATE, etc.) |
| `src/misc/init_modules.py` | Dynamic module loading dictionaries |
| `src/broker/PTBrokerEI.py` | Latest PT broker with estimation-based integration |
| `src/demand/TravelerModels.py` | All request types and decision logic |
| `src/evaluation/intermodal.py` | Intermodal-specific metrics |
| `Input_Parameters.md` | Complete parameter documentation |

## Output Files

Results are stored in `studies/<study_name>/results/<scenario_name>/`:

| File | Description |
|------|-------------|
| `00_config.json` | Complete scenario configuration with all parameters |
| `00_simulation.log` | Simulation execution log |
| `1_user-stats.csv` | Per-request data: pickup/dropoff times, fares, modal state |
| `1_user-stats_parent.csv` | Parent request stats for intermodal (aggregates sub-trips) |
| `2-<op_id>_op-stats.csv` | Vehicle task log: status, positions, occupancy, routes per timestep |
| `2_vehicle_types.csv` | Vehicle type definitions |
| `3-<op_id>_op-dyn_atts.csv` | Dynamic fleet attributes over time |
| `standard_eval.csv` | Aggregated KPIs per operator |
| `standard_mod-<op_id>_veh_eval.csv` | Per-vehicle evaluation metrics |
| `final_state.csv` | End-of-simulation fleet state |

### Key Evaluation Metrics

**Standard scenarios** (`standard_eval.csv`):
- `number users`, `modal split` - Demand served
- `waiting time`, `detour time`, `travel time` - Service quality
- `% fleet utilization`, `occupancy`, `% empty vkm` - Fleet efficiency
- `mod revenue`, `mod fix costs`, `mod var costs` - Economics
- `shared rides [%]` - Pooling effectiveness

**Intermodal scenarios** (with `evaluation_method: intermodal_evaluation`):
- `DRT_only_count`, `FM_count`, `LM_count`, `FLM_count` - Trip type breakdown
- `DRT_only_service_rate`, `FM_service_rate`, `LM_service_rate`, `FLM_service_rate` - Per-type service rates
- `PT_Wait_Time_*`, `LM_Wait_Time_*` - Modal-specific wait times
- `op0_fleet_utilization`, `op0_occupancy`, `op0_empty_vkm` - AMoD operator metrics

### User Stats Columns

Key columns in `1_user-stats.csv`:
- `request_id`, `sub_trip_id` - IDs (sub_trip_id for intermodal legs)
- `is_parent_request` - True for original request, False for sub-requests
- `rq_time`, `earliest_pickup_time` - Request timing
- `start`, `end` - Origin/destination node IDs (format: `node_id;-1;-1`)
- `offers` - Encoded offer string: `op_id:t_wait:X;t_drive:Y;fare:Z;vid:V`
- `modal_state` / `modal_state_value` - Trip mode (0=DRT_only, 1=FM, 2=LM, 3=FLM)
- `pickup_time`, `dropoff_time`, `fare` - Actual service details

### Vehicle Stats Columns

Key columns in `2-*_op-stats.csv`:
- `status` - Vehicle state: `route`, `boarding`, `idle`, `charging`
- `start_pos`, `end_pos` - Position at task start/end
- `driven_distance` - Distance traveled in this task
- `start_soc`, `end_soc` - Battery state of charge (for EVs)
- `rq_on_board` - Request IDs currently in vehicle
- `occupancy` - Number of passengers
- `route`, `trajectory` - Node sequence and timestamps

## Configuration Format

### constant_config.csv
Two-column CSV with `Input_Parameter_Name` and `Parameter_Value`:
```csv
Input_Parameter_Name,Parameter_Value
sim_env,ImmediateDecisionsSimulation
network_name,example_network
op_max_wait_time,300
op_vr_control_func_dict,func_key:distance_and_user_times_with_walk;vot:0.45
```

### scenario.csv
Multi-column CSV where each row is a scenario variation:
```csv
scenario_name,op_module,rq_file,broker_type,broker_maas_detour_time_factor
example_im_ptbroker,PoolingIRSOnly,example_100_intermodal.csv,PTBroker,
example_im_ptbrokerEI_mdtf30,PoolingIRSOnly,example_100_intermodal.csv,PTBrokerEI,30
```

### Intermodal Configuration
For PT+AMoD scenarios:
- `broker_type: PTBroker` or `PTBrokerEI` - Broker handling intermodal coordination
- `broker_maas_detour_time_factor` - Detour estimation factor for PTBrokerEI
- `pt_operator_type: PTControlBasic` - PT operator module
- `gtfs_name` - GTFS data folder name
- `pt_simulation_start_date` - GTFS date reference (YYYYMMDD)
- `rq_type: BasicIntermodalRequest` - Request type supporting intermodal
- `evaluation_method: intermodal_evaluation` - Use intermodal KPIs

## Example Scenarios

The `run_examples.py` script runs various scenario types (in `studies/example_study/`):

| Scenario Pattern | Description |
|------------------|-------------|
| `example_pool_*` | Basic ride-pooling with batch optimization |
| `example_charge_*` | Pooling with EV charging infrastructure |
| `example_depot_*` | Fleet size control (time/utilization based) |
| `example_broker_*` | Multi-operator broker scenarios (broker/user/independent decisions) |
| `example_rpp_*` | Ride-parcel-pooling (combined passenger + parcel) |
| `example_im_ptbroker*` | Intermodal PT+AMoD with PTBroker/PTBrokerEI |

Key fleet control modules (`op_module`):
- `PoolingIRSOnly` - Insertion heuristic, immediate decisions
- `PoolingIRSBatchOptimization` - Batch optimization with Gurobi/ORTools
- `RidePoolingBatchAssignmentFleetcontrol` - Full batch assignment

## Dependencies

- Python 3.10 (Conda environment)
- Optional: `gurobipy` (commercial optimizer), `ortools` (open-source)
- C++ extensions: Cython-based road router and RAPTOR PT router

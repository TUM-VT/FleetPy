# Ingolstadt RQ3 — predictive routing for AMoD (paper 2)

Study skeleton for the FleetPy-driven SUMO co-simulation. Cloned from
`studies/fleetpy_sumo_coupling/` (P-template, docs/6 recommended path step 1).

Companion planning docs live in the vaas repo: `docs/6_rq3_amod_fleet_selection.md`
(plan of record) and `docs/0_papers_todos.md` (open items).

## Repo discipline

Everything here is **FleetPy-side** and belongs on the fork branch
`yunfei/predictive_routing`. The vaas-side artifacts (`run_rq3_routing.sh`,
`export_gnn_tt.py`, `aggregate_rq3_kpis.py`, `coverage_probe.py`) stay in the
vaas repo. Do not cross the streams.

## Run

```bash
conda activate fleetpy
export SUMO_HOME=/usr/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:."   # see note below
python src/coupling/SUMO/SUMOFleetPyServer.py \
    studies/ingolstadt_rq3/scenarios/constant_config.csv \
    studies/ingolstadt_rq3/scenarios/001_smoke_R1.csv \
    "<path to the day's .sumocfg>" \
    sumo info
```

`PYTHONPATH` must include `$SUMO_HOME/tools`: `environment.yml` does not install
`traci`, and `SUMOFleetPyServer.py` does not append the SUMO tools directory to
`sys.path` itself (it only documents that `SUMO_HOME` "should be set"). The `.`
entry is needed because the server imports `run_scenarios` from the repo root.

Verified: the entry point imports cleanly under the `fleetpy` env with
SUMO 1.27.0 (the template README says it was tested against 1.26.0).

The `.sumocfg` is not duplicated here: the runs point at the existing
Ingolstadt configs under `sumo_ingolstadt/simulation/Ingolstadt SUMO 365/`
in the vaas repo. The AMoD vType (`sumo/FP_vType.xml`) has to be added to that
day's additional-file list so SUMO accepts the vehicles FleetPy dispatches.

## What is already settled

| Item | Value | Source |
|---|---|---|
| Network | `ingolstadt` (41,938 nodes / 76,301 edges) | converted, docs/6 step 3 |
| Fleet | `amod:100` | docs/6 §2.2 |
| AMoD vType | `amod`, declared in `data/vehicles/amod.csv` + `sumo/FP_vType.xml` | docs/6 step 5 |
| Simulation env | `SUMOcontrolledSim` | docs/6 §4.1 |
| Routing engine | `NetworkBasicWithStoreCpp` | template default |
| TT update interval | `sumo_t_update = 300` s | matches the GNN's 5-min bins |
| edgeData interval | 300 s | same |
| Teleport threshold | 60 s | matches the vaas SUMO sim config |
| Sim clock | SUMO seconds, 25200 (07:00) to 68400 (19:00) | vaas `configs/prediction/rq2_sims/*.yaml` |

## Repositioning setup

Repositioning needs three things Ingolstadt did not have:

1. **A zone system.** Built by the vaas-side `scripts/routing/build_fleetpy_zones.py`.
   Two are checked in: `ingolstadt_grid1000` (73 zones) and `ingolstadt_grid1500`
   (35 zones).
2. **A forecast model.** `op_fc_type=perfect` plus `op_temporal_resolution`.
3. **Gurobi.** All seven repositioning modules import `gurobipy` (now in
   `environment.yml`). The bundled restricted licence needs no registration but
   caps models at **2000 variables**. PavoneFC uses `N^2 - N` for N zones, so
   **35 zones = 1,190 variables fits; 73 zones = 5,256 does not.** Use
   `ingolstadt_grid1500` unless a full licence is available. The bundled licence
   is also marked non-production and expires 2027-11-29.

Measured effect at 200 req/h (same demand, seed and window; see docs/6):
repositioning beat raising `op_max_wait_time` on every KPI at once — service
85.4→95.1%, wait 177→157 s, detour 211→202 s, utilisation 21.6→27.1%. Raising
the wait cap to 600 s reached 99.5% service only by letting mean wait grow to
284 s, so it relaxes the constraint rather than fixing the supply/demand mismatch.

## Scenario-design rules learned from the increment-A smoke run

- **Inset the evaluation interval from the simulation interval.** FleetPy's user
  stats count completed trips only, so requests arriving within roughly one mean
  trip duration of `end_time` never appear and every service KPI is biased. In
  the smoke run (mean wait+travel 981 s) the last logged arrival was 27938
  against an `end_time` of 28800. Simulate wider than you evaluate: at least
  ~20 min of tail, plus a warm-up so the fleet is not still spreading from its
  initial positions.
- **Match fleet size to demand.** At 60 requests/h the 100-vehicle fleet sat at
  10.4% utilisation, which would mute the R1-vs-R2 contrast by construction.
  Aim for a meaningfully loaded fleet (order 300-400 requests/h for 100
  vehicles at ~16 min per trip).
- **Cost anchor:** 842 s wall per simulated hour (~4.3x real time) at PR=10%
  with 100 AMoD vehicles on the full Ingolstadt network.

## Still open

- **`rq_file`** is a placeholder (`TBD_SEE_README.csv`) pending the D-demand
  decision. The smoke scenario cannot run until that is set.
- **Evaluation window.** `001_smoke_R1.csv` uses 07:00-08:00 as a plumbing
  smoke window only. The study window (event/peak ~3-4 h vs full 12 h) is a
  separate decision and the single biggest cost lever.
- **`op_module`.** `PoolingIRSOnly` is inherited from the template and pools
  requests. docs/6 §6 wants one request per vehicle at a time for the headline,
  so this likely needs to change to a non-pooling assignment module.
- **`op_max_wait_time` (300 s) and the fare parameters** are template defaults,
  not calibrated for Ingolstadt.

## Notes found while building this skeleton

1. **The travel-time injection point is a CSV file, not a code hook.**
   `SUMOFleetPyServer.run_coupled_simulation()` calls
   `routing_engine.load_tt_file(sim_time, ext_path=<csv>)` every
   `sumo_t_update` seconds. The file schema is `from_node,to_node,edge_tt`
   (`edge_var` is written but ignored by `NetworkBasic.load_tt_file`). So the
   GNN injection for R2 does not need an upstream patch: write the predicted
   table to that path in the same schema.

2. **Partial network coverage is handled natively.** `load_tt_file` only
   updates the edges present in the file and leaves every other edge at its
   previous travel time. The GNN's 4,296-link main-road coverage therefore
   needs no explicit fallback logic; uncovered edges simply keep the
   SUMO-derived or free-flow value.

3. **The SUMO-edge to FleetPy-node join already exists.**
   `data/networks/ingolstadt/base/edges.csv` carries `source_edge_id` (the SUMO
   edge id) alongside `from_node` / `to_node`, which is exactly the key
   `load_tt_file` indexes on. `setup_network_translation()` builds the dict both
   ways. No new mapping artifact is needed for D-tt-export.

4. **FleetPy already implements an FCD penetration filter.**
   `_filter_by_fcd_mode()` reads the `sumo_fcd_vehicles` scenario parameter in
   the form `op_<operator ids>-pv_<share>`, and derives travel times from the
   operator's own vehicles plus a random share of private vehicles. That is
   structurally the same sensing surface as design option C (AMoD always
   observed, plus a penetration share of background traffic). It covers the
   **ego travel-time** channel only, i.e. floating-car data. Surrounding-vehicle
   **density**, which is what distinguishes AVaS from FCD, still requires the
   vaas-side aggregator.

5. **AMoD vehicles are already identifiable without an id-prefix convention.**
   FleetPy names its SUMO vehicles `fp_<operator>_<vehicle>`, which
   `_filter_by_fcd_mode` matches on. The `amod_<n>` prefix proposed in docs/6
   §2.3 is not needed.

# -------------------------------------------------------------------------------------------------------------------- #
# external imports
# ----------------
import sys
import os
import multiprocessing as mp

fleetpy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(fleetpy_path)

from run_scenarios import run_scenarios

# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    """ this script runs all scenarios of the example study. You can use this script as template for your own run_private*.py script to run the scenarios of your own study. 
    The required parameters are stated in the documentation of the run_scenarios function."""
    mp.freeze_support()

    if len(sys.argv) > 1:
        run_scenarios(*sys.argv)
    else:
        import time

        scs_path = os.path.join(os.path.dirname(__file__), "scenarios")
        # Base Examples IRS only
        # ----------------------
        # a) Pooling in ImmediateOfferEnvironment
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_ir.csv")
        sc = os.path.join(scs_path, "example_ir_only.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)

        # Base Examples with Optimization (requires gurobi license!)
        # ----------------------------------------------------------
        # b) Pooling in BatchOffer environment
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_pool.csv")
        sc = os.path.join(scs_path, "example_pool.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)

        # c) Pooling in ImmediateOfferEnvironment
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_ir.csv")
        sc = os.path.join(scs_path, "example_ir_batch.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)

        # d) Pooling with RV heuristics in ImmediateOfferEnvironment (with doubled demand)
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_ir.csv")
        t0 = time.perf_counter()
        # no heuristic scenario
        sc = os.path.join(scs_path, "example_pool_noheuristics.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        # with heuristic scenarios
        t1 = time.perf_counter()
        sc = os.path.join(scs_path, "example_pool_heuristics.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        t2 = time.perf_counter()
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=2)
        t3 = time.perf_counter()
        print(f"Computation time without heuristics: {round(t1-t0, 1)} | with heuristics 1 CPU: {round(t2-t1,1)}"
              f"| with heuristics 2 CPU: {round(t3-t2,1)}")

        # g) Pooling with RV heuristic and Repositioning in ImmediateOfferEnvironment (with doubled demand and
        #       bad initial vehicle distribution)
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_ir_repo.csv")
        sc = os.path.join(scs_path, "example_ir_heuristics_repositioning.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        
        # h) Pooling with public charging infrastructure (low range vehicles)
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_charge.csv")
        sc = os.path.join(scs_path, "example_charge.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        
        # i) Pooling and active vehicle fleet size is controlled externally (time and utilization based)
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_depot.csv")
        sc = os.path.join(scs_path, "example_depot.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        
        # j) Pooling with public charging and fleet size control (low range vehicles)
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_depot_charge.csv")
        sc = os.path.join(scs_path, "example_depot_charge.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        
        # k) Pooling with multiprocessing
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_depot_charge.csv")
        # no heuristic scenario single core
        t0 = time.perf_counter()
        sc = os.path.join(scs_path, "example_depot_charge.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        print("Computation without multiprocessing took {}s".format(time.perf_counter() - t0))
        # no heuristic scenario multiple cores
        cores = 2
        t0 = time.perf_counter()
        sc = os.path.join(scs_path, "example_depot_charge.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=cores, n_parallel_sim=1)
        print("Computation with multiprocessing took {}s".format(time.perf_counter() - t0))
        print(" -> multiprocessing only usefull for large vehicle fleets")
        
        # l) Pooling - multiple operators and broker
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_broker.csv")
        sc = os.path.join(scs_path, "example_broker.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)
        
        # m) Ride-Parcel-Pooling example
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_rpp.csv")
        sc = os.path.join(scs_path, "example_rpp.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)

        # i) Semi-on-Demand Public Transit example
        # Run PTScheduleGen.py to generate required PT files first
        log_level = "info"
        cc = os.path.join(scs_path, "constant_config_sod.csv")
        sc = os.path.join(scs_path, "example_sod.csv")
        run_scenarios(cc, sc, log_level=log_level, n_cpu_per_sim=1, n_parallel_sim=1)

import os
import multiprocessing as mp
import sys
from time import time
import traceback

import pandas as pd

fleetpy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(fleetpy_path)

from run_scenarios import run_scenarios

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
SC_PATH = os.path.join(STUDY_DIR, "scenarios")
RESULTS_DIR = os.path.join(STUDY_DIR, "results")

# Skip scenarios whose results already exist. Override from the command line with
# --force / --rerun-all (or env WP1_RERUN_ALL=1) to re-run everything.
SKIP_EXISTING = False
# File written when a run finishes; its presence marks a scenario as complete.
COMPLETION_MARKER = "standard_eval.csv"


def has_results(scenario_name):
    return os.path.isfile(os.path.join(RESULTS_DIR, scenario_name, COMPLETION_MARKER))


def build_scenario_file(scenario_file, skip_existing):
    """Return a scenario cfg path with completed scenarios filtered out.

    If nothing is skipped, the original file is returned unchanged. Otherwise a
    filtered copy is written next to it and its path returned.
    """
    df = pd.read_csv(scenario_file)
    if not skip_existing:
        return scenario_file, len(df), 0

    done_mask = df["scenario_name"].apply(has_results)
    remaining = df[~done_mask]
    skipped = int(done_mask.sum())
    if skipped == 0:
        return scenario_file, len(df), 0

    # Named after the input file (not a fixed "scenario_cfg_todo.csv") so multiple cluster jobs
    # running different scenario_cfg files in parallel against the same SC_PATH don't clobber
    # each other's filtered "todo" file.
    stem = os.path.splitext(os.path.basename(scenario_file))[0]
    filtered_path = os.path.join(SC_PATH, f"{stem}_todo.csv")
    remaining.to_csv(filtered_path, index=False)
    return filtered_path, len(remaining), skipped


def _resolve_scenario_file(arg):
    """Resolve a scenario_cfg path argument against SC_PATH, falling back to cwd (or an absolute
    path unchanged) -- same convention as generate_scenarios.py's ranges-file resolution."""
    if os.path.isabs(arg):
        return arg
    in_dir = os.path.join(SC_PATH, arg)
    return in_dir if os.path.exists(in_dir) else arg


if __name__ == "__main__":
    mp.freeze_support()

    # override: run everything regardless of existing results
    rerun_all = ("--force" in sys.argv or "--rerun-all" in sys.argv
                 or os.environ.get("WP1_RERUN_ALL") == "1")
    skip_existing = SKIP_EXISTING and not rerun_all

    # optional first non-flag argument: which scenario_cfg CSV to run (e.g. one of a cluster
    # job's split scenario_cfg_final_N.csv files) -- defaults to scenario_cfg.csv if omitted.
    cfg_args = [a for a in sys.argv[1:] if not a.startswith("--")]

    try:
        start_time = time()
        cc = os.path.join(SC_PATH, "const_cfg.yaml")
        sc = _resolve_scenario_file(cfg_args[0]) if cfg_args else os.path.join(SC_PATH, "scenario_cfg.csv")

        sc_to_run, n_run, n_skipped = build_scenario_file(sc, skip_existing)
        if n_skipped:
            print(f"Skipping {n_skipped} scenario(s) with existing results; running {n_run}. "
                  f"Pass --force to re-run all.")
        if n_run == 0:
            print("Nothing to run — all scenarios already have results.")
        else:
            run_scenarios(cc, sc_to_run, log_level="info", n_cpu_per_sim=1, n_parallel_sim=10)

        end_time = time()
        print(f"Computation time: {end_time - start_time} seconds")
    except:
        traceback.print_exc()

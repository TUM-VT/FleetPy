#!/usr/bin/env python3
"""Run the paper-based PTBroker revision scenarios with dynamic scheduling."""

import os

# Set these before importing NumPy, pandas, or FleetPy in either parent or worker.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
import csv
import datetime
import fcntl
import logging
import multiprocessing
import queue
import random
import signal
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from uuid import uuid4

STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[1]
CONSTANT_CONFIG = STUDY_DIR / "scenarios" / "constant_config.csv"
SCENARIOS_DIR = STUDY_DIR / "scenarios"
RESULTS_DIR = STUDY_DIR / "results"
GROUPS = tuple(
    f"{strategy}_c{capacity}"
    for strategy in ("cc", "ei100_off", "payg", "ei60_off", "ei60_on")
    for capacity in (4, 8, 20)
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--group", choices=GROUPS, help="run only this group")
    selection.add_argument("--list-groups", action="store_true", help="list available groups")
    selection.add_argument("--hardware-check", action="store_true",
                           help="show a hardware sample without starting simulations")
    parser.add_argument("--workers", type=positive_int, default=24,
                        help="parallel scenarios (default: 24, for the target 14900K)")
    parser.add_argument("--progress-interval", type=positive_int, default=30,
                        help="progress update interval in seconds (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="validate without starting simulations")
    return parser.parse_args(argv)


def prepare_scenarios(groups):
    """Validate the whole selection before starting any simulation."""
    from src.misc.config import ConstantConfig, ScenarioConfig

    constants = ConstantConfig(str(CONSTANT_CONFIG))
    study_name = STUDY_DIR.name
    if constants.get("study_name") not in (None, study_name):
        raise ValueError("The constant config study_name does not match the study directory")
    constants.update(study_name=study_name, n_cpu_per_sim=1, evaluate=1,
                     log_level="info", keep_old=False)
    scenarios = []
    names = set()
    demand_files = set()
    for group in groups:
        group_scenarios = ScenarioConfig(str(SCENARIOS_DIR / f"{group}.csv"))
        if not group_scenarios:
            raise ValueError(f"No scenarios are configured for {group}")
        for scenario in group_scenarios:
            name = scenario.get("scenario_name")
            if (not isinstance(name, str) or name in ("", ".", "..")
                    or "/" in name or "\\" in name):
                raise ValueError(f"Invalid scenario name: {name!r}")
            if name in names:
                raise ValueError(f"Duplicate scenario name: {name}")
            if os.path.lexists(RESULTS_DIR / name):
                raise FileExistsError(f"Scenario output already exists; no simulations started: {name}")
            demand_name = scenario.get("rq_file")
            if not isinstance(demand_name, str) or not demand_name:
                raise ValueError(f"Missing demand file for {name}")
            names.add(name)
            demand_files.add(demand_name)
            scenarios.append(constants + scenario)

    demand_dir = ROOT / "data" / "demand" / constants["demand_name"] / "matched" / constants["network_name"]
    expected_modes = {"intra": "MONOMODAL", "inter": "FIRSTLASTMILE"}
    for name in sorted(demand_files):
        path = demand_dir / name
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            if not {"rq_type", "query_modes"}.issubset(reader.fieldnames or ()):
                raise ValueError(f"Missing request mode fields in {path}")
            for row_number, row in enumerate(reader, start=2):
                if row["rq_type"] not in expected_modes or expected_modes[row["rq_type"]] != row["query_modes"]:
                    raise ValueError(f"Non-paper request mode in {path} at line {row_number}")
    return scenarios


_worker_events = None


def initialize_worker(events):
    global _worker_events
    _worker_events = events
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_worker(scenario, log_dir):
    """Capture each simulation's Python and native console output separately."""
    name = scenario["scenario_name"]
    log_path = Path(log_dir) / f"{name}.log"
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    try:
        with log_path.open("w", buffering=1) as output:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            _worker_events.put((name, os.getpid(), time.monotonic()))
            try:
                # FleetPy clears existing output directories; check again at dispatch time.
                if os.path.lexists(RESULTS_DIR / name):
                    raise FileExistsError(f"Scenario output already exists: {name}")
                from run_scenarios import run_single_simulation
                run_single_simulation(scenario)
                return name, True, ""
            except BaseException as exc:
                traceback.print_exc()
                return name, False, f"{type(exc).__name__}: {exc}"
            finally:
                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)
                    handler.close()
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def format_duration(seconds):
    return str(datetime.timedelta(seconds=max(0, int(seconds))))


class HardwareMonitor:
    """Read Linux host utilization and Intel CPU temperature without extra dependencies."""

    def __init__(self, proc_root=Path("/proc"), hwmon_root=Path("/sys/class/hwmon")):
        self.proc_root = Path(proc_root)
        self.hwmon_root = Path(hwmon_root)
        self.previous_cpu = None

    def _cpu_percent(self):
        try:
            fields = (self.proc_root / "stat").read_text().splitlines()[0].split()
            if fields[0] != "cpu" or len(fields) < 5:
                raise ValueError("Invalid aggregate CPU counters")
            counters = [int(value) for value in fields[1:9]]
            if any(value < 0 for value in counters):
                raise ValueError("Invalid aggregate CPU counters")
            total = sum(counters)
            idle = counters[3] + (counters[4] if len(counters) > 4 else 0)
        except (OSError, ValueError, IndexError):
            self.previous_cpu = None
            return None
        previous = self.previous_cpu
        self.previous_cpu = (total, idle)
        if previous is None:
            return None
        total_delta, idle_delta = total - previous[0], idle - previous[1]
        if total_delta <= 0 or idle_delta < 0 or idle_delta > total_delta:
            return None
        return 100 * (total_delta - idle_delta) / total_delta

    def _memory(self):
        try:
            values = {}
            for line in (self.proc_root / "meminfo").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    fields = value.split()
                    if len(fields) != 2 or fields[1] != "kB":
                        raise ValueError("Invalid memory counters")
                    values[key] = int(fields[0])
            total, available = values["MemTotal"], values["MemAvailable"]
            if total <= 0 or not 0 <= available <= total:
                raise ValueError("Invalid memory counters")
            return (total - available) / 1048576, total / 1048576
        except (OSError, ValueError, IndexError, KeyError):
            return None, None

    def _temperature(self):
        packages = []
        cores = []
        try:
            devices = sorted(self.hwmon_root.glob("hwmon*"))
        except OSError:
            return None, None
        for device in devices:
            try:
                if (device / "name").read_text().strip().lower() != "coretemp":
                    continue
                inputs = sorted(device.glob("temp*_input"))
            except OSError:
                continue
            for path in inputs:
                try:
                    label = path.with_name(path.name.replace("_input", "_label")).read_text().strip()
                    temperature = int(path.read_text().strip()) / 1000
                    if not -40 <= temperature <= 150:
                        continue
                except (OSError, ValueError):
                    continue
                if label.lower().startswith("package id "):
                    packages.append((temperature, label))
                elif label.lower().startswith("core "):
                    cores.append((temperature, label))
        if packages:
            temperature, label = max(packages)
            return temperature, f"coretemp {label}"
        if cores:
            temperature, label = max(cores)
            return temperature, f"coretemp max {label}"
        return None, None

    def sample(self):
        cpu = self._cpu_percent()
        memory_used, memory_total = self._memory()
        temperature, label = self._temperature()
        return {"cpu_percent": cpu, "memory_used_gib": memory_used,
                "memory_total_gib": memory_total, "temperature_c": temperature,
                "temperature_label": label}


def format_hardware(sample):
    cpu = f"{sample['cpu_percent']:.1f}%" if sample["cpu_percent"] is not None else "N/A"
    if sample["memory_used_gib"] is None or sample["memory_total_gib"] is None:
        memory = "N/A"
    else:
        used, total = sample["memory_used_gib"], sample["memory_total_gib"]
        memory = f"{used:.1f}/{total:.1f} GiB ({100 * used / total:.1f}%)"
    if sample["temperature_c"] is None:
        temperature = "N/A"
    else:
        temperature = f"{sample['temperature_c']:.1f} C ({sample['temperature_label']})"
    return f"Host CPU: {cpu} | RAM used/total: {memory} | CPU temperature: {temperature}"


def print_progress(total, succeeded, failed, pending, active, started, monitor, interrupted=False):
    now = time.monotonic()
    elapsed = now - started
    done = succeeded + failed
    eta = elapsed * (total - done) / done if done and not interrupted else None
    remaining = format_duration(eta) if eta is not None else "unknown"
    estimated_total = format_duration(elapsed + eta) if eta is not None else "unknown"
    suffix = " | Stopping after active scenarios" if interrupted else ""
    print(f"Completed: {done}/{total} ({100 * done / total:.1f}%) | Failed: {failed} | Running: {len(active)} | "
          f"Elapsed: {format_duration(elapsed)} | ETA (rough): {remaining} | "
          f"Estimated total (rough): {estimated_total}{suffix}", flush=True)
    print(format_hardware(monitor.sample()), flush=True)
    for name, (pid, begin) in sorted(active.items()):
        if name in pending:
            print(f"  PID {pid} | {format_duration(now - begin)} | {name}", flush=True)


def run_scenarios(scenarios, workers, progress_interval, log_dir):
    """Keep only one assigned scenario per worker and schedule the next on completion."""
    scenarios = list(scenarios)
    random.Random(20261110).shuffle(scenarios)
    total = len(scenarios)
    iterator = iter(scenarios)
    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                   initializer=initialize_worker, initargs=(events,))
    pending = {}
    active = {}
    succeeded = failed = 0
    interrupted = False
    interrupt_reported = False
    monitor = HardwareMonitor()
    start = time.monotonic()
    next_report = start

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    def submit_one():
        if interrupted:
            return False
        try:
            scenario = next(iterator)
        except StopIteration:
            return False
        pending[executor.submit(run_worker, scenario, str(log_dir))] = scenario["scenario_name"]
        return True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
        for _ in range(min(workers, total)):
            submit_one()
        while pending:
            done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            if interrupted and not interrupt_reported:
                print("Interrupt received: no more scenarios will be scheduled; waiting for active scenarios.", flush=True)
                interrupt_reported = True
            while True:
                try:
                    name, pid, begin = events.get_nowait()
                except queue.Empty:
                    break
                if name in pending.values():
                    active[name] = (pid, begin)
            for future in done:
                name = pending.pop(future)
                active.pop(name, None)
                result_name, success, error = future.result()
                if result_name != name:
                    raise RuntimeError(f"Unexpected worker result: {result_name}, expected {name}")
                if success:
                    succeeded += 1
                else:
                    failed += 1
                    print(f"FAILED {name}: {error} (see {log_dir / (name + '.log')})", flush=True)
                submit_one()
            if time.monotonic() >= next_report or not pending:
                print_progress(total, succeeded, failed, pending.values(), active, start, monitor, interrupted)
                next_report = time.monotonic() + progress_interval
        return 130 if interrupted else (1 if failed else 0)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        events.close()
        events.join_thread()
        signal.signal(signal.SIGINT, previous_sigint)


def main(argv=None):
    args = parse_args(argv)
    if args.list_groups:
        print("\n".join(GROUPS))
        return 0
    if args.hardware_check:
        monitor = HardwareMonitor()
        monitor.sample()
        time.sleep(1)
        print(format_hardware(monitor.sample()))
        return 0
    groups = (args.group,) if args.group else GROUPS
    if args.dry_run:
        scenarios = prepare_scenarios(groups)
        print(f"Validated {len(groups)} group(s): {len(scenarios)} scenarios; workers={args.workers}.")
        print("Dry run: no simulations started.")
        return 0

    # Serialize runs for this study so concurrent invocations cannot clear each other's output.
    with (STUDY_DIR / ".run_revision.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another revision run is already active") from None
        scenarios = prepare_scenarios(groups)
        log_dir = STUDY_DIR / "run_logs" / (datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
        log_dir.mkdir(parents=True)
        print(f"Validated {len(groups)} group(s): {len(scenarios)} scenarios; workers={args.workers}.")
        print(f"Scenario console logs: {log_dir}", flush=True)
        return run_scenarios(scenarios, args.workers, args.progress_interval, log_dir)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())

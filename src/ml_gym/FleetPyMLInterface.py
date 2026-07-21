"""Run one or more FleetPy scenarios with shared ML Hook abstractions.

The interface accepts either one fully merged scenario parameter mapping or a
sequence of mappings. ``nr_parallel`` limits how many distinct scenarios run
concurrently; it never creates duplicate runs of one scenario.
"""

import sys
import os
import time
from collections import Counter, deque
from collections.abc import Mapping

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add fleetpy path

from src.misc.init_modules import load_simulation_environment
from src.ml_gym.hooks_manager import HookManager, Events
from src.ml_gym.Observers import AbstractObserver
from src.ml_gym.Actors import AbstractActor
import multiprocessing as mp

from typing import List, Optional


def run_single_simulation(scenario_parameters, hooks_manager, process_id):
    """Child-process entry point for one scenario and its HookManager."""
    SF = load_simulation_environment(scenario_parameters, hooks_manager, process_id)
    SF.run(process_id)


class FleetPyMLInterface:
    def __init__(self, scenario_parameters, nr_parallel=1):
        """Create an ML runner for every supplied FleetPy scenario.

        ``nr_parallel`` is the maximum number of distinct scenarios running at
        once.  It never creates duplicate copies of one scenario merely to fill
        worker slots.  A single parameter mapping and a sequence of parameter
        mappings are normalized to the same internal scenario list.
        """
        if not isinstance(nr_parallel, int) or nr_parallel < 1:
            raise ValueError("FleetPyMLInterface: nr_parallel must be a positive integer")

        # Normalize a single mapping and a sequence of mappings immediately so
        # the rest of the class only needs one multi-scenario code path.
        if isinstance(scenario_parameters, Mapping):
            self.scenario_parameters_list = [dict(scenario_parameters)]
        else:
            self.scenario_parameters_list = [
                dict(parameters) for parameters in scenario_parameters
            ]
        if not self.scenario_parameters_list:
            raise ValueError("FleetPyMLInterface: at least one scenario must be provided")

        # FleetPy derives its output directory from scenario_name. Duplicate
        # names could therefore make concurrent workers overwrite each other.
        scenario_names = [
            parameters.get("scenario_name")
            for parameters in self.scenario_parameters_list
        ]
        if any(name is None for name in scenario_names):
            raise ValueError("FleetPyMLInterface: every scenario requires a scenario_name")

        duplicate_names = sorted(
            name for name, count in Counter(scenario_names).items() if count > 1
        )
        if duplicate_names:
            raise ValueError(f"FleetPyMLInterface: scenario_name must be unique; duplicates: {duplicate_names}")

        self.nr_parallel = nr_parallel
        # Avoid process and queue overhead unless there are at least two
        # scenarios that can actually run concurrently.
        self._use_multiprocessing = (nr_parallel > 1 and len(self.scenario_parameters_list) > 1)

        # Each scenario owns a HookManager. In multiprocessing mode it also
        # owns a private queue pair identified by its stable scenario index.
        self.hook_managers = []
        for process_id in range(len(self.scenario_parameters_list)):
            if self._use_multiprocessing:
                queues = {process_id: (mp.Queue(), mp.Queue())}
            else:
                queues = None
            self.hook_managers.append(HookManager(queues))

        self.fleetpy_processes = []

    def _selected_hook_managers(self, scenario_index: Optional[int]):
        """Select all HookManagers or the manager for one scenario."""
        if scenario_index is None:
            return self.hook_managers
        if not 0 <= scenario_index < len(self.hook_managers):
            raise IndexError(f"FleetPyMLInterface: scenario_index {scenario_index} is out of range")
        return [self.hook_managers[scenario_index]]

    def register_observer(self, event: Events, observer: AbstractObserver, scenario_index: Optional[int] = None):
        """Register an Observer for every scenario or one selected scenario."""
        for hook_manager in self._selected_hook_managers(scenario_index):
            hook_manager.add_observer(event, observer)

    def register_actor(self, event: Events, actor: AbstractActor,
                       scenario_index: Optional[int] = None):
        """Register an Actor for every scenario or one selected scenario."""
        for hook_manager in self._selected_hook_managers(scenario_index):
            hook_manager.add_actor(event, actor)

    def couple_actors_to_observers(self, event: Events, actors: List[AbstractActor],
                                   observers: List[AbstractObserver],
                                   scenario_index: Optional[int] = None):
        """Restrict Actor/Observer coupling globally or for one scenario."""
        for hook_manager in self._selected_hook_managers(scenario_index):
            hook_manager.couple_actors_to_observers(event, actors, observers)

    def reply_to_slave_processes(self, scenario_indices=None):
        """Service queued Actor requests from the selected active workers."""
        if scenario_indices is None:
            scenario_indices = range(len(self.hook_managers))
        for scenario_index in scenario_indices:
            self.hook_managers[scenario_index].reply_to_slave_processes()

    def run(self):
        """Run every scenario, using multiprocessing only when it adds concurrency."""
        if not self._use_multiprocessing:
            # Sequential execution avoids multiprocessing startup overhead but
            # still runs every supplied scenario row.
            for scenario_parameters, hook_manager in zip(
                    self.scenario_parameters_list, self.hook_managers):
                simulation = load_simulation_environment(scenario_parameters, hook_manager)
                simulation.run()
            return

        # Use a dynamic pending queue instead of fixed process batches. When a
        # short scenario finishes, the next pending scenario immediately takes
        # its slot without waiting for other, longer scenarios to finish.
        pending_scenarios = deque(enumerate(self.scenario_parameters_list))
        active_processes = {}
        failed_scenarios = []
        max_parallel = min(self.nr_parallel, len(self.scenario_parameters_list))

        while active_processes or pending_scenarios:
            # Fill every currently available process slot.
            while len(active_processes) < max_parallel and pending_scenarios:
                scenario_index, scenario_parameters = pending_scenarios.popleft()
                process = mp.Process(
                    target=run_single_simulation,
                    args=(
                        scenario_parameters,
                        self.hook_managers[scenario_index],
                        scenario_index,
                    ),
                )
                self.fleetpy_processes.append(process)
                active_processes[scenario_index] = process
                process.start()

            # Queue-based Actors block their simulation worker while waiting
            # for the parent process to compute and return an action.
            self.reply_to_slave_processes(active_processes)

            # Reap completed workers. Vacated slots are refilled at the start
            # of the next scheduler iteration, so there is no batch barrier.
            for scenario_index, process in list(active_processes.items()):
                if not process.is_alive():
                    process.join()
                    if process.exitcode != 0:
                        failed_scenarios.append(
                            self.scenario_parameters_list[scenario_index]["scenario_name"]
                        )
                    del active_processes[scenario_index]

            if active_processes:
                # Prevent a busy loop while keeping Actor-message latency low.
                time.sleep(0.001)

        if failed_scenarios:
            raise RuntimeError(
                "FleetPyMLInterface: simulations failed for scenarios "
                f"{failed_scenarios}"
            )

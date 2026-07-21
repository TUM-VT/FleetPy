"""Collect decision-aligned Fleet State data without Gym."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add FleetPy path

from src.FleetSimulationBase import build_operator_attribute_dicts
from src.misc.globals import *
from src.ml_gym.Actors.writers import JSONWriter
from src.ml_gym.FleetPyMLInterface import FleetPyMLInterface
from src.ml_gym.Observers.fleet_state_observers import FleetStateObserver
from src.ml_gym.hooks_manager import Events
import src.misc.config as config


FLEET_STATE_EVENTS = {
    "ImmediateDecisionsSimulation": Events.OBSERVE_FLEET_STATE_BEFORE_IMMEDIATE_REQUEST_SUBMISSION,
    "BatchOfferSimulation": Events.OBSERVE_FLEET_STATE_BEFORE_BATCH_TIME_TRIGGER,
}


def get_fleet_state_event(simulation_environment):
    """Return the Fleet State event matching the simulation control mode."""
    try:
        return FLEET_STATE_EVENTS[simulation_environment]
    except KeyError as error:
        supported_environments = ", ".join(sorted(FLEET_STATE_EVENTS))
        raise ValueError(
            f"Fleet State collection supports {supported_environments}, "
            f"but received '{simulation_environment}'."
        ) from error


def get_field_list(value):
    """Convert one configured field or a field sequence to a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def get_custom_fields(scenario_parameters):
    """Read a custom Fleet State schema from scenario parameters."""
    return {
        "veh": get_field_list(scenario_parameters.get(G_ML_FS_CUSTOM_VEH)),
        "leg": get_field_list(scenario_parameters.get(G_ML_FS_CUSTOM_LEG)),
        "stop": get_field_list(scenario_parameters.get(G_ML_FS_CUSTOM_STOP)),
    }


if __name__ == "__main__":

    MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if len(sys.argv) >= 3:
        const_config = sys.argv[1]
        sc_config = sys.argv[2]
    else:
        # default: ml_test study
        scs_path = os.path.join(MAIN_DIR, "studies", "ml_test", "scenarios")
        const_config = os.path.join(scs_path, "constant_config_ir.csv")
        sc_config = os.path.join(scs_path, "example_ml_fleet_state.csv")

    # Optional third argument: maximum number of scenarios running at once.
    nr_parallel = int(sys.argv[3]) if len(sys.argv) >= 4 else 1

    constant_cfg = config.ConstantConfig(const_config)
    scenario_cfgs = config.ScenarioConfig(sc_config)

    study_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(const_config))))
    constant_cfg[G_STUDY_NAME] = study_name
    constant_cfg["n_cpu_per_sim"] = 1
    constant_cfg["evaluate"] = 1
    constant_cfg["log_level"] = "info"

    # Combine the constant configuration with every row of the scenario file.
    fleetpy_configs = []
    for scenario_cfg in scenario_cfgs:
        fleetpy_config = constant_cfg + scenario_cfg
        fleetpy_config[G_SKIP_OUTPUT] = 0
        fleetpy_configs.append(fleetpy_config)

    # The interface runs all scenario rows and uses nr_parallel as a concurrency limit.
    fp_ml_interface = FleetPyMLInterface(fleetpy_configs, nr_parallel=nr_parallel)

    output_files = []
    for scenario_index, fleetpy_config in enumerate(fleetpy_configs):
        event = get_fleet_state_event(fleetpy_config[G_SIM_ENV])
        detail_level = fleetpy_config.get(G_ML_FS_DETAIL) or "medium"
        custom_fields = get_custom_fields(fleetpy_config) if detail_level == "custom" else None

        operator_parameters = build_operator_attribute_dicts(
            fleetpy_config,
            fleetpy_config[G_NR_OPERATORS],
            prefix="op_",
        )
        output_dir = get_directory_dict(fleetpy_config, operator_parameters)[G_DIR_OUTPUT]
        output_file = os.path.join(output_dir, "fleet_states.jsonl")
        output_files.append(output_file)

        # Register each scenario separately because event, schema, and output differ by row.
        fp_ml_interface.register_observer(
            event,
            FleetStateObserver(detail_level=detail_level, custom_fields=custom_fields),
            scenario_index=scenario_index,
        )
        fp_ml_interface.register_actor(
            event,
            JSONWriter(output_file),
            scenario_index=scenario_index,
        )

        print(
            f"Configure scenario '{fleetpy_config[G_SCENARIO_NAME]}' with "
            f"Fleet State detail level '{detail_level}'."
        )

    # Let collection failures propagate so callers and CI receive a non-zero
    # exit status. Only advertise output files after every scenario succeeds.
    fp_ml_interface.run()

    print("Fleet State JSONL files:")
    for output_file in output_files:
        print(f"  {output_file}")

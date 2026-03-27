import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))) # add fleetpy path

from src.ml_gym.FleetPyMLInterface import FleetPyMLInterface
from src.misc.globals import *

from src.ml_gym.hooks_manager import Events
from src.ml_gym.actors.repositioning import ZoneBasedRepositioningActor
import src.misc.config as config
import random

import traceback
import multiprocessing as mp

class RandomReposition(ZoneBasedRepositioningActor):

    def compute_action(self, observation, process_id) -> list[tuple[int, int]]:
        # compute random repositioning action based on observation
        print("\nRandom Repositioning: compute random repositioning action")
        print(f"Observation: {observation}")
        sim_time = observation["sim_time"]
        zone_to_fc_rq_origins = observation["zone_to_fc_rq_origins"]
        zone_to_fc_rq_destinations = observation["zone_to_fc_rq_destinations"]
        zone_to_idle_vehilces = observation["zone_to_idle_vehilces"]
        zone_to_overall_available_vehilces = observation["zone_to_overall_available_vehilces"]
        zone_to_current_repositioning_vehicles = observation["zone_to_current_repositioning_vehicles"]

        list_repo_targets = []
        for zone_id, value in zone_to_fc_rq_origins.items():
            if zone_id >= 0:
                for _ in range(int(value)):
                    list_repo_targets.append(zone_id)
        list_repo_origins = []
        for zone_id, value in zone_to_idle_vehilces.items():
            if zone_id > 0:
                for _ in range(int(value)):
                    list_repo_origins.append(zone_id)

        list_repo_actions = []
        while len(list_repo_targets) > 0 and len(list_repo_origins) > 0:
            origin = random.choice(list_repo_origins)
            target = random.choice(list_repo_targets)
            list_repo_actions.append((origin, target))
            list_repo_targets.remove(target)
            list_repo_origins.remove(origin)
        return list_repo_actions

if __name__ == "__main__":

    MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if len(sys.argv) >= 3:
        const_config = sys.argv[1]
        sc_config = sys.argv[2]
    else:
        # default: ml_test study
        scs_path = os.path.join(MAIN_DIR, "studies", "MLtest", "scenarios")
        const_config = os.path.join(scs_path, "constant_config.csv")
        sc_config = os.path.join(scs_path, "sc_config_repo.csv")

    constant_cfg = config.ConstantConfig(const_config)
    scenario_cfgs = config.ScenarioConfig(sc_config)

    study_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(const_config))))
    constant_cfg[G_STUDY_NAME] = study_name
    constant_cfg["n_cpu_per_sim"] = 1
    constant_cfg["evaluate"] = 1
    constant_cfg["log_level"] = "info"

    fleetpy_config = constant_cfg + scenario_cfgs[0]
    
    # init FleetPyMLInterface
    fp_ml_interface = FleetPyMLInterface(fleetpy_config, nr_parallel=1)
    
    # define event for interaction between FleetPy and ML environment

    event = Events.OBSERVE_BEFORE_REPOSITIONING
    from src.ml_gym.observers.repositioning_observers import SimTimeObserver, DemandForecastObserver, ZoneBasedVehicleStatesObserver
    sim_observer = SimTimeObserver()
    fp_ml_interface.register_observer(event, sim_observer)
    fp_ml_interface.register_observer(event, DemandForecastObserver())
    fp_ml_interface.register_observer(event, ZoneBasedVehicleStatesObserver())

    # register actor
    fp_ml_interface.register_actor(event, RandomReposition())

    event = Events.OBSERVE_FLEET_STATE_AFTER_RECEIVING_STATUS_UPDATE
    from src.ml_gym.observers.fleet_control_observers import FleetStateObserver
    fp_ml_interface.register_observer(event, FleetStateObserver())

    from src.ml_gym.writers import JSONWriter
    output_file = os.path.join(scs_path, "fleet_stats_output.json")
    fp_ml_interface.register_actor(event, JSONWriter(output_file))

    output_file = os.path.join(scs_path, "reposition_observer_output.json")
    repo_writer = JSONWriter(output_file)
    fp_ml_interface.register_actor(Events.OBSERVE_BEFORE_REPOSITIONING, repo_writer)

    # A coupling between specific actors and observers is also possible using the following method
    #fp_ml_interface.couple_actors_to_observers(Events.OBSERVE_BEFORE_REPOSITIONING, [repo_writer], [sim_observer])
    
    # run interface
    try:
        fp_ml_interface.run()
    except Exception as e:
        print("Error in FleetPyMLInterface:")
        traceback.print_exc()
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))) # add fleetpy path

from src.ml_gym.FleetPyGymInterface import FleetPyGym
from src.misc.globals import *
from src.ml_gym.hooks_manager import Events
from src.ml_gym.Observers.repositioning_observers import SimTimeObserver, DemandForecastObserver, ZoneBasedVehicleStatesObserver

from src.ml_gym.Actors.repositioning import ZoneBasedRepositioningActor
from src.misc.config import ConstantConfig, ScenarioConfig
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOConfig
import random


class FleetPyRepoRL(FleetPyGym):

    def __init__(self, config):
        constant_cfg = ConstantConfig(config["constant_cfg_path"])
        scenario_cfgs = ScenarioConfig(config["var_cfg_path"])

        study_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(config["constant_cfg_path"]))))
        constant_cfg[G_STUDY_NAME] = study_name
        constant_cfg["n_cpu_per_sim"] = 1
        constant_cfg["evaluate"] = 1
        constant_cfg["log_level"] = "info"

        # **** Note: it is possible to select different configuration for individual workers using the worker index. ****
        # scenario_inx = (config.worker_index-1) % len(scenario_cfgs)
        # print(f"Worker {config.worker_index} using scenario config {scenario_inx}")

        scenario_inx = 0
        process_id = 0
        fleetpy_config = constant_cfg + scenario_cfgs[scenario_inx]
        # Change the scenario name according to worker index
        if config.worker_index > 0:
            process_id = config.worker_index - 1
            fleetpy_config[G_SCENARIO_NAME] = fleetpy_config[G_SCENARIO_NAME] + f"_worker_{process_id}"

        super().__init__(fleetpy_config, process_id)
        self.nr_zones = config["nr_zones"]
        fleet_size = sum(fleetpy_config["op_fleet_composition"].values())

        self.action_space = spaces.MultiDiscrete(self.nr_zones * self.nr_zones * [fleet_size])
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(3 * self.nr_zones,), dtype=np.float32)

        # Register FleetPy observers
        event = Events.OBSERVE_BEFORE_REPOSITIONING
        sim_observer = SimTimeObserver()
        self.register_observer(event, sim_observer)
        self.register_observer(event, DemandForecastObserver())
        self.register_observer(event, ZoneBasedVehicleStatesObserver())

        # register FleetPy actors
        self.register_actor(event, ZoneBasedRepositioningActor())

    def translate_observation(self, observation, actor_type, event: Events):
        sim_time = observation["sim_time"]
        zone_to_fc_rq_origins = observation["zone_to_fc_rq_origins"]
        zone_to_fc_rq_destinations = observation["zone_to_fc_rq_destinations"]
        zone_to_idle_vehilces = observation["zone_to_idle_vehilces"]
        zone_to_overall_available_vehilces = observation["zone_to_overall_available_vehilces"]
        zone_to_current_repositioning_vehicles = observation["zone_to_current_repositioning_vehicles"]

        all_zone_ids = list(zone_to_idle_vehilces.keys())
        all_zone_ids.remove(-1)

        idle = np.array([zone_to_idle_vehilces.get(zone_id, 0) for zone_id in all_zone_ids], dtype=np.float32)
        req_origins = np.array([zone_to_fc_rq_origins.get(zone_id, 0)  for zone_id in all_zone_ids], dtype=np.float32)
        req_destinations = np.array([zone_to_fc_rq_destinations.get(zone_id, 0)  for zone_id in all_zone_ids], dtype=np.float32)

        processed_observation = np.concatenate([idle, req_origins, req_destinations], axis=0).astype(np.float32)
        return processed_observation

    def translate_action(self, observation, action, actor_type, event: Events):
        zone_to_fc_rq_origins = observation["zone_to_fc_rq_origins"]
        zone_to_idle_vehilces = observation["zone_to_idle_vehilces"]

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

    def reward(self, observation, action, actor_type, event):
        return 0.001

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

    fleetpy_config = {"nr_zones": 6,
                      "constant_cfg_path": const_config,
                      "var_cfg_path": sc_config
                      }

    config = (
        PPOConfig()
        .environment(FleetPyRepoRL, env_config=fleetpy_config,
        ).env_runners(num_env_runners=0).learners(num_learners=0)
    )
    algo = config.build_algo()
    print(algo.train())




    

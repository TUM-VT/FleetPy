import sys
import os

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))) # add fleetpy path

from src.ml_gym.FleetPyGymInterface import FleetPyGym
from src.misc.globals import *
from src.ml_gym.hooks_manager import Events
from src.ml_gym.Observers.repositioning_observers import SimTimeObserver, DemandForecastObserver, ZoneBasedVehicleStatesObserver

from src.ml_gym.Actors.repositioning import ZoneBasedRepositioningActor
import src.misc.config as config
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOConfig
import random


class RLReposition(ZoneBasedRepositioningActor):

    def translate_action(self, observation, action):
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


class FleetPyRepoRL(FleetPyGym):

    def __init__(self, config):
        fleetpy_config = config["fleetpy_config"]
        super().__init__(fleetpy_config)
        self.nr_zones = config["nr_zones"]
        fleet_size = sum(fleetpy_config["op_fleet_composition"].values())

        self.action_space = spaces.MultiDiscrete(self.nr_zones*[fleet_size])
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(3 * fleet_size * self.nr_zones,), dtype=np.float32)

        # Register FleetPy observers
        event = Events.OBSERVE_BEFORE_REPOSITIONING
        sim_observer = SimTimeObserver()
        self.register_observer(event, sim_observer)
        self.register_observer(event, DemandForecastObserver())
        self.register_observer(event, ZoneBasedVehicleStatesObserver())

        # register FleetPy actors
        self.register_actor(event, RLReposition())

    def translate_observation(self, observation):
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

    def reward(self, observation, action):
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

    constant_cfg = config.ConstantConfig(const_config)
    scenario_cfgs = config.ScenarioConfig(sc_config)

    study_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(const_config))))
    constant_cfg[G_STUDY_NAME] = study_name
    constant_cfg["n_cpu_per_sim"] = 1
    constant_cfg["evaluate"] = 1
    constant_cfg["log_level"] = "info"

    fleetpy_config = constant_cfg + scenario_cfgs[0]

    fleetpy_config = {"nr_zones": 6,
                      "fleetpy_config": fleetpy_config,
                      }

    config = (
        PPOConfig()
        .environment(FleetPyRepoRL, env_config=fleetpy_config,
        ).env_runners(num_env_runners=0).learners(num_learners=0)
    )
    algo = config.build()
    print(algo.train())




    

"""
Example: RL-based vehicle repositioning with FleetPy + Ray RLlib (PPO)
=======================================================================

This file demonstrates how to wrap FleetPy as a Gymnasium environment and
train a PPO agent to control vehicle repositioning decisions.

To adapt this example for a different problem:
  1. Replace or extend RLReposition.translate_action() to decode your agent's
     output into the format FleetPy expects.
  2. Replace or extend FleetPyRepoRL with your own FleetPyGym subclass:
       - define action_space and observation_space for your problem
       - register different observers / actors for a different Events hook
       - implement translate_observation() to flatten the observation dict
       - implement reward() with a meaningful signal
  3. Swap PPOConfig for any other RLlib algorithm config.

Run:
    python example_fleetpy_gym_repo_ray.py <constant_config.csv> <sc_config.csv>
    python example_fleetpy_gym_repo_ray.py          # uses the built-in MLtest study
"""

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


class RLReposition(ZoneBasedRepositioningActor):
    """Actor that translates the RL agent's output into (origin, target) zone pairs for FleetPy.

    ZoneBasedRepositioningActor.translate_action() is the only method you need to override.
    It receives the raw observation dict and the raw action produced by the RL network, and
    must return a list of (origin_zone_id, target_zone_id) tuples. FleetPy then moves one
    idle vehicle per tuple from the origin zone to the target zone.

    The current implementation ignores the RL action and instead does a random demand-driven
    matching — replace this logic with your actual action decoding once you have a trained policy.
    """

    def translate_action(self, observation, action):
        """Convert the RL agent's action into a list of zone-to-zone repositioning moves.

        :param observation: merged dict from all registered observers, contains at minimum:
            - "zone_to_fc_rq_origins"  (dict[zone_id -> forecasted departures])
            - "zone_to_idle_vehilces"  (dict[zone_id -> idle vehicle count])
        :param action: raw output of the RL network (MultiDiscrete array in this example);
            ignored here — replace with real decoding logic for your trained agent.
        :return: list of (origin_zone_id, target_zone_id) tuples; one vehicle moves per tuple.
        """
        print("incoming action: ", action)
        zone_to_fc_rq_origins = observation["zone_to_fc_rq_origins"]
        zone_to_idle_vehilces = observation["zone_to_idle_vehilces"]

        # Build a pool of target zones weighted by forecasted demand (one entry per expected trip).
        # Replace this with logic that decodes `action` directly once you have a trained policy.
        list_repo_targets = []
        for zone_id, value in zone_to_fc_rq_origins.items():
            if zone_id >= 0:
                for _ in range(int(value)):
                    list_repo_targets.append(zone_id)

        # Build a pool of origin zones weighted by idle vehicle count (one entry per idle vehicle).
        list_repo_origins = []
        for zone_id, value in zone_to_idle_vehilces.items():
            if zone_id > 0:
                for _ in range(int(value)):
                    list_repo_origins.append(zone_id)

        # Randomly match origins to targets until one pool is exhausted.
        list_repo_actions = []
        while len(list_repo_targets) > 0 and len(list_repo_origins) > 0:
            origin = random.choice(list_repo_origins)
            target = random.choice(list_repo_targets)
            list_repo_actions.append((origin, target))
            list_repo_targets.remove(target)
            list_repo_origins.remove(origin)

        return list_repo_actions


class FleetPyRepoRL(FleetPyGym):
    """Gymnasium environment for RL-based vehicle repositioning in FleetPy.

    Subclasses FleetPyGym, which handles the gymnasium.Env interface and runs
    FleetPy in a background thread. This class is responsible for:
      - loading FleetPy scenario configs
      - defining the action and observation spaces
      - registering observers (what to read from FleetPy) and an actor (what to write back)
      - implementing translate_observation() and reward()

    Expected keys in `config` (passed as env_config to RLlib):
        constant_cfg_path (str): path to FleetPy constant_config.csv
        var_cfg_path      (str): path to FleetPy scenario config CSV
        nr_zones          (int): number of zones in the zone system
    """

    def __init__(self, config):
        # --- Load FleetPy configs -------------------------------------------
        constant_cfg = ConstantConfig(config["constant_cfg_path"])
        scenario_cfgs = ScenarioConfig(config["var_cfg_path"])

        study_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(config["constant_cfg_path"]))))
        constant_cfg[G_STUDY_NAME] = study_name
        constant_cfg["n_cpu_per_sim"] = 1
        constant_cfg["evaluate"] = 1
        constant_cfg["log_level"] = "info"

        # Select which scenario config to use.
        # When running multiple Ray workers you can use config.worker_index to assign
        # each worker a different scenario, e.g.:
        #   scenario_inx = (config.worker_index - 1) % len(scenario_cfgs)
        scenario_inx = 0
        fleetpy_config = constant_cfg + scenario_cfgs[scenario_inx]

        # Give each Ray rollout worker its own output folder so result files don't collide.
        if config.worker_index > 0:
            fleetpy_config[G_SCENARIO_NAME] = fleetpy_config[G_SCENARIO_NAME] + f"_worker_{config.worker_index}"

        super().__init__(fleetpy_config)

        # --- Define Gymnasium spaces ----------------------------------------
        self.nr_zones = config["nr_zones"]
        fleet_size = sum(fleetpy_config["op_fleet_composition"].values())

        # Action: for each of the nr_zones x nr_zones zone-pairs, how many vehicles to move.
        # MultiDiscrete means each element is independently bounded by fleet_size.
        # Adapt this to match the action representation your policy network produces.
        self.action_space = spaces.MultiDiscrete(self.nr_zones * self.nr_zones * [fleet_size])

        # Observation: flat vector of [idle, forecast_origins, forecast_destinations] per zone.
        # Shape = 3 * nr_zones. Adjust shape and bounds to match your translate_observation() output.
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(3 * self.nr_zones,), dtype=np.float32)

        # --- Register observers and actor ------------------------------------
        # All three observers fire at the same hook point so their dicts are merged
        # into one combined observation passed to translate_observation() and reward().
        event = Events.OBSERVE_BEFORE_REPOSITIONING
        sim_observer = SimTimeObserver()
        self.register_observer(event, sim_observer)
        self.register_observer(event, DemandForecastObserver())
        self.register_observer(event, ZoneBasedVehicleStatesObserver())

        # The actor pauses the simulation, hands the observation to the gym loop,
        # waits for the RL action, then writes it back into FleetPy.
        self.register_actor(event, RLReposition())

    def translate_observation(self, observation):
        """Flatten the raw FleetPy observation dict into a fixed-size numpy vector.

        This is one of two methods you must implement when subclassing FleetPyGym.
        The output shape must match self.observation_space.

        Current encoding (length = 3 * nr_zones):
            [idle_z0, ..., idle_zN, origins_z0, ..., origins_zN, destinations_z0, ..., destinations_zN]

        :param observation: merged dict from all registered observers.
        :return: np.ndarray of shape (3 * nr_zones,), dtype float32.
        """
        print("translate observation", observation)
        zone_to_fc_rq_origins = observation["zone_to_fc_rq_origins"]
        zone_to_fc_rq_destinations = observation["zone_to_fc_rq_destinations"]
        zone_to_idle_vehilces = observation["zone_to_idle_vehilces"]
        zone_to_overall_available_vehilces = observation["zone_to_overall_available_vehilces"]
        zone_to_current_repositioning_vehicles = observation["zone_to_current_repositioning_vehicles"]

        # Zone -1 is a FleetPy placeholder for vehicles not yet assigned to any zone; exclude it.
        all_zone_ids = list(zone_to_idle_vehilces.keys())
        all_zone_ids.remove(-1)

        idle = np.array([zone_to_idle_vehilces.get(zone_id, 0) for zone_id in all_zone_ids], dtype=np.float32)
        req_origins = np.array([zone_to_fc_rq_origins.get(zone_id, 0)  for zone_id in all_zone_ids], dtype=np.float32)
        req_destinations = np.array([zone_to_fc_rq_destinations.get(zone_id, 0)  for zone_id in all_zone_ids], dtype=np.float32)

        processed_observation = np.concatenate([idle, req_origins, req_destinations], axis=0).astype(np.float32)
        return processed_observation

    def reward(self, observation, action, actor_type):
        """Compute the scalar reward signal returned to the RL agent after each step.

        This is the second method you must implement when subclassing FleetPyGym.
        The observation dict contains the same keys as in translate_observation().

        The current implementation returns a constant placeholder — replace it with a
        meaningful signal, e.g.:
            - negative mean passenger wait time
            - number of served requests
            - negative total repositioning distance

        :param observation: merged dict from all registered observers.
        :param action: the action that was sent to the actor (raw RL output).
        :param actor_type: type of the actor that triggered this step.
        :return: scalar float reward.
        """
        return 0.001  # TODO: replace with a meaningful reward signal


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

    # env_config is forwarded to FleetPyRepoRL.__init__ as the `config` argument.
    fleetpy_config = {"nr_zones": 6,
                      "constant_cfg_path": const_config,
                      "var_cfg_path": sc_config
                      }

    # num_env_runners=0 runs rollouts in the main process (easier for debugging).
    # Increase num_env_runners to parallelize data collection across multiple FleetPy instances.
    config = (
        PPOConfig().training(train_batch_size=48, minibatch_size=48)
        .environment(FleetPyRepoRL, env_config=fleetpy_config,
        ).env_runners(num_env_runners=0).learners(num_learners=0)
    )
    algo = config.build()
    print(algo.train())

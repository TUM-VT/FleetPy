import sys
import os

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) ))) # add fleetpy path

from src.ml_gym.FleetPyGymInterface import FleetPyGym
from src.misc.globals import *
from src.ml_gym.hooks_manager import Events
from src.ml_gym.Observers.repositioning_observers import SimTimeObserver, DemandForecastObserver, ZoneBasedVehicleStatesObserver, ZoneBasedCurrentDemandObserver

from src.ml_gym.Actors.repositioning import ZoneBasedRepositioningActor
from src.misc.config import ConstantConfig, ScenarioConfig
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOConfig # TODO: you can use other libraries that rely on gym
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

        :param observation: merged dict from all registered observers
        :param action: raw output of the RL network (MultiDiscrete array in this example);
        :return: list of (origin_zone_id, target_zone_id) tuples; one vehicle moves per tuple.
        """
        print("incoming action: ", action)

        # TODO: translate that into your action here.
        # the output format should be a list of (origin_zone_id, target_zone_id) tuples, e.g.:
        # return [(0, 2), (0, 2), (1, 3)]  # move 2 vehicles from zone 0 to 2, and 1 vehicle from zone 1 to 3
        list_repo_actions = []

        return list_repo_actions


class TakashiRLRepo(FleetPyGym):
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
        # TODO: replace with your actual action space. The current shape is just a placeholder and doesn't reflect any real constraints (e.g. available idle vehicles in origin zones).
        self.action_space = spaces.MultiDiscrete(self.nr_zones * self.nr_zones * [fleet_size])

        # Observation: flat vector of [idle, forecast_origins, forecast_destinations] per zone.
        # Shape = 3 * nr_zones. Adjust shape and bounds to match your translate_observation() output.
        # TODO: replace with your actual observation space. The current shape and bounds are just placeholders and may not reflect the true range of values in the observation.
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(3 * self.nr_zones,), dtype=np.float32)

        # --- Register observers and actor ------------------------------------
        # All three observers fire at the same hook point so their dicts are merged
        # into one combined observation passed to translate_observation() and reward().
        event = Events.OBSERVE_BEFORE_REPOSITIONING
        # TODO: what observations do you want to read from FleetPy? Implement them as AbstractObserver subclasses and register them here. The current ones are just examples.
        sim_observer = SimTimeObserver()
        self.register_observer(event, sim_observer)
        self.register_observer(event, DemandForecastObserver())
        self.register_observer(event, ZoneBasedVehicleStatesObserver())
        self.register_observer(event, ZoneBasedCurrentDemandObserver())

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
        # TODO: implement your actual observation translation logic here. The current implementation is just an example that combines some of the observed values into a flat vector, but you can customize it as needed based on what your observers return and what information you want to feed into the RL policy.
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
        scs_path = os.path.join(MAIN_DIR, "studies", "ml_test", "scenarios")
        const_config = os.path.join(scs_path, "constant_config.csv")
        sc_config = os.path.join(scs_path, "sc_config_repo.csv")

    # env_config is forwarded to FleetPyRepoRL.__init__ as the `config` argument.
    fleetpy_config = {"nr_zones": 6,
                      "constant_cfg_path": const_config,
                      "var_cfg_path": sc_config
                      }

    # num_env_runners=0 runs rollouts in the main process (easier for debugging).
    # Increase num_env_runners to parallelize data collection across multiple FleetPy instances.
    # TODO: adjust the RLlib config as needed (e.g. learning algorithm, hyperparameters, number of workers, etc.). The current config is just a placeholder to get you started.
    # TODO: or use other lib like stable-baselines3 or your own training loop instead of RLlib if you prefer. The key part is that the environment (FleetPyRepoRL) can be used with any library that supports gymnasium.Env.
    config = (
        PPOConfig().training(train_batch_size=48, minibatch_size=48)
        .environment(TakashiRLRepo, env_config=fleetpy_config,
        ).env_runners(num_env_runners=0).learners(num_learners=0)
    )
    algo = config.build()
    print(algo.train())
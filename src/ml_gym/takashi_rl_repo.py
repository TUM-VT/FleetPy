import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)) )))

from src.ml_gym.FleetPyGymInterface import FleetPyGym
from src.ml_gym.Observers.repositioning_observers import SimTimeObserver, FutureDropoffObserver, FutureRepositioningCompletionObserver, IdleVehiclesObserver, UnservedRequestsObserver, FutureRequestsObserver, TravelTimeMatrixObserver
from src.ml_gym.Actors.repositioning import ZoneBasedRepositioningActor
from src.misc.globals import *
from src.ml_gym.hooks_manager import Events
from src.misc.config import ConstantConfig, ScenarioConfig

from gymnasium import spaces
from stable_baselines3 import PPO

from scipy.optimize import linprog

np.set_printoptions(precision=2, suppress=True)

class RLReposition(ZoneBasedRepositioningActor):
    """Actor that translates the RL agent's output into (origin, target) zone pairs for FleetPy.
    ZoneBasedRepositioningActor.translate_action() is the only method you need to override.
    It receives the raw observation dict and the raw action produced by the RL network, and
    must return a list of (origin_zone_id, target_zone_id) tuples. FleetPy then moves one
    idle vehicle per tuple from the origin zone to the target zone.
    The current implementation ignores the RL action and instead does a random demand-driven
    matching — replace this logic with your actual action decoding once you have a trained policy.
    """
    def __init__(self):
        self.nr_zones = fleetpy_config["nr_zones"]
        self.zone_ids = list(range(self.nr_zones))

    def translate_action(self, observation, action):
        """Convert the RL agent's action into a list of zone-to-zone repositioning moves.
        :param observation: merged dict from all registered observers
        :param action: raw output of the RL network (MultiDiscrete array in this example);
        :return: list of (origin_zone_id, target_zone_id) tuples; one vehicle moves per tuple.
        """
        # print("incoming action: ", action)

        # TODO: translate that into your action here.
        # the output format should be a list of (origin_zone_id, target_zone_id) tuples, e.g.:
        # return [(0, 2), (0, 2), (1, 3)]  # move 2 vehicles from zone 0 to 2, and 1 vehicle from zone 1 to 3
        
        actions, _, _, _, _ = repo_optimization(action, observation, self.zone_ids, self.nr_zones)

        return actions

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

        super().__init__(fleetpy_config)

        self.tau = int(
            fleetpy_config["op_repo_horizons"][1]
            / fleetpy_config["op_repo_timestep"]
            )

        # --- Define Gymnasium spaces ----------------------------------------
        self.nr_zones = config["nr_zones"]
        self.zone_ids = list(range(self.nr_zones))
        fleet_size = sum(fleetpy_config["op_fleet_composition"].values())

        # Action: for each of the nr_zones x nr_zones zone-pairs, how many vehicles to move.
        # MultiDiscrete means each element is independently bounded by fleet_size.
        # Adapt this to match the action representation your policy network produces.
        # TODO: replace with your actual action space. The current shape is just a placeholder and doesn't reflect any real constraints (e.g. available idle vehicles in origin zones).
        self.action_space = spaces.Box(
            low=-5,
            high=5,
            shape=(self.nr_zones,),
            dtype=np.float32
        )

        # Observation: flat vector of [idle, forecast_origins, forecast_destinations] per zone.
        # Shape = 3 * nr_zones. Adjust shape and bounds to match your translate_observation() output.
        # TODO: replace with your actual observation space. The current shape and bounds are just placeholders and may not reflect the true range of values in the observation.
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=((3 * self.tau + 2) * self.nr_zones,),
            dtype=np.float32
        )

        # --- Register observers and actor ------------------------------------
        # All three observers fire at the same hook point so their dicts are merged
        # into one combined observation passed to translate_observation() and reward().
        event = Events.OBSERVE_BEFORE_REPOSITIONING

        # TODO: what observations do you want to read from FleetPy? Implement them as AbstractObserver subclasses and register them here. The current ones are just examples.
        sim_observer = SimTimeObserver()
        self.register_observer(event, sim_observer) # t
        self.register_observer(event, FutureDropoffObserver(self.tau)) # x_i^(t+k)
        self.register_observer(event, FutureRepositioningCompletionObserver()) # y_i^(t+k)
        self.register_observer(event, IdleVehiclesObserver()) # z_i^t
        self.register_observer(event, UnservedRequestsObserver()) # ru_i^t
        self.register_observer(event, FutureRequestsObserver(self.tau)) # rf_i^(t+k)
        self.register_observer(event, TravelTimeMatrixObserver()) # tt_i,j

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
        # print("translate observation", observation)
        # TODO: implement your actual observation translation logic here. The current implementation is just an example that combines some of the observed values into a flat vector, but you can customize it as needed based on what your observers return and what information you want to feed into the RL policy.
        
        zone_to_future_dropoffs = observation["zone_to_future_dropoffs"]
        zone_to_future_repo_completions = observation["zone_to_future_repo_completions"]
        zone_to_idle_vehicles = observation["zone_to_idle_vehicles"]
        zone_to_unserved_requests = observation["zone_to_unserved_requests"]
        zone_to_forecasted_requests = observation["zone_to_forecasted_requests"]

        dropoffs = np.array([
            zone_to_future_dropoffs[k].get(zone_id, 0)
            for k in range(1, self.tau + 1)
            for zone_id in self.zone_ids
        ], dtype=np.float32)

        repo_completions = np.array([
            zone_to_future_repo_completions[k].get(zone_id, 0)
            for k in range(1, self.tau + 1)
            for zone_id in self.zone_ids
        ], dtype=np.float32)

        idles = np.array([zone_to_idle_vehicles.get(zone_id, 0) for zone_id in self.zone_ids], dtype=np.float32)

        unserved_rq = np.array([zone_to_unserved_requests.get(zone_id, 0) for zone_id in self.zone_ids], dtype=np.float32)

        forecasted_rq = np.array([
            zone_to_forecasted_requests[k].get(zone_id, 0)
            for k in range(1, self.tau + 1)
            for zone_id in self.zone_ids
        ], dtype=np.float32)

        processed_observation = np.concatenate([dropoffs, repo_completions, idles, unserved_rq, forecasted_rq], axis=0).astype(np.float32)

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
        # check
        # print("reward called, cumulative_unserved:", observation["cumulative_unserved"])

        actions, dn, dn_int, u, tt_matrix = repo_optimization(action, observation, self.zone_ids, self.nr_zones)
        
        if dn_int is None:
            return -1e6

        # Weights
        w1 = 1
        w2 = 0.001
        w3 = 0.1

        # Cost terms
        cost_unserved = observation["cumulative_unserved"]
        cost_travel = np.sum(dn_int * tt_matrix)
        cost_deviation = np.sum(u)

        reward = - (w1 * cost_unserved + w2 * cost_travel + w3 * cost_deviation)

        #print(f"cost_unserved  = {cost_unserved:.3f}")
        #print(f"cost_travel    = {cost_travel:.3f}")
        #print(f"cost_deviation = {cost_deviation:.3f}")
        #print(f"reward         = {reward:.3f}")

        return float(reward)

# Optimize repositioning based on action (desired proportion)
def repo_optimization(action, observation, zone_ids, nr_zones):
    # Softmax
    exp_action = np.exp(action - np.max(action))
    dp = exp_action / np.sum(exp_action)
    zone_to_idle = observation["zone_to_idle_vehicles"] # no. of idle vehicles by zones
    tt_matrix = observation["tt_matrix"] # matrix of travel times among zones
    
    Z = nr_zones
    z = np.array([zone_to_idle.get(i, 0) for i in zone_ids])

    total_idle = np.sum(z) # total number of idle vehicles
    target = dp * total_idle # desired no. of idle vehicles by zones

    n_d = Z * Z
    n_u = Z
    c = []

    # Vectorize tt_matrix
    for i in range(Z):
        for j in range(Z):
            c.append(tt_matrix[i, j])
    
    # Add the deviation penalty term in the objective function
    lam = 100 # Lagrangerian multiplier
    c += [lam] * Z
    c = np.array(c)
    
    # Constraints: Ax <= b
    A = []
    b = []

    # Constraint 1: siguma_j(dn_i,j) < z_i (outflow from zone i must be equal or smaller than idle vehcles)
    for i in range(Z):
        row = np.zeros(n_d + n_u)
        for j in range(Z):
            row[i * Z + j] = 1
        A.append(row)
        b.append(z[i])
    
    # Constraint 2: - u_j + sigma_i(dn_ij) - sigma_k(dn_jk) <= dp_j * sigma_k(z_k) - z_j 
    for j in range(Z):
        row = np.zeros(n_d + n_u)

        # inflow
        for i in range(Z):
            row[i * Z + j] += 1
        # outflow
        for k in range(Z):
            row[j * Z + k] -= 1
        #u_j
        row[n_d + j] = -1

        A.append(row)
        b.append(target[j] - z[j])

    # Constraint 3: - u_j - sigma_i(dn_ij) + sigma_k(dn_jk) <= - dp_j * sigma_k(z_k) + z_j
    for j in range(Z):
        row = np.zeros(n_d + n_u)

        # inflow
        for i in range(Z):
            row[i * Z + j] -= 1
        # outflow
        for k in range(Z):
            row[j * Z + k] += 1
        #u_j
        row[n_d + j] = -1

        A.append(row)
        b.append(z[j] - target[j])
    
    A = np.array(A)
    b = np.array(b)

    # Boundary
    bounds = [(0, None)] * (n_d + n_u)

    # Solve the optimization problem by linprog
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    # Extract only repositioning components from the solution vector and reconstruct dn_ij
    if not res.success:
        return [], None, None, tt_matrix
    x = res.x[:n_d]
    u = res.x[n_d:]
    dn = x.reshape((Z, Z))
    
    # Largest remainder method
    dn_int = np.floor(dn).astype(int)
    remainder = dn - dn_int

    for i in range(Z):
        desired = np.sum(dn[i])
        current = np.sum(dn_int[i])
        extra = int(np.round(desired - current))

        if extra <= 0:
            continue
        order = np.argsort(-remainder[i])

        for j in order:
            if extra == 0:
                break
            if current < z[i]:
                dn_int[i, j] += 1
                current += 1
                extra -= 1

    # Convert into FleetPy format
    actions = []

    for i in range(Z):
        for j in range(Z):
            for _ in range(dn_int[i, j]):
                actions.append((zone_ids[i], zone_ids[j]))

    return actions, dn, dn_int, u, tt_matrix

# run RL
if __name__ == "__main__":

    MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Condition for default or manual input of configuration files　
    if len(sys.argv) >= 3: # case when 3 arguments are input in terminal (arg0:exe file, arg1:const_config, arg2:scenario_config)
        const_config = sys.argv[1]
        sc_config = sys.argv[2]
    else:
        # default (without arguments) paths for 2 csv files
        scs_path = os.path.join(MAIN_DIR, "studies", "ml_test", "scenarios") # studies/ml_test/scenarios
        const_config = os.path.join(scs_path, "const_cfg_manhattan_case_study.yaml") # studies/ml_test/scenarios/constant_config.csv
        sc_config = os.path.join(scs_path, "scenario_cfg_manhattan_ml_takashi.csv")

    # env_config is forwarded to FleetPyRepoRL.__init__ as the `config` argument.
    fleetpy_config = {"nr_zones": 63,
                    "constant_cfg_path": const_config,
                    "var_cfg_path": sc_config
                    }

    # num_env_runners=0 runs rollouts in the main process (easier for debugging).
    # Increase num_env_runners to parallelize data collection across multiple FleetPy instances.
    # TODO: adjust the RLlib config as needed (e.g. learning algorithm, hyperparameters, number of workers, etc.). The current config is just a placeholder to get you started.
    # TODO: or use other lib like stable-baselines3 or your own training loop instead of RLlib if you prefer. The key part is that the environment (FleetPyRepoRL) can be used with any library that supports gymnasium.Env.

    env = TakashiRLRepo(fleetpy_config)
    
    model = PPO('MlpPolicy',
                env,
                learning_rate=3e-4,
                n_steps=24,
                batch_size=24,
                n_epochs=10,
                gamma=0.99,
                clip_range=0.2,
                ent_coef=0,
                vf_coef=0.5,
                verbose=1, # log detail→0:none, 1:standard, 2:detail debug
                tensorboard_log="./tensorboard/"
                )
    
    model.learn(total_timesteps=100, tb_log_name="PPO_FleetPy")

    model.save("ppo_repo_model_2")
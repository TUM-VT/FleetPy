import sys
import os
import numpy as np

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

        dp = action / (np.sum(action) + 1e-8) # normalize the action (avoid 0 division)
        zone_to_idle = observation["zone_to_idle_vehicles"]
        tt_matrix = observation["tt_matrix"]

        # Zone -1 is a FleetPy placeholder for vehicles not yet assigned to any zone; exclude it.
        all_zone_ids = get_zone_ids(observation)
        
        Z = len(all_zone_ids)
        z = np.array([zone_to_idle[i] for i in all_zone_ids])

        total_idle = np.sum(z)
        target = dp * total_idle

        n_d = Z * Z
        n_u = Z

        c = []

        for i in range(Z):
            for j in range(Z):
                c.append(tt_matrix[i, j])
        
        lam = 10.0
        c += [lam] * Z
        c = np.array(c)
        
        # Constraints
        A = []
        b = []

        # Constraint 1: siguma_j(dn_i,j) < z_i
        for i in range(Z):
            row = np.zeros(n_d + n_u)
            for j in range(Z):
                row[i * Z + j] = 1
            A.append(row)
            b.append(z[i])
        
        # Constraint 2: u_j >= imbalance
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

        # Constraint 3: u_j >= -imbalance
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

        # boundary
        bounds = [(0, None)] * (n_d + n_u)

        # solve
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        if not res.success:
            return []
        x = res.x[:n_d]

        # reconstruct d_ij
        d_matrix = x.reshape((Z, Z))

        # convert into FleetPy format
        actions = []

        for i in range(Z):
            for j in range(Z):
                move = max(0, int(np.floor(d_matrix[i, j])))
                for _ in range(move):
                    actions.append((all_zone_ids[i], all_zone_ids[j]))

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

        horizon = fleetpy_config["op_repo_horizons"][1]
        resolution = fleetpy_config["op_temporal_resolution"]
        self.tau = int(horizon / resolution)

        # --- Define Gymnasium spaces ----------------------------------------
        self.nr_zones = config["nr_zones"]
        fleet_size = sum(fleetpy_config["op_fleet_composition"].values())

        # Action: for each of the nr_zones x nr_zones zone-pairs, how many vehicles to move.
        # MultiDiscrete means each element is independently bounded by fleet_size.
        # Adapt this to match the action representation your policy network produces.
        # TODO: replace with your actual action space. The current shape is just a placeholder and doesn't reflect any real constraints (e.g. available idle vehicles in origin zones).
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
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
        print("translate observation", observation)
        # TODO: implement your actual observation translation logic here. The current implementation is just an example that combines some of the observed values into a flat vector, but you can customize it as needed based on what your observers return and what information you want to feed into the RL policy.
        
        zone_to_future_dropoffs = observation["zone_to_future_dropoffs"]
        zone_to_future_repo_completions = observation["zone_to_future_repo_completions"]
        zone_to_idle_vehicles = observation["zone_to_idle_vehicles"]
        zone_to_unserved_requests = observation["zone_to_unserved_requests"]
        zone_to_forecasted_requests = observation["zone_to_forecasted_requests"]

        # Zone -1 is a FleetPy placeholder for vehicles not yet assigned to any zone; exclude it.
        all_zone_ids = get_zone_ids(observation)

        dropoffs = np.array([
            zone_to_future_dropoffs[k].get(zone_id, 0)
            for k in range(1, self.tau + 1)
            for zone_id in all_zone_ids
        ], dtype=np.float32)

        repo_completions = np.array([
            zone_to_future_repo_completions[k].get(zone_id, 0)
            for k in range(1, self.tau + 1)
            for zone_id in all_zone_ids
        ], dtype=np.float32)

        idles = np.array([zone_to_idle_vehicles.get(zone_id, 0) for zone_id in all_zone_ids], dtype=np.float32)

        unserved_rq = np.array([zone_to_unserved_requests.get(zone_id, 0) for zone_id in all_zone_ids], dtype=np.float32)

        forecasted_rq = np.array([
            zone_to_forecasted_requests[k].get(zone_id, 0)
            for k in range(1, self.tau + 1)
            for zone_id in all_zone_ids
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
        print("reward called, cumulative_unserved:", observation["cumulative_unserved"])

        # observation
        zone_to_idle = observation["zone_to_idle_vehicles"]
        tt_matrix = observation["tt_matrix"]
        cost_unserved = observation["cumulative_unserved"]
        
        all_zones_ids = get_zone_ids(observation)
        Z = len(all_zones_ids)
        z = np.array([zone_to_idle[i] for i in all_zones_ids])

        # action
        dp = action / (np.sum(action) + 1e-8)
        total_idle = np.sum(z)
        target = dp * total_idle

        # optimization
        n_d = Z * Z
        n_u = Z
        
        c = []
        for i in range(Z):
            for j in range(Z):
                c.append(tt_matrix[i, j])
        
        lam = 10.0
        c += [lam] * Z
        c = np.array(c)

        A = []
        b = []

        # outflow contraints
        for i in range(Z):
            row = np.zeros(n_d + n_u)
            for j in range(Z):
                row[i * Z + j] = 1
            A.append(row)
            b.append(z[i])
        
        # deviation constraints
        for j in range(Z):
            row = np.zeros(n_d + n_u)
            for i in range(Z):
                row[i * Z + j] += 1
            for k in range(Z):
                row[j * Z + k] -= 1
            row[n_d + j] = -1
            A.append(row)
            b.append(target[j] - z[j])
        
            row = np.zeros(n_d + n_u)
            for i in range(Z):
                row[i * Z + j] -= 1
            for k in range(Z):
                row[j * Z + k] += 1
            row[n_d + j] = -1
            A.append(row)
            b.append(z[j] - target[j])

        bounds = [(0, None)] * (n_d + n_u)


        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        if not res.success:
            return -1e6
        
        x = res.x
        dn = x[:n_d].reshape((Z, Z))
        u = x[n_d:]

        # weights
        w1 = 1.0
        w2 = 1.0
        w3 = 1.0


        # cost terms 2, 3
        cost_travel = np.sum(dn * tt_matrix)
        cost_deviation = np.sum(u)

        reward = - (w1 * cost_unserved + w2 * cost_travel + w3 * cost_deviation)

        for hook_list in self._hook_manager._hooks.values():
            for hook in hook_list:
                observers, _ = hook.get_observers_actors()
                for obs in self._observers:
                    if isinstance(obs, UnservedRequestsObserver):
                        obs.cumulative_unserved = 0
        
        return float(reward)

def get_zone_ids(observation):
    # Exclude zone no. -1
    zone_dict = observation.get("zone_to_idle_vehicles", {})
    ids = sorted(zone_dict.keys())
    if -1 in ids:
        ids.remove(-1)
    return ids

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
        const_config = os.path.join(scs_path, "constant_config.csv") # studies/ml_test/scenarios/constant_config.csv
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

    env = TakashiRLRepo(fleetpy_config)
    
    model = PPO('MlpPolicy',
                env,
                verbose=1, # how detailed log is output→　0:none, 1:standard, 2:detail debug
                learning_rate=3e-4, # how much parameters are changed in 1 update
                n_steps=1024, # how many information steps are collected from environment before update
                batch_size=64 # number of data used in 1 gradient update
                )
    
    model.learn(total_timesteps=100000)
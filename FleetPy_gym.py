import gymnasium as gym
from gymnasium import spaces
import numpy as np

# import FleetPy modules
from src.misc.globals import *
import src.misc.config as config
from src.misc.init_modules import load_simulation_environment
from src.RLBatchOfferSimulation import RLBatchOfferSimulation

class FleetPyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, RL_config):
        super(FleetPyEnv, self).__init__()
        use_case = RL_config["use_case"]
        action_no = RL_config["action_no"]
        start_config_i = RL_config["start_config_i"]
        self.use_case = use_case
        self.action_no = action_no
        # Initialize your FleetPy simulation here using the config argument if necessary
        scs_path = os.path.join(os.path.dirname(__file__), "studies", "SoDZonal", "scenarios")

        cc = os.path.join(scs_path, "constant_config_pool.csv")
        if use_case == "train" or use_case == "baseline":
            log_level = "info"
            sc = os.path.join(scs_path, "zonal_RL.csv")
        elif use_case == "test" or use_case == "baseline_test":
            log_level = "debug"
            sc = os.path.join(scs_path, "example_test.csv")


        constant_cfg = config.ConstantConfig(cc)
        scenario_cfgs = config.ScenarioConfig(sc)
        const_abs = os.path.abspath(cc)
        study_name = os.path.basename(os.path.dirname(os.path.dirname(const_abs)))

        constant_cfg[G_STUDY_NAME] = study_name
        constant_cfg["n_cpu_per_sim"] = 1
        constant_cfg["evaluate"] = 1
        constant_cfg["log_level"] = log_level
        constant_cfg["keep_old"] = False

        if use_case == "train" or use_case == "baseline":
            constant_cfg["skip_file_writing"] = 1
        else:
            constant_cfg["skip_file_writing"] = 0

        # combine constant and scenario parameters into verbose scenario parameters
        for i, scenario_cfg in enumerate(scenario_cfgs):
            scenario_cfgs[i] = constant_cfg + scenario_cfg
        self.scenario_cfgs = scenario_cfgs
        self.current_config_i = start_config_i

        print(f"Loading simulation environment {self.current_config_i}...")
        self.SF: RLBatchOfferSimulation = load_simulation_environment(self.scenario_cfgs[self.current_config_i])
        self.SF.run(RL_init=True)
        self.sim_time = self.SF.start_time

        # self.gamma = 0.98

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when assuming discrete actions:
        self.n_action = 3  # number of actions
        # self.action_space = spaces.Discrete(self.n_action)  # Adjust N to your number of actions
        self.n_action_boundary = (2-1) * 2 + 1 # move a zone boundary left/right or do nothing
        if action_no == 2:
            self.action_space = spaces.MultiDiscrete([self.n_action, self.n_action_boundary])
        elif action_no == 1:
            self.action_space = spaces.Discrete(self.n_action)
        else:
            raise ValueError("Invalid action number")
        # Example for observation space that is an array:
        self.n_state = 13
        # state_val_low = np.zeros(self.n_state)
        # state_val_high = np.array([500,10000,500,5,500,10000,500,50,50,500,500])
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.n_state,), dtype=np.float32)  # Adjust shape and range

        self.n_zone = 2
        self.last_dept = [0] * self.n_zone


    def step(self, action):
        # Execute one time step within the environment
        # You should interact with your FleetPy simulation here based on the action and return the next state, reward, done, and info
        # for sim_time in range(self.SF.start_time, self.SF.end_time, self.SF.time_step):

        # skip the step that do not deploy vehicles?
        if self.sim_time > self.SF.end_time:
            raise ValueError("Simulation has ended. Please reset the environment.")

        # zonal_veh_deployed = None
        # accumulated_reward = 0
        # while zonal_veh_deployed is None and self.sim_time <= self.SF.end_time:

        if self.action_no == 1:
            action = action, self.n_action_boundary-1

        if self.use_case == "baseline" or self.use_case == "baseline_test":
            zonal_headway = 8
            action_z = self.n_zone
            for z in range(self.n_zone):
                self.last_dept[z] += 1
            for z in range(self.n_zone):
                if self.last_dept[z] >= zonal_headway:
                    action_z = z
                    self.last_dept[z] = 0
                    break
            action = action_z, self.n_action_boundary-1

        observation, reward, done, truncated, info, zonal_veh_deployed = self.SF.step(self.sim_time, action)
            # accumulated_reward = reward + accumulated_reward / self.gamma

        # skip first 30 minute reward (initialization)
        if self.sim_time <= self.SF.start_time + 60*30:
            reward = 0

        self.sim_time += self.SF.time_step

        return observation, reward, done, truncated, info

    def reset(self, seed=None, options=None, eval_result=False):
        # Reset the state of the environment to an initial state
        # This often involves restarting the FleetPy simulation
        super().reset(seed=seed)

        if eval_result:
            # record stats
            self.SF.record_stats()

            # save final state, record remaining travelers and vehicle tasks
            self.SF.save_final_state()
            self.SF.record_remaining_assignments()
            self.SF.demand.record_remaining_users()
            if not self.SF.skip_output:
                self.SF.evaluate()

        # move run_single_simulation() here to handle scenario iteration
        self.SF: RLBatchOfferSimulation = load_simulation_environment(self.scenario_cfgs[self.current_config_i])
        self.current_config_i += 1
        if self.current_config_i >= len(self.scenario_cfgs):
            self.current_config_i = 0

        self.SF.run(RL_init=True)
        self.sim_time = self.SF.start_time

        observation, reward, done, truncated, info, zonal_veh_deployed \
            = self.SF.step(self.sim_time, (self.n_action-1, self.n_action_boundary-1)) # do nothing at first timestep
        self.sim_time += self.SF.time_step

        # TODO: look into truncated setting

        return observation, None  # Return the initial observation

    def render(self, mode='human', close=False):
        # Render the environment to the screen or another output. This is optional and may not be needed for FleetPy.
        pass

    def close(self):
        # Perform any cleanup when the environment is closed
        # TODO: carry out metric evaluation
        pass

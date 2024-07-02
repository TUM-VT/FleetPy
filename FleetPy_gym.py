import gymnasium as gym
from gymnasium import spaces
import numpy as np

# import FleetPy modules
from src.misc.globals import *
import src.misc.config as config
from src.misc.init_modules import load_simulation_environment
from src.RLBatchOfferSimulation import RLBatchOfferSimulation

from typing import List
import logging

LOG = logging.getLogger(__name__)


class FleetPyEnv(gym.Env):
    """
    Custom FleetPy environment for Gymnasium API
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, rl_config):
        # Initialize the FleetPy environment
        super(FleetPyEnv, self).__init__()
        use_case: str = rl_config["use_case"]
        action_no = rl_config["action_no"]
        start_config_i = rl_config["start_config_i"]
        cc_file = rl_config["cc_file"]
        sc_file = rl_config["sc_file"]
        self.use_case: str = use_case
        self.action_no = action_no
        # Initialize your FleetPy simulation here using the config argument if necessary
        scs_path = os.path.join(os.path.dirname(__file__), "studies", "SoDZonal", "scenarios")

        cc = os.path.join(scs_path, cc_file)
        # sc = os.path.join(scs_path, "zonal_RL.csv")
        sc = os.path.join(scs_path, sc_file)
        if use_case == "train" or use_case == "baseline" or use_case == "zbaseline" or use_case.endswith("result"):
            log_level = "info"
            # sc = os.path.join(scs_path, "zonal_RL.csv")
        elif use_case == "test" or use_case == "baseline_test" or use_case == "zbaseline_test":
            log_level = "debug"
            # sc = os.path.join(scs_path, "example_test.csv")

        constant_cfg = config.ConstantConfig(cc)
        scenario_cfgs = config.ScenarioConfig(sc)
        const_abs = os.path.abspath(cc)
        study_name = os.path.basename(os.path.dirname(os.path.dirname(const_abs)))

        constant_cfg[G_STUDY_NAME] = study_name
        constant_cfg["n_cpu_per_sim"] = 1
        constant_cfg["evaluate"] = 1
        constant_cfg["log_level"] = log_level
        constant_cfg["keep_old"] = False

        if use_case == "train" or use_case == "baseline" or use_case == "zbaseline":
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
        self.SF.run(rl_init=True)
        self.sim_time = self.SF.start_time

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when assuming discrete actions:
        # TODO: load this from parameters instead of hard code
        self.n_action = 2 + 1 + 1  # number of actions (2 zones, do nothing, regular)
        # self.n_action = 2 + 1  # number of actions (2 zones, do nothing)
        self.n_action_boundary = (2 - 1) * 2 + 1  # move a zone boundary left/right or do nothing
        if action_no == 2:
            self.action_space = spaces.MultiDiscrete([self.n_action, self.n_action_boundary])
        elif action_no == 1:
            self.action_space = spaces.Discrete(self.n_action)
        else:
            raise ValueError("Invalid action number")
        # Example for observation space that is an array:

        # TODO: load this from parameters instead of hard code
        self.n_state = 19
        # state_val_low = np.zeros(self.n_state)
        # state_val_high = np.array([500,10000,500,5,500,10000,500,50,50,500,500])
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.n_state,),
                                            dtype=np.float32)  # Adjust shape and range

        self.n_zone = self.SF.scenario_parameters[G_PT_N_ZONES]

        self.last_dept = [0] * (self.n_zone + 3)
        # if self.use_case.startswith("baseline"):
        #     self.last_dept[-1] = self.SF.start_time - 5 * 60
        if self.use_case.startswith("zbaseline"):
            self.last_dept[0] = self.SF.start_time - 15 * 60
            self.last_dept[1] = self.SF.start_time - 5 * 60

        self.n_SAV_avail = 4
        self.zonal_regular_headway_s = self.SF.scenario_parameters["pt_regular_headway"]

    def step(self, action):
        # Execute one time step within the environment
        # You should interact with your FleetPy simulation here based on the action
        # and return the next state, reward, done, and info
        # for sim_time in range(self.SF.start_time, self.SF.end_time, self.SF.time_step):

        # skip the step that do not deploy vehicles?
        if self.sim_time > self.SF.end_time:
            raise ValueError("Simulation has ended. Please reset the environment.")

        # zonal_veh_deployed = None
        # accumulated_reward = 0
        # while zonal_veh_deployed is None and self.sim_time <= self.SF.end_time:

        if self.action_no == 1:
            action_z = action
        # if action_z > self.n_zone:
        #     action_z = -1

        regular_headway_s = 5 * 60

        # regular_penalty = 0
        if self.use_case.startswith("baseline"):
            # for z in range(self.n_zone + 1):
            #     self.last_dept[z] += 1

            action_z = self.n_zone
            z = -1
            if self.last_dept[z] + regular_headway_s <= self.sim_time:
                action_z = z
            # if action_z != self.n_zone:
            #     self.last_dept[action_z] = self.sim_time

        elif self.use_case.startswith("zbaseline"):
            # regular_headway_s = 15 * 60
            zonal_headway_s = 20 * 60

            # for z in range(self.n_zone + 1):
            #     self.last_dept[z] += 1

            action_z = self.n_zone
            for z in range(self.n_zone):
                if self.last_dept[z] + zonal_headway_s <= self.sim_time:
                    action_z = z
                    break
            z = -1
            if self.last_dept[z] + self.zonal_regular_headway_s <= self.sim_time:
                action_z = z
            # if action_z != self.n_zone:
            #     self.last_dept[action_z] = self.sim_time

        else:
            # if action_z > self.n_zone:
            #     action_z = -1
            z = -1
            if self.last_dept[z] + self.zonal_regular_headway_s <= self.sim_time:
                # if action_z != -1:
                #     regular_penalty = 5
                action_z = z

        action = action_z, self.n_action_boundary - 1
        if action_z != self.n_zone:
            # print(action_z)
            z = action_z
            # if z > self.n_zone:
            #     z = -1
            self.last_dept[z] = self.sim_time
        # observation, reward, done, truncated, info, zonal_veh_deployed = self.SF.step(self.sim_time, action)
        # accumulated_reward = reward + accumulated_reward / self.gamma

        observation, reward, done, truncated, info, zonal_veh_deployed, n_sav_avail = self.SF.step(self.sim_time,
                                                                                                   action)
        # reward -= regular_penalty
        self.n_SAV_avail = n_sav_avail
        self.sim_time += self.SF.time_step
        # fast forward till next dispatchment
        # while self.sim_time % regular_headway_s != 0 and not done:
        #     action = self.n_zone, self.n_action_boundary-1
        #     if self.sim_time % regular_headway_s != 0 and not done:
        #         observation, reward, done, truncated, info, zonal_veh_deployed = self.SF.step(self.sim_time, action)
        #         self.sim_time += self.SF.time_step

        # skip first 60 minute reward (initialization)
        if self.sim_time <= self.SF.start_time + 60 * 60:
            reward = 0

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

        self.SF.run(rl_init=True)
        self.sim_time = self.SF.start_time

        observation, reward, done, truncated, info, zonal_veh_deployed, _ \
            = self.SF.step(self.sim_time,
                           (self.n_action - 1, self.n_action_boundary - 1))  # do nothing at first timestep
        # self.sim_time += self.SF.time_step
        self.n_SAV_avail = 4

        self.last_dept = [0] * (self.n_zone + 3)
        if self.use_case.startswith("zbaseline"):
            self.last_dept[0] = self.SF.start_time - 15 * 60
            self.last_dept[1] = self.SF.start_time - 5 * 60

        # TODO: look into truncated setting
        return observation, None  # Return the initial observation

    def render(self, mode='human', close=False):
        # Render the environment to the screen or another output. This is optional and may not be needed for FleetPy.
        pass

    def close(self):
        # Perform any cleanup when the environment is closed
        pass

    def action_masks(self) -> List[bool]:
        """
        Return action masks
        """
        # check regular time
        z = -1
        if self.last_dept[z] + self.zonal_regular_headway_s <= self.sim_time:
            masks = [False] * (self.n_action - 1) + [True]  # represent sending regular vehicles
        else:
            masks = [True] * (self.n_action - 1) + [False]

            no_sav_avail = self.n_SAV_avail
            if no_sav_avail == 0:
                for i in range(self.n_zone):
                    masks[i] = False
        LOG.debug(f"{self.sim_time} Action masks: {masks}")
        return masks

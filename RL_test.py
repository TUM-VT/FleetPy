import os.path
import sys

from FleetPy_gym import FleetPyEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

import datetime
import warnings

# Wrapper method to create a vectorized environment
def make_fleetpy_env(env_class, n_envs=1, env_kwargs=None):
    """
    Utility function for creating a vectorized FleetPy environment.

    :param env_class: The environment class to be vectorized, should be FleetPyEnv.
    :param n_envs: The number of environments to run in parallel.
    :param env_kwargs: A dictionary of keyword arguments to pass to the environment class.
    :return: A vectorized Gym environment.
    """
    # vec_env = DummyVecEnv
    vec_env = SubprocVecEnv
    env = make_vec_env(env_class, n_envs=n_envs, env_kwargs=env_kwargs, vec_env_cls=vec_env)
    return env

def standard_ppo(use_case, action_no, model_name, iter, cc_file, sc_file, n_envs=1, start_config_i=0, masked=False):
    # Create the environment
    # env = make_vec_env(env_name, n_envs=1)
    # model_name = "ppo_fleetpy_singleaction_1000"
    # model_name = "ppo_fleetpy_multiaction_2000"
    # model_name = "ppo_fleetpy_baseline_1000"
    RL_config = {
        "use_case": use_case,
        "action_no": action_no,
        "start_config_i": start_config_i,
        "cc_file": cc_file,
        "sc_file": sc_file,
    }
    # env = FleetPyEnv(rl_config)
    # use parallelized environment

    # model_name = "ppo_fleetpy_baseline_multiaction"
    model_dir = "./models/"
    log_dir = "./logs/"
    # logger = configure(log_dir, ["stdout", "tensorboard"])
    model_path = os.path.join(model_dir, model_name + ".zip")

    if use_case=="train" or use_case=="baseline" or use_case=="zbaseline":
        env = make_fleetpy_env(FleetPyEnv, n_envs=n_envs, env_kwargs={"rl_config": RL_config})
        # check if model exists


        if os.path.exists(model_path):
            if masked:
                model = MaskablePPO.load(model_path, env)
            else:
                model = PPO.load(model_path, env)

            print("PPO Model loaded!")
        else:
            # Initialize the PPO agent
            if masked:
                model = MaskablePPO("MlpPolicy",
                            env, n_steps=180,
                            learning_rate=0.003,
                            # batch_size=180,
                            gamma=0.99, verbose=1, tensorboard_log=log_dir)
            else:
                model = PPO("MlpPolicy",
                            env, n_steps=180,
                            learning_rate=0.003,
                            clip_range=0.2,
                            # batch_size=180,
                            gamma=0.99, verbose=1, tensorboard_log=log_dir)
            print("PPO Model initialized!")

        # Train the agent
        total_timesteps = 180 * iter
        print(f"Training for {total_timesteps} timesteps")
        # add date to log name
        log_name = model_name + "_" + datetime.datetime.now().strftime("%Y%m%d")
        model.learn(total_timesteps=total_timesteps, tb_log_name=log_name)

        # Save the model
        model.save(model_path)
        print("Model trained and saved!")

    else:
        eval_env = FleetPyEnv(RL_config)
        assert os.path.exists(model_path), "Model does not exist"
        if masked:
            model = MaskablePPO.load(model_path, eval_env)
        else:
            model = PPO.load(model_path, eval_env)
        print("PPO Model loaded!")
        # Evaluate the trained model
        # eval_env = gym.make(env_name)

        obs, _ = eval_env.reset()
        total_rewards = 0
        total_timesteps = 180 * iter
        for itit in range(total_timesteps):

            # action_masks = eval_env.action_masks()
            if use_case.startswith("baseline") or use_case.startswith("zbaseline"):
                action = 0
            else:
                if masked:
                    action_masks = get_action_masks(eval_env)
                    action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
                else:
                    action, _states = model.predict(obs, deterministic=True)

            obs, rewards, dones, truncated, info = eval_env.step(action)
            total_rewards += rewards
            if dones:
                # break
                obs, _ = eval_env.reset()
            if itit % 100 == 0:
                print(f"Iteration: {itit} Cum. Reward: {total_rewards}")

        eval_env.close()
        print(f"Total Reward: {total_rewards}")


# Main execution
if __name__ == "__main__":
    # read argument for use_case, action_no, and model_name
    masked = False

    if len(sys.argv) == 2:
        # read from scenario file
        RL_scenario_file = sys.argv[1]
        file_path = os.path.join(os.path.dirname(__file__), "studies", "SoDZonal", "scenarios", RL_scenario_file)
        # read use_case, action_no, model_name, iter, cc_file, sc_file, n_envs, start_config_i from RL_scenario_file (csv)
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                if line.startswith("#"):
                    continue
                args = line.strip().split(",")
                use_case = args[0]
                action_no = int(args[1])
                model_name = args[2]
                iter = int(args[3])
                cc_file = args[4]
                sc_file = args[5]
                n_envs = int(args[6]) if len(args) >= 7 else 1
                start_config_i = int(args[7]) if len(args) >= 8 else 0
                masked = int(args[8]) > 0 if len(args) >= 9 else False
                print(f"Running scenario: {use_case}, {action_no}, {model_name}, {iter}, {cc_file}, {sc_file}, {n_envs}, {start_config_i}, {masked}")

                if masked:
                    warnings.filterwarnings("ignore", category=UserWarning)
                standard_ppo(use_case, action_no, model_name, iter, cc_file, sc_file, n_envs=n_envs,
                             start_config_i=start_config_i, masked=masked)
    else:
        if len(sys.argv) < 5:
            print("Usage: python RL_test.py <use_case> <action_no> <model_name> <iter> (<n_env>) (<start_config_i>) (<masked>)")
            sys.exit(1)
        use_case = sys.argv[1]
        action_no = int(sys.argv[2])
        model_name = sys.argv[3]
        iter = int(sys.argv[4])
        n_envs = int(sys.argv[5]) if len(sys.argv) >= 6 else 1
        start_config_i = int(sys.argv[6]) if len(sys.argv) >= 7 else 0

        assert use_case in ["train", "test", "test_result",
                            "baseline", "baseline_test", "baseline_result",
                            "zbaseline", "zbaseline_test", "zbaseline_result"], "Invalid use_case"
        assert action_no in [1, 2], "Invalid action_no"
        assert n_envs <= 10, "Too many environments"

        cc_file = "constant_config_pool.csv"
        # sc_file = "example_test2.csv"
        sc_file = "example_test.csv"

        # train_ppo()
        # use_case = "test"
        # use_case = "train"
        # use_case = "baseline"
        # use_case = "baseline_test"
        # action_no = 1
        if masked:
            warnings.filterwarnings("ignore", category=UserWarning)
            standard_ppo(use_case, action_no, model_name, iter, cc_file, sc_file, n_envs=n_envs,
                     start_config_i=start_config_i, masked=masked)
        else:
            standard_ppo(use_case, action_no, model_name, iter, cc_file, sc_file, n_envs=n_envs,
                         start_config_i=start_config_i)


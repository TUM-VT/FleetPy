import os.path
import sys

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.distributions import Categorical
# from src.fleetctrl.SoDZonalControlRL import SoDZonalControlRL
from FleetPy_gym import FleetPyEnv
# from src.fleetctrl.SoDZonalControlRL import FeedForwardNN
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
# from stable_baselines3.common.env_util import make_vec_env

def standard_ppo(use_case, action_no, model_name, iter, start_config_i=0):
    # Create the environment
    # env = make_vec_env(env_name, n_envs=1)
    # model_name = "ppo_fleetpy_singleaction_1000"
    # model_name = "ppo_fleetpy_multiaction_2000"
    # model_name = "ppo_fleetpy_baseline_1000"
    env = FleetPyEnv({"use_case": use_case, "action_no": action_no, "start_config_i": start_config_i})
    # model_name = "ppo_fleetpy_baseline_multiaction"
    model_dir = "./models/"
    log_dir = "./logs/"
    logger = configure(log_dir, ["stdout", "tensorboard"])

    # check if model exists
    model_path = os.path.join(model_dir, model_name+".zip")
    if os.path.exists(model_path):
        model = PPO.load(model_path, env)
        print("PPO Model loaded!")
    else:
        # Initialize the PPO agent
        model = PPO("MlpPolicy",
                    env, n_steps=180,
                    # batch_size=180,
                    gamma=0.98, verbose=1, tensorboard_log=log_dir)
        print("PPO Model initialized!")

    if use_case=="train" or use_case=="baseline":
        # Train the agent
        total_timesteps = 180*iter
        print(f"Training for {total_timesteps} timesteps")
        model.learn(total_timesteps=total_timesteps)

        # Save the model
        model.save(model_path)
        print("Model trained and saved!")


    # Evaluate the trained model
    # eval_env = gym.make(env_name)
    eval_env = FleetPyEnv({"use_case": use_case, "action_no": action_no, "start_config_i": 0})
    obs, _ = eval_env.reset()
    total_rewards = 0
    total_timesteps = 180*iter
    for itit in range(total_timesteps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, truncated, info = eval_env.step(action)
        total_rewards += rewards
        if dones:
            break
            # obs, _ = eval_env.reset()
        if itit % 100 == 0:
            print(f"Iteration: {itit} Cum. Reward: {total_rewards}")

    eval_env.close()
    print(f"Total Reward: {total_rewards}")


# def train_ppo(env_name='CartPole-v1', num_episodes=100, max_timesteps=1000):
#     env = gym.make(env_name)
#     in_dim = env.observation_space.shape[0]
#     out_dim = env.action_space.n
#
#     ppo = SoDZonalControlRL(in_dim, out_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99)
#
#     total_rewards = 0
#     for episode in range(num_episodes):
#         state = env.reset()[0]
#         done = False
#         timestep = 0
#
#         states, actions, rewards, old_log_probs, times = [], [], [], [], []
#
#         while not done and timestep < max_timesteps:
#             timestep += 1
#             action, log_prob, _ = ppo.select_action(state)
#             next_state, reward, done, _, _ = env.step(action)
#
#             states.append(state)
#             actions.append(action)
#             rewards.append(reward)
#             old_log_probs.append(log_prob)
#             times.append(timestep)
#
#             state = next_state
#
#             total_rewards += reward
#         ppo.ppo_update(states, actions, rewards, old_log_probs, times)
#
#         print(f"Episode {episode + 1}: Cum. Reward: {sum(rewards)}")
#
#     policy_losses, value_losses, discounted_rewards = ppo.return_losses()
#     # print(policy_losses)
#     # print(value_losses)
#     # print(f"Total Reward: {total_rewards}")
#     # print(discounted_rewards)
#     ppo.save_model("ppo_cartpole.pth")


# Main execution
if __name__ == "__main__":
    # read argument for use_case, action_no, and model_name

    if len(sys.argv) < 5:
        print("Usage: python RL_test.py <use_case> <action_no> <model_name> <iter>")
        sys.exit(1)
    use_case = sys.argv[1]
    action_no = int(sys.argv[2])
    model_name = sys.argv[3]
    iter = int(sys.argv[4])
    if len(sys.argv) == 6:
        start_config_i = int(sys.argv[5])
    else:
        start_config_i = 0
    assert use_case in ["train", "test", "baseline", "baseline_test"], "Invalid use_case"
    assert action_no in [1, 2], "Invalid action_no"

    # train_ppo()
    # use_case = "test"
    # use_case = "train"
    # use_case = "baseline"
    # use_case = "baseline_test"
    # action_no = 1
    standard_ppo(use_case, action_no, model_name, iter, start_config_i)

import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from torch.distributions import Categorical
# from src.fleetctrl.SoDZonalControlRL import SoDZonalControlRL
from FleetPy_gym import FleetPyEnv
# from src.fleetctrl.SoDZonalControlRL import FeedForwardNN
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def standard_ppo():
    # Create the environment
    # env = make_vec_env(env_name, n_envs=1)
    # model_name = "ppo_cartpole_baseline"
    env = FleetPyEnv({})
    model_name = "ppo_fleetpy_baseline"

    # check if model exists
    if os.path.exists(model_name+".zip"):
        model = PPO.load(model_name, env)
        print("PPO Model loaded!")
    else:
        # Initialize the PPO agent
        model = PPO("MlpPolicy", env, n_steps=180, verbose=1)

    # Train the agent
    # total_timesteps = (86400-75600)/60
    total_timesteps = 180*100
    print(f"Training for {total_timesteps} timesteps")
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save(model_name)
    print("Model trained and saved!")


    # Evaluate the trained model
    # eval_env = gym.make(env_name)
    eval_env = FleetPyEnv({})
    obs, _ = eval_env.reset()
    total_rewards = 0
    num_episodes = 180
    for itit in range(num_episodes):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, truncated, info = eval_env.step(action)
        total_rewards += rewards
        if dones:
            obs, _ = eval_env.reset()
        if itit % 100 == 0:
            print(f"Iteration: {itit} Cum. Reward: {total_rewards}")
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
    # train_ppo()
    standard_ppo()

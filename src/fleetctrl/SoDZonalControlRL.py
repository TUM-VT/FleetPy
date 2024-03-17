import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

# -------------------------------------------------------------------------------------------------------------------- #
# global variables
# ----------------
LOG = logging.getLogger(__name__)

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        # Actor
        # self.actor = nn.Sequential(
        #     nn.Linear(state_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, action_dim),
        #     nn.Softmax(dim=-1),
        # )
        # Actor
        self.actor_fc1 = nn.Linear(state_dim, 64)
        self.actor_fc2 = nn.Linear(64, action_dim)

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        # action_probs = self.actor(state)
        # Actor
        x = F.relu(self.actor_fc1(state))
        logits = self.actor_fc2(x)

        # Logit normalization
        logits = (logits - logits.mean(dim=1, keepdim=True)) / (logits.std(dim=1, keepdim=True) + 1e-7)
        action_probs = F.softmax(logits, dim=-1)

        state_values = self.critic(state)
        return action_probs, state_values

class SoDZonalControlRL:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, eps_clip=0.2, epochs=10):
        self.model = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs

        seed = 0
        torch.manual_seed(seed)

    def ppo_update(self, states, actions, rewards, old_log_probs, old_state_values):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)  # Convert rewards to float32
        old_log_probs = torch.stack([torch.tensor(log_prob, dtype=torch.float32) for log_prob in old_log_probs])
        old_state_values = torch.stack([torch.tensor(value, dtype=torch.float32) for value in old_state_values])

        # Discount rewards and normalize
        discounted_rewards = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        rewards_normalized = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        # Optimize model for epochs
        for _ in range(self.epochs):
            action_probs, state_values = self.model(states)
            LOG.debug(f"RL training action_probs: {action_probs} state_values: {state_values}")
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            advantages = rewards_normalized - old_state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (rewards_normalized - state_values).pow(2).mean()

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),
            #                                max_norm=1.0)  # max_norm is a hyperparameter you can tune
            self.optimizer.step()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, state_value = self.model(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        LOG.debug(f"RL action: {action} action_prob: {action_probs}")

        return action.item(), log_prob, state_value

    def load_model(self, path="model_checkpoint.pth"):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_model(self, path="model_checkpoint.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#         self.actor = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, action_dim),
#             # nn.Softmax(dim=-1),
#         )
#
#         # Assuming a shared standard deviation across action dimensions that doesn't depend on state
#         self.log_std = nn.Parameter(torch.zeros(action_dim))
#
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1),
#         )
#
#     def forward(self, state):
#         action_mean = self.actor(state)
#         state_value = self.critic(state)
#         return action_mean, state_value
#
#     def get_value(self, state):
#         """Return the estimated value of a given state using the critic network."""
#         with torch.no_grad():  # Ensure gradients are not computed for this operation
#             state_value = self.critic(state)
#         return state_value
#
# class SoDZonalControlRL:
#     def __init__(self, state_dim, action_dim):
#         """
#         :param state_dim: int: dimension of state space
#         :param action_dim: int: dimension of action space
#         """
#         self.policy = ActorCritic(state_dim, action_dim)
#         self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
#
#     def compute_returns(self, next_value, rewards, masks, gamma=0.99):
#         R = next_value
#         returns = []
#         for step in reversed(range(len(rewards))):
#             R = rewards[step] + gamma * R * masks[step]
#             returns.insert(0, R)
#         return returns
#
#     def compute_gae(self, rewards, states, masks, gamma=0.99, lambda_=0.95, normalize=True):
#         # Convert states to tensor for value estimation
#         torch_states = torch.tensor(states, dtype=torch.float)
#         # Estimate values for each state
#         _, values = self.policy(torch_states)
#         values = values.detach().numpy().flatten()  # Convert to numpy array and flatten
#         values = np.append(values, 0)  # Append 0 for the value of the terminal state
#
#         deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
#         advantages = np.zeros_like(rewards)
#         last_advantage = 0
#
#         for t in reversed(range(len(rewards) - 1)):
#             advantages[t] = last_advantage = deltas[t] + gamma * lambda_ * last_advantage
#
#         if normalize:
#             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
#
#         return advantages
#
#     def ppo_update(self, states, actions, log_probs_old, rewards, masks, clip_param=0.2) -> None:
#         torch_states = torch.tensor(states, dtype=torch.float)
#         # Compute advantages using the modified compute_gae which now estimates values
#         advantages = self.compute_gae(rewards, states, masks)
#         advantages = torch.tensor(advantages, dtype=torch.float)
#
#         # Estimate values again for use in value loss calculation
#         _, state_values = self.policy(torch_states)
#         returns = torch.tensor(self.compute_returns(state_values[-1], rewards, masks), dtype=torch.float).view(-1, 1)
#
#         actions = actions.detach()
#         log_probs_old = log_probs_old.detach()
#         returns = returns.detach()
#
#         action_mean, _ = self.policy(torch_states)  # Re-estimate to get updated action_probs
#         std = self.policy.log_std.exp()
#         dist = torch.distributions.Normal(action_mean, std)
#         log_probs = dist.log_prob(actions)
#         ratio = torch.exp(log_probs - log_probs_old)
#         surr1 = ratio * advantages
#         surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
#
#         policy_loss = -torch.min(surr1, surr2).mean()
#         value_loss = 0.5 * (returns - state_values).pow(2).mean()
#
#         self.optimizer.zero_grad()
#         (policy_loss + value_loss).backward()
#         self.optimizer.step()
#
#     def select_action(self, state_np):
#         state = torch.tensor(state_np, dtype=torch.float).unsqueeze(0)  # Add batch dimension
#         with torch.no_grad():
#             action_mean, _ = self.policy(state)
#         std = self.policy.log_std.exp()
#         # dist = torch.distributions.Normal(action_mean, std)
#         dist = torch.distributions.Normal(0, 1)
#         action = dist.sample()
#         log_prob = dist.log_prob(action).sum(axis=-1)  # Sum log probabilities for multi-dim action
#
#         # action_normal = (action.numpy()[0] - action_mean.detach().numpy()[0]) / std.detach().numpy()[0]
#         return action.numpy()[0], log_prob.detach()
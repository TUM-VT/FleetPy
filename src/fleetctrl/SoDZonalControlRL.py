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


class FeedForwardNN(nn.Module):
    """
        A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """
    def __init__(self, in_dim, out_dim, softmax=False):
        """
            Initialize the network and set up the layers.

            Parameters:
                in_dim - input dimensions as an int
                out_dim - output dimensions as an int

            Return:
                None
        """
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer1b = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, out_dim)

        self.softmax = softmax

    def forward(self, obs):
        """
            Runs a forward pass on the neural network.

            Parameters:
                obs - observation to pass as input

            Return:
                output - the output of our forward pass
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation1 = F.relu(self.layer1b(activation1))
        output = self.layer2(activation1)

        if self.softmax:
            logits = (output - output.mean(dim=1, keepdim=True)) / (output.std(dim=1, keepdim=True) + 1e-7)
            return F.softmax(logits, dim=-1)
        else:
            return output


# class ActorCriticNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCriticNetwork, self).__init__()
#         # Actor
#         # self.actor = nn.Sequential(
#         #     nn.Linear(state_dim, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, action_dim),
#         #     nn.Softmax(dim=-1),
#         # )
#         # Actor
#         self.actor_fc1 = nn.Linear(state_dim, 64)
#         self.actor_fc2 = nn.Linear(64, action_dim)
#
#         # Critic
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#         )
#
#     def forward(self, state):
#         # action_probs = self.actor(state)
#         # Actor
#         x = F.relu(self.actor_fc1(state))
#         logits = self.actor_fc2(x)
#
#         # Logit normalization
#         logits = (logits - logits.mean(dim=1, keepdim=True)) / (logits.std(dim=1, keepdim=True) + 1e-7)
#         action_probs = F.softmax(logits, dim=-1)
#
#         state_values = self.critic(state)
#         return action_probs, state_values


class SoDZonalControlRL:
    def __init__(self, state_dim, action_dim, lr_actor=5e-3, lr_critic=5e-3, gamma=0.98, eps_clip=0.2, epochs=10, seed = None):
        if seed is not None:
            torch.manual_seed(seed)
        # torch.device("mps")  # apple m1

        # self.model = ActorCriticNetwork(state_dim, action_dim)
        self.actor = FeedForwardNN(state_dim, action_dim, softmax=True)
        self.critic = FeedForwardNN(state_dim, 1)
        # self.model = nn.DataParallel(self.model)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs

        self.policy_losses = []
        self.value_losses = []
        self.discounted_rewards = []

    def ppo_update(self, states, actions, rewards, old_log_probs, times):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)  # Convert rewards to float32
        old_log_probs = torch.stack([torch.tensor(log_prob, dtype=torch.float32) for log_prob in old_log_probs])
        # old_state_values = torch.stack([torch.tensor(value, dtype=torch.float32) for value in old_state_values])

        # Discount rewards (Generalized Advantage Estimation GAE)
        discounted_rewards = []
        last_time = times[-1]
        R = 0
        for reward, timestep in zip(reversed(rewards), reversed(times)):
            R = reward + self.gamma ** (last_time - timestep) * R
            last_time = timestep
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        self.discounted_rewards.append(np.average(discounted_rewards.numpy()))
        # state_values = self.critic(states)

        # rewards_normalized = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        # Optimize model for epochs
        for _ in range(self.epochs):
            # action_probs, state_values = self.model(states)
            action_probs, state_values = self.actor(states), self.critic(states)
            LOG.debug(f"RL training action_probs: {action_probs} state_values: {state_values}")
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            advantages = discounted_rewards - state_values.detach()
            A_k = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

            # advantages = rewards_normalized - old_state_values.detach()
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * A_k
            policy_loss = -(torch.min(surr1, surr2)).mean()
            value_loss = nn.MSELoss()(state_values, discounted_rewards.unsqueeze(1))
            self.policy_losses.append(policy_loss.item())
            self.value_losses.append(value_loss.item())

            # loss = policy_loss + value_loss

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),
            #                                max_norm=1.0)  # max_norm is a hyperparameter you can tune
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state)
            state_value = self.critic(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        LOG.debug(f"RL action: {action} action_prob: {action_probs}")

        return action.item(), log_prob, state_value

    def load_model(self, path="model_checkpoint.pth"):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_model_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    def save_model(self, path="model_checkpoint.pth"):
        torch.save({
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def return_losses(self):
        return self.policy_losses, self.value_losses, self.discounted_rewards

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

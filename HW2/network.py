import math
import os.path

import torch
import torch.nn as nn
import numpy as np

"""
We need a critic and an actor network.

Critic networks: 
Evaluate how good an action is by estimating the Q-values (action-value function)
 for given state-action pairs. These networks guide the actor toward higher rewards.
 
Actor networks: 
Output a stochastic policy, 
which samples actions from a probability distribution
balancing exploration and exploitation.
"""

# from matplotlib import pyplot as plt

torch.seed = 9999
np.random.seed(9999)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.value.to(device)



    def forward(self, state):
        state.to(device)
        # Only takes state as input, not action
        return self.value(state)



class PolicyNetwork(nn.Module):


    def __init__(self, state_dim, action_dim, action_std=0.5):
        """
        Initialize the policy network for continuous action spaces.

        Args:
            state_dim: Dimension of the state space (input)
            action_dim: Dimension of the action space (output)
            action_std: Initial standard deviation for exploration (default: 0.5)

        Creates a neural network with two hidden layers of 256 units each,
        and initializes a learnable log standard deviation parameter for
        stochastic action selection.
        """
        super(PolicyNetwork, self).__init__()

        # Neural network for predicting action means
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # Learnable parameter for standard deviation
        # Initialized with log of action_std for numerical stability
        self.log_std = nn.Parameter(torch.ones(action_dim) * math.log(action_std))

        # Action range for clipping (typical for continuous control tasks)
        self.action_range = 1.0


    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Environment state tensor

        Returns:
            mean: Mean of the action distribution, constrained to [-1, 1]
            std: Standard deviation of the action distribution

        Moves the state to the correct device and applies the neural network
        to get the mean action values, then constrains them with tanh.
        The standard deviation is computed by exponentiating the learned log_std.
        """
        state.to(device)
        # Apply tanh to constrain mean to [-1, 1] range
        mean = torch.tanh(self.actor(state))
        std = self.log_std.exp()
        return mean, std


    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy distribution.

        Args:
            state: Environment state tensor
            deterministic: If True, return the mean action without sampling (default: False)

        Returns:
            action: Action tensor, either sampled from distribution or mean if deterministic

        If deterministic is True, returns the mean action directly.
        Otherwise, samples from a Normal distribution with the predicted mean and std,
        then clips the action to ensure it stays within the valid range.
        """
        mean, std = self.forward(state)

        if deterministic:
            return mean

        # Create a Normal distribution with the predicted mean and std
        dist = torch.distributions.Normal(mean, std)
        # Sample an action from the distribution
        action = dist.sample()

        # Clip actions to ensure they stay within [-1, 1] range
        action = torch.clamp(action, -self.action_range, self.action_range)
        return action


    def evaluate(self, state, action):
        """
        Evaluate the log probability and entropy of actions.

        Args:
            state: Environment state tensor
            action: Action tensor to evaluate

        Returns:
            log_prob: Log probability of the action under current policy (for each batch element)
            entropy: Entropy of the policy distribution (for each batch element)

        Used during training to compute policy loss and entropy bonus.
        Higher log probability means the action is more likely under current policy.
        Higher entropy encourages exploration by making the policy distribution more uniform.
        """
        mean, std = self.forward(state)
        # Create a Normal distribution with the predicted mean and std
        dist = torch.distributions.Normal(mean, std)
        # Calculate log probability of the action
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        # Calculate entropy of the distribution
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy




import math
import os

import torch.optim as optim

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
    def __init__(self, state_dim, lr=3e-4):
        super(ValueNetwork, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.value = self.value.to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)



    def forward(self, state):
        state = state.to(device)
        # Only takes state as input, not action
        return self.value(state).cpu()

    def get_loss(self, trajectory_tensor):
        """
        generate MSE for value network
        :param trajectory_tensor: contains batch data sampled from the generated data
        :param epilson: small value for clipping
        :return:
        """

        cur_state_tensor = trajectory_tensor['states'].detach().clone()
        reward_to_go_tensor = trajectory_tensor['returns'].detach().clone()

        value_tensor = self.forward(cur_state_tensor)

        # Calculate mean squared error between predicted values and actual returns
        value_loss = torch.mean((value_tensor - reward_to_go_tensor) ** 2)

        return value_loss

    def save_weights(self, filepath="./Value_nn_weight", timestep=None):
        """
        Save the model weights to a file
        :param filepath: path to save the model weights
        """

        if timestep is not None:
            filepath = f"{filepath}_{timestep}.pth"
        else:
            filepath = f"{filepath}.pth"

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        """
        Load the model weights from a file
        :param filepath: path to load the model weights from
        """
        if not os.path.exists(filepath):
            print(f"No weights file found at {filepath}")
            return False

        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model weights loaded from {filepath}")
        return True


class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, action_std=0.5, lr=3e-4):
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

        # Initialize optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=lr)



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

    def get_action_probability(self, state, action):
        """
        Calculate the probability density of taking a specific action given the state.

        Args:
            state: Environment state tensor
            action: The action tensor for which to compute probability

        Returns:
            prob: The probability density of the action
        """
        # Get mean and std from forward pass
        mean, std = self.forward(state)

        # Create a Normal distribution
        dist = torch.distributions.Normal(mean, std)

        # Get log probability density
        log_prob = dist.log_prob(action)

        # Sum across only the last dimension to get joint probabilities
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1)  # Use -1 to always refer to the last dimension

        # Convert from log probability to probability
        prob = torch.exp(log_prob)

        return prob


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

        # Create a Normal distribution
        normal = torch.distributions.Normal(mean, std)

        # Get log probability density
        log_prob = normal.log_prob(action)

        # Sum across only the last dimension to get joint probabilities
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1)  # Use -1 to always refer to the last dimension

        # Convert from log probability to probability
        prob = torch.exp(log_prob)

        return action, prob

    def get_loss(self, trajectory_tensor, epilson=0.2):
        """
        :param trajectory_tensor: contains batch data sampled from the generated data
        :param epilson: small value for clipping
        :return:
        """

        cur_state_tensor = trajectory_tensor['states'].detach().clone()
        action_tensor = trajectory_tensor['actions'].detach().clone()
        old_probs_tensor = trajectory_tensor['old_probs'].detach().clone()
        advantage_tensor = trajectory_tensor['advantages'].detach().clone()


        # find the new probability of outputing current action given state
        new_prob_tensor = self.get_action_probability(cur_state_tensor, action_tensor)
        prob_ratio = old_probs_tensor / new_prob_tensor

        clipped_prob_ratio = torch.clamp(prob_ratio, min=1-epilson, max=1+epilson)

        # Calculate the surrogate objectives
        surrogate1 = prob_ratio * advantage_tensor
        surrogate2 = clipped_prob_ratio * advantage_tensor

        # Take the minimum of the two surrogate objectives
        # Use negative because we want to maximize the objective, but optimizers minimize
        policy_loss_tensor = -torch.min(surrogate1, surrogate2)

        # Average over all timesteps and trajectories
        policy_loss = policy_loss_tensor.mean()

        return policy_loss

    def save_weights(self, filepath="./Policy_nn_weight", timestep=None):
        """
        Save the model weights to a file
        :param filepath: path to save the model weights
        """

        if timestep is not None:
            filepath = f"{filepath}_{timestep}.pth"
        else:
            filepath = f"{filepath}.pth"

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        """
        Load the model weights from a file
        :param filepath: path to load the model weights from
        """
        if not os.path.exists(filepath):
            print(f"No weights file found at {filepath}")
            return False

        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model weights loaded from {filepath}")
        return True

    # def evaluate(self, state, action):
    #     """
    #     Evaluate the log probability and entropy of actions.
    #
    #     Args:
    #         state: Environment state tensor
    #         action: Action tensor to evaluate
    #
    #     Returns:
    #         log_prob: Log probability of the action under current policy (for each batch element)
    #         entropy: Entropy of the policy distribution (for each batch element)
    #
    #     Used during training to compute policy loss and entropy bonus.
    #     Higher log probability means the action is more likely under current policy.
    #     Higher entropy encourages exploration by making the policy distribution more uniform.
    #     """
    #     mean, std = self.forward(state)
    #     # Create a Normal distribution with the predicted mean and std
    #     dist = torch.distributions.Normal(mean, std)
    #     # Calculate log probability of the action
    #     log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
    #     # Calculate entropy of the distribution
    #     entropy = dist.entropy().sum(dim=-1, keepdim=True)
    #     return log_prob, entropy




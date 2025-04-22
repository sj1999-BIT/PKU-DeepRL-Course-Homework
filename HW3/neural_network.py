import math
import os

import gym
import torch.optim as optim

import torch
import torch.nn as nn
import numpy as np

from dataKeys import STATES, NEXT_STATES, ACTIONS, REWARD, LOG_PROB
from data import append_values_to_file
from tqdm import tqdm

"""
We need a neural network that models the environment, 
predicts the next state given the action and current state.
"""

# from matplotlib import pyplot as plt

torch.seed = 9999
np.random.seed(9999)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nn_learn(net: nn.Module, sample, filepath="./data.txt"):
    """
    Standard function for all network to generate loss and learn
    :param filename: store the data to this file
    :param net:
    :param sample:
    :param path: store the weights
    :return:
    """
    loss = net.get_loss(sample)
    # backpropagation
    net.optimizer.zero_grad()
    loss.backward()
    net.optimizer.step()

    # save the data
    append_values_to_file(loss.detach().item(), filepath)
    net.save_weights()


class DynamicsNetwork(nn.Module):

    def __init__(self, env: gym.Env, model_name=None, lr=3e-4):
        """
        Initialize the policy network for continuous action spaces.
        Creates a neural network with two hidden layers of 256 units each
        """
        super(DynamicsNetwork, self).__init__()

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Neural network for predicting next state given current state and action
        self.actor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        ).to(device)

        # Action range for clipping (typical for continuous control tasks)
        self.action_range = 1.0

        # Initialize optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # we can save the weights using the name
        self.name = model_name

    def forward(self, combined_input):
        """
        Forward pass through the network.

        Args:
            combined_input: Environment state tensor + action tensor

        Returns:
            next_state:Environment state tensor
        """
        # Move tensors to the correct device
        combined_input = combined_input.to(device)

        # Apply network to predict next state
        next_state = self.actor(combined_input)

        return next_state

    def get_next_state(self, cur_state, action):
        """
        :param cur_state: numpy array
        :param action: : numpy array
        :return: next state
        """

        if isinstance(cur_state, np.ndarray):
            cur_state = torch.tensor(cur_state, dtype=torch.float32)

        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)

        if len(cur_state.shape) == 1:
            state_action_tensor = torch.cat([cur_state, action], dim=0)

        if len(cur_state.shape) == 2:
            state_action_tensor = torch.cat([cur_state, action], dim=1)

        if len(cur_state.shape) > 2:
            raise Exception(f"invalid tensor size for cur_state {cur_state}")
            pass

        return self.forward(state_action_tensor).detach().cpu().numpy()

    def get_loss(self, sample_data):
        """
        generate MSE loss function
        :param data: sample data is a dict contains states, next_states, reward, action
        :return: MSE of the model output next state and ground truth
        """

        states_tensor = sample_data[STATES]
        actions_tensor = sample_data[ACTIONS]

        state_action_tensor = torch.cat([states_tensor, actions_tensor], dim=1)

        next_state_tensor = sample_data[NEXT_STATES]

        # Get model predictions
        pred_tensor = self.forward(state_action_tensor)

        # Calculate MSE loss using PyTorch's built-in function
        loss = torch.nn.functional.mse_loss(pred_tensor, next_state_tensor.to(device))

        return loss

    def save_weights(self, filepath="./dynamics_nn_weight", timestep=None):
        """
        Save the model weights to a file
        :param filepath: path to save the model weights
        """

        if self.name is not None:
            filepath = f"{filepath}_{self.name}.pth"
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


class ValueNetwork(nn.Module):
    """
    Using a valueNetwork to simulate the
    """

    def __init__(self, env, lr=3e-4):
        super(ValueNetwork, self).__init__()

        state_dim = env.observation_space.shape[0]

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

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

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

        cur_state_tensor = trajectory_tensor[STATES].detach().clone()
        reward_to_go_tensor = trajectory_tensor[REWARD].detach().clone()

        value_tensor = self.forward(cur_state_tensor)

        # Calculate mean squared error between predicted values and actual returns
        value_loss = torch.mean((value_tensor - reward_to_go_tensor) ** 2)

        return value_loss

    def save_weights(self, filepath="./Value_nn_weight", timestep=None):
        """
        Save the model weights to a file
        :param filepath: path to save the model weights
        """

        # create a data folder
        if not os.path.exists("./data"):
            os.makedirs("./data")

        if timestep is not None:
            filepath = f"./data/{filepath}_{timestep}.pth"
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

    def __init__(self, env, action_std=0.5, lr=3e-4):
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

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

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

    def get_entropy(self):
        """
        Calculate the entropy of the policy distribution.

        Args:
            mean: Mean of the action distribution
            log_std: Log standard deviation of the action distribution

        Returns:
            Entropy of the policy distribution
        """
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + self.log_std
        return entropy.sum(dim=-1)

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

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

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

        return log_prob

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

        return action, log_prob

    def get_loss(self, training_batch, epilson=0.2, entropy_coef=0.5):

        cur_state_tensor = training_batch[STATES].detach().clone()
        best_action_tensor = training_batch[ACTIONS].detach().clone()
        old_log_probs_tensor = training_batch[LOG_PROB].detach().clone()
        reward_tensor = training_batch[REWARD].detach().clone()

        # find the new probability of outputing best action given state
        new_log_prob_tensor = self.get_action_probability(cur_state_tensor, best_action_tensor)
        prob_ratio = torch.exp(new_log_prob_tensor - old_log_probs_tensor)

        clipped_prob_ratio = torch.clamp(prob_ratio, min=1 - epilson, max=1 + epilson)

        # Calculate the surrogate objectives
        surrogate1 = prob_ratio * reward_tensor
        surrogate2 = clipped_prob_ratio * reward_tensor

        # Take the minimum of the two surrogate objectives
        # Use negative because we want to maximize the objective, but optimizers minimize
        policy_loss_tensor = -torch.min(surrogate1, surrogate2)
        policy_loss = policy_loss_tensor.mean()

        # get what current policy outputs
        cur_action_tensor, _ = self.get_action(cur_state_tensor)

        # add mse loss between what action we pick now vs the best action
        policy_loss += 0.5 * torch.nn.functional.smooth_l1_loss(cur_action_tensor, best_action_tensor)

        # Average over all timesteps and trajectories
        # addin entropy to promote more exploration

        entropy_val = self.get_entropy()

        policy_loss += entropy_coef * entropy_val

        return policy_loss

    def save_weights(self, filepath="./Policy_nn_weight"):
        """
        Save the model weights to a file
        :param filepath: path to save the model weights
        """

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

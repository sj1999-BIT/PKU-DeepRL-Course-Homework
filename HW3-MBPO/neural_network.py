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
For MBPO, the dynamic network outputs a state and a reward guassian distribution
similar to policy network, we need to output a mean and sample from it
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


class PredictiveNetwork(nn.Module):

    def __init__(self, env: gym.Env, model_name=None, next_state_std=0.5, reward_std=0.5, lr=3e-4):
        """
        Initialize the policy network for continuous action spaces.
        Creates a neural network with two hidden layers of 256 units each
        """
        super(PredictiveNetwork, self).__init__()

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.state_dim = state_dim

        # Neural network for predicting next state given current state and action
        self.actor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim+1)
        ).to(device)

        self.log_next_state_std = nn.Parameter(torch.ones(state_dim) * math.log(next_state_std)).to(device)

        self.log_reward_std = nn.Parameter(torch.ones(1) * math.log(reward_std)).to(device)

        # Action range for clipping (typical for continuous control tasks)
        self.action_range = 1.0

        # Initialize optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # we can save the weights using the name
        self.name = model_name

    def forward(self, combined_input):
        """
            Forward pass through the network to predict next state and reward distributions.

            Args:
                combined_input: Concatenated state and action tensors [batch_size, state_dim + action_dim]

            Returns:
                next_state: Sampled next state from the predicted distribution
                reward: Sampled reward from the predicted distribution
                next_state_mean: Mean of next state distribution
                next_state_std: Standard deviation of next state distribution
                reward_mean: Mean of reward distribution
                reward_std: Standard deviation of reward distribution
            """
        # Move tensors to the correct device
        combined_input = combined_input.to(device)

        # Apply network to predict next state and reward
        output = self.actor(combined_input)

        if len(output.shape) == 1:
            next_state_mean = output[:-1]  # All columns except the last one
            reward_mean = output[-1].unsqueeze(0)  # Last column, reshape to [batch_size, 1]

        # for batch
        if len(output.shape) == 2:
            # Split the output into next_state and reward
            next_state_mean = output[:, :-1]  # All columns except the last one
            reward_mean = output[:, -1].unsqueeze(1)  # Last column, reshape to [batch_size, 1]

        next_state_std = self.log_next_state_std.exp()
        reward_std = self.log_reward_std.exp()

        next_state_distribution = torch.distributions.Normal(next_state_mean, next_state_std)
        # Sample an action from the distribution
        next_state = next_state_distribution.rsample()

        reward_distribution = torch.distributions.Normal(reward_mean, reward_std)
        # Sample an action from the distribution
        reward = reward_distribution.rsample()

        return next_state, reward, next_state_distribution, reward_distribution

    def get_next_state_and_reward(self, cur_state, action):
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

        if len(cur_state.shape) > 2 or len(cur_state.shape) < 1:
            raise Exception(f"invalid tensor size for cur_state {cur_state}")

        next_state, reward, next_state_distribution, reward_distribution = self.forward(state_action_tensor)

        return next_state, reward, next_state_distribution, reward_distribution

    def get_loss(self, sample_data):
        """
        Generate negative log-likelihood loss for probabilistic dynamics model
        :param sample_data: sample data is a dict containing states, next_states, rewards, actions
        :return: NLL loss for the predicted Gaussian distributions
        """
        # Extract input tensors
        states_tensor = sample_data[STATES]
        actions_tensor = sample_data[ACTIONS]

        # Concatenate states and actions as input
        state_action_tensor = torch.cat([states_tensor.to(device), actions_tensor], dim=1)

        # Extract ground truth tensors, pass to same device for calculation
        next_states_tensor = sample_data[NEXT_STATES].to(device)
        rewards_tensor = sample_data[REWARD].to(device)

        output = self.actor(state_action_tensor.to(device))

        # Get model predictions
        next_state_mean = output[:, :self.state_dim]
        reward_mean = output[:, -1].unsqueeze(1)

        next_state_var = self.log_next_state_std.exp().pow(2).expand_as(next_state_mean).to(device)
        reward_var = self.log_reward_std.exp().pow(2).expand_as(reward_mean).to(device)

        # Calculate negative log-likelihood loss
        state_loss = 0.5 * torch.mean(
            torch.log(next_state_var) +
            torch.pow(next_states_tensor - next_state_mean, 2) / next_state_var
        )

        reward_loss = 0.5 * torch.mean(
            torch.log(reward_var) +
            torch.pow(rewards_tensor - reward_mean, 2) / reward_var
        )

        # Sum the losses
        total_loss = state_loss + reward_loss

        return total_loss

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

    def __init__(self, env, lr=3e-4):
        super(ValueNetwork, self).__init__()

        state_dim = env.observation_space.shape[0]

        self.value = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)

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

        cur_state_tensor = trajectory_tensor['states'].clone()
        reward_to_go_tensor = trajectory_tensor['returns'].clone()

        value_tensor = self.forward(cur_state_tensor).squeeze(1).to(device)

        # Calculate mean squared error between predicted values and actual returns
        value_loss = torch.mean((value_tensor - reward_to_go_tensor) ** 2)

        return value_loss

    def save_weights(self, filepath="Value_nn_weight", timestep=None):
        """
        Save the model weights to a file
        :param filepath: path to save the model weights
        """

        # # create a data folder
        # if not os.path.exists("./data"):
        #     os.makedirs("./data")

        if timestep is not None:
            filepath = f"./{filepath}_{timestep}.pth"
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
        ).to(device)

        # Learnable parameter for standard deviation
        # Initialized with log of action_std for numerical stability
        self.log_std = nn.Parameter(torch.ones(action_dim) * math.log(action_std)).to(device)

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

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        # Apply tanh to constrain mean to [-1, 1] range
        mean = torch.tanh(self.actor(state.to(device)))
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

    def get_loss(self, trajectory_tensor, epilson=0.2):
        """
            Calculate the PPO (Proximal Policy Optimization) policy loss using clipped surrogate objective.

            This implements the PPO clipped surrogate objective function that prevents too large
            policy updates by clipping the probability ratio between old and new policies.

            Args:
                trajectory_tensor (dict): Dictionary containing batched trajectory data with keys:
                    - 'states': Tensor of environment states
                    - 'actions': Tensor of actions taken in those states
                    - 'old_log_probs': Tensor of log probabilities from when actions were sampled
                    - 'advantages': Tensor of calculated advantage estimates
                epsilon (float): Clipping parameter that limits how much the policy can change
                                 in a single update (default: 0.2)

            Returns:
                torch.Tensor: Scalar policy loss value to be minimized

            Note:
                The function computes the negative of the minimum between the standard objective
                and the clipped objective, which makes it suitable for minimization with
                gradient descent optimizers.
            """

        cur_state_tensor = trajectory_tensor['states'].detach().clone()
        action_tensor = trajectory_tensor['actions'].detach().clone()
        old_log_probs_tensor = trajectory_tensor['old_log_probs'].detach().clone()
        advantage_tensor = trajectory_tensor['advantages'].detach().clone()


        # find the new probability of outputing current action given state
        new_log_prob_tensor = self.get_action_probability(cur_state_tensor, action_tensor)
        prob_ratio = torch.exp(new_log_prob_tensor - old_log_probs_tensor)

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

        # create a data folder
        # if not os.path.exists("./data"):
        #     os.makedirs("./data")

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

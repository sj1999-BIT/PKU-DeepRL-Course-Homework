import math
import os

import torch.optim as optim

import torch
import torch.nn as nn
import numpy as np

"""
Reinforcement Learning Network Architectures
-------------------------------------------
This module implements neural network architectures for Policy Gradient methods
in continuous action space environments.

The module provides two main components:

1. ValueNetwork (Critic): 
   Evaluates state values to help with advantage estimation. The value network
   estimates the expected return (cumulative reward) from a given state, which
   is used to compute advantages for policy updates.

2. PolicyNetwork (Actor):
   Generates actions by outputting a stochastic policy that defines a probability 
   distribution over the action space. This network balances exploration and 
   exploitation by sampling actions from the learned distribution.

The implementation supports:
- Continuous action spaces using normal distributions
- PPO-style policy optimization with clipped objectives
- Saving and loading of model weights
- GPU acceleration when available

These networks are designed to work together in actor-critic reinforcement
learning algorithms: Proximal Policy Optimization (PPO).
"""
torch.seed = 9999
np.random.seed(9999)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def save_weights(self, filepath="./data/Value_nn_weight", timestep=None):
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

    def save_weights(self, filepath="./data/Policy_nn_weight", timestep=None):
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





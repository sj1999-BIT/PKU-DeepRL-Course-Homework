import math
import os
import random

import gym
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from dataKeys import STATES, NEXT_STATES, ACTIONS, REWARD, LOG_PROB
from data import append_values_to_file
from tqdm import tqdm

"""
QMIX requires temporal relationships.

This means it can only receive states individually, cannot be in batch!
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


class AgentNetwork(nn.Module):

    def __init__(self, agent_id, obs_dim=18, action_dim=5, lr=3e-4, hidden_dim=128):
        """
        Agent Network:
        - Input: Individual observation (18-dim) + previous action (one-hot encoded)
        - Hidden Layer 1: Dense layer with ReLU activation
        - GRU layer to maintain action-observation history
        - Hidden Layer 2: Dense layer with ReLU activation
        - Output: Q-values for 5 actions
        """
        super(AgentNetwork, self).__init__()

        # Input dimensions: observation + one-hot encoded previous action
        self.input_dim = obs_dim + 1
        self.action_dim = action_dim

        # First fully connected layer
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)

        # GRU layer for maintaining action-observation history
        self.gru = nn.GRU(hidden_dim, hidden_dim)

        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer: Q-values for each action
        self.q_values = nn.Linear(hidden_dim, action_dim)

        # Initialize optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # we can save the weights using the name
        self.name = f"agent_{agent_id}"

        # maintain a hidden layer
        self.hidden = None

        # move model to cuda
        self.to(device)

    def re_initiate_hidden_layer(self):
        self.hidden = None

    def forward(self, obs, prev_action):
        """
        Forward pass of the agent network

        Args:
            obs: Agent's observation (obs_dim)
            prev_action: One-hot encoded previous action (action_dim)

        Returns:
            q_values: Q-values for each action
            new_hidden: New hidden state
        """

        # convert to tensor for usage
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        if not isinstance(prev_action, torch.Tensor):
            prev_action = torch.tensor(prev_action, dtype=torch.float32)

        # move the inputs to same device
        # Concatenate observation and previous action alone the last dim
        x = torch.cat([obs, prev_action], dim=-1).to(device)

        # First layer
        x = F.relu(self.fc1(x))

        # unsqueeze each input to convert to length sequence
        x = x.unsqueeze(-2)

        # GRU layer
        x, hidden_layer = self.gru(x, self.hidden)

        self.hidden = hidden_layer

        x = x.squeeze(-2)

        # Second layer
        x = F.relu(self.fc2(x))

        # Output Q-values

        q_values = self.q_values(x).squeeze()

        return q_values

    def select_action(self, obs, prev_action, epsilon=0.1):
        """
        Select action using epsilon-greedy policy

        Args:
            obs: Agent's observation
            prev_action: One-hot encoded previous action
            epsilon: Exploration rate (probability of selecting a random action)

        Returns:
            action: Selected action index
            Q-value: q_value of the selected action
        """
        # Get Q-values from forward pass
        with torch.no_grad():
            q_values = self.forward(obs, prev_action)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Exploration: select random action
            action = random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: select action with highest Q-value
            action = torch.argmax(q_values).item()

        return action, q_values.cpu().numpy()[action]

    def save_weights(self, filepath=f"./nn_weight", timestep=None):
        """
        Save the model weights to a file
        :param filepath: path to save the model weights
        """

        if self.name is not None:
            filepath = f"{self.name}_{filepath}.pth"
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


class PositiveWeightGenerator(nn.Module):
    def __init__(self, input_dim=54, hidden_dim=32):
        super(PositiveWeightGenerator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear layer that takes input_dim and outputs hidden_dim
        self.weight_generator = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # Apply absolute value function to ensure outputs are positive
        positive_weights = torch.abs(self.weight_generator(x))
        return positive_weights


class MixNetwork(nn.Module):

    # define a layer that gives non negative weights and unristricted bias
    class NonNegLinear(nn.Linear):
        def forward(self, input):
            return F.linear(input, self.weight.clamp(min=0.), self.bias)

    def __init__(self, lr=3e-4, hidden_dim=256):
        """
        Mixing Network:
        - Input: Individual agent Q-values dim: (3)

        - first hyperparameter layer: takes in global state dim： (54), generate first hidden layer weights (absolute)

        - first hidden layer: takes in 3 input values, outputs hidden_dim

        - output: hidden_dim

        - concate with unmodified global_state, goes through a leaky_relu

        - second hyperparameter layer: takes in global state dim： (54), generate second hidden layer weights (absolute)

        - second hidden layer: takes in hidden_dim + global_state, outputs a single values

        - output: dim 1

        - third hyperparameter layer: takes in global state dim： (54), perform RELU, pass into a final layer to generate
          a single value

        - addition with the output

        -output: 1

        - Hidden Layer: Dense layer with positive weights
        - Output: Single joint Q-value
        """
        super(MixNetwork, self).__init__()

        # Input: Q-values from 3 individual agent networks
        self.input_dim = 3
        # Output: a single Q-value generated from 3 Q-values
        self.output_dim = 1
        # perform addition of global state at each layer, concate 3 * 18 agent observations together
        self.global_state_dim = 18 * 3

        # First fully connected layer
        self.fc1 = self.PositiveWeightGenerator(self.global_state_dim, hidden_dim)

        # Second fully connected layer
        self.fc2 = self.NonNegLinear(hidden_dim + self.global_state_dim, 1)

        # Hypernetwork for final bias
        # takes in the final global state to generate the last addition to the !_total
        self.fc2_bias = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize optimizers
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # we can save the weights using the name
        self.name = f"mixNetwork"

        # move model to cuda
        self.to(device)

    def forward(self, agent_q_values, global_state):
        """
        Forward pass of the mixing network

        Args:
            agent_q_values (Tensor): Individual Q-values from each agent [batch_size, n_agents]
            global_state (Tensor): Global state information [batch_size, state_dim]

        Returns:
            Tensor: Joint Q-value (Qtot)
        """

        # convert to tensor for usage
        if not isinstance(agent_q_values, torch.Tensor):
            agent_q_values = torch.tensor(agent_q_values, dtype=torch.float32).to(device)

        # convert to tensor for usage
        if not isinstance(global_state, torch.Tensor):
            global_state = torch.tensor(global_state, dtype=torch.float32).to(device)

        # Concatenate agent Q-values with global state
        inputs = torch.cat([agent_q_values, global_state], dim=-1).to(device)

        # First layer with ELU activation (as specified in the paper)
        x = F.elu(self.fc1(inputs))

        # Concatenate hidden layer output with global state again
        x = torch.cat([x, global_state], dim=-1).to(device)

        # Second layer (output layer)
        q_tot = self.fc2(x)

        # Add the final bias from global state
        fc2_bias = self.fc2_bias(global_state)
        q_tot += fc2_bias

        return q_tot


    def select_action(self, obs, prev_action, epsilon=0.1):
        """
        Select action using epsilon-greedy policy

        Args:
            obs: Agent's observation
            prev_action: One-hot encoded previous action
            epsilon: Exploration rate (probability of selecting a random action)

        Returns:
            action: Selected action index
            Q-value: q_value of the selected action
        """
        # Get Q-values from forward pass
        with torch.no_grad():
            q_values = self.forward(obs, prev_action)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Exploration: select random action
            action = random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: select action with highest Q-value
            action = torch.argmax(q_values).item()

        return action, q_values.cpu().numpy()[action]

    def save_weights(self, filepath=f"./nn_weight", timestep=None):
        """
        Save the model weights to a file
        :param filepath: path to save the model weights
        """

        if self.name is not None:
            filepath = f"{self.name}_{filepath}.pth"
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


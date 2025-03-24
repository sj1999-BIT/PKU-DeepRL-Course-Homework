'''
Implement the DQN agent here
'''
import math
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
from visual import plot_value_distribution

import numpy as np
import numpy as np
from PIL import Image
import cv2

# from matplotlib import pyplot as plt

torch.seed = 9999
np.random.seed(9999)


print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")




def preprocess_state(state):
    """
    Preprocess state for DQN based on input shape.

    Args:
        state: Input state with shape 4 * x, where x can be:
            - (128,) - 1D vector
            - (210, 160) - 2D grayscale image
            - (210, 160, 3) - 2D RGB image

    Returns:
        Processed state with shape (4, 84, 84)
    """
    # Determine shape and type
    shape = state.shape

    # Process based on detected shape
    if len(shape) == 2 and shape[1] == 128:
        # 1D vector input (4, 128)
        processed_frames = []
        for i in range(shape[0]):
            # Reshape 1D vector to 2D (8, 16)
            frame = state[i].reshape(8, 16)
            # Resize to (84, 84)
            frame = cv2.resize(frame, (168, 168), interpolation=cv2.INTER_AREA)

            processed_frames.append(frame)

    elif len(shape) == 3 and shape[1:] == (210, 160):
        # 2D grayscale input (4, 210, 160)
        processed_frames = []
        for i in range(shape[0]):
            # Resize to (84, 84)
            frame = cv2.resize(state[i], (168, 168), interpolation=cv2.INTER_AREA)
            processed_frames.append(frame)

    elif len(shape) == 4 and shape[1:] == (210, 160, 3):
        # RGB input (4, 210, 160, 3)
        processed_frames = []
        for i in range(shape[0]):
            # Resize to (84, 84) while keeping RGB channels
            frame = cv2.resize(state[i], (168, 168), interpolation=cv2.INTER_AREA)

            # Convert to grayscale to match requested (4, 84, 84) shape
            # but still preserve some color information using weighted conversion
            # Emphasize red channel (better for certain game environments)
            frame = np.dot(frame, [0.6, 0.3, 0.1])

            processed_frames.append(frame)



    else:
        raise ValueError(f"Unsupported input shape: {shape}. Expected (x, 128), (x, 210, 160), or (x, 210, 160, 3)")

    while len(processed_frames) < 4:
        processed_frames.insert(0, np.zeros_like(processed_frames[0]))

    # Stack and keep 3 channels
    processed_state = np.stack(processed_frames, axis=0)

    # Normalize pixel values to [0, 1]
    processed_state = processed_state.astype(np.float32) / 255.0

    return processed_state


class Q_agent(nn.Module):

    def __init__(self, load_path=None):
        # can take in different state inputs
        super(Q_agent, self).__init__()

        # Kernel size of 3, stride of 3 will reduce 84 to 28
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1)
        # Kernel size of 3, stride of 3 will reduce 84 to 28
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=3)

        self.fc1 = nn.Linear(4 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(0.2)

        # map output index to actual
        self.action_map = {0: 0, 1: 2, 2: 3}

        # Load weights if a path is provided
        if load_path:
            self.load_weights(load_path)


    def convert_raw_frames_to_input_state(self, input_raw_frames):
        formatted_input = preprocess_state(input_raw_frames)  # format input into (4, 168, 168)

        # Flatten the entire array into a single dimension
        return torch.from_numpy(formatted_input).float().to(device='cuda')



    def forward(self, input_raw_frames):
        """
        :param input_raw_frames: a stack of 4 raw frames, can be in ram, rgb or grayscale
        :return: an int from (0, 2, 3), indicating action to take.
        """

        # convert raw frames into model inputs
        x = self.convert_raw_frames_to_input_state(input_raw_frames)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x.flatten()

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # final output of tensor size 3
        x = self.fc5(x)

        # apply softmax to output, no softmax
        # x = F.log_softmax(x)

        return x.cpu()

    def get_action(self, input_raw_frames):
        q_val = self.forward(input_raw_frames)
        # get action based on highest q-val
        return self.action_map[torch.argmax(q_val).item()], max(q_val).detach().numpy().item()

    def loss_function(self, batch_data, target_network, discount_factor=0.9):
        """
        :param batch_data: size 64, contains current state, action, reward, q_val and next state
        :param target_network: the network that we wanted to improve
        :param discount_factor: to discount the value output
        :return: PyTorch tensor with gradients enabled
        """
        # Extract current states (needed for gradient calculation)
        print("extracting current states from batch")
        batch_current_states = [np.array(data[0]) for data in batch_data]

        # Extract actions
        batch_actions = [data[1] for data in batch_data]

        # Extract rewards and next states
        print("extract out all the rewards obtained")
        batch_rewards = torch.tensor([data[2] for data in batch_data], dtype=torch.float32)

        print("extracting next states from batch")
        batch_next_states = [np.array(data[-1]) for data in batch_data]

        # Forward current states through our model to get predicted Q-values
        # This creates tensors that are connected to the computational graph
        print("forward current states through our network")
        predicted_q_values = []
        for i, state in enumerate(batch_current_states):
            # Get all Q-values for the current state
            q_values = self.forward(state)
            # Select the Q-value for the action that was taken
            action_idx = list(self.action_map.keys())[list(self.action_map.values()).index(batch_actions[i])]
            predicted_q_values.append(q_values[action_idx])

        # Stack the individual tensors into a batch tensor
        predicted_q_tensor = torch.stack(predicted_q_values)

        # Compute target Q-values using the target network
        print("target_network forward each next states to get max q_output")
        target_q_values = []
        for next_state in batch_next_states:
            with torch.no_grad():  # No need to track gradients for target computation
                q_output = target_network.forward(next_state)
                target_q_values.append(torch.max(q_output).item())

        target_q_tensor = batch_rewards + discount_factor * torch.tensor(target_q_values, dtype=torch.float32)

        # Calculate MSE loss
        loss = F.mse_loss(predicted_q_tensor, target_q_tensor)

        return loss

    def save_weights(self, save_path, title=None):
        """
        Save the model weights to the specified path

        :param save_path: Path where the model weights will be saved
        :param title: Optional title to be saved with the model weights
        """

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        save_dict = {
            'state_dict': self.state_dict(),
            'title': title
        }
        torch.save(save_dict, save_path)
        print(f"Model weights saved to {save_path}" + (f" with title: {title}" if title else ""))

    def load_weights(self, load_path):
        """
        Load model weights from the specified path

        :param load_path: Path to the saved model weights
        :return: The title of the loaded weights if available, None otherwise
        """
        try:
            checkpoint = torch.load(load_path)

            # Handle both formats: direct state_dict or dict with state_dict and title
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'])
                title = checkpoint.get('title')
                print(f"Model weights loaded from {load_path}" + (f" with title: {title}" if title else ""))
                # Move model to GPU if available
                if torch.cuda.is_available():
                    self.to('cuda')
                return title
            else:
                # Handle legacy format (direct state_dict)
                self.load_state_dict(checkpoint)
                print(f"Model weights loaded from {load_path}")
                # Move model to GPU if available
                if torch.cuda.is_available():
                    self.to('cuda')
                return None
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return None

    def combine_weights(self, other_agent, ratio=0.5):
        """
        Combine weights from another Q_agent with the current agent's weights using a specified ratio

        :param other_agent: Another Q_agent whose weights will be combined with this agent
        :param ratio: Weight given to the current agent's parameters (between 0 and 1)
                     Current = ratio * current + (1 - ratio) * other
        :return: self, for method chaining
        """

        if not 0 <= ratio <= 1:
            raise ValueError("ratio must be between 0 and 1")

        # Get state dictionaries for both models
        current_state_dict = self.state_dict()
        other_state_dict = other_agent.state_dict()

        # Ensure both models have the same parameters
        if current_state_dict.keys() != other_state_dict.keys():
            raise ValueError("Model architectures don't match")

        # Create a new state dictionary with combined weights
        combined_state_dict = {}

        for key in current_state_dict.keys():
            # Linear interpolation between the two weights
            combined_state_dict[key] = current_state_dict[key] * ratio + other_state_dict[key] * (1 - ratio)

        # Load the combined weights into the current model
        self.load_state_dict(combined_state_dict)

        print(f"Weights combined with ratio {ratio:.4f} (self) to {1 - ratio:.4f} (other)")
        return self


class ImprovedQAgent(nn.Module):

    def __init__(self, load_path=None, epsilon=1.0):
        # can take in different state inputs
        super(ImprovedQAgent, self).__init__()

        # Keep the same architecture but improve initialization
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=3)

        self.fc1 = nn.Linear(4 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        # Initialize final layer with smaller weights to prevent one action dominating
        self.fc5 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(0.2)

        # map output index to actual
        self.action_map = {0: 0, 1: 2, 2: 3}

        # Exploration parameters
        self.epsilon = epsilon  # Start with high exploration
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        # Apply proper weight initialization
        self._initialize_weights()

        # Load weights if a path is provided
        if load_path:
            self.load_weights(load_path)

        # Move to GPU if available
        if torch.cuda.is_available():
            self.to('cuda')

    def _initialize_weights(self):
        """Apply better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                if m == self.fc5:  # Final layer
                    # Smaller initialization for output layer
                    nn.init.uniform_(m.weight, -0.03, 0.03)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)

    def convert_raw_frames_to_input_state(self, input_raw_frames):
        formatted_input = preprocess_state(input_raw_frames)  # format input into (4, 168, 168)
        return torch.from_numpy(formatted_input).float().to(device='cuda')

    def forward(self, input_raw_frames):
        """
        :param input_raw_frames: a stack of 4 raw frames, can be in ram, rgb or grayscale
        :return: Q-values for each action
        """
        # convert raw frames into model inputs
        x = self.convert_raw_frames_to_input_state(input_raw_frames)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.flatten()

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Add dropout after first FC layer
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # final output of tensor size 3
        x = self.fc5(x)

        return x.cpu()

    def get_action(self, input_raw_frames, training=True):
        """
        Select action using epsilon-greedy policy during training

        :param input_raw_frames: a stack of 4 raw frames
        :param training: whether to use epsilon-greedy policy
        :return: selected action and its Q-value
        """
        # Exploration: select random action with probability epsilon
        if training and random.random() < self.epsilon:
            action_idx = random.randint(0, 2)  # Choose random index
            action = self.action_map[action_idx]

            # Still compute Q-values for monitoring purposes
            q_val = self.forward(input_raw_frames)
            return action, q_val[action_idx].detach().numpy().item()
        else:
            # Exploitation: select best action
            q_val = self.forward(input_raw_frames)
            action_idx = torch.argmax(q_val).item()
            return self.action_map[action_idx], max(q_val).detach().numpy().item()

    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            return True
        return False

    def loss_function(self, batch_data, target_network, discount_factor=0.99):
        """
        Standard DQN loss function with improved stability

        :param batch_data: size 64, contains current state, action, reward, q_val and next state
        :param target_network: the network that we wanted to improve
        :param discount_factor: higher discount factor for longer-term rewards (0.99 instead of 0.9)
        :return: PyTorch tensor with gradients enabled
        """
        # Extract current states (needed for gradient calculation)
        batch_current_states = [np.array(data[0]) for data in batch_data]

        # Extract actions
        batch_actions = [data[1] for data in batch_data]

        # Extract rewards and next states
        batch_rewards = torch.tensor([data[2] for data in batch_data], dtype=torch.float32)
        batch_next_states = [np.array(data[-1]) for data in batch_data]

        # Forward current states through our model to get predicted Q-values
        predicted_q_values = []
        for i, state in enumerate(batch_current_states):
            q_values = self.forward(state)
            action_idx = list(self.action_map.keys())[list(self.action_map.values()).index(batch_actions[i])]
            predicted_q_values.append(q_values[action_idx])

        # Stack the individual tensors into a batch tensor
        predicted_q_tensor = torch.stack(predicted_q_values)

        # Standard DQN: use target network for both action selection and evaluation
        target_q_values = []
        for next_state in batch_next_states:
            with torch.no_grad():  # No need to track gradients for target computation
                target_q_output = target_network.forward(next_state)
                target_q_values.append(torch.max(target_q_output).item())

        # Calculate target Q values with reward and discounted future reward
        target_q_tensor = batch_rewards + discount_factor * torch.tensor(target_q_values, dtype=torch.float32)

        # Use Huber loss instead of MSE for better stability
        loss = F.smooth_l1_loss(predicted_q_tensor, target_q_tensor)

        return loss

    # Keep existing save_weights, load_weights, and combine_weights methods

    def combine_weights(self, other_agent, ratio=0.5):
        """
        Combine weights from another Q_agent with the current agent's weights using a specified ratio

        :param other_agent: Another Q_agent whose weights will be combined with this agent
        :param ratio: Weight given to the current agent's parameters (between 0 and 1)
                     Current = ratio * current + (1 - ratio) * other
        :return: self, for method chaining
        """

        if not 0 <= ratio <= 1:
            raise ValueError("ratio must be between 0 and 1")

        # Get state dictionaries for both models
        current_state_dict = self.state_dict()
        other_state_dict = other_agent.state_dict()

        # Ensure both models have the same parameters
        if current_state_dict.keys() != other_state_dict.keys():
            raise ValueError("Model architectures don't match")

        # Create a new state dictionary with combined weights
        combined_state_dict = {}

        for key in current_state_dict.keys():
            # Linear interpolation between the two weights
            combined_state_dict[key] = current_state_dict[key] * ratio + other_state_dict[key] * (1 - ratio)

        # Load the combined weights into the current model
        self.load_state_dict(combined_state_dict)

        print(f"Weights combined with ratio {ratio:.4f} (self) to {1 - ratio:.4f} (other)")
        return self





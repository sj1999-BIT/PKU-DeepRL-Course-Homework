'''
Implement the DQN agent here
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        for i in range(4):
            # Reshape 1D vector to 2D (8, 16)
            frame = state[i].reshape(8, 16)
            # Resize to (84, 84)
            frame = cv2.resize(frame, (168, 168), interpolation=cv2.INTER_AREA)

            processed_frames.append(frame)

        # Stack and set channels to 1
        processed_state = np.stack(processed_frames, axis=0)

    elif len(shape) == 3 and shape[1:] == (210, 160):
        # 2D grayscale input (4, 210, 160)
        processed_frames = []
        for i in range(4):
            # Resize to (84, 84)
            frame = cv2.resize(state[i], (168, 168), interpolation=cv2.INTER_AREA)
            processed_frames.append(frame)

        # Stack and set channels to 1
        processed_state = np.stack(processed_frames, axis=0)

    elif len(shape) == 4 and shape[1:] == (210, 160, 3):
        # RGB input (4, 210, 160, 3)
        processed_frames = []
        for i in range(4):
            # Resize to (84, 84) while keeping RGB channels
            frame = cv2.resize(state[i], (168, 168), interpolation=cv2.INTER_AREA)

            # Convert to grayscale to match requested (4, 84, 84) shape
            # but still preserve some color information using weighted conversion
            # Emphasize red channel (better for certain game environments)
            frame = np.dot(frame, [0.6, 0.3, 0.1])

            processed_frames.append(frame)

        # Stack and keep 3 channels
        processed_state = np.stack(processed_frames, axis=0)

    else:
        raise ValueError(f"Unsupported input shape: {shape}. Expected (4, 128), (4, 210, 160), or (4, 210, 160, 3)")

    # Normalize pixel values to [0, 1]
    processed_state = processed_state.astype(np.float32) / 255.0

    return processed_state


class Q_agent(nn.Module):

    def __init__(self):
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



    def forward(self, input_raw_frames):
        """
        :param input_raw_frames: a stack of 4 raw frames, can be in ram, rgb or grayscale
        :return: an int from (0, 2, 3), indicating action to take.
        """

        formatted_input = preprocess_state(input_raw_frames) # format input into (4, 168, 168)

        # Flatten the entire array into a single dimension
        x = torch.from_numpy(formatted_input).float().to(device='cuda')

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








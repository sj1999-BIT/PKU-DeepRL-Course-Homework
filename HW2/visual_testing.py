"""
render environment and check how the model performed
"""

from network import PolicyNetwork
# Replay Buffer
from tqdm import tqdm
from collections import deque
import random
import gymnasium as gym
import numpy as np
import torch


# Create the environment
env = gym.make("HalfCheetah-v5", render_mode="human")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# second 1000 epoch, agent moving with larger action, showing some form of shifting forward.
# large movement penalise more than the shift in forward, result in a decrease in reward accumulated
weight_path = "./results and data/2_1900_epoch/Policy_nn_weight.pth"

# first 1000 epoch, agent move randomly with small action, not really moving forward, but also small penalise
# weight_path = "./results and data/1_700 epoch/Policy_nn_weight.pth"

pNet = PolicyNetwork(obs_dim, action_dim)

pNet.load_weights(weight_path)

if __name__ == "__main__":
    env = gym.make("HalfCheetah-v5", render_mode="human")
    total_reward = 0
    done = False

    state, _ = env.reset()
    while not done:
        with torch.no_grad():
            # Convert current states to tensor
            states_tensor = torch.FloatTensor(np.array(state))

            # Get actions using policy network (stay in tensor form)
            actions_tensor, probs_tensor = pNet.get_action(states_tensor)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(actions_tensor.numpy())
            done = terminated or truncated

            total_reward += reward

            print(f"accumulated reward {total_reward}")




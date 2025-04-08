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
weight_path = "./Policy_nn_weight.pth"

# first 1000 epoch, agent move randomly with small action
# weight_path = "./results and data/1_700 epoch/Policy_nn_weight.pth"

pNet = PolicyNetwork(obs_dim, action_dim)

pNet.load_weights(weight_path)

# Set camera to track mode to follow the agent
if hasattr(env.unwrapped, 'viewer') and env.unwrapped.viewer is not None:
    env.unwrapped.viewer.cam.trackbodyid = 0  # Track the main body


def get_reward():
    env = gym.make("HalfCheetah-v5", render_mode=None)
    total_reward = 0
    state, _ = env.reset()
    for _ in range(1000):

        with torch.no_grad():
            # Convert current states to tensor
            states_tensor = torch.FloatTensor(np.array(state))

            # Get actions using policy network
            actions_tensor, probs_tensor = pNet.get_action(states_tensor)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(actions_tensor.numpy())
            total_reward += reward
            state = next_state

    print(f"total reward: {total_reward}")
    return total_reward

if __name__ == "__main__":
    # env = gym.make("HalfCheetah-v5", render_mode="human")
    # total_reward = 0
    # done = False
    #
    # timestep = 0
    #
    # # Add time control for frame rate limiting
    # import time
    #
    # TARGET_FPS = 60
    # FRAME_TIME = 1.0 / 1  # seconds per frame
    #
    # state, _ = env.reset()
    # for _ in range(1000):
    #     frame_start = time.time()
    #
    #     with torch.no_grad():
    #         # Convert current states to tensor
    #         states_tensor = torch.FloatTensor(np.array(state))
    #
    #         # Get actions using policy network
    #         actions_tensor, probs_tensor = pNet.get_action(states_tensor)
    #
    #         # Step environment
    #         next_state, reward, terminated, truncated, _ = env.step(actions_tensor.numpy())
    #         done = terminated or truncated
    #
    #         total_reward += reward
    #         timestep += 1
    #         state = next_state
    #
    #         print(f"timestep {timestep}: accumulated reward {total_reward}")
    #
    #     # # Calculate time spent on this frame
    #     # frame_time = time.time() - frame_start
    #     #
    #     # # Sleep to maintain target frame rate
    #     # if frame_time < FRAME_TIME:
    #     #     time.sleep(FRAME_TIME - frame_time)
    get_reward()
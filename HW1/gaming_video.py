"""
For recording of video of agent playing the game.
"""

from model import Q_agent
import torch.optim as optim

import random
from replayDataBuffer import get_training_data, get_reward_data
from visual import plot_progress_data
from data import save_array_to_file, append_values_to_file, load_array_from_file
import gymnasium as gym
import numpy as np
import ale_py
from model import Q_agent, QAgentWithEpilsonAndMoreDepth
from collections import deque
import random
from tqdm import tqdm

# Register Atari Learning Environment environments with Gymnasium
gym.register_envs(ale_py)

# Create Pong environment without rendering for faster data collection
env = gym.make("ALE/Pong-v5", render_mode="human", obs_type="rgb")

if __name__=="__main__":

    # initialise 2 agents
    q_agent = QAgentWithEpilsonAndMoreDepth("naive/target_agent_weights.pth")
    q_agent.to(device='cuda')

    for i in range(10):

        # Initialize new game
        done = False
        obs, info = env.reset()  # BUG FIX: env.reset() returns (obs, info) in newer Gymnasium


        # Make a copy of current state for recording
        # BUG: This line creates an empty copy since current_state is empty at the start of each game
        # Should be moved to after filling current_state

        # maintain 4 consecutive frames in the queue for current state
        current_state = deque(maxlen=4)
        current_state.append(obs)


        while not done:
            env.render()
            input_state = np.array(current_state)
            action, _ = q_agent.get_action(input_state)

            # Take action and observe result
            obs, reward, _, _, info = env.step(action)  # BUG FIX: Unpack 5 values
            if reward != 0:
                done = True

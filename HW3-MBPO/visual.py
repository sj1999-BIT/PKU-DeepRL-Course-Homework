import torch

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

from data import load_array_from_file
from neural_network import PolicyNetwork, PredictiveNetwork
from Ensembler import Ensembler
from main import load_NN

"""
Reinforcement Learning Visualization and Evaluation
--------------------------------------------------
This module provides utilities for:
1. Visualizing training metrics from saved data files
2. Evaluating trained policies in a rendered environment
3. Displaying performance metrics

The visualization functions help analyze the training progress by plotting
various metrics like policy loss, value loss, and rewards over time.

The simulation function allows for visual evaluation of the trained policy
by loading model weights and running the policy in a rendered environment.
"""


def plot_progress_data(data, save_plot=False, plot_file_title=None):
    # After training is complete, plot the loss graph

    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(f'{plot_file_title} over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    if save_plot:
        if plot_file_title is None:
            plot_file_title = "plot.png"
        plt.savefig(plot_file_title)
    plt.show()


def simulate_policy(path=".", mode=None):
    # initiate new environment
    env = gym.make("Hopper-v5", render_mode=mode)
    total_reward = 0
    state, _ = env.reset()

    pNet = PolicyNetwork(env)
    pNet.load_weights(f"{path}/Policy_nn_weight.pth")

    for _ in range(1000):
        with torch.no_grad():

            # Get actions using policy network
            actions_tensor, _ = pNet.get_action(state)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(actions_tensor.numpy())
            total_reward += reward
            state = next_state

            # Check if episode is done
            if terminated or truncated:
                break

    print(f"total reward: {total_reward}")
    return total_reward


# Example usage:
if __name__ == "__main__":
    for model_index in range(10):
        filename = f"model_{model_index}_loss"
        loss_arr = load_array_from_file(f"./{filename}.txt")
        plot_progress_data(loss_arr, save_plot=True, plot_file_title=filename)


import torch

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

from data import load_array_from_file
from network import PolicyNetwork

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


def simulate_policy(weight_path="./data/Policy_nn_weight.pth"):
    # initiate new environment
    env = gym.make("HalfCheetah-v5", render_mode="human")
    total_reward = 0
    state, _ = env.reset()

    # initiate new policynetwork with the saved weights
    pNet = PolicyNetwork(env)
    pNet.load_weights(weight_path)

    for _ in range(1000):
        with torch.no_grad():
            # Convert current states to tensor
            states_tensor = torch.FloatTensor(np.array(state))

            # Get actions using policy network
            actions_tensor, probs_tensor = pNet.get_action(states_tensor)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(actions_tensor.cpu().numpy())
            total_reward += reward
            state = next_state

    print(f"total reward: {total_reward}")
    return total_reward


# Example usage:
if __name__ == "__main__":

    # plot graphs for the collected data
    folder_name = "./data/"

    policy_loss_arr = load_array_from_file(f"{folder_name}/policy_loss.txt")
    plot_progress_data(policy_loss_arr, save_plot=True, plot_file_title="policy_loss")

    policy_loss_arr = load_array_from_file(f"{folder_name}/value_loss.txt")
    plot_progress_data(policy_loss_arr, save_plot=True, plot_file_title="value_loss")

    policy_loss_arr = load_array_from_file(f"{folder_name}/reward_plot.txt")
    plot_progress_data(policy_loss_arr, save_plot=True, plot_file_title="reward")

    # simulation of the policy
    simulate_policy()


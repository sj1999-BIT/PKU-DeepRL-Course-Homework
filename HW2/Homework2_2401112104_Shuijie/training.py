from network import ValueNetwork, PolicyNetwork
from trajectoryProcessor import TrajectoryProcessor
from data import append_values_to_file

import gymnasium as gym
import torch
import numpy as np

"""
Proximal Policy Optimization (PPO) Training Module
-------------------------------------------------

This module implements a complete training loop for PPO reinforcement learning algorithm
applied to continuous control tasks such as the HalfCheetah environment.

Key components:
1. Training loop for iterative policy and value network updates
2. Environment management for parallel trajectory collection
3. Periodic model saving and performance evaluation
4. Metrics tracking and data logging

The training follows the PPO algorithm with the following workflow:
- Collect trajectory data from multiple parallel environments
- Compute advantages using Generalized Advantage Estimation (GAE)
- Perform multiple epochs of policy and value network updates on sampled mini-batches
- Periodically evaluate policy performance and save model checkpoints
- Log training metrics for later analysis

This implementation uses tensor-based operations for efficient computation and
supports both CPU and GPU acceleration through PyTorch.
"""



def single_epoch_update(policy_net: PolicyNetwork, value_net: ValueNetwork, data_processor: TrajectoryProcessor, envs: [gym.Env],
                        batch_size=64):
    """
    In each epoch we aim the following things
    1. collect data using the policy network and the replayBuffer
    2. random sample training batches to train both network till convergence.
    :param batch_size: batch size used from random sample to train network
    :param envs: initiated environments for simulation
    :param data_processor:
    :param policy_net:
    :param value_net:
    :return:
    """

    """
    collect data for using policy and value network with the initiated envs
    The trajectory tensors contain keys:

    states: Environment states at each timestep, initially shaped [timesteps, trajectories, state_dim]
    actions: Actions taken by the policy, initially shaped [timesteps, trajectories, action_dim]
    rewards: Rewards received after each action, initially shaped [timesteps, trajectories]
    old_probs: Action probabilities from the policy that generated the data, initially shaped [timesteps, trajectories]
    next_states: States after taking actions, initially shaped [timesteps, trajectories, state_dim]
    dones: Boolean flags indicating terminal states, initially shaped [timesteps, trajectories]
    advantages: Computed advantage estimates using GAE, likely shaped [timesteps, trajectories]
    returns: Computed returns (discounted sum of rewards), likely shaped [timesteps, trajectories]
    """
    trajectory_tensor = data_processor.generate_data(envs, policy_net, value_net)


    policy_loss_arr = []
    value_loss_arr = []

    for _ in range(10):
        # extract a random batch of data for training.
        batch_training_data_tensor = data_processor.random_sample(trajectory_tensor, batch_size)

        # policy network learning
        policy_loss = policy_net.get_loss(batch_training_data_tensor)
        policy_net.optimizer.zero_grad()
        policy_loss.backward()
        policy_net.optimizer.step()

        # value network learning
        value_loss = value_net.get_loss(batch_training_data_tensor)
        value_net.optimizer.zero_grad()
        value_loss.backward()
        value_net.optimizer.step()


        # save the loss value for tracking
        policy_loss_arr.append(policy_loss.detach().item())
        value_loss_arr.append(value_loss.detach().item())

    append_values_to_file(policy_loss_arr, "./data/policy_loss.txt")
    # plot_progress_data(policy_loss_arr, save_plot=True, plot_file_title="policy_loss")

    append_values_to_file(value_loss_arr, "./data/value_loss.txt")
    # plot_progress_data(value_loss_arr, save_plot=True, plot_file_title="value_loss")


def get_reward(weight_path="./Policy_nn_weight.pth"):
    # initiate new environment
    env = gym.make("HalfCheetah-v5", render_mode=None)
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
            next_state, reward, terminated, truncated, _ = env.step(actions_tensor.numpy())
            total_reward += reward
            state = next_state

    print(f"total reward: {total_reward}")
    return total_reward



if __name__ == "__main__":
    # Create the environment
    env = gym.make("HalfCheetah-v5", render_mode=None)
    pNet = PolicyNetwork(env)
    pNet.load_weights("./data/Policy_nn_weight.pth")
    vNet = ValueNetwork(env)
    vNet.load_weights("./data/Value_nn_weight.pth")

    # Pre-create batch of environments
    env_num = 10  # Process 10 environments at a time
    envs = [gym.make(env.unwrapped.spec.id) for _ in range(env_num)]

    time_step = 0
    reward_arr = []

    # initialise buffer
    processor = TrajectoryProcessor()

    while True:

        if time_step % 100 == 0:
            # every 100 timestep we save the weights
            pNet.save_weights()
            vNet.save_weights()

        single_epoch_update(pNet, vNet, processor, envs, batch_size=64)

        # record the new reward obtained from the updated model
        reward = get_reward()
        append_values_to_file(reward, "./data/reward_plot.txt")

        time_step += 1

        print(f"current time_step {time_step}, modal reward {reward}")


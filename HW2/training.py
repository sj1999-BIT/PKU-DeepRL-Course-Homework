import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from network import ValueNetwork, PolicyNetwork
from replayBuffer import ReplayBuffer
from data import append_values_to_file
from visual import plot_progress_data

from tqdm import tqdm



def ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, log_probs_old, advantages,
               returns, clip_ratio=0.2, epochs=10):
    for _ in range(epochs):
        # Get current policy evaluation
        log_probs, entropy = policy_net.evaluate(states, actions)

        # Compute policy ratio (π_θ / π_θ_old)
        ratio = torch.exp(log_probs - log_probs_old)

        # Compute clipped objective
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()

        # Add entropy bonus
        policy_loss = policy_loss - 0.01 * entropy.mean()

        # Update policy
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # Update value function
        value_pred = value_net(states)
        value_loss = F.mse_loss(value_pred, returns)

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()


def single_epoch_update(policy_net: PolicyNetwork, value_net: ValueNetwork, buffer: ReplayBuffer, envs: [gym.Env],
                        batch_size=64):
    """
    In each epoch we aim the following things
    1. collect data using the policy network and the replayBuffer
    2. random sample training batches to train both network till convergence.
    :param batch_size: batch size used from random sample to train network
    :param envs: initiated environments for simulation
    :param buffer:
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
    trajectory_tensor = buffer.generate_data(envs, policy_net, value_net)

    policy_loss = None

    policy_loss_arr = []
    value_loss_arr = []

    convergence_threshold = 1e-4


    for _ in range(10):
        # extract a random batch of data for training.
        batch_training_data_tensor = buffer.random_sample(trajectory_tensor, batch_size)

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

    append_values_to_file(policy_loss_arr, "./policy_loss.txt")
    # plot_progress_data(policy_loss_arr, save_plot=True, plot_file_title="policy_loss")

    append_values_to_file(value_loss_arr, "./value_loss.txt")
    # plot_progress_data(value_loss_arr, save_plot=True, plot_file_title="value_loss")

def get_reward(policy_net: PolicyNetwork):
    """
    Test the policy_net ability to play, get the overall reward
    :param policy_net:
    :return:
    """

    env = gym.make("HalfCheetah-v5", render_mode=None)
    total_reward = 0
    done = False

    state, _ = env.reset()
    while not done:
        with torch.no_grad():
            # Convert current states to tensor
            states_tensor = torch.FloatTensor(np.array(state))

            # Get actions using policy network (stay in tensor form)
            actions_tensor, probs_tensor = policy_net.get_action(states_tensor)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(actions_tensor.numpy())
            done = terminated or truncated

            total_reward += reward

    return total_reward



if __name__ == "__main__":
    # Create the environment
    env = gym.make("HalfCheetah-v5", render_mode=None)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    pNet = PolicyNetwork(obs_dim, action_dim)
    vNet = ValueNetwork(obs_dim)

    # Pre-create batch of environments
    batch_size = 10  # Process 10 environments at a time
    envs = [gym.make(env.unwrapped.spec.id) for _ in range(batch_size)]

    time_step = 0
    reward_arr = []

    # initialise buffer
    buffer = ReplayBuffer(capacity=10)

    while True:

        if time_step % 100 == 0:
            # every 100 timestep we save the weights
            pNet.save_weights()
            vNet.save_weights()
            append_values_to_file(reward_arr, "./reward_plot.txt")
            reward_arr = []

        single_epoch_update(pNet, vNet, buffer, envs, batch_size=64)

        reward = get_reward(pNet)

        reward_arr.append(reward)

        time_step += 1

        print(f"current time_step {time_step}, modal reward {reward}")


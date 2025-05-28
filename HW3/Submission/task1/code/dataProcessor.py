"""
Define a class that keeps track of all the transitions we have gathered.

1. initialised by gathering random transitions up to 100000 and store them
2. can generate training samples for DynamicsNeuralNetwork for training
3. can add in new transitions as we perform bootstrapping
"""
import math
from collections import deque
from tqdm import tqdm
from neural_network import PolicyNetwork, ValueNetwork, DynamicsNetwork
from Ensembler import Ensembler

import torch
from dataKeys import STATES, NEXT_STATES, ACTIONS, REWARD, LOG_PROB

import numpy as np
import gymnasium as gym


class DataProcessor:
    def __init__(self, env: gym.Env, pNet: PolicyNetwork, maxSize=100000, need_initialise=True):
        self.env = env

        # initialise with 10000 random transitions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.transitionBuffer = {
            STATES: torch.empty((0, state_dim), dtype=torch.float32),
            NEXT_STATES: torch.empty((0, state_dim), dtype=torch.float32),
            REWARD: torch.empty((0,), dtype=torch.float32),
            ACTIONS: torch.empty((0, action_dim), dtype=torch.float32)
        }

        if need_initialise:
            self.initial_collect_data(pNet, transition_size=10000)

        self.maxBufferSize = maxSize

        # new update, we take into consideration possible early termination
        self.healthy_state_range = [-100, 100]
        self.healthy_z_range = [0.7, float('inf')]
        self.healthy_angle_range = [-0.2, 0.2]


    def update_transition_buffer(self):
        """
        Update all keys in the transition buffer and maintain the maximum length for each key.

        Args:
            result (dict): Dictionary containing new data to add to the buffer
            max_buffer_size (int): Maximum allowed buffer size for each key
        """

        # Then check if any key exceeds the maximum size
        current_size = self.transitionBuffer[list(self.transitionBuffer.keys())[0]].shape[0]
        if current_size > self.maxBufferSize:
            # Generate random indices to keep
            indices_to_keep = torch.randperm(current_size)[:self.maxBufferSize]

            # Apply the same indices to all keys to keep the data aligned
            for key in self.transitionBuffer.keys():
                self.transitionBuffer[key] = self.transitionBuffer[key][indices_to_keep]

    def initial_collect_data(self, pNet: PolicyNetwork, transition_size=10000):
        """
        Aim to collect 10,000 transitions from random action.
        Each transitions contains an state, action and next state and the reward.

        state: array size 11, value ranged (-INF, INF)
        action: array size 3, value ranged (-1.0, 1.0)

        :return: transition array given size
        """

        states = []
        actions = []
        rewards = []
        next_states = []

        state, _ = self.env.reset()

        try:
            for step in tqdm(range(transition_size), desc="getting transitions"):

                state_tensor = torch.tensor(state, dtype=torch.float32)

                states.append(state_tensor)
                # get action from policyNetwork
                action_tensor, _ = pNet.get_action(state_tensor)
                actions.append(action_tensor)
                # Step environment
                next_state, reward, terminated, truncated, _ = self.env.step(action_tensor.cpu().numpy())

                next_states.append(torch.tensor(next_state, dtype=torch.float32))
                rewards.append(torch.tensor(reward, dtype=torch.float32))
                state = next_state

                # Check if episode is done
                if terminated or truncated:
                    # reset env if terminated
                    state, _ = self.env.reset()
        finally:
            # Ensure environment is properly closed even if an exception occurs
            self.env.close()

        self.transitionBuffer = {
            STATES: torch.stack(states),
            ACTIONS: torch.stack(actions),
            REWARD: torch.stack(rewards),
            NEXT_STATES: torch.stack(next_states)
        }

    def check_termination(self, next_state_tensor):
        """
        Check if the predicted next state would cause termination
        """
        batch_size = next_state_tensor.shape[0]
        terminated = torch.zeros(batch_size, dtype=torch.bool)

        # Condition 1: Check other state elements are in healthy range
        elements_to_check = next_state_tensor[:, 1:]  # All elements except first

        if not isinstance(elements_to_check, torch.Tensor):
            elements_to_check = torch.tensor(elements_to_check, dtype=torch.float32)

        terminated |= (elements_to_check.any(dim=1) < self.healthy_state_range[0])
        terminated |= (elements_to_check.any(dim=1) > self.healthy_state_range[1])

        # Condition 2: Check height (z position)
        z_height = next_state_tensor[:, 0]
        terminated |= (z_height < self.healthy_z_range[0])

        # Condition 3: Check angle
        angle = next_state_tensor[:, 1]
        terminated |= (angle < self.healthy_angle_range[0])
        terminated |= (angle > self.healthy_angle_range[1])

        return terminated.float().unsqueeze(-1)

    def generate_trajectory_reward_batch(self, cur_state, pNet: PolicyNetwork, vNet: ValueNetwork,
                                         ensemble_dNet: Ensembler,
                                         trajectory_num=1000, trajectory_len=30, discount_factor=0.9):
        """
        Generate multiple trajectories in parallel and find the rewards

        Args:
            cur_state: current state of the environment
            pNet: policy network to infer actions given states
            vNet: value network to infer expected returns given states
            ensemble_dNet: dynamic network to infer next states
            trajectory_num: number of trajectories to simulate in parallel
            trajectory_len: length of each trajectory
            discount_factor: discount factor for future rewards

        Returns:
            actions: first actions for all trajectories
            log_probs: log probabilities of the first actions
            total_rewards: total rewards for all trajectories
        """

        state_batch = np.array([cur_state, ]).repeat(trajectory_num, 0)
        # Repeat the current state for each trajectory
        states_batch_tensor = torch.tensor(state_batch, dtype=torch.float32)  # Shape: [batch_size, state_dim]

        # Get initial actions and log probs for all trajectories
        actions_tensor, log_probs_tensor = pNet.get_action(states_batch_tensor)  # Shape: [batch_size, action_dim]

        # Store initial actions and log probs
        output_actions = actions_tensor.clone()
        output_log_probs = log_probs_tensor.clone()

        # Initialize total rewards
        total_rewards = torch.zeros(trajectory_num)

        # Create discount factors tensor
        discount_powers = torch.tensor([discount_factor ** i for i in range(trajectory_len)])

        # Track which trajectories are still active
        active_mask = torch.ones(trajectory_num, dtype=torch.bool)

        # Simulate trajectories
        for step in range(trajectory_len):
            # Get next states for all trajectories
            next_state_tensor = ensemble_dNet.get_best_next_state(states_batch_tensor,
                                                                  actions_tensor)  # Shape: [batch_size, state_dim]

            # Check termination based on Hopper environment rules
            terminated = self.check_termination(next_state_tensor).bool().squeeze()

            # update the terminated trajectories
            active_mask &= ~terminated

            # Keep track of active trajectories
            active_count = active_mask.sum().item()
            # print(f"Active trajectories: {active_count}")

            # Stop if all trajectories are inactive
            if active_count == 0:
                # print("All trajectories terminated. Stopping.")
                break  # or return, depending on your loop structure

            # Get values for all next states
            values = vNet.forward(next_state_tensor)  # Shape: [batch_size, 1]

            # Zero out values for terminated trajectories
            values = values * active_mask.float().unsqueeze(-1)

            # Add discounted values to total rewards
            total_rewards += values.squeeze() * discount_powers[step]

            # Update states
            states_batch_tensor = next_state_tensor

            # Get next actions
            actions_tensor, _ = pNet.get_action(states_batch_tensor)

        return output_actions, output_log_probs, total_rewards

    def model_predictive_control(self, cur_state, pNet: PolicyNetwork, vNet: ValueNetwork, ensemble_dNet: Ensembler,
                                 trajectory_num=1000, trajectory_len=30, discount_factor=0.9):
        """
        Accelerated MPC using tensor operations

        Args:
            cur_state: current state of the environment
            pNet: policy network to infer actions given states
            vNet: value network to infer expected returns given states
            ensemble_dNet: dynamic network to infer next states
            trajectory_num: number of trajectories to simulate in parallel
            trajectory_len: length of each trajectory
            discount_factor: discount factor for future rewards

        Returns:
            best_action: action with highest expected return
            action_log_prob: log probability of the best action
        """
        # Run all trajectories in parallel
        actions, log_probs, rewards = self.generate_trajectory_reward_batch(
            cur_state, pNet, vNet, ensemble_dNet, trajectory_num, trajectory_len, discount_factor
        )

        # Find trajectory with max reward
        max_reward_idx = torch.argmax(rewards)

        # Get best action and its log probability
        best_action = actions[max_reward_idx]
        action_log_prob = log_probs[max_reward_idx]

        return best_action, action_log_prob.detach()

    def generate_policy_training_data(self, pNet: PolicyNetwork, vNet: ValueNetwork, ensemble_dNet: Ensembler,
                               trajectory_num=1000, trajectory_len=10, discount_factor=0.9, time_step_num=1000):
        """
        given we have the policyNet, valueNet and the enseembler, we can repeatedly apply MPC to get new data.
        We need to collect for each MPC
        1. cur_state, next_state, action, reward: add to current buffer
        2. log_best-action_probability, what is the odds of current model taking this action.

        we collect 1000 steps by default
        :return:
        """
        states = []
        best_actions = []
        log_prob_best_actions = []
        rewards = []
        next_states = []

        cur_state, _ = self.env.reset()

        for step in tqdm(range(time_step_num), desc="using MPC to generate best transitions"):
            # get best action
            best_action_tensor, log_prob_tensor = self.model_predictive_control(cur_state, pNet, vNet, ensemble_dNet,
                                                                                trajectory_num=trajectory_num,
                                                                                trajectory_len=trajectory_len,
                                                                                discount_factor=discount_factor)

            # Step environment
            next_state, reward, terminated, truncated, _ = self.env.step(best_action_tensor.detach().cpu().numpy())

            states.append(torch.tensor(cur_state, dtype=torch.float32))
            best_actions.append(best_action_tensor)
            next_states.append(torch.tensor(next_state, dtype=torch.float32))
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            log_prob_best_actions.append(log_prob_tensor)

            cur_state = next_state

            # Check if episode is done
            if terminated or truncated:
                # reset env if terminated
                cur_state, _ = self.env.reset()

        result = {
            STATES: torch.stack(states),
            ACTIONS: torch.stack(best_actions),
            REWARD: torch.stack(rewards),
            NEXT_STATES: torch.stack(next_states),
            LOG_PROB: torch.stack(log_prob_best_actions)
        }

        # add new data into buffer
        self.transitionBuffer[STATES] = torch.cat([self.transitionBuffer[STATES], result[STATES]], dim=0)
        self.transitionBuffer[NEXT_STATES] = torch.cat([self.transitionBuffer[NEXT_STATES], result[NEXT_STATES]], dim=0)
        self.transitionBuffer[REWARD] = torch.cat([self.transitionBuffer[REWARD], result[REWARD]], dim=0)
        self.transitionBuffer[ACTIONS] = torch.cat([self.transitionBuffer[ACTIONS], result[ACTIONS]], dim=0)

        # maintain fixed size
        self.update_transition_buffer()

        return result

    def random_sample(self, tensors_dict=None, keys_to_sample=None, batch_size=64):
        """
        Extract a batch of random samples from specified keys in a dictionary of tensors.

        Args:
            tensors_dict: Dictionary containing tensors
            keys_to_sample: List of keys to extract samples from
            batch_size: Number of samples to extract

        Returns:
            Dictionary containing batched tensors of randomly sampled data for the specified keys
        """

        # default setting
        if tensors_dict is None:
            tensors_dict = self.transitionBuffer

        if keys_to_sample is None:
            keys_to_sample = tensors_dict.keys()

        # Check if keys exist in the dictionary
        for key in keys_to_sample:
            if key not in tensors_dict:
                raise ValueError(f"Key '{key}' not found in tensor dictionary")

        # Get the length of the first tensor to determine valid indices
        first_key = list(keys_to_sample)[0]
        data_length = len(tensors_dict[first_key])

        if data_length < batch_size:
            raise ValueError(f"only {data_length} data collected, not enough for batch size {batch_size}")

        # Generate random indices - only once for all tensors
        rand_indices = torch.randint(low=0, high=data_length, size=(batch_size,))

        # Initialize the result dictionary
        result = {}

        # Extract samples for each requested key
        for key in keys_to_sample:
            # Check if we're dealing with a tensor
            if isinstance(tensors_dict[key], torch.Tensor):
                # Direct indexing for tensors is more efficient
                result[key] = tensors_dict[key][rand_indices]
            else:
                # Handle non-tensor types (lists, numpy arrays, etc.)
                # Convert to numpy first if it's not already a tensor
                samples = [tensors_dict[key][i.item()] for i in rand_indices]
                result[key] = torch.tensor(samples, dtype=torch.float32)

        return result


if __name__ == "__main__":
    test_env = gym.make("Hopper-v5", render_mode=None)

    test_pNet = PolicyNetwork(test_env)

    tp = DataProcessor(test_env, test_pNet)

    tp.random_sample()

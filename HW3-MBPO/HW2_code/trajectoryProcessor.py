from tqdm import tqdm
import gymnasium as gym
import numpy as np
import torch

from network import ValueNetwork, PolicyNetwork

"""
Trajectory Processing for Reinforcement Learning
-----------------------------------------------

This module provides the TrajectoryProcessor class, which handles collection, processing,
and sampling of trajectory data for reinforcement learning algorithms, specifically
designed for Policy Gradient methods like PPO (Proximal Policy Optimization).

Key features:
- Parallel trajectory collection across multiple environments
- Generalized Advantage Estimation (GAE) for robust policy learning
- Efficient tensor-based operations for improved performance
- Structured data format for mini-batch training

The implementation efficiently processes trajectory data by:
1. Collecting state, action, reward, and done signals across multiple environments
2. Computing advantages and returns using GAE
3. Organizing data in tensor format for efficient training
4. Providing random sampling functionality for mini-batch updates

Typical usage involves:
1. Creating a TrajectoryProcessor instance
2. Generating trajectory data using policy and value networks
3. Sampling batches for training network updates

This approach is particularly effective for on-policy algorithms where fresh
trajectory data is needed for each training iteration.
"""

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrajectoryProcessor:
    """
    Handles the collection and processing of trajectory data for reinforcement learning.

    This class is responsible for:
    1. Generating trajectory data by executing a policy network in environments
    2. Computing Generalized Advantage Estimation (GAE) for collected trajectories
    3. Processing and formatting trajectory data into tensor format for efficient training
    4. Providing sampling methods to extract training batches from trajectory data

    Unlike a traditional replay buffer that stores experiences over time, this class
    primarily focuses on collecting fresh trajectories for each training iteration and
    preparing them with the necessary information (advantages, returns) for policy
    optimization algorithms like PPO.
    """

    def calculate_gae_tensors(self, states, rewards, next_states, dones, value_network, gamma=0.99, lambda_=0.95):
        """
        Calculate Generalized Advantage Estimation for batched trajectories using tensors

        Args:
            states: Tensor of states [time_steps, batch_size, state_dim]
            rewards: Tensor of rewards [time_steps, batch_size]
            next_states: Tensor of next states [time_steps, batch_size, state_dim]
            dones: Tensor of done flags [time_steps, batch_size]
            value_network: Value network to estimate state values
            gamma: Discount factor (default: 0.99)
            lambda_: GAE lambda parameter (default: 0.95)

        Returns:
            advantages: Tensor of advantages [time_steps, batch_size]
            returns: Tensor of returns [time_steps, batch_size]
        """
        # Get shapes for reshaping
        batch_size = states.size(1)
        time_steps = states.size(0)
        state_dim = states.size(2)

        # Flatten the batches for efficient value network evaluation
        flat_states = states.reshape(-1, state_dim)
        flat_next_states = next_states.reshape(-1, state_dim)

        # Get value estimates (single batch processing)
        with torch.no_grad():
            flat_values = value_network.forward(flat_states)
            flat_next_values = value_network.forward(flat_next_states)

        # Reshape values back to [time_steps, batch_size]
        values = flat_values.reshape(time_steps, batch_size)
        next_values = flat_next_values.reshape(time_steps, batch_size)

        # Ensure rewards and dones have the right shape
        if rewards.dim() == 3:  # If rewards has shape [time_steps, batch_size, 1]
            rewards = rewards.squeeze(-1)
        if dones.dim() == 3:  # If dones has shape [time_steps, batch_size, 1]
            dones = dones.squeeze(-1)

        # Calculate td-error (δₜ = rₜ + γV(s_{t+1}) - V(sₜ))
        deltas = rewards + gamma * next_values * (~dones).float() - values

        # Initialize advantages tensor
        advantages = torch.zeros_like(deltas)

        # Calculate GAE by working backwards
        last_gae = torch.zeros(batch_size, device=states.device)

        for t in reversed(range(time_steps)):
            # For done episodes, reset the last_gae
            mask = (~dones[t]).float()
            last_gae = deltas[t] + gamma * lambda_ * mask * last_gae
            advantages[t] = last_gae

        # Calculate returns (G = A + V)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def generate_data(self, envs: [gym.Env], policyNet: PolicyNetwork, valueNet: ValueNetwork):
        """
        Generates training data by collecting trajectories using the provided policy network,
        storing all data in tensor form for faster processing.

        This method runs the provided policy in multiple environments simultaneously, collecting
        state transitions, actions, rewards, and other relevant information. It then calculates
        advantages and returns using GAE for policy optimization.

        Args:
            envs: List of gym environments to collect trajectories from
            policyNet: The policy network used to select actions
            valueNet: The value network for GAE calculation

        Returns:
            A dictionary containing tensors for states, actions, rewards, old_log_probs,
            advantages, returns, and other trajectory data
        """

        # Initialize tensor lists for current batch
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_old_log_probs = []
        batch_next_states = []
        batch_dones = []

        # Initialize environment states
        current_states = []
        is_done = torch.zeros(len(envs), dtype=torch.bool, device=device)

        for i, e in enumerate(envs):
            state, _ = e.reset()
            current_states.append(state)

        # Process steps for fixed episode length
        for trajectory_step in tqdm(range(1000), desc="Collecting replay experience"):  # Fixed step limit
            # Convert current states to tensor
            states_tensor = torch.FloatTensor(np.array(current_states)).to(device)

            # Get actions using policy network (stay in tensor form)
            actions_tensor, log_probs_tensor = policyNet.get_action(states_tensor)

            # Store current states
            batch_states.append(states_tensor)
            batch_actions.append(actions_tensor)
            batch_old_log_probs.append(log_probs_tensor)

            # Process environment steps
            new_states = []
            rewards = []
            dones = []

            # Step through environments (still need numpy for gym)
            for i, (e, state, action, done) in enumerate(zip(envs, current_states,
                                                             actions_tensor.cpu().numpy(),
                                                             is_done)):
                if done:
                    # If environment is already done, reuse last state
                    new_states.append(state)
                    rewards.append(0)
                    dones.append(True)
                    continue

                # Step environment
                next_state, reward, terminated, truncated, _ = e.step(action)
                done = terminated or truncated

                # Store results
                new_states.append(next_state)
                rewards.append(reward)
                dones.append(done)

            # Convert new data to tensors
            rewards_tensor = torch.FloatTensor(rewards).to(device)
            next_states_tensor = torch.FloatTensor(np.array(new_states)).to(device)
            dones_tensor = torch.BoolTensor(dones).to(device)

            # Store step results
            batch_rewards.append(rewards_tensor)
            batch_next_states.append(next_states_tensor)
            batch_dones.append(dones_tensor)

            # Update for next iteration
            current_states = new_states
            is_done = dones_tensor

            # Break if all environments are done
            if is_done.all():
                break

        # Stack tensors along time dimension (making each tensor shape [time_steps, batch_size, ...])
        trajectory_tensors = {
            'states': torch.stack(batch_states),
            'actions': torch.stack(batch_actions),
            'rewards': torch.stack(batch_rewards),
            'old_log_probs': torch.stack(batch_old_log_probs),
            'next_states': torch.stack(batch_next_states),
            'dones': torch.stack(batch_dones)
        }

        # Calculate advantages and returns as tensors
        advantages, returns = self.calculate_gae_tensors(
            trajectory_tensors['states'],
            trajectory_tensors['rewards'],
            trajectory_tensors['next_states'],
            trajectory_tensors['dones'],
            valueNet
        )

        # Add to trajectory tensors
        trajectory_tensors['advantages'] = advantages
        trajectory_tensors['returns'] = returns

        # need to rearrange the tensor for sampling
        for key in trajectory_tensors.keys():
            if trajectory_tensors[key].dim() == 3:
                trajectory_tensors[key] = trajectory_tensors[key].permute(1, 0, 2)
            if trajectory_tensors[key].dim() == 2:
                trajectory_tensors[key] = trajectory_tensors[key].permute(1, 0)

        return trajectory_tensors

    def random_sample(self, trajectory_tensors, batch_size=64):
        """
        Extract a batch of random samples from the trajectory data using efficient
        PyTorch sampling.

        Args:
            trajectory_tensors: Dictionary containing trajectory tensors
            batch_size: Number of samples to extract

        Returns:
            Dictionary containing batched tensors of randomly sampled trajectory data
        """
        dim_x = trajectory_tensors['states'].shape[0]
        dim_y = trajectory_tensors['states'].shape[1]

        # Calculate total number of samples available
        total_samples = dim_x * dim_y

        # Generate unique random indices
        flat_indices = torch.randperm(total_samples)[:batch_size]

        # Convert flat indices to 2D coordinates
        x_indices = flat_indices // dim_y
        y_indices = flat_indices % dim_y

        # Extract tensors using the sampled indices
        batch_tensors = {}
        for key, tensor in trajectory_tensors.items():
            # Initialize a list to store sampled values for each key
            sampled_values = []

            for x, y in zip(x_indices, y_indices):
                # Extract the value at the specified coordinates
                sampled_values.append(tensor[x, y])

            # Stack the sampled values to create a batch tensor
            batch_tensors[key] = torch.stack(sampled_values)

        return batch_tensors

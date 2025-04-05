# Replay Buffer
from collections import deque
import random
import gymnasium as gym
import numpy as np
import torch

from network import ValueNetwork, PolicyNetwork


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, trajectory):
        self.buffer.append(trajectory)

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        # Convert to tensors
        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        reward = torch.FloatTensor(np.array(reward).reshape(-1, 1)).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        done = torch.FloatTensor(np.array(done).reshape(-1, 1)).to(device)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def calculate_gae(self, trajectory, value_network, gamma=0.99, lambda_=0.95):
        """
        Calculate Generalized Advantage Estimation for the current trajectory
        :param trajectory:
        :param value_network:
        :param gamma:
        :param lambda_:
        :return: returns is the
        """

        # Extract all transitions from buffer as one batch
        states = []
        rewards = []
        next_states = []
        dones = []

        for state, _, reward, next_state, done in trajectory:
            states.append(state)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)

        # Get value estimates
        with torch.no_grad():
            values = value_network.forward(states)
            next_values = value_network.forward(next_states)

        # Calculate advantages
        deltas = rewards + gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(deltas)

        # Calculate GAE by working backwards through the buffer
        last_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1 or dones[t]:
                last_gae = 0
            advantages[t] = last_gae = deltas[t] + gamma * lambda_ * (1 - dones[t]) * last_gae

        # Calculate returns (used for training value function)
        returns = advantages + values

        return advantages, returns

    def generate_data(self, env: gym.Env, policyNet: PolicyNetwork, valueNet: ValueNetwork):
        """
        Generates training data by collecting 100 trajectories using the provided policy network,
        then clips all trajectories to match the length of the shortest trajectory for consistent batching.

        :param env: The gym environment to collect trajectories from
        :param policyNet: The policy network used to select actions
        :return: A list of clipped trajectories, where each trajectory is a list of
                 (state, action, reward, next_state, done) tuples
        """

        all_trajectories = []

        # Pre-create a batch of initial states to get actions for all environments at once
        batch_size = 10  # Process 10 environments at a time
        envs = [gym.make(env.unwrapped.spec.id) for _ in range(batch_size)]

        # Generate trajectories in batches
        for batch_start in range(0, 100, batch_size):
            # Initialize a batch of trajectories
            current_batch_trajectories = [[] for _ in range(batch_size)]
            batch_states = []
            batch_dones = [False] * batch_size

            # Reset all environments in the batch
            for i, e in enumerate(envs):
                state, _ = e.reset()
                batch_states.append(state)

            # Run until all episodes in the batch are done
            for _ in range(100):  # Max steps per episode
                # Convert states to tensor for policy network (batch processing)
                states_tensor = torch.FloatTensor(np.array(batch_states))

                # Get actions for all active environments at once
                with torch.no_grad():
                    actions = policyNet.get_action(states_tensor).numpy()

                # Step all environments and collect transitions
                new_batch_states = []
                for i, (e, state, action, done) in enumerate(zip(envs, batch_states, actions, batch_dones)):
                    if done:
                        new_batch_states.append(state)
                        continue

                    next_state, reward, terminated, truncated, _ = e.step(action)
                    done = terminated or truncated

                    # Store transition
                    transition = (state, action, reward, next_state, float(done))
                    current_batch_trajectories[i].append(transition)

                    # Update for next iteration
                    new_batch_states.append(next_state)
                    batch_dones[i] = done

                # Update states for next iteration
                batch_states = new_batch_states

                # Check if all environments are done
                if all(batch_dones):
                    break

            # Add completed trajectories to our collection
            all_trajectories.extend(current_batch_trajectories)

        # Find length of shortest trajectory
        min_trajectory_length = min(len(trajectory) for trajectory in all_trajectories)

        # Clip all trajectories to the minimum length
        clipped_trajectories = [trajectory[:min_trajectory_length] for trajectory in all_trajectories]

        # Store in replay buffer
        for trajectory in clipped_trajectories:
            # Generate the advantage and return used for training policy and value agent
            advantages, returns = self.calculate_gae(trajectory, valueNet)

            # Combine the original trajectory data with calculated advantages and returns
            # Each element becomes (state, action, reward, next_state, done, advantage, return)
            trajectory_with_gae = []
            for i in range(len(trajectory)):
                state, action, reward, next_state, done = trajectory[i]
                advantage = advantages[i].item()  # Convert tensor to scalar
                cur_return = returns[i].item()  # Convert tensor to scalar

                trajectory_with_gae.append((state, action, reward, next_state, done, advantage, cur_return))

            # Push the enhanced trajectory to the buffer
            self.buffer.append(trajectory_with_gae)
        return clipped_trajectories




# Create the environment
env = gym.make("HalfCheetah-v5", render_mode=None)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
pNet = PolicyNetwork(obs_dim, action_dim)
vNet = ValueNetwork(obs_dim)

########################### dummy code #####################
# Function to generate random actions within the action space
def my_policy(observation):
    # For HalfCheetah, actions are continuous values in a specific range
    # The action space is typically a Box with shape (6,) and values between -1 and 1
    return np.random.uniform(-1, 1, size=(6,))


# for testing
if __name__ == "__main__":
    import time

    # initialise buffer
    buffer = ReplayBuffer(capacity=10)

    # Measure execution time
    start_time = time.time()
    result = buffer.generate_data(env, pNet, vNet)
    end_time = time.time()

    # Print result and execution time
    print(result)
    print(f"Time taken to run generate_data: {end_time - start_time:.4f} seconds")

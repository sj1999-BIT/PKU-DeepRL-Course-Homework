# Replay Buffer
from tqdm import tqdm
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

    def generate_data(self, envs: [gym.Env], policyNet: PolicyNetwork, valueNet: ValueNetwork, batch_size=20):
        """
        Generates training data by collecting 100 trajectories using the provided policy network,
        storing all data in tensor form for faster processing.

        :param env: The gym environment to collect trajectories from
        :param policyNet: The policy network used to select actions
        :param valueNet: The value network for GAE calculation
        :return: A list of trajectories with tensors for states, actions, rewards, old_probs, advantages, and returns
        """

        device = next(policyNet.parameters()).device

        # Initialize storage for all trajectory tensors
        all_trajectory_tensors = []

        # Create tqdm progress bar for the iteration
        for batch_start in tqdm(range(batch_size), desc="Collecting replay experience"):
            # Initialize tensor lists for current batch
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_old_probs = []
            batch_next_states = []
            batch_dones = []

            # Initialize environment states
            current_states = []
            is_done = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for i, e in enumerate(envs):
                state, _ = e.reset()
                current_states.append(state)

            # Process steps for fixed episode length
            for step in range(100):  # Fixed step limit
                # Convert current states to tensor
                states_tensor = torch.FloatTensor(np.array(current_states)).to(device)

                # Get actions using policy network (stay in tensor form)
                actions_tensor, probs_tensor = policyNet.get_action(states_tensor)

                # Store current states
                batch_states.append(states_tensor)
                batch_actions.append(actions_tensor)
                batch_old_probs.append(probs_tensor)

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
                'old_probs': torch.stack(batch_old_probs),
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

            # Add to overall collection
            all_trajectory_tensors= trajectory_tensors
        # No need to clip trajectories as all tensors have same time dimension per batch
        # Store in buffer
        # self.tensor_buffer = all_trajectory_tensors

        # reorganise for
        for key in all_trajectory_tensors.keys():
            if all_trajectory_tensors[key].dim() == 3:
                all_trajectory_tensors[key] = all_trajectory_tensors[key].permute(1, 0, 2)
            if all_trajectory_tensors[key].dim() == 2:
                all_trajectory_tensors[key] = all_trajectory_tensors[key].permute(1, 0)


        return all_trajectory_tensors

    def random_sample(self, trajectory_tensors, batch_size=64):
        """
        Extract out a batch size random sample from the generated data
        :param trajectory_tensors:
        :param batch_size:
        :return:
        """

        dim_x = trajectory_tensors['states'].shape[0]
        dim_y = trajectory_tensors['states'].shape[1]

        list_indies = []

        while len(list_indies) < batch_size:
            cur_index = (int(random.random() * dim_x), int(random.random() * dim_y))
            if cur_index not in list_indies:
                list_indies.append(cur_index)

        # Extract tensors using the sampled indices
        batch_tensors = {}
        for key, tensor in trajectory_tensors.items():
            # Initialize a list to store sampled values for each key
            sampled_values = []

            for x, y in list_indies:
                # Extract the value at the specified coordinates
                sampled_values.append(tensor[x, y])

            # Stack the sampled values to create a batch tensor
            batch_tensors[key] = torch.stack(sampled_values)

        return batch_tensors



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

    # Pre-create batch of environments
    batch_size = 20  # Process 10 environments at a time
    envs = [gym.make(env.unwrapped.spec.id) for _ in range(batch_size)]

    # initialise buffer
    buffer = ReplayBuffer(capacity=10)

    # Measure execution time
    start_time = time.time()
    trajectory_tensor = buffer.generate_data(envs, pNet, vNet)
    end_time = time.time()


    print(pNet.get_loss())

    # Print result and execution time
    print(trajectory_tensor)
    print(f"Time taken to run generate_data: {end_time - start_time:.4f} seconds")

    print(buffer.random_sample(trajectory_tensor, 64))

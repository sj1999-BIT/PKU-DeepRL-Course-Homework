import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random
import math
import time

from network import ValueNetwork, PolicyNetwork
from replayBuffer import ReplayBuffer
from training import ppo_update

# Assuming device is defined earlier in the code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_ppo(env_name, num_episodes=1000, batch_size=64, lr_policy=3e-4, lr_value=1e-3, 
              gamma=0.99, buffer_capacity=100000, update_interval=0, clip_ratio=0.2,
              ppo_epochs=10, lambda_=0.95, save_path="ppo_model.pt"):
    """
    Main training function for Proximal Policy Optimization (PPO) algorithm
    
    Args:
        env_name: Name of the Gym environment
        num_episodes: Total number of episodes to train for
        batch_size: Size of batches for training
        lr_policy: Learning rate for policy network
        lr_value: Learning rate for value network
        gamma: Discount factor
        buffer_capacity: Capacity of the replay buffer
        update_interval: Number of episodes between PPO updates
        clip_ratio: PPO clipping parameter
        ppo_epochs: Number of epochs to optimize on the same data
        lambda_: GAE lambda parameter
        save_path: Path to save the trained model
        
    Returns:
        Trained policy and value networks
    """
    # Create environment
    env = gym.make(env_name)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize networks
    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)

    
    # Initialize optimizers
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr_policy)
    optimizer_value = optim.Adam(value_net.parameters(), lr=lr_value)
    
    # Initialize replay buffer
    buffer = ReplayBuffer(buffer_capacity)
    
    # Tracking variables
    episode_rewards = []
    avg_rewards = []
    best_avg_reward = -float('inf')
    
    start_time = time.time()
    
    # Training loop
    for episode in range(num_episodes):
        # Collect data using current policy
        buffer.generate_data(env, policy_net, value_net)
        
        # Update policy only every update_interval episodes
        # if (episode + 1) % update_interval == 0:
        # Sample batch_size number of trajectories from buffer
        batch_data = random.sample(buffer.buffer, batch_size)

        # Unpack the batch data
        states, actions, rewards, next_states, dones, advantages, returns = zip(*batch_data)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        advantages = torch.FloatTensor(np.array(advantages).reshape(-1, 1)).to(device)
        returns = torch.FloatTensor(np.array(returns).reshape(-1, 1)).to(device)

        # Get log probs of actions under old policy (for importance ratio)
        with torch.no_grad():
            log_probs_old, _ = policy_net.evaluate(states, actions)

        # Update policy and value network
        ppo_update(
            policy_net=policy_net,
            value_net=value_net,
            optimizer_policy=optimizer_policy,
            optimizer_value=optimizer_value,
            states=states,
            actions=actions,
            log_probs_old=log_probs_old,
            advantages=advantages,
            returns=returns,
            clip_ratio=clip_ratio,
            epochs=ppo_epochs
        )

        # Evaluate current policy
        if (episode + 1) % 10 == 0:
            eval_rewards = []
            for _ in range(5):  # Run 5 evaluation episodes
                state, _ = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        action = policy_net.get_action(state_tensor, deterministic=True).squeeze().cpu().numpy()
                    
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    state = next_state
                
                eval_rewards.append(episode_reward)
            
            avg_reward = np.mean(eval_rewards)
            avg_rewards.append(avg_reward)
            
            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                # Save both networks
                torch.save({
                    'policy_net': policy_net.state_dict(),
                    'value_net': value_net.state_dict()
                }, save_path)
            
            elapsed_time = time.time() - start_time
            print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Best Avg Reward: {best_avg_reward:.2f} | Time: {elapsed_time:.2f}s")
        
    print(f"Training completed! Best average reward: {best_avg_reward:.2f}")
    
    # Load best model
    checkpoint = torch.load(save_path)
    policy_net.load_state_dict(checkpoint['policy_net'])
    value_net.load_state_dict(checkpoint['value_net'])
    
    return policy_net, value_net


# Example usage
if __name__ == "__main__":
    # Example environment (continuous control task)
    env_name = "Pendulum-v1"  # Or any other continuous control environment
    
    # Train the agent
    policy_net, value_net = train_ppo(
        env_name=env_name,
        num_episodes=500,
        batch_size=64,
        lr_policy=3e-4,
        lr_value=1e-3,
        gamma=0.99,
        buffer_capacity=100000,
        update_interval=5,
        clip_ratio=0.2,
        ppo_epochs=10,
        lambda_=0.95,
        save_path="ppo_pendulum_model.pt"
    )
    
    # Test the trained policy
    env = gym.make(env_name, render_mode="human")
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Select action using the trained policy (deterministic mode for evaluation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = policy_net.get_action(state_tensor, deterministic=True).squeeze().cpu().numpy()
        
        # Take step in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
        
        # Render the environment
        env.render()
    
    print(f"Test episode finished with total reward: {total_reward}")
    env.close()
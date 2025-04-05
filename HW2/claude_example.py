import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import random
from collections import deque

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Neural Networks for SAC
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2 architecture (for twin Q trick)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2


# Gaussian Policy (Actor)
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(GaussianPolicy, self).__init__()

        self.max_action = max_action

        # Actor network for mean
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Mean and log_std layers
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)

        # Initialize weights of the mean layer
        nn.init.uniform_(self.mean_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_linear.bias, -3e-3, 3e-3)

        # Initialize weights of the log_std layer
        nn.init.uniform_(self.log_std_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_linear.bias, -3e-3, 3e-3)

    def forward(self, state):
        x = self.actor(state)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Create normal distribution and reparameterize
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick

        # Enforce action bounds (Tanh squashing)
        action = torch.tanh(x_t)

        # Calculate log probability with correction for Tanh squashing
        log_prob = normal.log_prob(x_t)

        # Correct log_prob for the Tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action * self.max_action, log_prob


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

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


# SAC Agent
class SAC:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, alpha=0.2,
                 lr=3e-4, batch_size=256, buffer_size=1000000, alpha_auto_tune=True):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.alpha_auto_tune = alpha_auto_tune

        # Initialize actor (policy)
        self.actor = GaussianPolicy(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Initialize critics (twin Q networks)
        self.critic = QNetwork(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Initialize target critics
        self.critic_target = QNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Freeze target critics
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Automatic entropy tuning
        if alpha_auto_tune:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)

        if evaluate:
            # When evaluating, use mean action (no noise)
            with torch.no_grad():
                mean, _ = self.actor(state)
                return torch.tanh(mean).cpu().numpy()[0]
        else:
            # During training, sample from the policy
            with torch.no_grad():
                action, _ = self.actor.sample(state)
                return action.cpu().numpy()[0]

    def update_parameters(self):
        # Sample from replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # Update critic
        with torch.no_grad():
            # Get next action and its log_prob
            next_action, next_log_prob = self.actor.sample(next_state)

            # Compute target Q values
            next_q1_target, next_q2_target = self.critic_target(next_state, next_action)
            next_q_target = torch.min(next_q1_target, next_q2_target)

            # Account for entropy in target Q
            next_q_target = next_q_target - self.alpha * next_log_prob

            # Compute target using Bellman equation
            target_q = reward + (1 - done) * self.gamma * next_q_target

        # Current Q estimates
        current_q1, current_q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        new_actions, log_prob = self.actor.sample(state)
        q1, q2 = self.critic(state, new_actions)
        min_q = torch.min(q1, q2)

        # Actor loss with entropy
        actor_loss = (self.alpha * log_prob - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (temperature parameter for entropy)
        if self.alpha_auto_tune:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # Soft update of target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# Training function for Half-Cheetah with SAC
def train_sac():
    env_name = "HalfCheetah-v5"
    env = gym.make(env_name, render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize SAC agent
    sac_agent = SAC(state_dim, action_dim, max_action)

    # Training hyperparameters
    max_episodes = 1000
    max_timesteps = 1000
    random_steps = 5000  # Collect random experience before training
    update_every = 50  # Update networks every n steps
    eval_every = 10  # Evaluate policy every n episodes

    # Tracking variables
    rewards = []
    episode_timesteps = 0
    total_timesteps = 0

    # Start with random exploration
    state, _ = env.reset()
    for _ in range(random_steps):
        # env.render()
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store in replay buffer
        sac_agent.replay_buffer.push(state, action, reward, next_state, float(done))

        state = next_state if not done else env.reset()[0]

    # Training loop
    print(f"initiate training")
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_timesteps = 0

        for t in range(max_timesteps):
            total_timesteps += 1
            episode_timesteps += 1

            # Select action
            action = sac_agent.select_action(state)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store in replay buffer
            sac_agent.replay_buffer.push(state, action, reward, next_state, float(done))

            state = next_state
            episode_reward += reward

            # Update parameters
            if total_timesteps % update_every == 0:
                for _ in range(update_every):
                    sac_agent.update_parameters()

            if done:
                break

        rewards.append(episode_reward)

        # Print episode stats
        if (episode + 1) % eval_every == 0:
            avg_reward = np.mean(rewards[-eval_every:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

    # Close the environment
    env.close()

    return sac_agent


if __name__ == "__main__":
    # Train SAC on Half-Cheetah
    agent = train_sac()
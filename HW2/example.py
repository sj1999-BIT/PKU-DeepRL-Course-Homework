import gymnasium as gym
import numpy as np

# Assuming these are defined elsewhere in your code
num_episodes = 1000
batch_size = 64

# Create the environment
env = gym.make("HalfCheetah-v5", render_mode="human")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


########################### dummy code #####################
# Function to generate random actions within the action space
def my_policy(observation):
    # For HalfCheetah, actions are continuous values in a specific range
    # The action space is typically a Box with shape (6,) and values between -1 and 1
    return np.random.uniform(-1, 1, size=(6,))



if __name__ == "__main__":
    # Training loop
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            env.render()
            action = my_policy(obs)
            next_obs, reward, done, _, _ = env.step(action)
            print(f"reward: {reward}, action: {next_obs}")
            # my_buffer.push(obs, next_obs, action, reward, done)
            obs = next_obs


        # Sample from replay buffer and update policy
        # batch = my_buffer.sample(batch_size)
        # my_policy.train(batch)
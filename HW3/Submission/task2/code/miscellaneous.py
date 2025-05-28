"""
Temporary storage of random functions for future possible uses.
"""

import gymnasium as gym
import numpy as np

from tqdm import tqdm


def simulate_env():
    # initiate new environment
    env = gym.make("Hopper-v5", render_mode="human")
    total_reward = 0
    state, _ = env.reset()

    action_dim = env.action_space.shape[0]

    try:
        # Add tqdm progress bar
        for step in tqdm(range(1000), desc="Simulation Progress"):
            # Random sample action from Box(-1.0, 1.0, (3,), float32)
            action = np.random.uniform(-1.0, 1.0, size=(action_dim,))

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

            # Check if episode is done
            if terminated or truncated:
                break

        print(f"total reward: {total_reward}")
        return total_reward
    finally:
        # Ensure environment is properly closed even if an exception occurs
        env.close()
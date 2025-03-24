"""
Dedicated to generating training data for DQN based on gaming experience.
"""
import gymnasium as gym
import numpy as np
import ale_py
from model import Q_agent
from collections import deque
import random
from tqdm import tqdm

# Register Atari Learning Environment environments with Gymnasium
gym.register_envs(ale_py)

# Create Pong environment without rendering for faster data collection
env = gym.make("ALE/Pong-v5", render_mode=None, obs_type="rgb")

def get_reward_data(q_agent, num_games=10):
    """
    Collect reward for playing 10 games
    :param q_agent: q_agent used to play the game
    :param num_games: accumulate number of games
    :return: get accumulated reward.
    """
    total_reward = 0

    pbar = tqdm(range(num_games), ncols=100)

    for i in pbar:

        # Initialize new game
        done = False
        obs, info = env.reset()  # BUG FIX: env.reset() returns (obs, info) in newer Gymnasium



        # Make a copy of current state for recording
        # BUG: This line creates an empty copy since current_state is empty at the start of each game
        # Should be moved to after filling current_state

        # maintain 4 consecutive frames in the queue for current state
        current_state = deque(maxlen=4)
        current_state.append(obs)


        while not done:
            # env.render()
            input_state = np.array(current_state)
            action, _ = q_agent.get_action(input_state)

            # Take action and observe result
            obs, reward, _, _, info = env.step(action)  # BUG FIX: Unpack 5 values
            if reward != 0:
                done = True

        total_reward += reward

        # update progress bar
        pbar.set_description(f"game {i} got accumulated reward {total_reward}")

    return total_reward



# def get_training_data(q_agent:Q_agent, num_games=10):
#     """
#     Collect training data for DQN by playing Pong games and storing state transitions.
#
#     Args:
#         num_games (int): Number of games to collect data from. Default is 10.
#
#     Returns:
#         list: Collection of transitions [current_state, action, reward, next_state]
#              where states are deques of 4 consecutive observations.
#     """
#
#     # Storage for collected transitions
#     data_collected = []
#
#     # Deques to maintain the 4 consecutive frames that form a state
#     current_state = deque(maxlen=4)
#     next_state = deque(maxlen=4)
#
#     # Create progress bar to track data collection
#     # BUG FIX: Use num_games instead of hardcoded 10
#     pbar = tqdm(range(num_games), ncols=100)
#
#     for i in pbar:
#         # Update progress bar description with current data size
#         pbar.set_description(f"Getting DQN training data, sized {len(data_collected)}")
#
#         # Initialize new game
#         done = False
#         obs, info = env.reset()  # BUG FIX: env.reset() returns (obs, info) in newer Gymnasium
#
#         # initialised initial state as 4 frames of all zeroes
#         for _ in range(4):
#             current_state.append(np.zeros_like(obs))
#
#         # Make a copy of current state for recording
#         # BUG: This line creates an empty copy since current_state is empty at the start of each game
#         # Should be moved to after filling current_state
#
#         while not done:
#             # # Fill current_state with initial observations if it's not full yet
#             # while len(current_state) < 4:
#             #     action = random.choice([0, 2, 3])  # NOOP, RIGHT, or LEFT
#             #     obs, reward, terminated, truncated, info = env.step(action)  # BUG FIX: Unpack 5 values
#             #     if reward != 0:
#             #         done = True
#             #         break
#             #     current_state.append(obs.copy())
#
#             # # Skip rest of loop if we couldn't fill current_state
#             # if len(current_state) < 4:
#             #     continue
#
#             # NOW is the correct time to save a copy of current_state
#             recorded_cur_state = current_state.copy()
#
#             # Get action from DQN agent
#             input_state = np.array(current_state)
#             recorded_action,_ = q_agent.get_action(input_state)
#             action = recorded_action
#
#             # Take the action and get the next observation
#             obs, reward, terminated, truncated, info = env.step(action)  # BUG FIX: Unpack 5 values
#             if reward != 0:
#                 done = True
#             recorded_reward = reward
#
#
#             # Continue building next_state until we have 4 frames
#             while len(next_state) < 4:
#                 # Start building the next state
#                 next_state.append(obs.copy())
#                 current_state.append(obs.copy())
#
#                 if done:  # Stop if game ends
#                     break
#
#                 # Get next action based on updated current_state
#                 action, q_val = q_agent.get_action(np.array(current_state))
#
#                 # Take action and observe result
#                 obs, reward, terminated, truncated, info = env.step(action)  # BUG FIX: Unpack 5 values
#                 done = terminated or truncated  # BUG FIX: Check both terminated and truncated
#
#             # Save transition if we gathered a complete next_state
#             if len(next_state) == 4:
#                 recorded_next_state = next_state.copy()
#                 data_collected.append([recorded_cur_state, action, recorded_reward, q_val, recorded_next_state])
#
#             # Reset states for next transition
#             # BUG FIX: Instead of clearing both states, we should keep current observations
#             # and shift the window, but this implementation resets completely
#             next_state = deque(maxlen=4)
#
#             # reinitiate current state
#             current_state = deque(maxlen=4)
#             for _ in range(4):
#                 current_state.append(np.zeros_like(obs))
#
#     return data_collected


import matplotlib.pyplot as plt
import os


def get_training_data(q_agent, num_games=10, save_images=False, save_dir="state_images"):
    """
    Collect training data for DQN by playing Pong games and storing state transitions.
    Optionally saves images of states showing the 4 frames side by side.

    Args:
        q_agent : The agent to use for action selection
        num_games (int): Number of games to collect data from. Default is 10.
        save_images (bool): Whether to save images of states. Default is False.
        save_dir (str): Directory to save state images. Default is "state_images".

    Returns:
        list: Collection of transitions [current_state, action, reward, next_state]
             where states are deques of 4 consecutive observations.
    """
    # Create save directory if needed
    if save_images and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Function to save state as an image
    def save_state_image(state, filename):
        """Save an image of a state with 4 frames rendered side by side."""
        if not save_images:
            return

        # Convert state from deque to numpy array if needed
        if isinstance(state, deque):
            state_array = np.array(list(state))
        else:
            state_array = np.array(state)

        # Create a figure with 4 subplots side by side
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Plot each frame
        for i in range(4):
            frame = state_array[i]
            # Normalize if needed (assuming values between 0-255 or 0-1)
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)

            # Display the frame
            axes[i].imshow(frame, cmap='gray')
            axes[i].set_title(f"Frame {i + 1}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close(fig)

    # Storage for collected transitions
    data_collected = []

    # Deques to maintain the 4 consecutive frames that form a state
    current_state = deque(maxlen=4)
    next_state = deque(maxlen=4)

    # Create progress bar to track data collection
    pbar = tqdm(range(num_games), ncols=100)

    for i in pbar:
        # Update progress bar description with current data size
        pbar.set_description(f"Getting DQN training data, sized {len(data_collected)}")

        # Initialize new game
        done = False
        obs, info = env.reset()  # BUG FIX: env.reset() returns (obs, info) in newer Gymnasium
        obs, _, _, _, info = env.step(0)  # random step as it seems that the first frame has RGB problem

        # Initialize initial state as 3 frames of all zeroes with the start frame
        for _ in range(3):
            current_state.append(np.zeros_like(obs))
        current_state.append(obs.copy())

        while not done:
            # Save a copy of current_state
            recorded_cur_state = current_state.copy()

            # Get action from DQN agent
            input_state = np.array(current_state)
            recorded_action, q_val = q_agent.get_action(input_state)
            action = recorded_action

            # Take the action and get the next observation
            obs, reward, terminated, truncated, info = env.step(action)  # BUG FIX: Unpack 5 values
            if reward != 0:
                done = True
            recorded_reward = reward

            # Continue building next_state until we have 4 frames
            while len(next_state) < 4:
                # Start building the next state
                next_state.append(obs.copy())
                current_state.append(obs.copy())

                if done:  # Stop if game ends
                    break

                # Get next action based on updated current_state
                action, q_val = q_agent.get_action(np.array(current_state))

                # Take action and observe result
                obs, reward, terminated, truncated, info = env.step(action)  # BUG FIX: Unpack 5 values
                if reward != 0:
                    done = True

            # Save transition if we gathered a complete next_state
            if len(next_state) == 4:
                recorded_next_state = next_state.copy()



                data_collected.append([recorded_cur_state, action, recorded_reward, q_val, recorded_next_state])

            # Reset states for next transition
            next_state = deque(maxlen=4)

    # Save images
    if save_images:
        # Counter for image filenames
        image_counter = 0

        for cur_state,_, _, _, next_state in data_collected:
            save_state_image(cur_state, f"state_{image_counter}_current.png")
            save_state_image(next_state, f"state_{image_counter}_next.png")
            image_counter += 1

    return data_collected


def validate_training_data(data_collected):
    """
    Validates the collected DQN training data and returns the percentage of valid transitions.

    Args:
        data_collected (list): List of [current_state, action, reward, next_state] transitions

    Returns:
        float: Percentage of valid transitions (0.0 to 100.0)
        str: Description of errors found, or success message
        list: Indices of invalid transitions
    """
    if not data_collected:
        return 0.0, "Error: No data collected", []

    valid_actions = [0, 2, 3]  # NOOP, RIGHT, LEFT for Pong
    invalid_indices = []
    error_messages = []

    for i, (current_state, action, reward, q_val, next_state) in enumerate(data_collected):
        is_valid = True

        # Check current_state
        if not isinstance(current_state, deque):
            error_messages.append(f"Transition {i}: current_state is not a deque")
            is_valid = False
        elif len(current_state) != 4:
            error_messages.append(f"Transition {i}: current_state has {len(current_state)} frames instead of 4")
            is_valid = False

        # Check action
        if action not in valid_actions:
            error_messages.append(f"Transition {i}: invalid action {action}")
            is_valid = False

        # Check reward
        if not isinstance(reward, (int, float)):
            error_messages.append(f"Transition {i}: reward is not a number")
            is_valid = False

        # Check next_state
        if not isinstance(next_state, deque):
            error_messages.append(f"Transition {i}: next_state is not a deque")
            is_valid = False
        elif len(next_state) != 4:
            error_messages.append(f"Transition {i}: next_state has {len(next_state)} frames instead of 4")
            is_valid = False

        # If basic structure checks pass, perform deeper validation
        if is_valid:
            # Check observation types and shapes
            for j, obs in enumerate(current_state):
                if not isinstance(obs, np.ndarray):
                    error_messages.append(f"Transition {i}, current_state frame {j}: observation is not a numpy array")
                    is_valid = False
                    break
                if len(obs.shape) != 3 or obs.shape[2] != 3:  # Assuming RGB observations
                    error_messages.append(
                        f"Transition {i}, current_state frame {j}: wrong observation shape {obs.shape}")
                    is_valid = False
                    break

            for j, obs in enumerate(next_state):
                if not isinstance(obs, np.ndarray):
                    error_messages.append(f"Transition {i}, next_state frame {j}: observation is not a numpy array")
                    is_valid = False
                    break
                if len(obs.shape) != 3 or obs.shape[2] != 3:  # Assuming RGB observations
                    error_messages.append(f"Transition {i}, next_state frame {j}: wrong observation shape {obs.shape}")
                    is_valid = False
                    break

            # Check for identical consecutive frames
            if is_valid:
                current_frames = list(current_state)
                for j in range(3):
                    if np.array_equal(current_frames[j], current_frames[j + 1]):
                        error_messages.append(
                            f"Transition {i}: consecutive frames {j} and {j + 1} in current_state are identical")
                        is_valid = False
                        break

                next_frames = list(next_state)
                for j in range(3):
                    if np.array_equal(next_frames[j], next_frames[j + 1]):
                        error_messages.append(
                            f"Transition {i}: consecutive frames {j} and {j + 1} in next_state are identical")
                        is_valid = False
                        break

                # Check transition between states
                if action != 0 and np.array_equal(current_frames[-1], next_frames[0]):
                    error_messages.append(
                        f"Transition {i}: last frame of current_state identical to first frame of next_state despite action {action}")
                    is_valid = False

        # If any check failed, add to invalid indices
        if not is_valid:
            invalid_indices.append(i)

    # Calculate percentage of valid transitions
    valid_count = len(data_collected) - len(invalid_indices)
    percentage_valid = (valid_count / len(data_collected)) * 100 if data_collected else 0.0

    # Prepare return message
    if not error_messages:
        message = f"All {len(data_collected)} transitions are valid (100%)"
    else:
        # Limit number of error messages shown to avoid overwhelming output
        sample_errors = error_messages[:5]
        message = f"Found {len(error_messages)} errors. Sample errors:\n" + "\n".join(sample_errors)
        if len(error_messages) > 5:
            message += f"\n... and {len(error_messages) - 5} more errors."

    return percentage_valid, message, invalid_indices


# Example usage:
if __name__ == "__main__":
    # Initialize DQN agent that will choose actions
    q_agent = Q_agent()
    q_agent.to(device='cuda')

    data = get_training_data(q_agent, num_games=1, save_images=True)

    # # Validate and get percentage
    percentage, message, invalid_indices = validate_training_data(data)
    #
    # print(f"Validation complete: {percentage:.2f}% of transitions are valid")
    # print(message)
    #
    # if invalid_indices:
    #     print(f"Invalid transitions at indices: {invalid_indices[:10]}" +
    #           ("..." if len(invalid_indices) > 10 else ""))

    get_reward_data(q_agent, num_games=10)
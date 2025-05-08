"""
For recording of video of agent playing the game.
"""

import gymnasium as gym
import numpy as np
import ale_py

from neural_network import PolicyNetwork
from collections import deque
import time
import cv2
import os
from tqdm import tqdm

# gym.register_envs(ale_py)



def record_gameplay(duration=90):
    """Record gameplay for specified duration in seconds"""

    # Create Pong environment with rendering for video recording
    env = gym.make("Hopper-v5", render_mode="rgb_array")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = "./Submission/task1/video.mp4"
    fps = 30
    frame_width, frame_height = 480, 480  # Default Atari frame size
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize agent
    pNet = PolicyNetwork(env)
    pNet.load_weights("./Policy_nn_weight.pth")

    # Setup timers
    start_time = time.time()
    end_time = start_time + duration

    # Game loop
    for step in tqdm(range(duration), desc="rendering"):
        # Reset environment for new game if needed
        current_state, info = env.reset()


        done = False

        # Episode loop
        while not done and time.time() < end_time:
            # Render and record frame
            frame = env.render()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Get action from agent
            input_state = np.array(current_state)
            action_tensor, _ = pNet.get_action(input_state)

            # Take action
            current_state, reward, terminated, truncated, info = env.step(action_tensor.numpy())

            # Check if done
            done = terminated or truncated

            # # Optional: add small delay to control frame rate
            # time.sleep(1/fps)

    # Release resources
    video_writer.release()
    print(f"Video saved to {os.path.abspath(video_path)}")

if __name__ == "__main__":
    record_gameplay(duration=20)  # Record for 90 seconds
"""
For recording of video of agent playing the game.
"""

import gymnasium as gym
import numpy as np
import ale_py
from model import Q_agent, QAgentWithEpilsonAndMoreDepth
from collections import deque
import time
import cv2
import os

gym.register_envs(ale_py)

# Create Pong environment with rendering for video recording
env = gym.make("ALE/Pong-v5", render_mode="rgb_array", obs_type="rgb")

def record_gameplay(duration=90):
    """Record gameplay for specified duration in seconds"""

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = "pong_gameplay.mp4"
    fps = 30
    frame_width, frame_height = 160, 210  # Default Atari frame size
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize agent
    q_agent = QAgentWithEpilsonAndMoreDepth("naive/target_agent_weights.pth")
    q_agent.to(device='cuda')

    # Setup timers
    start_time = time.time()
    end_time = start_time + duration

    # Game loop
    while time.time() < end_time:
        # Reset environment for new game if needed
        obs, info = env.reset()

        # Initialize state queue
        current_state = deque(maxlen=4)
        for _ in range(4):  # Fill with initial observation to have 4 frames
            current_state.append(obs)

        done = False

        # Episode loop
        while not done and time.time() < end_time:
            # Render and record frame
            frame = env.render()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Get action from agent
            input_state = np.array(current_state)
            action, _ = q_agent.get_action(input_state)

            # Take action
            obs, reward, terminated, truncated, info = env.step(action)

            # Update state
            current_state.append(obs)

            # Check if done
            done = terminated or truncated

            # Optional: add small delay to control frame rate
            time.sleep(1/fps)

    # Release resources
    video_writer.release()
    print(f"Video saved to {os.path.abspath(video_path)}")

if __name__ == "__main__":
    record_gameplay(duration=90)  # Record for 90 seconds
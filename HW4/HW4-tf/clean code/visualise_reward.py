from random import random

import cv2
import numpy as np
from pettingzoo.mpe import simple_spread_v3

from dataKeys import AGENT_ZERO, AGENT_ONE, AGENT_TWO
from environment_functions import calculate_reward


def generate_action_dict():
    return {
        AGENT_ZERO: int(random() * 5),
        AGENT_ONE: int(random() * 5),
        AGENT_TWO: int(random() * 5),
    }


def add_reward_display(frame, overlap_reward, collision_punish, step):
    """Extend frame with white space on right and display reward."""
    # Original frame dimensions
    height, width, channels = frame.shape

    # Create a wider frame with white space on the right
    # Add 200 pixels for the reward display
    extended_width = width + 200
    extended_frame = np.ones((height, extended_width, channels), dtype=np.uint8) * 255

    # Copy the original frame to the left side
    extended_frame[:, :width, :] = frame

    # Add text with reward information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(extended_frame, f"Step: {step}", (width + 10, 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(extended_frame, f"overlap: {overlap_reward}\n", (width + 10, 70), font, 0.7, (0, 0, 0), 2)
    cv2.putText(extended_frame, f"collision: {collision_punish}", (width + 10, 110), font, 0.7, (0, 0, 0), 2)


    # Add a dividing line
    cv2.line(extended_frame, (width, 0), (width, height), (200, 200, 200), 2)

    return extended_frame

env = simple_spread_v3.parallel_env(render_mode='rgb_array')

COLLISION_THRESHOLD = 0.6
OVERLAP_THRESHOLD = 0.2

if __name__ == "__main__":

    # observations is a dictionary og agent_id to their observations
    cur_observation_dict, infos = env.reset()
    initial_frame = env.render()
    height, width, _ = initial_frame.shape

    # Calculate the size of the enlarged frame
    enlarged_width = (width + 200)  # Original + reward panel, all x4
    enlarged_height = height

    # Create a resizable window (but with initial size set appropriately)
    window_name = "Simple Spread Environment with Rewards (4x)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, enlarged_width, enlarged_height)

    step = 0

    # Initialize video writer
    frames = []

    while env.agents:

        cur_observation_dict, rewards, terminations, truncations, infos = env.step(generate_action_dict())

        frame = env.render()

        cal_reward, overlap_r, collision_p = calculate_reward(cur_observation_dict)

        step += 1

        extended_frame = add_reward_display(frame, overlap_r, collision_p, step)

        enlarged_extended_frame = cv2.resize(extended_frame, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)

        # Display the frame
        cv2.imshow(window_name, cv2.cvtColor(enlarged_extended_frame, cv2.COLOR_RGB2BGR))

        cv2.waitKey(150)


    env.close()

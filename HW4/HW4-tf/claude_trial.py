import math
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from dataKeys import AGENT_ZERO,AGENT_ONE, AGENT_TWO

from random import random

import cv2
import os



def get_dist(x, y):
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))


def break_local_observation_to_arr(local_observation_arr):
    """
    [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]
    :param local_observation_arr:
    :return:
    """
    dict = {}
    dict['cur_vel'] = local_observation_arr[:2]
    dict['cur_pos'] = local_observation_arr[2:4]
    dict['landmark_rel_positions'] = local_observation_arr[4:10]
    dict['other_agent_rel_positions'] = local_observation_arr[10:14]
    dict['communication'] = local_observation_arr[14:]

    return dict


def get_agent_dist_punish(observation_dict: dict):
    """
    """
    agent_pos = []
    for agent_id, local_observation in observation_dict.items():
        agent_pos.append(local_observation[2:4])

    agent_dist = []
    for i in range(3):
        for j in range(i+1, 3):
            agent_dist.append(get_dist(agent_pos[i][0] - agent_pos[j][0], agent_pos[i][1] - agent_pos[j][1]))

    punish = 0

    # print(f"agent dist {agent_dist}")

    for cur_dist in agent_dist:
        if cur_dist >= 1:
            continue
        punish += (1 - cur_dist) / 3

        if cur_dist < COLLISION_THRESHOLD:
            # print(f"punish for collision at dist {cur_dist}")
            punish += 1

    return punish

def get_landmark_dist_arr(landmark_rel_pos_arr):
    result = []
    for i in range(3):
        x_index = i * 2
        cur_landmark_rel_pos = landmark_rel_pos_arr[x_index:x_index+2]
        result.append(get_dist(cur_landmark_rel_pos[0], cur_landmark_rel_pos[1]))

    return result


def get_landmark_dist_reward(observation_dict: dict):

    dist_mat = []
    for agent_id, local_observation in observation_dict.items():
        dist_mat.append(get_landmark_dist_arr(local_observation[4:10]))

    dist_mat = np.array(dist_mat)

    reward = 0

    mini_dist_arr = []

    for i in range(3):
        cur_landmark_dist = dist_mat[:, i]
        mini_dist = np.min(cur_landmark_dist)
        mini_dist_arr.append(mini_dist)

        if mini_dist >= 1:
            continue

        reward += (1-mini_dist) / 3

        if mini_dist < OVERLAP_THRESHOLD:
            # print(f"reward for overlap at dist {mini_dist}")
            reward += 1

    # print(f"mini distance landmark to agent {mini_dist_arr}")

    return reward


def calculate_reward(observation_dict: dict):

    overlap_reward = get_landmark_dist_reward(observation_dict)
    collision_punish = get_agent_dist_punish(observation_dict)

    reward = overlap_reward - collision_punish

    return reward, overlap_reward, collision_punish

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

        # frames.append(frame)



    env.close()

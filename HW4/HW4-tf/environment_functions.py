import math
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from dataKeys import AGENT_ZERO,AGENT_ONE, AGENT_TWO

from random import random



def get_dist(x, y):
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))

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

env = simple_spread_v3.parallel_env(render_mode='human')

COLLISION_THRESHOLD = 0.6
OVERLAP_THRESHOLD = 0.2

if __name__ == "__main__":

    # observations is a dictionary og agent_id to their observations
    cur_observation_dict, infos = env.reset()

    while env.agents:
        cur_observation_dict, rewards, terminations, truncations, infos = env.step(generate_action_dict())
        print(calculate_reward(cur_observation_dict))

    env.close()

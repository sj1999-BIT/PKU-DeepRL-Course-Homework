import math
import os.path

import numpy as np
from pettingzoo.mpe import simple_spread_v3
from dataKeys import AGENT_ZERO,AGENT_ONE, AGENT_TWO
env = simple_spread_v3.parallel_env(render_mode=None)
from qmix import QMIX
from environment_functions import calculate_reward
from data import append_values_to_file

from tqdm import tqdm

# landmark needs to be this close to be considered covered
THRESHOLD = 0.1


from qmix import QMIX

env.reset()
action = 0

qmix_agent = QMIX(file_path="./06_normalized_rewards/")



if __name__ == "__main__":



    for i in tqdm(range(32), desc="collecting data for correlation"):
        # observations is a dictionary og agent_id to their observations
        cur_observation_dict, infos = env.reset()

        while env.agents:
            actions_dict, q_total = qmix_agent.get_actions_with_q_total(cur_observation_dict, epilson=1)
            cur_observation_dict, rewards, terminations, truncations, infos = env.step(actions_dict)
            # reward, _, _ = calculate_reward(cur_observation_dict)
            reward = 0
            for id, cur_reward in rewards.items():
                reward += cur_reward
            append_values_to_file(reward, "./correlation_reward.txt")
            append_values_to_file(q_total.numpy()[0][0], "./correlation_q_total.txt")



    env.close()
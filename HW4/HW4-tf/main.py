import math
import os.path

import numpy as np
from pettingzoo.mpe import simple_spread_v3
from dataKeys import AGENT_ZERO,AGENT_ONE, AGENT_TWO
env = simple_spread_v3.parallel_env(render_mode='human')
from qmix import QMIX
from environment_functions import calculate_reward
from data import append_values_to_file

"""
5-7: install new environment that can run tensorflow with gpu, but there is problem with cuda, may need to downgrade it to 11.2.
"""

# landmark needs to be this close to be considered covered
THRESHOLD = 0.1


from qmix import QMIX

env.reset()
action = 0

qmix_agent = QMIX(file_path="./08_5epoch/")



if __name__ == "__main__":

    # observations is a dictionary og agent_id to their observations
    cur_observation_dict, infos = env.reset()

    while env.agents:
        actions_dict, q_total = qmix_agent.get_actions_with_q_total(cur_observation_dict, epilson=0)
        cur_observation_dict, rewards, terminations, truncations, infos = env.step(actions_dict)
        reward = calculate_reward(cur_observation_dict)
        print(f"reward: {reward}, q_total: {q_total.numpy()}")

    env.close()
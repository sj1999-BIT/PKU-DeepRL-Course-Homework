from pettingzoo.mpe import simple_spread_v3
from dataKeys import AGENT_ZERO,AGENT_ONE, AGENT_TWO
env = simple_spread_v3.parallel_env(render_mode='human')

"""
5-7: install new environment that can run tensorflow with gpu, but there is problem with cuda, may need to downgrade it to 11.2.
"""

from qmix import QMIX

env.reset()
action = 0

qmix_agent = QMIX()

if __name__ == "__main__":

    # observations is a dictionary og agent_id to their observations
    observation_dict, infos = env.reset()

    while env.agents:

        actions_dict = qmix_agent.get_actions(observation_dict)
        print(actions_dict)
        observations, rewards, terminations, truncations, infos = env.step(actions_dict)


    env.close()
from pettingzoo.mpe import simple_spread_v3
from tqdm import tqdm

from environment_functions import calculate_reward
from data import append_values_to_file
from qmix import QMIX


def collect_reward_q_total_correlation_data(weights_file_path):
    env = simple_spread_v3.parallel_env(render_mode=None)

    qmix_agent = QMIX(file_path=weights_file_path)

    for i in tqdm(range(32), desc="collecting data for correlation"):
        # observations is a dictionary og agent_id to their observations
        cur_observation_dict, infos = env.reset()

        while env.agents:
            actions_dict, q_total = qmix_agent.get_actions_with_q_total(cur_observation_dict, epilson=1)
            cur_observation_dict, rewards, terminations, truncations, infos = env.step(actions_dict)
            reward, _, _ = calculate_reward(cur_observation_dict)
            append_values_to_file(reward, "./correlation_reward.txt")
            append_values_to_file(q_total.numpy()[0][0], "./correlation_q_total.txt")
    env.close()


if __name__ == "__main__":
    collect_reward_q_total_correlation_data("./06_normalized_rewards/")

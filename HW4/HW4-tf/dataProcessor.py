from pettingzoo.mpe import simple_spread_v3
from dataKeys import AGENT_ZERO, AGENT_ONE, AGENT_TWO, ACTIONS, STATES, NEXT_STATES, Q_TOTALS, REWARDS
from tqdm import tqdm
from qmix import QMIX, convert_dict_to_arr
from tf_neural_network import AgentNetwork

import numpy as np
import tensorflow as tf

import torch


class DataProcessor:
    def __init__(self, state_dim=18, maxSize=4000, env=None):

        if env is None:
            self.env = simple_spread_v3.parallel_env(render_mode=None)
        else:
            self.env = env

        self.fullEpisodeBuffer = []

        self.maxBufferSize = maxSize

    def initiate_empty_buffer(self):
        """
        Generate empty buffer for collection of data for a single episode
        :return: dict containing essentials for training
        """

        return {
            STATES: [],
            NEXT_STATES: [],
            REWARDS: [],
            ACTIONS: []
        }

    def update_transition_buffer(self):
        """
        Maintain max number of full episodes stored in the buffer.

        Args:
            result (dict): Dictionary containing new data to add to the buffer
            max_buffer_size (int): Maximum allowed buffer size for each key
        """

        # Then check if any key exceeds the maximum size
        current_size = len(self.fullEpisodeBuffer)
        if current_size > self.maxBufferSize:
            # Generate random indices to keep
            indices_to_keep = torch.randperm(current_size)[:self.maxBufferSize]

            # Apply the same indices to all keys to keep the data aligned
            self.fullEpisodeBuffer = self.fullEpisodeBuffer[indices_to_keep]

    def collect_single_full_episode(self, qmix_agent: QMIX):
        """
        collect a full episode of qmix agent playing in the environment.
        :param qmix_agent:
        :return:
        """
        single_episode_buffer = self.initiate_empty_buffer()

        # initiate new environment
        observation_dict, infos = self.env.reset()

        while self.env.agents:
            # record current state
            single_episode_buffer[STATES].append(convert_dict_to_arr(observation_dict))

            # get actions for previous state
            actions_dict = qmix_agent.get_actions(observation_dict)

            # record action taken for current state
            single_episode_buffer[ACTIONS].append(convert_dict_to_arr(actions_dict))

            observation_dict, reward_dict, terminations, truncations, infos = self.env.step(actions_dict)

            # record next state
            single_episode_buffer[NEXT_STATES].append(convert_dict_to_arr(observation_dict))

            # record reward
            single_episode_buffer[REWARDS].append(sum(convert_dict_to_arr(reward_dict)))

        single_episode_buffer[STATES] = np.array(single_episode_buffer[STATES])
        single_episode_buffer[ACTIONS] = np.array(single_episode_buffer[ACTIONS])
        single_episode_buffer[NEXT_STATES] = np.array(single_episode_buffer[NEXT_STATES])
        single_episode_buffer[REWARDS] = np.array(single_episode_buffer[REWARDS])

        return single_episode_buffer

    def collect_data(self, qmix_agent: QMIX, data_size=400):
        total_reward = 0
        for step in tqdm(range(data_size), desc="collecting full episode data"):
            single_episode_buffer = self.collect_single_full_episode(qmix_agent)
            total_reward += np.sum(single_episode_buffer[REWARDS])
            self.fullEpisodeBuffer.append(single_episode_buffer)
        return total_reward / data_size

    def random_sample(self, batch_size=64):
        """
        Select a random sample of size of the episode
        :param batch_size:
        :return:
        """
        # Randomly select 10 indices without replacement
        indices = np.random.choice(len(self.fullEpisodeBuffer), size=batch_size, replace=False)

        # Create new array with selected values
        new_array = np.array([self.fullEpisodeBuffer[i] for i in indices])
        # Or, more directly: new_array = array[indices]

        return new_array






if __name__ == "__main__":
    dp = DataProcessor()

    test_qmix_agent = QMIX()

    dp.collect_data(test_qmix_agent, data_size=32)

    sample_episode_arr = dp.random_sample(batch_size=32)

    test_qmix_agent.training(sample_episode_arr, batch_size=32)

    # single_agent = AgentNetwork()
    #
    # single_agent.call(episode[STATES][:, :18], episode[ACTIONS][:, :1])
    #
    # hidden_state_arr_batch = single_agent.hidden_state
    # single_agent.hidden_state = None
    # for cur_state, prev_action in zip(episode[STATES], episode[ACTIONS]):
    #     single_agent.call(cur_state[:18], prev_action[:1])
    #
    # hidden_state_arr_loop = single_agent.hidden_state
    #
    # # Check if tensors are exactly equal
    # print(tf.reduce_mean(tf.abs(hidden_state_arr_loop-hidden_state_arr_batch)))
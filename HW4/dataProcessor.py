from pettingzoo.mpe import simple_spread_v3
from dataKeys import AGENT_ZERO, AGENT_ONE, AGENT_TWO, ACTIONS, STATES, NEXT_STATES, Q_TOTALS, REWARD
from neural_networks import AgentNetwork, MixNetwork
from tqdm import tqdm

import torch

import numpy as np

"""
Due to the nature of using RNN in QMIX, data collection needs to be in full episodes (off line).

instead of collecting independent transitions, needs to be a full trajectory.
"""

class DataProcessor:
    def __init__(self, state_dim=18, maxSize=10000, env=None):

        if env is None:
            self.env = simple_spread_v3.parallel_env(render_mode=None)
        else:
            self.env = env

        # # for collection of data
        # self.transitionBuffer = {
        #     STATES: torch.empty((0, state_dim * 3), dtype=torch.float32),
        #     NEXT_STATES: torch.empty((0, state_dim * 3), dtype=torch.float32),
        #     REWARD: torch.empty((0,), dtype=torch.float32),
        #     Q_TOTALS: torch.empty((0, 1), dtype=torch.float32)
        # }
        #
        # an array containing full episodes of transitions
        self.fullEpisodeBuffer = []

        self.maxBufferSize = maxSize

    def initiate_empty_buffer(self):
        """
        Generate empty buffer for collection of data for a single episode
        :return: dict containing essentials for training
        """

        return {
            STATES: torch.empty((0, 18 * 3), dtype=torch.float32),
            NEXT_STATES: torch.empty((0, 18 * 3), dtype=torch.float32),
            REWARD: torch.empty((0,), dtype=torch.float32),
            ACTIONS: torch.empty((3,), dtype=torch.float32),
            Q_TOTALS: torch.empty((0, 1), dtype=torch.float32)
        }

    def update_transition_buffer(self):
        """
        Update all keys in the transition buffer and maintain the maximum length for each key.

        Args:
            result (dict): Dictionary containing new data to add to the buffer
            max_buffer_size (int): Maximum allowed buffer size for each key
        """

        # Then check if any key exceeds the maximum size
        current_size = self.transitionBuffer[list(self.transitionBuffer.keys())[0]].shape[0]
        if current_size > self.maxBufferSize:
            # Generate random indices to keep
            indices_to_keep = torch.randperm(current_size)[:self.maxBufferSize]

            # Apply the same indices to all keys to keep the data aligned
            for key in self.transitionBuffer.keys():
                self.transitionBuffer[key] = self.transitionBuffer[key][indices_to_keep]

    def collect_data(self, agent_dict: {}, mixNet: MixNetwork, data_size=400):
        for step in tqdm(range(data_size), desc="collecting full episode data"):
            self.fullEpisodeBuffer.append(self.collect_single_full_episode(agent_dict, mixNet))

    def collect_single_full_episode(self, agent_dict: {}, mixNet: MixNetwork):
        """
        Collect a single dict of transitions for a single episode
        :param mixNet:
        :return:
        """

        # Initialize lists to collect transitions
        states_list = []
        next_states_list = []
        rewards_list = []
        actions_list = []
        q_totals_list = []

        # simple helper function to help convert dict to array
        def convert_dict_to_arr(input_dict):
            result = []
            for key in input_dict.keys():
                if isinstance(input_dict[key], (list, np.ndarray)):
                    result.extend(input_dict[key])
                else:
                    result.append(input_dict[key])

            return result

        cur_episode_buffer = self.initiate_empty_buffer()

        prev_action_dict = {
            AGENT_ZERO: 0,
            AGENT_ONE: 0,
            AGENT_TWO: 0
        }

        actions_dict = {
            AGENT_ZERO: 0,
            AGENT_ONE: 0,
            AGENT_TWO: 0
        }


        # observations is a dictionary og agent_id to their observations
        observation_dict, infos = self.env.reset()

        while self.env.agents:

            # for inputs to mix Network
            input_Q_val = []


            for agent_id in actions_dict.keys():
                cur_action, cur_Q_value = agent_dict[agent_id].select_action(observation_dict[agent_id],
                                                                             [prev_action_dict[agent_id], ])
                actions_dict[agent_id] = cur_action

                # store Q_val
                input_Q_val.append(cur_Q_value)

            # get the current global state and save it
            cur_state_arr = convert_dict_to_arr(observation_dict)
            states_list.append(torch.tensor(cur_state_arr, dtype=torch.float32))

            # to collect actions
            action_arr = convert_dict_to_arr(actions_dict)
            actions_list.append(torch.tensor(action_arr, dtype=torch.float32))


            q_totals_list.append(mixNet.forward(input_Q_val, cur_state_arr))

            prev_action_dict = actions_dict
            observation_dict, reward_dict, terminations, truncations, infos = self.env.step(actions_dict)

            # get the current global state
            next_state_arr = convert_dict_to_arr(observation_dict)
            next_states_list.append(torch.tensor(next_state_arr, dtype=torch.float32))
            rewards_list.append(torch.tensor(sum(convert_dict_to_arr(reward_dict)), dtype=torch.float32))

        # Convert lists to tensors all at once
        states_batch = torch.stack(states_list)
        next_states_batch = torch.stack(next_states_list)
        rewards_batch = torch.stack(rewards_list)
        actions_batch = torch.stack(actions_list)
        q_totals_batch = torch.stack(q_totals_list)

        # a full episode
        # we dont record the q_values here as we will generate it later for training
        episode_dict = {
            'STATES': states_batch,
            'NEXT_STATES': next_states_batch,
            'REWARD': rewards_batch,
            'ACTIONS': actions_batch,
            # 'Q_TOTALS': q_totals_batch
        }

        return episode_dict

    def random_sample(self, batch_size=64):
        """

        :param batch_size:
        :return:
        """
        # Randomly select 10 indices without replacement
        indices = np.random.choice(len(self.fullEpisodeBuffer), size=batch_size, replace=False)

        # Create new array with selected values
        new_array = np.array([self.fullEpisodeBuffer[i] for i in indices])
        # Or, more directly: new_array = array[indices]

        return new_array

    def generate_q_total(self, episode_dict):
        """

        :param episode_dict:
        :return:
        """




if __name__ == "__main__":
    test_agent_dict = {
        AGENT_ZERO: AgentNetwork(0),
        AGENT_ONE: AgentNetwork(1),
        AGENT_TWO: AgentNetwork(2)
    }

    test_mixNet = MixNetwork()

    dp = DataProcessor()

    dp.collect_data(test_agent_dict, test_mixNet, data_size=1)

    # dp.collect_single_full_episode(test_agent_dict, test_mixNet)
    sample = dp.random_sample(batch_size=1)

    print(sample)

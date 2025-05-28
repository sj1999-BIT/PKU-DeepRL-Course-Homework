import os.path

import numpy as np

from dataKeys import AGENT_ZERO, AGENT_ONE, AGENT_TWO, STATES, Q_VALS
from helper import convert_dict_to_arr
from tf_neural_network import AgentNetwork, MixNetwork


class QMIX:
    def __init__(self, file_path=None):

        # store the hidden state for each agent
        self.agent_hidden_state_dict = {
            AGENT_ZERO: None,
            AGENT_ONE: None,
            AGENT_TWO: None
        }

        # initialise the agent network, all three agents share this network
        self.agent = AgentNetwork()

        # to record the current agent's recorded past actions
        self.prev_action_dict = {
            AGENT_ZERO: 0,
            AGENT_ONE: 0,
            AGENT_TWO: 0
        }

        self.filepath = file_path

        # initiate a mixAgent
        self.mixAgent = MixNetwork()

        # load model
        if file_path is not None and os.path.exists(file_path):
            dummy_obs = np.zeros(18)
            dummy_prev_action = np.zeros(1)

            self.agent(dummy_obs, dummy_prev_action)
            self.agent.load_model(file_path)

            dummy_q_vals = np.zeros(3)
            dummy_global_states = np.zeros(54)
            dummy_dict = {Q_VALS:dummy_q_vals, STATES:dummy_global_states}

            self.mixAgent(dummy_dict)
            self.mixAgent.load_model(file_path)




    def get_actions(self, observation_dict, epilson):
        """
        observation_dict is an dictionary mapping each agent id to their respective local observation
        :param observation_dict:
        :return: a dict of size 3 mapping agent id to their actions for each observation.
        """
        actions_dict = {}
        for agent_id in self.agent_hidden_state_dict.keys():
            cur_action, cur_Q_value = self.agent.select_action(
                observation_dict[agent_id],
                [self.prev_action_dict[agent_id], ],
                self.agent_hidden_state_dict[agent_id],
                epsilon=epilson
            )

            # update hidden state
            self.agent_hidden_state_dict[agent_id] = self.agent.hidden_state

            actions_dict[agent_id] = cur_action
            self.prev_action_dict[agent_id] = cur_action

        return actions_dict

    def get_actions_with_q_total(self, observation_dict, epilson):

        actions_dict = {}
        mix_input_dict = {}

        mix_input_dict[STATES] = convert_dict_to_arr(observation_dict)
        mix_input_dict[Q_VALS] = []


        for agent_id in self.agent_hidden_state_dict.keys():
            cur_action, cur_Q_value = self.agent.select_action(
                observation_dict[agent_id],
                [self.prev_action_dict[agent_id], ],
                self.agent_hidden_state_dict[agent_id],
                epsilon=epilson
            )

            mix_input_dict[Q_VALS].append(cur_Q_value)

            # update hidden state
            self.agent_hidden_state_dict[agent_id] = self.agent.hidden_state

            actions_dict[agent_id] = cur_action
            self.prev_action_dict[agent_id] = cur_action

        q_total = self.mixAgent.call(mix_input_dict)

        return actions_dict, q_total

    def save_weights(self):
        self.agent.save_model(filepath=self.filepath)












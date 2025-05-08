import os.path
import random

import numpy as np
import tensorflow as tf

from dataKeys import AGENT_ZERO,AGENT_ONE, AGENT_TWO, ACTIONS, STATES, Q_VALS, REWARDS, AGENT_WEIGHT_NAME
from tf_neural_network import AgentNetwork, MixNetwork
from typing import Dict
from tqdm import tqdm



"""
An assemble class the contains all the agents and mix network to make an assembled action.

Modify: each time we run a full episode we load a new agent network from a given pathway
"""


def convert_arr_to_dict(input_arr: np.ndarray):
    """
    Need to break the inputs into separate inputs for each of the agents
    :param input_arr:
    :return: dictionary mapping agent id to their inputs
    """
    result_dict = {}
    if len(input_arr.shape) == 1:
        # a single timestep.
        division = int(len(input_arr) / 3)
        result_dict[AGENT_ZERO] = input_arr[:division]
        result_dict[AGENT_ONE] = input_arr[division:division * 2]
        result_dict[AGENT_TWO] = input_arr[division * 2:]

    elif len(input_arr.shape) == 2:
        # a full episode of timesteps
        division = int(len(input_arr[0]) / 3)
        result_dict[AGENT_ZERO] = input_arr[:, :division]
        result_dict[AGENT_ONE] = input_arr[:, division:division * 2]
        result_dict[AGENT_TWO] = input_arr[:, division * 2:]

    else:
        raise Exception(f"invalid input of shape {len(input_arr.shape)}")

    return result_dict


def convert_dict_to_arr(input_dict):
    result = []
    for key in input_dict.keys():
        if isinstance(input_dict[key], (list, np.ndarray)):
            result.extend(input_dict[key])
        else:
            result.append(input_dict[key])
    return result


def random_sample(target_arr, sample_size):
    # Randomly select 10 indices without replacement
    indices = np.random.choice(len(target_arr), size=sample_size, replace=False)

    if isinstance(target_arr, tf.Tensor):
        return tf.gather(target_arr, indices)

    # Create new array with selected values
    return target_arr[indices]


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

        if file_path is not None and os.path.exists(file_path):
            dummy_obs = np.zeros(18)
            dummy_prev_action = np.zeros(1)

            self.agent(dummy_obs, dummy_prev_action)
            self.agent.load_model(file_path)

        # to record the current agent's recorded past actions
        self.prev_action_dict = {
            AGENT_ZERO: 0,
            AGENT_ONE: 0,
            AGENT_TWO: 0
        }

        self.filepath = file_path

    # def reset(self):
    #     """
    #     clear hidden state for each agent and previous actions
    #     :return:
    #     """
    #     for agent_id, agent in self.agent_dict.items():
    #         agent.hidden_state = None
    #
    #     # past action back to 0
    #     self.prev_action_dict = {
    #         AGENT_ZERO: 0,
    #         AGENT_ONE: 0,
    #         AGENT_TWO: 0
    #     }

    # def save_models(self, filepath="./"):
    #     self.mixNet.save_weights(filepath=filepath)
    #
    #     # aim to save an averaged versionl
    #     agent_vars = self.agent_dict[AGENT_ZERO].trainable_variables

    def get_actions(self, observation_dict):
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
                self.agent_hidden_state_dict[agent_id]
            )

            # update hidden state
            self.agent_hidden_state_dict[agent_id] = self.agent.hidden_state

            actions_dict[agent_id] = cur_action
            self.prev_action_dict[agent_id] = cur_action

        return actions_dict

    def save_weights(self):
        self.agent.save_model(filepath=self.filepath)












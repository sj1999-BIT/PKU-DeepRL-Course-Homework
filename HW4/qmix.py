"""
Here we combine the policy agents and the mix neural network into a single class.

This is for convenience of generating q_totals
"""

from neural_networks import MixNetwork, AgentNetwork
from dataKeys import AGENT_ZERO, AGENT_ONE, AGENT_TWO

class QMIX:
    def __init__(self, agent_weights_path, mixNetwork_path):
        """
        Initiate new class
        :param agent_weights_path:
        :param mixNetwork_path:
        """

        self.agent_dict = {
            AGENT_ZERO: AgentNetwork(0),
            AGENT_ONE: AgentNetwork(1),
            AGENT_TWO: AgentNetwork(2)
        }

        for agent_id, agent in self.agent_dict.items():
            agent.load_weights(agent_weights_path)

        self.mixNet = MixNetwork()
        self.mixNet.load_weights(mixNetwork_path)

    def _re_initiate_hidden_layer(self):
        """
        Each agent neural network hidden layer reset to zero.
        Called each time before running the episode dict to generate Q_total
        """
        for agent_id, agent in self.agent_dict.items():
            agent.re_initiate_hidden_layer()



    def generate_Q_total(self, episode_dict):
        """
        given a full episode of transitions, generate the
        :param episode_dict:
        :return:
        """
        # first re_initiate the hidden layer
        self._re_initiate_hidden_layer()






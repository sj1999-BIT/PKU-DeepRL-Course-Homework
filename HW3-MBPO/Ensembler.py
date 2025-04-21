"""
To ensure ensembler bootstrap is easier implemented, we create class that
simultaneously control and regulate many neural networks of the same class.
"""

from neural_network import PredictiveNetwork, nn_learn
from data import append_values_to_file
from tqdm import tqdm

import numpy as np



class Ensembler:
    def __init__(self, dynamic_Arr: [PredictiveNetwork, ]):
        self.dynamic_Arr = dynamic_Arr
        self.size = len(dynamic_Arr)

    def get_random_next_state_and_reward(self, cur_state, action):
        """
        Randomly select a next state prediction from one of the dynamics networks
        :param cur_state: current state
        :param action: action
        :return: randomly selected next state prediction and reward
        """
        # Get predictions from each dynamics network in dynamic_Arr
        all_predictions = []
        all_rewards = []
        all_next_state_distribution = []
        all_reward_distribution = []

        for dynamic_model in self.dynamic_Arr:
            next_state, reward, next_state_distribution, reward_distribution = \
                dynamic_model.get_next_state_and_reward(cur_state, action)
            all_predictions.append(next_state)
            all_rewards.append(reward)
            all_next_state_distribution.append(next_state_distribution)
            all_reward_distribution.append(reward_distribution)

        # Randomly select one of the model's predictions
        random_idx = np.random.randint(0, len(self.dynamic_Arr))
        selected_next_state = all_predictions[random_idx]
        selected_reward = all_rewards[random_idx]
        next_state_distribution = all_next_state_distribution[random_idx]
        reward_distribution = all_reward_distribution[random_idx]

        return selected_next_state, selected_reward, next_state_distribution, reward_distribution

    def train_models(self, batch_sample_arr, path="."):

        # train Dynamic NN
        for model_index in range(self.size):
            # training current model
            cur_model = self.dynamic_Arr[model_index]
            # generate batch data for training
            sample = batch_sample_arr[model_index]

            # model learn, save the weights
            nn_learn(cur_model, sample, filepath=f"{path}/model_{model_index}_loss.txt")

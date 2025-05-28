"""
To ensure ensembler bootstrap is easier implemented, we create class that
simultaneously control and regulate many neural networks of the same class.
"""

from neural_network import DynamicsNetwork, nn_learn
from data import append_values_to_file
from tqdm import tqdm

import numpy as np



class Ensembler:
    def __init__(self, dynamic_Arr: [DynamicsNetwork, ]):
        self.dynamic_Arr = dynamic_Arr
        self.size = len(dynamic_Arr)

    def get_best_next_state(self, cur_state, action):
        """
        Run through all dynamics networks and average results
        :param cur_state: current state
        :param action: action
        :return: average next state prediction
        """
        # Initialize an array to store predictions from each model
        all_predictions = []

        # Get predictions from each dynamics network in dynamic_Arr
        for dynamic_model in self.dynamic_Arr:
            next_state = dynamic_model.get_next_state(cur_state, action)
            all_predictions.append(next_state)

        # Convert list of predictions to numpy array
        all_predictions = np.array(all_predictions)

        # Average the predictions along the first axis (across models)
        average_prediction = np.mean(all_predictions, axis=0)

        return average_prediction

    def train_models(self, batch_sample_arr, path="."):

        # train Dynamic NN
        for model_index in range(self.size):
            # training current model
            cur_model = self.dynamic_Arr[model_index]
            # generate batch data for training
            sample = batch_sample_arr[model_index]

            # model learn, save the weights
            nn_learn(cur_model, sample, filepath=f"{path}/model_{model_index}_loss.txt")

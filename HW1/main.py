from model import Q_agent
import torch.optim as optim

import random
from replayDataBuffer import get_training_data, get_reward_data
from visual import plot_progress_data, plot_frequency
from data import save_array_to_file, append_values_to_file, load_array_from_file

"""
first check what is wrong with the training data.
"""

if __name__=="__main__":

    # initialise 2 agents
    # q_agent = ImprovedQAgent()
    # q_agent.to(device='cuda')


    # get around 300 transitions
    # each transition is recorded_cur_state, action, recorded_reward, q_val, recorded_next_state
    # training_data = get_training_data(q_agent, num_games=100)
    #
    # action_arr = [data[1] for data in training_data]
    #
    # action_choice = [0,2,3]
    #
    # plot_frequency(action_arr, action_choice, save_plot=True)

    # loss_values = load_array_from_file("loss.json")
    reward_values = []
    reward_values.extend(load_array_from_file("results/iter 1 improved 1K training/loss.json"))

    plot_progress_data(reward_values, save_plot=True, plot_file_title="results/iter 1 improved 1K training/loss_plot.png")













from model import DDQN
import torch.optim as optim

import random
from replayDataBuffer import get_training_data, get_reward_data
from visual import plot_progress_data
from data import save_array_to_file, append_values_to_file, load_array_from_file

"""
we need to do the following things
1. save the trained model so that we can run it in the future
2. added in the update function to transfer the weights over
3. record down the loss function, reward and q-value output overtime.

summary
a full training pipeline and data collection pipeline. 
"""

if __name__=="__main__":

    # Initialize a list to store loss values
    loss_values = []
    # initialize a list to store target_network performance
    reward_values = []

    # initialise 2 agents
    q_agent = DDQN(load_path="naive/q_agent_weights.pth")
    q_agent.to(device='cuda')

    target_agent = DDQN(load_path="naive/q_agent_weights.pth")
    target_agent.to(device='cuda')


    # generate 2 new JSON files to store the data
    loss_value_filename = "loss.json"
    save_array_to_file([], loss_value_filename)

    reward_value_filename = "reward.json"
    save_array_to_file([], reward_value_filename)


    # set optimizer
    optimizer = optim.Adam(q_agent.parameters(), lr=0.001)

    epoch_num = 10000

    for epoch in range(epoch_num):

        print(f"current epoch {epoch}")

        # one batch
        optimizer.zero_grad()

        # get around 300 transitions
        training_data = get_training_data(q_agent, num_games=30)


        # Randomly select 64 elements without replacement
        # training_batch = random.sample(training_data, 64)

        # show how data distributed
        # plot_value_distribution([data[2] for data in training_batch])

        loss_tensor = q_agent.loss_function(training_data, target_agent)

        loss_value = loss_tensor.item()

        # Store the loss value
        loss_values.append(loss_value)  # .item() converts the tensor to a Python number

        append_values_to_file(loss_value, loss_value_filename)

        # backpropagate
        loss_tensor.backward()
        # update optimizer
        optimizer.step()

        # modify target agent with the weights of the q_agent
        target_agent.combine_weights(q_agent, ratio=0.999)

        reward = get_reward_data(target_agent, num_games=10)

        reward_values.append(reward)
        append_values_to_file(reward, reward_value_filename)

        if epoch % 100 == 0:
            ##### save the weights
            q_agent.save_weights("./naive/q_agent_weights.pth")
            target_agent.save_weights("./naive/target_agent_weights.pth")
            plot_progress_data(loss_values, save_plot=True, plot_file_title="results/iter 1 improved 1K training/loss_plot.png")
            plot_progress_data(reward_values, save_plot=True, plot_file_title="reward_plot.png")










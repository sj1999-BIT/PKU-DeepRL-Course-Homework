from model import Q_agent
import torch.optim as optim

import random
from replayDataBuffer import get_training_data

if __name__=="__main__":

    # Initialize a list to store loss values
    loss_values = []

    # initialise 2 agents
    q_agent = Q_agent()
    q_agent.to(device='cuda')

    target_agent = Q_agent()
    target_agent.to(device='cuda')


    # set optimizer
    optimizer = optim.Adam(q_agent.parameters(), lr=0.001)

    epoch = 10

    for _ in range(epoch):

        # one batch
        optimizer.zero_grad()

        # get around 300 transitions
        training_data = get_training_data(q_agent, num_games=3)

        # Randomly select 64 elements without replacement
        training_batch = random.sample(training_data, 64)

        # show how data distributed
        # plot_value_distribution([data[2] for data in training_batch])

        loss = q_agent.loss_function(training_batch, target_agent)

        # Store the loss value
        loss_values.append(loss.item())  # .item() converts the tensor to a Python number

        # backpropagate
        loss.backward()
        # update optimizer
        optimizer.step()



    # After training is complete, plot the loss graph
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(loss_values)
    plt.title('Training Loss over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()





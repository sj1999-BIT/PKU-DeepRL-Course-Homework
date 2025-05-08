import os.path

import numpy as np
import tensorflow as tf

from tf_neural_network import AgentNetwork, MixNetwork
from helper import random_sample, convert_arr_to_dict
from tqdm import tqdm
from dataKeys import STATES, ACTIONS, AGENT_ZERO, AGENT_ONE, AGENT_TWO, Q_VALS, REWARDS, REWARD_FILENAME, LOSS_FILENAME
from data import append_values_to_file
from dataProcessor import DataProcessor
from qmix import QMIX


def build_models():
    """
    Tf models needs to be build before they can load weights
    :return:
    """

    dummy_obs = np.zeros(18)
    dummy_prev_action = np.zeros(1)

    agent = AgentNetwork()
    agent(dummy_obs, dummy_prev_action)

    dummy_global_state = np.zeros(54)
    dummy_q_vals = np.zeros(3)

    mixNet = MixNetwork()
    mixNet({STATES: dummy_global_state, Q_VALS: dummy_q_vals})

    return agent, mixNet


def generate_mix_NN_inputs_for_full_episode(episode_dict, agent: AgentNetwork, mixNet: MixNetwork):
    """
    MixNetwork needs the Q-val from each agent and the current state to generate the Q-total.

    We assume that inputs are always in temporal sequences, so each agent has the correct hidden state

    Inputs is a full episode.

    key point is that we only initiate one agent so as to make sure the backpropogation gradient only works on
    one agent.

    :return:
    """

    cur_state_arr = episode_dict[STATES]
    cur_action_arr = episode_dict[ACTIONS]

    # assume first action before episode starts is always 0
    prev_action_arr = np.vstack((np.array([[0, 0, 0]]), episode_dict[ACTIONS]))[:-1]

    def get_chosen_q_val(q_val_tensor, cur_agent_id):
        """
        it could be a single (5,) q_val for a single transition, or
        a full episode of q_val outputs
        :param cur_agent_id: specific agent id
        :param q_val_arr: outputed q_val
        :return:
        """
        cur_action_arr = np.squeeze(cur_action_dict[cur_agent_id], axis=1)

        # need to add in batch_dims to ensure each action acts as an indices to corresponding q_val outputs
        return tf.gather(q_val_tensor, cur_action_arr, batch_dims=1)

    cur_state_dict = convert_arr_to_dict(cur_state_arr)
    prev_action_dict = convert_arr_to_dict(prev_action_arr)
    cur_action_dict = convert_arr_to_dict(cur_action_arr)

    mixNet_inputs_dict = {}

    # collect the q_values
    q_val_dict = {}
    for agent_id in cur_state_dict.keys():
        # must stay in tensor form for the gradient to track during backpropagation
        # run the same agent through each batch
        # since its not single input, hidden state is not maintained
        cur_q_val_output_tensor = agent.call(cur_state_dict[agent_id], prev_action_dict[agent_id])

        chosen_q_val_tensor = get_chosen_q_val(cur_q_val_output_tensor, agent_id)
        q_val_dict[agent_id] = chosen_q_val_tensor

    # record the corresponding state
    mixNet_inputs_dict[STATES] = cur_state_arr

    # create a (25, 3) tensor
    mixNet_inputs_dict[Q_VALS] = tf.concat([tf.expand_dims(q_val_dict[AGENT_ZERO], 1),
                                            tf.expand_dims(q_val_dict[AGENT_ONE], 1),
                                            tf.expand_dims(q_val_dict[AGENT_TWO], 1)
                                            ],
                                           axis=1)

    return mixNet_inputs_dict


def get_TD_error(episode_dict, agent: AgentNetwork, mixNet: MixNetwork, discount_factor=0.9):
    """
    An array of sampled_episode_arr is provided, get the Q_total for each time_step.
    Generate the TD_error for each time_step
    :return:
    """

    # get the inputs for mixNet
    mixNet_inputs = generate_mix_NN_inputs_for_full_episode(episode_dict, agent, mixNet)

    # pass the inputs to mixNet to generate the q_total
    # must stay in tf form
    q_total = tf.squeeze(mixNet.call(mixNet_inputs), -1)

    # TD is the MSE of reward + next_state_q_total * discount - cur_state_q_total
    cur_state_q_total = q_total[:-1]
    next_state_q_total = q_total[1:]
    reward = tf.convert_to_tensor(episode_dict[REWARDS][:-1], dtype=tf.float32)

    # Calculate the TD error
    TD_error = tf.square(reward + next_state_q_total * discount_factor - cur_state_q_total)

    return TD_error


def training(agent, mixNet, sampled_episode_arr, epoch_num=20, batch_size=32, learning_rate=0.0005):
    """
    file_path: we will load one agent network and one mix agent, all updates will be performed on them.
    A sample of full episodes are provided.
    Each epoch, we generate the TD error for every transition.
    Then we randomly sample a batch of 32 TD error and sum them up
    :return:
    """

    # data filepath for saving the training progress
    loss_file_location = os.path.join(file_path, LOSS_FILENAME)

    # Track all trainable variables for gradient computation
    trainable_variables = []

    trainable_variables.extend(agent.trainable_variables)
    trainable_variables.extend(mixNet.trainable_variables)

    # Create optimizers for agent networks and mix network, need to clip due to potentially very large loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)  # Example with TensorFlow

    for epoch in tqdm(range(epoch_num), "training QMIX: "):
        # Track the variables for this episode
        with tf.GradientTape() as tape:
            tape.watch(trainable_variables)

            # Initialize TD_error_arr as an empty tensor
            TD_error_tensor = tf.constant([], dtype=tf.float32)

            # Iteratively add TD errors from each episode
            for episode_dict in sampled_episode_arr:
                # Assuming self.get_TD_error returns a tensor
                cur_TD_error = get_TD_error(episode_dict, agent, mixNet)

                # Handle first iteration
                if tf.equal(tf.size(TD_error_tensor), 0):
                    TD_error_tensor = cur_TD_error
                else:
                    # Concatenate with existing errors
                    TD_error_tensor = tf.concat([TD_error_tensor, cur_TD_error], axis=0)

            sampled_TD_error = random_sample(TD_error_tensor, sample_size=batch_size)

            TD_loss = tf.reduce_sum(sampled_TD_error)
            print(TD_loss.numpy())
            append_values_to_file(TD_loss.numpy(), loss_file_location)

        # Compute and apply gradients
        gradients = tape.gradient(TD_loss, trainable_variables)

        # Apply gradients separately
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    # save models
    agent.save_model(filepath=file_path)
    mixNet.save_model(filepath=file_path)


if __name__ == "__main__":
    file_path = "./data/"

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    dp = DataProcessor()

    # initial data collection
    qmix_agent = QMIX(file_path=file_path)
    current_reward = dp.collect_data(qmix_agent, data_size=2)
    print(f"average current reward {current_reward}")

    append_values_to_file(current_reward, os.path.join(file_path, REWARD_FILENAME))

    # take the used agent from q
    agent, mixNet = build_models()

    agent.load_model(filepath=file_path)
    mixNet.load_model(filepath=file_path)

    # save the agent, this is mainly for first initialise
    qmix_agent.save_weights()

    while True:
        # in the paper its updated every 200 episodes, lack of computation power so I downscale it

        # get a sample of full episode
        sampled_episode_arr = dp.random_sample(batch_size=2)

        # training of the models
        training(agent, mixNet, sampled_episode_arr, epoch_num=20, batch_size=32, learning_rate=0.0005)

        # collect more date with the update model
        qmix_agent = QMIX(file_path=file_path)
        current_reward = dp.collect_data(qmix_agent, data_size=32)

        print(f"average current reward {current_reward}")

        append_values_to_file(current_reward, os.path.join(file_path, REWARD_FILENAME))
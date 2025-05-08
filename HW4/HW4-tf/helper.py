"""
some helper functions
"""
import numpy as np
import tensorflow as tf

from dataKeys import AGENT_ZERO, AGENT_ONE, AGENT_TWO


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
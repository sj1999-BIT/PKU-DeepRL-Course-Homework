import tensorflow as tf
import os
import numpy as np
import torch
import random
import datetime

from dataKeys import Q_VALS, Q_TOTALS, STATES, AGENT_WEIGHT_NAME, MIX_WEIGHT_NAME


class AgentNetwork(tf.keras.Model):
    def __init__(self, input_dim=18, hidden_dim=128, output_dim=5):
        """
        Agent Network:
        - Input: Individual observation (18-dim) + previous action (one-hot encoded)
        - Hidden Layer 1: Dense layer with ReLU activation
        - GRU layer to maintain action-observation history
        - Hidden Layer 2: Dense layer with ReLU activation
        - Output: Q-values for 5 actions
        :param input_dim: input dimension
        :param hidden_dim: hidden layer dimension
        :param output_dim: output dimension
        :param learning_rate: learning rate for optimizer
        """
        super(AgentNetwork, self).__init__()

        self.action_dim = output_dim

        # Define layers

        # first layer that takes in inputs
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim+1,))

        # In your __init__ method
        self.gru = tf.keras.layers.GRU(hidden_dim,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       reset_after=False)  # Try setting this to False, causes it to use a much slower RNN

        self.dense2 = tf.keras.layers.Dense(output_dim)
        # Initialize hidden state
        self.hidden_state = None
        self.hidden_dim = hidden_dim

    def reset_hidden_state(self, x):
        """
        Reset the hidden state to zeros based on the input tensor x

        Args:
            x: Input tensor of shape (batch_size, features) or (batch_size, timesteps, features)
        """
        # Get the batch size from x
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)

        # Create a zeros tensor with shape (batch_size, hidden_dim)
        # This is the correct shape for GRUCell hidden state
        self.hidden_state = tf.zeros([x.shape[1], x.shape[-1]], dtype=tf.float32)

        return self.hidden_state

    def call(self, obs, prev_action):
        """
        Forward pass of the agent network

        Args:
            obs: Agent's observation (obs_dim)
            prev_action: One-hot encoded previous action (action_dim)

        Returns:
            q_values: Q-values for each action
            new_hidden: New hidden state
        """

        # Check if inputs are already tensors before converting
        if not isinstance(obs, tf.Tensor):
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)

        if not isinstance(prev_action, tf.Tensor):
            prev_action = tf.convert_to_tensor(prev_action, dtype=tf.float32)

        # Convert prev_action to float32 to match obs dtype
        # somehow got bug that causes it to be float64
        prev_action = tf.cast(prev_action, dtype=tf.float32)

        # Determine if we have a single state or multiple timesteps
        is_single_input = False
        if len(obs.shape) == 1:  # Single state case
            is_single_input = True
            # Add batch and time dimensions: (obs_dim,) -> (1, 1, obs_dim)
            obs = tf.expand_dims(tf.expand_dims(obs, 0), 0)
            prev_action = tf.expand_dims(tf.expand_dims(prev_action, 0), 0)
        elif len(obs.shape) == 2:  # Multiple timesteps case
            # Add batch dimension: (timesteps, obs_dim) -> (1, timesteps, obs_dim)
            obs = tf.expand_dims(obs, 0)
            prev_action = tf.expand_dims(prev_action, 0)

        # Concatenate observation and previous action along the last dim
        x = tf.concat([obs, prev_action], axis=-1)

        x = self.dense1(x)

        # Initialize hidden state if it doesn't exist
        if self.hidden_state is None:
            self.reset_hidden_state(x)

        if is_single_input:
            # single time steps we need to maintain the hidden state
            x, self.hidden_state = self.gru(x, initial_state=[self.hidden_state])
        else:
            # for multistep full episode, gru automatic process the hidden states
            x, self.hidden_state = self.gru(x)

        # Pass through output layer
        q_values = self.dense2(x)

        if is_single_input:
            q_values = tf.squeeze(q_values, axis=[0, 1])  # Remove batch and time dimensions
        else:
            q_values = tf.squeeze(q_values, axis=0)  # Remove only batch dimension

        return q_values

    def select_action(self, obs, prev_action, hidden_state=None, epsilon=0.1):
        """
        Select action using epsilon-greedy policy

        Args:
            obs: Agent's observation
            prev_action: One-hot encoded previous action
            epsilon: Exploration rate (probability of selecting a random action)

        Returns:
            action: Selected action index
            Q-value: q_value of the selected action
        """

        # update hidden state
        if hidden_state is not None:
            self.hidden_state = hidden_state

        # Get Q-values from forward pass
        q_values = self.call(obs, prev_action)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Exploration: select random action
            action = random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: select action with highest Q-value
            action = tf.argmax(q_values, axis=-1).numpy()

        return action, q_values.numpy()[action]

    def save_model(self, filepath="./"):
        """
        Save model weights using NumPy
        :param filepath: Path to save weights
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

            # Build model if needed
            if not self.built:
                dummy_obs = tf.zeros((1, 18), dtype=tf.float32)
                dummy_prev_action = tf.zeros((1, 1), dtype=tf.float32)
                _ = self(dummy_obs, dummy_prev_action)

            # Create full filepath
            if os.path.isdir(filepath):
                weights_path = os.path.join(filepath, AGENT_WEIGHT_NAME)
            else:
                weights_path = filepath

            # Ensure .npy extension
            if not weights_path.endswith('.npy'):
                weights_path += '.npy'

            # Get weights as NumPy arrays
            weights = self.get_weights()

            # Save as NumPy file
            np.save(weights_path, {'weights': weights}, allow_pickle=True)
            print(f"Weights saved to {weights_path}")
            return True

        except Exception as e:
            print(f"Error saving weights: {str(e)}")
            import traceback
            print(f"Stack trace: {traceback.format_exc()}")
            return False

    def load_model(self, filepath="./"):
        """
        Load weights using NumPy
        :param filepath: Path to load weights from
        """
        try:
            # Handle directory vs file path
            if os.path.isdir(filepath):
                weights_path = os.path.join(filepath, AGENT_WEIGHT_NAME)
            else:
                weights_path = filepath

            # Ensure .npy extension
            if not weights_path.endswith('.npy'):
                weights_path += '.npy'

            # Check if file exists
            if not os.path.exists(weights_path):
                print(f"Weights file not found: {weights_path}")
                return False

            # Build model if needed
            if not self.built:
                dummy_obs = tf.zeros((1, 18), dtype=tf.float32)
                dummy_prev_action = tf.zeros((1, 1), dtype=tf.float32)
                _ = self(dummy_obs, dummy_prev_action)

            # Load weights from NumPy file
            data = np.load(weights_path, allow_pickle=True).item()

            if 'weights' in data:
                # Use direct weights from file
                self.set_weights(data['weights'])
            elif 'model_weights' in data:
                # Handle legacy format
                self.set_weights(data['model_weights'])
            else:
                print(f"No weights found in file: {weights_path}")
                return False

            print(f"Weights loaded from {weights_path}")
            return True

        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            import traceback
            print(f"Stack trace: {traceback.format_exc()}")
            return False

class MixNetwork(tf.keras.Model):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=1):
        """
        Mixing Network:
        - Input: Individual agent Q-values dim: (3)

        - first hyperparameter layer: takes in global state dim： (54), generate first hidden layer weights (absolute)

        - first hidden layer: takes in 3 input values, outputs hidden_dim

        - output: hidden_dim

        - concate with unmodified global_state, goes through a leaky_relu

        - second hyperparameter layer: takes in global state dim： (54), generate second hidden layer weights (absolute)

        - second hidden layer: takes in hidden_dim + global_state, outputs a single values

        - output: dim 1

        - third hyperparameter layer: takes in global state dim： (54), perform RELU, pass into a final layer to generate
          a single value

        - addition with the output

        -output: 1

        - Hidden Layer: Dense layer with positive weights
        - Output: Single joint Q-value
        """
        super(MixNetwork, self).__init__()

        # need for create new layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # hyper_network - takes in global state generates weights for first hidden layer
        self.hyper_fc_to_first_layer_weights = tf.keras.layers.Dense(input_dim * hidden_dim)

        # hyper_network - takes in global state generates bias for first hidden layer
        self.hyper_fc_to_first_layer_bias = tf.keras.layers.Dense(hidden_dim)

        # hyper_network - takes in global state generates weights for second hidden layer
        self.hyper_fc_to_sec_layer_weights = tf.keras.layers.Dense(hidden_dim * output_dim)

        # hyper_network - takes in global state generates bias for second hidden layer
        self.hyper_fc_to_sec_layer_bias = tf.keras.layers.Dense(output_dim, activation='relu')

    def call(self, mix_input_dict):
        """
        input dict consist of cur states and the q_vals generated
        :param mix_input_dict:
        :return:
        """
        # Check if inputs are already tensors before converting
        if not isinstance(mix_input_dict[Q_VALS], tf.Tensor):
            q_vals_inputs = tf.convert_to_tensor(mix_input_dict[Q_VALS], dtype=tf.float32)
        else:
            q_vals_inputs = mix_input_dict[Q_VALS]

        if not isinstance(mix_input_dict[STATES], tf.Tensor):
            global_state_inputs = tf.convert_to_tensor(mix_input_dict[STATES], dtype=tf.float32)
        else:
            global_state_inputs = mix_input_dict[STATES]

        if len(q_vals_inputs.shape) != len(global_state_inputs.shape):
            raise Exception(f"invalid data inputted: q_val input dim is {q_vals_inputs.shape}, "
                            f"whereas global state inputs dim is {global_state_inputs.shape}")

        batch_size = 1 # by default
        if len(q_vals_inputs.shape) == 1:
            # single input, expand it to batch size 1
            q_vals_inputs = tf.expand_dims(q_vals_inputs, 0)
            global_state_inputs = tf.expand_dims(global_state_inputs, 0)
        elif len(q_vals_inputs.shape) == 2:
            # input is in batch
            batch_size = q_vals_inputs.shape[0]
        else:
            raise Exception(f"invalid data dimension: {q_vals_inputs.shape}")

        # First hypernetwork layer - generate weights for first hidden layer, shaped [25, 3, 128]
        w1 = tf.abs(self.hyper_fc_to_first_layer_weights(global_state_inputs))
        reshaped_weights = tf.reshape(w1, [batch_size, self.input_dim, self.hidden_dim])

        b1 = self.hyper_fc_to_first_layer_bias(global_state_inputs)
        reshaped_bias = tf.reshape(b1, [batch_size, self.hidden_dim])


        # perform matrix multiplication to simulate passing through neural network
        # [25, 1, 3] @ [25, 3, 128] -> [25, 1, 128]
        x = tf.matmul(tf.expand_dims(q_vals_inputs, 1), reshaped_weights)

        # [25, 1, 128] + [25, 1, 128] -> [25, 1, 128]
        x = x + tf.expand_dims(reshaped_bias, 1)

        # second hypernetwork layer - generate weights for first hidden layer, shaped [25,128,1]
        w2 = tf.abs(self.hyper_fc_to_sec_layer_weights(global_state_inputs))
        reshaped_weights = tf.reshape(w2, [batch_size, self.hidden_dim, self.output_dim])

        # , shaped [25, 1]
        b2 = self.hyper_fc_to_sec_layer_bias(global_state_inputs)
        reshaped_bias = tf.reshape(b2, [batch_size, self.output_dim])

        # actual forward pass in mix NN
        # perform matrix multiplication to simulate passing through neural network
        # [25, 1, 128] @ [25, 128, 1] -> [25, 1, 1]
        x = tf.matmul(x, reshaped_weights)

        # [25, 1, 1] + [25, 1, 1] -> [25, 1, 1]
        x = x + tf.expand_dims(reshaped_bias, 1)

        x = tf.squeeze(x, 1)

        return x

    def save_model(self, filepath="./"):
        """
        Save model weights using NumPy
        :param filepath: Path to save weights
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

            # Build model if needed
            if not self.built:
                dummy_obs = tf.zeros((1, 18), dtype=tf.float32)
                dummy_prev_action = tf.zeros((1, 1), dtype=tf.float32)
                _ = self(dummy_obs, dummy_prev_action)

            # Create full filepath
            if os.path.isdir(filepath):
                weights_path = os.path.join(filepath, MIX_WEIGHT_NAME)
            else:
                weights_path = filepath

            # Ensure .npy extension
            if not weights_path.endswith('.npy'):
                weights_path += '.npy'

            # Get weights as NumPy arrays
            weights = self.get_weights()

            # Save as NumPy file
            np.save(weights_path, {'weights': weights}, allow_pickle=True)
            print(f"Weights saved to {weights_path}")
            return True

        except Exception as e:
            print(f"Error saving weights: {str(e)}")
            import traceback
            print(f"Stack trace: {traceback.format_exc()}")
            return False

    def load_model(self, filepath="./"):
        """
        Load weights using NumPy
        :param filepath: Path to load weights from
        """
        try:
            # Handle directory vs file path
            if os.path.isdir(filepath):
                weights_path = os.path.join(filepath, MIX_WEIGHT_NAME)
            else:
                weights_path = filepath

            # Ensure .npy extension
            if not weights_path.endswith('.npy'):
                weights_path += '.npy'

            # Check if file exists
            if not os.path.exists(weights_path):
                print(f"Weights file not found: {weights_path}")
                return False

            # Build model if needed
            if not self.built:
                dummy_obs = tf.zeros((1, 18), dtype=tf.float32)
                dummy_prev_action = tf.zeros((1, 1), dtype=tf.float32)
                _ = self(dummy_obs, dummy_prev_action)

            # Load weights from NumPy file
            data = np.load(weights_path, allow_pickle=True).item()

            if 'weights' in data:
                # Use direct weights from file
                self.set_weights(data['weights'])
            elif 'model_weights' in data:
                # Handle legacy format
                self.set_weights(data['model_weights'])
            else:
                print(f"No weights found in file: {weights_path}")
                return False

            print(f"Weights loaded from {weights_path}")
            return True

        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            import traceback
            print(f"Stack trace: {traceback.format_exc()}")
            return False


    # def save_weights(self, filepath="./"):
    #     """
    #     Save the model weights to a file
    #     :param filepath: path to save the model weights
    #     :param timestep: optional timestep to include in filename
    #     """
    #     # Create directory if it doesn't exist
    #     os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    #
    #     filepath = os.path.join(filepath, MIX_WEIGHT_NAME)
    #
    #     # Create dictionary to save
    #     checkpoint = {
    #         'model_weights': self.get_weights()
    #     }
    #
    #     # Save as numpy array
    #     np.save(filepath, checkpoint, allow_pickle=True)
    #     print(f"Model weights saved to {filepath}")
    #
    # def load_weights(self, filepath="./"):
    #     """
    #     Load the model weights from a file
    #     :param filepath: path to load the model weights from
    #     """
    #
    #     filepath = os.path.join(filepath, f"{MIX_WEIGHT_NAME}.npy")
    #
    #     if not os.path.exists(filepath):
    #         print(f"No weights file found at {filepath}")
    #         return False
    #
    #     try:
    #         # Load the checkpoint
    #         checkpoint = np.load(filepath, allow_pickle=True).item()
    #
    #         # Load model weights
    #         self.set_weights(checkpoint['model_weights'])
    #
    #         print(f"Model weights loaded from {filepath}")
    #         return True
    #     except Exception as e:
    #         print(f"Error loading weights: {str(e)}")
    #         return False

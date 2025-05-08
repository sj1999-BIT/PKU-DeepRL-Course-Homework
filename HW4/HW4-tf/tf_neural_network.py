import tensorflow as tf
import os
import numpy as np
import datetime


class AgentNetwork(tf.keras.Model):
    def __init__(self, input_dim=18, hidden_dim=128, output_dim=1):
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

        # Define layers

        # first layer that takes in inputs
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim+1,))

        self.gru = tf.keras.layers.GRU(hidden_dim,
                                       return_sequences=True,
                                       return_state=False,
                                       recurrent_initializer='glorot_uniform')

        self.dense2 = tf.keras.layers.Dense(output_dim)

    def call(self, inputs, training=False):
        """
        Forward pass of the neural network
        :param inputs: input tensor
        :param training: whether in training mode
        :return: output tensor
        """
        x = self.dense1(inputs)
        x = self.gru(x)

        return self.dense2(x)

    def save_weights(self, filepath="./", timestep=None):
        """
        Save the model weights to a file
        :param filepath: path to save the model weights
        :param timestep: optional timestep to include in filename
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        if self.name_str is not None:
            filepath = f"{self.name_str}_{filepath}"

        if timestep is not None:
            filepath = f"{filepath}_{timestep}"

        filepath = f"{filepath}.h5"

        # Save optimizer state
        opt_weights = self.optimizer.get_weights()
        opt_config = self.optimizer.get_config()

        # Create dictionary to save
        checkpoint = {
            'model_weights': self.get_weights(),
            'optimizer_weights': opt_weights,
            'optimizer_config': opt_config
        }

        # Save as numpy array
        np.save(filepath, checkpoint, allow_pickle=True)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        """
        Load the model weights from a file
        :param filepath: path to load the model weights from
        """
        if not filepath.endswith('.h5'):
            filepath = f"{filepath}.h5"

        if not os.path.exists(filepath):
            print(f"No weights file found at {filepath}")
            return False

        try:
            # Load the checkpoint
            checkpoint = np.load(filepath, allow_pickle=True).item()

            # Load model weights
            self.set_weights(checkpoint['model_weights'])

            # Recreate optimizer with saved config
            self.optimizer = tf.keras.optimizers.get(checkpoint['optimizer_config'])
            self.optimizer.set_weights(checkpoint['optimizer_weights'])

            print(f"Model weights loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            return False

    # Alternative TensorFlow native saving method
    def save_model(self, filepath="."):
        """
        Save the entire model using TensorFlow's native format
        :param filepath: path to save the model
        """
        if self.name_str is not None:
            filepath = f"{self.name_str}_{filepath}"

        self.save(filepath)
        print(f"Full model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load the entire model using TensorFlow's native format
        :param filepath: path to load the model from
        """
        if not os.path.exists(filepath):
            print(f"No model found at {filepath}")
            return False

        try:
            loaded_model = tf.keras.models.load_model(filepath)
            self.set_weights(loaded_model.get_weights())
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Define parameters
    batch_size = 5
    sequence_length = 10
    input_dim = 18  # As per your model
    hidden_dim = 64  # Using smaller value for demonstration
    output_dim = 1

    # Create the model
    model = AgentNetwork(input_dim, hidden_dim, output_dim)

    # Create random input data
    # Shape: [batch_size, sequence_length, input_dim+1]
    # The +1 accounts for the previous action (as mentioned in your model's docstring)
    input_batch = tf.random.normal([sequence_length, input_dim + 1])

    # Method 1: Process the entire batch at once
    print("Processing entire batch at once...")
    batch_output = model(input_batch)
    print(f"Batch output shape: {batch_output.shape}")

    # Method 2: Process each example individually in a loop
    print("\nProcessing individual examples in a loop...")
    individual_outputs = []
    for i in range(batch_size):
        # Extract a single example (keeping the sequence dimension)
        single_input = input_batch[i] # Shape: [1, seq_len, input_dim+1]
        single_output = model(single_input)
        individual_outputs.append(single_output)

    # Stack individual outputs to create a batch
    stacked_outputs = tf.concat(individual_outputs, axis=0)
    print(f"Stacked individual outputs shape: {stacked_outputs.shape}")

    # Verify the outputs are the same
    difference = tf.reduce_max(tf.abs(batch_output - stacked_outputs))
    print(f"\nMaximum absolute difference: {difference.numpy()}")

    if difference < 1e-6:
        print("RESULT: Both methods produce identical outputs!")
    else:
        print("RESULT: Outputs differ slightly (possibly due to floating-point precision).")

    # Examine a specific example for detailed comparison
    example_idx = 0
    print(f"\nDetailed comparison for example {example_idx}:")
    print(f"Batch method output[0,0,:5]: {batch_output[example_idx, 0, :5].numpy()}")
    print(f"Loop method output[0,0,:5]: {stacked_outputs[example_idx, 0, :5].numpy()}")
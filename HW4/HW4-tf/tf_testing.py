import tensorflow as tf


# Define your models
class Model1(tf.keras.Model):
    def __init__(self):
        super(Model1, self).__init__()
        # Define layers for model 1
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


class Model2(tf.keras.Model):
    def __init__(self):
        super(Model2, self).__init__()
        # Define layers for model 2
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)  # Output layer

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


# Initialize models
model1 = Model1()
model2 = Model2()

# Define optimizer - you can use different optimizers for each model if needed
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# Training step function
@tf.function  # Optional: use tf.function for better performance
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        # Forward pass through model 1
        intermediate_outputs = model1(inputs)

        # Forward pass through model 2
        final_outputs = model2(intermediate_outputs)

        # Calculate loss
        loss = tf.keras.losses.MSE(targets, final_outputs)

    # Get gradients for both models
    # The tape automatically tracks the entire computation graph
    gradients = tape.gradient(loss, model1.trainable_variables + model2.trainable_variables)

    # Split gradients for each model if needed
    model1_grads = gradients[:len(model1.trainable_variables)]
    model2_grads = gradients[len(model1.trainable_variables):]

    # Apply gradients to both models
    optimizer.apply_gradients(zip(gradients, model1.trainable_variables + model2.trainable_variables))

    return loss

epochs=100

# Training loop
for epoch in range(epochs):
    # Loop through your dataset
    for x_batch, y_batch in dataset:
        loss = train_step(x_batch, y_batch)
    print(f"Epoch {epoch}, Loss: {loss.numpy()}")
# Report for Homework 1

## Introduction
My idea is to directly implement the neural network structure from the 
slides using tensorflow and test out how it performs.

1. input state is a stack of raw pixels from last 4 frames using RGB
2. target sample 64 batch of transitions.
3. use the soft update sample of w* = (1-r)w + rW*, r=0.999

### first 1000 epoch training

Training failed as the loss function drops to zero rapidly.
Target network unable to show improvement overtime.
First revision shows that model just only use action left. This 
could be because the q-network initial weights caused the model
to only rely on left action.

The loss function quickly diminished, indicating that the Q-network
was unable to improve further.

### next 1000 epoch training

I aim to improve my results by application the following changes
1. Replace Q-network with a greedy epilson agent so that the agent can explore other actions
2. increase replay buffer size and data collected. Previously I was using only 10 games and sample 64 transitions from it. Now I will increase it to 100 games.

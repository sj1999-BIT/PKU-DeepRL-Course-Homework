# Report for Homework 1

## Introduction
My idea is to directly implement the neural network structure from the 
slides using tensorflow and test out how it performs.

1. input state is a stack of raw pixels from last 4 frames using RGB
2. target sample 64 batch of transitions.
3. use the soft update sample of w* = (1-r)w + rW*, r=0.999
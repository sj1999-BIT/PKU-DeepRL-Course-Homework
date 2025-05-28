"""
this is where the planning policy is generated
"""
import torch
import numpy as np


class ModelPredictiveControl:
    def __init__(self, dynamics_models, horizon=20, num_candidates=1000, num_elites=100,
                 iterations=3, std_decay=0.9):
        """
        Initialize MPC with multiple dynamics models.

        Args:
            dynamics_models: List of 10 dynamics models
            horizon: Planning horizon (timesteps to look ahead)
            num_candidates: Number of action sequences to sample
            num_elites: Number of elite samples to keep (num_candidates * elite_fraction)
            iterations: Number of CEM iterations
            std_decay: Standard deviation decay factor
        """
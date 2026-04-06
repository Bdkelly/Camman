"""
noise — Ornstein-Uhlenbeck (OU) process for temporally correlated exploration.

The OU process generates smooth, mean-reverting noise that is well-suited for
exploring continuous action spaces in physical control tasks. Unlike independent
Gaussian noise, consecutive OU samples are correlated, producing smoother
action perturbations that better match the inertia of real hardware.

The discrete update rule is:
    dx = theta * (mu - x) + sigma * N(0, 1)
    x  = x + dx
"""

import numpy as np
import copy


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise.

    Generates noise that reverts toward a mean ``mu`` at rate ``theta`` while
    being perturbed by Gaussian noise with standard deviation ``sigma``.

    Args:
        size (int): Dimensionality of the noise vector (should match action_size).
        seed (int): Random seed for reproducibility. Default: 0.
        mu (float): Long-run mean of the process. Default: 0.0.
        theta (float): Mean-reversion rate. Higher values revert faster. Default: 0.15.
        sigma (float): Volatility (std-dev of the Gaussian perturbation). Default: 0.2.

    Attributes:
        state (np.ndarray): Current internal state of the process, shape (size,).
    """

    def __init__(self, size, seed=0, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the process state to the mean ``mu``."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Advance the process by one step and return the new noise sample.

        Returns:
            np.ndarray: Noise vector of shape (size,).
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
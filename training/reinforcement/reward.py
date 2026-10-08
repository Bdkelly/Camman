"""Bounded pan-only reward: vertical error is not controllable by this hardware."""

import numpy as np

from training.reinforcement.config import RWD_WEIGHTS


class RewardSystem:
    def __init__(self, reward_weights=None):
        self.reward_weights = {**RWD_WEIGHTS, **(reward_weights or {})}
        self.reset()

    def calculate_reward(self, dx, dy, pan_action, is_detected=True):
        weights = self.reward_weights
        centering = (
            weights["centering_peak"] * np.exp(-weights["centering_decay"] * dx**2)
            if is_detected
            else -weights["lost_ball_penalty"]
        )
        bonus = weights["window_bonus"] if is_detected and abs(dx) < 0.1 else 0.0
        return float(
            centering
            + bonus
            - weights["effort"] * pan_action**2
            - weights["stability"] * (pan_action - self.prev_action) ** 2
        )

    def update_prev_action(self, pan_action):
        self.prev_action = float(pan_action)

    def reset(self):
        self.prev_action = 0.0

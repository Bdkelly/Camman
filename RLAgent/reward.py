"""
reward — Multi-component reward function for camera pan tracking.

The reward signal balances four objectives:

1. **Centering** — Gaussian reward that peaks when the ball is at the frame centre,
   decaying smoothly as the ball drifts away.
2. **Effort** — L1 penalty on the magnitude of the pan action to discourage
   unnecessary movement.
3. **Stability** — L1 penalty on action acceleration (change from the previous
   action) to produce smooth, jerk-free trajectories.
4. **Window bonus** — Flat bonus awarded whenever the ball is within the centre
   band (|dx| < 0.125), reinforcing fine-grained alignment.

A large flat penalty is applied whenever the ball is not detected, discouraging
the agent from manoeuvring the ball out of the camera's field of view.
"""

import numpy as np


class RewardSystem:
    """Computes the per-step reward signal for camera pan training.

    Weights are passed in at construction so the same class can be reused
    with different reward schemes without code changes.

    Args:
        reward_weights (dict): Mapping of reward component names to scalar
            coefficients. Expected keys (with fallback defaults):

            - ``centering_peak`` (float): Peak value of the centering Gaussian. Default 500.0.
            - ``centering_decay`` (float): Decay rate of the centering Gaussian. Default 10.0.
            - ``effort`` (float): L1 penalty coefficient on |pan_action|. Default 0.01.
            - ``stability`` (float): L1 penalty coefficient on action acceleration. Default 1.0.
            - ``window_bonus`` (float): Flat bonus when |dx| < 0.125. Default 100.0.
            - ``lost_ball_penalty`` (float): Flat penalty when ball is not detected. Default 1000.0.

    Attributes:
        prev_action (np.ndarray): The pan action from the previous step, shape (1,).
            Updated by ``update_prev_action()`` and reset to zero by ``reset()``.
    """

    def __init__(self, reward_weights):
        self.reward_weights = reward_weights
        self.prev_action = np.zeros(1, dtype=np.float32)

    def calculate_reward(self, dx, dy, pan_action, is_detected=True):
        """Compute the reward for one environment step.

        Args:
            dx (float): Normalised horizontal offset of ball from frame centre (−1 to 1).
            dy (float): Normalised vertical offset of ball from frame centre (−1 to 1).
            pan_action (float): The pan command that was sent this step.
            is_detected (bool): Whether the ball was found by the detector. Default True.

        Returns:
            float: Scalar reward. Negative values represent penalties.
        """
        if not is_detected:
            return -self.reward_weights.get('lost_ball_penalty', 1000.0)

        c1_peak = self.reward_weights.get('centering_peak', 500.0)
        c1_decay = self.reward_weights.get('centering_decay', 10.0)
        c2 = self.reward_weights.get('effort', 0.01)
        c3 = self.reward_weights.get('stability', 1.0)
        window_bonus = self.reward_weights.get('window_bonus', 100.0)

        distance_squared = dx**2 + dy**2
        R_centering = c1_peak * np.exp(-c1_decay * distance_squared)

        R_effort = -c2 * np.abs(pan_action)

        acceleration = pan_action - self.prev_action[0]
        R_stability = -c3 * np.abs(acceleration)

        reward = R_centering + R_effort + R_stability

        if abs(dx) < 0.125:
            reward += window_bonus

        return reward

    def update_prev_action(self, pan_action):
        """Record the current pan action for use in the next stability calculation.

        Args:
            pan_action (float): The pan command that was executed this step.
        """
        self.prev_action[0] = pan_action

    def reset(self):
        """Reset the previous-action memory to zero at the start of an episode."""
        self.prev_action = np.zeros(1, dtype=np.float32)
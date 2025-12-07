import numpy as np

class RewardSystem:
    def __init__(self, reward_weights):
        self.reward_weights = reward_weights
        self.prev_action = np.zeros(1, dtype=np.float32)

    def calculate_reward(self, dx, dy, pan_action, is_detected=True):
        if not is_detected:
            # High penalty for losing the ball
            return -self.reward_weights.get('lost_ball_penalty', 1000.0)
        
        # --- Weight Retrieval & Adjustment ---
        # The key change is using C1 for the peak of the centering BONUS
        c1_peak = self.reward_weights.get('centering_peak', 500.0)  # Large positive bonus when centered
        c1_decay = self.reward_weights.get('centering_decay', 10.0) # How fast the bonus decays
        c2 = self.reward_weights.get('effort', 0.01)
        c3 = self.reward_weights.get('stability', 1.0)
        window_bonus = self.reward_weights.get('window_bonus', 100.0)
        
        # --- Reward Calculation ---

        # 1. R_centering (Proximity Bonus): This is now a positive bonus that peaks at c1_peak when dx=dy=0.
        distance_squared = dx**2 + dy**2
        R_centering = c1_peak * np.exp(-c1_decay * distance_squared)
        # When dx=dy=0, R_centering = c1_peak * e^0 = c1_peak (e.g., 500.0)
        
        # 2. R_effort (Action Penalty)
        R_effort = -c2 * np.abs(pan_action)
        
        # 3. R_stability (Acceleration Penalty)
        acceleration = pan_action - self.prev_action[0]
        R_stability = -c3 * np.abs(acceleration)

        # Combine all rewards
        reward = R_centering + R_effort + R_stability
        
        # 4. R_window (Explicit Window Bonus) - Keep this as a clear incentive to stay inside the boundary.
        if abs(dx) < 0.125:
            reward += window_bonus
            
        return reward

    def update_prev_action(self, pan_action):
        self.prev_action[0] = pan_action

    def reset(self):
        self.prev_action = np.zeros(1, dtype=np.float32)
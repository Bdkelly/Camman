import numpy as np

class RewardSystem:
    def __init__(self, reward_weights):
        self.reward_weights = reward_weights
        self.prev_action = np.zeros(1, dtype=np.float32)

    def calculate_reward(self, dx, dy, pan_action, is_detected=True):
        if not is_detected:
            # High penalty for losing the ball
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
        self.prev_action[0] = pan_action

    def reset(self):
        self.prev_action = np.zeros(1, dtype=np.float32)
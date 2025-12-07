import pytest
import numpy as np
from RLAgent.reward import RewardSystem

def test_reward_calculation():
    reward_system = RewardSystem(reward_weights={'centering': 100.0, 'effort': 1.0, 'stability': 20.0})
    
    # Test case 1: Ball is perfectly centered, no effort, no acceleration
    reward = reward_system.calculate_reward(dx=0, dy=0, pan_action=0)
    assert reward == 0.0

    # Test case 2: Ball is off-center
    reward = reward_system.calculate_reward(dx=0.1, dy=0.1, pan_action=0.5)
    expected_reward = -100.0 * (0.1**2 + 0.1**2) - 1.0 * np.abs(0.5) - 20.0 * np.abs(0.5 - 0)
    assert np.isclose(reward, expected_reward)

    # Test case 3: Test stability component
    reward_system.update_prev_action(0.5)
    reward = reward_system.calculate_reward(dx=0, dy=0, pan_action=0.2)
    expected_reward = -1.0 * np.abs(0.2) - 20.0 * np.abs(0.2 - 0.5)
    assert np.isclose(reward, expected_reward)

def test_reward_reset():
    reward_system = RewardSystem(reward_weights={})
    reward_system.update_prev_action(0.5)
    reward_system.reset()
    assert reward_system.prev_action[0] == 0.0

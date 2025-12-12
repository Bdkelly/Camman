import pytest
import numpy as np
from RLAgent.reward import RewardSystem

def test_reward_calculation():
    reward_system = RewardSystem(reward_weights={'centering': 100.0, 'effort': 1.0, 'stability': 20.0})
    
    reward = reward_system.calculate_reward(dx=0, dy=0, pan_action=0)
    assert np.isclose(reward, 600.0)

    reward = reward_system.calculate_reward(dx=0.1, dy=0.1, pan_action=0.5)
    expected_reward = -100.0 * (0.1**2 + 0.1**2) - 1.0 * np.abs(0.5) - 20.0 * np.abs(0.5 - 0)
    assert np.isclose(reward, expected_reward)

    expected_centering = 500.0 * np.exp(-10.0 * (0.1**2 + 0.1**2))
    expected_effort = -1.0 * 0.5
    expected_stability = -20.0 * 0.5
    expected_window = 100.0
    
    expected_reward = expected_centering + expected_effort + expected_stability + expected_window
    assert np.isclose(reward, expected_reward)

def test_reward_reset():
    reward_system = RewardSystem(reward_weights={})
    reward_system.update_prev_action(0.5)
    reward_system.reset()
    assert reward_system.prev_action[0] == 0.0

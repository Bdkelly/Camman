import pytest

from training.reinforcement.reward import RewardSystem


def test_pan_only_reward_prefers_center_and_ignores_vertical_error():
    rewards = RewardSystem()
    centered = rewards.calculate_reward(0, 0, 0)
    assert centered == pytest.approx(1.25)
    assert rewards.calculate_reward(0, 0.4, 0) == centered
    assert rewards.calculate_reward(0.4, 0, 0) < centered
    assert rewards.calculate_reward(0, 0, 0, False) < 0


def test_effort_and_action_change_penalized_and_reset():
    rewards = RewardSystem()
    moving = rewards.calculate_reward(0, 0, 0.5)
    assert moving < rewards.calculate_reward(0, 0, 0)
    rewards.update_prev_action(0.5)
    assert rewards.calculate_reward(0, 0, 0.5) > moving
    rewards.reset()
    assert rewards.prev_action == 0

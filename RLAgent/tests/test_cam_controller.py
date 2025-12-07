import pytest
import numpy as np
import cv2
from RLAgent.camController import CameraControlEnv
from RLAgent.reward import RewardSystem

@pytest.fixture
def env(mocker):
    cap = mocker.Mock()
    cap.get.side_effect = [1280, 720, 0] # W, H, POS_FRAMES
    cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    
    detection_model = mocker.Mock()
    transform = mocker.Mock()
    reward_system = RewardSystem(reward_weights={})
    
    return CameraControlEnv(
        cap=cap,
        detection_model=detection_model,
        transform=transform,
        device='cpu',
        frame_center_x=640,
        frame_center_y=360,
        max_action=1.0,
        reward_system=reward_system
    )

def test_reset(env, mocker):
    mocker.patch('RLAgent.camController.get_ball_detection', return_value=([{'box': (100, 100, 200, 200)}], np.zeros((720, 1280, 3), dtype=np.uint8)))
    state, _ = env.reset()
    assert state.shape == (3,)
    assert np.isclose(state[0], (150 - 640) / 1280)
    assert np.isclose(state[1], (150 - 360) / 720)
    assert state[2] == 0.0

def test_step(env, mocker):
    mocker.patch('RLAgent.camController.get_ball_detection', return_value=([{'box': (100, 100, 200, 200)}], np.zeros((720, 1280, 3), dtype=np.uint8)))
    action = np.array([0.5])
    next_state, reward, done, _, _ = env.step(action)
    assert next_state.shape == (3,)
    assert not done

def test_detect_ball(env, mocker):
    mocker.patch('RLAgent.camController.get_ball_detection', return_value=([{'box': (100, 100, 200, 200)}], np.zeros((720, 1280, 3), dtype=np.uint8)))
    env.current_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    ball_x, ball_y, _ = env.detect_ball()
    assert ball_x == 150
    assert ball_y == 150

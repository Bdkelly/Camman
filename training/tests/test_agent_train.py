from unittest.mock import Mock

import numpy as np
import pytest

from backend.policy import ActorPolicy
from training.reinforcement.train import train_agent, vidget


def test_training_produces_deployable_actor(tmp_path, mocker):
    cap = Mock()
    mocker.patch("training.reinforcement.train.vidget", return_value=(cap, 640, 480))
    mocker.patch("training.reinforcement.train.load_model_from_path", return_value=Mock())
    environment = Mock()
    state = np.array([0.1, 0.2, 0.0, 1.0], dtype=np.float32)
    environment.reset.return_value = (state, None)
    environment.step.return_value = (state, 1.0, True, np.array([0.0]), None)
    mocker.patch("training.reinforcement.train.CameraControlEnv", return_value=environment)
    weights = train_agent(
        "game.mp4", "detector.pth", num_episodes=1, max_t=2, checkpoint_dir=tmp_path, device="cpu"
    )
    assert weights
    assert (tmp_path / "critic_episode_final.pth").is_file()
    policy = ActorPolicy(tmp_path / "actor_episode_final.pth")
    assert policy.choose_action(state).shape == (1,)
    cap.release.assert_called_once()


def test_environment_failure_releases_video(tmp_path, mocker):
    cap = Mock()
    mocker.patch("training.reinforcement.train.vidget", return_value=(cap, 640, 480))
    mocker.patch("training.reinforcement.train.load_model_from_path", return_value=Mock())
    mocker.patch(
        "training.reinforcement.train.CameraControlEnv", side_effect=RuntimeError("bad frames")
    )
    with pytest.raises(RuntimeError, match="bad frames"):
        train_agent("game.mp4", "detector.pth", 1, 1, checkpoint_dir=tmp_path, device="cpu")
    cap.release.assert_called_once()


def test_bad_capture_is_released(mocker):
    cap = Mock()
    cap.isOpened.return_value = False
    mocker.patch("training.reinforcement.train.cv2.VideoCapture", return_value=cap)
    assert vidget("missing.mp4") == (None, 0, 0)
    cap.release.assert_called_once()

import json

import numpy as np
import pytest
import torch

from backend.policy import ActorPolicy
from backend.tracking import TrackingController
from training.reinforcement.tracks import TrackSequence, extract_tracks, synthetic_tracks
from training.reinforcement.train import train_agent


@pytest.fixture
def tracks(tmp_path):
    path = tmp_path / "tracks.npz"
    synthetic_tracks(300).save(path)
    return path


def small_run(tracks, output, episodes=4, **kwargs):
    return train_agent(
        tracks_paths=[tracks],
        checkpoint_dir=output,
        device="cpu",
        cpu_threads=1,
        num_episodes=episodes,
        max_t=20,
        options={
            "batch_size": 8,
            "memory_size": 128,
            "warmup_steps": 8,
            "eval_every": 2,
            "checkpoint_every": 2,
            "seed": 42,
        },
        **kwargs,
    )


def test_training_export_reload_control_and_resume_are_real(tmp_path, tracks, mocker):
    # No mocked actor/critic/environment: this exercises actual optimization.
    hardware = mocker.patch(
        "backend.hardware.serial_connection.open_connection",
        side_effect=AssertionError("Training must not open hardware"),
    )
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    result = small_run(tracks, full)
    small_run(tracks, resumed, episodes=2)
    with (resumed / "metrics.jsonl").open("a") as log:
        log.write('{"episode": 99}\npartial line')
    restored = train_agent(
        checkpoint_dir=resumed,
        resume=resumed / "training_latest.pth",
        num_episodes=4,
        device="cpu",
        cpu_threads=1,
    )
    assert result["updates"] > 0 and result["updates"] == restored["updates"]
    resumed_rows = [json.loads(row) for row in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert [row["episode"] for row in resumed_rows] == [1, 2, 3, 4]
    original = torch.load(full / "training_latest.pth", weights_only=True)
    after_resume = torch.load(resumed / "training_latest.pth", weights_only=True)
    for key in ("actor", "critic", "actor_target", "critic_target"):
        for name, value in original["agent"][key].items():
            torch.testing.assert_close(value, after_resume["agent"][key][name], rtol=0, atol=0)
    assert after_resume["agent"]["actor_optimizer"]["state"]
    assert after_resume["agent"]["critic_optimizer"]["state"]
    assert after_resume["agent"]["replay"]["size"] > 0
    # End-of-file and time-limit truncations must not suppress Bellman bootstrap.
    assert not original["agent"]["replay"]["data"]["done"].any()
    policy = ActorPolicy(full / "actor_episode_best.pth")
    assert policy.metadata["training"]["updates"] > 0
    mocker.patch("backend.tracking.time.monotonic", return_value=10)
    control = TrackingController(policy.spec)
    action = control.update([{"box": (65, 45, 75, 55)}], None, 100, 100, agent=policy)
    assert np.isfinite(action) and -1 <= action <= 1
    rows = [json.loads(row) for row in (full / "metrics.jsonl").read_text().splitlines()]
    best_reward = max(row["validation"]["mean_reward"] for row in rows if "validation" in row)
    assert policy.metadata["training"]["validation"]["mean_reward"] == best_reward
    hardware.assert_not_called()


def test_zero_updates_does_not_export_random_actor(tmp_path, tracks):
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="No optimizer updates"):
        train_agent(
            tracks_paths=[tracks],
            num_episodes=1,
            max_t=2,
            checkpoint_dir=output,
            device="cpu",
            options={"warmup_steps": 100},
            cpu_threads=1,
        )
    assert (output / "training_latest.pth").is_file()  # Can continue later.
    assert not list(output.glob("actor_*.pth"))


def test_resume_rejects_changed_data_and_settings(tmp_path, tracks):
    output = tmp_path / "run"
    small_run(tracks, output, episodes=2)
    with pytest.raises(ValueError, match="differs"):
        train_agent(
            resume=output / "training_latest.pth",
            checkpoint_dir=output,
            num_episodes=4,
            options={"batch_size": 16},
        )
    synthetic_tracks(300, seed=9).save(tracks)
    with pytest.raises(ValueError, match="data differs"):
        train_agent(resume=output / "training_latest.pth", checkpoint_dir=output, num_episodes=4)


def test_train_validation_duplicates_rejected(tmp_path, tracks):
    with pytest.raises(ValueError, match="identical"):
        train_agent(
            tracks_paths=[tracks], validation_paths=[tracks], checkpoint_dir=tmp_path / "run"
        )


def test_track_roundtrip_and_one_pass_extraction(tmp_path, mocker):
    cap = mocker.Mock()
    cap.isOpened.return_value = True
    cap.get.return_value = 30
    cap.read.side_effect = [(True, np.zeros((100, 100, 3), np.uint8))] * 4 + [(False, None)]
    mocker.patch("training.reinforcement.tracks.cv2.VideoCapture", return_value=cap)
    detect = mocker.patch(
        "training.reinforcement.tracks.get_ball_detection",
        return_value=([{"box": (40, 40, 60, 60)}], None),
    )
    data = extract_tracks("game.mp4", mocker.Mock(), "cpu")
    assert detect.call_count == 4 and data.detected.all()
    np.testing.assert_array_equal(data.centers, np.full((4, 2), 0.5))
    cap.release.assert_called_once()
    path = tmp_path / "cache.npz"
    data.save(path)
    loaded = TrackSequence.load(path)
    assert loaded.fingerprint() == data.fingerprint()
    assert loaded.metadata == data.metadata


def test_extraction_error_releases_video(mocker):
    cap = mocker.Mock()
    cap.isOpened.return_value = True
    cap.get.return_value = 30
    cap.read.return_value = (True, np.zeros((20, 20, 3), np.uint8))
    mocker.patch("training.reinforcement.tracks.cv2.VideoCapture", return_value=cap)
    mocker.patch(
        "training.reinforcement.tracks.get_ball_detection", side_effect=RuntimeError("bad frames")
    )
    with pytest.raises(RuntimeError, match="bad frames"):
        extract_tracks("game.mp4", mocker.Mock(), "cpu")
    cap.release.assert_called_once()

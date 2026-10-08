import numpy as np
import pytest

from backend.control import ControlSpec, ObservationBuilder
from backend.tracking import TrackingController
from training.reinforcement.environment import CameraControlEnv, SimulationSpec
from training.reinforcement.tracks import TrackSequence


def sequence(x=0.5, frames=100, fps=30):
    return TrackSequence(np.tile([x, 0.5], (frames, 1)), np.ones(frames, bool), fps)


def test_pan_direction_and_calibrated_time():
    env = CameraControlEnv(
        [sequence()], simulation=SimulationSpec(motor_response_s=0), training=False
    )
    before, _ = env.reset(options={"offset": -0.2})
    after, reward, terminated, truncated, info = env.step([0.5])
    assert after.shape == (7,)
    assert after[0] < before[0]  # Rightward camera motion moves the ball left in the image.
    assert after[2] == 0.5 and after[3] == 1
    assert after[0] == pytest.approx(before[0] - 0.5 * 0.1 * 30 / 90)
    assert info["dt"] == pytest.approx(0.1)
    assert not terminated and not truncated and np.isfinite(reward)


def test_hidden_target_is_not_leaked_and_eventually_stops():
    tracks = sequence()
    tracks.centers[3:] = [0.95, 0.5]
    env = CameraControlEnv([tracks], training=False)
    env.reset()
    state, _, _, _, info = env.step([0.5])
    assert not info["visible"] and not state[3]
    assert state[0] == state[1] == state[4] == 0
    for _ in range(7):
        state, _, _, _, info = env.step([0.5])
    assert info["executed_action"] == 0


def test_eof_is_truncation_and_requires_reset():
    env = CameraControlEnv([sequence(frames=4)], training=False)
    env.reset()
    _, _, terminated, truncated, _ = env.step([0])
    assert truncated and not terminated
    with pytest.raises(RuntimeError, match="Reset"):
        env.step([0])


def test_training_and_live_observations_match(mocker):
    spec = ControlSpec()
    env = CameraControlEnv([sequence()], spec, SimulationSpec(motor_response_s=0), training=False)
    state, _ = env.reset(options={"offset": -0.2})
    clock = mocker.patch("backend.tracking.time.monotonic", return_value=0)
    controller = TrackingController(spec)
    actor = mocker.Mock(choose_action=mocker.Mock(return_value=[0.5]))

    def update_from(state):
        x, y = (state[:2] + 0.5) * 1000
        controller.update([{"box": (x - 5, y - 5, x + 5, y + 5)}], None, 1000, 1000, agent=actor)

    update_from(state)
    np.testing.assert_allclose(controller.last_state, state, atol=1e-6)
    state, _, _, _, _ = env.step([0.5])
    clock.return_value = 0.1
    update_from(state)
    np.testing.assert_allclose(controller.last_state, state, atol=1e-5)


def test_observation_latency_missing_and_timing_validation():
    builder = ObservationBuilder()
    builder.observe((0.5, 0.5), 0, 10, measurement_time=9.9)
    state = builder.observe((0.6, 0.5), 0.4, 10.2, measurement_time=10.1, action_dt=0.2)
    np.testing.assert_allclose(state, [0.1, 0, 0.4, 1, 0.5, 2, 0.2])
    state = builder.observe(None, 0.4, 10.4)
    assert state[3] == 0 and state[6] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        builder.observe((0.5, 0.5), 0, 10.3)
    with pytest.raises(ValueError):
        builder.observe(None, 0, 10.5, action_dt=float("nan"))

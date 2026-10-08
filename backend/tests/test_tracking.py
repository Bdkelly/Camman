from unittest.mock import Mock

import numpy as np
import pytest
import serial
import torch

from backend.actor import Actor
from backend.control import STATE_FIELDS, ControlSpec
from backend.policy import ActorPolicy, export_actor
from backend.tracking import TrackingController


def connection():
    return Mock(write=Mock(side_effect=len))


def ball(x=0.5, y=0.5):
    return [{"box": (x * 1000 - 5, y * 1000 - 5, x * 1000 + 5, y * 1000 + 5)}]


def test_basic_velocity_control_and_rate_limit(mocker):
    clock = mocker.patch("backend.tracking.time.monotonic", return_value=10)
    controller, ser = TrackingController(), connection()
    controller.update(ball(0.8), ser, 1000, 1000)
    ser.write.assert_called_once_with(b"V:0.9000\n")
    controller.update(ball(0.8), ser, 1000, 1000)
    assert ser.write.call_count == 1
    clock.return_value = 10.2
    controller.update(ball(), ser, 1000, 1000)
    ser.write.assert_called_with(b"Stop\n")


def test_actor_receives_contract_and_actual_previous_action(mocker):
    clock = mocker.patch("backend.tracking.time.monotonic", return_value=10)
    agent, ser, controller = Mock(), connection(), TrackingController()
    agent.choose_action.return_value = [0.250012]
    controller.update(ball(), ser, 1000, 1000, agent=agent)
    np.testing.assert_array_equal(agent.choose_action.call_args.args[0], [0, 0, 0, 1, 0, 1, 0])
    clock.return_value = 10.2
    controller.update([], ser, 1000, 1000, agent=agent)
    np.testing.assert_allclose(agent.choose_action.call_args.args[0], [0, 0, 0.25, 0, 0, 2, 0.4])
    ser.write.assert_called_with(b"V:0.2500\n")
    clock.return_value = 10.51
    controller.update([], ser, 1000, 1000, interval=10, agent=agent)
    ser.write.assert_called_with(b"Stop\n")  # Timeout bypasses the rate limit.
    assert agent.choose_action.call_count == 2


@pytest.mark.parametrize("reason", ["disabled", "missing", "stale", "invalid"])
def test_fail_closed_when_motion_cannot_continue(mocker, reason):
    clock = mocker.patch("backend.tracking.time.monotonic", return_value=10)
    controller, ser = TrackingController(), connection()
    controller.update(ball(0.8), ser, 1000, 1000)
    clock.return_value = 11
    if reason == "disabled":
        controller.tick(ser, enabled=False)
    elif reason == "missing":
        controller.tick(ser)
    elif reason == "stale":
        controller.update(ball(0.8), ser, 1000, 1000, observed_at=10.1)
    else:
        with pytest.raises(ValueError):
            controller.update(
                ball(0.8), ser, 1000, 1000, agent=Mock(choose_action=Mock(return_value=[np.nan]))
            )
    ser.write.assert_called_with(b"Stop\n")
    assert controller.prev_action == 0


def test_manual_pulse_then_stop(mocker):
    clock = mocker.patch("backend.tracking.time.monotonic", return_value=10)
    controller, ser = TrackingController(invert_pan=True), connection()
    controller.manual("Right", ser)
    ser.write.assert_called_with(b"V:-0.2500\n")
    controller.tick(ser, enabled=False)
    assert controller.prev_action == 0.25
    clock.return_value = 10.21
    controller.tick(ser, enabled=False)
    ser.write.assert_called_with(b"Stop\n")


def test_stationary_actor_is_rate_limited_without_repeated_stop(mocker):
    clock = mocker.patch("backend.tracking.time.monotonic", return_value=10)
    controller, ser, actor = TrackingController(), connection(), Mock()
    actor.choose_action.return_value = [0]
    controller.update(ball(), ser, 1000, 1000, agent=actor)
    clock.return_value = 10.01
    controller.update(ball(), ser, 1000, 1000, agent=actor)
    assert actor.choose_action.call_count == 1
    ser.write.assert_not_called()


def test_tick_refreshes_wire_without_changing_policy_decision_time(mocker):
    clock = mocker.patch("backend.tracking.time.monotonic", return_value=10)
    controller, ser = TrackingController(), connection()
    controller.update(ball(0.8), ser, 1000, 1000)
    clock.return_value = 10.25
    controller.tick(ser)
    assert ser.write.call_count == 2
    assert controller.last_command_time == 10
    clock.return_value = 10.51
    controller.tick(ser)
    ser.write.assert_called_with(b"Stop\n")


def test_actor_checkpoint_inference_and_real_serial_loopback(tmp_path, mocker):
    actor = Actor(len(STATE_FIELDS), 1, 1.0).eval()
    with torch.no_grad():
        actor.fc3.weight.zero_()
        actor.fc3.bias.fill_(0.5)
    path = tmp_path / "actor.pth"
    export_actor(actor, path, ControlSpec())
    policy = ActorPolicy(path)
    state = np.array([0.1, 0.2, 0, 1, 0, 1, 0], dtype=np.float32)
    with torch.no_grad():
        expected = actor(torch.from_numpy(state).unsqueeze(0)).numpy().reshape(-1)
    np.testing.assert_allclose(policy.choose_action(state), expected)
    assert not policy.actor.training and not hasattr(policy, "critic_local")
    mocker.patch("backend.tracking.time.monotonic", return_value=10)
    with serial.serial_for_url("loop://", timeout=0.1) as ser:
        controller = TrackingController(policy.spec)
        controller.update(ball(0.6), ser, 1000, 1000, agent=policy)
        assert ser.readline() == b"V:0.4621\n"
        controller.stop(ser)
        assert ser.readline() == b"Stop\n"


@pytest.mark.parametrize("invalid", ["legacy", "scale", "nonfinite"])
def test_reject_incompatible_policy(tmp_path, invalid):
    path = tmp_path / "actor.pth"
    actor = Actor(len(STATE_FIELDS), 1, 1.0)
    saved = export_actor(actor, path, ControlSpec())
    if invalid == "legacy":
        saved = actor.state_dict()
    elif invalid == "scale":
        saved["action_schema"] = "absolute_position"
    else:
        saved["actor_state_dict"]["fc3.bias"].fill_(float("nan"))
    torch.save(saved, path)
    with pytest.raises(ValueError):
        ActorPolicy(path)

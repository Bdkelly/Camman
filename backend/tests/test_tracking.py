from unittest.mock import Mock

import numpy as np
import torch

from backend.actor import Actor
from backend.policy import ActorPolicy
from backend.tracking import TrackingController


def test_basic_control_and_rate_limit(mocker):
    clock = mocker.patch("backend.tracking.time.monotonic", return_value=10)
    control, serial = TrackingController(), Mock()
    boxes = [{"box": (0, 100, 100, 200)}]
    control.update(boxes, serial, 640, 480, 1.0)
    serial.write.assert_called_once_with(b"Left\n")
    control.update(boxes, serial, 640, 480, 1.0)
    assert serial.write.call_count == 1
    clock.return_value = 12
    control.update([{"box": (300, 100, 340, 200)}], serial, 640, 480, 1.0)
    serial.write.assert_called_with(b"Stop\n")


def test_actor_receives_detection_and_previous_action(mocker):
    mocker.patch("backend.tracking.time.monotonic", side_effect=[10, 12])
    agent, serial, controller = Mock(), Mock(), TrackingController()
    agent.choose_action.return_value = np.array([0.25])
    controller.update([{"box": (300, 220, 340, 260)}], serial, 640, 480, 1, agent=agent)
    np.testing.assert_array_equal(agent.choose_action.call_args.args[0], [0, 0, 0, 1])
    controller.update([], serial, 640, 480, 1, agent=agent)
    np.testing.assert_array_equal(agent.choose_action.call_args.args[0], [0, 0, 0.25, 0])
    serial.write.assert_called_with(b"P:0.25,T:0.00\n")


def test_actor_checkpoint_inference_matches_network(tmp_path):
    actor = Actor(4, 1, 1.0).eval()
    path = tmp_path / "actor.pth"
    torch.save(actor.state_dict(), path)
    policy = ActorPolicy(path)
    state = np.array([0.1, 0.2, 0.0, 1.0], dtype=np.float32)
    with torch.no_grad():
        expected = actor(torch.from_numpy(state).unsqueeze(0)).numpy().reshape(-1)
    np.testing.assert_allclose(policy.choose_action(state), expected)
    assert not policy.actor.training
    assert not hasattr(policy, "critic_local")
    assert not hasattr(policy, "memory")

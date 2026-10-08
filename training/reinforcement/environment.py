"""Pan a virtual viewport over cached wide-view trajectories. Never opens hardware."""

import math
from dataclasses import dataclass

import numpy as np

from backend.control import ControlSpec, ObservationBuilder, guarded_pan
from training.reinforcement.reward import RewardSystem

# Version 2 matches CAMMAN/1 Stop: cancel motion rather than coast through zero.
SIMULATOR_VERSION = 2


@dataclass(frozen=True)
class SimulationSpec:
    view_fraction: float = 0.5
    motor_response_s: float = 0.08
    dropout: float = 0.05
    randomization: float = 0.2

    def __post_init__(self):
        if not 0 < self.view_fraction < 1 or not 0 <= self.motor_response_s <= 2:
            raise ValueError("View fraction must be 0..1 and motor response 0..2 seconds")
        if not 0 <= self.dropout < 1 or not 0 <= self.randomization < 1:
            raise ValueError("Dropout/randomization must be 0..1")


class CameraControlEnv:
    def __init__(
        self,
        sequences,
        spec=None,
        simulation=None,
        *,
        max_steps=1000,
        seed=0,
        training=True,
        reward_system=None,
    ):
        if (
            not sequences
            or max_steps < 1
            or any(not sequence.detected.any() for sequence in sequences)
        ):
            raise ValueError(
                "Each training/validation trajectory needs detections and at least two frames"
            )
        self.sequences = sequences
        self.spec = spec or ControlSpec()
        self.simulation = simulation or SimulationSpec()
        self.max_steps, self.training = max_steps, training
        self.rng = np.random.default_rng(seed)
        self.observations = ObservationBuilder(self.spec)
        self.reward_system = reward_system or RewardSystem()
        self.finished = True

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        sequence_index = options.get("sequence", int(self.rng.integers(len(self.sequences))))
        self.sequence = self.sequences[sequence_index]
        self.stride = max(1, round(self.spec.period * self.sequence.fps))
        candidates = np.flatnonzero(self.sequence.detected[:-1])
        if not len(candidates):
            raise ValueError("Trajectory has no observable starting frame before EOF")
        self.index = int(
            options.get("start", self.rng.choice(candidates) if self.training else candidates[0])
        )
        if not 0 <= self.index < len(self.sequence.centers) - 1:
            raise ValueError("Episode start must precede the last frame")
        width = self.simulation.view_fraction
        offset = options.get("offset", self.rng.uniform(-0.35, 0.35) if self.training else 0.0)
        self.camera_center = float(
            np.clip(self.sequence.centers[self.index, 0] + offset * width, width / 2, 1 - width / 2)
        )
        self.gain = (
            self.rng.uniform(1 - self.simulation.randomization, 1 + self.simulation.randomization)
            if self.training
            else 1.0
        )
        self.time = self.index / self.sequence.fps
        self.previous_pan = self.velocity = 0.0
        self.steps = 0
        self.finished = False
        self.reward_system.reset()
        self.observations.reset()
        return self._observe(self.spec.period)

    def _observe(self, dt):
        world_x, world_y = self.sequence.centers[self.index]
        x = (world_x - self.camera_center) / self.simulation.view_fraction + 0.5
        known = bool(self.sequence.detected[self.index])
        visible = bool(known and 0 <= x <= 1)
        observed = visible and (not self.training or self.rng.random() >= self.simulation.dropout)
        state = self.observations.observe(
            (x, world_y) if observed else None, self.previous_pan, self.time, action_dt=dt
        )
        info = {
            "visible": visible,
            "observed": observed,
            "known_target": known,
            "dx": float(x - 0.5),
            "dy": float(world_y - 0.5),
            "dt": dt,
            "camera_center": self.camera_center,
            "frame_index": self.index,
            "executed_action": self.previous_pan,
        }
        return state, info

    def step(self, action):
        if self.finished:
            raise RuntimeError("Reset the environment before stepping")
        executed = guarded_pan(action, self.spec, self.observations.target_age)
        stride = self.stride
        if self.training and self.simulation.randomization and stride > 1:
            stride = max(1, stride + int(self.rng.choice([-1, 0, 0, 0, 1])))
        next_index = min(self.index + stride, len(self.sequence.centers) - 1)
        dt = (next_index - self.index) / self.sequence.fps
        tau = self.simulation.motor_response_s
        if executed == 0:
            # The deployed firmware calls stopMotion() for Stop/V:0, including
            # commands zeroed by the deadband or lost-target guard.
            self.velocity = integrated_velocity = 0.0
        elif tau:
            decay = math.exp(-dt / tau)
            integrated_velocity = executed * dt + (self.velocity - executed) * tau * (1 - decay)
            self.velocity = executed + (self.velocity - executed) * decay
        else:
            self.velocity = executed
            integrated_velocity = executed * dt
        width = self.simulation.view_fraction
        center = self.camera_center + integrated_velocity * self.spec.pan_rate * width * self.gain
        self.camera_center = float(np.clip(center, width / 2, 1 - width / 2))
        if center != self.camera_center:
            self.velocity = 0.0
        self.index = next_index
        self.time += dt
        self.steps += 1
        self.previous_pan = executed
        state, info = self._observe(dt)
        reward = self.reward_system.calculate_reward(
            info["dx"], info["dy"], executed, info["visible"]
        )
        self.reward_system.update_prev_action(executed)
        # EOF and the step budget truncate a continuing task; neither is an
        # absorbing failure state. DDPG must still bootstrap from this last state.
        truncated = self.index == len(self.sequence.centers) - 1 or self.steps >= self.max_steps
        self.finished = truncated
        return state, reward, False, truncated, info

"""The observation/action contract shared by simulation and camera deployment."""

import math
from dataclasses import asdict, dataclass

import numpy as np

STATE_FIELDS = (
    "dx",
    "dy",
    "previous_pan",
    "detected",
    "image_velocity_x",
    "dt_ratio",
    "target_age_ratio",
)
ACTION_SCHEMA = "normalized_pan_velocity_v1"


@dataclass(frozen=True)
class ControlSpec:
    control_hz: float = 10.0
    camera_hfov_deg: float = 90.0
    max_pan_speed_deg_s: float = 30.0
    lost_target_timeout_s: float = 0.5
    max_frame_age_s: float = 0.5
    deadband: float = 0.02

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid control setting {name}: {value}")
        if not 1 <= self.control_hz <= 60:
            raise ValueError("control_hz must be 1..60")
        if not 1 <= self.camera_hfov_deg < 180 or self.max_pan_speed_deg_s <= 0:
            raise ValueError("Set a valid camera horizontal FOV and positive pan speed")
        if self.lost_target_timeout_s <= 0 or self.max_frame_age_s <= 0 or self.deadband >= 1:
            raise ValueError("Timeouts must be positive and deadband must be less than 1")

    @property
    def period(self):
        return 1.0 / self.control_hz

    @property
    def pan_rate(self):
        """Camera widths per second at normalized action +1."""
        return self.max_pan_speed_deg_s / self.camera_hfov_deg


def normalized_pan(action):
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError("Pan action must contain one finite number")
    return float(np.clip(value[0], -1.0, 1.0))


def guarded_pan(action, spec, target_age):
    action = normalized_pan(action)
    if target_age >= spec.lost_target_timeout_s or abs(action) < spec.deadband:
        return 0.0
    # State history and simulation use the same quantization as the wire format.
    return round(action, 4)


def box_center(boxes, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    if not boxes:
        return None
    coords = np.asarray(boxes[0]["box"], dtype=np.float64)
    if coords.shape != (4,) or not np.isfinite(coords).all():
        raise ValueError("Ball box must have four finite coordinates")
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        float(np.clip((x1 + x2) / (2 * width), 0, 1)),
        float(np.clip((y1 + y2) / (2 * height), 0, 1)),
    )


class ObservationBuilder:
    def __init__(self, spec=None):
        self.spec = spec or ControlSpec()
        self.reset()

    def reset(self):
        self.last_time = self.last_seen = self.last_measurement = None
        self.last_dx = None
        self.target_age = self.spec.lost_target_timeout_s

    def observe(self, center, previous_pan, now, *, measurement_time=None, action_dt=None):
        measurement_time = now if measurement_time is None else measurement_time
        if not math.isfinite(now) or not math.isfinite(measurement_time):
            raise ValueError("Observation timestamps must be finite")
        if self.last_time is not None and now < self.last_time:
            raise ValueError("Observation time must be monotonic")
        dt = (
            action_dt
            if action_dt is not None
            else (self.spec.period if self.last_time is None else now - self.last_time)
        )
        if not math.isfinite(dt) or dt < 0 or measurement_time > now + 1e-6:
            raise ValueError("Observation timing must be finite, nonnegative and not in the future")
        if self.last_measurement is not None and measurement_time < self.last_measurement:
            raise ValueError("Measurement time must be monotonic")
        dx = dy = velocity = 0.0
        if center is not None:
            if len(center) != 2 or not np.isfinite(center).all():
                raise ValueError("Ball center must contain two finite coordinates")
            dx, dy = np.clip(center, 0, 1) - 0.5
            elapsed = (
                0 if self.last_measurement is None else measurement_time - self.last_measurement
            )
            if self.last_dx is not None and elapsed > 0:
                velocity = np.clip((dx - self.last_dx) / elapsed, -5, 5)
            self.last_dx, self.last_measurement = dx, measurement_time
            self.last_seen = measurement_time
        else:
            self.last_dx = self.last_measurement = None
        self.target_age = (
            self.spec.lost_target_timeout_s
            if self.last_seen is None
            else max(0, now - self.last_seen)
        )
        self.last_time = now
        return np.array(
            [
                dx,
                dy,
                normalized_pan(previous_pan),
                float(center is not None),
                velocity,
                np.clip(dt / self.spec.period, 0, 10),
                np.clip(self.target_age / self.spec.lost_target_timeout_s, 0, 2),
            ],
            dtype=np.float32,
        )

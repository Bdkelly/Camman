"""Policy execution and actuator lifecycle, usable with Qt, headless or dry-run sinks."""

import math
import time

from backend.control import ControlSpec, ObservationBuilder, box_center, guarded_pan
from backend.hardware.protocol import velocity_command
from backend.hardware.serial_connection import send_agent_command


class TrackingController:
    def __init__(self, spec=None, *, invert_pan=False, proportional_gain=3.0, clock=None):
        self.spec = spec or ControlSpec()
        self.observations = ObservationBuilder(self.spec)
        self.invert_pan = invert_pan
        self.proportional_gain = proportional_gain
        self.last_command_time = None
        self.prev_action = 0.0
        self.last_state = None
        self._manual_until = None
        self._last_write = None
        self._clock = clock

    def _now(self):
        # An injected monotonic clock makes simulated deployment checks
        # deterministic without changing the process-wide time functions.
        return self._clock() if self._clock is not None else time.monotonic()

    def _send(self, action, ser, log, now, *, force=False):
        command = velocity_command(action, invert=self.invert_pan)
        if action == 0 and self.prev_action == 0 and not force:
            return 0.0
        if ser is None:
            if log:
                log(f"[dry-run] {command}")
        else:
            send_agent_command(ser, command, log)
        # Only record an action once the complete write succeeds.
        self.prev_action = action
        self.last_command_time = now
        self._last_write = now
        return action

    def stop(self, ser, *, log=None, force=False, reset=False):
        action = self._send(0.0, ser, log, self._now(), force=force)
        self._manual_until = None
        if reset:
            self.observations.reset()
            self.last_state = None
        return action

    def tick(self, ser, *, enabled=True, log=None):
        """Call even when no video frame arrives, to expire motion requests."""
        now = self._now()
        if self._manual_until is not None:
            if now < self._manual_until:
                return
            self.stop(ser, log=log, reset=True)
        seen = self.observations.last_seen
        if not enabled or seen is None or now - seen >= self.spec.lost_target_timeout_s:
            self.stop(ser, log=log, reset=not enabled)
        elif self.prev_action and self._last_write is not None and now - self._last_write >= 0.2:
            # Keep velocity alive during a slow policy cadence. A blocked worker
            # stops refreshing and the firmware watchdog takes over.
            send_agent_command(ser, velocity_command(self.prev_action, invert=self.invert_pan), log)
            self._last_write = now

    def manual(self, command, ser, *, log=None, duration=0.2):
        if command not in {"Left", "Right", "Stop"}:
            raise ValueError("Manual command must be Left, Right or Stop")
        if command == "Stop":
            return self.stop(ser, log=log, force=True, reset=True)
        if not 0 < duration <= 0.5:
            raise ValueError("Manual pulse duration must be 0..0.5 seconds")
        now = self._now()
        action = -0.25 if command == "Left" else 0.25
        self._send(action, ser, log, now, force=True)
        self._manual_until = now + duration
        return action

    def update(
        self, boxes, ser, width, height, interval=None, *, agent=None, log=None, observed_at=None
    ):
        now = self._now()
        observed_at = now if observed_at is None else observed_at
        interval = self.spec.period if interval is None else interval
        try:
            if not math.isfinite(interval) or interval < 0:
                raise ValueError("Command interval must be finite and nonnegative")
            if not math.isfinite(observed_at) or observed_at > now + 1e-6:
                raise ValueError("Invalid frame observation timestamp")
            if self._manual_until is not None:
                if now < self._manual_until:
                    return None
                self.stop(ser, log=log, reset=True)
            center = box_center(boxes, width, height)
            if now - observed_at > self.spec.max_frame_age_s:
                center = None
            dt = (
                self.spec.period if self.last_command_time is None else now - self.last_command_time
            )
            state = self.observations.observe(
                center, self.prev_action, now, measurement_time=observed_at, action_dt=dt
            )
            self.last_state = state
            if self.observations.target_age >= self.spec.lost_target_timeout_s:
                return self.stop(ser, log=log)
            if self.last_command_time is not None and now - self.last_command_time < interval:
                return None
            # With no actor, use the same velocity protocol and normalized units.
            action = (
                agent.choose_action(state)
                if agent is not None
                else self.proportional_gain * state[0]
                if state[3]
                else 0.0
            )
            action = guarded_pan(action, self.spec, self.observations.target_age)
            result = self._send(action, ser, log, now)
            # A stationary policy still consumes one decision interval, even
            # when its duplicate Stop does not need another serial write.
            self.last_command_time = now
            return result
        except Exception:
            # Invalid observations/policy output must not leave the last velocity active.
            self.stop(ser, log=log, force=True, reset=True)
            raise

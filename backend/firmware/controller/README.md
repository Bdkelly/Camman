# CAMMAN/1 pan controller

This is the supported serial firmware for Camman's normalized pan-velocity
policy. It targets a classic ESP32 Dev Module plus a **STEP/DIR stepper driver**.
It does not drive a motor directly from GPIO. The parent directory's older
sketches are retained as references; their position protocol is incompatible.

## Wiring and calibration

Edit the constants at the top of [src/main.cpp](src/main.cpp) for your mechanism
before uploading. These are example settings, not measurements of your rig:

| Setting | Example | Must match |
| --- | --- | --- |
| STEP / DIR | GPIO 14 / 27 | Your ESP32 and driver wiring; these differ from the old sketch |
| Motor steps per revolution | 200 | Motor specification |
| Microsteps | 16 | Physical driver switches/configuration |
| Gear ratio | 1 | Motor revolutions per camera revolution |
| Maximum pan speed | 30 degrees/s | Actor's `--max-pan-speed-deg-s` training setting |
| Acceleration | 120 degrees/s² | Mechanism capability; tune the simulation response too |
| Travel | ±90 degrees | Available travel from the centered boot position |
| Command watchdog | 750 ms | Host must refresh velocity faster than this |

Connect driver STEP/DIR and a common signal ground; use a driver/motor supply
appropriate for the hardware and ESP32-compatible logic levels. Driver enable,
current limiting and power wiring depend on your driver. The example has no
limit-switch inputs or homing routine: **manually center before boot**. Software
travel limits are step estimates and cannot detect missed steps or lost position.
Measure actual pan speed and direction. `--invert-pan` reverses the application
output while retaining the policy's convention that positive pans right.

## Build and upload

Run from the repository root. PlatformIO Core supplies the compiler and Arduino
framework. The project pins Espressif32 7.1.3 and AccelStepper 1.64.0.

```sh
python -m pip install 'platformio>=6.1,<7'
pio settings set enable_telemetry No
pio run -d backend/firmware/controller
```

After configuring the wiring/calibration, upload to the correct board/port:

```sh
pio run -d backend/firmware/controller --target upload --upload-port COM3
```

Use `/dev/ttyUSB0` or your assigned port on Linux. Compilation has been checked;
flashing, physical direction, speed, stopping distance and travel limits need
validation on your mechanism. The Python/native tests never flash a board.

## Protocol for ESP32 or another microcontroller

USB/UART is **115200 baud, 8N1**, ASCII, one command per newline. CRLF is accepted.
Only `HELLO` and `Stop` need responses; ordinary velocity commands have no reply
so unread acknowledgements cannot fill the serial buffer.

| Host line | Controller response / behavior |
| --- | --- |
| `HELLO` | `CAMMAN/1 VELOCITY` followed by newline |
| `V:0.2500` | Request +25% of calibrated maximum pan speed |
| `V:-1.0000` | Request full speed left |
| `V:0.0000` or `Stop` | Stop stepping immediately; `Stop` replies `Stopping` |
| Malformed, non-finite, out-of-range or overlong line | Stop, then reply `ERR invalid command` when buffer space permits |
| No valid velocity command for 750 ms | Stop, independently of Python |

The application sends `Stop`, then requires the exact version greeting at
connection time. Implement these semantics on an Arduino, RP2040, STM32 or other
controller to reuse the Python backend. Normalize velocity to the same measured
maximum speed, enforce a watchdog and local travel limits, and keep stepping
nonblocking. `V:` is deliberately distinct from legacy absolute-position `P:`.
The greeting verifies protocol support, **not** physical calibration.

The motor loop ramps velocity, brakes near software limits and runs
`AccelStepper.runSpeed()` without blocking serial reads. `Stop` and watchdog
expiry cancel motion immediately; they do not perform a deceleration ramp.
A mechanical system can still coast or miss steps, so measure the result.

## First controlled movement

The following command performs a 200 ms pulse, then sends `Stop` before closing:

```sh
python -m backend.tools.motormove --port COM3 --pan 0.25 --duration 0.2
```

Use `--command Stop` to test the handshake/stop path without requesting motion.
The utility accepts pulses up to 0.5 seconds. Full tracking setup is in the
[backend guide](../../README.md#tracking-and-actor-inference).

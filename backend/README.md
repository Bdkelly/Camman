# Backend: detection, tracking and hardware

[Project overview](../README.md) · [Frontend guide](../frontend/README.md) · [Training guide](../training/README.md)

The backend provides the Python code that turns a video frame into a ball
detection and, optionally, a camera-pan command. It contains the shared detector
and actor definitions, checkpoint loading, tracking rules and hardware transport.

It is an **in-process library**, not a web service. There is no backend server,
HTTP port or `python -m backend` launcher to start. Run the frontend to use it
interactively, run `camman-track` without a GUI, import it in a Python program,
or run the diagnostics below.

## How it connects to the other components

| Caller | Uses the backend for | Receives |
| --- | --- | --- |
| [Frontend](../frontend/README.md) | Loading detector/actor weights, detecting balls and sending control commands | Boxes, annotated frames, actions and log callbacks |
| [Detector training](../training/detection/train.py) | Constructing the same detector architecture the application loads | A trainable PyTorch model |
| [Camera-policy training](../training/reinforcement/train.py) | Shared observations, calibrated actions and actor architecture | Versioned deployable actor; detector runs once when caching tracks |
| [Annotation tools](../training/data/auto_annotate.py) | Running a trained detector over images | Candidate bounding boxes |

The backend does not import the frontend, Qt, Albumentations or the training
agent. Application inference creates an `ActorPolicy` containing only the actor
network; the critic, optimizers and replay buffer belong to training. Hardware
logging uses an optional callable such as `print`, which the frontend can adapt
to a Qt signal.

## Install and run

For **Jetson Orin Nano Super**, follow the [native installation and tuning
guide](JETSON.md). It avoids replacing Jetson's CUDA/vision binaries with generic
pip wheels and includes USB, CSI and hardware-decoded video examples.

Use the [project setup instructions](../README.md#install) to create and activate
an environment. Run these commands from the repository root. If using NVIDIA,
install a matching CUDA-enabled PyTorch/torchvision pair with the
[official selector](https://pytorch.org/get-started/locally/) before the project:

```sh
python -m pip install -e .
python -m backend.tools.check_device
```

This installs the backend dependencies without the Qt or training extras.
`check_device` reports PyTorch/torchvision/OpenCV versions, CUDA availability,
GPU name and GStreamer support. Add `--device cuda --verify-ops` to execute NMS
and ROIAlign on the GPU. It is a device diagnostic, not a performance benchmark.

For the complete application, install `.[frontend]` and launch `camman` as
described in the [frontend guide](../frontend/README.md).

## Run detection from Python

First obtain a detector using the [training walkthrough](../training/README.md).
Save the following example as a local script in the repository root and run it
with the active environment. Replace the image/model paths with your files.
It performs inference without starting Qt or connecting to hardware.

```python
from pathlib import Path

import cv2
import torch

from backend.detection import FrameTransform, get_ball_detection
from backend.models import load_model_from_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_from_path("artifacts/detector_v1/trained_model_final.pth", device)
transform = FrameTransform()
frame = cv2.imread("data/game_01/frames/frame_00000.jpg")
if frame is None:
    raise FileNotFoundError("Check the input image path")

detections, annotated = get_ball_detection(
    model, frame, transform, device, confidence_threshold=0.98
)
print(detections)

Path("artifacts").mkdir(exist_ok=True)
if not cv2.imwrite("artifacts/detection-preview.jpg", annotated):
    raise OSError("Could not write the preview")
```

Each detection is a dictionary with `box`, `label` and `score`. `box` is a tuple
of `(x_min, y_min, x_max, y_max)` in **original-frame pixels**, `label` is `Ball`,
and `score` is the confidence value. No detection produces an empty list.

The input frame is a BGR NumPy image, as returned by OpenCV. The backend converts
it to RGB, applies the shared preprocessing, runs inference, and maps boxes back
to the original dimensions. By default it returns at most one ball and draws on
the supplied frame. Use `frame.copy()` to preserve the original, `draw=False` to
skip drawing, or `max_detections=None` to return all accepted ball boxes. For a
video loop, construct the model and transform **once**, outside the loop.

## Model files and compatibility

| File | Loader / purpose |
| --- | --- |
| Detector `trained_model_final.pth` | `load_model_from_path(path, device)` loads the raw detector state dictionary |
| Detector `checkpoint_epoch_N.pth` | The same loader extracts `model_state_dict` from the training checkpoint |
| Actor `actor_episode_best.pth` or `actor_episode_final.pth` | `ActorPolicy(path, device)` loads a camera-control policy |
| Critic `critic_episode_*.pth` | Used by training; not loaded by the application |

The detector is currently Faster R-CNN with a ResNet-50/FPN backbone and two
class IDs: background `0`, ball `1`. Existing detector loading is offline: it
builds the architecture without downloading COCO weights and then loads the
specified state dictionary. Missing files and incompatible weights raise an
error. The loader preserves the frozen batch-normalization convention of legacy
COCO-initialized checkpoints and supports ordinary batch normalization for
checkpoints trained with `--no-pretrained`.

`FrameTransform` preserves the existing 640×640 resize and external ImageNet
normalization. Faster R-CNN also normalizes internally. Coordinate changes to
training and preprocessing must therefore be coordinated with inference and
model retraining; do not change this convention for just one component.
There is currently no NCNN/ONNX/TensorRT loading path.

## Deployment profiles and measurement

[runtime.py](runtime.py) contains application-only settings. Jetson auto-detection
selects CUDA FP16 autocast, 640px internal detector inputs, two Torch CPU threads
and one OpenCV thread. Standard deployment keeps FP32 and 800px detector input.
Explicit options override these defaults; training does not apply the profile.
The external 640px preprocessing and checkpoint normalization stay unchanged.

To apply the same profile from Python:

```python
from backend.runtime import resolve_runtime

runtime = resolve_runtime(profile="jetson", device="cuda")
runtime.apply()
model = load_model_from_path(
    "artifacts/detector_v1/trained_model_final.pth",
    runtime.device,
    detector_size=runtime.detector_size,
)
detections, annotated = get_ball_detection(
    model, frame, FrameTransform(), runtime.device, precision=runtime.precision
)
```

This continues the imports/frame setup in the earlier Python example. The model
loader's default `detector_size=None` and detection's default `precision="fp32"`
preserve existing training/annotation callers. FP16 requires CUDA. Compare
detection accuracy after changing resolution or precision.

[capture.py](capture.py) opens explicit OpenCV/GStreamer backends and provides
`FrameReader`: live sources retain one newest waiting frame; files apply back
pressure and preserve order. Capture owns read/release on its reader thread.
[tools/benchmark.py](tools/benchmark.py) measures batch-one inference without
Qt or motors:

```sh
python -m backend.tools.benchmark --profile jetson --device cuda --source data/game_01/game.mp4 --model artifacts/detector_v1/trained_model_final.pth --frames 200 --output artifacts/benchmarks/jetson.json
```

Warmup is excluded; CUDA is synchronized for timing. Reports include mean/p50/
p95 detection time, capture time, sequential throughput and peak PyTorch CUDA
allocation. This is not GUI FPS, total board RAM usage, or an accuracy score.
See [the Jetson guide](JETSON.md#benchmark-and-tune) for a baseline comparison
and the remaining hardware validation.

## Tracking and actor inference

Run the complete capture → detection → policy → controller pipeline without Qt:

```sh
camman-track --source 0 --model artifacts/detector_v1/trained_model_final.pth --actor artifacts/policy_v1/actor_episode_best.pth --profile jetson --device cuda
```

Omitting `--serial-port` is a **dry-run**: decisions are computed and logged, but
no port is opened. Add `--serial-port /dev/ttyUSB0` (or `COM3` on Windows) to drive
a configured controller. Headless tracking starts immediately; Ctrl+C, EOF,
inference failures and frame limits send `Stop` before closing. `--frames 300`
limits a trial. It supports the GUI's capture, device and precision options,
`--inference-fps`, `--confidence` and `--invert-pan`.
`python -m backend.tools.track` is equivalent.

`TrackingController` is reusable by the GUI, headless runner or a future TUI.
Create it with `TrackingController(policy.spec)`, call `update` with each frame's
boxes and capture timestamp (`observed_at`), and call `tick` regularly even when
frames stop arriving. Call `stop(..., force=True)` in your shutdown path. A custom
sink can provide a pyserial-compatible `write(bytes)` returning the byte count.

[control.py](control.py) defines the seven-value observation used by **both**
the simulator and live application:

| Field | Meaning |
| --- | --- |
| `dx`, `dy` | Target offset from image center in fractions of width/height; zero when missing |
| `previous_pan` | Last executed normalized pan request, before wiring inversion |
| `detected` | Target present in the current usable frame |
| `image_velocity_x` | Horizontal image motion in frame widths/s, clipped to ±5; reset across gaps |
| `dt_ratio` | Elapsed decision time divided by the policy's configured period, clipped to 0–10 |
| `target_age_ratio` | Time since the last observed target divided by the loss timeout, clipped to 0–2 |

The actor outputs **one pan velocity in [-1, 1]**; +1 means right at the saved
maximum degrees/s. It does not output a position or control tilt. Its versioned
checkpoint stores field order, action semantics, architecture, horizontal FOV,
maximum speed, decision frequency, deadband and freshness/loss timeouts. The GUI
uses the saved frequency and locks its interval slider for an actor. Inference
can run slower than that rate; the observation reports elapsed time, but slow
hardware still needs a policy trained/evaluated for its measured cadence.

Without an actor, the controller uses proportional pan (`3 × dx`, bounded to
[-1, 1]) and stops immediately on missing detections. With an actor, short gaps
can be predicted through, but missing/stale observations trigger a forced stop
at the configured loss timeout (default 0.5 s). Frames older than the freshness
limit (default 0.5 s) do not refresh target visibility. Frame age starts when
OpenCV returns a frame; camera/decoder buffering is additional latency.

Invalid actor output stops motion. Disabling tracking/inference stops it on the
next worker iteration, and the firmware watchdog covers a blocked/crashed host.
Manual left/right are bounded 200 ms velocity pulses. The GUI's **Stop Motion**
button disables automatic control and queues `Stop`; it is not a hardware stop
switch. Large inference calls can delay host commands, so firmware-side timeout
and limits remain necessary.

Legacy raw actor weights are rejected: their old four-value state and pan scale
cannot be safely inferred. [Retrain using the new pipeline](../training/README.md#optional-train-the-camera-control-policy).
Detector weights keep their existing compatibility.

## Hardware connections

Use the [supported firmware and calibration guide](firmware/controller/README.md).
`open_connection("COM3")` uses 115200 baud and bounded read/write timeouts.
`open_connection("auto")` probes for a `Stopping` reply; an explicitly requested
controller must then pass `verify_velocity_controller`. The frontend disables
motion on failed handshakes, while headless startup reports an error. Omitting
a port leaves the application in dry-run mode.

| Message, before newline | Meaning |
| --- | --- |
| `HELLO` → `CAMMAN/1 VELOCITY` | Required version/capability handshake |
| `V:0.2500` | +25% normalized pan velocity |
| `V:-0.2500` | −25% normalized pan velocity |
| `Stop` → `Stopping` | Stop motion |

Serial writes are checked for complete delivery. The host refreshes active
velocity during `tick`; the firmware stops after 750 ms without a valid velocity
command. Calibration is a separate physical measurement: the handshake cannot
verify motor wiring, speed, gearing or camera FOV. Other microcontrollers can
implement the same newline protocol and lifecycle.

```sh
python -m backend.tools.probe
python -m backend.tools.motormove --port COM3 --command Stop
python -m backend.tools.motormove --port COM3 --pan 0.25 --duration 0.2
```

The last command moves real hardware briefly and always attempts `Stop` before
closing. Bluetooth discovery helpers remain experimental; the tracking loop
uses serial. No BLE motor-control integration is implied by a status indicator.

## Code map and tests

| Location | Responsibility |
| --- | --- |
| [models.py](models.py) | Shared detector factory and checkpoint loading |
| [detection.py](detection.py) | Preprocessing, prediction filtering, coordinate scaling and overlays |
| [actor.py](actor.py), [policy.py](policy.py) | Shared actor architecture and inference-only wrapper |
| [tracking.py](tracking.py) | Rate-limited policy execution, freshness checks, manual pulses and stop lifecycle |
| [control.py](control.py) | Shared observation/action contract and camera calibration |
| [config.py](config.py) | Writable detector directory and `CAMMAN_MODELS_DIR` override |
| [runtime.py](runtime.py), [capture.py](capture.py) | Deployment profiles and bounded camera/file capture |
| [hardware/](hardware/) | Serial commands, port probing and Bluetooth helpers |
| [tools/](tools/) | Headless tracking, diagnostics and detector benchmarking |
| [firmware/](firmware/) | Supported velocity firmware plus legacy reference variants |
| [tests/](tests/) | Model, transport, tracking and component-boundary checks |

For tests, use the full development environment: the training extras support
the preprocessing comparison, and Qt is required by the shared test plugin:

```sh
python -m pip install -e ".[frontend,training,dev]"
python -m pytest backend/tests
```

If CUDA is unavailable, verify the active environment and PyTorch installation.
For model key/shape errors, check that the file is a detector rather than an
actor/critic and matches this architecture. If port probing misses a board after
reset, pass its known port explicitly. Tests include serial loopback and the firmware parser (with `g++` installed).
Build the ESP32 project separately with PlatformIO; physical platform behavior
still requires validation on your rig.

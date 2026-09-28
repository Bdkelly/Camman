# Backend: detection, tracking and hardware

[Project overview](../README.md) · [Frontend guide](../frontend/README.md) · [Training guide](../training/README.md)

The backend provides the Python code that turns a video frame into a ball
detection and, optionally, a camera-pan command. It contains the shared detector
and actor definitions, checkpoint loading, tracking rules and hardware transport.

It is an **in-process library**, not a web service. There is no backend server,
HTTP port or `python -m backend` launcher to start. Run the frontend to use it
interactively, import it in a Python program, or run the diagnostics below.

## How it connects to the other components

| Caller | Uses the backend for | Receives |
| --- | --- | --- |
| [Frontend](../frontend/README.md) | Loading detector/actor weights, detecting balls and sending control commands | Boxes, annotated frames, actions and log callbacks |
| [Detector training](../training/detection/train.py) | Constructing the same detector architecture the application loads | A trainable PyTorch model |
| [Camera-policy training](../training/reinforcement/train.py) | Detector inference and the shared actor architecture | Ball positions and compatible actor weights |
| [Annotation tools](../training/data/auto_annotate.py) | Running a trained detector over images | Candidate bounding boxes |

The backend does not import the frontend, Qt, Albumentations or the training
agent. Application inference creates an `ActorPolicy` containing only the actor
network; the critic, optimizers and replay buffer belong to training. Hardware
logging uses an optional callable such as `print`, which the frontend can adapt
to a Qt signal.

## Install and run

Use the [project setup instructions](../README.md#install) to create and activate
an environment. Run these commands from the repository root. If using NVIDIA,
install a matching CUDA-enabled PyTorch/torchvision pair with the
[official selector](https://pytorch.org/get-started/locally/) before the project:

```sh
python -m pip install -e .
python -m backend.tools.check_device
```

This installs the backend dependencies without the Qt or training extras.
`check_device` reports the PyTorch version, CUDA availability and GPU name when
available. It is a device diagnostic, not a performance benchmark.

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
model = load_model_from_path(
    "artifacts/detector_v1/trained_model_final.pth", device
)
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

## Tracking and actor inference

`TrackingController.update(boxes, ser, width, height, interval, agent=None,
log=None)` owns the last command time and previous action. Reuse one controller
across frames so command throttling and action history work.

With an actor, it constructs a four-element state:

| State value | Meaning |
| --- | --- |
| `dx` | Ball horizontal offset from frame center, divided by frame width |
| `dy` | Ball vertical offset from frame center, divided by frame height |
| `previous_action` | Last actor pan command |
| `is_detected` | `1.0` when a box is present, otherwise `0.0` |

When no ball is detected, `dx` and `dy` are zero and the detection flag is zero;
the actor can still choose a command. Without an actor, the controller uses the
first box and a 50-pixel horizontal dead zone: left, right, or stop when centered.
Basic tracking sends no new command when there is no box. With no serial
connection, the controller sends nothing. `interval` limits automatic command
frequency; it does not change inference FPS.

Application actor inference defaults to a pan scale of `1.0`; the training
simulation uses `5.0`. The state dictionary does not store this setting. See the
[training deployment notes](../training/README.md#optional-train-the-camera-control-policy)
and calibrate the policy/firmware combination before motor use.

## Hardware connections

`open_connection("COM3")` opens a known port at 115200 baud. Linux ports may
look like `/dev/ttyUSB0`; use the actual port assigned to your device.
`open_connection("auto")` probes ports for a `Stopping` response to `Stop`.
`open_connection(None)` returns `None` for operation without a platform.

The transport appends one newline to each command:

| Action | Serial command before the newline |
| --- | --- |
| Manual/basic left | `Left` |
| Manual/basic right | `Right` |
| Basic tracking centered | `Stop` |
| Actor pan | `P:<value>,T:0.00` |

The caller owns the connection and must close it. The frontend's worker does
this on exit. To run diagnostics explicitly:

```sh
python -m backend.tools.probe
python -m backend.tools.motormove --port COM3 --command Stop
```

The probe opens serial ports and sends `Stop`. The motor tool also accepts
`--command Left`, `--command Right`, or `--pan 0.5`; those commands move real
hardware. Consult the [firmware README](firmware/README.md) for the retained
sketch variants and their known limitations.

Bluetooth code currently provides discovery/connection helpers with placeholder
UUIDs. The application control loop uses serial; a Bluetooth status indicator
does not make motor control use Bluetooth. BLE firmware and UUID integration
remain unfinished.

## Code map and tests

| Location | Responsibility |
| --- | --- |
| [models.py](models.py) | Shared detector factory and checkpoint loading |
| [detection.py](detection.py) | Preprocessing, prediction filtering, coordinate scaling and overlays |
| [actor.py](actor.py), [policy.py](policy.py) | Shared actor architecture and inference-only wrapper |
| [tracking.py](tracking.py) | Rate-limited control and four-value actor state |
| [config.py](config.py) | Writable detector directory and `CAMMAN_MODELS_DIR` override |
| [hardware/](hardware/) | Serial commands, port probing and Bluetooth helpers |
| [tools/](tools/) | Device and platform diagnostics |
| [firmware/](firmware/) | Preserved ESP32 source variants |
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
reset, pass its known port explicitly. These tests use mocked serial devices;
firmware compilation and physical platform behavior require separate testing.

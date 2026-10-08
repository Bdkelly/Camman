# Camman

Camman detects a ball in a camera/video feed and can send pan commands to an
ESP32 camera platform. The desktop application and server training tools share
the same detector and actor definitions.

## Three components

| Directory | Responsibility | Entry point |
| --- | --- | --- |
| [frontend/](frontend/README.md) | PyQt windows, video presentation, UI signals and worker adapter | `camman` or `python -m frontend` |
| [backend/](backend/README.md) | Detector/actor inference, tracking lifecycle, serial protocol and ESP32 firmware | `camman-track`, frontend imports, diagnostics in `backend.tools` |
| [training/](training/README.md) | Detector training, reinforcement learning, annotation review, data cleaning and video tools | `camman-train`, `camman-train-agent`, `camman-clean`, `camman-annotate` |

The frontend and training tools depend on the backend. The backend does not
import Qt, training code, or Albumentations. The live application loads only the
actor network; critics, optimizers and replay buffers stay in `training/`.
Tests live inside each component's `tests/` directory.

New to the project? Start with the frontend guide to run a video, the training
guide to build your first detector, or the backend guide to integrate inference
into Python code. Each guide explains its inputs, outputs, connections and
current limitations.

## Install

**Jetson Orin Nano Super:** follow the [Jetson deployment guide](backend/JETSON.md)
for the native CUDA/OpenCV installation, runtime profile, hardware video input
and on-device benchmarks. Its installation steps replace the generic pip steps
below so the Jetson binary packages are retained.

For a fresh checkout of the development branch:

```sh
git clone --branch dev https://github.com/Bdkelly/Camman.git
cd Camman
```

Run the setup and component commands from this repository root, where
`pyproject.toml` lives. If you already have a checkout, use that copy.

Use Python 3.10 or newer in a virtual environment (validation uses Python 3.12).
Install a matching PyTorch/torchvision build for your hardware first using the
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/).
For example, create an environment with `python -m venv .venv`, then activate it:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[frontend]"
```

```bash
# Linux/macOS
source .venv/bin/activate
python -m pip install -e ".[frontend]"
```

Install `.[training]` on the training server, or `.[frontend,training,dev]` for
development. `pip install -e .` installs backend dependencies without Qt or
training augmentation dependencies. OpenCV is the headless build: Qt displays
the application video, and RL training runs without a display.

## Run the application

```bash
camman --source 0
camman --source /path/to/game.mp4 --model /path/to/detector.pth
camman --source 0 --model /path/to/detector.pth --actor /path/to/actor.pth --serial-port COM3
python -m backend.tools.check_device
camman --profile jetson --device cuda --source 0 --model /path/to/detector.pth
```

`--source` accepts a camera index, video path, URL, or an explicit GStreamer
pipeline with `--capture-backend gstreamer`. `--device cpu` or `--device
cuda:0` overrides automatic device selection. The app starts in video preview;
use **Start Inference** and **Start CamMan Agent** when ready. Without an actor,
tracking uses proportional pan velocity. Hardware is optional: pass an explicit port
or `--serial-port auto` to connect. Port probing sends `Stop`, not a pan command.

The Models window stores detector weights in `~/.camman/models` (on Windows,
`%USERPROFILE%\.camman\models`). Set `CAMMAN_MODELS_DIR` to override it. An explicit
`--model` takes priority; otherwise the first alphabetically sorted `.pth` in
that directory is selected. Keep actor weights outside this detector directory.
Both raw detector state dictionaries and training checkpoints containing
`model_state_dict` can be loaded. Missing or incompatible weights are reported;
preview remains available. No model weights are included in this repository.

## Training and data preparation

Training happens on the server. Input datasets and generated artifacts belong
outside the source packages, for example under the ignored `data/` and
`artifacts/` directories.

```bash
python -m training.data.video extract /path/to/game.mp4 data/game/frames
camman-clean --annotations data/game/labels.json --images data/game/frames --output data/game/clean.json
camman-train --annotations data/game/clean.json --images data/game/frames --output artifacts/detection --epochs 42
camman-cache-tracks --video /path/to/game.mp4 --model artifacts/detection/trained_model_final.pth --output data/tracks/game.npz
camman-train-agent --tracks data/tracks/game.npz --output artifacts/reinforcement
```

Detector training supports `--batch-size`, `--workers`, `--device` and
`--no-pretrained`. The default initializes from COCO weights and may download
them; loading an existing checkpoint for inference never requests those weights.
Agent training uses cached trajectories, a shared calibrated velocity contract,
held-out validation against basic controllers and full resumable checkpoints.
It saves a deployable actor only after actual optimizer updates. Follow the [training walkthrough](training/README.md) for labeling,
cleaning, training, reviewing results and using the saved weights in the app.

## Camera-control deployment

Use the [actor–critic walkthrough](training/README.md#optional-train-the-camera-control-policy)
to cache footage, train, resume, evaluate and deploy. `camman-track` runs the
full tracking pipeline without Qt; omit `--serial-port` for a dry-run.

```bash
camman-track --source 0 --model artifacts/detection/trained_model_final.pth --actor artifacts/reinforcement/actor_episode_best.pth
```

The [supported ESP32 firmware](backend/firmware/controller/README.md) implements
CAMMAN/1 normalized velocity, handshake, watchdog and software travel limits.
Other microcontrollers can implement the same protocol. Match measured motor
speed and camera FOV to the actor's saved calibration. Legacy raw actor weights
need retraining, and legacy `P:` position firmware must be replaced; existing
detector checkpoints remain supported.

Check the complete actor-training-to-command path without a detector or motor:

```bash
camman-check-agent-pipeline --output artifacts/pipeline_check
```

This trains a synthetic test actor, reloads it and verifies velocity/Stop output
through serial loopback. Read `verification.json` for the result; it does not
qualify the model for real games or test physical hardware. See the
[training guide](training/README.md#software-only-trial-without-a-detector-or-controller)
for server GPU use and the full game-training workflow.

## Development

```bash
python -m pip install -e ".[frontend,training,dev]"
python -m pytest
python -m build
```

For headless Linux tests, set `QT_QPA_PLATFORM=offscreen`. Set
`NO_ALBUMENTATIONS_UPDATE=1` to disable the augmentation library's startup version
check. Tests include real CPU actor/critic learning, deterministic resume, serial
loopback and a native firmware parser check when `g++` is available. Video and
physical devices are mocked; tests do not download weights or operate hardware.

## Migration from the old layout

| Previous location | New location/replacement |
| --- | --- |
| `guiapp/`, `apprunner.py` | `frontend/`, `camman` or `python -m frontend` |
| Three `utils/models.py` copies | `backend/models.py` |
| `guiapp/utils/vidpro.py`, `RLAgent/ballfind.py` | `backend/detection.py`, `backend/tracking.py`, `frontend/video.py` |
| `guiapp/utils/ser_*`, `bluecon.py` | `backend/hardware/` |
| `guiapp/Platform/`, `Platform_testing/`, `cudatest.py` | `backend/firmware/`, `backend/tools/` |
| `RLAgent/` | `training/reinforcement/`; shared actor in `backend/actor.py` |
| `training/train.py` | `training/detection/train.py` |
| Dataset/cleaning code in `utils/` | `training/data/clean.py`, `training/data/dataset.py` |
| `infer/`, inference experiments in `utils/oneUse/` | `training/data/auto_annotate.py`, `review.py`, `combine.py`, `metadata.py` |
| `videoedit/`, `infer/back2vid.py` | `training/data/video.py` |
| Existing annotation JSON | `training/data/examples/legacy/` |

Generated coverage reports, build/egg metadata, OS cache files, an empty mobile
export script, an unfinished JSON splitter, duplicate detector/inference/video
implementations and one byte-identical annotation copy were removed. A dummy
annotation example was also removed. Unique annotation files and firmware
variants were retained; their old versions also remain in Git history.

This branch retains Faster R-CNN and the PyQt frontend. NCNN export, a compact
Pi model and a TUI remain separate follow-up work. Historical image normalization
is retained for checkpoint compatibility; model accuracy and physical camera
behavior need validation on real data/hardware after this reorganization.

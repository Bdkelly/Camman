# Camman

Camman detects a ball in a camera/video feed and can send pan commands to an
ESP32 camera platform. The desktop application and server training tools share
the same detector and actor definitions.

## Three components

| Directory | Responsibility | Entry point |
| --- | --- | --- |
| `frontend/` | PyQt windows, video presentation, UI signals and worker adapter | `camman` or `python -m frontend` |
| `backend/` | Detector inference, actor inference, tracking control, serial/Bluetooth, firmware | Imported by the frontend; diagnostics in `backend.tools` |
| `training/` | Detector training, reinforcement learning, annotation review, data cleaning and video tools | `camman-train`, `camman-train-agent`, `camman-clean`, `camman-annotate` |

The frontend and training tools depend on the backend. The backend does not
import Qt, training code, or Albumentations. The live application loads only the
actor network; critics, optimizers and replay buffers stay in `training/`.
Tests live inside each component's `tests/` directory.

## Install

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
```

`--source` accepts a camera index or video path. `--device cpu` or `--device
cuda:0` overrides automatic device selection. The app starts in video preview;
use **Start Inference** and **Start CamMan Agent** when ready. Without an actor,
tracking uses left/right/stop rules. Hardware is optional: pass an explicit port
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
camman-train-agent --video /path/to/game.mp4 --model artifacts/detection/trained_model_final.pth --output artifacts/reinforcement
```

Detector training supports `--batch-size`, `--workers`, `--device` and
`--no-pretrained`. The default initializes from COCO weights and may download
them; loading an existing checkpoint for inference never requests those weights.
Agent training supports `--episodes` and `--steps` and saves best/final actors
and critics. See `training/README.md` for annotation formats and review commands.

## Development

```bash
python -m pip install -e ".[frontend,training,dev]"
python -m pytest
python -m build
```

For headless Linux tests, set `QT_QPA_PLATFORM=offscreen`. Set
`NO_ALBUMENTATIONS_UPDATE=1` to disable the augmentation library's startup version
check. Tests use synthetic images and mocked video/serial devices; they do not
download trained weights or operate hardware.

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

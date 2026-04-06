# Camman — Intelligent Ball Tracking Camera System

Camman is an autonomous camera pan control system that combines real-time ball detection with deep reinforcement learning. A Faster R-CNN model detects a ball in each video frame; a DDPG agent then continuously adjusts the camera's pan angle to keep the ball centered. Hardware commands are sent to an ESP32 microcontroller over serial or Bluetooth. A PyQt5 GUI provides live video monitoring and manual overrides.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
  - [RLAgent](#rlagent)
  - [GUI Application (guiapp)](#gui-application-guiapp)
  - [Training (training)](#training-training)
  - [Inference (infer)](#inference-infer)
  - [Utilities (utils)](#utilities-utils)
- [Training the RL Agent](#training-the-rl-agent)
- [Training the Detection Model](#training-the-detection-model)
- [Configuration](#configuration)
- [Hardware Setup](#hardware-setup)
- [Running Tests](#running-tests)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         GUI (PyQt5)                         │
│  MainWindow ──► VideoThread (QThread)                       │
└───────────────────────┬─────────────────────────────────────┘
                        │ frames
                        ▼
           ┌────────────────────────┐
           │  Ball Detection        │
           │  Faster R-CNN          │
           │  (ResNet50-FPN)        │
           └───────────┬────────────┘
                       │ bounding box
                       ▼
           ┌────────────────────────┐
           │  State Computation     │
           │  [dx, dy, pan, det]    │
           └───────────┬────────────┘
                       │ 4-D state
                       ▼
           ┌────────────────────────┐
           │  DDPG Agent            │
           │  Actor → pan action    │
           └───────────┬────────────┘
                       │ P:<angle>
                       ▼
           ┌────────────────────────┐
           │  ESP32 Platform        │
           │  Serial / Bluetooth    │
           └────────────────────────┘
```

**State vector** (4-D): `[dx, dy, prev_pan_action, is_detected]`

- `dx` / `dy` — normalised offset of ball centre from frame centre (−1 to 1)
- `prev_pan_action` — last pan command sent (provides velocity context)
- `is_detected` — 0 or 1 float flag

**Action** (1-D continuous): pan angle in `[−MAX_ACTION, MAX_ACTION]` degrees

---

## Project Structure

```
Camman/
├── RLAgent/                  # Reinforcement learning agent
│   ├── ActorNet.py           # Policy (actor) network
│   ├── CriticNet.py          # Q-value (critic) network
│   ├── RLAgent.py            # DDPG agent (experience replay, soft updates)
│   ├── agentTrain.py         # Training loop entry point
│   ├── ballfind.py           # Ball detection helper for RL loop
│   ├── camController.py      # Gym-style camera environment
│   ├── config.py             # All hyperparameters and paths
│   ├── reward.py             # Multi-component reward function
│   └── utils/
│       ├── models.py         # Faster R-CNN factory
│       └── noise.py          # Ornstein-Uhlenbeck exploration noise
├── guiapp/                   # PyQt5 desktop application
│   ├── camman.py             # Application entry point
│   ├── platform_screen.py    # Platform connectivity dialog
│   ├── threads/
│   │   └── video_threads.py  # VideoThread: detection + agent + display
│   └── utils/
│       ├── bluecon.py        # Bluetooth LE communication (bleak)
│       ├── models.py         # Model factory wrapper for GUI context
│       ├── ser_con.py        # Serial communication helpers
│       ├── ser_val.py        # Serial port validation
│       └── vidpro.py         # Video processing and tracking control
├── training/
│   └── train.py              # Faster R-CNN training pipeline
├── infer/
│   ├── inference.py          # Single-image / batch inference
│   ├── back2vid.py           # Reconstruct annotated video from JSON
│   └── ballfill.py           # Annotation utility
├── utils/                    # Shared dataset and data utilities
│   ├── models.py             # Faster R-CNN factory (canonical)
│   ├── datasets.py           # PyTorch Dataset (Label Studio JSON)
│   ├── jsonreader.py         # Annotation parsing helpers
│   ├── cleanCall.py          # Data cleaning
│   └── redodata.py           # Data preprocessing
├── setup.py                  # Package metadata and dependencies
└── pytest.ini                # Test configuration
```

---

## Installation

**Requirements:** Python 3.9+, CUDA-capable GPU recommended.

```bash
git clone https://github.com/bdkelly/camman.git
cd camman
pip install -e .
```

Core dependencies (installed automatically via `setup.py`):

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Detection model and RL networks |
| `opencv-python` | Frame capture and annotation |
| `albumentations` | Image transforms |
| `PyQt5` | Desktop GUI |
| `pyserial` | Wired serial to ESP32 |
| `bleak` | Bluetooth LE to ESP32 |
| `numpy` | Numerical ops |

---

## Quick Start

### Launch the GUI

```bash
python -m guiapp.camman
# or, after pip install -e .:
camman
```

1. The **video feed** panel shows the live camera stream.
2. Click **Start Inference** to enable ball detection overlays.
3. Click **Start CamMan Agent** to let the DDPG agent drive the camera.
4. Use **Manual Left / Manual Right** to override with fixed commands.
5. Open **Platform** to check serial / Bluetooth connectivity.
6. Open **Models** to hot-swap the detection model checkpoint.

---

## Modules

### RLAgent

The core reinforcement learning subsystem implementing the DDPG algorithm for continuous camera pan control.

| File | Key Class / Function | Description |
|------|----------------------|-------------|
| `RLAgent.py` | `RLAgent` | Full DDPG agent: actor-critic, replay buffer, learning |
| `ActorNet.py` | `Actor` | Policy network: state → pan action |
| `CriticNet.py` | `Critic` | Q-value network: (state, action) → scalar |
| `camController.py` | `CameraControlEnv` | Episode environment: reset/step/detect |
| `reward.py` | `RewardSystem` | Gaussian centering + effort + stability rewards |
| `agentTrain.py` | `train_agent()` | Episode training loop with OU noise |
| `ballfind.py` | `get_ball_detection()` | Single-frame detection with coordinate rescaling |
| `utils/noise.py` | `OUNoise` | Ornstein-Uhlenbeck process for exploration |
| `config.py` | — | All hyperparameters (edit before training) |

### GUI Application (guiapp)

A PyQt5 desktop application providing live monitoring and control.

| File | Key Class | Description |
|------|-----------|-------------|
| `camman.py` | — | `main()` entry point |
| `ui/main_screen.py` | `MainWindow` | Primary window with video, controls, log |
| `threads/video_threads.py` | `VideoThread` | Background thread for capture/detect/act |
| `utils/vidpro.py` | — | `videorun()`, `track_control()`, `init_video_comp()` |
| `utils/ser_con.py` | — | `find_esp32()`, `send_agent_command()`, manual pan |
| `utils/bluecon.py` | `ESP32Controller` | Async Bluetooth LE connection |
| `platform_screen.py` | `PlatformWindow` | Connectivity status dialog |

### Training (training)

```bash
# Edit paths inside train.py then run:
python training/train.py
```

`trainer()` in `training/train.py` trains a Faster R-CNN ResNet50-FPN model on a custom ball dataset annotated with Label Studio. Features:

- Albumentations augmentation pipeline
- Cosine/step learning rate schedule with linear warmup
- Gradient clipping
- Checkpoint saving each epoch

### Inference (infer)

```bash
python infer/inference.py
```

Runs the trained detection model on individual images or directories and outputs annotated images plus JSON bounding-box files.

### Utilities (utils)

| File | Description |
|------|-------------|
| `models.py` | `get_fasterrcnn_model_single_class(num_classes)` — model factory |
| `datasets.py` | `objectdata` — PyTorch Dataset wrapping Label Studio JSON annotations |
| `jsonreader.py` | Parse and filter Label Studio bounding box annotations |

---

## Training the RL Agent

1. **Prepare a detection model** (see [Training the Detection Model](#training-the-detection-model)) and note its `.pth` path.
2. **Prepare a training video** — a video file containing the ball you want to track.
3. **Edit `RLAgent/config.py`**:
   ```python
   VIDEO_PATH  = 'path/to/your/video.mp4'
   MODEL_PATH  = 'path/to/detection_model.pth'
   NUM_EPISODES = 200    # increase for better convergence
   ```
4. **Run training**:
   ```bash
   python RLAgent/agentTrain.py
   ```
   A window shows the live frame with the agent's control window (green rectangle). Press **Q** to stop early.
5. Checkpoints are saved to `checkpoints/` every 100 episodes and whenever a new best average score is achieved.
6. Copy the best checkpoint to `guiapp/agentModel/agentModel.pth` for use in the GUI.

**Key hyperparameters** (all in `config.py`):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `LR_ACTOR` | `1e-4` | Actor learning rate |
| `LR_CRITIC` | `1e-3` | Critic learning rate |
| `GAMMA` | `0.99` | Discount factor |
| `SOFT_UPDATE` | `1e-3` | Polyak averaging coefficient τ |
| `BATCH_SIZE` | `128` | Replay batch size |
| `MEMORY_SIZE` | `2048` | Replay buffer capacity |
| `MAX_ACTION` | `5.0` | Maximum pan angle (degrees) |
| `NOISE_SIGMA` | `0.2` | Initial OU noise standard deviation |
| `NOISE_DECAY` | `0.999` | Per-episode noise decay |

**Reward components** (`RWD_WEIGHTS` in `config.py`):

| Key | Default | Description |
|-----|---------|-------------|
| `centering_peak` | `500.0` | Peak reward when ball is centred (Gaussian) |
| `centering_decay` | `10.0` | Controls width of centering Gaussian |
| `effort` | `0.01` | Penalty coefficient on `|pan_action|` |
| `stability` | `1.0` | Penalty coefficient on action acceleration |
| `window_bonus` | `100.0` | Bonus when `|dx| < 0.125` (ball in centre band) |
| `lost_ball_penalty` | `100.0` | Flat penalty per step when ball not detected |

---

## Training the Detection Model

1. **Annotate images** using [Label Studio](https://labelstud.io/) with a single `Ball` class. Export in JSON format.
2. **Edit paths** in `training/train.py` (dataset directory, annotation JSON, output model path).
3. **Run training**:
   ```bash
   python training/train.py
   ```

The model is a pre-trained Faster R-CNN ResNet50-FPN with a replaced 2-class head (background + Ball). The canonical factory function is:

```python
from utils.models import get_fasterrcnn_model_single_class
model = get_fasterrcnn_model_single_class(num_classes=2)
```

---

## Configuration

All RL training parameters live in `RLAgent/config.py`. Key sections:

```python
# Paths (must be updated before training)
VIDEO_PATH     = 'path/to/training_video.mp4'
MODEL_PATH     = 'path/to/detection_model.pth'
CHECKPOINT_DIR = 'checkpoints'

# RL architecture
STATE_SIZE  = 4    # [dx, dy, prev_action, is_detected]
ACTION_SIZE = 1    # pan angle
MAX_ACTION  = 5.0  # degrees

# Training
NUM_EPISODES = 200
MAX_T        = 5000   # max steps per episode
BATCH_SIZE   = 128
MEMORY_SIZE  = 2048
```

---

## Hardware Setup

The system communicates with an **ESP32** microcontroller that drives a pan-tilt servo platform.

**Serial (wired)**
- Default baud rate: `115200`
- The application automatically scans available ports and connects to the first valid ESP32 port.

**Bluetooth LE**
- Uses the `bleak` library for async BLE scanning.
- The `PlatformWindow` dialog shows real-time connection status for both interfaces.

**Command protocol**

| Command | Example | Description |
|---------|---------|-------------|
| Pan | `P:2.35` | Set pan angle to 2.35° |
| Pan + Tilt | `P:1.00,T:0.00` | Set pan and tilt (tilt currently unused) |
| Manual Left | `P:0.5` | Fixed left pan step |
| Manual Right | `P:1.5` | Fixed right pan step |
| Stop | `Stop` | Halt platform movement |

---

## Running Tests

```bash
pytest
```

Test files are located in `RLAgent/tests/` and `guiapp/tests/`. Coverage includes:

- DDPG agent forward/learn pass (`test_rl_agent.py`)
- Training loop smoke test (`test_agentTrain.py`)
- Camera environment step/reset (`test_cam_controller.py`)
- Reward computation (`test_reward.py`)
- GUI main window (`test_camman.py`)
- Video thread lifecycle (`test_videoThread.py`)
- Serial connection helpers (`test_ser_con.py`)
- Video processing functions (`test_vidpro.py`)
- Model loading (`test_models.py`)
- Platform screen (`test_platformS.py`)

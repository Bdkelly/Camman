# Jetson Orin Nano Super deployment

[Project](../README.md) · [Frontend](../frontend/README.md) · [Backend](README.md)

Training stays on your RTX server. Copy the detector `.pth` and optional
versioned camera-control actor `.pth` to the Jetson. Detector checkpoints do
not need retraining for this profile. Legacy bare actor weights must be
retrained with the new observation/velocity contract. The app remains PyQt5, with video inside its window.

These changes have CPU/headless regression coverage. **Jetson FPS, CUDA FP16
accuracy, camera pipelines and motor behavior still need testing on your kit.**
No target-hardware speedup or real-time frame rate is claimed.

## Runtime settings

| Setting | Standard profile | Jetson profile |
| --- | --- | --- |
| Device | CUDA if available, otherwise CPU | Same; CPU fallback is logged |
| Precision | FP32 | CUDA FP16 autocast; FP32 on CPU |
| Actual Faster R-CNN input | 800 × 800 | 640 × 640 |
| External checkpoint preprocessing | 640 × 640, historical normalization | Same |
| PyTorch CPU threads | Library default | 2 |
| OpenCV CPU threads | Library default | 1 |
| Preview ceiling | 30 FPS | 15 FPS |
| Inference ceiling | Unlimited | 15 FPS |
| Optional actor policy | Same device as detector | CPU |

`--profile auto` selects Jetson from `/proc/device-tree/model`. In a container
without that device tree, pass `--profile jetson`. Use `--device cuda` to fail
explicitly if CUDA is unavailable.

Previously, preprocessing resized to 640 and torchvision enlarged it to 800.
The Jetson profile changes torchvision's internal resize, reducing backbone
input pixels by 36%. This is **not a measured 36% latency improvement**.
Smaller balls may become harder to detect; compare recall on held-out games.
`--detector-size 800` restores the original size with the other optimizations.

FP16 uses `torch.autocast`, preserving PyTorch's operator precision rules.
Parameters stay FP32. Raw and wrapped training checkpoints, FrozenBatchNorm,
and historical double normalization remain supported. Training defaults do
not change.

## Install the native Jetson packages

Use a JetPack release supporting your Orin Nano Super. NVIDIA added Super mode
in JetPack 6.2; JetPack 7.2 extends its newer stack to Orin. Match the PyTorch,
torchvision and CUDA binaries to **your installed JetPack and Python versions**.
Desktop CUDA wheels are not an interchangeable Jetson installation recipe.

Run from the repository root on the Jetson. This example uses system OpenCV
and PyQt, and NumPy 1.x for compatibility with JetPack 6.x system OpenCV.
On another native stack, retain the NumPy version its binary packages require.

```bash
sudo apt-get update
sudo apt-get install python3-venv python3-pip python3-opencv python3-pyqt5 \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad nvidia-l4t-gstreamer
/usr/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install 'numpy>=1.24,<2' 'Pillow>=10,<13' \
  'pyserial>=3.5,<4' 'bleak>=0.21,<4'
```

Install CUDA-enabled PyTorch and matching torchvision in this environment using
[NVIDIA's instructions](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
and [compatibility table](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html).
A working system-wide pair can be inherited by the environment. If building
torchvision, follow its [matching-version build instructions](https://github.com/pytorch/vision#installation)
with CUDA enabled. An import alone does not verify CUDA NMS and ROIAlign.

Then install Camman without replacing those binary packages:

```bash
python -m pip install --no-deps -e .
python -m backend.tools.check_device --device cuda --verify-ops
gst-inspect-1.0 nvv4l2decoder
gst-inspect-1.0 nvvidconv
```

The diagnostic prints package versions, the loaded OpenCV path and GStreamer
support, and executes NMS/ROIAlign on CUDA. Pipeline input needs `GStreamer:
True`. If torchvision fails to import, fix the binary pair before launching.

The generic `pip install -e '.[frontend]'` pulls in `opencv-python-headless`,
which does not supply GStreamer. The Jetson installation deliberately uses
vendor/system packages, bypassing those generic wheel requirements with
`--no-deps`. Keep this environment separate from training and avoid reinstalling
the generic extras into it. The native OpenCV build must support GStreamer and
its open/read timeout properties for pipeline input.

## Launch

Run in the Jetson's graphical desktop. USB camera example:

```bash
camman --profile jetson --device cuda --source 0 --capture-backend v4l2 \
  --capture-width 1280 --capture-height 720 --capture-fps 30 \
  --model artifacts/detector_v1/trained_model_final.pth
```

Camera dimensions/FPS are requests; inspect the actual settings in the startup
log. Omit these flags for the camera defaults. Smaller capture sizes save work
but may remove small-ball detail. Click **Start Inference** after loading.
Add `--serial-port /dev/ttyUSB0` for your platform and optionally
`--actor artifacts/policy_v1/actor_episode_best.pth`. Check the actual port and
Linux serial permissions. The app does not change power modes or clocks.

Recorded video:

```bash
camman --profile jetson --device cuda --source data/game_01/game.mp4 \
  --model artifacts/detector_v1/trained_model_final.pth
```

Files keep every frame and play more slowly if inference cannot keep up.
Live inputs drain capture continuously, retaining only the newest waiting
frame. Override classification with `--source-mode live` or `file`.

## Headless policy deployment

To operate without Qt, use the same backend through `camman-track`:

```bash
camman-track --profile jetson --device cuda --source 0 --capture-backend v4l2 \
  --model artifacts/detector_v1/trained_model_final.pth \
  --actor artifacts/policy_v1/actor_episode_best.pth
```

This starts immediately in dry-run mode. After validating the
[controller firmware/calibration](firmware/controller/README.md), add
`--serial-port /dev/ttyUSB0`. Use `--invert-pan` only if the measured wiring
reverses the intended camera direction. The actor runs on CPU and uses its
saved decision frequency, FOV, speed and timeout contract. Train using the
measured deployment cadence and latency; an actor cannot compensate for an
arbitrarily slow detector. Critic/replay/checkpoint state stays on the server.

The serial handshake requires CAMMAN/1 velocity support. Frame freshness and
lost-target checks stop stale motion; shutdown and failures attempt Stop. The
ESP32 firmware has its own 750 ms watchdog because a blocked Python inference
call cannot send immediate commands. No physical Jetson/motor validation is
claimed by the software tests.

## Hardware video input

Explicit GStreamer pipelines enable NVIDIA's decoder/converter. An ordinary
`--source game.mp4` does not promise hardware decoding. Pipelines must end in
a BGR `appsink`, without a `gst-launch-1.0` prefix. Adjust these examples for
your source codec, camera modes and installed plugins. Camman passes pipeline
strings to OpenCV, never to a shell.

H.264 MP4, preserving every frame:

```bash
camman --profile jetson --device cuda --capture-backend gstreamer --source-mode file \
  --source 'filesrc location="/absolute/path/game.mp4" ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=false sync=false' \
  --model artifacts/detector_v1/trained_model_final.pth
```

For H.265 MP4 use `h265parse`. Other containers need the appropriate demuxer.
Live RTSP H.264:

```bash
camman --profile jetson --device cuda --capture-backend gstreamer --source-mode live \
  --source 'rtspsrc location="rtsp://camera-address/stream" latency=100 drop-on-latency=true ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true sync=false' \
  --model artifacts/detector_v1/trained_model_final.pth
```

An Argus-compatible CSI camera supporting this mode:

```bash
camman --profile jetson --device cuda --capture-backend gstreamer --source-mode live \
  --source 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true sync=false' \
  --model artifacts/detector_v1/trained_model_final.pth
```

BGR still crosses into CPU arrays for OpenCV/Qt, then into PyTorch. This is not
an end-to-end zero-copy or DeepStream pipeline. Camera, network and decoder
buffering still contribute latency.

## Benchmark and tune

Use the same checkpoint, video and power/cooling configuration for both runs:

```bash
python -m backend.tools.benchmark --profile standard --device cuda \
  --source data/game_01/game.mp4 --model artifacts/detector_v1/trained_model_final.pth \
  --warmup 10 --frames 200 --output artifacts/benchmarks/standard.json

python -m backend.tools.benchmark --profile jetson --device cuda \
  --source data/game_01/game.mp4 --model artifacts/detector_v1/trained_model_final.pth \
  --warmup 10 --frames 200 --output artifacts/benchmarks/jetson.json
```

`camman-benchmark` is the equivalent installed command. Reports include settings
and library versions, mean/p50/p95 detection latency, capture time, sequential
throughput and peak PyTorch CUDA allocation. Batch-one detection uses CUDA
synchronization and excludes warmup. It includes preprocessing, transfers and
output filtering; it excludes drawing, Qt and serial. No frames are dropped,
app FPS caps are not applied, and file playback is unpaced. Short videos report
the actual sample count; a live camera can limit throughput to its capture rate.

Detection count is **not** an accuracy metric, and peak CUDA allocation is not
total Jetson RAM. Monitor RAM, temperatures and clocks with `tegrastats`, and
inspect power mode with `sudo nvpmodel -q`. Select a mode appropriate to your
flashed image and cooling using NVIDIA's instructions; mode numbers vary.
Record those settings alongside the reports.

Try one change at a time:

- `--precision fp32` to compare numerical stability with FP16.
- `--detector-size 800` for original resolution; 512 for a more aggressive
  reduction needing a small-ball recall check.
- `--cpu-threads 1`, `2`, or `4` while monitoring latency and CPU use.
- In the app, `--inference-fps 10 --preview-fps 15` to cap work, or
  `--inference-fps 0` to remove the inference ceiling.

The GUI shows preview production rate, detection rate, last detection time,
frame age since OpenCV returned it, and application capture drops. Frame age
excludes buffering inside the camera/decoder. Preview FPS is a ceiling: while
inference is active, video follows detector cadence. Controls remain responsive,
but a slow detector cannot produce smooth 30 FPS annotated video. Compare
missed balls and jitter on representative labeled games as well as timing,
particularly around the existing 0.98 confidence threshold.

## Limits

Faster R-CNN ResNet-50/FPN is still a substantial model. If measured latency
remains too high, train and validate a smaller detector on the server and add
its inference adapter. TensorRT FP16/INT8 can be evaluated then; a Faster R-CNN
`.pth` is not a ready TensorRT engine. INT8 requires representative calibration
and a small-ball accuracy check. This branch has no TensorRT/NCNN loader.

Manual commands are coalesced and sent between inference calls, off the UI
thread; this is not a separate real-time control loop. Capture is released by
its reader thread. FFmpeg/GStreamer have configured open/read timeouts; a stuck
native camera driver can defer release until its read returns. Closing can
also wait for an active model load or inference call.

## References

- [JetPack 6.2 Super mode](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/)
- [JetPack 7.2 Orin support](https://developer.nvidia.com/blog/deploy-agentic-ready-ai-at-the-edge-with-memory-efficiency-in-nvidia-jetpack-7-2/)
- [NVIDIA accelerated GStreamer](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html)
- [PyTorch autocast](https://docs.pytorch.org/docs/2.6/amp.html)
- [OpenCV capture backends and timeouts](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html)

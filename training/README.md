# Model training and data cleaning

`detection/` trains the ball detector. `reinforcement/` trains the camera-control
actor/critic. `data/` contains shared dataset loading and offline preparation
tools. All paths are supplied as CLI arguments; no developer-specific paths are
used by runnable code.

## Annotation format

The cleaner accepts legacy inference lists, the former flat pixel-coordinate
records, Label Studio video sequences (`box` or `annotations/result/value`),
and this cleaned format:

```json
[
  {"frame": "frame_00000.jpg", "boxes": [[100, 50, 120, 70]]},
  {"frame": "frame_00001.jpg", "boxes": []}
]
```

Boxes are `[x_min, y_min, x_max, y_max]` in **image pixels**. Every box is the
single `Ball` class. An empty list is an explicitly annotated negative image.
The cleaner groups boxes by image, preserves negative frames, clips boxes to
image bounds, and rejects invalid boxes or missing images. Disabled Label
Studio annotations are skipped; missing annotations are not assumed negative.
Frame extraction starts at `frame_00000.jpg`; check the annotation export's
frame numbering and extraction FPS before preparing a dataset.

Clean each video with its own frame directory. Combined legacy files containing
multiple video-source metadata entries are rejected by the cleaner to prevent
same-name frames from different games being merged. Existing multi-video JSON
is retained as historical data, not a ready-to-train dataset.

## Offline tools

```bash
camman-annotate --images data/game/frames --model artifacts/detection/trained_model_final.pth --output artifacts/review/candidates.json --preview-dir artifacts/review/images
python -m training.data.review --annotations artifacts/review/candidates.json --images artifacts/review/images --output artifacts/review/edited
python -m training.data.video assemble artifacts/review/images artifacts/review/game.mp4 --fps 30
python -m training.data.combine first.json second.json combined.json
python -m training.data.metadata --annotations labels.json --video game --images data/game/frames --output labels_with_source.json
```

The annotation generator loads the model once for the directory and retains
frames with no detection. Review candidates before treating them as ground
truth. The manual review window requires Python's Tk support and a graphical
desktop; extraction requires `ffmpeg` on PATH. Use `--help` for each command.
Combining lists preserves entries and metadata without resolving conflicting
frame names. Use it only when the source identity remains unambiguous.

## Compatibility and remaining model work

The training loader now passes pixel boxes to Albumentations' `pascal_voc`
transform and represents each image once with all its targets. It raises an
error for missing images instead of fabricating a dummy frame. New training
runs will therefore differ from runs using the old coordinate conversion.

For this structural cleanup, the existing 640px resize and external ImageNet
normalization are preserved in training and inference. Faster R-CNN also
normalizes internally; correcting that convention should be a coordinated
retraining/checkpoint-format change, not a silent change to deployed weights.
The training script still needs a held-out validation split and model-selection
metrics before training quality can be evaluated. This branch does not add NCNN
export or claim Raspberry Pi real-time performance.

RL training retains its existing `MAX_ACTION=5.0` simulation scale. Application
actor inference retains its existing `max_action=1.0` scale for the serial pan
interface. State-dict weights do not encode this scale; calibrate the selected
firmware and deployment policy before using a newly trained actor on hardware.

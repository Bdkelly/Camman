# Ball-detection dataset: images, labels and honest evaluation

This guide is for **ball detection**, not actor–critic trajectory training.
A sample is one original image plus every ball box you intend the detector to
recognize in that image. Images without a visible ball are useful reviewed
negative samples. Video files themselves are not inputs to `camman-train`.

## Recommended layout

Keep original images under a common root with unique game/session directories.
Separate JSON manifests decide which images belong to each split:

| Path | Contents |
| --- | --- |
| `data/ball/images/game_001/frame_00000.jpg` | Original frame from game 001 |
| `data/ball/images/game_001/frame_00015.jpg` | Another reviewed frame from that game |
| `data/ball/images/game_015/frame_00120.jpg` | Frame from a different game, assigned to validation |
| `data/ball/images/game_018/frame_00080.jpg` | Frame from another game, reserved for final testing |
| `data/ball/train.json` | Only training-game records |
| `data/ball/validation.json` | Only validation-game records |
| `data/ball/test.json` | Only untouched test-game records |

All three manifests can use `--images data/ball/images`. Alternatively, use
separate image roots and pass `--validation-images` when training. Image filenames
in JSON are relative to their image root; forward slashes also work on Windows.

For example, `train.json` could contain:

```json
[
  {"frame": "game_001/frame_00000.jpg", "boxes": [[948, 512, 966, 530]]},
  {"frame": "game_001/frame_00015.jpg", "boxes": []},
  {"frame": "game_002/frame_00240.jpg", "boxes": [[85, 240, 97, 252], [1510, 720, 1528, 738]]}
]
```

`validation.json` uses **exactly the same schema**, with different games:

```json
[
  {"frame": "game_015/frame_00120.jpg", "boxes": [[422, 315, 436, 329]]},
  {"frame": "game_016/frame_00090.jpg", "boxes": []}
]
```

So does `test.json`:

```json
[
  {"frame": "game_018/frame_00080.jpg", "boxes": [[1100, 600, 1116, 616]]},
  {"frame": "game_019/frame_00450.jpg", "boxes": []}
]
```

These are illustrative coordinates, not annotations to apply to your footage.
Each referenced file must exist. Keep each frame once in its manifest. Training
accepts JPEG/PNG and other Pillow-readable RGB-convertible images. Save orientation
into the pixels before annotation; avoid relying on EXIF viewer rotation.

## Box meaning and annotation quality

Boxes are **pixel xyxy** in the original stored image, not normalized YOLO values:

- `[x_min, y_min, x_max, y_max]`, origin at the top-left.
- `0 <= x_min < x_max <= image_width`.
- `0 <= y_min < y_max <= image_height`.
- No class field is required: every box is class `Ball` (training label 1).
- `boxes: []` explicitly means “reviewed; no visible ball to label.”

For `[948, 512, 966, 530]`, the box is 18 pixels wide and 18 pixels high. Draw a
tight box around the visible ball; do not include the player or a large patch of
field. Label identifiable blurred/partially occluded balls consistently. Do not
guess a hidden ball's location. Exclude ambiguous/unreviewed frames rather than
silently marking them negative. All recognizable balls of the selected class
should be labelled; if spare balls appear, decide the class policy consistently
across every split. This detector alone does not identify which ball is in play.

Keep original pixels: no drawn boxes, labels, tracking crosshairs or synthetic
prediction overlays. Include far-away tiny balls, close balls, aerial plays,
fast motion, partial occlusion, field edges, difficult lighting and camera angles.
Include hard negatives: white shoes, field markings, jerseys, spectators and
empty field. Automatically generated labels must be reviewed, especially missed
balls incorrectly proposed as negatives.

There is no fixed number of images or positive/negative ratio that guarantees a
useful model. Coverage of varied games and difficult cases matters more than
thousands of nearly identical adjacent frames. Match validation/test conditions
to real deployment, including stretches when the ball is absent.

## Split by game/session before selecting frames

A starting allocation is **70% training / 15% validation / 15% test by games**.
For 20 independent games that could mean 14/3/3. Exact frame counts need not follow
that ratio. With fewer games, report the limited evaluation coverage honestly.

| Split | How it is used | Must not be used for |
| --- | --- | --- |
| Training | Weight updates; random flip/brightness augmentation | Reporting final generalization |
| Validation | Choose best epoch, confidence threshold and settings | Gradient updates |
| Test | One final evaluation after choices are fixed | Repeated tuning/model selection |

**Do not randomly split neighboring frames from the same video.** Keep all clips
from the same match/session in the same split, including alternate camera views,
re-encoded copies and augmented copies. Otherwise the model can see almost the
same scene during training and testing. Prefer additional held-out venues,
lighting conditions or cameras when evaluating transfer to a new setup.

For annotation efficiency, start by extracting sparse frames (for example 1–3 FPS),
then add targeted samples of hard/fast plays. This is a sampling starting point,
not a requirement: too sparse a sample can miss important appearances. Never
change extraction FPS after creating labels without remapping frame indices.

The trainer does **not** automatically split a manifest. It rejects exact resolved
file-path overlap between train and validation. It cannot detect duplicated pixels
under different names or determine which games belong together; you must enforce
session-level separation. Test-set isolation is also your responsibility.

## Check, train and test

Clean each manifest separately:

```sh
camman-clean --annotations data/ball/train.json --images data/ball/images --output data/ball/train.clean.json
camman-clean --annotations data/ball/validation.json --images data/ball/images --output data/ball/validation.clean.json
camman-clean --annotations data/ball/test.json --images data/ball/images --output data/ball/test.clean.json
```

Cleaning groups labels, clips coordinates, removes exact duplicate boxes and
checks image existence/geometry. It cannot certify label accuracy or game separation.
Review overlays in an annotation tool before training. The command also accepts
legacy formats described in the [training guide](README.md#3-create-or-import-labels).

Train on the RTX server (use a fresh output directory):

```sh
camman-train --annotations data/ball/train.clean.json --images data/ball/images --validation-annotations data/ball/validation.clean.json --epochs 42 --batch-size 4 --workers 4 --prefetch 2 --precision auto --device cuda:0 --cache-dir artifacts/image_cache --output artifacts/detector_v2
```

The default internal detector size stays 800px for continuity. Add
`--detector-size 640` to test a less expensive backbone input; measure the effect
on small-ball recall before adopting it. The historical external resize and
normalization remain unchanged. The best checkpoint is selected by validation
**F1 at a fixed confidence and IoU**, both 0.5 by default. Validation uses FP32
without training augmentations. The first epoch wins equal-score ties.

Once settings are fixed, evaluate the chosen model on untouched test games:

```sh
camman-evaluate-detector --model artifacts/detector_v2/trained_model_best.pth --annotations data/ball/test.clean.json --images data/ball/images --device cuda:0 --workers 4 --batch-size 4 --confidence 0.5 --iou 0.5 --detector-size 800 --output artifacts/detector_v2/test.json
```

Use the **same internal detector size** as training (640 if you changed it).
Evaluation reports true/false positives, missed balls, precision, recall, F1 and
how many negative images contain false alarms. It matches boxes one-to-one in
score order. These fixed-threshold metrics are **not COCO mAP**. A ball can be
approximately located but fail the IoU threshold, particularly when very small.
Inspect example misses and false positives as well as summary numbers.

The inference application historically defaults to a higher confidence threshold
(0.98). If deploying at that threshold, choose/verify it on validation, then run
the final test with `--confidence 0.98` as well. Test at deployment precision and
resolution too; `--precision fp16` is available for CUDA evaluation. Test results
do not measure camera-control accuracy, latency or Jetson throughput.

No ground-truth labels are required when running the application on live footage.
Only the trained detector file is deployed; the validation/test images stay on
the evaluation machine.

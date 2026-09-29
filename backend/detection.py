"""Frame preprocessing and ball detection shared by the application and tools."""

from contextlib import nullcontext

import cv2
import numpy as np
import torch


class FrameTransform:
    """Preserve the 640px/ImageNet preprocessing used by existing checkpoints.

    This normalization is historical (Faster R-CNN also normalizes internally).
    Changing it requires retraining and an explicit checkpoint migration.
    """

    def __init__(self, size=640):
        self.size = size
        self.mean = np.array((0.485, 0.456, 0.406), dtype=np.float32) * 255
        self.scale = 1 / (np.array((0.229, 0.224, 0.225), dtype=np.float32) * 255)

    def __call__(self, *, image):
        return self._normalize(cv2.resize(image, (self.size, self.size)))

    def from_bgr(self, frame):
        # Resize before colour conversion: avoid a full-resolution RGB copy.
        small = cv2.resize(frame, (self.size, self.size))
        return self._normalize(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    def _normalize(self, image):
        image = image.astype(np.float32)
        image = (image - self.mean) * self.scale
        return {"image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))}


def get_ball_detection(
    model,
    frame,
    transform,
    device,
    confidence_threshold=0.98,
    *,
    max_detections=1,
    draw=True,
    precision="fp32",
):
    """Return pixel-coordinate ball boxes and an optionally annotated BGR frame."""
    device = torch.device(device)
    if precision not in {"fp32", "fp16"} or (precision == "fp16" and device.type != "cuda"):
        raise ValueError("Use fp32, or fp16 on a CUDA device")
    if isinstance(transform, FrameTransform):
        tensor = transform.from_bgr(frame)["image"].to(device)
    else:
        tensor = transform(image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))["image"].to(device)
    input_height, input_width = tensor.shape[-2:]
    amp = torch.autocast("cuda", dtype=torch.float16) if precision == "fp16" else nullcontext()
    with torch.inference_mode(), amp:
        prediction = model([tensor])[0]
    all_scores = prediction["scores"].float()
    keep = (all_scores >= confidence_threshold) & (prediction["labels"] == 1)
    # Limit on-device, then make one small host transfer. Keep coordinates in
    # float32 even under autocast; ROI/NMS retain their own autocast policies.
    boxes = prediction["boxes"][keep][:max_detections].float()
    scores = all_scores[keep][:max_detections, None]
    results = torch.cat((boxes, scores), dim=1).cpu().numpy()
    height, width = frame.shape[:2]
    scale = np.array([width / input_width, height / input_height] * 2)
    detected = []
    for result in results:
        box, score = result[:4], result[4]
        x1, y1, x2, y2 = (box * scale).astype(int)
        coords = (
            int(np.clip(x1, 0, width)),
            int(np.clip(y1, 0, height)),
            int(np.clip(x2, 0, width)),
            int(np.clip(y2, 0, height)),
        )
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        detected.append({"box": coords, "label": "Ball", "score": float(score)})
        if draw:
            cv2.rectangle(frame, coords[:2], coords[2:], (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Ball {score:.2f}",
                (coords[0], coords[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
    return detected, frame

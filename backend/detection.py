"""Frame preprocessing and ball detection shared by the application and tools."""

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
        image = cv2.resize(image, (self.size, self.size)).astype(np.float32)
        image = (image - self.mean) * self.scale
        return {"image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))}


def get_ball_detection(
    model, frame, transform, device, confidence_threshold=0.98, *, max_detections=1, draw=True
):
    """Return pixel-coordinate ball boxes and an optionally annotated BGR frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(image=rgb)["image"].to(device)
    input_height, input_width = tensor.shape[-2:]
    with torch.inference_mode():
        prediction = model([tensor])[0]
    keep = (prediction["scores"] >= confidence_threshold) & (prediction["labels"] == 1)
    boxes = prediction["boxes"][keep].cpu().numpy()
    scores = prediction["scores"][keep].cpu().numpy()
    height, width = frame.shape[:2]
    scale = np.array([width / input_width, height / input_height] * 2)
    detected = []
    for box, score in zip(boxes[:max_detections], scores[:max_detections]):
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

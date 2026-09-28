"""Ball dataset: one image with all its boxes, including empty negative targets."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from backend.detection import FrameTransform


class BallDataset(Dataset):
    def __init__(self, images_dir, records, transform=None):
        self.images_dir = Path(images_dir)
        self.records = records
        self.transform = transform
        if not self.images_dir.is_dir():
            raise FileNotFoundError(self.images_dir)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.images_dir / record["frame"]) as image:
            array = np.asarray(image.convert("RGB"))
        boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.ones(len(boxes), dtype=np.int64)
        if self.transform is not None:
            # Albumentations pascal_voc expects pixels, not coordinates divided by image size.
            result = self.transform(image=array, bboxes=boxes, labels=labels)
            tensor, boxes, labels = result["image"], result["bboxes"], result["labels"]
        else:
            height, width = array.shape[:2]
            tensor = FrameTransform()(image=array)["image"]
            boxes = boxes * np.array([640 / width, 640 / height] * 2, dtype=np.float32)
        target = {
            "boxes": torch.as_tensor(np.asarray(boxes), dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(np.asarray(labels), dtype=torch.int64),
        }
        return tensor, target

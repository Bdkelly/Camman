"""Ball dataset: one image with all its boxes, including empty negative targets."""

import hashlib
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from backend.detection import FrameTransform


class BallDataset(Dataset):
    def __init__(self, images_dir, records, transform=None, *, cache_dir=None):
        self.images_dir = Path(images_dir)
        self.records = records
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(self.images_dir)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        array, scale = self._image(self.images_dir / record["frame"])
        boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4) * scale
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

    def _image(self, path):
        """Cache deterministic RGB/resize only; never cache random augmentations.

        Keys include absolute source path, size and mtime_ns. Editing a source
        invalidates its entry on the next read. Files must not change mid-run.
        Each compressed source gets a ~1.2 MB uncompressed NPZ on local storage.
        Atomic replacement permits separate workers/runs to share the directory.
        """
        if self.cache_dir is None:
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB")), np.ones(4, dtype=np.float32)
        stat = path.stat()
        key = hashlib.sha256(
            f"rgb-linear-640-v1:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
        ).hexdigest()
        cached = self.cache_dir / f"{key}.npz"
        if cached.is_file():
            with np.load(cached, allow_pickle=False) as data:
                array, scale = data["image"], data["scale"]
            if array.shape != (640, 640, 3) or array.dtype != np.uint8 or scale.shape != (4,):
                raise ValueError(f"Invalid image cache: {cached}; remove it and retry")
            return array, scale
        with Image.open(path) as image:
            width, height = image.size
            array = cv2.resize(np.asarray(image.convert("RGB")), (640, 640))
        scale = np.array([640 / width, 640 / height] * 2, dtype=np.float32)
        fd, temporary = tempfile.mkstemp(prefix=key, suffix=".tmp", dir=self.cache_dir)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez(stream, image=array, scale=scale)
            os.replace(temporary, cached)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return array, scale

"""DataLoader options shared by detector training and held-out evaluation."""

import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info


def collate_fn(batch):
    return tuple(zip(*batch))


def seed_worker(worker_id):
    # Avoid each loader process launching another pool of OpenCV threads.
    cv2.setNumThreads(1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    transform = get_worker_info().dataset.transform
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)


def make_loader(dataset, *, batch_size, workers, device, shuffle=False, seed=0, prefetch=2):
    if not len(dataset):
        raise ValueError("Dataset must contain at least one reviewed image")
    if batch_size < 1 or workers < 0 or prefetch < 1:
        raise ValueError("Batch size/prefetch must be positive and workers non-negative")
    options = {}
    if workers:
        options.update(
            persistent_workers=True,
            prefetch_factor=prefetch,
            worker_init_fn=seed_worker,
            multiprocessing_context="spawn",
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.device(device).type == "cuda",
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
        **options,
    )


def precision_mode(device, precision):
    if precision not in {"auto", "fp32", "fp16"}:
        raise ValueError("Precision must be auto, fp32 or fp16")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; check the torch/driver installation")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 requires CUDA; use fp32 on CPU")
    return device.type == "cuda" and precision != "fp32"


def grad_scaler(enabled):
    # torch.amp.GradScaler was added after the project's minimum torch 2.2.
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)

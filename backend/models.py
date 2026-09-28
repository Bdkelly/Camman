"""Shared detector construction and checkpoint loading; no UI dependencies."""

from pathlib import Path

import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d


def get_fasterrcnn_model_single_class(num_classes=2, *, pretrained=True):
    """Use COCO weights for new training, or build offline when loading weights."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _restore_frozen_batch_norm(module):
    """Match the COCO factory's FrozenBatchNorm and eps without downloading it."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, FrozenBatchNorm2d(child.num_features, eps=0.0))
        else:
            _restore_frozen_batch_norm(child)


def load_model_from_path(model_path, device):
    """Load a raw state dict or a detector training checkpoint without downloads.

    Missing/incompatible weights raise an error instead of silently running a
    randomly initialized ball head.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Detector checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = get_fasterrcnn_model_single_class(pretrained=False)
    # Torchvision uses ordinary BatchNorm for a weights=None backbone, whereas
    # the original COCO-initialized models use FrozenBatchNorm with eps=0.
    # Frozen checkpoints have no batch counters. Keep ordinary BatchNorm when
    # loading a checkpoint trained with --no-pretrained instead.
    if not any(key.endswith("num_batches_tracked") for key in state_dict):
        _restore_frozen_batch_norm(model)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

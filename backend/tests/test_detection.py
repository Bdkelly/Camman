from unittest.mock import Mock

import numpy as np
import pytest
import torch

from backend.detection import FrameTransform, get_ball_detection
from backend.models import load_model_from_path


def test_detection_scales_pixels_and_filters_background():
    model = Mock(
        return_value=[
            {
                "boxes": torch.tensor([[100, 100, 200, 200], [0, 0, 50, 50], [50, 50, 80, 80]]),
                "labels": torch.tensor([1, 0, 1]),
                "scores": torch.tensor([0.99, 1.0, 0.5]),
            }
        ]
    )
    frame = np.zeros((320, 1280, 3), dtype=np.uint8)
    boxes, output = get_ball_detection(model, frame, FrameTransform(), "cpu", draw=False)
    assert boxes == [{"box": (200, 50, 400, 100), "label": "Ball", "score": pytest.approx(0.99)}]
    assert output is frame
    assert model.call_args.args[0][0].shape == (3, 640, 640)


def test_detection_empty_and_multiple():
    prediction = {
        "boxes": torch.tensor([[5, 5, 10, 10], [20, 20, 30, 30]]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.99, 0.995]),
    }
    model = Mock(return_value=[prediction])
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    boxes, _ = get_ball_detection(model, frame, FrameTransform(), "cpu", max_detections=None)
    assert len(boxes) == 2
    boxes, _ = get_ball_detection(model, frame, FrameTransform(), "cpu", confidence_threshold=1.0)
    assert boxes == []


def test_missing_checkpoint_never_builds_random_model(tmp_path, mocker):
    factory = mocker.patch("backend.models.get_fasterrcnn_model_single_class")
    with pytest.raises(FileNotFoundError):
        load_model_from_path(tmp_path / "missing.pth", "cpu")
    factory.assert_not_called()


@pytest.mark.parametrize("wrapped", [False, True])
def test_checkpoint_formats_load_offline(tmp_path, mocker, wrapped):
    source = torch.nn.Linear(2, 1)
    checkpoint = source.state_dict()
    if wrapped:
        checkpoint = {"model_state_dict": checkpoint, "epoch": 2}
    path = tmp_path / "detector.pth"
    torch.save(checkpoint, path)
    target = torch.nn.Linear(2, 1)
    factory = mocker.patch("backend.models.get_fasterrcnn_model_single_class", return_value=target)
    loaded = load_model_from_path(path, "cpu")
    factory.assert_called_once_with(pretrained=False)
    assert not loaded.training
    torch.testing.assert_close(loaded.weight, source.weight)


def test_legacy_preprocessing_matches_training():
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    image = np.random.default_rng(1).integers(0, 256, (30, 50, 3), dtype=np.uint8)
    legacy = A.Compose([A.Resize(640, 640), A.Normalize(), ToTensorV2()])(image=image)["image"]
    torch.testing.assert_close(FrameTransform()(image=image)["image"], legacy, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("frozen", [True, False])
def test_checkpoint_preserves_backbone_normalization(tmp_path, mocker, frozen):
    from torchvision.ops.misc import FrozenBatchNorm2d

    norm = FrozenBatchNorm2d(3, eps=0.0) if frozen else torch.nn.BatchNorm2d(3)
    source = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), norm).eval()
    norm.running_var.fill_(0.1)
    path = tmp_path / "normalization.pth"
    torch.save(source.state_dict(), path)
    target = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.BatchNorm2d(3))
    mocker.patch("backend.models.get_fasterrcnn_model_single_class", return_value=target)
    loaded = load_model_from_path(path, "cpu")
    image = torch.ones(1, 3, 10, 10)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(image), source(image), rtol=1e-6, atol=1e-6)
    assert isinstance(loaded[1], FrozenBatchNorm2d) is frozen

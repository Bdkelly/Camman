import json
from unittest.mock import Mock

import albumentations as A
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from training.data.auto_annotate import annotate
from training.data.clean import load_annotations
from training.data.dataset import BallDataset


@pytest.fixture
def images(tmp_path):
    Image.new("RGB", (100, 50), "white").save(tmp_path / "frame_00000.jpg")
    Image.new("RGB", (100, 50), "black").save(tmp_path / "frame_00001.jpg")
    return tmp_path


def test_clean_preserves_all_boxes_and_negative_frames(images):
    data = [
        {
            "frame_00000.jpg": [
                {"x_min": 10, "y_min": 5, "x_max": 20, "y_max": 10},
                {"x_min": 60, "y_min": 20, "x_max": 70, "y_max": 30},
            ]
        },
        {"frame_00001.jpg": []},
    ]
    path = images / "raw.json"
    path.write_text(json.dumps(data))
    records = load_annotations(path, images)
    assert records == [
        {"frame": "frame_00000.jpg", "boxes": [[10, 5, 20, 10], [60, 20, 70, 30]]},
        {"frame": "frame_00001.jpg", "boxes": []},
    ]
    transform = A.Compose(
        [A.Resize(100, 200), A.Normalize(), ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    dataset = BallDataset(images, records, transform)
    image, target = dataset[0]
    assert image.shape == (3, 100, 200)
    torch.testing.assert_close(
        target["boxes"], torch.tensor([[20.0, 10.0, 40.0, 20.0], [120.0, 40.0, 140.0, 60.0]])
    )
    assert target["labels"].tolist() == [1, 1]
    _, negative = dataset[1]
    assert negative["boxes"].shape == (0, 4)
    assert negative["labels"].shape == (0,)


def test_label_studio_uses_height_for_y_and_skips_disabled(images):
    sequence = [
        {"frame": 0, "x": 10, "y": 20, "width": 20, "height": 10},
        {"frame": 1, "enabled": False},
    ]
    path = images / "ls.json"
    path.write_text(json.dumps([{"box": [{"sequence": sequence}]}]))
    assert load_annotations(path, images) == [
        {"frame": "frame_00000.jpg", "boxes": [[10, 10, 30, 15]]}
    ]


def test_clean_roundtrip_and_duplicate_frame_grouping(images):
    records = [
        {"frame": "frame_00000.jpg", "boxes": [[10, 5, 20, 10]]},
        {"frame": "frame_00001.jpg", "boxes": []},
    ]
    path = images / "clean.json"
    path.write_text(json.dumps(records))
    assert load_annotations(path, images) == records
    legacy = [
        {"frame": "frame_00000.jpg", "x_min": x, "y_min": 5, "x_max": x + 5, "y_max": 10}
        for x in [10, 40]
    ]
    path.write_text(json.dumps(legacy))
    loaded = load_annotations(path, images)
    assert len(loaded) == 1
    assert len(loaded[0]["boxes"]) == 2


def test_missing_image_is_not_replaced_by_dummy_data(images):
    data = BallDataset(images, [{"frame": "missing.jpg", "boxes": []}])
    with pytest.raises(FileNotFoundError):
        data[0]


def test_multi_video_names_are_not_silently_merged(images):
    path = images / "multiple.json"
    path.write_text(json.dumps([{"videos": {"first": {}}}, {"videos": {"second": {}}}]))
    with pytest.raises(ValueError, match="Multiple video"):
        load_annotations(path, images)


def test_invalid_box_is_reported(images):
    path = images / "bad.json"
    path.write_text(json.dumps([{"frame": "frame_00000.jpg", "boxes": [[20, 10, 10, 15]]}]))
    with pytest.raises(ValueError, match="Invalid box"):
        load_annotations(path, images)


def test_annotator_loads_once_and_keeps_negative_frames(images, mocker):
    loader = mocker.patch("training.data.auto_annotate.load_model_from_path", return_value=Mock())
    mocker.patch(
        "training.data.auto_annotate.get_ball_detection",
        return_value=([], np.zeros((50, 100, 3), dtype=np.uint8)),
    )
    output = images / "candidates.json"
    records = annotate(images, "detector.pth", output, device="cpu")
    loader.assert_called_once()
    assert len(records) == 2
    assert all(not boxes for record in records for boxes in record.values())
    assert len(load_annotations(output, images)) == 2

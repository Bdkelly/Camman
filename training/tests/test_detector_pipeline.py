import json
import os
import socket
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from training.data.dataset import BallDataset
from training.detection.evaluate import DetectionCounts, evaluate
from training.detection.loading import make_loader, precision_mode
from training.detection.train import trainer


def test_cache_matches_uncached_and_reuses_pixels(tmp_path, mocker):
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 256, (73, 121, 3), dtype=np.uint8)).save(tmp_path / "frame.jpg")
    records = [{"frame": "frame.jpg", "boxes": [[10, 5, 20, 15], [30, 20, 90, 65]]}]
    original = BallDataset(tmp_path, records)[0]
    cached = BallDataset(tmp_path, records, cache_dir=tmp_path / "cache")
    actual = cached[0]
    torch.testing.assert_close(original[0], actual[0], rtol=0, atol=0)
    torch.testing.assert_close(original[1]["boxes"], actual[1]["boxes"])
    decoder = mocker.patch(
        "training.data.dataset.Image.open", side_effect=AssertionError("Decoded again")
    )
    torch.testing.assert_close(cached[0][0], actual[0])
    decoder.assert_not_called()


def test_cache_invalidation_and_negative_targets(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (80, 40), "white").save(path)
    data = BallDataset(
        tmp_path, [{"frame": "frame.png", "boxes": []}], cache_dir=tmp_path / "cache"
    )
    before, target = data[0]
    assert target["boxes"].shape == (0, 4)
    stat = path.stat()
    Image.new("RGB", (80, 40), "black").save(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
    after, _ = data[0]
    assert not torch.equal(before, after)
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 2
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_cache_does_not_freeze_augmentation(tmp_path):
    Image.new("RGB", (100, 50)).save(tmp_path / "frame.png")
    calls = []

    def transform(**values):
        calls.append(True)
        return {
            "image": torch.zeros(3, 640, 640) + len(calls),
            "bboxes": values["bboxes"],
            "labels": values["labels"],
        }

    dataset = BallDataset(
        tmp_path,
        [{"frame": "frame.png", "boxes": [[10, 5, 20, 10]]}],
        transform,
        cache_dir=tmp_path / "cache",
    )
    first, t1 = dataset[0]
    second, t2 = dataset[0]
    assert not torch.equal(first, second)
    torch.testing.assert_close(t1["boxes"], t2["boxes"])


def test_loader_worker_configuration(tmp_path):
    Image.new("RGB", (20, 20)).save(tmp_path / "frame.png")
    dataset = BallDataset(tmp_path, [{"frame": "frame.png", "boxes": []}])
    single = make_loader(dataset, batch_size=1, workers=0, device="cpu")
    assert not single.persistent_workers and not single.pin_memory
    parallel = make_loader(dataset, batch_size=1, workers=1, device="cpu", prefetch=3)
    assert parallel.persistent_workers and parallel.prefetch_factor == 3


def test_spawned_loader_iteration(tmp_path):
    if hasattr(socket, "AF_UNIX"):
        try:
            probe = socket.socket(socket.AF_UNIX)
        except PermissionError:
            pytest.skip("Sandbox blocks Unix sockets required for PyTorch tensor sharing")
        else:
            probe.close()
    Image.new("RGB", (20, 20)).save(tmp_path / "frame.png")
    dataset = BallDataset(tmp_path, [{"frame": "frame.png", "boxes": []}])
    parallel = make_loader(dataset, batch_size=1, workers=1, device="cpu")
    parallel.timeout = 15
    for _ in range(2):
        assert next(iter(parallel))[0][0].shape == (3, 640, 640)


def prediction(boxes, scores):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores),
        "labels": torch.ones(len(boxes), dtype=torch.int64),
    }


def test_metrics_duplicates_misses_and_negative_frames():
    counts = DetectionCounts()
    truth = {"boxes": torch.tensor([[0.0, 0, 10, 10], [20, 20, 30, 30]])}
    counts.update(prediction([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8]), truth)
    counts.update(prediction([[0, 0, 10, 10]], [0.7]), {"boxes": torch.empty(0, 4)})
    counts.update(prediction([[0, 0, 10, 10]], [0.1]), {"boxes": torch.empty(0, 4)})
    result = counts.result()
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (
        1,
        2,
        1,
    )
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.5
    assert result["f1"] == 0.4
    assert result["negative_images"] == 2 and result["false_alarm_images"] == 1


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.transform = SimpleNamespace(min_size=(800,), max_size=1333)

    def forward(self, images, targets=None):
        if self.training:
            return {"loss": self.weight.square()}
        assert not torch.is_grad_enabled()
        return [prediction([[64, 64, 128, 128]], [0.9]) for _ in images]


def test_training_validation_best_selection_and_metadata(tmp_path, mocker):
    for name in ("train", "val"):
        Image.new("RGB", (100, 50)).save(tmp_path / f"{name}.png")
    model = TinyDetector()
    mocker.patch("training.detection.train.fmodel", return_value=model)
    output = tmp_path / "run"
    trainer(
        [{"frame": "train.png", "boxes": [[10, 5, 20, 10]]}],
        tmp_path,
        2,
        validation_data=[{"frame": "val.png", "boxes": [[10, 5, 20, 10]]}],
        output=output,
        workers=0,
        batch_size=1,
        device="cpu",
        pretrained=False,
        cache_dir=tmp_path / "cache",
        detector_size=640,
    )
    best = torch.load(output / "trained_model_best.pth", weights_only=True)
    final = torch.load(output / "trained_model_final.pth", weights_only=True)
    assert best["validation"]["f1"] == 1
    assert best["epoch"] == 0  # Ties preserve the earliest best validation checkpoint.
    assert final["weight"] < best["model_state_dict"]["weight"]
    assert best["config"]["detector_size"] == 640
    assert model.transform.min_size == (640,)
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert all(m["images_per_second"] > 0 for m in metrics)
    assert len(metrics) == 2
    with pytest.raises(ValueError, match="not empty"):
        trainer([{"frame": "train.png", "boxes": []}], tmp_path, 1, output=output, device="cpu")


def test_split_leakage_is_rejected_before_training(tmp_path):
    records = [{"frame": "frame.png", "boxes": [[1, 1, 2, 2]]}]
    with pytest.raises(ValueError, match="share images"):
        trainer(
            records, tmp_path, 1, validation_data=records, device="cpu", output=tmp_path / "run"
        )
    assert not (tmp_path / "run").exists()


def test_cpu_fp16_rejected():
    with pytest.raises(ValueError, match="requires CUDA"):
        precision_mode("cpu", "fp16")


def test_evaluate_restores_model_mode():
    model = TinyDetector().train()
    model.roi_heads = SimpleNamespace(score_thresh=0.05)
    loader = [([torch.zeros(3, 640, 640)], [{"boxes": torch.tensor([[64, 64, 128, 128]])}])]
    assert evaluate(model, loader, "cpu", confidence=0.01)["f1"] == 1
    assert model.training and model.roi_heads.score_thresh == 0.05


def test_evaluation_cli_preserves_numeric_precision(tmp_path, mocker):
    from training.detection.evaluate import main

    Image.new("RGB", (100, 50)).save(tmp_path / "val.png")
    labels = tmp_path / "val.json"
    labels.write_text(json.dumps([{"frame": "val.png", "boxes": [[10, 5, 20, 10]]}]))
    mocker.patch("training.detection.evaluate.load_model_from_path", return_value=TinyDetector())
    report = tmp_path / "metrics.json"
    main(
        [
            "--model",
            "test.pth",
            "--annotations",
            str(labels),
            "--images",
            str(tmp_path),
            "--device",
            "cpu",
            "--workers",
            "0",
            "--output",
            str(report),
        ]
    )
    metrics = json.loads(report.read_text())
    assert metrics["precision"] == 1.0
    assert metrics["inference_precision"] == "fp32"

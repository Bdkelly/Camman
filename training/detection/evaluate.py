"""Evaluate ball boxes at a fixed confidence/IoU threshold (not COCO mAP)."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torchvision.ops import box_iou

from backend.models import load_model_from_path
from training.data.clean import load_annotations
from training.data.dataset import BallDataset
from training.detection.loading import make_loader, precision_mode


class DetectionCounts:
    def __init__(self, confidence=0.5, iou=0.5):
        if not 0 <= confidence <= 1 or not 0 < iou <= 1:
            raise ValueError("Confidence must be 0..1; IoU must be >0 and <=1")
        self.confidence, self.iou = confidence, iou
        self.tp = self.fp = self.fn = self.images = self.negative_images = (
            self.false_alarm_images
        ) = 0

    def update(self, prediction, target):
        boxes = prediction["boxes"].detach().float().cpu()
        scores = prediction["scores"].detach().float().cpu()
        labels = prediction["labels"].detach().cpu()
        truth = target["boxes"].detach().float().cpu()
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise ValueError("Detector returned non-finite predictions")
        keep = (scores >= self.confidence) & (labels == 1)
        boxes = boxes[keep][scores[keep].argsort(descending=True)]
        overlaps = box_iou(boxes, truth)
        matched = set()
        for row in overlaps:
            # Match each prediction to the best still-unmatched target, once.
            candidates = [i for i in range(len(truth)) if i not in matched]
            best = max(candidates, key=lambda i: float(row[i])) if candidates else None
            if best is not None and row[best] >= self.iou:
                matched.add(best)
                self.tp += 1
            else:
                self.fp += 1
        self.fn += len(truth) - len(matched)
        self.images += 1
        if not len(truth):
            self.negative_images += 1
            self.false_alarm_images += bool(len(boxes))

    def result(self):
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "true_positives": self.tp,
            "false_positives": self.fp,
            "false_negatives": self.fn,
            "images": self.images,
            "ground_truth_boxes": self.tp + self.fn,
            "negative_images": self.negative_images,
            "false_alarm_images": self.false_alarm_images,
            "confidence": self.confidence,
            "iou": self.iou,
            "metric": "single-class fixed-threshold box precision/recall/F1; not COCO mAP",
        }


@torch.inference_mode()
def evaluate(model, loader, device, *, precision="fp32", confidence=0.5, iou=0.5):
    amp = precision_mode(device, precision)
    device = torch.device(device)
    counts = DetectionCounts(confidence, iou)
    was_training = model.training
    roi_heads = getattr(model, "roi_heads", None)
    original_threshold = getattr(roi_heads, "score_thresh", None)
    if original_threshold is not None:
        # Permit evaluating a threshold below torchvision's default 0.05 filter.
        roi_heads.score_thresh = min(original_threshold, confidence)
    model.eval()
    try:
        for images, targets in loader:
            images = [image.to(device, non_blocking=device.type == "cuda") for image in images]
            with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
                predictions = model(images)
            if len(predictions) != len(targets):
                raise ValueError("Detector prediction count does not match batch")
            for prediction, target in zip(predictions, targets):
                counts.update(prediction, target)
    finally:
        if original_threshold is not None:
            roi_heads.score_thresh = original_threshold
        model.train(was_training)
    return counts.result()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True, help="Write metrics JSON to a new file")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--cache-dir")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16"), default="fp32")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--detector-size", type=int, default=800)
    args = parser.parse_args(argv)
    if args.detector_size < 128 or args.detector_size > 1536 or args.detector_size % 32:
        parser.error("Detector size must be a multiple of 32 between 128 and 1536")
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; choose a new metrics filename")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    precision_mode(device, args.precision)
    DetectionCounts(args.confidence, args.iou)
    records = load_annotations(args.annotations, args.images)
    dataset = BallDataset(args.images, records, cache_dir=args.cache_dir)
    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        prefetch=args.prefetch,
    )
    model = load_model_from_path(args.model, device, detector_size=args.detector_size)
    result = evaluate(
        model, loader, device, precision=args.precision, confidence=args.confidence, iou=args.iou
    )
    result.update(
        model=str(Path(args.model).resolve()),
        annotations=str(Path(args.annotations).resolve()),
        detector_size=args.detector_size,
        inference_precision=args.precision,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

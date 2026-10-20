"""Train the detector with bounded data prefetch, optional cache/AMP and held-out metrics."""

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import LinearLR, SequentialLR, StepLR

from backend.models import get_fasterrcnn_model_single_class as fmodel
from training.data.clean import load_annotations
from training.data.dataset import BallDataset
from training.detection.evaluate import DetectionCounts, evaluate
from training.detection.loading import grad_scaler, make_loader, precision_mode


def training_transform(seed):
    # Preserve historical preprocessing; cache is placed before stochastic transforms.
    transform = A.Compose(
        [
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)
    return transform


def check_disjoint(train, train_root, validation, validation_root):
    paths = {(Path(train_root) / r["frame"]).resolve() for r in train}
    overlap = paths & {(Path(validation_root) / r["frame"]).resolve() for r in validation}
    if overlap:
        raise ValueError(f"Training/validation share images, e.g. {next(iter(overlap))}")


def trainer(
    jsondata,
    imgdir,
    num_epochs,
    *,
    output="artifacts/detection",
    batch_size=8,
    workers=4,
    device=None,
    pretrained=True,
    precision="auto",
    prefetch=2,
    cache_dir=None,
    detector_size=800,
    seed=0,
    validation_data=None,
    validation_images=None,
    eval_every=1,
    confidence=0.5,
    iou=0.5,
):
    if num_epochs < 1 or batch_size < 1 or workers < 0 or prefetch < 1 or eval_every < 1:
        raise ValueError("Epochs/batch size/prefetch/eval interval must be positive; workers >= 0")
    if detector_size < 128 or detector_size > 1536 or detector_size % 32:
        raise ValueError("Detector size must be a multiple of 32 between 128 and 1536")
    DetectionCounts(confidence, iou)
    if not jsondata:
        raise ValueError("Training data is empty")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = precision_mode(device, precision)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    validation_images = validation_images or imgdir
    if validation_data is not None:
        if not validation_data or not any(r["boxes"] for r in validation_data):
            raise ValueError("Validation needs reviewed images including positive ball examples")
        check_disjoint(jsondata, imgdir, validation_data, validation_images)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; use a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    dataset = BallDataset(imgdir, jsondata, training_transform(seed), cache_dir=cache_dir)
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        device=device,
        shuffle=True,
        seed=seed,
        prefetch=prefetch,
    )
    val_loader = None
    if validation_data is not None:
        val_loader = make_loader(
            BallDataset(validation_images, validation_data, cache_dir=cache_dir),
            batch_size=batch_size,
            workers=workers,
            device=device,
            seed=seed,
            prefetch=prefetch,
        )
    config = {
        "epochs": num_epochs,
        "batch_size": batch_size,
        "workers": workers,
        "device": str(device),
        "precision": "fp16" if amp else "fp32",
        "prefetch": prefetch,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "detector_size": detector_size,
        "seed": seed,
        "pretrained": pretrained,
        "eval_every": eval_every,
        "validation_precision": "fp32",
        "confidence": confidence,
        "iou": iou,
        "images": str(Path(imgdir).resolve()),
        "validation_images": str(Path(validation_images).resolve()) if val_loader else None,
        "torch": str(torch.__version__),
        "preprocessing": "640px RGB, external ImageNet normalization",
    }
    (output / "run.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for name, records in (
        ("train-records.json", jsondata),
        ("validation-records.json", validation_data),
    ):
        if records is not None:
            (output / name).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Using {device}; precision {config['precision']}; internal size {detector_size}px")
    model = fmodel(2, pretrained=pretrained).to(device)
    if hasattr(model, "transform"):
        # Avoid the invisible 640->800 resize when a smaller size is requested.
        model.transform.min_size = (detector_size,)
        model.transform.max_size = detector_size
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0001)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.2, total_iters=5),
            StepLR(optimizer, step_size=10, gamma=0.1),
        ],
        milestones=[5],
    )
    scaler = grad_scaler(amp)
    best_f1 = -1.0
    for epoch in range(num_epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        total = torch.zeros((), device=device)
        seen, wait_s = 0, 0.0
        iterator = iter(loader)
        while True:
            waiting = time.perf_counter()
            try:
                images, targets = next(iterator)
            except StopIteration:
                break
            wait_s += time.perf_counter() - waiting
            images = [im.to(device, non_blocking=device.type == "cuda") for im in images]
            targets = [
                {k: v.to(device, non_blocking=device.type == "cuda") for k, v in t.items()}
                for t in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
                losses = sum(model(images, targets).values())
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=not amp
            )
            scaler.step(optimizer)
            scaler.update()
            # Keep loss on device: no loss.item() synchronization every batch.
            total += losses.detach() * len(images)
            seen += len(images)
        average = total.item() / seen
        if not math.isfinite(average):
            raise RuntimeError(
                "Non-finite training loss; no model saved for this epoch. Try fp32 and inspect labels."
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        scheduler.step()
        metrics = {
            "epoch": epoch + 1,
            "loss": average,
            "train_seconds": seconds,
            "images_per_second": seen / seconds,
            "loader_wait_seconds": wait_s,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val_loader and ((epoch + 1) % eval_every == 0 or epoch + 1 == num_epochs):
            validation_start = time.perf_counter()
            metrics["validation"] = evaluate(
                model, val_loader, device, confidence=confidence, iou=iou
            )
            metrics["validation_seconds"] = time.perf_counter() - validation_start
            if metrics["validation"]["f1"] > best_f1:
                best_f1 = metrics["validation"]["f1"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "validation": metrics["validation"],
                    },
                    output / "trained_model_best.pth",
                )
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        print(json.dumps(metrics))
        if (epoch + 1) % 10 == 0 or epoch + 1 == num_epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": config,
                    "loss": average,
                },
                output / f"checkpoint_epoch_{epoch + 1}.pth",
            )
    torch.save(model.state_dict(), output / "trained_model_final.pth")
    print(f"Training complete: {output}. Evaluate on untouched games before deployment.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--epochs", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default="artifacts/detection")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16"), default="auto")
    parser.add_argument("--prefetch", type=int, default=2, help="Batches prefetched per worker")
    parser.add_argument("--cache-dir", help="Optional local SSD cache of resized RGB images")
    parser.add_argument("--detector-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-annotations", help="Separate held-out game labels")
    parser.add_argument("--validation-images", help="Defaults to --images")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Fixed validation score threshold"
    )
    parser.add_argument("--iou", type=float, default=0.5, help="Validation box-matching threshold")
    args = parser.parse_args(argv)
    if args.validation_images and not args.validation_annotations:
        parser.error("--validation-images requires --validation-annotations")
    records = load_annotations(args.annotations, args.images)
    validation = (
        load_annotations(args.validation_annotations, args.validation_images or args.images)
        if args.validation_annotations
        else None
    )
    trainer(
        records,
        args.images,
        args.epochs,
        output=args.output,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        pretrained=not args.no_pretrained,
        precision=args.precision,
        prefetch=args.prefetch,
        cache_dir=args.cache_dir,
        detector_size=args.detector_size,
        seed=args.seed,
        validation_data=validation,
        validation_images=args.validation_images,
        eval_every=args.eval_every,
        confidence=args.confidence,
        iou=args.iou,
    )


if __name__ == "__main__":
    main()

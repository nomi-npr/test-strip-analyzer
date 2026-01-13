#!/usr/bin/env python
"""Training script aligned with the paper replication plan."""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from training.dataset import (
    StripDataset,
    TransformConfig,
    build_eval_transforms,
    build_positive_transforms,
    build_train_transforms,
    compute_pos_weight,
    compute_sample_weights,
)
from training.evaluation import compute_metrics, tune_thresholds
from training.models import FocalLoss, build_model

PAPER_JOINT_RATIO = (6.0, 2.0, 2.0, 1.0)


@dataclass
class RunConfig:
    approach: str
    task: str
    model: str
    optimizer: str
    epochs: int
    batch_size: int
    lr: float
    loss: str
    use_pos_weight: bool
    use_class_weights: bool
    target_pos_ratio: tuple[float, ...] | None
    target_joint_ratio: tuple[float, float, float, float] | None
    input_size: int
    normalization: str
    num_workers: int
    seed: int
    device: str
    dataset_split: str
    pretrained: bool
    freeze_backbone: bool
    head_hidden: int
    betas: tuple[float, float] | None
    epsilon: float | None
    momentum: float | None
    init_from: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train strip classifier (paper pipeline).")
    parser.add_argument("--task", choices=["binary", "multilabel"], default="multilabel")
    parser.add_argument(
        "--model",
        choices=["vanilla_cnn", "nasnetmobile", "densenet121", "resnet50"],
        default="vanilla_cnn",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument(
        "--target-pos-ratio",
        type=str,
        default=None,
        help="Comma-separated target positive ratios, e.g., 0.3 or 0.3,0.3",
    )
    parser.add_argument(
        "--target-joint-ratio",
        type=str,
        default=None,
        help="Comma-separated joint ratios (neg,t1,t2,both) or 'paper'",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--normalization", choices=["minmax", "imagenet"], default="minmax")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--positive-augment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu | cuda | mps | auto",
    )
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--num-interop-threads", type=int, default=None)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument(
        "--epoch-samples-multiplier",
        type=float,
        default=1.0,
        help="Multiply number of samples per epoch when using sampling.",
    )
    parser.add_argument(
        "--positive-sample-multiplier",
        type=float,
        default=1.0,
        help="Extra weight multiplier for samples with any positive label.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip missing image files instead of raising an error.",
    )
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument(
        "--synthetic-train-csv",
        type=Path,
        default=None,
        help="Optional synthetic training manifest.",
    )
    parser.add_argument(
        "--synthetic-test-csv",
        type=Path,
        default=None,
        help="Optional synthetic test manifest.",
    )
    parser.add_argument(
        "--include-synthetic-train",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--evaluate-synthetic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--images-root", type=Path, default=Path("data/images"))
    parser.add_argument(
        "--synthetic-images-root",
        type=Path,
        default=Path("data/synthetic"),
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("training/runs"))
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Optional path to a model checkpoint to initialize weights.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "recall", "precision"],
        default="f1",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print training progress.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Batches between training progress logs.",
    )
    parser.add_argument(
        "--export-cpu-optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export a CPU-optimized TorchScript model.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str) -> torch.device:
    device = device.lower()
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def export_cpu_optimized(model: nn.Module, input_size: int, output_path: Path) -> None:
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    example = torch.randn(1, 3, input_size, input_size)
    traced = torch.jit.trace(model_cpu, example)
    try:
        traced = torch.jit.freeze(traced)
    except Exception:
        pass
    try:
        optimized = torch.jit.optimize_for_inference(traced)
    except Exception:
        optimized = traced
    torch.jit.save(optimized, output_path)


def resolve_target_pos_ratio(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    parts = [float(p.strip()) for p in value.split(",") if p.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        return (parts[0],)
    if len(parts) == 2:
        return (parts[0], parts[1])
    raise ValueError("--target-pos-ratio must have one or two comma-separated values")


def resolve_target_joint_ratio(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if value.strip().lower() in {"paper", "default"}:
        return PAPER_JOINT_RATIO
    parts = [float(p.strip()) for p in value.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--target-joint-ratio must have four comma-separated values")
    return (parts[0], parts[1], parts[2], parts[3])


def build_dataset(
    split_csv: Path,
    images_root: Path,
    transform: object,
    label_mode: str,
    positive_transform: object | None,
    skip_missing: bool,
) -> StripDataset:
    return StripDataset(
        split_csv=split_csv,
        images_root=images_root,
        transform=transform,
        positive_transform=positive_transform,
        label_mode=label_mode,
        skip_missing=skip_missing,
    )


def _concat_datasets(datasets: list[StripDataset]) -> tuple[torch.utils.data.Dataset, np.ndarray]:
    if len(datasets) == 1:
        dataset = datasets[0]
        return dataset, dataset.get_labels()
    labels = np.concatenate([ds.get_labels() for ds in datasets], axis=0)
    dataset = ConcatDataset(datasets)
    return dataset, labels


def _count_missing(dataset: torch.utils.data.Dataset) -> int:
    if hasattr(dataset, "get_missing_paths"):
        return len(dataset.get_missing_paths())
    if isinstance(dataset, ConcatDataset):
        return sum(_count_missing(ds) for ds in dataset.datasets)
    return 0


def build_dataloaders(
    args: argparse.Namespace,
    config: TransformConfig,
    target_pos_ratio: tuple[float, ...] | None,
    target_joint_ratio: tuple[float, float, float, float] | None,
) -> tuple[DataLoader, DataLoader, DataLoader | None, DataLoader | None, np.ndarray, int]:
    train_tf = build_train_transforms(config)
    val_tf = build_eval_transforms(config)
    positive_tf = build_positive_transforms(config) if args.positive_augment else None

    label_mode = args.task

    train_datasets: list[StripDataset] = [
        build_dataset(
            args.train_csv,
            args.images_root / "train",
            transform=train_tf,
            label_mode=label_mode,
            positive_transform=positive_tf,
            skip_missing=args.skip_missing,
        )
    ]

    if args.include_synthetic_train:
        if args.task != "binary":
            raise ValueError("Synthetic training is only supported for binary task.")
        train_datasets.append(
            build_dataset(
                args.synthetic_train_csv,
                args.synthetic_images_root,
                transform=train_tf,
                label_mode=label_mode,
                positive_transform=positive_tf,
                skip_missing=args.skip_missing,
            )
        )

    train_dataset, labels = _concat_datasets(train_datasets)

    val_dataset = build_dataset(
        args.val_csv,
        args.images_root / "val",
        transform=val_tf,
        label_mode=label_mode,
        positive_transform=None,
        skip_missing=args.skip_missing,
    )

    test_dataset = None
    if args.test_csv is not None and args.test_csv.exists():
        test_dataset = build_dataset(
            args.test_csv,
            args.images_root / "test",
            transform=val_tf,
            label_mode=label_mode,
            positive_transform=None,
            skip_missing=args.skip_missing,
        )

    synthetic_test_dataset = None
    if args.evaluate_synthetic and args.synthetic_test_csv is not None:
        synthetic_test_dataset = build_dataset(
            args.synthetic_test_csv,
            args.synthetic_images_root,
            transform=val_tf,
            label_mode=label_mode,
            positive_transform=None,
            skip_missing=args.skip_missing,
        )

    sampler = None
    shuffle = True
    effective_samples = len(labels)
    use_sampling = (
        target_pos_ratio is not None
        or target_joint_ratio is not None
        or args.positive_sample_multiplier > 1.0
        or args.epoch_samples_multiplier != 1.0
    )
    if use_sampling:
        weights = compute_sample_weights(
            labels,
            target_pos_ratio=target_pos_ratio,
            target_joint_ratio=target_joint_ratio,
            positive_boost=args.positive_sample_multiplier,
        )
        num_samples = max(1, int(len(weights) * max(args.epoch_samples_multiplier, 1.0)))
        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
        shuffle = False
        effective_samples = num_samples

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        pin_memory=args.pin_memory,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
            pin_memory=args.pin_memory,
        )

    synthetic_test_loader = None
    if synthetic_test_dataset is not None:
        synthetic_test_loader = DataLoader(
            synthetic_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
            pin_memory=args.pin_memory,
        )

    return train_loader, val_loader, test_loader, synthetic_test_loader, labels, effective_samples


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    task: str,
    verbose: bool = True,
    log_interval: int = 50,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    best_metric = -1.0
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            if task == "binary":
                labels = labels.long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            if verbose and log_interval > 0 and batch_idx % log_interval == 0:
                print(
                    f"Epoch {epoch}/{epochs} | step {batch_idx}/{len(train_loader)} | "
                    f"loss {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader.dataset)
        val_logits, val_labels = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_logits, val_labels, task=task)
        val_score = float(val_metrics.get("macro_f1", val_metrics.get("f1", 0.0)))

        history.append({"epoch": epoch, "train_loss": train_loss, "val_score": val_score})
        if verbose:
            print(
                f"Epoch {epoch}/{epochs} done | train_loss {train_loss:.4f} | "
                f"val_score {val_score:.4f}"
            )

        if val_score > best_metric:
            best_metric = val_score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_logits, val_labels = evaluate(model, val_loader, device)
    return {"history": history, "best_score": best_metric}, val_logits, val_labels


def write_run_readme(run_dir: Path, metrics: dict[str, object]) -> None:
    content = [
        "# Training Run Summary",
        "",
        "## Metrics",
        "",
        json.dumps(metrics, indent=2),
        "",
        "## Next Steps",
        "- Compare real vs synthetic test metrics",
        "- Update the results summary table",
        "- Run cross-validation and paired t-test if needed",
    ]
    (run_dir / "README.md").write_text("\n".join(content), encoding="utf-8")


def _resolve_default_csv(task: str, split: str) -> Path:
    suffix = "binary" if task == "binary" else "multilabel"
    return Path(f"data/splits/{split}_{suffix}.csv")


def _resolve_default_synthetic_csv(split: str) -> Path:
    return Path(f"data/splits/synthetic_{split}.csv")


def _compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor | None:
    labels = np.asarray(labels).astype(int)
    pos = labels.sum()
    neg = labels.shape[0] - pos
    if pos == 0 or neg == 0:
        return None
    weights = np.array([1.0, float(neg / pos)], dtype=np.float32)
    return torch.tensor(weights, device=device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    if args.num_interop_threads is not None:
        torch.set_num_interop_threads(args.num_interop_threads)

    if args.train_csv is None:
        args.train_csv = _resolve_default_csv(args.task, "train")
    if args.val_csv is None:
        args.val_csv = _resolve_default_csv(args.task, "val")
    if args.test_csv is None:
        args.test_csv = _resolve_default_csv(args.task, "test")

    if args.include_synthetic_train and args.synthetic_train_csv is None:
        args.synthetic_train_csv = _resolve_default_synthetic_csv("train")
    if args.evaluate_synthetic and args.synthetic_test_csv is None:
        args.synthetic_test_csv = _resolve_default_synthetic_csv("test")
    if args.include_synthetic_train and args.synthetic_train_csv is not None:
        if not args.synthetic_train_csv.exists():
            print(
                "Warning: Synthetic train manifest not found at "
                f"{args.synthetic_train_csv}. Disabling synthetic training."
            )
            args.include_synthetic_train = False
    if args.evaluate_synthetic and args.synthetic_test_csv is not None:
        if not args.synthetic_test_csv.exists():
            print(
                "Warning: Synthetic test manifest not found at "
                f"{args.synthetic_test_csv}. Disabling synthetic evaluation."
            )
            args.evaluate_synthetic = False

    if args.task == "multilabel" and args.target_joint_ratio is None:
        args.target_joint_ratio = "paper"

    device = resolve_device(args.device)
    if args.verbose:
        print(f"Using device: {device}")

    target_pos_ratio = resolve_target_pos_ratio(args.target_pos_ratio)
    target_joint_ratio = resolve_target_joint_ratio(args.target_joint_ratio)
    config = TransformConfig(input_size=args.input_size, normalization=args.normalization)

    if args.verbose:
        print("Building dataloaders...")
    train_loader, val_loader, test_loader, synthetic_test_loader, train_labels, effective_samples = (
        build_dataloaders(args, config, target_pos_ratio, target_joint_ratio)
    )
    if args.verbose:
        print(
            f"Loaded data | train {len(train_loader.dataset)} | "
            f"val {len(val_loader.dataset)} | "
            f"test {len(test_loader.dataset) if test_loader is not None else 0}"
        )
        missing_train = _count_missing(train_loader.dataset)
        missing_val = _count_missing(val_loader.dataset)
        missing_test = _count_missing(test_loader.dataset) if test_loader is not None else 0
        if missing_train or missing_val or missing_test:
            print(
                "Warning: skipped missing images | "
                f"train {missing_train} | val {missing_val} | test {missing_test}"
            )
        if synthetic_test_loader is not None:
            print(f"Synthetic test {len(synthetic_test_loader.dataset)}")
        if args.epoch_samples_multiplier != 1.0 or args.positive_sample_multiplier > 1.0:
            print(
                f"Sampling enabled | epoch_samples_multiplier={args.epoch_samples_multiplier} | "
                f"positive_sample_multiplier={args.positive_sample_multiplier}"
            )
            print(f"Effective samples per epoch: {effective_samples}")

    if args.verbose:
        print(f"Loading model {args.model} (pretrained={args.pretrained})...")
    model = build_model(
        args.model,
        num_outputs=2,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        input_size=args.input_size,
        head_hidden=args.head_hidden,
    )
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    if args.init_from is not None:
        if not args.init_from.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.init_from}")
        state = torch.load(args.init_from, map_location=device)
        model.load_state_dict(state, strict=True)
        if args.verbose:
            print(f"Initialized weights from {args.init_from}")

    criterion: nn.Module
    pos_weight = None
    class_weights = None
    if args.task == "binary":
        if args.use_class_weights:
            class_weights = _compute_class_weights(train_labels, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        if args.use_pos_weight:
            pos_weight_np = compute_pos_weight(train_labels)
            pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
            if args.verbose:
                print(f"Using pos_weight: {pos_weight_np.tolist()}")
        if args.loss == "focal":
            criterion = FocalLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        betas = None
        epsilon = None
        momentum = args.momentum
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.epsilon,
        )
        betas = (args.beta1, args.beta2)
        epsilon = args.epsilon
        momentum = None

    dataset_split = "real"
    if args.include_synthetic_train:
        dataset_split = "real+synthetic"

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.runs_dir / f"paper_{args.task}_{args.model}_{dataset_split}_{args.input_size}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.verbose:
        print(f"Starting training for {args.epochs} epochs | run_dir={run_dir}")

    train_summary, val_logits, val_labels = train_loop(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=args.epochs,
        task=args.task,
        verbose=args.verbose,
        log_interval=args.log_interval,
    )

    thresholds = None
    val_metrics = None
    if args.task == "multilabel":
        thresholds = tune_thresholds(val_logits, val_labels, metric=args.threshold_metric)
        val_metrics = compute_metrics(val_logits, val_labels, thresholds=thresholds, task=args.task)
        if args.verbose:
            print(f"Tuned thresholds: {thresholds.tolist()}")
    else:
        val_metrics = compute_metrics(val_logits, val_labels, task=args.task)

    test_metrics = None
    if test_loader is not None:
        test_logits, test_labels = evaluate(model, test_loader, device)
        test_metrics = compute_metrics(
            test_logits,
            test_labels,
            thresholds=thresholds,
            task=args.task,
        )
        if args.verbose:
            print("Evaluated on test split.")

    synthetic_test_metrics = None
    if synthetic_test_loader is not None:
        synth_logits, synth_labels = evaluate(model, synthetic_test_loader, device)
        synthetic_test_metrics = compute_metrics(
            synth_logits,
            synth_labels,
            thresholds=thresholds,
            task=args.task,
        )
        if args.verbose:
            print("Evaluated on synthetic test split.")

    metrics = {
        "train_summary": train_summary,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "synthetic_test_metrics": synthetic_test_metrics,
        "thresholds": None if thresholds is None else thresholds.tolist(),
    }

    config_obj = RunConfig(
        approach=f"paper_{args.task}",
        task=args.task,
        model=args.model,
        optimizer=args.optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss=args.loss,
        use_pos_weight=args.use_pos_weight,
        use_class_weights=args.use_class_weights,
        target_pos_ratio=target_pos_ratio,
        target_joint_ratio=target_joint_ratio,
        input_size=args.input_size,
        normalization=args.normalization,
        num_workers=args.num_workers,
        seed=args.seed,
        device=str(device),
        dataset_split=dataset_split,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        head_hidden=args.head_hidden,
        betas=betas,
        epsilon=epsilon,
        momentum=momentum,
        init_from=None if args.init_from is None else str(args.init_from),
    )

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config_obj), handle, indent=2)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    torch.save(model.state_dict(), run_dir / "model.pt")
    if args.export_cpu_optimized:
        cpu_model_path = run_dir / "model_cpu.ts"
        export_cpu_optimized(model, args.input_size, cpu_model_path)
        if args.verbose:
            print(f"Saved CPU-optimized model to {cpu_model_path}")
    write_run_readme(run_dir, metrics)

    print(f"Run artifacts written to {run_dir}")


if __name__ == "__main__":
    main()

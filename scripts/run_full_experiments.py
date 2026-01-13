#!/usr/bin/env python
"""Orchestrate paper experiment grid."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper experiment grid.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["binary", "multilabel"],
        choices=["binary", "multilabel"],
    )
    parser.add_argument(
        "--input-sizes",
        nargs="+",
        type=int,
        default=[224, 200, 128],
    )
    parser.add_argument(
        "--baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include vanilla CNN baselines.",
    )
    parser.add_argument(
        "--transfer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include transfer learning models.",
    )
    parser.add_argument(
        "--baseline-optimizers",
        nargs="+",
        default=["adam", "sgd"],
        choices=["adam", "sgd"],
    )
    parser.add_argument(
        "--transfer-optimizer",
        default="adam",
        choices=["adam", "sgd"],
    )
    parser.add_argument(
        "--transfer-models",
        nargs="+",
        default=["nasnetmobile", "densenet121", "resnet50"],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--normalization", choices=["minmax", "imagenet"], default="minmax")
    parser.add_argument("--runs-dir", type=Path, default=Path("training/runs"))
    parser.add_argument("--log-interval", type=int, default=50)
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
    parser.add_argument(
        "--synthetic-finetune",
        choices=["best", "all", "none"],
        default="best",
        help="Which real-trained models to fine-tune with synthetic data.",
    )
    parser.add_argument("--synthetic-epochs", type=int, default=None)
    parser.add_argument("--synthetic-lr", type=float, default=None)
    parser.add_argument(
        "--auto-build-synthetic-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build synthetic manifests if missing and synthetic data exists.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--force-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow parallel jobs even when MPS is detected (may be slower/unstable).",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-summary",
        type=Path,
        default=Path("training/runs/paper_experiment_summary.json"),
    )
    return parser.parse_args()


@dataclass
class Job:
    tag: str
    cmd: list[str]
    task: str
    model: str
    optimizer: str
    input_size: int


@dataclass
class RunResult:
    tag: str
    returncode: int
    run_dir: str | None = None


def _stream_process(cmd: list[str], tag: str) -> RunResult:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    run_dir = None
    assert process.stdout is not None
    for line in process.stdout:
        if "Run artifacts written to" in line:
            run_dir = line.strip().split("Run artifacts written to", 1)[1].strip()
        print(f"[{tag}] {line}", end="")
    returncode = process.wait()
    return RunResult(tag=tag, returncode=returncode, run_dir=run_dir)


def _maybe_force_sequential(device: str, parallel: int, force_parallel: bool) -> int:
    if parallel <= 1:
        return parallel
    if force_parallel:
        return parallel
    if device in {"mps", "auto"}:
        try:
            import torch

            if torch.backends.mps.is_available():
                print("Detected MPS; forcing --parallel=1 to avoid GPU contention.")
                return 1
        except Exception:
            return 1
    return parallel


def _build_train_cmd(
    args: argparse.Namespace,
    task: str,
    model: str,
    optimizer: str,
    input_size: int,
    epochs: int | None = None,
    lr: float | None = None,
    init_from: str | None = None,
    include_synthetic: bool = False,
    evaluate_synthetic: bool = False,
    pretrained: bool | None = None,
    freeze_backbone: bool | None = None,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "training/train.py",
        "--task",
        task,
        "--model",
        model,
        "--optimizer",
        optimizer,
        "--epochs",
        str(epochs if epochs is not None else args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--input-size",
        str(input_size),
        "--normalization",
        args.normalization,
        "--runs-dir",
        str(args.runs_dir),
        "--log-interval",
        str(args.log_interval),
    ]
    if lr is not None:
        cmd.extend(["--lr", str(lr)])
    if pretrained is None:
        pretrained = model != "vanilla_cnn"
    if freeze_backbone is None:
        freeze_backbone = model != "vanilla_cnn"
    if pretrained:
        cmd.append("--pretrained")
    if freeze_backbone:
        cmd.append("--freeze-backbone")
    if init_from is not None:
        cmd.extend(["--init-from", init_from])
    if task == "binary":
        if include_synthetic:
            cmd.append("--include-synthetic-train")
        if evaluate_synthetic:
            cmd.append("--evaluate-synthetic")
    return cmd


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_metric(metrics: dict | None, task: str) -> float | None:
    if not metrics:
        return None
    if task == "binary":
        return metrics.get("f1") or metrics.get("accuracy")
    return metrics.get("macro_f1")


def _select_best_run(
    run_dirs: Iterable[str],
    task: str,
    allowed_models: set[str],
) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    best_score = -1.0
    for run_dir in run_dirs:
        config_path = Path(run_dir) / "config.json"
        metrics_path = Path(run_dir) / "metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        config = _load_json(config_path)
        if config.get("task") != task:
            continue
        if config.get("dataset_split", "real") != "real":
            continue
        if config.get("model") not in allowed_models:
            continue
        metrics = _load_json(metrics_path)
        score = _extract_metric(metrics.get("val_metrics"), task)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best = {"run_dir": run_dir, "config": config, "val_score": score}
    return best


def _ensure_synthetic_manifests(args: argparse.Namespace) -> bool:
    synthetic_train_manifest = Path("data/splits/synthetic_train.csv")
    synthetic_test_manifest = Path("data/splits/synthetic_test.csv")
    if synthetic_train_manifest.exists() and synthetic_test_manifest.exists():
        return True
    if not args.auto_build_synthetic_manifest:
        return False
    pos_dir = Path("data/synthetic/positive")
    neg_dir = Path("data/synthetic/negative")
    if not pos_dir.exists() or not neg_dir.exists():
        return False
    print("Building synthetic manifests from data/synthetic/positive and negative...")
    cmd = ["uv", "run", "scripts/build_synthetic_manifest.py"]
    result = _stream_process(cmd, "synthetic-manifest")
    return result.returncode == 0


def _run_jobs(jobs: Iterable[Job], parallel: int, continue_on_error: bool) -> dict[str, RunResult]:
    results: dict[str, RunResult] = {}
    if parallel <= 1:
        for job in jobs:
            result = _stream_process(job.cmd, job.tag)
            results[job.tag] = result
            if result.returncode != 0 and not continue_on_error:
                print(f"Job {job.tag} failed with code {result.returncode}")
                sys.exit(result.returncode)
        return results

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_map = {executor.submit(_stream_process, job.cmd, job.tag): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            result = future.result()
            results[job.tag] = result
            if result.returncode != 0 and not continue_on_error:
                print(f"Job {job.tag} failed with code {result.returncode}")
                sys.exit(result.returncode)
    return results


def main() -> None:
    args = parse_args()
    args.parallel = _maybe_force_sequential(args.device, args.parallel, args.force_parallel)

    jobs: list[Job] = []
    if args.baseline:
        for task in args.tasks:
            for input_size in args.input_sizes:
                for optimizer in args.baseline_optimizers:
                    tag = f"baseline-{task}-vanilla_cnn-{optimizer}-{input_size}"
                    cmd = _build_train_cmd(
                        args,
                        task,
                        "vanilla_cnn",
                        optimizer,
                        input_size,
                        include_synthetic=False,
                        evaluate_synthetic=False,
                    )
                    jobs.append(
                        Job(
                            tag=tag,
                            cmd=cmd,
                            task=task,
                            model="vanilla_cnn",
                            optimizer=optimizer,
                            input_size=input_size,
                        )
                    )

    if args.transfer:
        for task in args.tasks:
            for input_size in args.input_sizes:
                for model in args.transfer_models:
                    tag = f"transfer-{task}-{model}-{args.transfer_optimizer}-{input_size}"
                    cmd = _build_train_cmd(
                        args,
                        task,
                        model,
                        args.transfer_optimizer,
                        input_size,
                        include_synthetic=False,
                        evaluate_synthetic=False,
                    )
                    jobs.append(
                        Job(
                            tag=tag,
                            cmd=cmd,
                            task=task,
                            model=model,
                            optimizer=args.transfer_optimizer,
                            input_size=input_size,
                        )
                    )

    print(f"Starting real-data training jobs: {len(jobs)} (parallel={args.parallel})")
    results = _run_jobs(jobs, args.parallel, args.continue_on_error)

    real_run_dirs = [res.run_dir for res in results.values() if res.run_dir]

    synthetic_results: dict[str, RunResult] = {}
    if args.include_synthetic_train or args.evaluate_synthetic:
        has_manifest = _ensure_synthetic_manifests(args)
        if not has_manifest:
            print("Warning: synthetic manifests not available; skipping synthetic fine-tune.")
        else:
            if args.synthetic_finetune == "none":
                print("Synthetic fine-tune disabled via --synthetic-finetune=none.")
            else:
                allowed = set(args.transfer_models)
                if args.synthetic_finetune == "best":
                    best = _select_best_run(real_run_dirs, "binary", allowed)
                    if best is None:
                        print("Warning: no eligible real runs found for synthetic fine-tune.")
                    else:
                        cfg = best["config"]
                        run_dir = best["run_dir"]
                        epochs = args.synthetic_epochs or int(cfg.get("epochs", args.epochs))
                        lr = args.synthetic_lr or float(cfg.get("lr", 1e-3))
                        tag = f"synthetic-finetune-{cfg.get('model')}-{cfg.get('input_size')}"
                        cmd = _build_train_cmd(
                            args,
                            "binary",
                            cfg.get("model"),
                            cfg.get("optimizer"),
                            int(cfg.get("input_size", args.input_sizes[0])),
                            epochs=epochs,
                            lr=lr,
                            init_from=str(Path(run_dir) / "model.pt"),
                            include_synthetic=True,
                            evaluate_synthetic=args.evaluate_synthetic,
                            pretrained=bool(cfg.get("pretrained", True)),
                            freeze_backbone=bool(cfg.get("freeze_backbone", True)),
                        )
                        synthetic_results[tag] = _stream_process(cmd, tag)
                else:
                    for run_dir in real_run_dirs:
                        config_path = Path(run_dir) / "config.json"
                        if not config_path.exists():
                            continue
                        cfg = _load_json(config_path)
                        if cfg.get("task") != "binary":
                            continue
                        if cfg.get("dataset_split", "real") != "real":
                            continue
                        if cfg.get("model") not in allowed:
                            continue
                        epochs = args.synthetic_epochs or int(cfg.get("epochs", args.epochs))
                        lr = args.synthetic_lr or float(cfg.get("lr", 1e-3))
                        tag = f"synthetic-finetune-{cfg.get('model')}-{cfg.get('input_size')}-{Path(run_dir).name}"
                        cmd = _build_train_cmd(
                            args,
                            "binary",
                            cfg.get("model"),
                            cfg.get("optimizer"),
                            int(cfg.get("input_size", args.input_sizes[0])),
                            epochs=epochs,
                            lr=lr,
                            init_from=str(Path(run_dir) / "model.pt"),
                            include_synthetic=True,
                            evaluate_synthetic=args.evaluate_synthetic,
                            pretrained=bool(cfg.get("pretrained", True)),
                            freeze_backbone=bool(cfg.get("freeze_backbone", True)),
                        )
                        synthetic_results[tag] = _stream_process(cmd, tag)

    summary = {
        "real_results": {k: vars(v) for k, v in results.items()},
        "synthetic_results": {k: vars(v) for k, v in synthetic_results.items()},
        "jobs": [vars(job) for job in jobs],
    }
    args.save_summary.parent.mkdir(parents=True, exist_ok=True)
    args.save_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {args.save_summary}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Aggregate paper experiment runs into summary JSON/Markdown."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paper experiment results.")
    parser.add_argument("--runs-dir", type=Path, default=Path("training/runs"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("training/runs/paper_results_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/paper_results_summary.md"),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_metric(metrics: dict | None, task: str) -> float | None:
    if not metrics:
        return None
    if task == "binary":
        return metrics.get("f1") or metrics.get("accuracy")
    return metrics.get("macro_f1")


def main() -> None:
    args = parse_args()
    runs = []
    for run_dir in sorted(args.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        config = _load_json(config_path)
        if not str(config.get("approach", "")).startswith("paper"):
            continue
        metrics = _load_json(metrics_path)
        task = config.get("task", "")
        runs.append(
            {
                "run_dir": str(run_dir),
                "task": task,
                "model": config.get("model"),
                "input_size": config.get("input_size"),
                "dataset_split": config.get("dataset_split"),
                "optimizer": config.get("optimizer"),
                "val_metric": _extract_metric(metrics.get("val_metrics"), task),
                "test_metric": _extract_metric(metrics.get("test_metrics"), task),
                "synthetic_metric": _extract_metric(
                    metrics.get("synthetic_test_metrics"), task
                ),
            }
        )

    summary = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "runs": runs,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper Results Summary",
        "",
        "| Task | Model | Input Size | Dataset | Val Metric | Test Metric | Synthetic Metric |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        lines.append(
            "| {task} | {model} | {input_size} | {dataset_split} | {val_metric} | {test_metric} | {synthetic_metric} |".format(
                task=run["task"],
                model=run["model"],
                input_size=run["input_size"],
                dataset_split=run["dataset_split"],
                val_metric=run["val_metric"],
                test_metric=run["test_metric"],
                synthetic_metric=run["synthetic_metric"],
            )
        )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")

    print("Summary written to", args.output_json)
    print("Markdown written to", args.output_md)


if __name__ == "__main__":
    main()

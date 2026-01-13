#!/usr/bin/env python
"""Run rule-based strip analysis and report metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from training.features.rule_based import RuleBasedConfig, analyze_image
from training.preprocessing.labels import LabelConfig, normalize_labels, read_labels_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based strip analysis.")
    parser.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    parser.add_argument("--images-root", type=Path, default=Path("data/images/train"))
    parser.add_argument("--output", type=Path, default=Path("training/runs/rule_based_results.json"))
    parser.add_argument("--band-start", type=float, default=0.35)
    parser.add_argument("--band-end", type=float, default=0.95)
    parser.add_argument("--control-threshold", type=float, default=0.02)
    parser.add_argument("--test-threshold", type=float, default=0.005)
    parser.add_argument("--blur-sigma", type=float, default=3.0)
    parser.add_argument("--blue-row-threshold", type=float, default=0.2)
    parser.add_argument("--blue-search-ratio", type=float, default=0.6)
    parser.add_argument("--blue-min-run", type=int, default=10)
    parser.add_argument("--use-membrane-anchor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--membrane-window-ratio", type=float, default=0.08)
    parser.add_argument("--membrane-search-start", type=float, default=0.05)
    parser.add_argument("--membrane-search-end", type=float, default=0.6)
    parser.add_argument("--membrane-brightness-weight", type=float, default=0.5)
    parser.add_argument("--membrane-texture-weight", type=float, default=0.35)
    parser.add_argument("--membrane-blue-weight", type=float, default=0.15)
    parser.add_argument("--membrane-offset-ratio", type=float, default=0.03)
    parser.add_argument("--use-template-matching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--line-kernel", type=str, default="1,2,3,2,1")
    parser.add_argument("--control-window", type=str, default="0.15,0.45")
    parser.add_argument("--test1-window", type=str, default="0.45,0.65")
    parser.add_argument("--test2-window", type=str, default="0.65,0.90")
    parser.add_argument("--line-percentile", type=float, default=85.0)
    parser.add_argument("--background-percentile", type=float, default=50.0)
    parser.add_argument("--background-window-ratio", type=float, default=0.08)
    parser.add_argument("--template-spacing", type=str, default="0.16,0.16")
    parser.add_argument("--template-tolerance", type=float, default=0.10)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    metrics = {}
    per_label = {}
    for idx in range(y_true.shape[1]):
        per_label[f"test{idx+1}"] = {
            "precision": float(precision_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
            "recall": float(recall_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
            "f1": float(f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
        }
    metrics["per_label"] = per_label
    metrics["macro_f1"] = float(np.mean([m["f1"] for m in per_label.values()]))
    metrics["macro_precision"] = float(np.mean([m["precision"] for m in per_label.values()]))
    metrics["macro_recall"] = float(np.mean([m["recall"] for m in per_label.values()]))
    metrics["joint_accuracy"] = float((y_true == y_pred).all(axis=1).mean())
    return metrics


def main() -> None:
    args = parse_args()
    df = read_labels_csv(args.labels)
    df = normalize_labels(df)
    for col in (LabelConfig().test1_label_col, LabelConfig().test2_label_col):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    config = RuleBasedConfig(
        band_start_ratio=args.band_start,
        band_end_ratio=args.band_end,
        control_threshold=args.control_threshold,
        test_threshold=args.test_threshold,
        blur_sigma=args.blur_sigma,
        blue_row_threshold=args.blue_row_threshold,
        blue_search_ratio=args.blue_search_ratio,
        blue_min_run=args.blue_min_run,
        use_membrane_anchor=args.use_membrane_anchor,
        membrane_window_ratio=args.membrane_window_ratio,
        membrane_search_start_ratio=args.membrane_search_start,
        membrane_search_end_ratio=args.membrane_search_end,
        membrane_brightness_weight=args.membrane_brightness_weight,
        membrane_texture_weight=args.membrane_texture_weight,
        membrane_blue_weight=args.membrane_blue_weight,
        membrane_offset_ratio=args.membrane_offset_ratio,
        use_template_matching=args.use_template_matching,
        line_kernel=tuple(float(x) for x in args.line_kernel.split(",") if x),
        control_window=tuple(float(x) for x in args.control_window.split(",") if x),
        test1_window=tuple(float(x) for x in args.test1_window.split(",") if x),
        test2_window=tuple(float(x) for x in args.test2_window.split(",") if x),
        line_percentile=args.line_percentile,
        background_percentile=args.background_percentile,
        background_window_ratio=args.background_window_ratio,
        template_spacing=tuple(float(x) for x in args.template_spacing.split(",") if x),
        template_tolerance=args.template_tolerance,
    )

    preds = []
    truths = []
    details = []
    for _, row in df.iterrows():
        path = args.images_root / row[LabelConfig().image_path_col]
        result, pred = analyze_image(path, config)
        preds.append(pred)
        truths.append((row[LabelConfig().test1_label_col], row[LabelConfig().test2_label_col]))
        details.append({"image_path": row[LabelConfig().image_path_col], **result.to_dict()})

    y_true = np.asarray(truths, dtype=int)
    y_pred = np.asarray(preds, dtype=int)
    metrics = compute_metrics(y_true, y_pred)

    output = {
        "metrics": metrics,
        "config": {
            "band_start": args.band_start,
            "band_end": args.band_end,
            "control_threshold": args.control_threshold,
            "test_threshold": args.test_threshold,
            "blur_sigma": args.blur_sigma,
            "blue_row_threshold": args.blue_row_threshold,
            "blue_search_ratio": args.blue_search_ratio,
            "blue_min_run": args.blue_min_run,
            "use_membrane_anchor": args.use_membrane_anchor,
            "membrane_window_ratio": args.membrane_window_ratio,
            "membrane_search_start": args.membrane_search_start,
            "membrane_search_end": args.membrane_search_end,
            "membrane_brightness_weight": args.membrane_brightness_weight,
            "membrane_texture_weight": args.membrane_texture_weight,
            "membrane_blue_weight": args.membrane_blue_weight,
            "membrane_offset_ratio": args.membrane_offset_ratio,
            "use_template_matching": args.use_template_matching,
            "line_kernel": args.line_kernel,
            "control_window": args.control_window,
            "test1_window": args.test1_window,
            "test2_window": args.test2_window,
            "line_percentile": args.line_percentile,
            "background_percentile": args.background_percentile,
            "background_window_ratio": args.background_window_ratio,
            "template_spacing": args.template_spacing,
            "template_tolerance": args.template_tolerance,
        },
        "details": details,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

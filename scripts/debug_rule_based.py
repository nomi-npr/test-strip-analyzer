#!/usr/bin/env python
"""Generate debug visuals for rule-based strip analysis."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.features import rule_based
from training.preprocessing.labels import LabelConfig, normalize_labels, read_labels_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug rule-based strip analysis.")
    parser.add_argument("--split-csv", type=Path, default=Path("data/splits/val.csv"))
    parser.add_argument("--images-root", type=Path, default=Path("data/processed/inspry/val"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/runs/rule_based_debug"))
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--select",
        choices=["any", "positive", "negative", "mixed"],
        default="mixed",
        help="Select sample types based on labels.",
    )
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
    return parser.parse_args()


def _overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    color = np.zeros_like(image)
    color[:, :, 1] = 255
    overlay[mask == 0] = (overlay[mask == 0] * 0.2).astype(np.uint8)
    overlay[mask > 0] = (0.7 * overlay[mask > 0] + 0.3 * color[mask > 0]).astype(np.uint8)
    return overlay


def _annotate_rectified(
    image: np.ndarray,
    config: rule_based.RuleBasedConfig,
    result: rule_based.LineResult,
) -> np.ndarray:
    annotated = image.copy()
    start = result.band_start
    end = result.band_end
    if result.membrane_start is not None and result.membrane_end is not None:
        cv2.line(
            annotated,
            (0, result.membrane_start),
            (annotated.shape[1], result.membrane_start),
            (0, 255, 255),
            2,
        )
        cv2.line(
            annotated,
            (0, result.membrane_end),
            (annotated.shape[1], result.membrane_end),
            (0, 255, 255),
            2,
        )
    cv2.line(annotated, (0, start), (annotated.shape[1], start), (255, 255, 0), 2)
    cv2.line(annotated, (0, end), (annotated.shape[1], end), (255, 255, 0), 2)

    for row, color, label in [
        (result.control_row, (0, 255, 0), "C"),
        (result.test1_row, (255, 0, 0), "T1"),
        (result.test2_row, (255, 0, 255), "T2"),
    ]:
        if row is None:
            continue
        cv2.line(annotated, (0, row), (annotated.shape[1], row), color, 2)
        cv2.putText(
            annotated,
            label,
            (5, max(10, row - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return annotated


def _save_profile(
    profile: np.ndarray, result: rule_based.LineResult, out_path: Path
) -> None:
    height = len(profile)
    start = result.band_start
    end = result.band_end
    plt.figure(figsize=(4, 6))
    plt.plot(profile, np.arange(height))
    plt.gca().invert_yaxis()
    plt.axhline(start, color="yellow", linestyle="--")
    plt.axhline(end, color="yellow", linestyle="--")
    if result.membrane_start is not None and result.membrane_end is not None:
        plt.axhline(result.membrane_start, color="cyan", linestyle="--")
        plt.axhline(result.membrane_end, color="cyan", linestyle="--")
    plt.xlabel("Line score")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    for col in (LabelConfig().test1_label_col, LabelConfig().test2_label_col):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if args.select == "any":
        return df.sample(n=min(args.max_samples, len(df)), random_state=args.seed)

    positives = df[(df[LabelConfig().test1_label_col] == 1) | (df[LabelConfig().test2_label_col] == 1)]
    negatives = df[(df[LabelConfig().test1_label_col] == 0) & (df[LabelConfig().test2_label_col] == 0)]

    if args.select == "positive":
        return positives.sample(n=min(args.max_samples, len(positives)), random_state=args.seed)
    if args.select == "negative":
        return negatives.sample(n=min(args.max_samples, len(negatives)), random_state=args.seed)

    # mixed
    rng = random.Random(args.seed)
    indices = []
    for subset in [positives, negatives]:
        if subset.empty:
            continue
        idx = subset.index[rng.randrange(len(subset))]
        indices.append(idx)
    remaining = args.max_samples - len(indices)
    if remaining > 0:
        rest = df.drop(index=indices, errors="ignore") if indices else df
        extra = rest.sample(n=min(remaining, len(rest)), random_state=args.seed).index.tolist()
        indices.extend(extra)
    return df.loc[indices]


def main() -> None:
    args = parse_args()
    df = read_labels_csv(args.split_csv)
    df = normalize_labels(df)
    df = _select_rows(df, args)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    config = rule_based.RuleBasedConfig(
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

    for _, row in df.iterrows():
        rel_path = row[LabelConfig().image_path_col]
        image_path = args.images_root / rel_path
        stem = Path(rel_path).stem

        image = rule_based._load_image(image_path)
        mask = rule_based.segment_strip_mask(image)
        rect_img, rect_mask = rule_based.rectify_strip(image, mask, config)
        profile = rule_based._row_profile(rect_img, rect_mask, config)
        result = rule_based.analyze_strip(rect_img, rect_mask, config)
        prediction = rule_based.predict_from_strengths(result, config)

        sample_dir = output_dir / stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(sample_dir / "01_original.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            str(sample_dir / "02_mask_overlay.png"),
            cv2.cvtColor(_overlay_mask(image, mask), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(sample_dir / "03_rectified.png"), cv2.cvtColor(rect_img, cv2.COLOR_RGB2BGR))
        annotated = _annotate_rectified(rect_img, config, result)
        cv2.imwrite(
            str(sample_dir / "04_rectified_annotated.png"),
            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
        )
        _save_profile(profile, result, sample_dir / "05_profile.png")

        meta = {
            "image_path": rel_path,
            "label": {
                "test1": int(row[LabelConfig().test1_label_col]),
                "test2": int(row[LabelConfig().test2_label_col]),
            },
            "prediction": {"test1": int(prediction[0]), "test2": int(prediction[1])},
            **result.to_dict(),
        }
        (sample_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        print(f"Saved debug for {rel_path} -> {sample_dir}")


if __name__ == "__main__":
    main()

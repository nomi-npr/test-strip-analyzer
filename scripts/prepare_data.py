#!/usr/bin/env python
"""Audit dataset integrity, normalize labels, and create stratified splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.preprocessing import (
    audit_dataset,
    infer_delimiter,
    read_labels_csv,
    write_audit_report,
)
from training.preprocessing.labels import LabelConfig, normalize_labels, write_manifest
from training.preprocessing.splits import (
    create_split_symlinks,
    create_stratified_splits,
    write_split_config,
    write_split_csvs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit data and create stratified splits.")
    parser.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    parser.add_argument("--images-root", type=Path, default=Path("original_images"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--images-out", type=Path, default=Path("data/images"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delimiter", type=str, default=None)
    parser.add_argument("--detect-near-duplicates", action="store_true")
    parser.add_argument("--near-dup-threshold", type=int, default=5)
    parser.add_argument("--max-dup-images", type=int, default=2000)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--clean-manifest",
        type=Path,
        default=Path("data/labels_clean.csv"),
        help="Where to write normalized labels CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.delimiter is None:
        args.delimiter = infer_delimiter(args.labels)

    report = audit_dataset(
        labels_path=args.labels,
        images_root=args.images_root,
        delimiter=args.delimiter,
        detect_near_duplicates=args.detect_near_duplicates,
        near_duplicate_threshold=args.near_dup_threshold,
        max_duplicate_images=args.max_dup_images,
        seed=args.seed,
    )
    write_audit_report(report, args.splits_dir / "audit_report.json")

    df = read_labels_csv(args.labels, delimiter=args.delimiter)
    df = normalize_labels(df)
    write_manifest(
        df,
        args.clean_manifest,
        delimiter=",",
        columns=[
            LabelConfig().image_path_col,
            LabelConfig().test1_label_col,
            LabelConfig().test2_label_col,
        ],
    )

    if args.audit_only:
        print(json.dumps(report.to_dict(), indent=2))
        return

    splits = create_stratified_splits(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    write_split_config(
        args.splits_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        total_rows=report.total_rows,
    )
    write_split_csvs(splits, args.splits_dir, delimiter=",")
    create_split_symlinks(
        splits,
        images_root=args.images_root,
        output_root=args.images_out,
        copy_files=args.copy_images,
    )

    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()

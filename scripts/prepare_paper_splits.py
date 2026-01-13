#!/usr/bin/env python
"""Create paper-aligned split CSVs for binary and multilabel tasks."""
from __future__ import annotations

import argparse
from pathlib import Path

from training.preprocessing import (
    build_binary_label,
    create_stratified_splits,
    infer_delimiter,
    normalize_labels,
    read_labels_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paper split CSVs.")
    parser.add_argument(
        "--strategy",
        choices=["existing", "fresh"],
        default="existing",
        help="Use existing train/val/test or create fresh splits.",
    )
    parser.add_argument("--labels", type=Path, default=Path("data/labels_clean.csv"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_existing(splits_dir: Path) -> dict[str, object]:
    splits = {}
    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}.csv"
        if path.exists():
            splits[split] = read_labels_csv(path)
    return splits


def _write_multilabel(df, out_path: Path) -> None:
    df = normalize_labels(df)
    df[["image_path", "test1_label", "test2_label"]].to_csv(out_path, index=False)


def _write_binary(df, out_path: Path) -> None:
    df = build_binary_label(df)
    df[["image_path", "binary_label"]].to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    args.splits_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, object]
    if args.strategy == "existing":
        splits = _load_existing(args.splits_dir)
        if not splits:
            raise FileNotFoundError("No existing splits found to reuse.")
    else:
        delimiter = infer_delimiter(args.labels)
        df = read_labels_csv(args.labels, delimiter=delimiter)
        df = normalize_labels(df)
        splits = create_stratified_splits(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    for split_name, df in splits.items():
        multilabel_path = args.splits_dir / f"{split_name}_multilabel.csv"
        binary_path = args.splits_dir / f"{split_name}_binary.csv"
        _write_multilabel(df, multilabel_path)
        _write_binary(df, binary_path)

    print("Paper split CSVs written to", args.splits_dir)


if __name__ == "__main__":
    main()

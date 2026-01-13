#!/usr/bin/env python
"""Create stratified k-fold splits for cross-validation."""
from __future__ import annotations

import argparse
from pathlib import Path

from training.preprocessing import (
    build_binary_label,
    create_stratified_kfolds,
    infer_delimiter,
    normalize_labels,
    read_labels_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create k-fold split CSVs.")
    parser.add_argument("--labels", type=Path, default=Path("data/labels_clean.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits/folds"))
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_multilabel(df, path: Path) -> None:
    df = normalize_labels(df)
    df[["image_path", "test1_label", "test2_label"]].to_csv(path, index=False)


def _write_binary(df, path: Path) -> None:
    df = build_binary_label(df)
    df[["image_path", "binary_label"]].to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    delimiter = infer_delimiter(args.labels)
    df = read_labels_csv(args.labels, delimiter=delimiter)
    df = normalize_labels(df)

    folds = create_stratified_kfolds(df, n_splits=args.folds, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for idx, split in enumerate(folds):
        fold_dir = args.output_dir / f"fold_{idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_df = split["train"]
        val_df = split["val"]

        _write_multilabel(train_df, fold_dir / "train_multilabel.csv")
        _write_multilabel(val_df, fold_dir / "val_multilabel.csv")
        _write_binary(train_df, fold_dir / "train_binary.csv")
        _write_binary(val_df, fold_dir / "val_binary.csv")

    print(f"Wrote {len(folds)} folds to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build synthetic train/test manifests from StyleGAN2-ADA outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic dataset manifests.")
    parser.add_argument("--synthetic-root", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--positive-dir", type=str, default="positive")
    parser.add_argument("--negative-dir", type=str, default="negative")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _collect_images(base_root: Path, root: Path, label: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            records.append({"image_path": str(path.relative_to(base_root)), "binary_label": label})
    return records


def main() -> None:
    args = parse_args()
    pos_root = args.synthetic_root / args.positive_dir
    neg_root = args.synthetic_root / args.negative_dir
    if not pos_root.exists() or not neg_root.exists():
        raise FileNotFoundError("Synthetic positive/negative directories not found.")

    records = _collect_images(args.synthetic_root, pos_root, 1) + _collect_images(
        args.synthetic_root, neg_root, 0
    )
    df = pd.DataFrame.from_records(records)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    train_count = int(round(len(df) * args.train_ratio))
    train_df = df.iloc[:train_count].copy()
    test_df = df.iloc[train_count:].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "synthetic_train.csv"
    test_path = args.output_dir / "synthetic_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Wrote", train_path)
    print("Wrote", test_path)


if __name__ == "__main__":
    main()

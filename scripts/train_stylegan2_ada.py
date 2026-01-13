#!/usr/bin/env python
"""Prepare class-specific datasets and optionally launch StyleGAN2-ADA training."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from training.preprocessing import build_binary_label, infer_delimiter, read_labels_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StyleGAN2-ADA on a single class.")
    parser.add_argument(
        "--labels-csv", type=Path, default=Path("data/splits/train_multilabel.csv")
    )
    parser.add_argument("--images-root", type=Path, default=Path("data/images/train"))
    parser.add_argument("--class-label", choices=["positive", "negative"], required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/synthetic/stylegan_datasets"),
        help="Where to build the class-specific dataset.",
    )
    parser.add_argument(
        "--stylegan-root",
        type=Path,
        default=None,
        help="Path to StyleGAN2-ADA repo (or set STYLEGAN2_ADA_ROOT).",
    )
    parser.add_argument("--outdir", type=Path, default=Path("training/gan_runs"))
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cfg", type=str, default="auto")
    parser.add_argument("--kimg", type=int, default=1000)
    parser.add_argument("--execute", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    delimiter = infer_delimiter(args.labels_csv)
    df = read_labels_csv(args.labels_csv, delimiter=delimiter)
    df = build_binary_label(df)

    target_value = 1 if args.class_label == "positive" else 0
    subset = df[df["binary_label"].astype(int) == target_value]
    dataset_dir = args.dataset_dir / args.class_label
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for rel_path in subset["image_path"].tolist():
        src = args.images_root / rel_path
        if not src.exists():
            continue
        dst = dataset_dir / src.name
        _symlink(src, dst)

    stylegan_root = args.stylegan_root or os.environ.get("STYLEGAN2_ADA_ROOT")
    if stylegan_root is None:
        raise RuntimeError("StyleGAN2-ADA root not provided. Set --stylegan-root or STYLEGAN2_ADA_ROOT.")

    train_cmd = [
        "python",
        "train.py",
        "--outdir",
        str(args.outdir),
        "--data",
        str(dataset_dir),
        "--gpus",
        str(args.gpus),
        "--cfg",
        args.cfg,
        "--kimg",
        str(args.kimg),
    ]

    print("Prepared dataset:", dataset_dir)
    print("Training command:", " ".join(train_cmd))
    if args.execute:
        subprocess.run(train_cmd, cwd=stylegan_root, check=True)


if __name__ == "__main__":
    main()

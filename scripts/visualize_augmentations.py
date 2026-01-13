#!/usr/bin/env python
"""Visualize augmented samples for quick sanity checks."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid

from training.dataset import (
    StripDataset,
    TransformConfig,
    build_positive_transforms,
    build_train_transforms,
    denormalize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a grid of augmented samples.")
    parser.add_argument(
        "--split-csv", type=Path, default=Path("data/splits/train_multilabel.csv")
    )
    parser.add_argument("--images-root", type=Path, default=Path("data/images/train"))
    parser.add_argument("--output", type=Path, default=Path("data/augmentations/train_grid.png"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--normalization", choices=["minmax", "imagenet"], default="minmax")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TransformConfig(input_size=args.input_size, normalization=args.normalization)
    train_tf = build_train_transforms(config)
    positive_tf = build_positive_transforms(config)

    dataset = StripDataset(
        split_csv=args.split_csv,
        images_root=args.images_root,
        transform=train_tf,
        positive_transform=positive_tf,
    )

    rng = random.Random(args.seed)
    indices = [rng.randrange(len(dataset)) for _ in range(args.num_samples)]
    images = []
    for idx in indices:
        image, _ = dataset[idx]
        images.append(image)

    grid = make_grid(images, nrow=int(args.num_samples ** 0.5))
    if config.normalization == "imagenet":
        grid = denormalize(grid, config.mean, config.std).clamp(0, 1)
    grid = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    output_image = Image.fromarray(grid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(args.output)
    print(f"Saved augmentation grid to {args.output}")


if __name__ == "__main__":
    main()

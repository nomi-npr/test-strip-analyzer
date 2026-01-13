#!/usr/bin/env python
"""Generate synthetic images from a StyleGAN2-ADA network."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate StyleGAN2-ADA samples.")
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stylegan-root",
        type=Path,
        default=None,
        help="Path to StyleGAN2-ADA repo (or set STYLEGAN2_ADA_ROOT).",
    )
    parser.add_argument("--execute", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stylegan_root = args.stylegan_root or os.environ.get("STYLEGAN2_ADA_ROOT")
    if stylegan_root is None:
        raise RuntimeError("StyleGAN2-ADA root not provided. Set --stylegan-root or STYLEGAN2_ADA_ROOT.")

    seeds = list(range(args.seed, args.seed + args.num_images))
    seed_arg = ",".join(str(seed) for seed in seeds)

    cmd = [
        "python",
        "generate.py",
        "--network",
        str(args.network),
        "--outdir",
        str(args.outdir),
        "--seeds",
        seed_arg,
    ]

    print("Generate command:", " ".join(cmd))
    if args.execute:
        subprocess.run(cmd, cwd=stylegan_root, check=True)


if __name__ == "__main__":
    main()

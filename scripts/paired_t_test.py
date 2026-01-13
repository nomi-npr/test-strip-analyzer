#!/usr/bin/env python
"""Run a paired t-test on two metric lists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.evaluation import paired_t_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired t-test for CV metrics.")
    parser.add_argument("--values-a", type=str, default=None)
    parser.add_argument("--values-b", type=str, default=None)
    parser.add_argument("--file-a", type=Path, default=None)
    parser.add_argument("--file-b", type=Path, default=None)
    return parser.parse_args()


def _parse_values(raw: str | None) -> list[float]:
    if raw is None:
        return []
    return [float(v.strip()) for v in raw.split(",") if v.strip()]


def _parse_file(path: Path | None) -> list[float]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected JSON array of floats")
    return [float(v) for v in data]


def main() -> None:
    args = parse_args()
    values_a = _parse_values(args.values_a) or _parse_file(args.file_a)
    values_b = _parse_values(args.values_b) or _parse_file(args.file_b)
    if not values_a or not values_b:
        raise ValueError("Provide values via --values-a/--values-b or --file-a/--file-b")

    result = paired_t_test(values_a, values_b)
    print(
        "t-stat={:.4f} p-value={} mean-diff={:.4f} n={}".format(
            result.t_stat,
            "{:.4g}".format(result.p_value) if result.p_value is not None else "NA",
            result.mean_diff,
            result.n,
        )
    )


if __name__ == "__main__":
    main()

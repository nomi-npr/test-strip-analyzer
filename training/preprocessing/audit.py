"""Dataset audit and duplicate detection utilities."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from training.preprocessing.labels import LabelConfig, build_joint_label, normalize_labels, read_labels_csv


@dataclass
class AuditReport:
    total_rows: int
    missing_files: list[str]
    positive_rates: dict[str, float]
    joint_counts: dict[str, int]
    duplicate_pairs: list[tuple[str, str, int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized)
    diff = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find_near_duplicates(
    image_paths: Iterable[Path],
    hash_size: int = 8,
    threshold: int = 5,
    max_images: int | None = 2000,
    seed: int = 42,
) -> list[tuple[str, str, int]]:
    paths = list(image_paths)
    if max_images is not None and len(paths) > max_images:
        rng = random.Random(seed)
        paths = rng.sample(paths, k=max_images)
    hashes: list[tuple[Path, int]] = []
    for path in paths:
        try:
            with Image.open(path) as img:
                hashes.append((path, _compute_dhash(img, hash_size=hash_size)))
        except Exception:
            continue
    duplicates: list[tuple[str, str, int]] = []
    for i, (path_a, hash_a) in enumerate(hashes):
        for path_b, hash_b in hashes[i + 1 :]:
            distance = _hamming_distance(hash_a, hash_b)
            if distance <= threshold:
                duplicates.append((str(path_a), str(path_b), distance))
    return duplicates


def audit_dataset(
    labels_path: Path,
    images_root: Path,
    delimiter: str | None = None,
    detect_near_duplicates: bool = False,
    near_duplicate_threshold: int = 5,
    max_duplicate_images: int | None = 2000,
    seed: int = 42,
) -> AuditReport:
    df = read_labels_csv(labels_path, delimiter=delimiter)
    df = normalize_labels(df)
    df = build_joint_label(df)

    missing_files: list[str] = []
    for rel_path in df[LabelConfig().image_path_col].tolist():
        full_path = images_root / rel_path
        if not full_path.exists():
            missing_files.append(rel_path)

    total_rows = len(df)
    test1_rate = float(df[LabelConfig().test1_label_col].mean())
    test2_rate = float(df[LabelConfig().test2_label_col].mean())
    positive_rates = {
        "test1_positive_rate": test1_rate,
        "test2_positive_rate": test2_rate,
    }

    joint_counts_series = df["joint_label"].value_counts().sort_index()
    joint_counts = {str(k): int(v) for k, v in joint_counts_series.to_dict().items()}

    duplicate_pairs: list[tuple[str, str, int]] = []
    if detect_near_duplicates:
        image_paths = [images_root / p for p in df[LabelConfig().image_path_col].tolist()]
        duplicate_pairs = find_near_duplicates(
            image_paths,
            threshold=near_duplicate_threshold,
            max_images=max_duplicate_images,
            seed=seed,
        )

    return AuditReport(
        total_rows=total_rows,
        missing_files=missing_files,
        positive_rates=positive_rates,
        joint_counts=joint_counts,
        duplicate_pairs=duplicate_pairs,
    )


def write_audit_report(report: AuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)

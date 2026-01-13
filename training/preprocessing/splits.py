"""Stratified split utilities and symlink creation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from training.preprocessing.labels import LabelConfig, build_joint_label, normalize_labels


@dataclass
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    created_at: str
    total_rows: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _allocate_counts(
    count: int, train_ratio: float, val_ratio: float, test_ratio: float
) -> tuple[int, int, int]:
    if count <= 0:
        return 0, 0, 0
    if count == 1:
        return 1, 0, 0
    if count == 2:
        return 1, 1, 0

    train_count = int(round(train_ratio * count))
    val_count = int(round(val_ratio * count))
    test_count = count - train_count - val_count

    # Ensure at least one sample per split when possible.
    if val_count == 0:
        val_count = 1
        train_count = max(train_count - 1, 1)
    if test_count == 0:
        test_count = 1
        train_count = max(train_count - 1, 1)

    # Fix any over-allocation due to rounding.
    while train_count + val_count + test_count > count:
        if train_count >= val_count and train_count >= test_count and train_count > 1:
            train_count -= 1
        elif val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break

    # Ensure at least one training sample.
    if train_count == 0:
        train_count = 1
        if val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1

    return train_count, val_count, test_count


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    df = normalize_labels(df)
    df = build_joint_label(df)

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for label, group in df.groupby("joint_label"):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        train_count, val_count, test_count = _allocate_counts(
            len(indices), train_ratio, val_ratio, test_ratio
        )
        train_indices.extend(indices[:train_count])
        val_indices.extend(indices[train_count : train_count + val_count])
        test_indices.extend(indices[train_count + val_count : train_count + val_count + test_count])

    train_df = df.loc[train_indices].sample(frac=1, random_state=seed)
    val_df = df.loc[val_indices].sample(frac=1, random_state=seed + 1)
    test_df = df.loc[test_indices].sample(frac=1, random_state=seed + 2)

    return {"train": train_df.reset_index(drop=True), "val": val_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}


def create_stratified_kfolds(
    df: pd.DataFrame,
    n_splits: int = 10,
    seed: int = 42,
) -> list[dict[str, pd.DataFrame]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    df = normalize_labels(df)
    df = build_joint_label(df)
    labels = df["joint_label"].astype(str).to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[dict[str, pd.DataFrame]] = []
    for train_idx, val_idx in skf.split(df, labels):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        folds.append({"train": train_df, "val": val_df})
    return folds


def write_split_config(
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    total_rows: int,
) -> None:
    config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        total_rows=total_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)


def write_split_csvs(
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
    config: LabelConfig | None = None,
    delimiter: str = ",",
) -> None:
    config = config or LabelConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, df in splits.items():
        df = normalize_labels(df, config)
        df = df[[config.image_path_col, config.test1_label_col, config.test2_label_col]]
        df.to_csv(output_dir / f"{split_name}.csv", index=False, sep=delimiter)


def _link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if dst.is_symlink():
        if dst.exists():
            return
        dst.unlink()
    elif dst.exists():
        return
    if copy_files:
        dst.write_bytes(src.read_bytes())
    else:
        dst.symlink_to(src.resolve())


def create_split_symlinks(
    splits: dict[str, pd.DataFrame],
    images_root: Path,
    output_root: Path,
    copy_files: bool = False,
    config: LabelConfig | None = None,
) -> None:
    config = config or LabelConfig()
    for split_name, df in splits.items():
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for rel_path in df[config.image_path_col].tolist():
            src = images_root / rel_path
            dst = split_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(src, dst, copy_files=copy_files)

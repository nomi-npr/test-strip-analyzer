"""Label IO and normalization utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

POSITIVE_VALUES = {"positive", "pos", "p", "1", "true", "t", "yes", "y"}
NEGATIVE_VALUES = {"negative", "neg", "n", "0", "false", "f", "no"}


@dataclass(frozen=True)
class LabelConfig:
    image_path_col: str = "image_path"
    test1_col: str = "test1_result"
    test2_col: str = "test2_result"
    test1_label_col: str = "test1_label"
    test2_label_col: str = "test2_label"
    binary_label_col: str = "binary_label"


def infer_delimiter(path: Path) -> str:
    header = path.read_text(encoding="utf-8").splitlines()[0]
    comma_count = header.count(",")
    semicolon_count = header.count(";")
    if semicolon_count > comma_count:
        return ";"
    return ","


def read_labels_csv(path: Path, delimiter: str | None = None) -> pd.DataFrame:
    if delimiter is None:
        delimiter = infer_delimiter(path)
    return pd.read_csv(path, sep=delimiter, dtype=str)


def _normalize_value(value: object) -> int:
    if value is None:
        raise ValueError("Label value is missing")
    raw = str(value).strip().lower()
    if raw in POSITIVE_VALUES:
        return 1
    if raw in NEGATIVE_VALUES:
        return 0
    if raw.isdigit():
        return 1 if int(raw) != 0 else 0
    raise ValueError(f"Unknown label value: {value}")


def normalize_labels(
    df: pd.DataFrame, config: LabelConfig | None = None
) -> pd.DataFrame:
    config = config or LabelConfig()
    df = df.copy()
    if config.test1_label_col in df.columns and config.test2_label_col in df.columns:
        df[config.test1_label_col] = df[config.test1_label_col].map(_normalize_value)
        df[config.test2_label_col] = df[config.test2_label_col].map(_normalize_value)
        return df
    df[config.test1_label_col] = df[config.test1_col].map(_normalize_value)
    df[config.test2_label_col] = df[config.test2_col].map(_normalize_value)
    return df


def build_binary_label(
    df: pd.DataFrame, config: LabelConfig | None = None, column_name: str | None = None
) -> pd.DataFrame:
    config = config or LabelConfig()
    column_name = column_name or config.binary_label_col
    if column_name in df.columns:
        return df
    df = normalize_labels(df, config)
    df = df.copy()
    df[column_name] = (
        (df[config.test1_label_col].astype(int) == 1)
        | (df[config.test2_label_col].astype(int) == 1)
    ).astype(int)
    return df


def build_joint_label(
    df: pd.DataFrame, config: LabelConfig | None = None, column_name: str = "joint_label"
) -> pd.DataFrame:
    config = config or LabelConfig()
    df = df.copy()
    df[column_name] = (
        df[config.test1_label_col].astype(str) + df[config.test2_label_col].astype(str)
    )
    return df


def write_manifest(
    df: pd.DataFrame, path: Path, delimiter: str = ",", columns: Iterable[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        df = df[list(columns)]
    df.to_csv(path, index=False, sep=delimiter)

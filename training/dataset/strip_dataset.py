"""Dataset definitions for strip images."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.preprocessing.labels import (
    LabelConfig,
    build_binary_label,
    infer_delimiter,
    normalize_labels,
    read_labels_csv,
)


class StripDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        images_root: Path,
        transform: Callable | None = None,
        positive_transform: Callable | None = None,
        return_path: bool = False,
        delimiter: str | None = None,
        config: LabelConfig | None = None,
        image_mode: str = "RGB",
        label_mode: str = "multilabel",
        label_columns: tuple[str, ...] | None = None,
        label_dtype: torch.dtype | None = None,
        skip_missing: bool = False,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.images_root = Path(images_root)
        self.transform = transform
        self.positive_transform = positive_transform
        self.return_path = return_path
        self.image_mode = image_mode
        self.config = config or LabelConfig()

        if delimiter is None:
            delimiter = infer_delimiter(self.split_csv)

        df = read_labels_csv(self.split_csv, delimiter=delimiter)
        self.label_mode = label_mode
        if label_mode == "binary":
            df = build_binary_label(df, self.config)
            if label_columns is None:
                label_columns = (self.config.binary_label_col,)
            self.labels = df[list(label_columns)].astype(int).to_numpy().reshape(-1)
            self.label_dtype = label_dtype or torch.long
        else:
            if self.config.test1_label_col not in df.columns:
                df = normalize_labels(df, self.config)
            if label_columns is None:
                label_columns = (self.config.test1_label_col, self.config.test2_label_col)
            self.labels = df[list(label_columns)].astype(float).to_numpy()
            self.label_dtype = label_dtype or torch.float32
        self.label_columns = label_columns
        self.df = df
        self.image_paths = df[self.config.image_path_col].tolist()
        self.missing_paths: list[Path] = []
        if skip_missing:
            self._filter_missing()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        rel_path = self.image_paths[idx]
        image_path = self.images_root / rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        with Image.open(image_path) as img:
            image = img.convert(self.image_mode)
        label = torch.tensor(self.labels[idx], dtype=self.label_dtype)

        if self.positive_transform is not None and label.any():
            image = self.positive_transform(image)
        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return image, label, str(image_path)
        return image, label

    def get_labels(self) -> np.ndarray:
        return np.asarray(self.labels)

    def _filter_missing(self) -> None:
        keep_mask = []
        labels = np.asarray(self.labels)
        for rel_path in self.image_paths:
            image_path = self.images_root / rel_path
            exists = image_path.exists()
            keep_mask.append(exists)
            if not exists:
                self.missing_paths.append(image_path)
        if not keep_mask:
            return
        mask = np.asarray(keep_mask, dtype=bool)
        if not mask.all():
            self.image_paths = list(np.asarray(self.image_paths)[mask])
            self.labels = labels[mask]

    def get_missing_paths(self) -> list[Path]:
        return list(self.missing_paths)

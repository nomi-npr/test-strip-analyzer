from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from training.preprocessing import audit_dataset, create_stratified_splits
from training.preprocessing.labels import infer_delimiter, normalize_labels


def _make_image(path: Path) -> None:
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    img.save(path)


def test_audit_and_split(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    images_root.mkdir()

    records = []
    for idx in range(40):
        label_combo = idx % 4
        test1 = "positive" if label_combo in (2, 3) else "negative"
        test2 = "positive" if label_combo in (1, 3) else "negative"
        filename = f"img_{idx}.jpg"
        _make_image(images_root / filename)
        records.append({"image_path": filename, "test1_result": test1, "test2_result": test2})

    labels_path = tmp_path / "labels.csv"
    pd.DataFrame.from_records(records).to_csv(labels_path, sep=";", index=False)

    assert infer_delimiter(labels_path) == ";"

    report = audit_dataset(labels_path, images_root, delimiter=";")
    assert report.total_rows == 40
    assert report.missing_files == []

    df = pd.read_csv(labels_path, sep=";")
    df = normalize_labels(df)
    splits = create_stratified_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert sum(len(split) for split in splits.values()) == 40

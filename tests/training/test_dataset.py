from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.dataset import StripDataset


def _make_image(path: Path) -> None:
    img = Image.new("RGB", (10, 10), color=(0, 255, 0))
    img.save(path)


def test_strip_dataset_loading(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    images_root = tmp_path / "images"
    images_root.mkdir()
    image_path = images_root / "sample.jpg"
    _make_image(image_path)

    csv_path = tmp_path / "split.csv"
    csv_path.write_text(
        "image_path,test1_label,test2_label\n" "sample.jpg,1,0\n", encoding="utf-8"
    )

    from torchvision import transforms

    dataset = StripDataset(
        split_csv=csv_path,
        images_root=images_root,
        transform=transforms.ToTensor(),
    )
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert label.shape == (2,)


def test_strip_dataset_labels(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    images_root.mkdir()
    for idx in range(3):
        _make_image(images_root / f"sample_{idx}.jpg")

    csv_path = tmp_path / "split.csv"
    csv_path.write_text(
        "image_path,test1_label,test2_label\n"
        "sample_0.jpg,1,0\n"
        "sample_1.jpg,0,1\n"
        "sample_2.jpg,0,0\n",
        encoding="utf-8",
    )

    dataset = StripDataset(split_csv=csv_path, images_root=images_root)
    labels = dataset.get_labels()
    assert np.array_equal(labels, np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))


def test_strip_dataset_binary_labels(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    images_root.mkdir()
    _make_image(images_root / "sample.jpg")

    csv_path = tmp_path / "split.csv"
    csv_path.write_text(
        "image_path,test1_label,test2_label\n" "sample.jpg,1,0\n",
        encoding="utf-8",
    )

    dataset = StripDataset(split_csv=csv_path, images_root=images_root, label_mode=\"binary\")
    _, label = dataset[0]
    assert int(label) == 1

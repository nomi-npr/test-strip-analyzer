"""Torchvision transforms for training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True)
class TransformConfig:
    input_size: int = 224
    normalization: str = "minmax"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


def _resize(config: TransformConfig) -> transforms.Resize:
    return transforms.Resize(
        (config.input_size, config.input_size),
        interpolation=InterpolationMode.BILINEAR,
    )


def _translate_fraction(config: TransformConfig, pixels: int = 10) -> tuple[float, float]:
    frac = min(max(pixels / float(config.input_size), 0.0), 0.5)
    return (frac, frac)


def _maybe_normalize(config: TransformConfig) -> transforms.Normalize | None:
    if config.normalization == "imagenet":
        return transforms.Normalize(mean=config.mean, std=config.std)
    return None


def build_train_transforms(config: TransformConfig) -> transforms.Compose:
    transforms_list: list[transforms.Transform] = [
        _resize(config),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(
            degrees=0,
            translate=_translate_fraction(config, pixels=10),
        ),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
    ]
    normalize = _maybe_normalize(config)
    if normalize is not None:
        transforms_list.append(normalize)
    return transforms.Compose(transforms_list)


def build_eval_transforms(config: TransformConfig) -> transforms.Compose:
    transforms_list: list[transforms.Transform] = [
        _resize(config),
        transforms.ToTensor(),
    ]
    normalize = _maybe_normalize(config)
    if normalize is not None:
        transforms_list.append(normalize)
    return transforms.Compose(transforms_list)


def build_positive_transforms(config: TransformConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            _resize(config),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=_translate_fraction(config, pixels=10),
            ),
            transforms.ColorJitter(brightness=0.2),
        ]
    )


def denormalize(
    tensor: torch.Tensor, mean: Iterable[float], std: Iterable[float]
) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, device=tensor.device)[:, None, None]
    std_tensor = torch.tensor(std, device=tensor.device)[:, None, None]
    return tensor * std_tensor + mean_tensor

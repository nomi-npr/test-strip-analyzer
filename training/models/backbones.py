"""Model builders for the paper replication."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torchvision import models


class VanillaCNN(nn.Module):
    def __init__(self, input_size: int, num_outputs: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            features = self.features(dummy)
            feature_dim = int(features.view(1, -1).shape[1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def _build_head(
    in_features: int, num_outputs: int, hidden_units: int = 256
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.BatchNorm1d(hidden_units),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_units, num_outputs),
    )


def _apply_freeze(model: nn.Module, trainable_modules: Iterable[nn.Module]) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True


def _build_resnet50(pretrained: bool) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        return models.resnet50(weights=weights)
    except (AttributeError, TypeError):
        return models.resnet50(pretrained=pretrained)


def _build_densenet121(pretrained: bool) -> nn.Module:
    try:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        return models.densenet121(weights=weights)
    except (AttributeError, TypeError):
        return models.densenet121(pretrained=pretrained)


def _build_nasnetmobile(pretrained: bool) -> nn.Module:
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "NASNetMobile requires timm. Install via `pip install timm` or the cnn extra."
        ) from exc
    model = timm.create_model(
        "nasnetamobile", pretrained=pretrained, num_classes=0, global_pool="avg"
    )
    return model


def build_model(
    backbone: str,
    num_outputs: int = 2,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    input_size: int = 224,
    head_hidden: int = 256,
) -> nn.Module:
    backbone = backbone.lower()
    if backbone in {"vanilla_cnn", "vanilla", "cnn"}:
        return VanillaCNN(input_size=input_size, num_outputs=num_outputs)
    if backbone in {"resnet50", "resnet"}:
        model = _build_resnet50(pretrained)
        in_features = model.fc.in_features
        head = _build_head(in_features, num_outputs, hidden_units=head_hidden)
        model.fc = head
        if freeze_backbone:
            _apply_freeze(model, [model.fc])
        return model
    if backbone in {"densenet121", "densenet"}:
        model = _build_densenet121(pretrained)
        in_features = model.classifier.in_features
        head = _build_head(in_features, num_outputs, hidden_units=head_hidden)
        model.classifier = head
        if freeze_backbone:
            _apply_freeze(model, [model.classifier])
        return model
    if backbone in {"nasnetmobile", "nasnet"}:
        model = _build_nasnetmobile(pretrained)
        if hasattr(model, "num_features"):
            in_features = int(model.num_features)
        else:
            in_features = model.feature_info[-1]["num_chs"]
        head = _build_head(in_features, num_outputs, hidden_units=head_hidden)
        model = nn.Sequential(model, nn.Flatten(), head)
        if freeze_backbone:
            _apply_freeze(model, [head])
        return model
    raise ValueError(f"Unsupported backbone: {backbone}")

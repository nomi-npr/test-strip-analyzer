"""Sampling utilities for balancing multi-label datasets."""
from __future__ import annotations

import numpy as np


def compute_pos_weight(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 1:
        pos = labels.sum()
        neg = labels.shape[0] - pos
        pos = max(pos, 1.0)
        return np.asarray([neg / pos], dtype=np.float64)
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    pos = np.clip(pos, 1, None)
    return neg / pos


def _normalize_ratios(ratios: np.ndarray) -> np.ndarray:
    ratios = np.asarray(ratios, dtype=np.float64)
    ratios = np.clip(ratios, 1e-6, None)
    return ratios / ratios.sum()


def compute_joint_sample_weights(
    labels: np.ndarray, target_ratios: tuple[float, float, float, float]
) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError("Joint ratio sampling expects label shape (N, 2)")
    target_ratios = _normalize_ratios(np.asarray(target_ratios))
    weights = np.ones(labels.shape[0], dtype=np.float64)
    counts = np.zeros(4, dtype=np.int64)
    masks = [
        (labels[:, 0] == 0) & (labels[:, 1] == 0),
        (labels[:, 0] == 1) & (labels[:, 1] == 0),
        (labels[:, 0] == 0) & (labels[:, 1] == 1),
        (labels[:, 0] == 1) & (labels[:, 1] == 1),
    ]
    for idx, mask in enumerate(masks):
        counts[idx] = int(mask.sum())
    total = labels.shape[0]
    for idx, mask in enumerate(masks):
        if counts[idx] == 0:
            continue
        current_ratio = counts[idx] / total
        factor = target_ratios[idx] / max(current_ratio, 1e-6)
        weights[mask] *= factor
    return weights


def compute_sample_weights(
    labels: np.ndarray,
    target_pos_ratio: tuple[float, ...] | None = None,
    positive_boost: float = 1.0,
    target_joint_ratio: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    labels = np.asarray(labels)
    weights = np.ones(labels.shape[0], dtype=np.float64)

    if labels.ndim == 1:
        if target_pos_ratio is None:
            if positive_boost <= 1.0:
                return weights
        if target_pos_ratio is not None:
            target = float(target_pos_ratio[0])
            target = min(max(target, 1e-3), 1.0 - 1e-3)
            pos_mask = labels == 1
            pos_count = pos_mask.sum()
            neg_count = labels.shape[0] - pos_count
            if pos_count > 0:
                factor = (target / (1.0 - target)) * (neg_count / pos_count)
                weights[pos_mask] *= factor
        if positive_boost > 1.0:
            weights[labels == 1] *= positive_boost
        return weights

    if target_joint_ratio is not None:
        weights *= compute_joint_sample_weights(labels, target_joint_ratio)
        return weights

    if target_pos_ratio is None:
        if positive_boost <= 1.0:
            return weights
        target_pos_ratio = tuple([0.5] * labels.shape[1])

    for idx, target in enumerate(target_pos_ratio):
        target = min(max(float(target), 1e-3), 1.0 - 1e-3)
        pos_mask = labels[:, idx] == 1
        pos_count = pos_mask.sum()
        neg_count = labels.shape[0] - pos_count
        if pos_count == 0:
            continue
        factor = (target / (1.0 - target)) * (neg_count / pos_count)
        weights[pos_mask] *= factor
    if positive_boost > 1.0:
        positive_mask = labels.sum(axis=1) > 0
        weights[positive_mask] *= positive_boost
    return weights

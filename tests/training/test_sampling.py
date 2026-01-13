from __future__ import annotations

import numpy as np

from training.dataset import compute_joint_sample_weights, compute_pos_weight, compute_sample_weights


def test_compute_pos_weight() -> None:
    labels = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    pos_weight = compute_pos_weight(labels)
    assert pos_weight.shape == (2,)
    assert pos_weight[0] > 0
    assert pos_weight[1] > 0


def test_compute_sample_weights() -> None:
    labels = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    weights = compute_sample_weights(labels, target_pos_ratio=(0.5, 0.5))
    assert weights.shape[0] == labels.shape[0]
    assert weights[0] > weights[2]
    assert weights[1] > weights[2]


def test_compute_sample_weights_positive_boost() -> None:
    labels = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    weights = compute_sample_weights(labels, positive_boost=3.0)
    assert weights[0] > weights[2]
    assert weights[1] > weights[2]


def test_compute_sample_weights_binary() -> None:
    labels = np.array([1, 0, 0, 1, 0])
    weights = compute_sample_weights(labels, target_pos_ratio=(0.6,))
    assert weights[0] > weights[1]


def test_compute_joint_sample_weights() -> None:
    labels = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    weights = compute_joint_sample_weights(labels, (6.0, 2.0, 2.0, 1.0))
    assert weights.shape[0] == labels.shape[0]

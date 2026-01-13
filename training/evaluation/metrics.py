"""Metric computation utilities."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _to_numpy(array: np.ndarray | Iterable) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def compute_binary_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    logits = _to_numpy(logits)
    labels = _to_numpy(labels).astype(int)
    probs = softmax(logits)
    preds = probs.argmax(axis=1)
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    return metrics


def compute_multilabel_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> dict[str, object]:
    logits = _to_numpy(logits)
    labels = _to_numpy(labels)
    probs = sigmoid(logits)

    if thresholds is None:
        thresholds = [0.5] * labels.shape[1]
    thresholds = np.asarray(list(thresholds))

    metrics: dict[str, object] = {"per_label": {}, "joint_accuracy": None}
    preds = (probs >= thresholds).astype(int)

    joint_accuracy = float((preds == labels).all(axis=1).mean())
    metrics["joint_accuracy"] = joint_accuracy

    per_label_accuracy = []
    for idx in range(labels.shape[1]):
        label_true = labels[:, idx]
        label_pred = preds[:, idx]
        label_prob = probs[:, idx]
        precision = precision_score(label_true, label_pred, zero_division=0)
        recall = recall_score(label_true, label_pred, zero_division=0)
        f1 = f1_score(label_true, label_pred, zero_division=0)
        acc = accuracy_score(label_true, label_pred)
        per_label_accuracy.append(acc)
        try:
            auroc = roc_auc_score(label_true, label_prob)
        except ValueError:
            auroc = None
        metrics["per_label"][f"test{idx + 1}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(acc),
            "auroc": None if auroc is None else float(auroc),
            "confusion_matrix": confusion_matrix(label_true, label_pred).tolist(),
        }

    metrics["macro_accuracy"] = float(np.mean(per_label_accuracy))
    metrics["macro_f1"] = float(
        np.mean([m["f1"] for m in metrics["per_label"].values()])
    )
    metrics["macro_precision"] = float(
        np.mean([m["precision"] for m in metrics["per_label"].values()])
    )
    metrics["macro_recall"] = float(
        np.mean([m["recall"] for m in metrics["per_label"].values()])
    )
    return metrics


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: Iterable[float] | None = None,
    task: str = "multilabel",
) -> dict[str, object]:
    if task == "binary":
        return compute_binary_metrics(logits, labels)
    return compute_multilabel_metrics(logits, labels, thresholds=thresholds)


def tune_thresholds(
    logits: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
    thresholds: Iterable[float] | None = None,
) -> np.ndarray:
    logits = _to_numpy(logits)
    labels = _to_numpy(labels)
    probs = sigmoid(logits)

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best_thresholds = []
    for idx in range(labels.shape[1]):
        label_true = labels[:, idx]
        best_score = -1.0
        best_threshold = 0.5
        for threshold in thresholds:
            preds = (probs[:, idx] >= threshold).astype(int)
            if metric == "recall":
                score = recall_score(label_true, preds, zero_division=0)
            elif metric == "precision":
                score = precision_score(label_true, preds, zero_division=0)
            else:
                score = f1_score(label_true, preds, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        best_thresholds.append(best_threshold)
    return np.asarray(best_thresholds)

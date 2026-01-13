"""Statistical evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PairedTTestResult:
    t_stat: float
    p_value: float | None
    mean_diff: float
    n: int


def paired_t_test(a: Iterable[float], b: Iterable[float]) -> PairedTTestResult:
    values_a = np.asarray(list(a), dtype=np.float64)
    values_b = np.asarray(list(b), dtype=np.float64)
    if values_a.shape != values_b.shape:
        raise ValueError("Paired t-test requires arrays of equal length.")
    diffs = values_a - values_b
    n = diffs.size
    mean_diff = float(diffs.mean())
    if n < 2:
        return PairedTTestResult(t_stat=float("nan"), p_value=None, mean_diff=mean_diff, n=n)
    std = float(diffs.std(ddof=1))
    if std == 0:
        return PairedTTestResult(t_stat=float("inf"), p_value=0.0, mean_diff=mean_diff, n=n)
    t_stat = mean_diff / (std / sqrt(n))
    p_value = None
    try:
        from scipy import stats

        p_value = float(stats.ttest_rel(values_a, values_b).pvalue)
    except Exception:
        p_value = None
    return PairedTTestResult(t_stat=float(t_stat), p_value=p_value, mean_diff=mean_diff, n=n)

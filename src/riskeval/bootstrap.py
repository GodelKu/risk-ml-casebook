"""Stratified percentile bootstrap for any metric of (y, p)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .metrics import Metric, _as_arrays


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    n_boot: int
    level: float

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "low": self.low, "high": self.high}


def bootstrap_ci(
    y: ArrayLike,
    p: ArrayLike,
    metric: Metric,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
    stratified: bool = True,
) -> Interval:
    """Percentile interval for ``metric(y, p)``.

    Stratified resampling keeps the number of positives fixed in every replicate,
    so metrics like ROC AUC are always defined. The point estimate is the metric
    on the full sample, not the bootstrap mean.
    """
    y_arr, p_arr = _as_arrays(y, p)
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if n_boot < 10:
        raise ValueError("n_boot must be at least 10")
    rng = np.random.default_rng(seed)
    point = float(metric(y_arr, p_arr))
    if stratified:
        pos = np.flatnonzero(y_arr == 1)
        neg = np.flatnonzero(y_arr == 0)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        if stratified:
            idx = np.concatenate(
                [rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
            )
        else:
            idx = rng.integers(0, y_arr.size, y_arr.size)
        stats[b] = metric(y_arr[idx], p_arr[idx])
    alpha = (1 - level) / 2
    low, high = np.quantile(stats, [alpha, 1 - alpha])
    return Interval(point=point, low=float(low), high=float(high), n_boot=n_boot, level=level)

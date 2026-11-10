"""Point metrics for binary risk scores.

All functions take labels ``y`` (0/1) and scores ``p`` (probabilities in [0, 1]).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

Metric = Callable[[NDArray[np.int_], NDArray[np.float64]], float]

_EPS = 1e-12


def _as_arrays(y: ArrayLike, p: ArrayLike) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    y_arr = np.asarray(y)
    p_arr = np.asarray(p, dtype=np.float64)
    if y_arr.shape != p_arr.shape or y_arr.ndim != 1:
        raise ValueError("y and p must be 1-D arrays of the same length")
    if y_arr.size == 0:
        raise ValueError("empty input")
    if not np.isin(y_arr, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    if np.isnan(p_arr).any() or (p_arr < 0).any() or (p_arr > 1).any():
        raise ValueError("scores must be probabilities in [0, 1] without NaN")
    return y_arr.astype(np.int_), p_arr


def roc_auc(y: ArrayLike, p: ArrayLike) -> float:
    """Area under the ROC curve. Needs both classes present."""
    y_arr, p_arr = _as_arrays(y, p)
    return float(roc_auc_score(y_arr, p_arr))


def pr_auc(y: ArrayLike, p: ArrayLike) -> float:
    """Average precision (step-wise area under the precision-recall curve)."""
    y_arr, p_arr = _as_arrays(y, p)
    return float(average_precision_score(y_arr, p_arr))


def brier(y: ArrayLike, p: ArrayLike) -> float:
    """Mean squared error between probability and outcome."""
    y_arr, p_arr = _as_arrays(y, p)
    return float(np.mean((p_arr - y_arr) ** 2))


def log_loss(y: ArrayLike, p: ArrayLike) -> float:
    """Binary cross-entropy with probabilities clipped away from 0 and 1."""
    y_arr, p_arr = _as_arrays(y, p)
    q = np.clip(p_arr, _EPS, 1 - _EPS)
    return float(-np.mean(y_arr * np.log(q) + (1 - y_arr) * np.log(1 - q)))


def calibration_table(y: ArrayLike, p: ArrayLike, n_bins: int = 10) -> list[dict[str, float]]:
    """Equal-mass bins: each bin holds about the same number of rows.

    Bin edges are quantiles of ``p``. Rows with the same score always land in the
    same bin, so the result does not depend on row order (a constant score gives
    one bin). Returns one dict per bin with count, mean prediction and observed rate.
    """
    y_arr, p_arr = _as_arrays(y, p)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    edges = np.unique(np.quantile(p_arr, np.linspace(0, 1, n_bins + 1)[1:-1]))
    bin_id = np.searchsorted(edges, p_arr, side="right")
    rows = []
    for b in np.unique(bin_id):
        idx = np.flatnonzero(bin_id == b)
        rows.append(
            {
                "count": int(idx.size),
                "mean_pred": float(p_arr[idx].mean()),
                "observed": float(y_arr[idx].mean()),
                "p_min": float(p_arr[idx].min()),
                "p_max": float(p_arr[idx].max()),
            }
        )
    return rows


def ece(y: ArrayLike, p: ArrayLike, n_bins: int = 10) -> float:
    """Expected calibration error on equal-mass bins, weighted by bin size."""
    table = calibration_table(y, p, n_bins)
    n = sum(r["count"] for r in table)
    return float(sum(r["count"] / n * abs(r["mean_pred"] - r["observed"]) for r in table))


METRICS: dict[str, Metric] = {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "brier": brier,
    "log_loss": log_loss,
    "ece": ece,
}

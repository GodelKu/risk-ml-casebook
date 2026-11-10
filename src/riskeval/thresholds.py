"""Decision thresholds from error costs or from a false positive budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .metrics import _as_arrays


@dataclass(frozen=True)
class Decision:
    threshold: float
    expected_cost: float
    tpr: float
    fpr: float
    flagged_share: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def bayes_threshold(c_fp: float, c_fn: float) -> float:
    """Optimal threshold for perfectly calibrated probabilities.

    Flag a case when p * c_fn > (1 - p) * c_fp, which gives p > c_fp / (c_fp + c_fn).
    """
    if c_fp <= 0 or c_fn <= 0:
        raise ValueError("costs must be positive")
    return c_fp / (c_fp + c_fn)


def _rates(y: np.ndarray, flagged: np.ndarray) -> tuple[float, float]:
    pos = y == 1
    neg = ~pos
    tpr = float(flagged[pos].mean()) if pos.any() else float("nan")
    fpr = float(flagged[neg].mean()) if neg.any() else float("nan")
    return tpr, fpr


def evaluate_threshold(
    y: ArrayLike, p: ArrayLike, threshold: float, c_fp: float = 1.0, c_fn: float = 1.0
) -> Decision:
    """Flag rows with p >= threshold and report rates and mean cost per row."""
    y_arr, p_arr = _as_arrays(y, p)
    flagged = p_arr >= threshold
    fp = np.sum(flagged & (y_arr == 0))
    fn = np.sum(~flagged & (y_arr == 1))
    tpr, fpr = _rates(y_arr, flagged)
    return Decision(
        threshold=float(threshold),
        expected_cost=float((c_fp * fp + c_fn * fn) / y_arr.size),
        tpr=tpr,
        fpr=fpr,
        flagged_share=float(flagged.mean()),
    )


def threshold_by_cost(
    y: ArrayLike, p: ArrayLike, c_fp: float, c_fn: float, grid: int = 1001
) -> Decision:
    """Empirical threshold that minimises mean cost per row on the given data.

    Pick it on validation data and apply it unchanged to test data.
    """
    if c_fp <= 0 or c_fn <= 0:
        raise ValueError("costs must be positive")
    y_arr, p_arr = _as_arrays(y, p)
    candidates = np.linspace(0.0, 1.0, grid)
    best: Decision | None = None
    for t in candidates:
        d = evaluate_threshold(y_arr, p_arr, float(t), c_fp, c_fn)
        if best is None or d.expected_cost < best.expected_cost - 1e-15:
            best = d
    assert best is not None
    return best


def threshold_at_fpr(y: ArrayLike, p: ArrayLike, alpha: float) -> Decision:
    """Lowest threshold whose false positive rate on these data is at most ``alpha``."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    y_arr, p_arr = _as_arrays(y, p)
    neg_scores = np.sort(p_arr[y_arr == 0])
    if neg_scores.size == 0:
        raise ValueError("need at least one negative")
    k = int(np.floor(alpha * neg_scores.size))  # negatives allowed above the threshold
    if k == 0:
        t = float(np.nextafter(neg_scores[-1], np.inf))
    else:
        t = float(np.nextafter(neg_scores[-k - 1], np.inf))
    return evaluate_threshold(y_arr, p_arr, t)


def baseline_costs(y: ArrayLike, c_fp: float, c_fn: float) -> dict[str, float]:
    """Mean cost per row for the two trivial policies: flag nobody, flag everybody."""
    y_arr = np.asarray(y)
    rate = float(np.mean(y_arr))
    return {"flag_none": c_fn * rate, "flag_all": c_fp * (1 - rate)}

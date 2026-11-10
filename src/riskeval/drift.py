"""Population stability index and metrics per time period."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .metrics import METRICS


def psi(expected: ArrayLike, actual: ArrayLike, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population stability index of ``actual`` relative to ``expected``.

    Bin edges are quantiles of ``expected``; empty bins get a small floor ``eps``.
    PSI = sum((a - e) * ln(a / e)). It is 0 for identical distributions.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[~np.isnan(e)]
    a = a[~np.isnan(a)]
    if e.size == 0 or a.size == 0:
        raise ValueError("empty input")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return 0.0 if np.allclose(a, e[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    e_share = np.histogram(e, edges)[0] / e.size
    a_share = np.histogram(a, edges)[0] / a.size
    e_share = np.maximum(e_share, eps)
    a_share = np.maximum(a_share, eps)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def temporal_metrics(
    df: pd.DataFrame, label: str, score: str, time: str, metrics: tuple[str, ...] = ("roc_auc",)
) -> list[dict[str, float | str]]:
    """Metrics per period plus PSI of the score against the first period."""
    rows: list[dict[str, float | str]] = []
    periods = sorted(df[time].unique())
    ref = df.loc[df[time] == periods[0], score].to_numpy()
    for t in periods:
        part = df[df[time] == t]
        row: dict[str, float | str] = {"period": str(t), "n": float(len(part))}
        y = part[label].to_numpy()
        p = part[score].to_numpy()
        for m in metrics:
            if m in ("roc_auc", "pr_auc") and len(np.unique(y)) < 2:
                warnings.warn(f"period {t}: only one class, {m} skipped", stacklevel=2)
                row[m] = float("nan")
            else:
                row[m] = METRICS[m](y, p)
        row["score_psi"] = psi(ref, p)
        rows.append(row)
    return rows

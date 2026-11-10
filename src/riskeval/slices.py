"""Metrics by group with intervals and a warning for small groups."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .bootstrap import bootstrap_ci
from .metrics import METRICS


def slice_metrics(
    y: ArrayLike,
    p: ArrayLike,
    groups: ArrayLike,
    metric: str = "roc_auc",
    min_n: int = 100,
    n_boot: int = 500,
    seed: int = 0,
) -> list[dict[str, object]]:
    """One row per group: size, positives, metric with a 95% interval, and a small-n flag.

    Groups smaller than ``min_n`` or without both classes are reported but not scored.
    """
    y_arr = np.asarray(y)
    p_arr = np.asarray(p, dtype=float)
    g_arr = np.asarray(groups).astype(str)
    fn = METRICS[metric]
    rows: list[dict[str, object]] = []
    for g in sorted(np.unique(g_arr)):
        mask = g_arr == g
        yg, pg = y_arr[mask], p_arr[mask]
        n_pos = int(yg.sum())
        row: dict[str, object] = {"group": g, "n": int(mask.sum()), "positives": n_pos}
        too_small = mask.sum() < min_n or n_pos == 0 or n_pos == mask.sum()
        row["small_sample"] = bool(too_small)
        if too_small:
            row.update({"point": None, "low": None, "high": None})
        else:
            ci = bootstrap_ci(yg, pg, fn, n_boot=n_boot, seed=seed)
            row.update(ci.as_dict())
        rows.append(row)
    return rows

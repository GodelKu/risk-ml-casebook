"""Safe loading of a predictions file: CSV only, size limit, explicit columns."""

from __future__ import annotations

import os

import pandas as pd

MAX_BYTES = 2 * 1024**3  # 2 GB


def read_predictions(
    path: str | os.PathLike[str],
    label: str,
    score: str,
    extra: tuple[str, ...] = (),
    max_bytes: int = MAX_BYTES,
) -> pd.DataFrame:
    """Read only the needed columns of a CSV file and validate them.

    No pickle, no eval, no network. Raises ValueError with a clear message.
    """
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(f"file is {size} bytes, limit is {max_bytes}")
    cols = [label, score, *[c for c in extra if c]]
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in cols if c not in header]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = pd.read_csv(path, usecols=cols)
    if df[label].isna().any() or df[score].isna().any():
        raise ValueError("label and score must not contain missing values")
    if not df[label].isin([0, 1]).all():
        raise ValueError("label column must contain only 0 and 1")
    if not df[score].between(0, 1).all():
        raise ValueError("score column must contain probabilities in [0, 1]")
    return df

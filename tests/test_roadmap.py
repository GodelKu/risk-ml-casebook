"""Planned extensions from docs/ROADMAP.md. Each test fails until its feature is implemented."""

from __future__ import annotations

import numpy as np
import pytest

import riskeval.bootstrap as bs
import riskeval.drift as dr
import riskeval.metrics as mt
import riskeval.thresholds as th


@pytest.mark.xfail(strict=True, reason="roadmap item 1")
def test_ex1_mcc() -> None:
    y = np.array([0, 0, 1, 1])
    assert mt.mcc(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5) == pytest.approx(1.0)
    assert mt.mcc(y, np.array([0.9, 0.8, 0.2, 0.1]), 0.5) == pytest.approx(-1.0)


@pytest.mark.xfail(strict=True, reason="roadmap item 2")
def test_ex2_uniform_bins() -> None:
    y = np.array([0, 1])
    rows = mt.calibration_table(y, np.array([0.05, 0.95]), n_bins=10, strategy="uniform")
    assert rows[0]["p_max"] < 0.1 and rows[-1]["p_min"] >= 0.9


@pytest.mark.xfail(strict=True, reason="roadmap item 3")
def test_ex3_paired_bootstrap_zero_for_identical() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    iv = bs.paired_bootstrap_ci(y, p, p, mt.roc_auc)
    assert iv.point == iv.low == iv.high == 0.0


@pytest.mark.xfail(strict=True, reason="roadmap item 4")
def test_ex4_cost_curve_endpoint() -> None:
    y = np.array([0, 0, 0, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    thresholds, costs = th.cost_curve(y, p, 1.0, 5.0, grid=11)
    assert thresholds[0] == 0.0
    assert costs[0] == pytest.approx(th.baseline_costs(y, 1.0, 5.0)["flag_all"])


@pytest.mark.xfail(strict=True, reason="roadmap item 5")
def test_ex5_psi_categorical() -> None:
    a = ["x", "y", "y", "z"]
    assert dr.psi_categorical(a, a) == pytest.approx(0.0)
    assert dr.psi_categorical(["x"] * 10, ["y"] * 10) > 1.0

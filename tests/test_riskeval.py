"""Tests with known answers. Synthetic data only."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from riskeval import (
    bayes_threshold,
    bootstrap_ci,
    brier,
    calibration_table,
    ece,
    log_loss,
    pr_auc,
    psi,
    roc_auc,
    slice_metrics,
    temporal_metrics,
    threshold_at_fpr,
    threshold_by_cost,
)
from riskeval.report import render, table


def calibrated_sample(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores p ~ Beta(2, 5) and labels y ~ Bernoulli(p): calibrated by construction."""
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 5, n)
    y = (rng.random(n) < p).astype(int)
    return y, p


# ---------- metrics ----------


def test_perfect_classifier_auc_is_one() -> None:
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert roc_auc(y, p) == 1.0
    assert pr_auc(y, p) == 1.0


def test_random_scores_auc_near_half() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 20000)
    p = rng.random(20000)
    assert abs(roc_auc(y, p) - 0.5) < 0.02


def test_constant_prevalence_score_has_pr_auc_equal_prevalence() -> None:
    y = np.array([0] * 80 + [1] * 20)
    p = np.full(100, 0.2)
    assert pr_auc(y, p) == pytest.approx(0.2)


def test_brier_and_log_loss_hand_computed() -> None:
    y = np.array([1, 0])
    p = np.array([0.8, 0.4])
    assert brier(y, p) == pytest.approx((0.04 + 0.16) / 2)
    assert log_loss(y, p) == pytest.approx(-(np.log(0.8) + np.log(0.6)) / 2)


def test_ece_near_zero_when_calibrated() -> None:
    y, p = calibrated_sample(200_000)
    assert ece(y, p) < 0.005


def test_ece_detects_overconfidence() -> None:
    y, p = calibrated_sample(50_000)
    sharp = np.clip(0.5 + 2 * (p - 0.5), 0, 1)
    assert ece(y, sharp) > ece(y, p) + 0.05


def test_calibration_table_equal_mass() -> None:
    y, p = calibrated_sample(1000)
    rows = calibration_table(y, p, n_bins=10)
    assert len(rows) == 10
    assert all(r["count"] == 100 for r in rows)
    assert [r["mean_pred"] for r in rows] == sorted(r["mean_pred"] for r in rows)


@pytest.mark.parametrize(
    "y,p",
    [
        ([0, 1], [0.5]),
        ([0, 2], [0.1, 0.2]),
        ([0, 1], [0.1, 1.5]),
        ([0, 1], [0.1, np.nan]),
        ([], []),
    ],
)
def test_bad_inputs_raise(y: list, p: list) -> None:
    with pytest.raises(ValueError):
        brier(np.array(y), np.array(p, dtype=float))


# ---------- bootstrap ----------


def test_bootstrap_interval_contains_point_and_is_reproducible() -> None:
    y, p = calibrated_sample(2000)
    a = bootstrap_ci(y, p, roc_auc, n_boot=200, seed=7)
    b = bootstrap_ci(y, p, roc_auc, n_boot=200, seed=7)
    assert a == b
    assert a.low <= a.point <= a.high


def test_stratified_bootstrap_keeps_positives_fixed() -> None:
    y = np.array([0] * 95 + [1] * 5)
    p = np.linspace(0.01, 0.99, 100)
    seen = []

    def count_pos(yy: np.ndarray, pp: np.ndarray) -> float:
        seen.append(int(yy.sum()))
        return float(yy.mean())

    bootstrap_ci(y, p, count_pos, n_boot=50, seed=0)
    assert set(seen) == {5}


@pytest.mark.slow
def test_bootstrap_coverage_close_to_nominal() -> None:
    """Brier of a fixed calibrated model: true value is E[p(1-p)] for p ~ Beta(2,5)."""
    a, b = 2.0, 5.0
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    true_brier = mean - (var + mean**2)  # E[p] - E[p^2]
    rng = np.random.default_rng(123)
    hits = 0
    n_sim = 400
    for s in range(n_sim):
        p = rng.beta(a, b, 400)
        y = (rng.random(400) < p).astype(int)
        ci = bootstrap_ci(y, p, brier, n_boot=300, seed=s, stratified=False)
        hits += ci.low <= true_brier <= ci.high
    coverage = hits / n_sim
    assert 0.93 <= coverage <= 0.97, coverage


# ---------- thresholds ----------


def test_bayes_threshold_formula() -> None:
    assert bayes_threshold(1, 5) == pytest.approx(1 / 6)
    assert bayes_threshold(1, 1) == 0.5


def test_empirical_cost_threshold_matches_theory_on_calibrated_data() -> None:
    y, p = calibrated_sample(400_000, seed=3)
    for c_fp, c_fn in [(1, 5), (1, 2), (2, 1)]:
        d = threshold_by_cost(y, p, c_fp, c_fn, grid=401)
        assert abs(d.threshold - bayes_threshold(c_fp, c_fn)) <= 0.02


def test_threshold_at_fpr_respects_budget() -> None:
    y, p = calibrated_sample(50_000, seed=5)
    for alpha in (0.01, 0.05, 0.2):
        d = threshold_at_fpr(y, p, alpha)
        assert d.fpr <= alpha
        assert d.fpr > alpha - 0.005  # not wastefully strict


# ---------- drift ----------


def test_psi_zero_for_identical_and_positive_for_shift() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=10_000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-12)
    assert psi(x, x + 1.0) > 0.25


def test_temporal_metrics_rows_per_period() -> None:
    y, p = calibrated_sample(3000)
    df = pd.DataFrame({"y": y, "p": p, "m": np.repeat([0, 1, 2], 1000)})
    rows = temporal_metrics(df, "y", "p", "m", ("roc_auc", "brier"))
    assert [r["period"] for r in rows] == ["0", "1", "2"]
    assert rows[0]["score_psi"] == pytest.approx(0.0, abs=1e-12)


# ---------- slices ----------


def test_slice_metrics_flags_small_groups() -> None:
    y, p = calibrated_sample(1050)
    g = np.array(["big"] * 1000 + ["tiny"] * 50)
    rows = {r["group"]: r for r in slice_metrics(y, p, g, n_boot=100)}
    assert rows["tiny"]["small_sample"] is True and rows["tiny"]["point"] is None
    assert rows["big"]["small_sample"] is False
    assert rows["big"]["low"] <= rows["big"]["point"] <= rows["big"]["high"]


# ---------- report and CLI ----------


def test_html_escapes_data_strings() -> None:
    page = render("t", [("groups", table([{"group": "<script>alert(1)</script>", "n": 3}]))])
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "http" not in page.split("<body>")[1]  # no external links in the body


def test_cli_end_to_end(tmp_path) -> None:
    y, p = calibrated_sample(3000)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": y,
            "p": p,
            "month": rng.integers(0, 3, 3000),
            "grp": rng.choice(["a", "<b>"], 3000),
        }
    )
    csv = tmp_path / "preds.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "r.html"
    js = tmp_path / "r.json"
    cmd = [
        sys.executable,
        "-m",
        "riskeval.cli",
        "report",
        str(csv),
        "--label",
        "y",
        "--score",
        "p",
        "--time",
        "month",
        "--group",
        "grp",
        "--n-boot",
        "100",
        "-o",
        str(out),
        "--json",
        str(js),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    page = out.read_text(encoding="utf-8")
    assert "&lt;b&gt;" in page and "<td><b></td>" not in page
    data = json.loads(js.read_text())
    assert data["n"] == 3000
    assert {m["metric"] for m in data["metrics"]} == {
        "roc_auc",
        "pr_auc",
        "brier",
        "log_loss",
        "ece",
    }


def test_cli_rejects_bad_labels(tmp_path) -> None:
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"y": [0, 2], "p": [0.1, 0.2]}).to_csv(csv, index=False)
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "riskeval.cli",
            "report",
            str(csv),
            "--label",
            "y",
            "--score",
            "p",
            "-o",
            str(tmp_path / "x.html"),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "only 0 and 1" in res.stderr


def test_calibration_bins_keep_ties_together_and_ignore_row_order() -> None:
    y = np.array([1] * 22 + [0] * 78)
    p = np.full(100, 0.25)
    rows = calibration_table(y, p, n_bins=10)
    assert len(rows) == 1
    assert ece(y, p) == pytest.approx(0.03)
    y2, p2 = calibrated_sample(5000, seed=9)
    perm = np.random.default_rng(0).permutation(5000)
    assert ece(y2, p2) == pytest.approx(ece(y2[perm], p2[perm]))

"""Case 2: account opening fraud (Bank Account Fraud suite, NeurIPS 2022, Base.csv).

Data: Jesus et al., "Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets
for ML Evaluation", NeurIPS 2022. Kaggle dataset sgpjesus/bank-account-fraud-dataset-neurips-2022,
license CC BY-NC-SA 4.0. The data are synthetic (generated from a real dataset) and are
not stored in this repository.

Usage:
    python cases/bank_fraud/run.py --data data/raw/baf_base.csv           # validation only
    python cases/bank_fraud/run.py --data data/raw/baf_base.csv --final   # opens test once

Protocol: out of time. Train on months 0-4, choose everything on month 5,
open months 6-7 once. The operating point is a 5 % false positive rate, the convention
of the BAF paper. Every number goes to results/bank_fraud.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from riskeval import bootstrap_ci, calibration_table, psi, temporal_metrics  # noqa: E402
from riskeval.metrics import METRICS, roc_auc  # noqa: E402
from riskeval.thresholds import evaluate_threshold, threshold_at_fpr  # noqa: E402

SEED = 42
LABEL = "fraud_bool"
CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
TRAIN_MONTHS, VAL_MONTHS, TEST_MONTHS = [0, 1, 2, 3, 4], [5], [6, 7]
TARGET_FPR = 0.05
OLDER = 50  # customer_age is binned by decade; 50 means 50-59


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {LABEL, "month", "customer_age", *CATEGORICAL}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"unexpected columns, missing {sorted(missing)}")
    return df


def features(df: pd.DataFrame, drop: list[str] | None = None) -> pd.DataFrame:
    """All columns except label and month. Constant columns are dropped by name."""
    x = df.drop(columns=[LABEL, "month", "device_fraud_count"], errors="ignore")
    for c in SENTINEL_COLS:  # -1 is documented as missing; keep the value, add a flag
        if c in x.columns:
            x[f"{c}_is_minus1"] = (x[c] == -1).astype(np.int8)
    for c in CATEGORICAL:
        x[c] = x[c].astype("category")
    return x.drop(columns=drop or [])


def preprocessor(cols: list[str]) -> ColumnTransformer:
    cats = [c for c in CATEGORICAL if c in cols]
    nums = [c for c in cols if c not in cats]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
            ("num", StandardScaler(), nums),
        ]
    )


def recall_at(threshold: float) -> Callable[[np.ndarray, np.ndarray], float]:
    def fn(y: np.ndarray, p: np.ndarray) -> float:
        return float(np.mean(p[y == 1] >= threshold))

    return fn


def tpr_at_fpr(y: np.ndarray, p: np.ndarray, alpha: float = TARGET_FPR) -> float:
    """Recall when the threshold is set on the same data to give FPR alpha (selection only)."""
    return threshold_at_fpr(y, p, alpha).tpr


def fit_models(xtr: pd.DataFrame, ytr: np.ndarray, xva: pd.DataFrame, yva: np.ndarray) -> dict:
    cols = list(xtr.columns)
    models: dict[str, object] = {}
    lr = Pipeline([("pre", preprocessor(cols)), ("m", LogisticRegression(max_iter=3000))])
    models["logistic_regression"] = lr.fit(xtr, ytr)

    best, best_key, grid_log = None, (-1.0, -1.0), []
    for lr_rate in (0.05, 0.1):
        for leaves in (31, 63):
            m = HistGradientBoostingClassifier(
                learning_rate=lr_rate,
                max_leaf_nodes=leaves,
                l2_regularization=1.0,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
                random_state=SEED,
                categorical_features="from_dtype",
            )
            m.fit(xtr, ytr)
            p = m.predict_proba(xva)[:, 1]
            row = {
                "learning_rate": lr_rate,
                "max_leaf_nodes": leaves,
                "val_tpr_at_5pct_fpr": tpr_at_fpr(yva, p),
                "val_roc_auc": roc_auc(yva, p),
                "n_iter": int(m.n_iter_),
            }
            grid_log.append(row)
            key = (row["val_tpr_at_5pct_fpr"], row["val_roc_auc"])
            if key > best_key:
                best, best_key = m, key
    models["gradient_boosting"] = best
    models["_grid"] = grid_log
    return models


def scores(models: dict, x: pd.DataFrame, prior: float) -> dict[str, np.ndarray]:
    out = {"prior_rate": np.full(len(x), prior)}
    for name in ("logistic_regression", "gradient_boosting"):
        out[name] = models[name].predict_proba(x)[:, 1]
    return out


def metric_block(y: np.ndarray, p: np.ndarray, n_boot: int) -> dict[str, dict[str, float]]:
    """Intervals for ranking metrics; point values for the rest (the test set is large)."""
    block: dict[str, dict[str, float]] = {}
    nan = float("nan")
    for name, fn in METRICS.items():
        if name in ("roc_auc", "pr_auc") and np.unique(p).size > 1:
            block[name] = bootstrap_ci(y, p, fn, n_boot=n_boot, seed=SEED).as_dict()
        else:
            block[name] = {"point": fn(y, p), "low": nan, "high": nan}
    return block


def group_fpr(y: np.ndarray, p: np.ndarray, older: np.ndarray, t: float) -> dict[str, float]:
    neg = y == 0
    fpr_old = float(np.mean(p[neg & older] >= t))
    fpr_young = float(np.mean(p[neg & ~older] >= t))
    return {
        "fpr_age_50_plus": fpr_old,
        "fpr_age_under_50": fpr_young,
        "ratio_young_over_old": fpr_young / fpr_old if fpr_old > 0 else float("inf"),
    }


def fairness_block(
    y: np.ndarray, p: np.ndarray, age: np.ndarray, t: float, n_boot: int
) -> dict[str, object]:
    """Predictive equality at a fixed threshold: FPR by age group, with a bootstrap interval
    for the ratio (resampling negatives within each age group)."""
    older = age >= OLDER
    point = group_fpr(y, p, older, t)
    rng = np.random.default_rng(SEED)
    neg_old = np.flatnonzero((y == 0) & older)
    neg_young = np.flatnonzero((y == 0) & ~older)
    flags = p >= t
    ratios = np.empty(n_boot)
    for b in range(n_boot):
        fo = flags[rng.choice(neg_old, neg_old.size)].mean()
        fy = flags[rng.choice(neg_young, neg_young.size)].mean()
        ratios[b] = fy / fo if fo > 0 else np.inf
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    return {
        **point,
        "ratio_low": float(lo),
        "ratio_high": float(hi),
        "n_negatives_50_plus": int(neg_old.size),
        "n_negatives_under_50": int(neg_young.size),
    }


def feature_psi(xtr: pd.DataFrame, xte: pd.DataFrame) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for c in xtr.columns:
        if c in CATEGORICAL or c.endswith("_is_minus1"):
            continue
        rows.append({"feature": c, "psi": psi(xtr[c].to_numpy(float), xte[c].to_numpy(float))})
    return sorted(rows, key=lambda r: -float(r["psi"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--final", action="store_true", help="evaluate the sealed test months once")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "bank_fraud.json")
    args = ap.parse_args()

    raw = args.data.read_bytes()
    df = load(args.data)
    tr = df[df["month"].isin(TRAIN_MONTHS)]
    va = df[df["month"].isin(VAL_MONTHS)]
    te = df[df["month"].isin(TEST_MONTHS)]
    ytr, yva, yte = (d[LABEL].to_numpy() for d in (tr, va, te))

    res: dict[str, object] = {
        "data": {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(df),
            "fraud_rate": float(df[LABEL].mean()),
            "fraud_rate_by_month": {
                str(k): float(v) for k, v in df.groupby("month")[LABEL].mean().items()
            },
            "split": {
                "train_months": TRAIN_MONTHS,
                "validation_months": VAL_MONTHS,
                "test_months": TEST_MONTHS,
                "train": len(tr),
                "validation": len(va),
                "test": len(te),
            },
        },
        "operating_point": {"target_fpr": TARGET_FPR, "chosen_on": "validation month"},
    }

    prior = float(ytr.mean())
    xtr, xva, xte = features(tr), features(va), features(te)
    models = fit_models(xtr, ytr, xva, yva)
    res["gb_grid"] = models.pop("_grid")
    s_va = scores(models, xva, prior)
    res["validation"] = {
        m: {**metric_block(yva, s, args.n_boot), "tpr_at_5pct_fpr": tpr_at_fpr(yva, s)}
        for m, s in s_va.items()
        if m != "prior_rate"
    }

    gb_va = s_va["gradient_boosting"]
    decision = threshold_at_fpr(yva, gb_va, TARGET_FPR)
    res["decision_on_validation"] = decision.as_dict()
    res["fairness_validation"] = fairness_block(
        yva, gb_va, va["customer_age"].to_numpy(), decision.threshold, args.n_boot
    )

    m_abl = fit_models(features(tr, ["customer_age"]), ytr, features(va, ["customer_age"]), yva)[
        "gradient_boosting"
    ]
    p_abl_va = m_abl.predict_proba(features(va, ["customer_age"]))[:, 1]
    d_abl = threshold_at_fpr(yva, p_abl_va, TARGET_FPR)
    res["ablation_validation"] = {
        "dropped": ["customer_age"],
        "decision": d_abl.as_dict(),
        "fairness": fairness_block(
            yva, p_abl_va, va["customer_age"].to_numpy(), d_abl.threshold, args.n_boot
        ),
    }

    if args.final:
        s_te = scores(models, xte, prior)
        gb_te = s_te["gradient_boosting"]
        t = decision.threshold
        applied = evaluate_threshold(yte, gb_te, t)
        per_month = []
        for mth in TEST_MONTHS:
            mask = te["month"].to_numpy() == mth
            per_month.append(
                {"month": mth, **evaluate_threshold(yte[mask], gb_te[mask], t).as_dict()}
            )
        p_abl_te = m_abl.predict_proba(features(te, ["customer_age"]))[:, 1]
        age_te = te["customer_age"].to_numpy()
        drift_frame = pd.DataFrame(
            {
                "y": np.concatenate([yva, yte]),
                "p": np.concatenate([gb_va, gb_te]),
                "month": np.concatenate([va["month"].to_numpy(), te["month"].to_numpy()]),
            }
        )
        res["test"] = {
            "sealed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": {m: metric_block(yte, s, args.n_boot) for m, s in s_te.items()},
            "decision_applied": applied.as_dict(),
            "recall_at_validation_threshold": bootstrap_ci(
                yte, gb_te, recall_at(t), n_boot=args.n_boot, seed=SEED
            ).as_dict(),
            "decision_by_month": per_month,
            "fairness": fairness_block(yte, gb_te, age_te, t, args.n_boot),
            "ablation_without_age": {
                "decision_applied": evaluate_threshold(yte, p_abl_te, d_abl.threshold).as_dict(),
                "fairness": fairness_block(yte, p_abl_te, age_te, d_abl.threshold, args.n_boot),
            },
            "score_by_month": temporal_metrics(
                drift_frame, "y", "p", "month", metrics=("roc_auc", "pr_auc")
            ),
            "feature_psi_train_vs_test": feature_psi(xtr, xte)[:8],
            "calibration_table_gb": calibration_table(yte, gb_te),
        }
        older = np.where(age_te >= OLDER, "50+", "under 50")
        preds = pd.DataFrame({"y": yte, "p": gb_te, "month": te["month"].to_numpy(), "age": older})
        preds.to_csv(ROOT / "data" / "fraud_test_preds.csv", index=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

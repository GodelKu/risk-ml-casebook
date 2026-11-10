"""Case 1: default of credit card clients (UCI dataset 350, CC BY 4.0).

Usage:
    python cases/credit_default/run.py --data data/credit.csv           # validation only
    python cases/credit_default/run.py --data data/credit.csv --final   # opens test once

The input may be the original .xls (needs xlrd) or a CSV export of it,
with the two original header rows or with a single header row.
Every number goes to results/credit_default.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from riskeval import (  # noqa: E402
    bootstrap_ci,
    calibration_table,
    evaluate_threshold,
    threshold_by_cost,
)
from riskeval.metrics import METRICS, roc_auc  # noqa: E402
from riskeval.thresholds import baseline_costs  # noqa: E402

SEED = 42
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
PROTECTED = ["SEX", "MARRIAGE", "AGE"]


def load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xls", ".xlsx"):
        df = pd.read_excel(path, header=1)
    else:
        df = pd.read_csv(path)
        if "X1" in df.columns:  # CSV exported with the first header row kept
            df = pd.read_csv(path, header=1)
    df = df.rename(columns={"default payment next month": "target", "PAY_0": "PAY_1"})
    expected = {"ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "target"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"unexpected columns, missing {sorted(missing)}")
    return df


def describe_codes(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Value counts for columns whose codes are only partly documented. Nothing is changed."""
    cols = ["EDUCATION", "MARRIAGE"] + [f"PAY_{i}" for i in range(1, 7)]
    return {c: {str(k): int(v) for k, v in df[c].value_counts().sort_index().items()} for c in cols}


def features(df: pd.DataFrame, drop: list[str] | None = None) -> pd.DataFrame:
    x = df.drop(columns=["ID", "target"])
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


def fit_models(xtr: pd.DataFrame, ytr: np.ndarray, xva: pd.DataFrame, yva: np.ndarray) -> dict:
    cols = list(xtr.columns)
    models: dict[str, object] = {}
    lr = Pipeline([("pre", preprocessor(cols)), ("m", LogisticRegression(max_iter=2000))])
    models["logistic_regression"] = lr.fit(xtr, ytr)

    best, best_auc, grid_log = None, -1.0, []
    for lr_rate in (0.05, 0.1):
        for leaves in (15, 31):
            for l2 in (0.0, 1.0):
                m = HistGradientBoostingClassifier(
                    learning_rate=lr_rate,
                    max_leaf_nodes=leaves,
                    l2_regularization=l2,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=30,
                    random_state=SEED,
                    categorical_features=[cols.index(c) for c in CATEGORICAL if c in cols],
                )
                m.fit(xtr, ytr)
                auc = roc_auc(yva, m.predict_proba(xva)[:, 1])
                grid_log.append(
                    {
                        "learning_rate": lr_rate,
                        "max_leaf_nodes": leaves,
                        "l2": l2,
                        "val_roc_auc": auc,
                        "n_iter": int(m.n_iter_),
                    }
                )
                if auc > best_auc:
                    best, best_auc = m, auc
    models["gradient_boosting"] = best
    models["_grid"] = grid_log

    mlp = Pipeline(
        [
            ("pre", preprocessor(cols)),
            (
                "m",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    alpha=1e-3,
                    early_stopping=True,
                    max_iter=200,
                    random_state=SEED,
                ),
            ),
        ]
    )
    models["mlp_sklearn"] = mlp.fit(xtr, ytr)
    return models


def scores(models: dict, x: pd.DataFrame, prior: float) -> dict[str, np.ndarray]:
    out = {"prior_rate": np.full(len(x), prior)}
    for name in ("logistic_regression", "gradient_boosting", "mlp_sklearn"):
        out[name] = models[name].predict_proba(x)[:, 1]
    return out


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q)).reshape(-1, 1)


def metric_block(y: np.ndarray, p: np.ndarray, n_boot: int) -> dict[str, dict[str, float]]:
    block = {}
    for name, fn in METRICS.items():
        if name in ("roc_auc", "pr_auc") and np.unique(p).size == 1:
            block[name] = {"point": fn(y, p), "low": float("nan"), "high": float("nan")}
            continue
        block[name] = bootstrap_ci(y, p, fn, n_boot=n_boot, seed=SEED).as_dict()
    return block


def paired_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int) -> dict[str, float]:
    """ROC AUC(a) - ROC AUC(b) on the same resampled rows, stratified."""
    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        deltas[b] = roc_auc(y[idx], p_a[idx]) - roc_auc(y[idx], p_b[idx])
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {"point": roc_auc(y, p_a) - roc_auc(y, p_b), "low": float(lo), "high": float(hi)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--final", action="store_true", help="evaluate the sealed test split once")
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--cost-fn", type=float, default=5.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "credit_default.json")
    args = ap.parse_args()

    raw = args.data.read_bytes()
    df = load(args.data)
    y = df["target"].to_numpy()
    idx = np.arange(len(df))
    i_tr, i_tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
    i_va, i_te = train_test_split(i_tmp, test_size=0.5, stratify=y[i_tmp], random_state=SEED)
    tr, va, te = df.iloc[i_tr], df.iloc[i_va], df.iloc[i_te]

    res: dict[str, object] = {
        "data": {
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(df),
            "features": int(df.shape[1] - 2),
            "default_rate": float(y.mean()),
            "split": {"train": len(tr), "validation": len(va), "test": len(te), "seed": SEED},
            "partly_documented_codes": describe_codes(df),
        },
        "cost_assumption": {
            "c_fp": args.cost_fp,
            "c_fn": args.cost_fn,
            "note": "illustration only, not a real lending cost",
        },
    }

    prior = float(tr["target"].mean())
    xtr, xva, xte = features(tr), features(va), features(te)
    models = fit_models(xtr, tr["target"].to_numpy(), xva, va["target"].to_numpy())
    res["gb_grid"] = models.pop("_grid")
    yva, yte = va["target"].to_numpy(), te["target"].to_numpy()
    s_va = scores(models, xva, prior)
    res["validation"] = {m: metric_block(yva, s, args.n_boot) for m, s in s_va.items()}

    # calibration maps are fitted on validation and judged only on test
    gb_va = s_va["gradient_boosting"]
    platt = LogisticRegression().fit(logit(gb_va), yva)
    iso = IsotonicRegression(out_of_bounds="clip").fit(gb_va, yva)

    def calibrate(p: np.ndarray) -> dict[str, np.ndarray]:
        return {"raw": p, "platt": platt.predict_proba(logit(p))[:, 1], "isotonic": iso.predict(p)}

    res["validation_calibration_raw_gb"] = calibration_table(yva, gb_va)

    # threshold chosen on validation with the raw boosting score
    decision = threshold_by_cost(yva, gb_va, args.cost_fp, args.cost_fn)
    res["decision_on_validation"] = {
        "chosen": decision.as_dict(),
        "baselines": baseline_costs(yva, args.cost_fp, args.cost_fn),
    }

    # ablation without SEX, MARRIAGE, AGE (validation)
    m_abl = fit_models(
        features(tr, PROTECTED), tr["target"].to_numpy(), features(va, PROTECTED), yva
    )["gradient_boosting"]
    p_abl_va = m_abl.predict_proba(features(va, PROTECTED))[:, 1]
    res["ablation_validation"] = {
        "dropped": PROTECTED,
        "roc_auc_full_minus_ablated": paired_delta(yva, gb_va, p_abl_va, args.n_boot),
    }

    if args.final:
        s_te = scores(models, xte, prior)
        gb_te = s_te["gradient_boosting"]
        cal_te = calibrate(gb_te)
        applied = evaluate_threshold(yte, gb_te, decision.threshold, args.cost_fp, args.cost_fn)
        p_abl_te = m_abl.predict_proba(features(te, PROTECTED))[:, 1]
        res["test"] = {
            "sealed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": {m: metric_block(yte, s, args.n_boot) for m, s in s_te.items()},
            "calibration_gb": {
                k: {"ece": METRICS["ece"](yte, v), "brier": METRICS["brier"](yte, v)}
                for k, v in cal_te.items()
            },
            "calibration_table_gb_raw": calibration_table(yte, gb_te),
            "decision_applied": applied.as_dict(),
            "baselines": baseline_costs(yte, args.cost_fp, args.cost_fn),
            "ablation_roc_auc_full_minus_ablated": paired_delta(yte, gb_te, p_abl_te, args.n_boot),
        }

        bands = pd.cut(te["AGE"], [0, 29, 39, 49, 200], labels=["<30", "30-39", "40-49", "50+"])
        preds = pd.DataFrame({"y": yte, "p": gb_te, "age_band": bands.astype(str)})
        preds.to_csv(ROOT / "data" / "credit_test_preds.csv", index=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

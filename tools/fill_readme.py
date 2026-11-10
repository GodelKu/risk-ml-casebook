"""Fill the tables in README.md from results/*.json.

Each table lives between ``<!-- BEGIN name -->`` and ``<!-- END name -->`` markers.
Run after the case scripts. ``tests/test_readme_sync.py`` fails if README and JSON disagree.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CREDIT = ROOT / "results" / "credit_default.json"

MODEL_NAMES = {
    "prior_rate": "Prior rate (no skill)",
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting (sklearn HGB)",
    "mlp_sklearn": "MLP (sklearn)",
}


def ci(block: dict[str, float], digits: int = 3) -> str:
    p, lo, hi = block["point"], block["low"], block["high"]
    if lo != lo or hi != hi:  # NaN
        return f"{p:.{digits}f}"
    return f"{p:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def credit_tables(d: dict) -> dict[str, str]:
    t = d["test"]
    lines = [
        "| Model | ROC AUC | PR AUC | Brier | ECE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in MODEL_NAMES.items():
        m = t["metrics"][key]
        lines.append(
            f"| {label} | {ci(m['roc_auc'])} | {ci(m['pr_auc'])} | {ci(m['brier'])} | "
            f"{ci(m['ece'])} |"
        )
    metrics = "\n".join(lines)

    dec = t["decision_applied"]
    base = t["baselines"]
    decision = "\n".join(
        [
            "| Policy on test | Mean cost per client | Flagged share | TPR | FPR |",
            "| --- | --- | --- | --- | --- |",
            f"| Threshold {dec['threshold']:.3f} chosen on validation | {dec['expected_cost']:.3f} "
            f"| {dec['flagged_share']:.3f} | {dec['tpr']:.3f} | {dec['fpr']:.3f} |",
            f"| Flag nobody | {base['flag_none']:.3f} | 0.000 | 0.000 | 0.000 |",
            f"| Flag everybody | {base['flag_all']:.3f} | 1.000 | 1.000 | 1.000 |",
        ]
    )
    cal = t["calibration_gb"]
    calibration = "\n".join(
        ["| Boosting scores | ECE | Brier |", "| --- | --- | --- |"]
        + [f"| {k} | {v['ece']:.4f} | {v['brier']:.4f} |" for k, v in cal.items()]
    )
    abl = t["ablation_roc_auc_full_minus_ablated"]
    ablation = (
        f"ROC AUC with all features minus ROC AUC without SEX, MARRIAGE and AGE: "
        f"**{ci(abl, 4)}** (paired stratified bootstrap on the test set)."
    )
    data = d["data"]
    summary = (
        f"{data['rows']} clients, {data['features']} features, default rate "
        f"{data['default_rate']:.4f}. Split {data['split']['train']}/"
        f"{data['split']['validation']}/{data['split']['test']} (seed {data['split']['seed']}). "
        f"Test set opened once at {t['sealed_at']}. Input CSV sha256 `{data['file_sha256'][:16]}…`."
    )
    return {
        "credit_summary": summary,
        "credit_metrics": metrics,
        "credit_decision": decision,
        "credit_calibration": calibration,
        "credit_ablation": ablation,
    }


def fill(text: str, tables: dict[str, str]) -> str:
    for name, body in tables.items():
        pattern = re.compile(rf"(<!-- BEGIN {name} -->\n).*?(<!-- END {name} -->)", re.S)
        if not pattern.search(text):
            raise SystemExit(f"marker {name} not found in README")
        text = pattern.sub(lambda m, b=body: m.group(1) + b + "\n" + m.group(2), text)
    return text


def main() -> int:
    tables = credit_tables(json.loads(CREDIT.read_text(encoding="utf-8")))
    new = fill(README.read_text(encoding="utf-8"), tables)
    if "--check" in sys.argv:
        return 0 if new == README.read_text(encoding="utf-8") else 1
    README.write_text(new, encoding="utf-8")
    print("README tables updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

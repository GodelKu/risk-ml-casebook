"""Command line: ``riskeval report preds.csv --label y --score p -o report.html``."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .bootstrap import bootstrap_ci
from .drift import temporal_metrics
from .io import read_predictions
from .metrics import METRICS, calibration_table
from .report import calibration_svg, render, table
from .slices import slice_metrics
from .thresholds import baseline_costs, bayes_threshold, evaluate_threshold, threshold_by_cost


def build_report(args: argparse.Namespace) -> tuple[str, dict[str, object]]:
    df = read_predictions(args.preds, args.label, args.score, (args.time or "", args.group or ""))
    y = df[args.label].to_numpy()
    p = df[args.score].to_numpy()
    summary: dict[str, object] = {
        "n": int(len(df)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
    }
    metric_rows = []
    for name, fn in METRICS.items():
        ci = bootstrap_ci(y, p, fn, n_boot=args.n_boot, seed=args.seed)
        metric_rows.append({"metric": name, **ci.as_dict()})
    summary["metrics"] = metric_rows

    calib = calibration_table(y, p, n_bins=10)
    summary["calibration"] = calib

    decision = threshold_by_cost(y, p, args.cost_fp, args.cost_fn)
    at_bayes = evaluate_threshold(
        y, p, bayes_threshold(args.cost_fp, args.cost_fn), args.cost_fp, args.cost_fn
    )
    summary["decision"] = {
        "cost_fp": args.cost_fp,
        "cost_fn": args.cost_fn,
        "chosen_on_this_data": decision.as_dict(),
        "bayes_threshold_if_calibrated": at_bayes.as_dict(),
        "baselines": baseline_costs(y, args.cost_fp, args.cost_fn),
    }

    sections = [
        (
            "Data",
            table(
                [
                    {
                        "rows": summary["n"],
                        "positives": summary["positives"],
                        "prevalence": summary["prevalence"],
                    }
                ]
            ),
        ),
        (
            "Metrics with 95% bootstrap intervals",
            table(metric_rows)
            + f"<p class='muted'>Stratified bootstrap, {args.n_boot} replicates, "
            f"seed {args.seed}. No-skill PR AUC equals the prevalence.</p>",
        ),
        ("Calibration (equal-mass bins)", calibration_svg(calib) + table(calib)),
        (
            "Cost-based decision",
            table(
                [
                    {"policy": "threshold chosen on this file", **decision.as_dict()},
                    {"policy": "c_fp/(c_fp+c_fn), valid if calibrated", **at_bayes.as_dict()},
                ]
            )
            + table(
                [
                    {"policy": k, "expected_cost": v}
                    for k, v in baseline_costs(y, args.cost_fp, args.cost_fn).items()
                ]
            )
            + "<p class='warn'>A threshold chosen on the same file it is evaluated on "
            "is optimistic. "
            "Choose it on validation data and apply it unchanged to test data.</p>",
        ),
    ]
    if args.time:
        rows = temporal_metrics(df, args.label, args.score, args.time, ("roc_auc", "brier"))
        summary["temporal"] = rows
        sections.append(("By period (score PSI against first period)", table(rows)))
    if args.group:
        slice_rows = slice_metrics(
            y, p, df[args.group].to_numpy(), "roc_auc", n_boot=min(args.n_boot, 500), seed=args.seed
        )
        summary["slices"] = slice_rows
        sections.append(
            (
                "By group (ROC AUC, 95% interval)",
                table(slice_rows)
                + "<p class='muted'>Groups under 100 rows or with one class are not scored.</p>",
            )
        )
    page = render(
        f"Model evaluation: {Path(args.preds).name}",
        sections,
        note=f"riskeval {__version__}. Data never leaves this machine.",
    )
    return page, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="riskeval")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="build an HTML evaluation report from a predictions CSV")
    rep.add_argument("preds")
    rep.add_argument("--label", required=True)
    rep.add_argument("--score", required=True)
    rep.add_argument("--time")
    rep.add_argument("--group")
    rep.add_argument("--cost-fp", type=float, default=1.0)
    rep.add_argument("--cost-fn", type=float, default=5.0)
    rep.add_argument("--n-boot", type=int, default=1000)
    rep.add_argument("--seed", type=int, default=0)
    rep.add_argument("-o", "--output", default="report.html")
    rep.add_argument("--json", help="also write the numbers to this JSON file")
    args = parser.parse_args(argv)
    try:
        page, summary = build_report(args)
    except (ValueError, OSError) as exc:
        print(f"riskeval: error: {html.unescape(str(exc))}", file=sys.stderr)
        return 2
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(page, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
        )
    print(f"wrote {args.output}")
    return 0


def _json_default(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


if __name__ == "__main__":
    raise SystemExit(main())

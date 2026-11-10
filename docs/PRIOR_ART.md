# Prior art (stage 0)

Checked on 2026-11-10. Lines marked *not verified* come from memory or search snippets, not from reading the tool's current docs.

| Tool | What it already does | What it does not do (as far as checked) |
| --- | --- | --- |
| [Fairlearn](https://fairlearn.org/main/user_guide/assessment/confidence_interval_estimation.html) | `MetricFrame` computes any metric overall and by group, with bootstrap intervals (`n_boot`, `ci_quantiles`, `random_state`). Verified on the linked page. | Cost-based thresholds, PSI drift, an offline single-file report. |
| [Deepchecks](https://docs.deepchecks.com/dev/tabular/auto_checks/model_evaluation/plot_calibration_score.html) | Calibration check with reliability curves and Brier score, many other tabular checks. Verified on the linked page. | The calibration page shows no intervals and no cost-based threshold. |
| Evidently | Classification quality and data drift reports, including PSI as one of the drift tests. *not verified in current docs* | Bootstrap intervals on every metric. *not verified* |
| scikit-learn | `roc_auc_score`, `average_precision_score`, `calibration_curve`, `CalibratedClassifierCV`, `FixedThresholdClassifier` and `TunedThresholdClassifierCV`. *versions not verified* | Intervals, drift, slices, a report. |

## What riskeval adds

A small, readable library (under 600 lines) that puts five things in one place with tests against known answers: stratified bootstrap intervals on every metric, equal-mass calibration and ECE, cost-based and FPR-budget thresholds with the trivial policies as baselines, PSI by period, and group slices with small-sample flags. One CLI command writes a single HTML file that works offline and on a phone.

## When to use something else

* Production monitoring with dashboards and many drift tests: use Evidently.
* Broad pre-deployment test suites: use Deepchecks.
* Serious fairness analysis with mitigation algorithms: use Fairlearn.

No maintained tool checked here covers more than 80 % of this exact combination, so the project continues. It is a teaching and portfolio project, not a replacement for these tools.

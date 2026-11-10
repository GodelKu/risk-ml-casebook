# risk-ml-casebook

Honestly evaluated risk models, plus `riskeval`, a small library that evaluates any binary risk model from a CSV of predictions and writes one offline HTML report.

**Case 1, credit card default (UCI 350, CC BY 4.0).** Gradient boosting reaches the ROC AUC shown below on a sealed 20 % test set that was opened once, with 95 % stratified bootstrap intervals for every metric, a calibration check, a decision threshold chosen on validation from an explicit cost assumption, and an ablation without sex, marital status and age. Every number in this file is copied by `tools/fill_readme.py` from `results/*.json`, which the scripts write; a test fails if they disagree.

This is a study on public Taiwanese data from 2005. It is not a model for lending decisions.

## riskeval in one command

```bash
pip install -e .
riskeval report preds.csv --label y --score p --group segment --cost-fp 1 --cost-fn 5 -o report.html
```

The input is a CSV with a 0/1 label and a probability. The report contains:

* ROC AUC, PR AUC, Brier, log loss and ECE, each with a 95 % stratified bootstrap interval;
* a calibration plot and table on equal-mass bins;
* the cost-minimising threshold, the threshold c_fp / (c_fp + c_fn) that is optimal if scores are calibrated, and the costs of flagging nobody and flagging everybody;
* optional metrics per period with the population stability index (PSI) of the score, and per group with small-sample warnings.

Data stays on your machine: no network calls, no telemetry, no pickle. Strings from the data are HTML-escaped.

## Case 1: default of credit card clients

Data: [UCI Machine Learning Repository, dataset 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), Yeh and Lien (2009), license CC BY 4.0. The original `.xls` was converted to CSV with LibreOffice; the data are not stored in this repository.

<!-- BEGIN credit_summary -->
30000 clients, 23 features, default rate 0.2212. Split 18000/6000/6000 (seed 42). Test set opened once at 2026-11-10T09:35:33+00:00. Input CSV sha256 `bf15b02eb5a2c34f…`.
<!-- END credit_summary -->

### Validation protocol

* Stratified 60/20/20 split with a fixed seed. Model choice, the boosting grid, calibration maps and the decision threshold use the training and validation parts only.
* The test part is evaluated only with `--final`, which records the time it was opened.
* The data have no time column, so a later-period check is impossible here (see Limitations).

### Results on the test set (95 % intervals)

<!-- BEGIN credit_metrics -->
| Model | ROC AUC | PR AUC | Brier | ECE |
| --- | --- | --- | --- | --- |
| Prior rate (no skill) | 0.500 | 0.221 | 0.172 [0.172, 0.172] | 0.000 [0.000, 0.000] |
| Logistic regression | 0.728 [0.709, 0.744] | 0.516 [0.490, 0.545] | 0.143 [0.140, 0.147] | 0.054 [0.046, 0.063] |
| Gradient boosting (sklearn HGB) | 0.786 [0.772, 0.800] | 0.566 [0.541, 0.592] | 0.133 [0.129, 0.137] | 0.012 [0.011, 0.025] |
| MLP (sklearn) | 0.782 [0.767, 0.796] | 0.561 [0.537, 0.588] | 0.134 [0.130, 0.138] | 0.008 [0.009, 0.022] |
<!-- END credit_metrics -->

The no-skill PR AUC equals the default rate. A constant score has no ranking, so its ROC AUC is 0.5 with no interval.

### Calibration

Platt and isotonic maps were fitted on validation scores and judged on test. On these data they did not improve the raw boosting scores, which were already close to calibrated.

<!-- BEGIN credit_calibration -->
| Boosting scores | ECE | Brier |
| --- | --- | --- |
| raw | 0.0118 | 0.1330 |
| platt | 0.0142 | 0.1330 |
| isotonic | 0.0193 | 0.1332 |
<!-- END credit_calibration -->

### Decision under an assumed cost

Assumption, for illustration only: a missed default costs 5 units, a wrongly flagged good client costs 1. The threshold was chosen on validation and applied unchanged to test.

<!-- BEGIN credit_decision -->
| Policy on test | Mean cost per client | Flagged share | TPR | FPR |
| --- | --- | --- | --- | --- |
| Threshold 0.145 chosen on validation | 0.545 | 0.508 | 0.805 | 0.423 |
| Flag nobody | 1.106 | 0.000 | 0.000 | 0.000 |
| Flag everybody | 0.779 | 1.000 | 1.000 | 1.000 |
<!-- END credit_decision -->

### Ablation without SEX, MARRIAGE and AGE

<!-- BEGIN credit_ablation -->
ROC AUC with all features minus ROC AUC without SEX, MARRIAGE and AGE: **0.0025 [-0.0005, 0.0055]** (paired stratified bootstrap on the test set).
<!-- END credit_ablation -->

An interval that contains zero means no measurable difference on this test set. In the United States, Regulation B (12 CFR 1002.6(b)) says a creditor shall not consider sex in any aspect of a credit transaction ((b)(9)), shall evaluate married and unmarried applicants by the same standards ((b)(8)), and may use age in an empirically derived, statistically sound scoring system only if the age of an elderly applicant is not assigned a negative factor or value ((b)(2)(ii)). Source: [law.cornell.edu](https://www.law.cornell.edu/cfr/text/12/1002.6). Removing the columns does not remove proxies for them.

## Limitations

* One cross-sectional snapshot from Taiwan in 2005. No time column, so no out-of-time test and no drift check for case 1.
* Some codes are only partly documented (`EDUCATION` 0, 5, 6; `MARRIAGE` 0; `PAY_*` values -2 and 0). They were counted in `results/credit_default.json` and left unchanged.
* The cost ratio 5:1 is an assumption, not a measured business cost. Real costs depend on exposure, recovery and interest.
* Stratified bootstrap keeps the class counts fixed, so intervals do not include uncertainty in the default rate itself.
* ECE is biased upward on small bins, and its percentile bootstrap interval can sit slightly above the point estimate.
* The environment used to build this repository had no LightGBM or PyTorch. Boosting uses scikit-learn's `HistGradientBoostingClassifier` and the neural network is scikit-learn's `MLPClassifier`.
* Test coverage percentage was not measured (no coverage tool in the build environment).
* Case 2 (account opening fraud, temporal split and drift) is planned and not yet included.

## Why not Evidently, Deepchecks or Fairlearn

They do more and are maintained by teams. `riskeval` is under 600 lines, every function is tested against a known answer, and one command gives intervals, calibration, a cost-based threshold, drift and slices in one offline file. See [docs/PRIOR_ART.md](docs/PRIOR_ART.md).

## Reproduce

```bash
pip install -e ".[dev]" && pytest -q
python cases/credit_default/run.py --data data/raw/uci350_credit.csv --final
python tools/fill_readme.py
```

Download the dataset from the UCI page above and export the `.xls` to `data/raw/uci350_credit.csv` (for example `libreoffice --headless --convert-to csv`), or pass the `.xls` directly if `xlrd` is installed.

MIT license. Developed with AI assistance.

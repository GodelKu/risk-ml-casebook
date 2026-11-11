# risk-ml-casebook

[![CI](https://github.com/GodelKu/risk-ml-casebook/actions/workflows/ci.yml/badge.svg)](https://github.com/GodelKu/risk-ml-casebook/actions/workflows/ci.yml) ![Python 3.10 to 3.13](https://img.shields.io/badge/python-3.10%20to%203.13-blue) ![License MIT](https://img.shields.io/badge/license-MIT-green)

Honestly evaluated risk models, plus `riskeval`, a small library that evaluates any binary risk model from a CSV of predictions and writes one offline HTML report.

**Case 1, credit card default (UCI 350, CC BY 4.0).** Gradient boosting reaches the ROC AUC shown below on a sealed 20 % test set that was opened once, with 95 % stratified bootstrap intervals for every metric, a calibration check, a decision threshold chosen on validation from an explicit cost assumption, and an ablation without sex, marital status and age. Every number in this file is copied by `tools/fill_readme.py` from `results/*.json`, which the scripts write; a test fails if they disagree.

**Case 2, account opening fraud (Bank Account Fraud suite, NeurIPS 2022).** One million synthetic applications over eight months with about 1 % fraud. The test is out of time: train on months 0 to 4, choose everything on month 5, open months 6 and 7 once. The operating point is a 5 % false positive rate fixed on validation, followed by a check of what that threshold actually does on later months, false positive rates by age group, and feature and score drift.

Both cases are studies on public data. Neither is a model for decisions about real people.

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

## Case 2: account opening fraud, out of time

Data: Base.csv of the [Bank Account Fraud suite](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) (Jesus et al., "Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation", NeurIPS 2022), license CC BY-NC-SA 4.0. The data are synthetic, generated from a real fraud dataset, and are not stored in this repository. Model card: [cases/bank_fraud/MODEL_CARD.md](cases/bank_fraud/MODEL_CARD.md).

<!-- BEGIN fraud_summary -->
1000000 applications, fraud rate 0.0110 overall, from 0.0087 to 0.0147 by month. Train months 0 to 4 (675666 rows), validation month 5 (119323), test months 6 and 7 (205011). Test months opened once at 2026-11-11T00:00:10+00:00. Input CSV sha256 `7bf10a37ce07e72e…`.
<!-- END fraud_summary -->

### Validation protocol

* Split by month, not at random, so the test measures the model on applications that arrive after training.
* Gradient boosting (4-point grid) is chosen by recall at 5 % FPR on month 5. Logistic regression is the baseline.
* The threshold is the lowest score with at most 5 % false positives on month 5 and is applied unchanged to months 6 and 7.

### Results on the test months (95 % intervals)

<!-- BEGIN fraud_metrics -->
| Model | ROC AUC | PR AUC | Brier |
| --- | --- | --- | --- |
| Prior rate (no skill) | 0.500 | 0.014 | 0.0139 |
| Logistic regression | 0.882 [0.876, 0.888] | 0.162 [0.151, 0.175] | 0.0129 |
| Gradient boosting (sklearn HGB) | 0.889 [0.883, 0.895] | 0.185 [0.173, 0.198] | 0.0126 |
<!-- END fraud_metrics -->

With 1.4 % fraud in the test months, PR AUC is the more telling number: no-skill PR AUC equals the fraud rate.

### What a threshold fixed on validation does later

<!-- BEGIN fraud_decision -->
| Where | Threshold | FPR | Recall (TPR) | Flagged share |
| --- | --- | --- | --- | --- |
| Validation month (threshold chosen here) | 0.0277 | 0.0500 | 0.542 | 0.0558 |
| Test months 6 and 7 | same | 0.0573 | 0.569 [0.553, 0.586] | 0.0645 |
| Test month 6 | same | 0.0650 | 0.579 | 0.0719 |
| Test month 7 | same | 0.0487 | 0.558 | 0.0562 |
<!-- END fraud_decision -->

The threshold was set for 5 % false positives, and the realised rate moved with the month. A fraud team plans its review capacity from this number, so in practice the threshold would be re-checked every month against the alert budget.

### False positive rate by age group

<!-- BEGIN fraud_fairness -->
| Test months 6 and 7 | Recall | FPR under 50 | FPR 50 and over | FPR ratio under 50 / 50+ |
| --- | --- | --- | --- | --- |
| All features | 0.569 | 0.0420 | 0.1366 | 0.307 [0.297, 0.318] |
| Without customer_age | 0.562 | 0.0469 | 0.1094 | 0.429 [0.413, 0.444] |
<!-- END fraud_fairness -->

Legitimate applicants aged 50 and over are flagged about three times as often as younger ones. Dropping `customer_age` narrows the gap at a small cost in recall but does not close it, which suggests that other features carry information about age. Following the BAF paper this is reported as predictive equality (equal false positive rates); which fairness criterion should apply is a policy decision, not a modelling one.

### Drift between months

<!-- BEGIN fraud_drift -->
| Month | Rows | ROC AUC | PR AUC | Score PSI vs month 5 |
| --- | --- | --- | --- | --- |
| 5 | 119323 | 0.893 | 0.178 | 0.0000 |
| 6 | 108168 | 0.890 | 0.172 | 0.0090 |
| 7 | 96843 | 0.891 | 0.207 | 0.0321 |

Largest feature shifts, train months vs test months (PSI): `velocity_4w` 3.739, `velocity_24h` 1.892, `velocity_6h` 1.089, `zip_count_4w` 0.491, `date_of_birth_distinct_emails_4w` 0.399.
<!-- END fraud_drift -->

Several input features shift strongly between the training and test months (PSI well above the common 0.25 rule of thumb), while the score distribution and ROC AUC stay almost stable. Feature PSI alone would raise alarms here; the score PSI and the monthly performance are what show whether the model still works.

## Limitations

* One cross-sectional snapshot from Taiwan in 2005. No time column, so no out-of-time test and no drift check for case 1.
* Some codes are only partly documented (`EDUCATION` 0, 5, 6; `MARRIAGE` 0; `PAY_*` values -2 and 0). They were counted in `results/credit_default.json` and left unchanged.
* The cost ratio 5:1 is an assumption, not a measured business cost. Real costs depend on exposure, recovery and interest.
* Stratified bootstrap keeps the class counts fixed, so intervals do not include uncertainty in the default rate itself.
* ECE is biased upward on small bins, and its percentile bootstrap interval can sit slightly above the point estimate.
* The environment used to build this repository had no LightGBM or PyTorch. Boosting uses scikit-learn's `HistGradientBoostingClassifier` and the neural network is scikit-learn's `MLPClassifier`.
* Test coverage percentage was not measured (no coverage tool in the build environment).
* Case 2 uses synthetic data, so its absolute numbers say little about any real bank. Only two months are held out, and the 5 % FPR operating point is a convention, not a measured cost. The grid search on month 5 makes the validation numbers slightly optimistic; the test numbers are the honest ones.

## Why not Evidently, Deepchecks or Fairlearn

They do more and are maintained by teams. `riskeval` is about 700 lines, its metrics, intervals and thresholds are tested against known answers, and one command gives intervals, calibration, a cost-based threshold, drift and slices in one offline file. See [docs/PRIOR_ART.md](docs/PRIOR_ART.md).

## Reproduce

```bash
pip install -e ".[dev]" && pytest -q
python cases/credit_default/run.py --data data/raw/uci350_credit.csv --final
python cases/bank_fraud/run.py --data data/raw/baf_base.csv --final
python tools/fill_readme.py
```

For case 2, download Base.csv from the Kaggle page above (an account is needed) and save it as `data/raw/baf_base.csv`. For case 1, download the dataset from the UCI page above and export the `.xls` to `data/raw/uci350_credit.csv` (for example `libreoffice --headless --convert-to csv`), or pass the `.xls` directly if `xlrd` is installed.

## Documentation

* [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md): every module, why it is built that way, and what was rejected.
* [docs/DESIGN_FAQ.md](docs/DESIGN_FAQ.md): questions a reviewer is likely to ask, with answers tied to code and results.
* [docs/PRIOR_ART.md](docs/PRIOR_ART.md): existing tools and where this project differs.
* [docs/ROADMAP.md](docs/ROADMAP.md): planned extensions, each with a failing test already written.
* Model cards: [case 1](cases/credit_default/MODEL_CARD.md), [case 2](cases/bank_fraud/MODEL_CARD.md).

## How this was built

I built this together with an AI coding assistant (Claude). The implementation was shared work: I wrote much of it with the assistant's help, and the assistant wrote the rest and drafted most of the documentation. I chose the problems and the datasets, agreed on the evaluation rules before any code was written, and supplied the data. The rules were set so that no code, mine or generated, can quietly produce a wrong number:

* metrics, intervals and thresholds have tests with known answers, including a simulation that checks bootstrap coverage;
* the test split is opened once, by a separate `--final` run that records the time;
* README tables are generated from the results JSON, and a test fails if the two disagree;
* CI runs lint, type checks and tests on Linux, macOS and Windows.

Commits written with the assistant carry a `Co-Authored-By` line. I am responsible for every claim in this README; if you find one the code does not support, please open an issue.

Author: Georgii Baiev, physics student at California State University, Northridge. MIT license.

# Walkthrough

Read this with the code open. For each module: what it does, why it is done this way, and what was rejected.

## src/riskeval/metrics.py

* `_as_arrays` validates once: 1-D, same length, labels only 0/1, scores in [0, 1], no NaN. Every public function calls it, so bad input fails loudly instead of producing a wrong number.
* `roc_auc`, `pr_auc` wrap scikit-learn. PR AUC is average precision, the step-wise area. Its no-skill value is the positive rate, not 0.5.
* `brier` is the mean squared error of the probability. `log_loss` clips to [1e-12, 1 - 1e-12] so a confident wrong prediction gives a large but finite number.
* `calibration_table` uses **equal-mass** bins: edges are quantiles of the scores. Equal-width bins were rejected because most credit scores sit below 0.3, which leaves the upper bins nearly empty. Ties always stay in one bin (`np.searchsorted` on unique edges), so the result does not depend on row order. An earlier version split the sorted rows into equal chunks; with a constant score that put all positives into one bin after resampling and gave a nonsense ECE. The test `test_calibration_bins_keep_ties_together_and_ignore_row_order` guards this.
* `ece` is the size-weighted mean of |mean predicted − observed rate| over bins.

## src/riskeval/bootstrap.py

* Percentile bootstrap: resample rows with replacement, recompute the metric, take the 2.5 % and 97.5 % quantiles.
* **Stratified** by default: positives and negatives are resampled separately, so every replicate has the same number of each. This keeps ROC AUC defined on rare-event data. Cost: the interval ignores uncertainty in the prevalence (see Limitations).
* The point estimate is the metric on the full sample, not the mean of replicates.
* Rejected: BCa intervals (more accurate in theory, harder to explain and to test), and normal approximations (bad for bounded metrics near 0 or 1).

## src/riskeval/thresholds.py

* `bayes_threshold`: if probabilities are calibrated, flag when p·c_fn > (1 − p)·c_fp, so the threshold is c_fp / (c_fp + c_fn). With 1 and 5 this is 1/6 ≈ 0.167.
* `threshold_by_cost` searches a grid of thresholds and returns the one with the lowest mean cost per row on the data given. Use validation data; the README warns that choosing and evaluating on the same file is optimistic.
* `threshold_at_fpr`: sorts negative scores and picks the lowest threshold that leaves at most ⌊α·n_neg⌋ negatives at or above it. `np.nextafter` moves just above a score so ties are excluded.
* `baseline_costs`: flagging nobody costs c_fn × prevalence; flagging everybody costs c_fp × (1 − prevalence). A model is only useful if it beats both.

## src/riskeval/drift.py

* `psi`: bins from quantiles of the reference sample, a small floor `eps` for empty bins, then Σ (a − e) ln(a / e). Zero for identical distributions. Common rule of thumb (not a law): below 0.1 stable, above 0.25 large shift.
* `temporal_metrics`: metrics per period plus the PSI of the score against the first period.

## src/riskeval/slices.py

Metric per group with a bootstrap interval. Groups under `min_n` rows or with one class are listed but not scored, to avoid confident numbers from tiny samples.

## src/riskeval/io.py and cli.py

* Reads only the needed CSV columns, checks file size, labels and score range. No pickle, no eval, no network.
* `cli.py` builds the report sections and optionally writes the same numbers to JSON.

## src/riskeval/report.py

One HTML string with inline CSS and an inline SVG calibration plot. Every data string passes through `html.escape`; the test injects `<script>alert(1)</script>`. CSS uses `prefers-color-scheme` for dark mode and `overflow-x:auto` wrappers so tables never widen the page at 360 px.

## cases/credit_default/run.py

1. `load` renames `default payment next month` to `target` and `PAY_0` to `PAY_1`, so repayment status and bill columns share month indices.
2. `describe_codes` counts partly documented codes and changes nothing.
3. Split 60/20/20, stratified, seed 42. All choices use train and validation.
4. Models: prior rate, logistic regression in a pipeline (one-hot plus scaling fitted on train only), gradient boosting grid on validation, MLP.
5. Calibration maps (Platt on the logit, isotonic) are fitted on validation and judged only on test.
6. Threshold by cost on validation, applied unchanged to test, compared with the two trivial policies.
7. Ablation: the same grid without SEX, MARRIAGE, AGE; the difference in ROC AUC uses a **paired** bootstrap, the same resampled rows for both models, which is much tighter than comparing two separate intervals.
8. `--final` opens the test set and writes the time.

## tools/fill_readme.py and tests/test_readme_sync.py

README tables are generated from JSON between markers. The sync test rebuilds them and fails if the README was edited by hand.

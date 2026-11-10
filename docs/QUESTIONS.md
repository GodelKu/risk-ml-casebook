# 25 interview questions

Answer each without looking, then check. File references point to where the answer lives in code.

1. **Why not report accuracy?** With 22 % defaults, always predicting "no default" scores 78 %. Use ROC AUC, PR AUC, Brier, log loss. (`metrics.py`)
2. **What is the no-skill PR AUC?** The positive rate, here 0.221. (`README`, test `test_constant_prevalence_score_has_pr_auc_equal_prevalence`)
3. **ROC AUC in one sentence?** The probability that a random defaulter gets a higher score than a random non-defaulter.
4. **Why does ROC AUC look good on rare events when the model is weak?** It is driven by the huge pool of easy negatives; PR AUC exposes poor precision.
5. **Brier vs log loss?** Both are proper scoring rules. Log loss punishes confident mistakes much harder; Brier is bounded and easier to read.
6. **What is ECE and how did you bin?** Weighted mean gap between predicted and observed rate; equal-mass quantile bins, ties kept together. (`calibration_table`)
7. **Why equal-mass bins?** Scores cluster at low values; equal-width bins at the top would be nearly empty.
8. **How are the intervals computed?** Stratified percentile bootstrap, 1000 replicates, fixed seed. (`bootstrap.py`)
9. **Why stratify, and what does it cost?** Keeps class counts fixed so AUC is always defined; ignores uncertainty in prevalence.
10. **How did you check the intervals are right?** Simulation with a known true Brier; 95 % intervals covered it in 93–97 % of runs. (`test_bootstrap_coverage_close_to_nominal`)
11. **Why a 60/20/20 split and not just cross-validation?** Validation picks the model, grid and threshold; a sealed test gives one honest final number.
12. **What stops you from peeking at the test set?** Test is only scored with `--final`, which records the time; README numbers come only from that run.
13. **Why can't you do an out-of-time test here?** The dataset has no date column.
14. **Why is the threshold not 0.5?** Errors cost differently; under 5:1 the calibrated optimum is 1/6. (`bayes_threshold`)
15. **Your chosen threshold was 0.145, not 0.167. Why?** It is the empirical minimum on validation; scores are not perfectly calibrated and the cost curve is flat near the minimum.
16. **Is the model useful at all?** Mean cost 0.545 per client on test versus 0.779 for flagging everybody and 1.106 for flagging nobody.
17. **Why calibrate on validation and judge on test?** Fitting and judging on the same data makes calibration look perfect.
18. **Did calibration help?** No; raw boosting ECE was already lowest. Isotonic overfit slightly.
19. **Why did you use HistGradientBoosting instead of LightGBM?** LightGBM was not installable in the build environment; HGB is the same family (histogram-based boosting) and the plan allowed it.
20. **How did you tune boosting?** 8-point grid (learning rate, leaves, L2) with early stopping inside training, chosen by validation ROC AUC. (`gb_grid` in JSON)
21. **What does the ablation show?** Dropping SEX, MARRIAGE and AGE changed test ROC AUC by about 0.0025 with an interval that includes zero.
22. **Why a paired bootstrap for the difference?** Both models are scored on the same resampled rows, which removes shared noise and gives a much tighter interval.
23. **Does dropping protected columns make the model fair?** No. Other features can act as proxies, and fairness needs outcome analysis by group.
24. **What is PSI and when would you use it here?** A measure of distribution shift between a reference and a new sample; it needs time periods, which case 1 lacks. (`drift.py`)
25. **What would you do next with real data?** Out-of-time validation, costs from actual exposure and recovery, monitoring PSI of score and key features, and checking outcomes by group.

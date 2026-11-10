# Five exercises

Each has a test in `tests/test_exercises.py` marked `xfail(strict=True)`. When your code is right the test passes, pytest reports XPASS as a failure, and you remove the marker. Commit each one yourself.

1. **Matthews correlation at a threshold.** Add `mcc(y, p, threshold)` to `metrics.py`. Known answer: perfect predictions give 1, inverted give −1.
2. **Equal-width calibration option.** Add `strategy="quantile" | "uniform"` to `calibration_table`. With uniform bins and 10 bins, edges are 0.1, 0.2, …; scores 0.05 and 0.95 land in the first and last bin.
3. **Bootstrap for the difference of two metrics.** Move the paired bootstrap from the case script into `bootstrap.py` as `paired_bootstrap_ci(y, p_a, p_b, metric)`. Known answer: identical scores give exactly 0 with a zero-width interval.
4. **Cost curve in the report.** Add `cost_curve(y, p, c_fp, c_fn, grid)` returning thresholds and mean costs. Known answer: at threshold 0 the cost equals the "flag everybody" baseline.
5. **PSI for categorical columns.** Add `psi_categorical(expected, actual)` using category shares instead of quantile bins. Known answer: identical label lists give 0; disjoint categories give a large positive value.

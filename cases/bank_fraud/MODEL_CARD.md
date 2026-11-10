# Model card: account opening fraud, case 2

**Model.** scikit-learn `HistGradientBoostingClassifier` with native categorical handling, chosen from a 4-point grid (learning rate 0.05 or 0.1, 31 or 63 leaves, L2 1.0) by recall at 5 % false positive rate on the validation month, ties broken by ROC AUC (`results/bank_fraud.json`, key `gb_grid`). A logistic regression (one-hot plus scaling) is the baseline.

**Intended use.** Teaching and portfolio: showing out-of-time evaluation, an operating point fixed in advance, drift and a fairness check on an imbalanced problem. Not for decisions about real people.

**Data.** Base.csv of the Bank Account Fraud suite (Jesus et al., NeurIPS 2022), 1 000 000 synthetic bank account applications generated from a real dataset, about 1.1 % fraud, 8 months. License CC BY-NC-SA 4.0. Not stored in this repository. `device_fraud_count` is constant in Base.csv and is dropped. For five columns the value -1 marks a missing value; the value is kept and a 0/1 flag is added.

**Evaluation.** Train months 0 to 4, validation month 5, test months 6 and 7 opened once. The threshold is the lowest score with at most 5 % false positives on the validation month; it is applied unchanged to the test months, so the realised test FPR can differ from 5 %. Ranking metrics carry 95 % stratified bootstrap intervals.

**Fairness.** Predictive equality by age group (under 50 versus 50 and over, following the BAF paper): false positive rates at the fixed threshold and their ratio with a bootstrap interval. A model without `customer_age` is reported next to it. Removing the column does not remove proxies for it.

**Known limitations.** Synthetic data, so absolute numbers say little about any real bank. Only two test months. The 5 % FPR operating point is a convention, not a measured cost. Grid search on the validation month makes the validation numbers slightly optimistic; the test numbers are the honest ones.

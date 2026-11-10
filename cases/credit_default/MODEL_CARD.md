# Model card: credit default, case 1

**Model.** scikit-learn `HistGradientBoostingClassifier`, chosen from an 8-point grid by validation ROC AUC (`results/credit_default.json`, key `gb_grid`). Native categorical handling for SEX, EDUCATION and MARRIAGE.

**Intended use.** Teaching and portfolio: showing how to evaluate a risk score honestly. Not for decisions about real people.

**Data.** UCI dataset 350, 30 000 credit card clients in Taiwan, 2005, CC BY 4.0. Target: default payment in the next month. 23 features: credit limit, sex, education, marital status, age, six months of repayment status, bill amounts and payment amounts.

**Evaluation.** Stratified 60/20/20 split, seed 42. Test opened once. Metrics with 95 % stratified bootstrap intervals are in the README and in `results/credit_default.json` under `test`.

**Decision.** Threshold chosen on validation to minimise mean cost under an assumed 5:1 cost of a missed default versus a false flag. The ratio is an illustration.

**Sensitive attributes.** An ablation without SEX, MARRIAGE and AGE is reported. Removing columns does not remove proxies. Under US Regulation B, sex may not be considered and married and unmarried applicants must be evaluated by the same standards (12 CFR 1002.6(b)(8), (b)(9)).

**Known limitations.** No time dimension, so no out-of-time validation. Partly documented codes left as they are. Costs are assumed. One country and one year.

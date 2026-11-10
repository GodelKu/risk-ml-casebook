# Commit plan

Commit in this order yourself (`git add <files>` then `git commit -m "<message>"`). Read each file before you commit it.

1. `.gitignore`, `LICENSE`, `pyproject.toml` — "Project skeleton: packaging, license, ignore data"
2. `src/riskeval/__init__.py`, `src/riskeval/metrics.py` — "Add point metrics and equal-mass calibration"
3. `src/riskeval/bootstrap.py` — "Add stratified percentile bootstrap"
4. `src/riskeval/thresholds.py` — "Add cost and FPR-budget thresholds with trivial baselines"
5. `src/riskeval/drift.py`, `src/riskeval/slices.py` — "Add PSI, per-period metrics and group slices"
6. `src/riskeval/io.py`, `src/riskeval/report.py`, `src/riskeval/cli.py` — "Add safe CSV loading, offline HTML report and CLI"
7. `tests/test_riskeval.py` — "Add known-answer tests, including bootstrap coverage"
8. `docs/PRIOR_ART.md` — "Document prior art and scope"
9. `cases/credit_default/run.py`, `cases/credit_default/MODEL_CARD.md` — "Add credit default case with sealed test split"
10. `results/credit_default.json`, `results/credit_default_report.json` — "Add case 1 results from the sealed run"
11. `tools/fill_readme.py`, `tests/test_readme_sync.py`, `README.md` — "Generate README tables from results"
12. `docs/WALKTHROUGH.md`, `docs/QUESTIONS.md`, `docs/EXERCISES.md`, `tests/test_exercises.py` — "Add walkthrough, interview questions and exercises"
13. `.github/workflows/ci.yml`, `CHANGELOG.md`, `COMMIT_PLAN.md` — "Add CI on three systems and changelog"

Then do the five exercises as five more commits of your own.

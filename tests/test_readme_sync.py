"""README tables must equal what tools/fill_readme.py builds from results/*.json."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_readme_matches_results() -> None:
    if not (ROOT / "results" / "credit_default.json").exists():
        pytest.skip("results not generated yet")
    spec = importlib.util.spec_from_file_location("fill_readme", ROOT / "tools" / "fill_readme.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import json

    tables = mod.credit_tables(json.loads((ROOT / "results" / "credit_default.json").read_text()))
    fraud = ROOT / "results" / "bank_fraud.json"
    if fraud.exists():
        tables.update(mod.fraud_tables(json.loads(fraud.read_text())))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert mod.fill(readme, tables) == readme, "run: python tools/fill_readme.py"

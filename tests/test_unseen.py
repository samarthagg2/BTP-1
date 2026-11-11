"""Sanity tests for Phase 5 (src.p5_unseen, src.p5_shap). Run: python3 -m pytest -q tests/"""
import json

import numpy as np
import pandas as pd
import pytest

from config import RESULTS_DIR
from src.features import load_features
from src.p5_unseen import verdicts


@pytest.fixture(scope="module")
def unseen():
    p = RESULTS_DIR / "p5_unseen.json"
    if not p.exists():
        pytest.skip("run python3 -m src.p5_unseen first")
    return json.load(open(p))


def test_unseen_counts(unseen):
    df = load_features()
    u = df[df.split == "unseen"]
    assert unseen["n"] == {"all": 39, "standard": 30, "newstructure": 9}
    assert len(u) == 39 and u.policy.str.contains("newstructure").sum() == 9


def test_predictions_cover_every_unseen_cell(unseen):
    P = pd.DataFrame(unseen["predictions"])
    assert len(P) == 39 and np.isfinite(P.values).all() and (P.values > 0).all()


def test_verdicts_follow_the_registered_rules(unseen):
    assert verdicts(unseen["hypotheses"]) == unseen["verdicts"]
    h = json.loads(json.dumps(unseen["hypotheses"]))
    h["H5"]["gp_f4_coverage_u1"] = 0.95                      # rule: < 0.90 -> supported
    assert verdicts(h)["H5"] is False


def test_shap_values_add_up_to_predictions():
    p = RESULTS_DIR / "p5_shap_values.csv"
    if not p.exists():
        pytest.skip("run python3 -m src.p5_shap first")
    S = json.load(open(RESULTS_DIR / "p5_shap.json"))
    V = pd.read_csv(p, index_col=0)
    U = json.load(open(RESULTS_DIR / "p5_unseen.json"))
    d = V[(V.model == "EN-F4") & V.set.str.startswith("u1")]
    total = d[S["EN-F4"]["features"]].sum(axis=1) + S["EN-F4"]["base_value"]
    pred = np.log10(pd.Series(U["predictions"]["en_physics"])[d.index])
    # stored predictions are rounded to 0.1 cycles -> ~1e-4 in log10 life
    assert np.allclose(total.values, pred.values, atol=5e-4)

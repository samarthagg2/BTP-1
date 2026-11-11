"""Sanity tests for src.features / src.evaluate (run: python3 -m pytest -q tests/)."""
import numpy as np
import pytest

from src.data import curve, load_dataset
from src.evaluate import constant_baseline, mape, rmse
from src.features import ALL_CANDIDATES, FEATURE_SETS, load_features

PAPER_SPLITS = ("train", "primary_test", "secondary_test")


@pytest.fixture(scope="module")
def df():
    return load_features()


def test_feature_sets_are_candidates():
    assert len(FEATURE_SETS["discharge"]) == 6 and len(FEATURE_SETS["full"]) == 9
    from src.features import PHYSICS_FEATURES
    for name, cols in FEATURE_SETS.items():
        assert set(cols) <= set(ALL_CANDIDATES) | (set(PHYSICS_FEATURES) if "physics" in name else set())


def test_one_row_per_cell(df):
    assert len(df) == 163 and df.index.is_unique
    assert df.split.value_counts().to_dict() == {"primary_test": 43, "train": 41,
                                                 "secondary_test": 40, "unseen": 39}


def test_no_missing_in_paper_splits(df):
    d = df[df.split.isin(PAPER_SPLITS)]
    nan = d[ALL_CANDIDATES].isna()
    # only b1c18's IR (stuck logger, cycles 2-53) may be missing
    assert set(d.index[nan.any(axis=1)]) <= {"b1c18"}
    assert np.isfinite(d[ALL_CANDIDATES].fillna(0).values).all()


def test_delta_q_uses_cycles_100_and_10(df):
    c = load_dataset()["cells"]["b1c1"]
    dq = curve(c, "Qdlin", 100) - curve(c, "Qdlin", 10)
    assert df.loc["b1c1", "dq_log_var"] == pytest.approx(np.log10(np.var(dq, ddof=1)))


def test_known_values(df):
    assert df.loc["b2c1", "dq_log_var"] == pytest.approx(-2.73, abs=0.01)
    assert df.loc["b1c1", "dq_log_var"] == pytest.approx(-5.01, abs=0.01)


def test_temperature_integral_is_plausible(df):
    # ~99 cycles x 43-72 min x ~30-36 degC; no pause / corrupt time stamp inflating it
    t = df.loc[df.split.isin(PAPER_SPLITS), "temp_integral_2_100"]
    assert t.between(99 * 40 * 28, 99 * 75 * 38).all()


def test_metrics_and_constant_baseline():
    y = np.array([100.0, 200.0])
    assert rmse(y, y) == 0 and mape(y, [110.0, 180.0]) == pytest.approx(10.0)
    assert constant_baseline([100.0, 1000.0], 1)[0] == pytest.approx(10 ** 2.5)


# ---------------------------------------------------------------- Phase 3 (F4)
import copy  # noqa: E402

from src.features import PHYSICS_FEATURES, margin_rate, overhang_after10  # noqa: E402


def test_physics_features_finite_for_every_cell(df):
    assert FEATURE_SETS["physics"] == PHYSICS_FEATURES
    assert np.isfinite(df[PHYSICS_FEATURES].values).all()
    assert (df.overhang_after10 >= 0).all()


def test_margin_rate_definition(df):
    # log10((QD2 - 0.88) / sd ΔQ) == log10(QD2 - 0.88) - 0.5 * log10 Var ΔQ
    expect = np.log10(df.qd_2 - 0.88) - 0.5 * df.dq_log_var
    assert np.allclose(df.margin_rate, expect, atol=1e-9)


def test_physics_features_use_no_cycle_after_100():
    c = load_dataset()["cells"]["b2c10"]
    before = margin_rate(c), overhang_after10(c)
    late = copy.deepcopy(c)
    s = late["summary"]
    for k in ("QD", "QD_clean", "QC"):
        s[k] = np.where(s["cycle"] > 100, 99.0, s[k])          # absurd values after cycle 100
    rows = np.where(s["cycle"][: late["curves"]["Qdlin"].shape[0]] > 100)[0]
    late["curves"]["Qdlin"][rows] = 99.0
    assert (margin_rate(late), overhang_after10(late)) == pytest.approx(before)

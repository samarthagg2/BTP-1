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
    for cols in FEATURE_SETS.values():
        assert set(cols) <= set(ALL_CANDIDATES)


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

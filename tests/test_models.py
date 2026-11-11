"""Sanity tests for Phase 4 (src.pls, src.models, src.horizon). Run: python3 -m pytest -q tests/"""
import warnings

import numpy as np
import pytest

from src.features import FEATURE_SETS, load_features
from src.horizon import build
from src.models import gp_kernel, gp_predict_std, make_regressor
from src.pls import N_BINS, BlockPLS, design_matrix


@pytest.fixture(scope="module")
def df():
    return load_features()


@pytest.fixture(scope="module")
def train(df):
    return df[df.split == "train"]


def test_horizon_at_100_reproduces_feature_table(df):
    h = build(100)
    cols = [c for c in h.columns if c in df.columns and c not in ("split", "batch", "cycle_life", "anomalous", "temp_suspect")]
    assert len(cols) >= 14
    a, b = h[cols].astype(float), df.loc[h.index, cols].astype(float)
    assert np.allclose(a.values, b.values, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_horizon_excludes_cells_dead_before_n():
    h = build(150)
    dead = h.index[h.cycle_life <= 150]
    assert list(dead) == ["b2c1"] and h.loc["b2c1", "dq_log_var"] != h.loc["b2c1", "dq_log_var"]   # NaN


def test_pls_design_matrices(train):
    for variant, width in (("raw", N_BINS), ("asinh", N_BINS), ("amp_shape", N_BINS + 1), ("f4_shape", N_BINS + 2)):
        X = design_matrix(train, variant)
        assert X.shape == (len(train), width) and np.isfinite(X).all()


def test_block_pls_gives_blocks_their_variance_share(train):
    X = design_matrix(train, "amp_shape")
    m = BlockPLS(n_scalar=1, n_components=2, weight=0.7).fit(X, np.log10(train.cycle_life.to_numpy()))
    Z = m._transform(X)
    share = Z[:, :1].var(0).sum() / Z.var(0).sum()
    assert share == pytest.approx(0.7, abs=1e-6)


def test_gp_kernel_bounds():
    k = gp_kernel(2)
    params = k.get_params()
    assert params["k1__k2__k2__length_scale_bounds"][0] == 1.0     # Matern length-scale >= 1 SD
    assert params["k2__noise_level_bounds"][0] == 1e-2              # noise floor


def test_new_models_predict_finite(train, df):
    cols = FEATURE_SETS["physics"]
    X, y = train[cols].to_numpy(), np.log10(train.cycle_life.to_numpy())
    test = df[df.split == "primary_test"][cols].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind in ("gp", "svr", "gam"):
            p = np.ravel(make_regressor(kind, len(cols)).fit(X, y).predict(test))
            assert p.shape == (len(test),) and np.isfinite(p).all() and (1.5 < p).all() and (p < 4.5).all()
        mu, sd = gp_predict_std(make_regressor("gp", len(cols)).fit(X, y), test)
        assert (sd > 0).all()

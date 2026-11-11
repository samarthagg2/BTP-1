"""P4: partial least squares (PLS) on the whole ΔQ(V) curve.

Why PLS: the ΔQ_100-10(V) curve has 1000 strongly correlated points and there
are only 41 training cells. PLS compresses the curve into a few weighted
averages (components) chosen to predict log10 life, then fits a linear model
on them. Unlike the paper's summary statistics it can use *where* in voltage
the curve changes.

Inputs (curves averaged into N_BINS voltage bins to reduce noise):
  A  'raw'        ΔQ in mAh
  B  'asinh'      asinh(ΔQ / 1 mAh): ~logarithmic for large |ΔQ|, linear near 0,
                  because life follows a power law in the ΔQ amplitude
  C  'amp_shape'  [log10 sd(ΔQ) | ΔQ / sd(ΔQ)]: amplitude and shape blocks
  D  'f4_shape'   [margin_rate, overhang_after10 | ΔQ / sd(ΔQ)]: F4 and shape
For C and D the two blocks are scaled to a chosen share of the total variance
(BlockPLS), otherwise one amplitude column is drowned by 100 shape columns.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold

from src.data import load_dataset
from src.features import delta_q

N_BINS = 100
VARIANTS = ("raw", "asinh", "amp_shape", "f4_shape")
N_SCALAR = {"raw": 0, "asinh": 0, "amp_shape": 1, "f4_shape": 2}


def binned_dq(cell, hi=100, lo=10, n_bins=N_BINS):
    """ΔQ_hi-lo(V) averaged into n_bins consecutive voltage bins (Ah)."""
    return delta_q(cell, hi, lo).reshape(n_bins, -1).mean(1)


def bin_voltages(vdlin, n_bins=N_BINS):
    return vdlin.reshape(n_bins, -1).mean(1)


def design_matrix(df, variant, hi=100):
    """Input matrix for the cells in df (index = cell ids, needs margin_rate/overhang_after10 for 'f4_shape')."""
    cells = load_dataset()["cells"]
    D = np.array([binned_dq(cells[c], hi=hi) for c in df.index])
    sd = D.std(1, ddof=1)
    if variant == "raw":
        return 1000 * D
    if variant == "asinh":
        return np.arcsinh(1000 * D)
    if variant == "amp_shape":
        return np.c_[np.log10(sd), D / sd[:, None]]
    if variant == "f4_shape":
        return np.c_[df[["margin_rate", "overhang_after10"]].to_numpy(), D / sd[:, None]]
    raise ValueError(variant)


class BlockPLS(BaseEstimator, RegressorMixin):
    """PLS on [scalar block | curve block]. The scalar block is z-scored, the curve
    block centred and divided by its RMS norm; the blocks then get `weight` and
    1 - weight of the total variance. With n_scalar = 0 this is plain PLS with
    optional per-column scaling (`scale`)."""

    def __init__(self, n_scalar=0, n_components=2, weight=0.5, scale=False):
        self.n_scalar, self.n_components, self.weight, self.scale = n_scalar, n_components, weight, scale

    def _transform(self, X):
        if self.n_scalar == 0:
            return X
        S, C = X[:, :self.n_scalar], X[:, self.n_scalar:]
        S = (S - self.ms_) / self.ss_ * np.sqrt(self.weight / self.n_scalar)
        C = (C - self.mc_) / self.nc_ * np.sqrt(1 - self.weight)
        return np.c_[S, C]

    def fit(self, X, y):
        X = np.asarray(X, float)
        if self.n_scalar:
            S, C = X[:, :self.n_scalar], X[:, self.n_scalar:]
            self.ms_, self.ss_ = S.mean(0), S.std(0)
            self.mc_ = C.mean(0)
            self.nc_ = np.sqrt(((C - self.mc_) ** 2).sum(1).mean())
        scale = self.scale if self.n_scalar == 0 else False
        self.pls_ = PLSRegression(n_components=self.n_components, scale=scale).fit(self._transform(X), y)
        return self

    def predict(self, X):
        return self.pls_.predict(self._transform(np.asarray(X, float))).ravel()


def param_grid(variant):
    if N_SCALAR[variant] == 0:
        return {"n_components": list(range(1, 11)), "scale": [False, True]}
    return {"n_components": list(range(1, 8)), "weight": [0.5, 0.7, 0.9, 0.97, 0.99]}


def make_pls(variant, seed=1, n_repeats=5):
    """GridSearchCV over the PLS settings with repeated 4-fold CV (training cells only)."""
    return GridSearchCV(BlockPLS(n_scalar=N_SCALAR[variant]), param_grid(variant),
                        cv=RepeatedKFold(n_splits=4, n_repeats=n_repeats, random_state=seed),
                        scoring="neg_root_mean_squared_error")

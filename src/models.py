"""P4: regression models other than the paper's elastic net (small-data friendly, no trees).

All models predict log10(cycle life) from z-scored features. Settings are chosen
on the training cells only.
  gp   Gaussian process: kernel = c1 * DotProduct (linear trend, as log life is
       close to linear in the log features)  +  c2 * Matern(nu=2.5, one length-scale
       per feature) (smooth non-linear deviations)  +  White noise. Hyper-parameters
       by marginal likelihood (no grid search); gives a predictive standard deviation.
       Bounds chosen on the training cells (unbounded, the GP interpolated the 41
       cells exactly): length-scales >= 1 SD of the z-scored feature (better or equal
       nested CV for F2 and F4) and a noise floor of 0.01 in normalised variance
       (~0.016 log10 life, about a third of the elastic net's CV residual).
  svr  Support vector regression, RBF kernel; C, gamma, epsilon by repeated 4-fold CV.
  gam  Additive spline model: one cubic spline per feature (linear extrapolation
       outside the training range) + ridge; knots and ridge strength by repeated CV.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVR

NEW_MODELS = ("gp", "svr", "gam")


def _cv(seed):
    return RepeatedKFold(n_splits=4, n_repeats=5, random_state=seed)


def gp_kernel(n_features):
    return (ConstantKernel(0.1, (1e-4, 1e2)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e2))
            + ConstantKernel(0.01, (1e-5, 1e1)) * Matern(length_scale=3 * np.ones(n_features),
                                                          length_scale_bounds=(1.0, 1e3), nu=2.5)
            + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-2, 1e0)))


def make_regressor(kind, n_features, seed=42):
    pre = [SimpleImputer(strategy="median"), StandardScaler()]
    if kind == "gp":
        return make_pipeline(*pre, GaussianProcessRegressor(kernel=gp_kernel(n_features), normalize_y=True,
                                                            n_restarts_optimizer=10, random_state=seed))
    if kind == "svr":
        grid = {"svr__C": [0.3, 1, 3, 10, 30, 100], "svr__gamma": ["scale", 0.01, 0.03, 0.1, 0.3],
                "svr__epsilon": [0.01, 0.03, 0.1]}
        return GridSearchCV(make_pipeline(*pre, SVR(kernel="rbf")), grid, cv=_cv(seed),
                            scoring="neg_root_mean_squared_error")
    if kind == "gam":
        grid = {"splinetransformer__n_knots": [3, 4, 5], "ridge__alpha": list(np.logspace(-3, 2, 11))}
        pipe = make_pipeline(*pre, SplineTransformer(degree=3, extrapolation="linear"), Ridge())
        return GridSearchCV(pipe, grid, cv=_cv(seed), scoring="neg_root_mean_squared_error")
    raise ValueError(kind)


def gp_predict_std(model, X):
    """Mean and standard deviation (log10 life) of a fitted GP pipeline."""
    *pre, gp = model.steps
    Z = X
    for _, step in pre:
        Z = step.transform(Z)
    mu, sd = gp[1].predict(Z, return_std=True)
    return mu, sd

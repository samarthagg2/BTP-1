"""Metrics and naive baselines. Metrics are computed on CYCLES (models predict log10 life)."""
import numpy as np


def rmse(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def mape(y, yhat):
    """Mean absolute percentage error (%), the paper's 'mean percent error'."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(yhat - y) / y) * 100)


def r2_log(y, yhat):
    ly, lh = np.log10(y), np.log10(yhat)
    return float(1 - np.sum((ly - lh) ** 2) / np.sum((ly - ly.mean()) ** 2))


def scores(y, yhat):
    return {"rmse": rmse(y, yhat), "mape": mape(y, yhat), "r2_log": r2_log(y, yhat), "n": int(len(y))}


def constant_baseline(train_life, n):
    """SI Note 3 'Constant': one value for every cell = the training mean of log10(life),
    i.e. the geometric mean (an intercept-only model on the log target). This reproduces
    SI Table 3 exactly (327/401/399/511 cycles); the arithmetic mean does not."""
    return np.full(n, float(10 ** np.mean(np.log10(train_life))))


def policy_baseline(train_policy, train_life, policies):
    """Geometric-mean training life of cells with the same charging policy; falls back
    to the constant baseline for policies not seen in training."""
    train_policy = [p.replace("-newstructure", "") for p in train_policy]
    lut = {}
    for p, l in zip(train_policy, train_life):
        lut.setdefault(p, []).append(l)
    gmean = lambda v: float(10 ** np.mean(np.log10(v)))
    fallback = gmean(train_life)
    return np.array([gmean(lut[p]) if p in lut else fallback
                     for p in (q.replace("-newstructure", "") for q in policies)])

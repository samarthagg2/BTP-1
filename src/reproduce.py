"""P2: reproduce Severson et al. 2019 Table 1 (variance / discharge / full models).

Protocol (paper Methods + SI):
  * target = log10(cycle life); fit + tune on the 41 training cells only
  * 'variance': OLS on one feature (no model selection)
  * 'discharge', 'full': z-score + elastic net, alpha & l1_ratio by repeated 4-fold CV
  * missing feature values (b1c18 IR, stuck logger) -> training-set median
  * 'full' excludes the 4 cells whose temperature probe lost contact (temp_suspect)
  * evaluated once on primary test (with / without the anomalous cell) and secondary test
Usage: python3 -m src.reproduce            -> results/p2_reproduction.json
"""
import json
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import RESULTS_DIR, SEED
from src.evaluate import constant_baseline, policy_baseline, scores
from src.features import FEATURE_SETS, load_features

L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

# Paper Table 1 (+ SI Table 3 'constant'): rmse/mape for train, primary, primary excl. anomaly, secondary
PAPER = {
    "constant":  {"train": (327, 29.6), "primary": (401, 34.9), "primary_excl": (399, 28.2), "secondary": (511, 36.1)},
    "variance":  {"train": (103, 14.1), "primary": (138, 14.7), "primary_excl": (138, 13.2), "secondary": (196, 11.4)},
    "discharge": {"train": (76, 9.8),   "primary": (91, 13.0),  "primary_excl": (86, 10.1),  "secondary": (173, 8.6)},
    "full":      {"train": (51, 5.6),   "primary": (118, 14.1), "primary_excl": (100, 7.5),  "secondary": (214, 10.7)},
}
# SI Fig. 5: full-model coefficients on z-scored features
PAPER_FULL_COEF = {"qd_2": 0.0218, "temp_integral_2_100": -0.0359, "chargetime_avg_2_6": 0.0392,
                   "dq_log_min": -0.0293, "dq_log_var": -0.0714, "qd_slope_2_100": -0.0127,
                   "qd_icpt_2_100": 2.96e-6, "ir_min_2_100": -0.00711, "ir_100_minus_2": -0.0132}


def make_model(name, seed=SEED):
    if name == "variance":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
    cv = RepeatedKFold(n_splits=4, n_repeats=10, random_state=seed)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         ElasticNetCV(l1_ratio=L1_RATIOS, n_alphas=100, cv=cv, max_iter=200000))


def uses_temperature(name):
    """Sets containing the temperature integral leave out the 4 cells whose probe lost contact (paper)."""
    return "temp_integral_2_100" in FEATURE_SETS[name]


def subsets(df, name):
    keep = ~df.temp_suspect if uses_temperature(name) else np.ones(len(df), bool)
    d = df[keep]
    return {"train": d[d.split == "train"],
            "primary": d[d.split == "primary_test"],
            "primary_excl": d[(d.split == "primary_test") & ~d.anomalous],
            "secondary": d[d.split == "secondary_test"]}


def fit_one(df, name, seed=SEED):
    cols = FEATURE_SETS[name]
    parts = subsets(df, name)
    tr = parts["train"]
    assert not tr[cols].isna().any().any(), "NaN in training features"
    model = make_model(name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(tr[cols].values, np.log10(tr.cycle_life.values))
    res = {}
    for k, d in parts.items():
        pred = 10 ** model.predict(d[cols].values)
        res[k] = scores(d.cycle_life.values, pred)
    reg = model[-1]
    info = {"coef": dict(zip(cols, map(float, reg.coef_))), "intercept": float(reg.intercept_)}
    if name != "variance":
        info.update(alpha=float(reg.alpha_), l1_ratio=float(reg.l1_ratio_),
                    n_selected=int(np.sum(np.abs(reg.coef_) > 1e-12)))
    return model, res, info


def baselines(df):
    parts = subsets(df, "variance")
    tr = parts["train"]
    out = {"constant": {}, "policy_mean": {}}
    for k, d in parts.items():
        out["constant"][k] = scores(d.cycle_life, constant_baseline(tr.cycle_life, len(d)))
        out["policy_mean"][k] = scores(d.cycle_life,
                                       policy_baseline(tr.policy, tr.cycle_life, d.policy))
    return out


def predictions(df, model, name):
    cols = FEATURE_SETS[name]
    d = df[~df.temp_suspect] if uses_temperature(name) else df
    d = d[d.split != "unseen"]
    return {cid: float(10 ** p) for cid, p in zip(d.index, model.predict(d[cols].values))}


def run(n_seeds=10, verbose=True):
    df = load_features()
    out = {"baselines": baselines(df), "models": {}, "stability": {}, "predictions": {}}
    for name in ("variance", "discharge", "full"):
        model, res, info = fit_one(df, name)
        out["models"][name] = {"scores": res, **info}
        out["predictions"][name] = predictions(df, model, name)
        if name != "variance":
            runs = [fit_one(df, name, seed=s)[1] for s in range(n_seeds)]
            out["stability"][name] = {
                k: {m: [float(np.min([r[k][m] for r in runs])), float(np.max([r[k][m] for r in runs]))]
                    for m in ("rmse", "mape")} for k in runs[0]}
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "p2_reproduction.json", "w") as fp:
        json.dump(out, fp, indent=2)
    if verbose:
        report(out)
    return out


def report(out):
    keys = ("train", "primary", "primary_excl", "secondary")
    print(f"{'model':10s} " + " ".join(f"{k:>22s}" for k in keys))
    rows = [("constant", out["baselines"]["constant"]), ("policy_mean", out["baselines"]["policy_mean"])]
    rows += [(n, out["models"][n]["scores"]) for n in ("variance", "discharge", "full")]
    for n, sc in rows:
        cells = []
        for k in keys:
            ours = f"{sc[k]['rmse']:.0f}/{sc[k]['mape']:.1f}%"
            p = PAPER.get(n, {}).get(k)
            cells.append(f"{ours} ({p[0]}/{p[1]}%)" if p else ours)
        print(f"{n:10s} " + " ".join(f"{c:>22s}" for c in cells))
    print("format: ours RMSE/MAPE (paper RMSE/MAPE)")
    for n in ("discharge", "full"):
        m = out["models"][n]
        print(f"\n{n}: alpha={m['alpha']:.4g} l1_ratio={m['l1_ratio']} selected {m['n_selected']}/{len(m['coef'])}")
        for f, w in m["coef"].items():
            pw = PAPER_FULL_COEF.get(f) if n == "full" else None
            print(f"   {f:22s} {w:+.4f}" + (f"   paper {pw:+.4f}" if pw is not None else ""))
        st = out["stability"][n]
        print("   CV-seed spread (10 seeds): " +
              ", ".join(f"{k} RMSE {v['rmse'][0]:.0f}-{v['rmse'][1]:.0f}" for k, v in st.items()))
    v = out["models"]["variance"]
    print(f"\nvariance: coef {v['coef']['dq_log_var']:+.4f} (z-scored)")


if __name__ == "__main__":
    run(n_seeds=int(sys.argv[1]) if len(sys.argv) > 1 else 10)

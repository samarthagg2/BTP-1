"""P4: PLS on the whole ΔQ(V) curve vs the paper's elastic net.

Pre-registered (fixed from training-cell nested CV, before any test cell was used):
  PRIMARY   = 'asinh'    (best whole-curve model: PLS on asinh(ΔQ / 1 mAh))
  SECONDARY = 'f4_shape' (does the curve shape add anything to F4?)
Both are evaluated once on the paper's splits; the other variants are reported
on the training cells only. The unseen batch is not used (Phase 5).

Usage: python3 -m src.p4_pls      -> results/p4_pls.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold

from config import RESULTS_DIR, VDLIN
from src.evaluate import scores
from src.features import FEATURE_SETS, load_features
from src.p3_evaluate import bootstrap, predictions, robust
from src.pls import VARIANTS, bin_voltages, design_matrix, make_pls
from src.reproduce import fit_one, make_model

PRIMARY, SECONDARY = "asinh", "f4_shape"


def nested_cv(df, n_repeats=10, seed=7):
    """Nested CV on the 41 training cells (identical outer folds for every model)."""
    tr = df[df.split == "train"]
    y = np.log10(tr.cycle_life.to_numpy())
    outer = list(RepeatedKFold(n_splits=4, n_repeats=n_repeats, random_state=seed).split(y))
    makers = {f"PLS {v}": (lambda v=v: make_pls(v), design_matrix(tr, v)) for v in VARIANTS}
    makers["OLS log sd(ΔQ)"] = (LinearRegression, design_matrix(tr, "amp_shape")[:, :1])
    makers["OLS F4"] = (LinearRegression, tr[FEATURE_SETS["physics"]].to_numpy())
    makers["EN discharge (F2)"] = (lambda: make_model("discharge"), tr[FEATURE_SETS["discharge"]].to_numpy())
    makers["EN F4"] = (lambda: make_model("physics"), tr[FEATURE_SETS["physics"]].to_numpy())
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (make, X) in makers.items():
            e = []
            for a, b in outer:
                m = make().fit(X[a], y[a])
                e.append(float(np.sqrt(np.mean((np.ravel(m.predict(X[b])) - y[b]) ** 2))))
            out[name] = {"rmse_log": float(np.mean(e)), "folds": e}
    return out


def test_eval(df):
    tr = df[df.split == "train"]
    y = np.log10(tr.cycle_life.to_numpy())
    paper = df[df.split != "unseen"]
    lo, hi = float(tr.cycle_life.min()), float(tr.cycle_life.max())
    res, preds, info = {}, {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for v in (PRIMARY, SECONDARY):
            gs = make_pls(v).fit(design_matrix(tr, v), y)
            p = pd.Series(10 ** gs.predict(design_matrix(paper, v)), index=paper.index)
            preds[f"pls_{v}"] = p
            info[f"pls_{v}"] = {"params": gs.best_params_}
            if v == PRIMARY:     # regression coefficients on the (unscaled) asinh curve, per voltage bin
                pls = gs.best_estimator_.pls_
                info[f"pls_{v}"]["coef_by_voltage"] = {"V": bin_voltages(VDLIN).round(4).tolist(),
                                                       "coef": np.ravel(pls.coef_).tolist()}
        for name in ("discharge", "physics"):
            preds[f"en_{name}"] = predictions(df, fit_one(df, name)[0], name)
    parts = {"train": paper.split == "train", "primary": paper.split == "primary_test",
             "primary_excl": (paper.split == "primary_test") & ~paper.anomalous,
             "secondary": paper.split == "secondary_test"}
    for k, p in preds.items():
        res[k] = {"scores": {s: scores(paper.cycle_life[m], p[paper.index[m]]) for s, m in parts.items()},
                  "robust": robust(df, p, lo, hi), **info.get(k, {})}
    res["bootstrap"] = bootstrap(df, preds, lo, hi, [("pls_asinh", "en_discharge"), ("pls_asinh", "en_physics"),
                                                     ("pls_f4_shape", "en_physics")])
    return res


def run(verbose=True):
    df = load_features()
    out = {"preregistered": {"primary": PRIMARY, "secondary": SECONDARY},
           "nested_cv_train": nested_cv(df), "test": test_eval(df)}
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "p4_pls.json", "w") as fp:
        json.dump(out, fp, indent=2)
    if verbose:
        print("nested CV on training cells (RMSE log10 life):")
        for k, v in out["nested_cv_train"].items():
            print(f"   {k:20s} {v['rmse_log']:.4f}")
        print("\ntest (RMSE cycles / % error):")
        for k, v in out["test"].items():
            if k == "bootstrap":
                continue
            s, r = v["scores"], v["robust"]
            print(f"   {k:14s} " + "  ".join(f"{q} {s[q]['rmse']:.0f}/{s[q]['mape']:.1f}%" for q in ("train", "primary", "primary_excl", "secondary"))
                  + f"  | in-range primary {r['primary']['rmse_in_range']:.0f} | median % {r['primary']['median_pct_err']:.1f}/{r['secondary']['median_pct_err']:.1f}"
                  + (f"  {v['params']}" if "params" in v else ""))
        for key, b in out["test"]["bootstrap"].items():
            print(f"   bootstrap {key}: " + "  ".join(f"{p} {w['rmse_diff']:+.0f} [{w['rmse_ci'][0]:+.0f},{w['rmse_ci'][1]:+.0f}]"
                                                    for p, w in b.items() if "-" in p))
    return out


if __name__ == "__main__":
    run()

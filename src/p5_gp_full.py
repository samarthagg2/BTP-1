"""EXPLORATORY (added after all test sets were used): Gaussian process on F3 + F4.

Not pre-registered. The primary, secondary and unseen sets had all been used
before this configuration was chosen, so its scores are an after-the-fact check,
not a clean test. Same GP as src/models.py (linear + Matern ARD + noise, bounded),
trained on the training cells without temperature-probe cells.

Usage: python3 -m src.p5_gp_full      -> results/p5_gp_full.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from config import RESULTS_DIR
from src.evaluate import scores
from src.features import FEATURE_SETS, load_features
from src.models import gp_predict_std, make_regressor
from src.reproduce import fit_one

CONFIGS = {"GP-F3+F4": ("gp", "full_physics"), "EN-F3+F4": ("en", "full_physics"), "GP-F4": ("gp", "physics"),
           "EN-F3": ("en", "full")}


def run(verbose=True):
    df = load_features()
    tr_all = df[df.split == "train"]
    u = df[df.split == "unseen"]
    ns = u.policy.str.contains("newstructure")
    sets = {"primary": df[(df.split == "primary_test")], "primary_excl": df[(df.split == "primary_test") & ~df.anomalous],
            "secondary": df[df.split == "secondary_test"], "u1_all": u, "u1_standard": u[~ns], "u1_newstructure": u[ns]}
    lo, hi = tr_all.cycle_life.min(), tr_all.cycle_life.max()
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (kind, fs) in CONFIGS.items():
            cols = FEATURE_SETS[fs]
            temp = "temp_integral_2_100" in cols
            tr = tr_all[~tr_all.temp_suspect] if temp else tr_all
            y = np.log10(tr.cycle_life.to_numpy())
            X = tr[cols].to_numpy()
            make = (lambda: make_regressor("gp", len(cols))) if kind == "gp" else (lambda: fit_one(df, fs)[0])
            model = make().fit(X, y) if kind == "gp" else fit_one(df, fs)[0]
            # nested CV on the training cells (identical outer folds)
            e = []
            for a, b in RepeatedKFold(n_splits=4, n_repeats=10, random_state=7).split(y):
                if kind == "gp":
                    m = make_regressor("gp", len(cols)).fit(X[a], y[a])
                else:
                    from src.reproduce import make_model
                    m = make_model(fs).fit(X[a], y[a])
                e.append(np.sqrt(np.mean((np.ravel(m.predict(X[b])) - y[b]) ** 2)))
            res = {"nested_cv_log": float(np.mean(e)),
                   "train": scores(tr.cycle_life, 10 ** np.ravel(model.predict(X)))}
            for k, d in sets.items():
                d = d[~d.temp_suspect] if temp else d
                p = 10 ** np.ravel(model.predict(d[cols].to_numpy()))
                s = scores(d.cycle_life, p)
                s["signed_pct"] = float(np.mean((p - d.cycle_life) / d.cycle_life) * 100)
                if k == "primary":
                    inr = ((d.cycle_life >= lo) & (d.cycle_life <= hi)).to_numpy()
                    s["rmse_in_range"] = float(np.sqrt(np.mean((p[inr] - d.cycle_life.to_numpy()[inr]) ** 2)))
                if kind == "gp":
                    mu, sd = gp_predict_std(model, d[cols].to_numpy())
                    s["coverage_95"] = float(np.mean(np.abs(np.log10(d.cycle_life.to_numpy()) - mu) <= 1.96 * sd))
                res[k] = s
            if kind == "gp":
                res["kernel"] = str(model[-1].kernel_)
            out[name] = res
    with open(RESULTS_DIR / "p5_gp_full.json", "w") as fp:
        json.dump(out, fp, indent=2)
    if verbose:
        for name, r in out.items():
            line = f"{name:9s} CV {r['nested_cv_log']:.4f} | train {r['train']['rmse']:.0f}"
            for k in ("primary", "secondary", "u1_all", "u1_standard", "u1_newstructure"):
                s = r[k]
                line += f" | {k} {s['rmse']:.0f}/{s['mape']:.1f}%" + (f" (in-range {s['rmse_in_range']:.0f})" if "rmse_in_range" in s else "") \
                        + (f" cov {s['coverage_95']:.2f}" if "coverage_95" in s else "")
            print(line)
        print("GP-F3+F4 kernel:", out["GP-F3+F4"]["kernel"])
    return out


if __name__ == "__main__":
    run()

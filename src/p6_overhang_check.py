"""P6 add-on: does overhang_after10 (OH) help when added to the paper's sets F1 and F2?

Exploratory (the test sets had already been used): elastic net with and without OH, nested CV on the
41 training cells (outer 4-fold x 5, seed 3) and one evaluation on primary, secondary and u1, with paired
bootstrap intervals (5000 resamples). Result: OH helps within the training batches only, so it is not used.

Usage: python3 -m src.p6_overhang_check   -> results/p6_overhang_check.json
"""
import json
import warnings

import numpy as np
from sklearn.model_selection import RepeatedKFold

from config import RESULTS_DIR
from src.features import FEATURE_SETS, load_features
from src.reproduce import make_model

OH = ["overhang_after10"]
SETS = {"F1": FEATURE_SETS["variance"], "F1+OH": FEATURE_SETS["variance"] + OH,
        "F2": FEATURE_SETS["discharge"], "F2+OH": FEATURE_SETS["discharge"] + OH}


def _model(cols):
    return make_model("variance" if len(cols) == 1 else "en")


def run(verbose=True):
    df = load_features()
    tr = df[df.split == "train"]
    lo, hi = tr.cycle_life.min(), tr.cycle_life.max()
    y = np.log10(tr.cycle_life.to_numpy())
    tests = {"primary": df[df.split == "primary_test"], "secondary": df[df.split == "secondary_test"],
             "u1": df[df.split == "unseen"]}
    out, preds = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, cols in SETS.items():
            X = tr[cols].to_numpy()
            e = [float(np.sqrt(np.mean((_model(cols).fit(X[a], y[a]).predict(X[b]) - y[b]) ** 2)))
                 for a, b in RepeatedKFold(n_splits=4, n_repeats=5, random_state=3).split(X)]
            m = _model(cols).fit(X, y)
            preds[name] = {k: 10 ** m.predict(d[cols].to_numpy()) for k, d in tests.items()}
            r = {k: float(np.sqrt(np.mean((preds[name][k] - d.cycle_life.to_numpy()) ** 2))) for k, d in tests.items()}
            d = tests["primary"]; inr = ((d.cycle_life >= lo) & (d.cycle_life <= hi)).to_numpy()
            r["primary_in_range"] = float(np.sqrt(np.mean((preds[name]["primary"][inr] - d.cycle_life.to_numpy()[inr]) ** 2)))
            out[name] = {"cv_rmse_log": float(np.mean(e)), "rmse": r,
                         "coef_oh": float(m[-1].coef_[-1]) if "overhang_after10" in cols else None}
    rng = np.random.default_rng(0)
    boot = {}
    for a, b in (("F1+OH", "F1"), ("F2+OH", "F2")):
        boot[f"{a} - {b}"] = {}
        for k, d in tests.items():
            yy = d.cycle_life.to_numpy()
            msk = ((yy >= lo) & (yy <= hi)) if k == "primary" else np.ones(len(yy), bool)
            pa, pb, yy = preds[a][k][msk], preds[b][k][msk], yy[msk]
            idx = rng.integers(0, len(yy), size=(5000, len(yy)))
            diff = np.sqrt(((pa - yy)[idx] ** 2).mean(1)) - np.sqrt(((pb - yy)[idx] ** 2).mean(1))
            boot[f"{a} - {b}"][k + ("_in_range" if k == "primary" else "")] = [
                float(diff.mean()), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    res = {"sets": out, "bootstrap": boot}
    with open(RESULTS_DIR / "p6_overhang_check.json", "w") as fp:
        json.dump(res, fp, indent=2)
    if verbose:
        for k, v in out.items():
            print(f"{k:6s} CV {v['cv_rmse_log']:.4f}  " + "  ".join(f"{q} {x:.0f}" for q, x in v["rmse"].items()))
        for k, v in boot.items():
            print(k, {q: [round(x) for x in iv] for q, iv in v.items()})
    return res


if __name__ == "__main__":
    run()

"""P3: test-set performance of the F4 physics set with the paper's model.

F4 was fixed on the training cells (src/screen_physics.py) before this script
was run; it must not be changed in response to these numbers.

For the paper's three sets, F4 and F3 + F4 (elastic net / OLS, protocol of
src/reproduce.py) it reports:
  * the official scores (RMSE in cycles, mean absolute % error) on train,
    primary, primary without the anomalous cell, and secondary;
  * the spread of test RMSE over 10 cross-validation seeds;
  * scores on the test cells whose observed life lies inside the training range
    (a model-independent rule: extrapolation is a separate question), and the
    median and 90th-percentile % error (robust, no exclusion);
  * paired bootstrap intervals for the RMSE / % error differences between
    discharge, full and F4, on identical cells.
The unseen batch (u1) is not used (Phase 5).

Usage: python3 -m src.p3_evaluate     -> results/p3_test_eval.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from config import RESULTS_DIR
from src.evaluate import constant_baseline, scores
from src.features import FEATURE_SETS, load_features
from src.reproduce import fit_one, uses_temperature

MODELS = ("variance", "discharge", "full", "physics", "full_physics")
SPLITS = {"primary": "primary_test", "secondary": "secondary_test"}
N_BOOT = 5000


def predictions(df, model, name):
    d = df[~df.temp_suspect] if uses_temperature(name) else df
    d = d[d.split != "unseen"]
    return pd.Series(10 ** model.predict(d[FEATURE_SETS[name]].to_numpy()), index=d.index)


def robust(df, pred, lo, hi):
    out = {}
    for key, split in SPLITS.items():
        d = df[(df.split == split) & df.index.isin(pred.index)]
        e = pred[d.index] - d.cycle_life
        pe = 100 * e.abs() / d.cycle_life
        inr = (d.cycle_life >= lo) & (d.cycle_life <= hi)
        out[key] = {"n": int(len(d)), "n_in_range": int(inr.sum()),
                    "out_of_range_cells": sorted(d.index[~inr]),
                    "rmse_in_range": float(np.sqrt((e[inr] ** 2).mean())), "mape_in_range": float(pe[inr].mean()),
                    "median_pct_err": float(pe.median()), "p90_pct_err": float(pe.quantile(0.9))}
    return out


def bootstrap(df, preds, lo, hi, pairs, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for key, split in SPLITS.items():
        d = df[(df.split == split) & ~df.temp_suspect]           # same cells for every model
        d = d[(d.cycle_life >= lo) & (d.cycle_life <= hi)]
        y = d.cycle_life.to_numpy()
        err = {k: preds[k][d.index].to_numpy() - y for k in preds}
        idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
        rmse = {k: np.sqrt((err[k][idx] ** 2).mean(1)) for k in err}
        pct = {k: (100 * np.abs(err[k]) / y)[idx].mean(1) for k in err}
        res = {"n": int(len(d)), "rmse": {k: [float(np.sqrt((err[k] ** 2).mean())),
                                              float(np.percentile(rmse[k], 2.5)), float(np.percentile(rmse[k], 97.5))]
                                          for k in err}}
        for a, b in pairs:
            dr, dp = rmse[a] - rmse[b], pct[a] - pct[b]
            res[f"{a}-{b}"] = {"rmse_diff": float(dr.mean()), "rmse_ci": [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))],
                               "a_better_pct": float(100 * np.mean(dr < 0)),
                               "pct_err_diff": float(dp.mean()), "pct_err_ci": [float(np.percentile(dp, 2.5)), float(np.percentile(dp, 97.5))]}
        out[key] = res
    return out


def run(verbose=True):
    df = load_features()
    tr = df[df.split == "train"]
    lo, hi = float(tr.cycle_life.min()), float(tr.cycle_life.max())
    out = {"train_life_range": [lo, hi], "models": {}}
    preds = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        const = pd.Series(constant_baseline(tr.cycle_life, 1)[0], index=df.index[df.split != "unseen"])
        out["models"]["constant"] = {"robust": robust(df, const, lo, hi)}
        for name in MODELS:
            model, res, info = fit_one(df, name)
            preds[name] = predictions(df, model, name)
            m = {"n_features": len(FEATURE_SETS[name]), "scores": res, "robust": robust(df, preds[name], lo, hi),
                 "coef": info["coef"]}
            if name != "variance":
                runs = [fit_one(df, name, seed=s)[1] for s in range(10)]
                m["seed_spread_rmse"] = {k: [float(min(r[k]["rmse"] for r in runs)), float(max(r[k]["rmse"] for r in runs))]
                                         for k in ("primary", "secondary")}
                m.update(alpha=info["alpha"], l1_ratio=info["l1_ratio"])
            out["models"][name] = m
    out["bootstrap"] = bootstrap(df, {k: preds[k] for k in ("discharge", "full", "physics")}, lo, hi,
                                 [("physics", "discharge"), ("physics", "full"), ("full", "discharge")])
    prim = df[df.split == "primary_test"]
    out["largest_primary_errors_physics"] = {
        c: {"life": float(prim.cycle_life[c]), **{k: float(preds[k][c] - prim.cycle_life[c]) for k in ("discharge", "full", "physics") if c in preds[k]}}
        for c in (preds["physics"][prim.index] - prim.cycle_life).abs().sort_values(ascending=False).index[:3]}
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "p3_test_eval.json", "w") as fp:
        json.dump(out, fp, indent=2)
    if verbose:
        report(out)
    return out


def report(out):
    print(f"training life range {out['train_life_range']}")
    for name, m in out["models"].items():
        if "scores" in m:
            s = m["scores"]
            print(f"{name:13s} " + "  ".join(f"{k} {s[k]['rmse']:.0f}/{s[k]['mape']:.1f}%" for k in ("train", "primary", "primary_excl", "secondary"))
                  + (f"   seeds {m['seed_spread_rmse']}" if "seed_spread_rmse" in m else ""))
        r = m["robust"]
        print(f"{'':13s} in-range primary {r['primary']['rmse_in_range']:.0f}/{r['primary']['mape_in_range']:.1f}% (n={r['primary']['n_in_range']}); "
              f"median %err {r['primary']['median_pct_err']:.1f} / {r['secondary']['median_pct_err']:.1f}")
    for key, b in out["bootstrap"].items():
        print(key, {k: (round(v["rmse_diff"], 1), [round(x) for x in v["rmse_ci"]], round(v["a_better_pct"])) for k, v in b.items() if "-" in k})


if __name__ == "__main__":
    run()

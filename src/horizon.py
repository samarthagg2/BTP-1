"""RQ4: prediction error against the number of early cycles N.

Recomputes the paper's features (SI Table 1, as in src/features.py) and the
F4 physics features using cycles <= N instead of <= 100:
  ΔQ = Q_N(V) - Q_10(V); capacity fit over cycles 2..N; temperature integral
  and minimum IR over 2..N; IR(N) - IR(2); margin_rate with ΔQ_N-10;
  overhang_after10 searched over 2..N. Charge time (cycles 2-6) is unchanged.
For each N every model is re-tuned on the training cells only (protocol of
src/reproduce.py). Cells that reached end of life at or before cycle N are
excluded (their life is already known); at N <= 147 none is.

Usage: python3 -m src.horizon [N ...]     -> results/horizon.json (default N = 20..150)
"""
import json
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from config import N_CACHE, PROC_DIR, RESULTS_DIR
from src.data import load_dataset, load_splits
from src.evaluate import scores
from src.features import (FEATURE_SETS, N_LO, _linfit, _log_abs, _window, delta_q,
                          margin_rate, overhang_after10, value_at)
from src.models import make_regressor
from src.reproduce import make_model, uses_temperature

# elastic-net / OLS sets of the paper and F4, plus the P4 Gaussian process on F4
MODELS = ("variance", "discharge", "full", "physics", "gp_physics")


def _spec(name, seed):
    """(feature-set name, model factory) for a horizon model."""
    if name.startswith("gp_"):
        fs = name[3:]
        return fs, lambda: make_regressor("gp", len(FEATURE_SETS[fs]), seed)
    return name, lambda: make_model(name, seed)
N_BOOT = 5000


def features_at(cell, extras, n):
    dq = delta_q(cell, hi=n, lo=N_LO)
    dev = dq - dq.mean()
    m2, m3, m4 = (dev ** 2).mean(), (dev ** 3).mean(), (dev ** 4).mean()
    slope, icpt = _linfit(cell, 2, n)
    _, q = _window(cell, "QD", 2, n)
    q2 = value_at(cell, "QD", 2, hi=n)
    _, ct = _window(cell, "chargetime", 2, 6)
    _, ir = _window(cell, "IR", 2, n)
    m = (extras["cycle"] >= 2) & (extras["cycle"] <= n)
    tint = extras["T_int"][m]
    with np.errstate(all="ignore"):
        return {
            "dq_log_min": _log_abs(dq.min()), "dq_log_var": _log_abs(np.var(dq, ddof=1)),
            "dq_log_skew": _log_abs(m3 / m2 ** 1.5), "dq_log_kurt": _log_abs(m4 / m2 ** 2),
            "qd_slope_2_100": slope, "qd_icpt_2_100": icpt, "qd_2": q2,
            "qd_max_minus_2": float(np.nanmax(q)) - q2,
            "chargetime_avg_2_6": float(np.nanmean(ct)) if np.isfinite(ct).any() else np.nan,
            "temp_integral_2_100": float(np.nansum(tint)) if np.isfinite(tint).all() else np.nan,
            "ir_min_2_100": float(np.nanmin(ir)) if np.isfinite(ir).any() else np.nan,
            "ir_100_minus_2": value_at(cell, "IR", n, hi=n) - value_at(cell, "IR", 2, hi=n),
            "margin_rate": margin_rate(cell, hi=n), "overhang_after10": overhang_after10(cell, n_max=n),
        }


def build(n):
    """Feature table at horizon n (column names as in features.csv; '_2_100' means '_2_n')."""
    assert n + 3 <= N_CACHE, f"cache holds {N_CACHE} cycles; rebuild with a larger N_CACHE"
    ds, sp = load_dataset(), load_splits()
    extras = pickle.load(open(PROC_DIR / "extras.pkl", "rb"))
    rows = []
    for split in ("train", "primary_test", "secondary_test"):
        for cid in sp[split]:
            c = ds["cells"][cid]
            alive = c["cycle_life"] > n                          # dead cells: life already known, no features
            rows.append({"cell": cid, "split": split, "batch": c["batch"], "cycle_life": float(c["cycle_life"]),
                         "anomalous": cid in sp["anomalous"], "temp_suspect": bool(c["temp_suspect"]),
                         **(features_at(c, extras[cid], n) if alive else {})})
    return pd.DataFrame(rows).set_index("cell")


def evaluate(n, seed=42):
    df = build(n)
    alive = df.cycle_life > n
    out = {"n": n, "dead_before_n": sorted(df.index[~alive]), "models": {}, "pred": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        for name in MODELS:
            fs, make = _spec(name, seed)
            cols = FEATURE_SETS[fs]
            d = df[alive & ~df.temp_suspect] if uses_temperature(fs) else df[alive]
            tr = d[d.split == "train"]
            model = make().fit(tr[cols].to_numpy(), np.log10(tr.cycle_life.to_numpy()))
            pred = lambda v: 10 ** np.ravel(model.predict(v[cols].to_numpy()))
            parts = {"train": tr, "primary_excl": d[(d.split == "primary_test") & ~d.anomalous],
                     "secondary": d[d.split == "secondary_test"]}
            out["models"][name] = {k: scores(v.cycle_life.to_numpy(), pred(v)) for k, v in parts.items()}
            out["pred"][name] = dict(zip(d.index, pred(d).tolist()))
    return out


def paired_change(res_a, res_b, df, seed=0):
    """Bootstrap of RMSE(b) - RMSE(a) for each model on the cells both horizons predict."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in MODELS:
        pa, pb = res_a["pred"][name], res_b["pred"][name]
        out[name] = {}
        for key, split in (("primary_excl", "primary_test"), ("secondary", "secondary_test")):
            cells = [c for c in pa if c in pb and df.loc[c, "split"] == split and not df.loc[c, "anomalous"]]
            y = df.loc[cells, "cycle_life"].to_numpy()
            ea = np.array([pa[c] for c in cells]) - y
            eb = np.array([pb[c] for c in cells]) - y
            idx = rng.integers(0, len(cells), size=(N_BOOT, len(cells)))
            d = np.sqrt((eb[idx] ** 2).mean(1)) - np.sqrt((ea[idx] ** 2).mean(1))
            out[name][key] = {"n": len(cells), "rmse_a": float(np.sqrt((ea ** 2).mean())),
                              "rmse_b": float(np.sqrt((eb ** 2).mean())), "diff": float(d.mean()),
                              "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                              "b_better_pct": float(100 * np.mean(d < 0))}
    return out


def paired_models(res, df, a, b, seed=0):
    """Bootstrap of RMSE(a) - RMSE(b) at one horizon, identical cells (negative: a better)."""
    rng = np.random.default_rng(seed)
    out = {}
    for key, split in (("primary_excl", "primary_test"), ("secondary", "secondary_test")):
        cells = [c for c in res["pred"][a] if c in res["pred"][b] and df.loc[c, "split"] == split and not df.loc[c, "anomalous"]]
        y = df.loc[cells, "cycle_life"].to_numpy()
        ea = np.array([res["pred"][a][c] for c in cells]) - y
        eb = np.array([res["pred"][b][c] for c in cells]) - y
        idx = rng.integers(0, len(cells), size=(N_BOOT, len(cells)))
        d = np.sqrt((ea[idx] ** 2).mean(1)) - np.sqrt((eb[idx] ** 2).mean(1))
        out[key] = {"diff": float(d.mean()), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out


def run(ns, verbose=True):
    res = {n: evaluate(n) for n in ns}
    df = build(max(ns))
    life = df.cycle_life
    out = {"horizons": {str(n): {"models": r["models"], "dead_before_n": r["dead_before_n"],
                                 "median_fraction_of_life_used": float(np.median(n / life[life > n]))}
                        for n, r in res.items()}}
    ref = 100 if 100 in res else ns[-1]
    out["reference_n"] = ref
    out["change_vs_reference"] = {str(n): paired_change(res[ref], res[n], df) for n in ns if n != ref}
    out["gp_vs_en_physics"] = {str(n): paired_models(res[n], df, "gp_physics", "physics") for n in ns}
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "horizon.json", "w") as fp:
        json.dump(out, fp, indent=2)
    if verbose:
        for n, h in out["horizons"].items():
            print(f"N = {n}: dead before N {h['dead_before_n']}, median share of life used {100 * h['median_fraction_of_life_used']:.0f}%")
            for name, sc in h["models"].items():
                print(f"   {name:9s} " + "  ".join(f"{k} {v['rmse']:.0f}/{v['mape']:.1f}%" for k, v in sc.items()))
        print(f"\nchange in test RMSE vs N={ref} (negative = better), paired bootstrap 95% CI:")
        for n, ch in out["change_vs_reference"].items():
            print(f"  N={n:>3s} " + " | ".join(f"{name} " + " ".join(f"{k[:4]} {w['diff']:+4.0f} [{w['ci'][0]:+.0f},{w['ci'][1]:+.0f}]"
                                                                       for k, w in v.items()) for name, v in ch.items()))
    return out


if __name__ == "__main__":
    run([int(a) for a in sys.argv[1:]] or [20, 30, 40, 50, 60, 80, 100, 150])

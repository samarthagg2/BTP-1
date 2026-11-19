"""P6 add-on: why do b2 and u1-standard cells live short? Test of the rest / temperature hypothesis.

Severson et al. (Methods) state the rests per batch: 2017-05-12 (b1) 1 min after 80 % SOC and 1 s after
discharge; 2017-06-30 (b2) 5 min and 5 min; 2018-04-12 (b3) 5 s at several points. u1 standard has 2 x 5 min,
u1 newstructure 2 x 10 s (measured from the raw traces).

Hypothesis: long rests let the cell cool, the next fast charge starts colder, lithium plating is more likely,
and life is shorter. Tests:
  1. same policy, different procedure: b1 vs b2, u1 standard vs newstructure;
  2. group level: temperature at the start of / during the fast charge against life;
  3. within a group and policy (rests identical): does a colder cell live shorter? Fixed-effects regression of
     log10 life on temperature with one intercept per (group, policy), cells with >= 2 replicates per policy.

Usage: python3 -m src.p6_rest_check   -> results/p6_rest_check.json
"""
import json

import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import BATCHES, RAW_DIR, RESULTS_DIR
from src.data import load_dataset
from src.features import load_features


def cell_temperatures(df):
    """Median over cycles 2-100 of: T at the first fast-charge sample, mean T during the fast charge."""
    cells = load_dataset()["cells"]
    out = pd.DataFrame(np.nan, index=df.index, columns=["T_start_fc", "T_mean_fc"])
    for b in sorted(df.batch.unique()):
        with h5py.File(RAW_DIR / BATCHES[b], "r") as f:
            for cid in df.index[df.batch == b]:
                if df.loc[cid, "temp_suspect"]:
                    continue
                g = f[f["batch"]["cycles"][cells[cid]["index_in_batch"], 0]]
                ts, tm = [], []
                for n in range(2, 101):
                    try:
                        get = lambda k: np.asarray(f[g[k][n - 1, 0]], dtype=float).ravel()
                        I, T, Qc = get("I"), get("T"), get("Qc")          # I in C-rate
                    except Exception:
                        continue
                    fc = np.flatnonzero((I > 1.5) & (Qc < 0.885))
                    if fc.size >= 5 and np.all(np.isfinite(T[fc])):
                        ts.append(T[fc[0]]); tm.append(T[fc].mean())
                if ts:
                    out.loc[cid] = [np.median(ts), np.median(tm)]
    return out


def run(verbose=True):
    df = load_features()
    df["group"] = np.where(df.policy.str.contains("newstructure") & (df.batch == "u1"), "u1 new",
                           np.where(df.batch == "u1", "u1 std", df.batch))
    df["base_policy"] = df.policy.str.replace("-newstructure", "", regex=False)
    df = df.join(cell_temperatures(df))
    df["log_life"] = np.log10(df.cycle_life)

    groups = {}
    for g, d in df.groupby("group"):
        ten = d[d.chargetime_avg_2_6 < 10.5]
        groups[g] = {"n": int(len(d)), "median_life": float(d.cycle_life.median()),
                     "median_life_10min_policies": float(ten.cycle_life.median()) if len(ten) else None,
                     "T_start_fc": float(d.T_start_fc.median()), "T_mean_fc": float(d.T_mean_fc.median())}
    shared = {}
    for a, b in (("b1", "b2"), ("u1 std", "u1 new")):
        pa, pb = df[df.group == a], df[df.group == b]
        for p in sorted(set(pa.base_policy) & set(pb.base_policy)):
            shared[f"{p}: {a} vs {b}"] = {a: pa[pa.base_policy == p].cycle_life.tolist(),
                                         b: pb[pb.base_policy == p].cycle_life.tolist(),
                                         "dT_start": float(pb[pb.base_policy == p].T_start_fc.median()
                                                           - pa[pa.base_policy == p].T_start_fc.median())}
    # within group and policy: fixed effects for (group, policy); exclude the anomalous cell
    d = df[np.isfinite(df.T_start_fc) & ~df.anomalous].copy()
    d["cell_key"] = d.group + "|" + d.base_policy
    d = d[d.groupby("cell_key").cell_key.transform("size") >= 2]
    within = {}
    for t in ("T_start_fc", "T_mean_fc"):
        x = d[t] - d.groupby("cell_key")[t].transform("mean")
        y = d.log_life - d.groupby("cell_key").log_life.transform("mean")
        res = sm.OLS(y, x).fit()
        lo, hi = res.conf_int().iloc[0]
        within[t] = {"slope_log10_per_degC": float(res.params.iloc[0]), "ci95": [float(lo), float(hi)],
                     "p": float(res.pvalues.iloc[0]), "n_cells": int(len(d)), "n_policy_groups": int(d.cell_key.nunique()),
                     "sd_within_degC": float(x.std())}
    # how much life change would the group temperature gaps imply, using the within-group slope?
    s = within["T_start_fc"]["slope_log10_per_degC"]
    implied = {f"{a} vs {b}": float(10 ** (s * (groups[b]["T_start_fc"] - groups[a]["T_start_fc"]))) for a, b in
               (("b1", "b2"), ("u1 new", "u1 std"))}
    res = {"groups": groups, "shared_policies": shared, "within_group_policy": within,
           "implied_life_ratio_from_temperature": implied,
           "cells": df[["group", "policy", "cycle_life", "T_start_fc", "T_mean_fc"]].round(3).to_dict(orient="index")}
    with open(RESULTS_DIR / "p6_rest_check.json", "w") as fp:
        json.dump(res, fp, indent=2, default=str)
    if verbose:
        print(pd.DataFrame(groups).T.round(2))
        for k, v in shared.items():
            print(k, v)
        for k, v in within.items():
            print(k, {q: (np.round(w, 4) if not isinstance(w, int) else w) for q, w in v.items()})
        print("implied life ratio from temperature alone:", {k: round(v, 3) for k, v in implied.items()})
    return res


if __name__ == "__main__":
    run()

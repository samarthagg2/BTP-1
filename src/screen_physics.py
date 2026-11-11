"""P3: screening of physics-informed feature candidates (TRAINING cells only).

The test sets are never used here. For every candidate we report:
  * Spearman rho with log10(life): all training cells, within b1, within b2
    (a feature whose sign flips between batches is a batch proxy, not physics);
  * within-batch OLS t-statistic of the candidate given dq_log_var
    (does it add information beyond ΔQ inside each batch?);
  * paired cross-validated gain (RepeatedKFold 4 x 100, OLS on z-scored inputs)
    on top of dq_log_var, on top of margin_rate, and on top of
    dq_log_var + a batch label (gain that a batch label cannot explain).
It also records the evidence behind the design: ΔQ-trajectory exponents,
protocol step durations per group, and a nested-CV comparison of feature sets
with the paper's elastic-net protocol.

Usage: python3 -m src.screen_physics      -> results/p3_screening.json (+ .csv)
"""
import json
import re
import sys
import warnings

import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import BATCHES, PROC_DIR, RAW_DIR, RESULTS_DIR, VDLIN
from src.data import curve, load_dataset
from src.features import FEATURE_SETS, SMOOTH_WIN, _window, load_features, value_at
from src.reproduce import make_model

PEAK_WIN = (VDLIN >= 3.0) & (VDLIN <= 3.3)     # main dQ/dV peak at 4C sits at 3.14-3.15 V
TRAJ_CYCLES = np.arange(20, 101, 10)

# candidate -> (mechanism, short definition); the two selected features are marked in SELECTED
CANDIDATES = {
    "dqdv_peak_h_delta": ("LAM / LLI", "height of main dQ/dV peak, cycle 100 - cycle 10"),
    "dqdv_peak_v_delta": ("OCV shift + overpotential", "voltage of main dQ/dV peak, cycle 100 - cycle 10"),
    "dqdv_area_delta": ("capacity of main graphite stage", "charge delivered 3.3->3.0 V, cycle 100 - cycle 10"),
    "dq_offset_2v": ("LLI (CC capacity)", "ΔQ_100-10 at 2.0 V"),
    "cv_hold_delta": ("kinetics", "charge from the 2.0 V CV hold, cycle 100 - cycle 10"),
    "ce_deficit": ("side reactions", "1 - median(QD/QC), cycles 2-100"),
    "ir_sqrt_slope": ("SEI growth ~ sqrt(t)", "slope of IR vs sqrt(cycle), cycles 2-100"),
    "temp_rise_mean": ("Joule heating", "mean (Tmax - Tmin), cycles 2-100"),
    "T_start_fastcharge": ("plating vs temperature", "median cell temperature at start of fast charge, cycles 2-100"),
    "plating_index": ("plating risk of the policy", "mean C-rate of the policy over 50-80% SOC"),
    "qd_curv_2_100": ("fade acceleration", "quadratic coefficient of QD vs cycle, 2-100"),
    "traj_exponent": ("fade acceleration", "p in amplitude(ΔQ_n-10) ~ (n-10)^p, n = 20..100"),
    "dq_multi_log_var": ("noise reduction", "log var of mean Q(91-100) - mean Q(6-15)"),
    "peak_cycle": ("overhang timing", "cycle of the early capacity peak (smoothed QD, 2-100)"),
    "t_half_rise": ("overhang timing", "cycle at which half of the early rise is reached"),
    "rise_before10": ("overhang", "capacity rise cycle 2 -> 10 (mAh)"),
    "overhang_after10": ("overhang hiding ΔQ", "capacity rise cycle 10 -> early peak (mAh)"),
    "margin_rate": ("budget / rate", "log10((QD(2) - 0.88 Ah) / sd ΔQ_100-10)"),
    "t_cycle": ("time-based ageing (protocol)", "median cycle duration, cycles 2-100 (min)"),
}
SELECTED = ["margin_rate", "overhang_after10"]


# ----------------------------------------------------------------- candidates
def _policy_c_rate_high_soc(policy):
    m = re.match(r"([\d.]+)C\((\d+)%\)-([\d.]+)C", policy.replace("-newstructure", ""))
    c1, q1, c2 = float(m[1]), float(m[2]) / 100, float(m[3])
    soc = np.linspace(0, 0.8, 801)
    return float(np.where(soc < q1, c1, c2)[soc >= 0.5].mean())


def _capacity_peak(cell):
    cyc, q = _window(cell, "QD", 2, 100)
    ok = np.isfinite(q)
    cyc, q = cyc[ok].astype(float), q[ok]
    sm_q = pd.Series(q).rolling(SMOOTH_WIN, center=True, min_periods=max(3, SMOOTH_WIN // 2)).median().to_numpy()
    k = int(np.nanargmax(sm_q))
    base, q10 = np.nanmedian(q[cyc <= 4]), np.nanmedian(q[(cyc >= 9) & (cyc <= 11)])
    rise = sm_q[k] - base
    half = cyc[int(np.argmax(sm_q >= base + 0.5 * rise))] if rise > 0 else 2.0
    return {"peak_cycle": cyc[k], "t_half_rise": half, "rise_before10": 1000 * (q10 - base)}


def _dq_trajectory(cell):
    q10 = curve(cell, "Qdlin", 10)
    amp = np.array([np.std(curve(cell, "Qdlin", n) - q10, ddof=1) for n in TRAJ_CYCLES])
    p, logk = np.polyfit(np.log10(TRAJ_CYCLES - 10), np.log10(amp), 1)
    resid = np.log10(amp) - (logk + p * np.log10(TRAJ_CYCLES - 10))
    r2 = 1 - np.sum(resid ** 2) / np.sum((np.log10(amp) - np.log10(amp).mean()) ** 2)
    return p, r2, amp


def candidate_table(df, extras, raw_temperature=True):
    cells = load_dataset()["cells"]
    tr = df[df.split == "train"]
    rows, traj = {}, {}
    for cid in tr.index:
        c = cells[cid]
        s = c["summary"]
        cyc = s["cycle"]
        w = (cyc >= 2) & (cyc <= 100)
        f = {}
        sm_d = {n: savgol_filter(curve(c, "dQdV", n), 51, 3) for n in (10, 100)}
        pk = {n: int(np.argmin(np.where(PEAK_WIN, sm_d[n], np.inf))) for n in sm_d}
        f["dqdv_peak_h_delta"] = -sm_d[100][pk[100]] + sm_d[10][pk[10]]
        f["dqdv_peak_v_delta"] = VDLIN[pk[100]] - VDLIN[pk[10]]
        q = {n: curve(c, "Qdlin", n) for n in (10, 100)}
        span = lambda a: a[PEAK_WIN].max() - a[PEAK_WIN].min()
        f["dqdv_area_delta"] = span(q[100]) - span(q[10])
        f["dq_offset_2v"] = q[100][-1] - q[10][-1]
        cv_hold = lambda n: value_at(c, "QD", n) - curve(c, "Qdlin", n)[-1]
        f["cv_hold_delta"] = cv_hold(100) - cv_hold(10)
        ce = s["QD_clean"][w] / s["QC"][w]
        f["ce_deficit"] = 1 - np.median(ce[np.isfinite(ce) & (ce > 0.9) & (ce < 1.1)])
        wi = w & np.isfinite(s["IR"])
        f["ir_sqrt_slope"] = np.polyfit(np.sqrt(cyc[wi]), s["IR"][wi], 1)[0]
        f["temp_rise_mean"] = np.nanmean(s["Tmax"][w] - s["Tmin"][w])
        f["plating_index"] = _policy_c_rate_high_soc(c["policy"])
        cq, qq = _window(c, "QD", 2, 100)
        ok = np.isfinite(qq)
        f["qd_curv_2_100"] = np.polyfit(cq[ok].astype(float), qq[ok], 2)[0]
        f["traj_exponent"], r2, amp = _dq_trajectory(c)
        traj[cid] = {"r2": r2, "amp_mAh": (1000 * amp).tolist()}
        mean_q = lambda a, b: np.mean([curve(c, "Qdlin", n) for n in range(a, b + 1)], axis=0)
        f["dq_multi_log_var"] = np.log10(np.var(mean_q(91, 100) - mean_q(6, 15), ddof=1))
        f.update(_capacity_peak(c))
        f["overhang_after10"] = df.loc[cid, "overhang_after10"]
        f["margin_rate"] = df.loc[cid, "margin_rate"]
        e = extras[cid]
        f["t_cycle"] = np.nanmedian(e["t_dur"][(e["cycle"] >= 2) & (e["cycle"] <= 100)])
        rows[cid] = f
    F = pd.DataFrame(rows).T.astype(float)
    F["T_start_fastcharge"] = fastcharge_temperature(tr) if raw_temperature else np.nan
    return F[list(CANDIDATES)], traj


def fastcharge_temperature(tr):
    """Median cell-can temperature at the first sample of the fast charge, cycles 2-100
    (raw traces). Cells whose probe lost contact (temp_suspect) -> NaN."""
    cells = load_dataset()["cells"]
    out = pd.Series(np.nan, index=tr.index)
    for b in sorted(tr.batch.unique()):
        with h5py.File(RAW_DIR / BATCHES[b], "r") as f:
            for cid in tr.index[tr.batch == b]:
                if tr.loc[cid, "temp_suspect"]:
                    continue
                g = f[f["batch"]["cycles"][cells[cid]["index_in_batch"], 0]]
                vals = []
                for n in range(2, 101):
                    get = lambda k: np.asarray(f[g[k][n - 1, 0]], dtype=float).ravel()
                    I, T, Qc = get("I"), get("T"), get("Qc")      # I is in C-rate
                    fc = np.flatnonzero((I > 1.5) & (Qc < 0.885))
                    if fc.size >= 5 and np.isfinite(T[fc[0]]):
                        vals.append(T[fc[0]])
                out[cid] = np.median(vals) if vals else np.nan
    return out


# ----------------------------------------------------------------- evaluation
def evaluate(F, tr, n_repeats=100, seed=1):
    y = np.log10(tr.cycle_life.to_numpy())
    b1 = (tr.batch == "b1").to_numpy()
    var = tr.dq_log_var.to_numpy()
    batch = (~b1).astype(float)
    margin = F["margin_rate"].to_numpy()
    splits = list(RepeatedKFold(n_splits=4, n_repeats=n_repeats, random_state=seed).split(y))

    def cv_err(Z):
        e = []
        for a, b in splits:
            m = make_pipeline(StandardScaler(), LinearRegression()).fit(Z[a], y[a])
            e.append(np.sqrt(np.mean((m.predict(Z[b]) - y[b]) ** 2)))
        return np.array(e)

    base = {"dq": cv_err(var[:, None]), "margin": cv_err(margin[:, None]),
            "dq_batch": cv_err(np.c_[var, batch])}
    out = {}
    for k in F:
        x = F[k].to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, np.nanmedian(x))
        rho = lambda mk: float(spearmanr(x[mk], y[mk])[0])
        t = {}
        for name, mk in (("b1", b1), ("b2", ~b1)):
            res = sm.OLS(y[mk], sm.add_constant(np.c_[var[mk], x[mk]])).fit()
            t[name] = (float(res.tvalues[2]), float(res.pvalues[2]))
        e_dq, e_m, e_db = cv_err(np.c_[var, x]), cv_err(np.c_[margin, x]), cv_err(np.c_[var, batch, x])
        out[k] = {
            "mechanism": CANDIDATES[k][0], "definition": CANDIDATES[k][1],
            "rho_all": rho(np.ones(len(y), bool)), "rho_b1": rho(b1), "rho_b2": rho(~b1),
            "abs_rho_with_dq_log_var": float(abs(spearmanr(x, var)[0])),
            "t_given_dq_b1": t["b1"][0], "p_given_dq_b1": t["b1"][1],
            "t_given_dq_b2": t["b2"][0], "p_given_dq_b2": t["b2"][1],
            "gain_on_dq_pct": float(100 * (base["dq"] - e_dq).mean() / base["dq"].mean()),
            "gain_on_margin_pct": (float(100 * (base["margin"] - e_m).mean() / base["margin"].mean())
                                   if k != "margin_rate" else 0.0),
            "gain_beyond_batch_pct": float(100 * (base["dq_batch"] - e_db).mean() / base["dq_batch"].mean()),
            "beyond_batch_better_folds_pct": float(100 * np.mean(base["dq_batch"] > e_db)),
            "selected": k in SELECTED,
        }
    out["_reference"] = {"cv_rmse_log_dq_log_var": float(base["dq"].mean()),
                         "cv_rmse_log_margin_rate": float(base["margin"].mean()),
                         "margin_better_than_dq_folds_pct": float(100 * np.mean(base["margin"] < base["dq"])),
                         "cv_rmse_log_dq_plus_batch": float(base["dq_batch"].mean())}
    return out


def compare_sets(df, n_repeats=10, seed=7):
    """Nested CV on the training cells with the paper's protocol (elastic net with
    inner RepeatedKFold(4, 10); OLS for one feature). The 4 temp-suspect cells are
    left out for every set so all sets see the same cells."""
    tr = df[(df.split == "train") & ~df.temp_suspect]
    y = np.log10(tr.cycle_life.to_numpy())
    sets = {"F1 variance": FEATURE_SETS["variance"], "F2 discharge": FEATURE_SETS["discharge"],
            "F3 full": FEATURE_SETS["full"], "F4 physics": FEATURE_SETS["physics"],
            "F3 + physics": FEATURE_SETS["full"] + FEATURE_SETS["physics"]}
    outer = list(RepeatedKFold(n_splits=4, n_repeats=n_repeats, random_state=seed).split(y))
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for name, cols in sets.items():
            X = tr[cols].to_numpy()
            el, ec = [], []
            for i, (a, b) in enumerate(outer):
                m = make_model("variance" if len(cols) == 1 else "full", seed=i).fit(X[a], y[a])
                p = m.predict(X[b])
                el.append(np.sqrt(np.mean((p - y[b]) ** 2)))
                ec.append(np.sqrt(np.mean((10 ** p - 10 ** y[b]) ** 2)))
            res[name] = {"n_features": len(cols), "rmse_log": float(np.mean(el)), "rmse_cycles": float(np.mean(ec)),
                         "fold_rmse_log": [float(v) for v in el]}
    ref = np.array(res["F2 discharge"]["fold_rmse_log"])
    for v in res.values():
        v["better_than_F2_folds_pct"] = float(100 * np.mean(np.array(v["fold_rmse_log"]) < ref))
    return {"n_cells": int(len(tr)), "outer": f"RepeatedKFold(4, {n_repeats})", "sets": res}


def step_breakdown(n_cells=6, cycles=(20, 60), seed=0):
    """Median duration (min) of each protocol step and the cell temperature at the
    start of the fast charge, for random cells of each batch / group (all batches:
    only the protocol and temperatures are read, no cycle lives)."""
    cells = load_dataset()["cells"]
    groups = {}
    for cid, c in cells.items():
        g = c["batch"] + ("-newstructure" if c["batch"] == "u1" and "newstructure" in c["policy"] else "")
        groups.setdefault(g, []).append(cid)
    rng = np.random.default_rng(seed)
    rows = []
    for g, ids in sorted(groups.items()):
        with h5py.File(RAW_DIR / BATCHES[g[:2]], "r") as f:
            for cid in rng.choice(ids, size=min(n_cells, len(ids)), replace=False):
                grp = f[f["batch"]["cycles"][cells[cid]["index_in_batch"], 0]]
                for n in cycles:
                    get = lambda k: np.asarray(f[grp[k][n - 1, 0]], dtype=float).ravel()
                    t, I, V, T, Qc = (get(k) for k in ("t", "I", "V", "T", "Qc"))
                    dt = np.diff(t, append=t[-1])
                    dt = np.where((dt >= 0) & (dt < 10), dt, 0)
                    d0 = np.flatnonzero(I < -1)[0]
                    pre = np.arange(len(t)) < d0
                    fc = np.flatnonzero(pre & (I > 1.5) & (Qc < 0.885))
                    rows.append({
                        "group": g,
                        "fast_charge": dt[fc].sum(),
                        "rest_after_80pct": dt[pre & (np.abs(I) < 0.01) & (Qc > 0.8) & (Qc < 0.95)].sum(),
                        "cc_1C": dt[pre & (np.abs(I - 1) < 0.05) & (V < 3.598)].sum(),
                        "cv_3.6V": dt[pre & (V >= 3.598) & (I > 0.005) & (I < 0.99) & (Qc > 0.8)].sum(),
                        "discharge_cc_4C": dt[~pre & (I < -3.9)].sum(),
                        "discharge_cv_2V": dt[~pre & (V <= 2.003) & (I < -0.005) & (I > -3.9)].sum(),
                        "rest_after_discharge": dt[~pre & (np.abs(I) < 0.005)].sum()
                        + dt[pre & (np.abs(I) < 0.005) & (Qc < 0.05)].sum(),
                        "total": dt.sum(),
                        "T_start_fastcharge": T[fc[0]] if fc.size else np.nan,
                    })
    d = pd.DataFrame(rows)
    return {g: {k: float(v) for k, v in r.items()} for g, r in d.groupby("group").median().round(2).iterrows()}


def run(raw=True, verbose=True):
    df = load_features()
    extras = pd.read_pickle(PROC_DIR / "extras.pkl")
    tr = df[df.split == "train"]
    F, traj = candidate_table(df, extras, raw_temperature=raw)
    ev = evaluate(F, tr)
    p = F["traj_exponent"]
    out = {"candidates": ev,
           "trajectory": {"median_exponent": float(p.median()), "q25": float(p.quantile(.25)), "q75": float(p.quantile(.75)),
                          "median_r2": float(np.median([v["r2"] for v in traj.values()])),
                          "median_amp_mAh": np.median([v["amp_mAh"] for v in traj.values()], axis=0).round(2).tolist(),
                          "cycles": TRAJ_CYCLES.tolist()},
           "feature_sets": compare_sets(df),
           "steps": step_breakdown() if raw else {}}
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "p3_screening.json", "w") as fp:
        json.dump(out, fp, indent=2)
    table = pd.DataFrame({k: v for k, v in ev.items() if not k.startswith("_")}).T
    table.to_csv(RESULTS_DIR / "p3_screening.csv")
    F.to_csv(RESULTS_DIR / "p3_candidates_train.csv")
    if verbose:
        report(out)
    return out


def report(out):
    t = pd.DataFrame({k: v for k, v in out["candidates"].items() if not k.startswith("_")}).T
    cols = ["rho_all", "rho_b1", "rho_b2", "t_given_dq_b1", "t_given_dq_b2", "gain_on_dq_pct", "gain_on_margin_pct",
            "gain_beyond_batch_pct", "selected"]
    print(t[cols].astype({c: float for c in cols[:-1]}).round(2).to_string())
    print("reference:", {k: round(v, 4) for k, v in out["candidates"]["_reference"].items()})
    print("ΔQ trajectory exponent: median %.2f (IQR %.2f-%.2f)" % (out["trajectory"]["median_exponent"],
          out["trajectory"]["q25"], out["trajectory"]["q75"]))
    for k, v in out["feature_sets"]["sets"].items():
        print(f"  {k:14s} {v['n_features']:2d} feat  RMSE log {v['rmse_log']:.4f}  cycles {v['rmse_cycles']:6.1f}  "
              f"better than F2 in {v['better_than_F2_folds_pct']:.0f}% of folds")
    if out["steps"]:
        print(pd.DataFrame(out["steps"]).T.to_string())


if __name__ == "__main__":
    run(raw="--no-raw" not in sys.argv)

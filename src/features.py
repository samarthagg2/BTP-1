"""Early-cycle features of Severson et al. 2019 (SI Table 1 + SI Note 1).

Every feature uses data from cycles <= 100 only. log = log10.
ΔQ(V) = Q_100(V) - Q_10(V) on the 1000-point Vdlin grid (via data.curve, which
substitutes the nearest valid cycle if a curve is corrupt).

All 20 candidate features of SI Table 1 are computed; FEATURE_SETS picks the
subsets the paper's three models use (the check marks in SI Table 1).

Implementation choices where SI Note 1 is ambiguous (documented in the report):
  * skewness / kurtosis: standard moment ratios m3/m2^1.5 and m4/m2^2
    (population moments; kurtosis NOT excess). The SI skewness formula has an
    inconsistent denominator, read as a typo.
  * QD features use QD_clean (glitch cycles = NaN, dropped from fits).
  * "max discharge capacity" = max of QD_clean over cycles 2..100.
  * single-cycle values (QD(2), IR(2), IR(100)) fall back to the nearest cycle
    with a finite value within +-MAX_FALLBACK cycles (inside 2..100).
  * temperature integral = sum over cycles 2..100 of trapz(T, t) (degC*min),
    computed from raw traces by src.extras.

Usage:  python3 -m src.features      -> data/processed/features.csv
"""
import pickle

import numpy as np
import pandas as pd

from config import PROC_DIR
from src.data import curve, load_dataset, load_splits

N_LO, N_HI = 10, 100          # ΔQ cycles
C_FIRST, C_LAST = 2, 100      # summary-feature window
MAX_FALLBACK = 3

FEATURE_SETS = {
    "variance": ["dq_log_var"],
    "discharge": ["dq_log_min", "dq_log_var", "dq_log_skew", "dq_log_kurt",
                  "qd_2", "qd_max_minus_2"],
    "full": ["dq_log_min", "dq_log_var", "qd_slope_2_100", "qd_icpt_2_100", "qd_2",
             "chargetime_avg_2_6", "temp_integral_2_100", "ir_min_2_100",
             "ir_100_minus_2"],
}
ALL_CANDIDATES = [
    "dq_log_min", "dq_log_mean", "dq_log_var", "dq_log_skew", "dq_log_kurt", "dq_log_2v",
    "qd_slope_2_100", "qd_icpt_2_100", "qd_slope_91_100", "qd_icpt_91_100",
    "qd_2", "qd_max_minus_2", "qd_100",
    "chargetime_avg_2_6", "temp_max_2_100", "temp_min_2_100", "temp_integral_2_100",
    "ir_2", "ir_min_2_100", "ir_100_minus_2",
]


# ---------------------------------------------------------------- helpers
def _series(cell, field):
    s = cell["summary"]
    key = "QD_clean" if field == "QD" else field
    return np.asarray(s["cycle"]), np.asarray(s[key], dtype=np.float64)


def _window(cell, field, lo=C_FIRST, hi=C_LAST):
    cyc, v = _series(cell, field)
    m = (cyc >= lo) & (cyc <= hi)
    return cyc[m], v[m]


def value_at(cell, field, n, lo=C_FIRST, hi=C_LAST, max_shift=MAX_FALLBACK):
    """Value of a summary field at cycle n, or the nearest finite one (ties -> earlier)."""
    cyc, v = _series(cell, field)
    for d in [0] + [s for k in range(1, max_shift + 1) for s in (-k, k)]:
        m = n + d
        if m < lo or m > hi:
            continue
        hit = np.where(cyc == m)[0]
        if hit.size and np.isfinite(v[hit[0]]):
            return float(v[hit[0]])
    return np.nan


def _linfit(cell, lo, hi):
    cyc, q = _window(cell, "QD", lo, hi)
    ok = np.isfinite(q)
    slope, icpt = np.polyfit(cyc[ok].astype(float), q[ok], 1)
    return float(slope), float(icpt)


def delta_q(cell, hi=N_HI, lo=N_LO):
    return curve(cell, "Qdlin", hi) - curve(cell, "Qdlin", lo)


def _log_abs(x):
    return float(np.log10(np.abs(x)))


# ---------------------------------------------------------------- features
def dq_features(cell):
    dq = delta_q(cell)
    mu = dq.mean()
    dev = dq - mu
    m2, m3, m4 = (dev ** 2).mean(), (dev ** 3).mean(), (dev ** 4).mean()
    return {
        "dq_log_min": _log_abs(dq.min()),
        "dq_log_mean": _log_abs(mu),
        "dq_log_var": _log_abs(np.var(dq, ddof=1)),
        "dq_log_skew": _log_abs(m3 / m2 ** 1.5),
        "dq_log_kurt": _log_abs(m4 / m2 ** 2),
        "dq_log_2v": _log_abs(dq[-1]),            # VDLIN[-1] == 2.0 V
    }


def qd_features(cell):
    s1, i1 = _linfit(cell, 2, 100)
    s2, i2 = _linfit(cell, 91, 100)
    _, q = _window(cell, "QD")
    q2 = value_at(cell, "QD", 2)
    return {
        "qd_slope_2_100": s1, "qd_icpt_2_100": i1,
        "qd_slope_91_100": s2, "qd_icpt_91_100": i2,
        "qd_2": q2, "qd_max_minus_2": float(np.nanmax(q)) - q2,
        "qd_100": value_at(cell, "QD", 100),
    }


def other_features(cell, extras):
    _, ct = _window(cell, "chargetime", 2, 6)
    _, tmax = _window(cell, "Tmax")
    _, tmin = _window(cell, "Tmin")
    _, ir = _window(cell, "IR")
    e = extras
    m = (e["cycle"] >= C_FIRST) & (e["cycle"] <= C_LAST)
    tint = e["T_int"][m]
    ir2, ir100 = value_at(cell, "IR", 2), value_at(cell, "IR", 100)
    with np.errstate(all="ignore"):
        return {
            "chargetime_avg_2_6": float(np.nanmean(ct)) if np.isfinite(ct).any() else np.nan,
            "temp_max_2_100": float(np.nanmax(tmax)),
            "temp_min_2_100": float(np.nanmin(tmin)),
            "temp_integral_2_100": float(np.nansum(tint)) if np.isfinite(tint).all() else np.nan,
            "ir_2": ir2,
            "ir_min_2_100": float(np.nanmin(ir)) if np.isfinite(ir).any() else np.nan,
            "ir_100_minus_2": ir100 - ir2,
        }


def cell_features(cell, extras):
    return {**dq_features(cell), **qd_features(cell), **other_features(cell, extras)}


def build(save=True):
    ds, sp = load_dataset(), load_splits()
    with open(PROC_DIR / "extras.pkl", "rb") as fp:
        extras = pickle.load(fp)
    split_of = {c: k for k in ("train", "primary_test", "secondary_test", "unseen") for c in sp[k]}
    rows = []
    for cid, split in split_of.items():
        c = ds["cells"][cid]
        rows.append({"cell": cid, "split": split, "batch": c["batch"], "policy": c["policy"],
                     "cycle_life": float(c["cycle_life"]),
                     "anomalous": cid in sp["anomalous"], "temp_suspect": bool(c["temp_suspect"]),
                     **cell_features(c, extras[cid])})
    df = pd.DataFrame(rows).set_index("cell")
    if save:
        df.to_csv(PROC_DIR / "features.csv")
        print(f"wrote {PROC_DIR / 'features.csv'}  ({len(df)} cells x {len(ALL_CANDIDATES)} features)")
    return df


def load_features():
    return pd.read_csv(PROC_DIR / "features.csv", index_col="cell")


if __name__ == "__main__":
    df = build()
    nan = df[ALL_CANDIDATES].isna().sum()
    print("NaN counts (non-zero):", nan[nan > 0].to_dict() or "none")

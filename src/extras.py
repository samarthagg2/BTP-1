"""Per-cycle quantities that need the raw within-cycle traces (not in the cache).

For every kept cell and every cycle 1..N_CACHE it stores
  t_dur   : measured cycle duration (min), excluding sample gaps > MAX_DT_MIN
  T_int   : integral of cell-can temperature over the measured time (degC * min)
            -> SI Note 1 "temperature integral" is the sum over cycles 2..100
  t_gap   : time inside the record excluded as gaps (pauses / corrupt time stamps)

Output: data/processed/extras.pkl = {cell_id: {"cycle": int[], "t_dur": float[], "T_int": float[], "t_gap": float[]}}
Usage:  python3 -m src.extras
"""
import pickle
import sys
import time

import h5py
import numpy as np

from config import BATCHES, N_CACHE, PROC_DIR, RAW_DIR
from src.data import load_dataset

OUT = PROC_DIR / "extras.pkl"


MAX_DT_MIN = 10.0   # sample intervals longer than this are gaps, not measured time


def _cycle_stats(f, cyc_group, record):
    """(duration, temperature integral, gap time) of one cycle record.
    Intervals between consecutive samples longer than MAX_DT_MIN are excluded
    from both duration and integral: they are test pauses (u1 ~cycle 48, b1
    cycles 12-13) or corrupt time stamps (b1c18 cycles 2-53 jump ~4000 min)."""
    t = np.asarray(f[cyc_group["t"][record, 0]], dtype=np.float64).ravel()
    T = np.asarray(f[cyc_group["T"][record, 0]], dtype=np.float64).ravel()
    if t.size < 2 or T.size != t.size:
        return np.nan, np.nan, np.nan
    ok = np.isfinite(t) & np.isfinite(T)
    if ok.sum() < 2:
        return np.nan, np.nan, np.nan
    t, T = t[ok], T[ok]
    dt = np.diff(t)
    keep = (dt >= 0) & (dt <= MAX_DT_MIN)
    seg = 0.5 * (T[1:] + T[:-1]) * dt
    return float(dt[keep].sum()), float(seg[keep].sum()), float(dt[~keep & (dt > 0)].sum())


def build(verbose=True):
    cells = load_dataset()["cells"]
    out = {}
    for b, fname in BATCHES.items():
        mine = {cid: c for cid, c in cells.items() if c["batch"] == b}
        if not mine:
            continue
        t0 = time.time()
        with h5py.File(RAW_DIR / fname, "r") as f:
            for cid, c in mine.items():
                cyc_group = f[f["batch"]["cycles"][c["index_in_batch"], 0]]
                n_rec = cyc_group["t"].shape[0]
                n = min(N_CACHE, n_rec)
                dur, tint, gap = (np.full(n, np.nan) for _ in range(3))
                for r in range(n):
                    dur[r], tint[r], gap[r] = _cycle_stats(f, cyc_group, r)
                # record r holds cycle r+1 (same convention as the cache; checked in src.checks)
                out[cid] = {"cycle": np.arange(1, n + 1), "t_dur": dur, "T_int": tint, "t_gap": gap}
        if verbose:
            print(f"{b}: {len(mine)} cells in {time.time() - t0:.0f}s", flush=True)
    with open(OUT, "wb") as fp:
        pickle.dump(out, fp)
    if verbose:
        print(f"wrote {OUT} ({len(out)} cells)")
    return out


if __name__ == "__main__":
    build(verbose="-q" not in sys.argv)

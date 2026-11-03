"""Convert a raw batch .mat (MATLAB v7.3 / HDF5) into a lightweight cache.

The cache keeps, for every cell:
  - descriptors: cycle_life, policy
    (barcode/channel_id are MATLAB `string` objects that h5py cannot decode)
  - summary: per-cycle arrays for ALL cycles (small)
  - curves: Qdlin, Tdlin, dQdV for the first N_CACHE cycle records only,
            as float32 arrays of shape (n_records, 1000)

Raw within-cycle time series (I, V, T, t, Qc, Qd) are NOT cached; they are
only needed for optional analyses and can be re-read from the .mat file.

Usage (from repo root):
    python3 -m src.convert b1          # one batch
    python3 -m src.convert all         # every batch whose .mat is present
"""
import pickle
import sys
import time

import h5py
import numpy as np

from config import BATCHES, N_CACHE, PROC_DIR, RAW_DIR, VDLIN

SUMMARY_FIELDS = {          # name in .mat -> name in cache
    "cycle": "cycle", "QDischarge": "QD", "QCharge": "QC", "IR": "IR",
    "Tmax": "Tmax", "Tavg": "Tavg", "Tmin": "Tmin", "chargetime": "chargetime",
}
CURVE_FIELDS = {"Qdlin": "Qdlin", "Tdlin": "Tdlin", "discharge_dQdV": "dQdV"}


def _str(f, ref):
    """Decode a MATLAB char array (UTF-16 code units) to str."""
    return f[ref][()].tobytes()[::2].decode()


def _arr(f, ref):
    return np.ravel(f[ref][()])


def convert_batch(name, n_cache=N_CACHE, verbose=True):
    path = RAW_DIR / BATCHES[name]
    t0 = time.time()
    cells = {}
    with h5py.File(path, "r") as f:
        b = f["batch"]
        n_cells = b["summary"].shape[0]
        vdlin = _arr(f, b["Vdlin"][0, 0]) if "Vdlin" in b else VDLIN
        for i in range(n_cells):
            s = f[b["summary"][i, 0]]
            summary = {new: np.ravel(s[old][()]).astype(np.float64)
                       for old, new in SUMMARY_FIELDS.items()}
            cyc = f[b["cycles"][i, 0]]
            n_rec = cyc["Qdlin"].shape[0]
            k = min(n_cache, n_rec)
            curves = {}
            for old, new in CURVE_FIELDS.items():
                rows = []
                for j in range(k):
                    a = _arr(f, cyc[old][j, 0]).astype(np.float32)
                    if a.size != 1000:   # empty/odd record -> NaN row
                        a = np.full(1000, np.nan, np.float32)
                    rows.append(a)
                curves[new] = np.vstack(rows) if rows else np.empty((0, 1000), np.float32)
            cells[f"{name}c{i}"] = {
                "batch": name,
                "index_in_batch": i,
                "cycle_life": float(np.ravel(f[b["cycle_life"][i, 0]][()])[0]),
                "policy": _str(f, b["policy_readable"][i, 0]).strip(),
                "n_cycle_records": int(n_rec),
                "summary": summary,
                "curves": curves,
            }
            if verbose and (i % 10 == 0 or i == n_cells - 1):
                print(f"  {name}: cell {i + 1}/{n_cells}  ({time.time() - t0:.0f}s)", flush=True)
    out = {"batch": name, "source": BATCHES[name], "vdlin": np.asarray(vdlin),
           "n_cache": n_cache, "cells": cells}
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    dst = PROC_DIR / f"{name}_raw.pkl"
    with open(dst, "wb") as fp:
        pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB, {len(cells)} cells, "
              f"{time.time() - t0:.0f}s)")
    return out


RAW_FIELDS = {"I": "I", "V": "V", "T": "T", "t": "t", "Qc": "Qc", "Qd": "Qd",
              "Qdlin": "Qdlin", "Tdlin": "Tdlin", "discharge_dQdV": "dQdV"}


def read_raw_cycle(name, cell_index, record):
    """Read one full cycle record (raw traces + curves) straight from the .mat.
    For plots/analyses that need data the cache leaves out. `record` is the
    record index (cycle number = record + 1)."""
    with h5py.File(RAW_DIR / BATCHES[name], "r") as f:
        cyc = f[f["batch"]["cycles"][cell_index, 0]]
        return {new: _arr(f, cyc[old][record, 0]).astype(np.float64)
                for old, new in RAW_FIELDS.items()}


def main(argv):
    names = list(BATCHES) if argv[:1] == ["all"] else argv
    for n in names:
        if not (RAW_DIR / BATCHES[n]).exists():
            print(f"skip {n}: {BATCHES[n]} not found")
            continue
        convert_batch(n)


if __name__ == "__main__":
    main(sys.argv[1:])

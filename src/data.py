"""Load the cleaned dataset and access cycles safely.

ALWAYS use `curve(cell, field, n)` to get a cycle's curve: it maps a cycle
NUMBER n to the right cached row via summary['cycle'], so the off-by-one
between record index and cycle number can never leak in.
"""
import json
import pickle
from functools import lru_cache

import numpy as np

from config import PROC_DIR


@lru_cache(maxsize=1)
def load_dataset():
    with open(PROC_DIR / "dataset.pkl", "rb") as fp:
        return pickle.load(fp)


@lru_cache(maxsize=1)
def load_splits():
    with open(PROC_DIR / "splits.json") as fp:
        return json.load(fp)


def life(cell):
    return float(cell["cycle_life"])


def row_of_cycle(cell, n):
    """Index of the cached curve row holding cycle number n."""
    cyc = cell["summary"]["cycle"]
    n_rows = cell["curves"]["Qdlin"].shape[0]
    hits = np.where(cyc[:n_rows] == n)[0]
    if hits.size != 1:
        raise KeyError(f"cycle {n} not uniquely found in cached rows "
                       f"(cycles {cyc[:3]}...{cyc[n_rows - 1]})")
    return int(hits[0])


MAX_SUBSTITUTE = 2   # how many cycles away a replacement curve may come from


def valid_cycle(cell, n, max_shift=MAX_SUBSTITUTE):
    """Cycle number to use for n: n itself if its curve is valid, otherwise the
    nearest valid cycle within +-max_shift (ties -> earlier cycle, so no future
    data beyond n+max_shift is ever used)."""
    valid = cell.get("curve_valid")
    for d in [0] + [s for k in range(1, max_shift + 1) for s in (-k, k)]:
        try:
            r = row_of_cycle(cell, n + d)
        except KeyError:
            continue
        if valid is None or valid[r]:
            return n + d
    raise ValueError(f"no valid curve within +-{max_shift} of cycle {n}")


def curve(cell, field, n, substitute=True):
    """Cached curve `field` (Qdlin / Tdlin / dQdV) for cycle number n.
    If cycle n's curve is invalid (glitch/corrupt), the nearest valid cycle
    within +-2 is used instead (see `valid_cycle`)."""
    m = valid_cycle(cell, n) if substitute else n
    return cell["curves"][field][row_of_cycle(cell, m)].astype(np.float64)


def summary_upto(cell, field, n_max, n_min=1, clean=True):
    """Summary values for cycle numbers n_min..n_max (inclusive)."""
    s = cell["summary"]
    key = "QD_clean" if (field == "QD" and clean and "QD_clean" in s) else field
    m = (s["cycle"] >= n_min) & (s["cycle"] <= n_max)
    return s["cycle"][m], s[key][m]

"""Build the cleaned, combined dataset and the fixed splits.

Severson rules follow the authors' public code exactly
(`Load Data.ipynb` in github.com/rdbraatz/data-driven-prediction-of-battery-
cycle-life-before-capacity-degradation):
  * b1: drop b1c8, b1c10, b1c12, b1c13, b1c22 (never reach 80 % capacity)
  * b2: cells b2c7, b2c8, b2c9, b2c15, b2c16 are continuations of
        b1c0..b1c4 -> append their summaries, add `add_len` to cycle life,
        then drop them from b2
  * b3: drop noisy channels b3c37, b3c2, b3c23, b3c32, b3c42, b3c43
  * order = b1 cells, then b2, then b3 (insertion order) -> splits:
        test_ind  = [0, 2, 4, ..., nb1+nb2-2] + [83]   (primary test, 43)
        train_ind = [1, 3, ..., nb1+nb2-3]              (train, 41)
        secondary = last nb3 cells                       (40)
Our additional rules (documented in the report):
  * IR == 0 -> NaN (missing measurement)
  * chargetime >= 19.99 min (saturation cap) -> NaN
  * single-cycle QD glitches (|QD - rolling median(5)| > 0.02 Ah) -> NaN
    in a separate array `QD_clean`; the raw `QD` is kept untouched
  * unseen set = 2018-02-20 cells with a finite cycle life
  * truncated policy strings repaired ('4C(31%)-5' -> '4C(31%)-5C')
  * cells with life < 200 flagged as `anomalous` (the paper's 148-cycle outlier)
  * placeholder records (QD = QC = 0; all of b1's cycle 1) -> NaN in every
    summary field; their Qdlin rows are already NaN
  * Qdlin rows flagged invalid if non-finite, max outside 0.5-1.2 Ah, or on a
    QD-glitch cycle (corrupt curves, e.g. u1c5 cycle 48 max = 294 Ah);
    src.data.curve() then substitutes the nearest valid cycle within +-2
  * temperature probe suspect if mean(Tmax - Tmin) over cycles 2-100 < 2 C
  * label check `life_from_qd` = first cycle with QD_clean < 0.88, else
    n_cycles + 1 (b1 stopped each cell just before crossing 0.88 Ah)
Data notes: b1 cycle 1 is an empty record in every cell (why the paper's
features start at cycle 2); b1 has isolated QD spikes inside cycles 1-100
(b1c0 @12: 1.54 Ah, b1c18 @40: 2.88 Ah) and IR = 0 at cycle 12/13 in several
cells -> features must use QD_clean and NaN-aware IR. b2 has a batch-wide QD glitch at cycles ~246-258 (test pause?);
it lies outside cycles 1-100 so it does not affect features.

Usage:  python3 -m src.clean
Writes: data/processed/dataset.pkl, data/processed/splits.json
"""
import json
import pickle
import re

import numpy as np

from config import EOL_AH, PROC_DIR

B1_DROP = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]
B1_KEYS = ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]
B2_KEYS = ["b2c7", "b2c8", "b2c9", "b2c15", "b2c16"]
ADD_LEN = [662, 981, 1060, 208, 482]
B3_DROP = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]
CHARGETIME_CAP = 19.99
TEMP_SWING_MIN = 2.0   # deg C; mean (Tmax - Tmin) over cycles 2-100 below this -> probe suspect
GLITCH_AH = 0.02


def _load(name):
    with open(PROC_DIR / f"{name}_raw.pkl", "rb") as fp:
        return pickle.load(fp)


def _rolling_median(x, w=5):
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.nanmedian(xp[i:i + w]) for i in range(len(x))])


def _fix_policy(p):
    """Repair truncated policy strings, e.g. '4C(31%)-5' -> '4C(31%)-5C'."""
    p = p.strip()
    if re.search(r"-\d+(\.\d+)?$", p):
        p += "C"
    return p


def _apply_our_rules(cell):
    cell["policy"] = _fix_policy(cell["policy"])
    s = cell["summary"]
    # placeholder records (b1: cycle 1 has QD = QC = IR = chargetime = 0, Qdlin NaN)
    empty = (s["QD"] == 0) & (s["QC"] == 0)
    cell["n_empty_records"] = int(empty.sum())
    for f in s:
        if f != "cycle":
            s[f] = np.where(empty, np.nan, s[f])
    s["IR"] = np.where(s["IR"] == 0, np.nan, s["IR"])
    s["chargetime"] = np.where(s["chargetime"] >= CHARGETIME_CAP, np.nan, s["chargetime"])
    qd = s["QD"].copy()
    med = _rolling_median(qd)
    glitch = np.abs(qd - med) > GLITCH_AH
    qd[glitch] = np.nan
    s["QD_clean"] = qd
    cell["n_qd_glitches"] = int(glitch.sum())
    # curve validity: a cached Qdlin row is valid if finite, physically plausible
    # (0.5 < max < 1.2 Ah) and its cycle is not a QD glitch. Glitch cycles can have
    # corrupt curves (e.g. u1c5 cycle 48: max Qdlin = 294 Ah).
    Q = cell["curves"]["Qdlin"]
    n_rows = Q.shape[0]
    qmax = np.nanmax(np.where(np.isfinite(Q), Q, -np.inf), axis=1)
    valid = np.isfinite(Q).all(axis=1) & (qmax > 0.5) & (qmax < 1.2) & ~glitch[:n_rows]
    cell["curve_valid"] = valid
    # temperature probe check: a probe that lost contact reads ~chamber air
    # (~30 C) with almost no within-cycle swing (healthy cells swing ~6 C)
    early = (s["cycle"] >= 2) & (s["cycle"] <= 100)
    cell["temp_swing"] = float(np.nanmean((s["Tmax"] - s["Tmin"])[early]))
    cell["temp_suspect"] = cell["temp_swing"] < TEMP_SWING_MIN
    # label recomputed as in the authors' LoadData.m: first cycle with QD < 0.88,
    # else (test stopped just before crossing) number of cycles + 1
    below = np.where(s["QD_clean"] < EOL_AH)[0]
    cell["life_from_qd"] = float(s["cycle"][below[0]]) if below.size else float(len(s["QD"]) + 1)
    return cell


def build():
    b1, b2, b3 = (_load(n)["cells"] for n in ("b1", "b2", "b3"))
    u1_raw = _load("u1")
    vdlin = u1_raw["vdlin"]

    for k in B1_DROP:
        del b1[k]
    for bk, ck, add in zip(B1_KEYS, B2_KEYS, ADD_LEN):
        s1, s2 = b1[bk]["summary"], b2[ck]["summary"]
        n1 = len(s1["cycle"])
        for f in s1:
            extra = s2[f] + n1 if f == "cycle" else s2[f]
            s1[f] = np.hstack((s1[f], extra))
        b1[bk]["cycle_life"] = b1[bk]["cycle_life"] + add
        b1[bk]["merged_from"] = ck
    for k in B2_KEYS:
        del b2[k]
    for k in B3_DROP:
        del b3[k]

    nb1, nb2, nb3 = len(b1), len(b2), len(b3)
    order = list(b1) + list(b2) + list(b3)
    cells = {**b1, **b2, **b3}

    test_ind = np.hstack((np.arange(0, nb1 + nb2, 2), 83))
    train_ind = np.arange(1, nb1 + nb2 - 1, 2)
    sec_ind = np.arange(nb1 + nb2 + nb3 - nb3, nb1 + nb2 + nb3)

    u1 = {k: v for k, v in u1_raw["cells"].items() if np.isfinite(v["cycle_life"])}
    for c in list(cells.values()) + list(u1.values()):
        _apply_our_rules(c)

    splits = {
        "train": [order[i] for i in train_ind],
        "primary_test": [order[i] for i in test_ind],
        "secondary_test": [order[i] for i in sec_ind],
        "unseen": list(u1),
        "unseen_newstructure": [k for k, v in u1.items() if "newstructure" in v["policy"]],
        # paper's outlier (life 148, "reaches 80 % rapidly"); report results with & without
        "anomalous": [k for k in order if cells[k]["cycle_life"] < 200],
        # suspected detached temperature probes (paper excludes 4 cells from the
        # "full" model for this reason; it does not name them)
        "temp_suspect": [k for k in order + list(u1) if {**cells, **u1}[k]["temp_suspect"]],
        "counts": {"b1": nb1, "b2": nb2, "b3": nb3},
    }
    dataset = {"vdlin": vdlin, "cells": {**cells, **u1}, "order": order + list(u1)}
    with open(PROC_DIR / "dataset.pkl", "wb") as fp:
        pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROC_DIR / "splits.json", "w") as fp:
        json.dump(splits, fp, indent=1)
    print({k: len(v) for k, v in splits.items() if isinstance(v, list)}, splits["counts"])
    return dataset, splits


if __name__ == "__main__":
    build()

"""P1 checkpoint: verify the cleaned dataset against the paper's ground truth.

Ground truth = Supplementary Table 9 of Severson et al. (2019), parsed to
data/reference/si_table9_cells.json (barcode, split, batch date, cycle life, policy).
Barcodes cannot be decoded from the .mat files, so cells are matched on
(split, batch, cycle life, policy).

Usage:  python3 -m src.checks
"""
import json
from collections import Counter

import numpy as np

from config import ROOT
from src.data import load_dataset, load_splits, row_of_cycle

BATCH_DATE = {"b1": "2017-05-12", "b2": "2017-06-30", "b3": "2018-04-12"}
SPLIT_NAME = {"train": "train", "primary_test": "prim. test", "secondary_test": "sec. test"}


# naming differences between SI Table 9 and the .mat records (same cells)
POLICY_ALIASES = {"4.36C(80%)-4.36C": "4.8C(80%)-4.8C"}   # 2018-04-12 batch


def _norm_policy(p):
    p = p.replace(" ", "").split("-newstructure")[0].upper()
    return POLICY_ALIASES.get(p, p)


def main():
    ds, sp = load_dataset(), load_splits()
    cells = ds["cells"]
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok &= bool(cond)

    print("== split sizes")
    for k, n in [("train", 41), ("primary_test", 43), ("secondary_test", 40)]:
        check(len(sp[k]) == n, f"{k}: {len(sp[k])} (expected {n})")
    print(f"  info  unseen: {len(sp['unseen'])} (expected 39)")
    check(len(sp["unseen"]) == 39, "unseen = 39")
    allsets = [set(sp[k]) for k in ("train", "primary_test", "secondary_test", "unseen")]
    check(sum(map(len, allsets)) == len(set().union(*allsets)), "splits are disjoint")

    print("== match against SI Table 9 (split, batch, cycle life)")
    si = json.load(open(ROOT / "data" / "reference" / "si_table9_cells.json"))
    want = Counter((r["split"], r["batch"], int(r["life"])) for r in si)
    have = Counter()
    for k in ("train", "primary_test", "secondary_test"):
        for cid in sp[k]:
            c = cells[cid]
            have[(SPLIT_NAME[k], BATCH_DATE[c["batch"]], int(round(c["cycle_life"])))] += 1
    missing, extra = want - have, have - want
    check(not missing and not extra, f"exact match of all {sum(want.values())} cells")
    for m in list(missing.elements())[:10]:
        print("     missing (in paper, not ours):", m)
    for e in list(extra.elements())[:10]:
        print("     extra   (ours, not in paper):", e)
    # Policy strings: SI Table 9's policy column is misaligned for 2017-05-12 from
    # its 9th row on (it keeps the policies of the 5 dropped b1 cells, e.g. it lists
    # 5.4C(40%)-3.6C twice). The .mat record of each cell is authoritative.
    want_p = Counter((r["batch"], _norm_policy(r["policy"])) for r in si if r["batch"] != "2017-05-12")
    have_p = Counter((BATCH_DATE[cells[c]["batch"]], _norm_policy(cells[c]["policy"]))
                     for k in SPLIT_NAME for c in sp[k] if cells[c]["batch"] != "b1")
    check(want_p == have_p, "policies match SI for 2017-06-30 and 2018-04-12 "
          "(2017-05-12 skipped: SI policy column misaligned there)")

    print("== labels")
    sev = [cid for k in ("train", "primary_test", "secondary_test") for cid in sp[k]]
    lives = np.array([cells[c]["cycle_life"] for c in sev])
    print(f"  info  cycle life range {lives.min():.0f}-{lives.max():.0f} (paper 148-2237)")
    diff = [(c, cells[c]["cycle_life"], cells[c]["life_from_qd"]) for c in sev
            if np.isfinite(cells[c]["life_from_qd"]) and abs(cells[c]["cycle_life"] - cells[c]["life_from_qd"]) > 2]
    print(f"  info  stored life vs first-QD<0.88 cycle differ by >2 for {len(diff)} cells", diff[:5])

    print("== cycle indexing & early-cycle curves")
    bad_idx, bad_nan, offsets = [], [], Counter()
    for cid in sev + sp["unseen"]:
        c = cells[cid]
        try:
            r10, r100 = row_of_cycle(c, 10), row_of_cycle(c, 100)
            offsets[(c["batch"], r100 - 100)] += 1
            if not (np.isfinite(c["curves"]["Qdlin"][r10]).all() and np.isfinite(c["curves"]["Qdlin"][r100]).all()):
                bad_nan.append(cid)
        except KeyError:
            bad_idx.append(cid)
    check(not bad_idx, f"cycles 10 and 100 found in cache for every cell {bad_idx[:5]}")
    check(not bad_nan, f"Qdlin finite at cycles 10 & 100 for every cell {bad_nan[:5]}")
    print("  info  (batch, row(cycle100) - 100) counts:", dict(offsets))

    print("== our cleaning rules")
    subs = []
    from src.data import valid_cycle
    for cid in sev + sp["unseen"]:
        for n in (10, 100):
            m = valid_cycle(cells[cid], n)
            if m != n:
                subs.append((cid, n, m))
    print(f"  info  cycles 10/100 replaced by a neighbouring valid cycle: {subs}")
    print(f"  info  anomalous: {sp['anomalous']}; temperature-probe suspects: {sp['temp_suspect']}")
    n_ir0 = sum(np.isnan(cells[c]["summary"]["IR"][:100]).any() for c in sev + sp["unseen"])
    n_gl = sum(cells[c]["n_qd_glitches"] > 0 for c in sev + sp["unseen"])
    print(f"  info  cells with missing IR in first 100 cycles: {n_ir0}; cells with QD glitches: {n_gl}")

    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    main()

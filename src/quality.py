"""Per-batch data-quality report (run on the raw caches, before cleaning).

Usage:  python3 -m src.quality b1 [b2 b3 u1]
"""
import pickle
import sys
from collections import Counter

import numpy as np

from config import EOL_AH, PROC_DIR


def _row(s, n):
    r = np.where(s["cycle"] == n)[0]
    return int(r[0]) if r.size == 1 else None


def report(name):
    d = pickle.load(open(PROC_DIR / f"{name}_raw.pkl", "rb"))
    cells = d["cells"]
    print(f"\n######## {name}: {d['source']}  ({len(cells)} cells)")
    issues = Counter()
    glitch_cycles = []
    for k, c in cells.items():
        s, Q = c["summary"], c["curves"]["Qdlin"]
        msgs = []
        life = c["cycle_life"]
        below = np.where(s["QD"] < EOL_AH)[0]
        qdlife = below[0] + 1 if below.size else np.nan
        if not np.isfinite(life):
            msgs.append("life=NaN")
        elif np.isfinite(qdlife) and abs(qdlife - life) > 2:
            msgs.append(f"life {life:.0f} vs QD<0.88 at {qdlife:.0f}")
        elif not np.isfinite(qdlife):
            msgs.append(f"never<0.88 (life {life:.0f}, {len(s['QD'])} cycles)")
        if s["cycle"][0] != 1 or len(np.unique(s["cycle"])) != len(s["cycle"]):
            msgs.append(f"cycle numbering starts {s['cycle'][0]:.0f}, dups {len(s['cycle']) - len(np.unique(s['cycle']))}")
        r10, r100 = _row(s, 10), _row(s, 100)
        if r10 is None or r100 is None or r100 >= Q.shape[0]:
            msgs.append("cycle 10/100 not in cache")
        else:
            for n, r in ((10, r10), (100, r100)):
                if not np.isfinite(Q[r]).all():
                    msgs.append(f"Qdlin NaN@{n}")
                elif np.nanmax(Q[r]) < 0.5:
                    msgs.append(f"Qdlin max {np.nanmax(Q[r]):.2f}@{n}")
        early = (s["cycle"] >= 2) & (s["cycle"] <= 100)
        if (s["IR"][early] == 0).any():
            msgs.append(f"IR=0 x{int((s['IR'][early] == 0).sum())}")
        ir = s["IR"][early][s["IR"][early] > 0]
        if ir.size and (ir.max() > 0.03 or ir.min() < 0.01):
            msgs.append(f"IR range {ir.min():.4f}-{ir.max():.4f}")
        ct = s["chargetime"][(s["cycle"] >= 2) & (s["cycle"] <= 6)]
        if (ct >= 19.99).any() or (ct < 7).any() or (ct > 15).any():
            msgs.append(f"chargetime2-6 {np.round(ct, 1)}")
        ta = s["Tavg"][early]
        if ta.size > 1 and (np.max(np.abs(np.diff(ta))) > 2 or ta.min() < 20 or ta.max() > 45):
            msgs.append(f"Tavg {ta.min():.1f}-{ta.max():.1f}, jump {np.max(np.abs(np.diff(ta))):.1f}")
        qd = s["QD"]
        med = np.array([np.median(qd[max(0, i - 2):i + 3]) for i in range(len(qd))])
        g = np.where(np.abs(qd - med) > 0.02)[0]
        glitch_cycles += [int(s["cycle"][i]) for i in g]
        g_early = [int(s["cycle"][i]) for i in g if s["cycle"][i] <= 100]
        if g_early:
            msgs.append(f"QD glitch in cycles<=100 at {g_early}")
        for m in msgs:
            issues[m.split(" ")[0]] += 1
        if msgs:
            print(f"  {k:7s} {c['policy'][:28]:28s} life {life:6.0f} | " + "; ".join(msgs))
    hist = Counter((g // 10) * 10 for g in glitch_cycles)
    print(f"  -- QD glitches (any cycle): {len(glitch_cycles)} in {len(set(glitch_cycles))} distinct cycles;"
          f" busiest 10-cycle bins: {hist.most_common(3)}")
    lives = np.array([c["cycle_life"] for c in cells.values()])
    print(f"  -- life range {np.nanmin(lives):.0f}-{np.nanmax(lives):.0f}; NaN {np.isnan(lives).sum()};"
          f" issue counts {dict(issues)}")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        report(n)

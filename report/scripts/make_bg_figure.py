"""Background figure: one test cycle of b2c10, fresh (cycle 10) vs end of life (cycle 559).
Reads the raw .mat traces. Run from repo root:  python3 report/scripts/make_bg_figure.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.convert import read_raw_cycle  # noqa: E402

mpl.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 7.5})
INK2 = "#52514e"
CELL, CYCLES = 10, {10: ("#2a78d6", "cycle 10 (fresh)"), 559: ("#eb6834", "cycle 559 (end of life)")}

fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True)
for n, (col, lab) in CYCLES.items():
    c = read_raw_cycle("b2", CELL, n - 1)          # record index = cycle - 1
    t = c["t"] - c["t"][0]
    qd = np.nanmax(c["Qd"])
    a.plot(t, c["I"], color=col, lw=1.1, label=f"{lab}, $Q_d$ = {qd:.3f} Ah")
    b.plot(t, c["V"], color=col, lw=1.1)
a.axhline(0, color=INK2, lw=0.5)
a.set_ylabel("current (C-rate)")
a.set_ylim(-5.2, 7.2)
for x, y, s in ((0.5, 5.6, "fast charge\n3.6C(9%)–5C"), (16, 1.5, "1C"), (26.5, 0.6, "CV 3.6 V"),
                (36, -4.9, "4C discharge"), (50, -2.4, "CV 2.0 V")):
    a.text(x, y, s, fontsize=6.5, color=INK2)

for v, s in ((3.6, "3.6 V limit"), (2.0, "2.0 V limit")):
    b.axhline(v, color=INK2, lw=0.5, ls=":")
    b.text(61, v + 0.03, s, fontsize=6.5, color=INK2, ha="right", va="bottom")
b.set_ylabel("terminal voltage (V)")
h, l = a.get_legend_handles_labels()
b.legend(h, l, fontsize=6.5, loc="lower left")
b.set_xlabel("time in cycle (min)")
a.set_title("(a)", loc="left", fontsize=7.5)
b.set_title("(b)", loc="left", fontsize=7.5)
fig.tight_layout(h_pad=0.4)
out = ROOT / "report" / "figures" / "fig_cycle.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

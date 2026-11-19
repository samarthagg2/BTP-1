"""RQ4 figure (column width, 2 rows): test RMSE against the number of early cycles N.
Reads results/horizon.json. Run from repo root:  python3 report/scripts/make_rq4_figure.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import RESULTS_DIR  # noqa: E402

mpl.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 7.5})
INK2 = "#52514e"
H = json.load(open(RESULTS_DIR / "horizon.json"))["horizons"]
ns = sorted(int(n) for n in H)
STY = {"variance": ("F1", "#1baf7a", "o", "-"), "discharge": ("EN-F2", "#2a78d6", "s", "-"),
       "full": ("EN-F3", "#6250d6", "^", "-")}
CONST = {"primary_excl": 399, "secondary": 511}
CLIP = 560

fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.3), sharex=True)
for ax, key, t in ((axes[0], "primary_excl", "(a) Primary test (without b2c1)"),
                   (axes[1], "secondary", "(b) Secondary test")):
    for m, (lab, col, mk, ls) in STY.items():
        y = np.array([H[str(n)]["models"][m][key]["rmse"] for n in ns])
        ax.plot(ns, np.minimum(y, CLIP), color=col, marker=mk, ms=3, lw=1.1, ls=ls, label=lab)
    ax.axhline(CONST[key], color=INK2, lw=0.7, ls=":")
    ax.text(152, CONST[key] + 10, "constant", fontsize=6, color=INK2, ha="right")
    ax.set(ylim=(0, 590), ylabel="RMSE (cycles)")
    ax.set_title(t, loc="left", fontsize=7)
axes[1].set_xlabel("early cycles used, $N$")
axes[1].set_xticks(ns)
axes[1].legend(fontsize=6.3, loc="upper right", bbox_to_anchor=(1, 0.84), ncol=3, handlelength=1.8, columnspacing=1.0)
axes[0].annotate("EN-F2: 1829 at N = 50", (50, CLIP), xytext=(62, 480), fontsize=6, color="#2a78d6",
                 arrowprops=dict(arrowstyle="-", color="#2a78d6", lw=0.5))
fig.tight_layout(h_pad=0.5)
out = ROOT / "report" / "figures" / "fig_rq4_horizon.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

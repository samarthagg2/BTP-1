"""RQ1 figure (column width): parity plots of the three reproduced elastic-net models.
Reads results/p2_reproduction.json. Run from repo root:  python3 report/scripts/make_rq1_figure.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import RESULTS_DIR  # noqa: E402
from src.features import load_features  # noqa: E402

mpl.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 7.5})
INK2 = "#52514e"
STYLE = {"train": ("train", "#2a78d6", "o"), "primary_test": ("primary", "#eb6834", "s"),
         "secondary_test": ("secondary", "#1baf7a", "^")}
TITLES = {"variance": "(a) F1 variance", "discharge": "(b) F2 discharge", "full": "(c) F3 full"}

R = json.load(open(RESULTS_DIR / "p2_reproduction.json"))
df = load_features()
fig, grid = plt.subplots(2, 2, figsize=(3.5, 3.4))
axes = [grid[0, 0], grid[0, 1], grid[1, 0]]
lim = (120, 3000)
for ax, name in zip(axes, TITLES):
    d = df.join(pd.Series(R["predictions"][name], name="pred"), how="inner")
    for split, (lab, col, mk) in STYLE.items():
        s = d[d.split == split]
        ax.scatter(s.cycle_life, s.pred, s=10, color=col, marker=mk, lw=0, label=lab, zorder=3)
    an = d[d.anomalous]
    ax.scatter(an.cycle_life, an.pred, s=45, facecolor="none", edgecolor=INK2, lw=0.6, zorder=4)
    ax.plot(lim, lim, color=INK2, lw=0.6, ls="--")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mpl.ticker.FixedLocator([200, 500, 1000, 2000]))
        axis.set_major_formatter(mpl.ticker.FixedFormatter(["200", "500", "1k", "2k"]))
        axis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_title(TITLES[name], loc="left", fontsize=7.5)
    ax.set_xlabel("observed life")
    ax.set_ylabel("predicted life")
axes[0].annotate("b2c1", (148, 270), (300, 150), fontsize=6.5, color=INK2,
                 arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2))
leg = grid[1, 1]
leg.axis("off")
h, l = axes[0].get_legend_handles_labels()
leg.legend(h, l, loc="center", fontsize=7.5, markerscale=2, title="cells", title_fontsize=7.5)
fig.tight_layout(w_pad=0.6, h_pad=0.6)
out = ROOT / "report" / "figures" / "fig_rq1_parity.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

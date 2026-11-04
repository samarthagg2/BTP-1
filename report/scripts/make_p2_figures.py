"""P2 figures: parity plots of the three reproduced models (cf. paper Fig. 3).
Run from repo root:  python3 report/scripts/make_p2_figures.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import RESULTS_DIR  # noqa: E402
from src.features import load_features  # noqa: E402

mpl.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.color": "#e4e3df",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "font.size": 8.5})
INK2 = "#52514e"
STYLE = {"train": ("train", "#2a78d6", "o"),
         "primary_test": ("primary test", "#eb6834", "s"),
         "secondary_test": ("secondary test", "#1baf7a", "^")}
TITLES = {"variance": "Variance (1 feature)", "discharge": "Discharge (6)", "full": "Full (9)"}

R = json.load(open(RESULTS_DIR / "p2_reproduction.json"))
df = load_features()

fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharex=True, sharey=True)
lim = (100, 3000)
for ax, name in zip(axes, ("variance", "discharge", "full")):
    pred = pd.Series(R["predictions"][name], name="pred")
    d = df.join(pred, how="inner")
    for split, (lab, col, mk) in STYLE.items():
        s = d[d.split == split]
        ax.scatter(s.cycle_life, s.pred, s=22, color=col, marker=mk, edgecolor="white",
                   linewidth=0.6, label=lab, zorder=3)
    an = d[d.anomalous]
    ax.scatter(an.cycle_life, an.pred, s=90, facecolor="none", edgecolor=INK2, lw=0.8, zorder=4)
    ax.annotate("b2c1 (anomalous)", (an.cycle_life.iloc[0], an.pred.iloc[0]), xytext=(160, 900),
                fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--", zorder=2)
    sc = R["models"][name]["scores"]
    ax.set_title(f"{TITLES[name]}\nRMSE primary {sc['primary']['rmse']:.0f} "
                 f"({sc['primary_excl']['rmse']:.0f}), secondary {sc['secondary']['rmse']:.0f} cycles",
                 loc="left", fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("observed cycle life")
axes[0].set_ylabel("predicted cycle life")
axes[0].legend(loc="lower right", fontsize=7.5)
fig.tight_layout()
out = ROOT / "report" / "figures" / "fig_p2_parity.png"
fig.savefig(out, bbox_inches="tight")
print(out)

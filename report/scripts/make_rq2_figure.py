"""RQ2 figure (column width, 2 rows): (a) EN-F2 vs GAM-F2 test predictions (GAM fails beyond the training
range); (b) GP on F2: predictions with 95% intervals.
Reads results/p4_models.json. Run from repo root:  python3 report/scripts/make_rq2_figure.py
"""
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import RESULTS_DIR  # noqa: E402
from src.features import FEATURE_SETS, load_features  # noqa: E402
from src.models import gp_predict_std, make_regressor  # noqa: E402

mpl.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 7.5})
INK, INK2 = "#2b2a28", "#52514e"
SPLIT = {"train": ("train", "#2a78d6", "o"), "primary_test": ("primary", "#eb6834", "s"),
         "secondary_test": ("secondary", "#1baf7a", "^")}
T = json.load(open(RESULTS_DIR / "p4_models.json"))["test"]
df = load_features()

fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 4.6))
lim = (120, 3000)


def logaxes(ax, ylabel):
    ax.plot(lim, lim, color=INK2, lw=0.6, ls="--")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="observed life", ylabel=ylabel)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mpl.ticker.FixedLocator([200, 500, 1000, 2000]))
        axis.set_major_formatter(mpl.ticker.FixedFormatter(["200", "500", "1k", "2k"]))
        axis.set_minor_formatter(mpl.ticker.NullFormatter())


# (a) EN-F2 vs GAM-F2 on the test cells: the GAM bends down beyond the training range
PR = T["predictions"]
test = df[df.split.isin(["primary_test", "secondary_test"])]
for m, lab, col, mk in (("en_discharge", "EN-F2", INK, "o"), ("gam_discharge", "GAM-F2", "#d03b3b", "D")):
    pr = test.index.map(PR[m]).to_numpy(dtype=float)
    a.scatter(test.cycle_life, pr, color=col, marker=mk, s=11, lw=0, alpha=0.8, label=lab, zorder=3)
c = "b1c2"
a.annotate("b1c2", (df.cycle_life[c], PR["gam_discharge"][c]), xytext=(900, 330), fontsize=6.5, color=INK2,
           arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
a.axvspan(tr_life_max := df[df.split == "train"].cycle_life.max(), lim[1], color="#efeee9", zorder=0, lw=0)
logaxes(a, "predicted life")
a.legend(fontsize=6.5, loc="upper left", markerscale=1.3)
a.set_title("(a) EN-F2 and GAM-F2 on the test cells (shaded: beyond training lives)", loc="left", fontsize=6.8)

# (b) GP-F2 predictions with 95 % intervals
tr = df[df.split == "train"]
cols = FEATURE_SETS["discharge"]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gp = make_regressor("gp", len(cols)).fit(tr[cols].to_numpy(), np.log10(tr.cycle_life.to_numpy()))
paper = df[df.split != "unseen"]
mu, sd = gp_predict_std(gp, paper[cols].to_numpy())
for split, (lab, col, mk) in SPLIT.items():
    msk = (paper.split == split).to_numpy()
    obs, mid = paper.cycle_life.to_numpy()[msk], 10 ** mu[msk]
    lo, hi = 10 ** (mu[msk] - 1.96 * sd[msk]), 10 ** (mu[msk] + 1.96 * sd[msk])
    b.errorbar(obs, mid, yerr=[mid - lo, hi - mid], fmt=mk, ms=3, color=col, ecolor=col, elinewidth=0.6,
               alpha=0.9, label=lab, zorder=3)
logaxes(b, "GP-F2 prediction")
cv = T["gp_discharge"]["coverage_95"]
b.set_title(f"(b) GP-F2 with 95% interval: covers {100 * cv['primary']:.0f}% primary, "
            f"{100 * cv['secondary']:.0f}% secondary", loc="left", fontsize=7)
b.legend(fontsize=6.5, loc="lower right", markerscale=1.3)

fig.tight_layout(h_pad=0.8)
out = ROOT / "report" / "figures" / "fig_rq2_models.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

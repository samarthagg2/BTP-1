"""P4 figures: PLS diagnosis, in-batch vs new-batch trade-off with GP intervals, and the RQ4 horizon.
Reads results/p4_pls.json, results/p4_models.json, results/horizon.json.
Run from repo root:  python3 report/scripts/make_p4_figures.py
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
import pandas as pd  # noqa: E402

from config import RESULTS_DIR, VDLIN  # noqa: E402
from src.features import FEATURE_SETS, load_features  # noqa: E402
from src.models import gp_predict_std, make_regressor  # noqa: E402
from src.pls import bin_voltages, design_matrix  # noqa: E402

mpl.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 8.5})
INK2 = "#52514e"
SPLIT = {"train": ("train", "#2a78d6", "o"), "primary_test": ("primary test", "#eb6834", "s"),
         "secondary_test": ("secondary test", "#1baf7a", "^")}
FIG = ROOT / "report" / "figures"
P = json.load(open(RESULTS_DIR / "p4_pls.json"))
M = json.load(open(RESULTS_DIR / "p4_models.json"))
H = json.load(open(RESULTS_DIR / "horizon.json"))
df = load_features()


def title(ax, s):
    ax.set_title(s, loc="left", fontsize=8.5)


# --- fig_p4_pls: why the whole-curve PLS fails on the secondary batch
V = bin_voltages(VDLIN)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
for split, (lab, col, _) in SPLIT.items():
    X = design_matrix(df[df.split == split], "asinh")
    axes[0].plot(V, X.mean(0), color=col, lw=1.4, label=lab)
    axes[0].fill_between(V, np.percentile(X, 10, 0), np.percentile(X, 90, 0), color=col, alpha=0.12, lw=0)
axes[0].invert_xaxis(); axes[0].set(xlabel="voltage (V)", ylabel=r"asinh($\Delta Q$ / 1 mAh)")
axes[0].legend(fontsize=6.5, loc="lower left"); title(axes[0], "(a) Input curves: mean, 10-90%")
c = P["test"]["pls_asinh"]["coef_by_voltage"]
axes[1].plot(c["V"], c["coef"], color=INK2, lw=1.2); axes[1].axhline(0, color="#999", lw=0.6)
axes[1].invert_xaxis(); axes[1].set(xlabel="voltage (V)", ylabel="PLS coefficient")
title(axes[1], "(b) Weight per voltage bin")
fig.tight_layout(); fig.savefig(FIG / "fig_p4_pls.png", bbox_inches="tight"); plt.close(fig)

# --- fig_p4_models: (a) in-batch vs new-batch error for every configuration, (b) GP-F4 with 95% intervals
T = M["test"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.15]})
STY = {"en": ("#52514e", "o"), "gp": ("#1baf7a", "s"), "svr": ("#6250d6", "^"), "gam": ("#eb6834", "D")}
NAME = {"en": "EN", "gp": "GP", "svr": "SVR", "gam": "GAM"}
OFF = {"en_discharge": (5, -3), "en_physics": (-30, -10), "gp_discharge": (4, 2), "gp_physics": (5, 2),
       "svr_discharge": (5, -2), "svr_physics": (5, 2), "gam_discharge": (5, -8), "gam_physics": (5, 2)}
for k, v in T.items():
    if k in ("bootstrap", "predictions"):
        continue
    m, fs = k.split("_", 1)
    col, mk = STY[m]
    x, y = v["robust"]["primary"]["rmse_in_range"], v["scores"]["secondary"]["rmse"]
    axes[0].scatter(x, y, color=col, marker=mk, s=40 if fs == "physics" else 26,
                    facecolor=col if fs == "physics" else "white", lw=1.2, zorder=3)
    axes[0].annotate(f"{NAME[m]}-{'F4' if fs == 'physics' else 'F2'}", (x, y), xytext=OFF[k], textcoords="offset points", fontsize=6.5)
axes[0].set(xlabel="primary RMSE, in training range (cycles)", ylabel="secondary RMSE (cycles)", ylim=(155, 215))
title(axes[0], "(a) Familiar batches vs new batch\nfilled: F4, open: F2")
paper = df[df.split != "unseen"]
tr = df[df.split == "train"]
cols = FEATURE_SETS["physics"]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gp = make_regressor("gp", len(cols)).fit(tr[cols].to_numpy(), np.log10(tr.cycle_life.to_numpy()))
mu, sd = gp_predict_std(gp, paper[cols].to_numpy())
for split, (lab, col, mk) in SPLIT.items():
    msk = (paper.split == split).to_numpy()
    obs = paper.cycle_life.to_numpy()[msk]
    lo_, hi_ = 10 ** (mu[msk] - 1.96 * sd[msk]), 10 ** (mu[msk] + 1.96 * sd[msk])
    axes[1].errorbar(obs, 10 ** mu[msk], yerr=[10 ** mu[msk] - lo_, hi_ - 10 ** mu[msk]], fmt=mk, ms=3.5, color=col,
                     ecolor=col, elinewidth=0.6, alpha=0.85, label=lab)
lim = (100, 3000)
axes[1].plot(lim, lim, color=INK2, lw=0.8, ls="--"); axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set(xlim=lim, ylim=lim, xlabel="observed cycle life", ylabel="GP-F4 prediction (95% interval)")
cv = T["gp_physics"]["coverage_95"]
title(axes[1], f"(b) GP on F4: interval covers {100 * cv['primary']:.0f}% primary,\n{100 * cv['secondary']:.0f}% secondary")
axes[1].legend(fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "fig_p4_models.png", bbox_inches="tight"); plt.close(fig)

# --- fig_p4_horizon: test RMSE against the number of cycles N
ns = sorted(int(n) for n in H["horizons"])
HSTY = {"discharge": ("EN F2 (discharge)", "#2a78d6", "s", "-"), "full": ("EN F3 (full)", "#6250d6", "^", "-"),
        "physics": ("EN F4", "#eb6834", "D", "-"), "gp_physics": ("GP F4", "#1baf7a", "o", "--")}
const = {"primary_excl": 399, "secondary": 511}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, key, t in ((axes[0], "primary_excl", "(a) Primary test (without b2c1)"), (axes[1], "secondary", "(b) Secondary test")):
    for m, (lab, col, mk, ls) in HSTY.items():
        y = np.array([H["horizons"][str(n)]["models"][m][key]["rmse"] for n in ns])
        ax.plot(ns, np.minimum(y, 560), color=col, marker=mk, ms=3.5, lw=1.2, ls=ls, label=lab)
    ax.axhline(const[key], color=INK2, lw=0.8, ls=":")
    ax.text(150, const[key] + 8, "constant", fontsize=6.5, color=INK2, ha="right")
    ax.set_ylim(0, 580); ax.set_xlabel("cycles used, $N$"); title(ax, t)
axes[0].set_ylabel("test RMSE (cycles)"); axes[0].legend(fontsize=6.5, loc="upper right")
axes[0].annotate("EN F2: 1829 / 428\nat N = 50 / 60", (55, 560), xytext=(62, 470), fontsize=6.3, color="#2a78d6",
                 arrowprops=dict(arrowstyle="-", color="#2a78d6", lw=0.6))
fig.tight_layout(); fig.savefig(FIG / "fig_p4_horizon.png", bbox_inches="tight"); plt.close(fig)
print("wrote fig_p4_pls / fig_p4_models / fig_p4_horizon")

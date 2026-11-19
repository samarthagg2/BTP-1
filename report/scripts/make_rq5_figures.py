"""RQ5 figures (column width): frozen models on the unseen batch u1, and the SHAP decomposition of the mean
prediction per test set. Reads results/p5_unseen.json, results/p5_shap.json.
Run from repo root:  python3 report/scripts/make_rq5_figures.py
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

mpl.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 7})
INK2 = "#52514e"
FIG = ROOT / "report" / "figures"
df = load_features()
tr = df[df.split == "train"]

# --- fig_rq5_parity: 2x2 parity on u1 for four frozen models
U = json.load(open(RESULTS_DIR / "p5_unseen.json"))
P = pd.DataFrame(U["predictions"])
uu = df.loc[P.index]
grp = np.where(uu.policy.str.contains("newstructure"), "newstructure", "standard")
fig, grid = plt.subplots(2, 2, figsize=(3.5, 3.5), sharex=True, sharey=True)
lim = (300, 2500)
for ax, (m, lab) in zip(grid.flat, (("en_variance", "(a) F1"), ("en_discharge", "(b) EN-F2"),
                                    ("en_full", "(c) EN-F3"), ("gp_discharge", "(d) GP-F2"))):
    for g, col, mk in (("standard", "#d03b3b", "o"), ("newstructure", "#6250d6", "s")):
        msk = grp == g
        ax.scatter(uu.cycle_life[msk], P[m][msk], color=col, marker=mk, s=12, edgecolor="white", lw=0.3,
                   label=g, zorder=3)
    ax.plot(lim, lim, color=INK2, lw=0.6, ls="--")
    ax.axhline(10 ** np.mean(np.log10(tr.cycle_life)), color="#8a8984", lw=0.6, ls=":")
    s = U["scores"][m]
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mpl.ticker.FixedLocator([300, 500, 1000, 2000]))
        axis.set_major_formatter(mpl.ticker.FixedFormatter(["300", "500", "1k", "2k"]))
        axis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_title(lab, loc="left", fontsize=7)
    ax.text(0.97, 0.04, f"RMSE {s['standard']['rmse']:.0f} / {s['newstructure']['rmse']:.0f}",
            transform=ax.transAxes, ha="right", fontsize=6, color=INK2)
for ax in grid[1]:
    ax.set_xlabel("observed life")
for ax in grid[:, 0]:
    ax.set_ylabel("predicted life")
grid[0, 1].legend(fontsize=6, loc="upper left", handletextpad=0.2, markerscale=1.3)
fig.tight_layout(w_pad=0.4, h_pad=0.6)
fig.savefig(FIG / "fig_rq5_parity.pdf", bbox_inches="tight"); plt.close(fig)

# --- fig_rq5_shap: mean SHAP per test set (stacked by feature group) vs the actual mean
S = json.load(open(RESULTS_DIR / "p5_shap.json"))
SETS = ["primary", "secondary", "u1_standard", "u1_newstructure"]
XL = ["prim.", "sec.", "u1\nstd", "u1\nnew"]
GROUPS = {"ΔQ min, var": ["dq_log_min", "dq_log_var"], "ΔQ skew, kurt": ["dq_log_skew", "dq_log_kurt"],
          "capacity": ["qd_2", "qd_max_minus_2", "qd_slope_2_100", "qd_icpt_2_100"],
          "temperature integral": ["temp_integral_2_100"], "charge time": ["chargetime_avg_2_6"],
          "IR": ["ir_min_2_100", "ir_100_minus_2"], "margin_rate": ["margin_rate"],
          "overhang_after10": ["overhang_after10"]}
COL = {"ΔQ min, var": "#2a78d6", "ΔQ skew, kurt": "#9cc3f0", "capacity": "#8a8984",
       "temperature integral": "#d03b3b", "charge time": "#e8a33d", "IR": "#6250d6", "margin_rate": "#1baf7a",
       "overhang_after10": "#eb6834"}
u = df[df.split == "unseen"]
ns = u.policy.str.contains("newstructure")
cells = {"primary": df[df.split == "primary_test"], "secondary": df[df.split == "secondary_test"],
         "u1_standard": u[~ns], "u1_newstructure": u[ns]}
fig, grid = plt.subplots(1, 2, figsize=(3.5, 2.6), sharey=True)
for ax, (m, t) in zip(grid.flat, (("EN-F2", "(a) EN-F2"), ("EN-F3", "(b) EN-F3"))):
    r = S[m]
    base = r["base_value"]
    for i, s in enumerate(SETS):
        mean = r["sets"][s]["mean"]
        pos = neg = 0.0
        for g, feats in GROUPS.items():
            if not any(f in mean for f in feats):
                continue
            v = sum(mean.get(f, 0.0) for f in feats)
            if abs(v) < 1e-4:
                continue
            if v >= 0:
                ax.bar(i, v, bottom=pos, color=COL[g], width=0.62, label=g, edgecolor="white", lw=0.3); pos += v
            else:
                ax.bar(i, v, bottom=neg, color=COL[g], width=0.62, label=g, edgecolor="white", lw=0.3); neg += v
        d = cells[s]
        if m == "EN-F3":
            d = d[~d.temp_suspect]
        actual = np.log10(d.cycle_life).mean() - base
        ax.plot([i - 0.38, i + 0.38], [actual, actual], color="black", lw=1.6, zorder=5,
                label="actual mean")
        ax.scatter([i], [pos + neg], marker="o", s=14, color="white", edgecolor="black", lw=0.8, zorder=6,
                   label="predicted mean")
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xticks(range(len(SETS))); ax.set_xticklabels(XL, fontsize=6)
    ax.set_title(t, loc="left", fontsize=7)
grid[0].set_ylabel(r"mean SHAP ($\log_{10}$ life)")
h, l = [], []
for ax in grid.flat:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l:
            h.append(hh); l.append(ll)
fig.tight_layout(w_pad=0.4, rect=(0, 0.2, 1, 1))
fig.legend(h, l, loc="lower center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, 0.0), handlelength=1.2,
           columnspacing=0.8)
fig.savefig(FIG / "fig_rq5_shap.pdf", bbox_inches="tight"); plt.close(fig)
print("wrote fig_rq5_parity / fig_rq5_shap")


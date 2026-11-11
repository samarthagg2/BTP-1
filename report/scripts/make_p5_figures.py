"""P5 figures: SHAP decomposition of the mean prediction per test set, and the overhang dependence.
Reads results/p5_shap.json and results/p5_shap_values.csv (python3 -m src.p5_shap).
Run from repo root:  python3 report/scripts/make_p5_figures.py
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
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 8.5})
INK2 = "#52514e"
FIG = ROOT / "report" / "figures"
S = json.load(open(RESULTS_DIR / "p5_shap.json"))
V = pd.read_csv(RESULTS_DIR / "p5_shap_values.csv", index_col=0)
df = load_features()
SETS = ["primary", "secondary", "u1_standard", "u1_newstructure"]
LAB = {"primary": "primary", "secondary": "secondary\n(b3)", "u1_standard": "u1\nstandard", "u1_newstructure": "u1\nnewstructure"}
GROUPS = {  # merge near-duplicate features for readability
    "ΔQ (min, var)": ["dq_log_min", "dq_log_var"], "ΔQ shape (skew, kurt)": ["dq_log_skew", "dq_log_kurt"],
    "capacity (QD2, max−QD2, fade)": ["qd_2", "qd_max_minus_2", "qd_slope_2_100", "qd_icpt_2_100"],
    "temperature integral": ["temp_integral_2_100"], "charge time": ["chargetime_avg_2_6"],
    "IR": ["ir_min_2_100", "ir_100_minus_2"], "margin_rate": ["margin_rate"], "overhang_after10": ["overhang_after10"]}
COL = {"ΔQ (min, var)": "#2a78d6", "ΔQ shape (skew, kurt)": "#9cc3f0", "capacity (QD2, max−QD2, fade)": "#8a8984",
       "temperature integral": "#d03b3b", "charge time": "#e8a33d", "IR": "#6250d6", "margin_rate": "#1baf7a",
       "overhang_after10": "#eb6834"}
u = df[df.split == "unseen"]
ns = u.policy.str.contains("newstructure")
cells = {"primary": df[df.split == "primary_test"], "secondary": df[df.split == "secondary_test"],
         "u1_standard": u[~ns], "u1_newstructure": u[ns]}

# --- fig_p5_shap: stacked mean SHAP per set vs the actual mean offset of log10 life
models = ["EN-F2", "EN-F3", "EN-F4", "GP-F4"]
fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
for ax, m in zip(axes, models):
    r = S[m]
    base = r["base_value"]
    for i, s in enumerate(SETS):
        mean = r["sets"][s]["mean"]
        pos, neg = 0.0, 0.0
        for g, feats in GROUPS.items():
            v = sum(mean.get(f, 0.0) for f in feats)
            if not any(f in mean for f in feats) or abs(v) < 1e-4:
                continue
            if v >= 0:
                ax.bar(i, v, bottom=pos, color=COL[g], width=0.6, label=g); pos += v
            else:
                ax.bar(i, v, bottom=neg, color=COL[g], width=0.6, label=g); neg += v
        d = cells[s]
        if m == "EN-F3":
            d = d[~d.temp_suspect]
        actual = np.log10(d.cycle_life).mean() - base
        ax.scatter([i], [actual], marker="_", s=400, color="black", lw=2, zorder=5, label="actual mean")
        ax.scatter([i], [pos + neg], marker="o", s=18, color="white", edgecolor="black", zorder=6, label="predicted mean")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(SETS))); ax.set_xticklabels([LAB[s] for s in SETS], fontsize=7)
    ax.set_title(f"{m} (base {10 ** base:.0f} cycles)", loc="left", fontsize=8.5)
axes[0].set_ylabel("mean SHAP, log$_{10}$ life vs base")
h, l = [], []
for ax in axes:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l:
            h.append(hh); l.append(ll)
fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(); fig.savefig(FIG / "fig_p5_shap.png", bbox_inches="tight"); plt.close(fig)

# --- fig_p5_overhang: SHAP of overhang_after10 against its value, EN-F4 vs GP-F4
tr = df[df.split == "train"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
STY = {"train": ("train", "#2a78d6", "o"), "primary": ("primary", "#eb6834", "s"), "secondary": ("secondary", "#1baf7a", "^"),
       "u1_standard": ("u1 standard", "#d03b3b", "D"), "u1_newstructure": ("u1 newstructure", "#6250d6", "v")}
for ax, m in zip(axes, ("EN-F4", "GP-F4")):
    d = V[V.model == m]
    for s, (lab, col, mk) in STY.items():
        x = d[d.set == s]
        vals = df.loc[x.index, "overhang_after10"]
        ax.scatter(vals, x["overhang_after10"], color=col, marker=mk, s=14, edgecolor="white", lw=0.3, label=lab)
    ax.axvspan(tr.overhang_after10.min(), tr.overhang_after10.max(), color="#f3f2ee", zorder=0)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set(xlabel="overhang_after10 (mAh)"); ax.set_title(f"{m}", loc="left", fontsize=8.5)
axes[0].set_ylabel("SHAP of overhang_after10 (log$_{10}$ life)")
axes[0].legend(fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(FIG / "fig_p5_overhang.png", bbox_inches="tight"); plt.close(fig)
print("wrote fig_p5_shap / fig_p5_overhang")

# --- fig_p5_parity: frozen models on the unseen batch u1
U = json.load(open(RESULTS_DIR / "p5_unseen.json"))
P = pd.DataFrame(U["predictions"])
uu = df.loc[P.index]
grp = np.where(uu.policy.str.contains("newstructure"), "newstructure", "standard")
fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), sharex=True, sharey=True)
lim = (250, 2500)
for ax, (m, lab) in zip(axes, (("en_discharge", "EN-F2 (discharge)"), ("en_full", "EN-F3 (full)"),
                               ("gp_physics", "GP-F4"), ("en_full_physics", "EN F3+F4"))):
    for g, col, mk in (("standard", "#d03b3b", "o"), ("newstructure", "#6250d6", "s")):
        msk = grp == g
        ax.scatter(uu.cycle_life[msk], P[m][msk], color=col, marker=mk, s=18, edgecolor="white", lw=0.4, label=g)
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
    ax.axhline(10 ** np.mean(np.log10(tr.cycle_life)), color="#8a8984", lw=0.6, ls=":")
    s = U["scores"][m]
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("observed cycle life")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mpl.ticker.FixedLocator([300, 500, 1000, 2000]))
        axis.set_major_formatter(mpl.ticker.ScalarFormatter()); axis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_title(f"{lab}\nRMSE {s['all']['rmse']:.0f}: std {s['standard']['rmse']:.0f}, new {s['newstructure']['rmse']:.0f}",
                 loc="left", fontsize=8)
axes[0].set_ylabel("predicted cycle life"); axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig_p5_parity.png", bbox_inches="tight"); plt.close(fig)
print("wrote fig_p5_parity")

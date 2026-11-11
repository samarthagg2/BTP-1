"""P3 figures: ΔQ trajectories, the overhang effect, and the margin/rate feature.
Training cells only (the test sets are not looked at in Phase 3).
Run from repo root:  python3 report/scripts/make_p3_figures.py
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from config import RESULTS_DIR, VDLIN  # noqa: E402
from src.data import curve, load_dataset  # noqa: E402
from src.features import SMOOTH_WIN, load_features  # noqa: E402

mpl.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 8.5})
INK2, C1, C2 = "#52514e", "#2a78d6", "#eb6834"
BATCH_STYLE = {"b1": ("b1 (2017-05-12)", C1, "o"), "b2": ("b2 (2017-06-30)", C2, "s")}
BLUES = LinearSegmentedColormap.from_list("blues", ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
FIG = ROOT / "report" / "figures"
cells = load_dataset()["cells"]
df = load_features()
tr = df[df.split == "train"]
S = json.load(open(RESULTS_DIR / "p3_screening.json"))
norm = LogNorm(tr.cycle_life.min(), tr.cycle_life.max())


def title(ax, s):
    ax.set_title(s, loc="left", fontsize=8.5)


# --- fig_p3_traj: ΔQ amplitude grows linearly over cycles 10-100
NS = np.array(S["trajectory"]["cycles"])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
for cid, r in tr.sort_values("cycle_life", ascending=False).iterrows():
    q10 = curve(cells[cid], "Qdlin", 10)
    amp = [1000 * np.std(curve(cells[cid], "Qdlin", n) - q10, ddof=1) for n in NS]
    axes[0].plot(NS - 10, amp, color=BLUES(norm(r.cycle_life)), lw=0.8, marker="o", ms=2)
x = np.array([10, 90])
axes[0].plot(x, 15.5 * x / 90, color=INK2, lw=1, ls="--", label="slope 1 (constant rate)")
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set(xlabel="cycles after cycle 10", ylabel=r"sd of $\Delta Q_{n-10}$ (mAh)")
axes[0].legend(fontsize=7, loc="upper left")
title(axes[0], "(a) Growth of the curve change, 41 training cells")
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=BLUES), ax=axes[0], label="cycle life")
p = pd.read_csv(RESULTS_DIR / "p3_candidates_train.csv", index_col=0)["traj_exponent"]
axes[1].hist(p, bins=np.arange(0, 1.45, 0.1), color=C1, edgecolor="white")
axes[1].axvline(1, color=INK2, lw=1, ls="--")
axes[1].set(xlabel="fitted exponent $p$", ylabel="cells")
title(axes[1], f"(b) Exponent: median {p.median():.2f}")
fig.tight_layout(); fig.savefig(FIG / "fig_p3_traj.png", bbox_inches="tight"); plt.close(fig)


# --- fig_p3_overhang: the early rise, where it shows in ΔQ, and its effect
def smooth_capacity(c):
    s = c["summary"]; cyc = s["cycle"]; q = s["QD_clean"]
    m = (cyc >= 2) & (cyc <= 100) & np.isfinite(q)
    sm_q = pd.Series(q[m]).rolling(SMOOTH_WIN, center=True, min_periods=4).median().to_numpy()
    return cyc[m], q[m], sm_q


fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
for cid, col in (("b1c1", C1), ("b2c33", C2), ("b1c21", "#6250d6")):
    c = cells[cid]; cyc, q, sm_q = smooth_capacity(c)
    axes[0].plot(cyc, 1000 * (q - np.nanmedian(q[cyc <= 4])), color=col, lw=0.6, alpha=0.4)
    axes[0].plot(cyc, 1000 * (sm_q - np.nanmedian(q[cyc <= 4])), color=col, lw=1.6,
                 label=f"{cid} (life {c['cycle_life']:.0f}; after-10 rise {df.loc[cid, 'overhang_after10']:.1f} mAh)")
axes[0].axvline(10, color=INK2, lw=0.8, ls=":")
axes[0].set(xlabel="cycle", ylabel="QD - QD(2-4) (mAh)", ylim=(-12, 8))
axes[0].legend(fontsize=6.3, loc="lower left")
title(axes[0], "(a) Early capacity rise (overhang)")
for cid, r in tr.sort_values("cycle_life", ascending=False).iterrows():
    c = cells[cid]
    axes[1].plot(1000 * (curve(c, "Qdlin", 40) - curve(c, "Qdlin", 2)), VDLIN, color=BLUES(norm(r.cycle_life)), lw=0.7)
axes[1].axvline(0, color=INK2, lw=0.8); axes[1].set_xlim(-25, 10)
axes[1].set(xlabel=r"$\Delta Q_{40-2}$ (mAh)", ylabel="voltage (V)")
title(axes[1], "(b) Gain at the low-voltage end")
y = np.log10(tr.cycle_life.to_numpy())
for b, (lab, col, mk) in BATCH_STYLE.items():
    d = tr[tr.batch == b]; yy = np.log10(d.cycle_life.to_numpy())
    s1, i1 = np.polyfit(d.dq_log_var, yy, 1)
    res = 100 * (10 ** (yy - (i1 + s1 * d.dq_log_var.to_numpy())) - 1)
    axes[2].scatter(d.overhang_after10, res, color=col, marker=mk, s=18, edgecolor="white", lw=0.5, label=lab)
axes[2].axhline(0, color=INK2, lw=0.8)
axes[2].set(xlabel="rise after cycle 10 (mAh)", ylabel="life vs. ΔQ-only fit in its batch (%)")
axes[2].legend(fontsize=7)
title(axes[2], "(c) Larger late rise: shorter than ΔQ predicts")
fig.tight_layout(); fig.savefig(FIG / "fig_p3_overhang.png", bbox_inches="tight"); plt.close(fig)

# --- fig_p3_margin: ΔQ variance vs margin/rate
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, col_name, xl in ((axes[0], "dq_log_var", r"$\log_{10}\mathrm{Var}(\Delta Q_{100-10})$"),
                         (axes[1], "margin_rate", r"$\log_{10}\,[(Q_{D,2}-0.88)/\mathrm{sd}(\Delta Q)]$")):
    for b, (lab, col, mk) in BATCH_STYLE.items():
        d = tr[tr.batch == b]
        ax.scatter(d[col_name], d.cycle_life, color=col, marker=mk, s=18, edgecolor="white", lw=0.5, label=lab)
    r = np.corrcoef(tr[col_name], y)[0, 1]
    ax.set_yscale("log"); ax.set_xlabel(xl)
    title(ax, f"{'(a) ΔQ variance' if col_name == 'dq_log_var' else '(b) Margin / rate'}: r = {r:+.2f}")
axes[0].set_ylabel("cycle life"); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig_p3_margin.png", bbox_inches="tight"); plt.close(fig)
print("wrote fig_p3_traj / fig_p3_overhang / fig_p3_margin")

# --- fig_p3_parity: F4 vs the discharge model on the paper's splits (test sets used once, F4 frozen)
import warnings  # noqa: E402

from src.features import FEATURE_SETS  # noqa: E402
from src.reproduce import fit_one  # noqa: E402

T = json.load(open(RESULTS_DIR / "p3_test_eval.json"))
STYLE = {"train": ("train", "#2a78d6", "o"), "primary_test": ("primary test", "#eb6834", "s"),
         "secondary_test": ("secondary test", "#1baf7a", "^")}
paper = df[df.split != "unseen"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True, sharey=True)
lim = (100, 3000)
for ax, name, lab in ((axes[0], "discharge", "Discharge (6 features)"), (axes[1], "physics", "F4 physics (2 features)")):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = fit_one(df, name)[0]
    pred = pd.Series(10 ** m.predict(paper[FEATURE_SETS[name]].to_numpy()), index=paper.index)
    for split, (sl, col, mk) in STYLE.items():
        d = paper[paper.split == split]
        ax.scatter(d.cycle_life, pred[d.index], color=col, marker=mk, s=16, edgecolor="white", lw=0.4, label=sl, zorder=3)
    for cid in ("b1c2", "b2c1"):
        ax.scatter([paper.cycle_life[cid]], [pred[cid]], s=80, facecolor="none", edgecolor=INK2, lw=0.8, zorder=4)
        ax.annotate(cid, (paper.cycle_life[cid], pred[cid]), xytext=(6, -10), textcoords="offset points", fontsize=7, color=INK2)
    lo_, hi_ = T["train_life_range"]
    ax.axvspan(lim[0], lo_, color="#f3f2ee", zorder=0); ax.axvspan(hi_, lim[1], color="#f3f2ee", zorder=0)
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
    s = T["models"][name]["scores"]; r = T["models"][name]["robust"]["primary"]
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("observed cycle life")
    title(ax, f"{lab}\nprimary {s['primary']['rmse']:.0f} (in range {r['rmse_in_range']:.0f}), secondary {s['secondary']['rmse']:.0f} cycles")
axes[0].set_ylabel("predicted cycle life"); axes[0].legend(fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "fig_p3_parity.png", bbox_inches="tight"); plt.close(fig)
print("wrote fig_p3_parity")

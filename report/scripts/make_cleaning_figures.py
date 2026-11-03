"""Figures for the Data Cleaning section. Needs data/processed/dataset.pkl.

Run from the repo root:  python3 report/scripts/make_cleaning_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_dataset, load_splits  # noqa: E402

OUT = ROOT / "report" / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
BAD = "#d03b3b"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
COL_W, FULL_W = 3.5, 7.16
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

ds, sp = load_dataset(), load_splits()
cells = ds["cells"]
stats = {}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def qdlin_max(c):
    Q = c["curves"]["Qdlin"]
    return np.array([np.nanmax(r) if np.isfinite(r).any() else np.nan for r in Q])


# ------------------------------------------------ Fig A: dataset overview (4 sets)
sets = [("train", "Train (41)"), ("primary_test", "Primary test (43)"),
        ("secondary_test", "Secondary test (40)"), ("unseen", "Unseen 2018-02-20 (39)")]
all_l = [cells[c]["cycle_life"] for k, _ in sets for c in sp[k]]
norm = LogNorm(min(all_l), max(all_l))
fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 1.9), sharey=True)
for ax, (k, title) in zip(axes, sets):
    for cid in sorted(sp[k], key=lambda c: -cells[c]["cycle_life"]):
        s = cells[cid]["summary"]
        ax.plot(s["cycle"], s["QD_clean"], color=BLUES(norm(cells[cid]["cycle_life"])), lw=0.5)
    ax.axhline(0.88, color=INK2, ls="--", lw=0.6)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Cycle number")
    ax.set_xlim(0, 2300)
    ax.set_ylim(0.85, 1.12)
axes[0].set_ylabel("Discharge capacity (Ah)")
sm = mpl.cm.ScalarMappable(norm=norm, cmap=BLUES)
cb = fig.colorbar(sm, ax=axes, pad=0.01, fraction=0.025)
cb.set_label("Cycle life")
cb.outline.set_visible(False)
cb.set_ticks([200, 500, 1000, 2000])
cb.set_ticklabels(["200", "500", "1000", "2000"])
cb.ax.minorticks_off()
save(fig, "fig_dataset_overview")

# ------------------------------------------------ Fig B: glitches (3 panels)
fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.0))
# (a) b1c18 spike inside cycles 1-100
c = cells["b1c18"]; s = c["summary"]
m = s["cycle"] <= 100
raw = np.where(np.isfinite(s["QD"]), s["QD"], np.nan)
ax = axes[0]
ax.plot(s["cycle"][m], np.clip(raw[m], None, 1.10), color=BAD, lw=0.8, label="raw (clipped)")
ax.plot(s["cycle"][m], s["QD_clean"][m], color=C1, lw=1.0, label="cleaned")
ax.annotate("2.88 Ah\n(off scale)", xy=(40, 1.10), xytext=(55, 1.085), fontsize=6.5,
            color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
ax.set_ylim(1.060, 1.102)
ax.set_title("(a) b1c18: single-cycle spike", loc="left")
ax.set_xlabel("Cycle number"); ax.set_ylabel("Discharge capacity (Ah)")
ax.legend(loc="lower left")
# (b) batch-wide dips in the unseen batch
ax = axes[1]
picks = ["u1c2", "u1c5", "u1c10", "u1c21", "u1c33"]
for cid, col in zip(picks, [C1, C2, C3, C4, INK2]):
    s = cells[cid]["summary"]; m = (s["cycle"] >= 30) & (s["cycle"] <= 90)
    ax.plot(s["cycle"][m], s["QD"][m], color=col, lw=0.8, label=cid)
for x in (47.5, 70):
    ax.axvspan(x - 2.5, x + 4.5, color="#f0efec", lw=0, zorder=0)
ax.set_title("(b) 2018-02-20: batch-wide dips", loc="left")
ax.set_xlabel("Cycle number")
ax.set_ylim(0.9, 1.13)
ax.legend(loc="lower left", ncol=2)
# (c) corrupt curve at the glitch cycle
ax = axes[2]
c = cells["u1c5"]; s = c["summary"]; qm = qdlin_max(c)
n = min(len(qm), 100)
ax.semilogy(s["cycle"][1:n], qm[1:n], color=C1, lw=0.9, marker="o", ms=1.5)
ax.axhspan(0.5, 1.2, color="#cde2fb", lw=0, zorder=0)
ax.text(60, 1.6, "valid band 0.5-1.2 Ah", fontsize=6.5, color=INK2)
r48 = int(np.where(s["cycle"] == 48)[0][0])
ax.annotate(f"cycle 48: {qm[r48]:.0f} Ah", xy=(48, qm[r48]), xytext=(58, qm[r48] * 0.6),
            fontsize=6.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
ax.set_title("(c) u1c5: corrupt Q(V) curve", loc="left")
ax.set_xlabel("Cycle number"); ax.set_ylabel("max of $Q_d(V)$ curve (Ah)")
fig.tight_layout(w_pad=1.0)
save(fig, "fig_clean_glitches")

# ------------------------------------------------ Fig C: temperature-probe check
fig, ax = plt.subplots(figsize=(COL_W, 1.9))
order = [("b1", "2017-05-12"), ("b2", "2017-06-30"), ("b3", "2018-04-12"), ("u1", "2018-02-20")]
ids_all = [c for k in ("train", "primary_test", "secondary_test", "unseen") for c in sp[k]]
rng = np.random.default_rng(0)
for i, (b, lab) in enumerate(order):
    ids = [c for c in ids_all if cells[c]["batch"] == b]
    sw = np.array([cells[c]["temp_swing"] for c in ids])
    sus = np.array([cells[c]["temp_suspect"] for c in ids])
    x = i + rng.uniform(-0.18, 0.18, len(ids))
    ax.scatter(x[~sus], sw[~sus], s=10, color=C1, edgecolor="white", lw=0.4, zorder=3)
    ax.scatter(x[sus], sw[sus], s=16, color=BAD, marker="D", edgecolor="white", lw=0.4, zorder=4)
    for xx, yy, cid in zip(x[sus], sw[sus], np.array(ids)[sus]):
        ax.text(xx + 0.08, yy, cid, fontsize=6, color=INK, va="center")
ax.axhline(2.0, color=INK2, ls="--", lw=0.6)
ax.text(0.5, 2.15, "threshold 2 °C", fontsize=6.5, color=INK2, ha="center")
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in order])
ax.set_ylabel("Mean $T_{max}-T_{min}$,\ncycles 2-100 (°C)")
ax.grid(axis="x", visible=False)
save(fig, "fig_clean_temp")

# ------------------------------------------------ numbers for the text
ids_sev = [c for k in ("train", "primary_test", "secondary_test") for c in sp[k]]
for f in ("QD_clean", "IR", "chargetime"):
    tot = nan = 0
    for c in ids_all:
        s = cells[c]["summary"]; m = (s["cycle"] >= 2) & (s["cycle"] <= 100)
        tot += m.sum(); nan += np.isnan(s[f][m]).sum()
    stats[f"nan_{f}"] = [int(nan), int(tot)]
stats["temp_suspect"] = sp["temp_suspect"]
stats["anomalous"] = sp["anomalous"]
stats["life_range_severson"] = [min(cells[c]["cycle_life"] for c in ids_sev),
                                max(cells[c]["cycle_life"] for c in ids_sev)]
for k, _ in sets:
    L = [cells[c]["cycle_life"] for c in sp[k]]
    stats[f"life_{k}"] = [min(L), float(np.median(L)), max(L)]
json.dump(stats, open(OUT / "stats_cleaning.json", "w"), indent=1)
print(json.dumps(stats, indent=1))

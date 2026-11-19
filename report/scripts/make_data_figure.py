"""Dataset figure: capacity fade, example ΔQ(V) curves, and log Var(ΔQ) vs life for all four batches.
Run from repo root:  python3 report/scripts/make_data_figure.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import EOL_AH, VDLIN  # noqa: E402
from src.data import curve, load_dataset  # noqa: E402
from src.features import load_features  # noqa: E402

mpl.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 7.5})
INK2 = "#52514e"
GROUPS = {"b1": ("b1 (train/primary)", "#2a78d6"), "b2": ("b2 (train/primary)", "#e8a33d"),
          "b3": ("b3 (secondary)", "#1baf7a"), "u1": ("u1 standard (unseen)", "#d03b3b"),
          "u1_new": ("u1 newstructure (unseen)", "#6250d6")}

cells = load_dataset()["cells"]
df = load_features()
df["grp"] = np.where((df.batch == "u1") & df.policy.str.contains("newstructure"), "u1_new", df.batch)

fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.16, 2.15), gridspec_kw={"width_ratios": [1.15, 1, 1]})

# (a) capacity fade
for cid, r in df.iterrows():
    s = cells[cid]["summary"]
    a.plot(s["cycle"], s["QD_clean"], color=GROUPS[r.grp][1], lw=0.45, alpha=0.6)
a.axhline(EOL_AH, color=INK2, lw=0.6, ls="--")
a.text(2300, EOL_AH + 0.004, "EOL 0.88 Ah", fontsize=6.5, color=INK2, ha="right")
a.set(xlim=(0, 2400), ylim=(0.85, 1.11), xlabel="cycle", ylabel="discharge capacity (Ah)")
a.set_title("(a) capacity fade", loc="left", fontsize=7.5)

# (b) ΔQ(V) for training cells spanning the life range
tr = df[(df.split == "train") & ~df.anomalous].sort_values("cycle_life")
pick = tr.iloc[np.linspace(0, len(tr) - 1, 5).round().astype(int)]
cmap = mpl.colormaps["Blues"]
lo, hi = np.log10(pick.cycle_life.min()), np.log10(pick.cycle_life.max())
for cid, r in pick.iterrows():
    dq = (curve(cells[cid], "Qdlin", 100) - curve(cells[cid], "Qdlin", 10)) * 1000
    shade = 0.35 + 0.6 * (np.log10(r.cycle_life) - lo) / (hi - lo)
    b.plot(dq, VDLIN, color=cmap(shade), lw=1.1, label=f"{r.cycle_life:.0f}")
b.axvline(0, color=INK2, lw=0.5)
b.set(xlabel=r"$\Delta Q_{100-10}(V)$ (mAh)", ylabel="voltage (V)", ylim=(2.0, 3.5))
b.legend(title="cycle life", fontsize=6.5, title_fontsize=6.5, loc="lower left")
b.set_title(r"(b) $\Delta Q_{100-10}(V)$, training cells", loc="left", fontsize=7.5)

# (c) log Var vs life
for g, (lab, col) in GROUPS.items():
    d = df[df.grp == g]
    c.scatter(d.dq_log_var, d.cycle_life, s=9, color=col, edgecolor="white", lw=0.3, label=lab)
r_tr = np.corrcoef(tr.dq_log_var, np.log10(tr.cycle_life))[0, 1]
c.text(0.97, 0.95, f"train: r = {r_tr:.2f}", transform=c.transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
c.set_yscale("log")
c.yaxis.set_major_locator(mpl.ticker.FixedLocator([200, 500, 1000, 2000]))
c.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter()); c.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
c.set(xlabel=r"$\log_{10}\mathrm{Var}(\Delta Q_{100-10})$", ylabel="cycle life")
c.set_title("(c) the key feature", loc="left", fontsize=7.5)

h, l = c.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.07), markerscale=1.6)
fig.tight_layout(w_pad=0.8)
out = ROOT / "report" / "figures" / "fig_data.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out, f"train r = {r_tr:.3f}")

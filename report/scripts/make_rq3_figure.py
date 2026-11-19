"""RQ3 figure (column width, 2x2): why the two F4 features work. Training cells only.
(a) ΔQ amplitude grows at a constant rate; (b) life vs margin_rate; (c) early capacity rise (overhang);
(d) life relative to a ΔQ-only fit vs the rise after cycle 10.
Run from repo root:  python3 report/scripts/make_rq3_figure.py
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

from config import RESULTS_DIR  # noqa: E402
from src.data import curve, load_dataset  # noqa: E402
from src.features import SMOOTH_WIN, load_features  # noqa: E402

mpl.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 7})
INK2 = "#52514e"
BATCH = {"b1": ("b1", "#2a78d6", "o"), "b2": ("b2", "#e8a33d", "s")}
BLUES = LinearSegmentedColormap.from_list("blues", ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
cells = load_dataset()["cells"]
df = load_features()
tr = df[df.split == "train"]
S = json.load(open(RESULTS_DIR / "p3_screening.json"))
norm = LogNorm(tr.cycle_life.min(), tr.cycle_life.max())
y = np.log10(tr.cycle_life.to_numpy())

fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(3.5, 3.35))

# (a) amplitude of ΔQ_{n-10} vs n-10
NS = np.array(S["trajectory"]["cycles"])
for cid, r in tr.sort_values("cycle_life", ascending=False).iterrows():
    q10 = curve(cells[cid], "Qdlin", 10)
    amp = [1000 * np.std(curve(cells[cid], "Qdlin", n) - q10, ddof=1) for n in NS]
    a.plot(NS - 10, amp, color=BLUES(norm(r.cycle_life)), lw=0.6)
x = np.array([10, 90])
a.plot(x, 15.5 * x / 90, color=INK2, lw=0.9, ls="--")
a.text(12, 9, "slope 1", fontsize=6, color=INK2)
a.set(xscale="log", yscale="log", xlabel="cycles after cycle 10", ylabel=r"sd $\Delta Q_{n-10}$ (mAh)")
p = pd.read_csv(RESULTS_DIR / "p3_candidates_train.csv", index_col=0)["traj_exponent"]
a.set_title("(a) $\\Delta Q$ grows linearly", loc="left", fontsize=7)
a.text(0.97, 0.04, f"median exponent {p.median():.2f}", transform=a.transAxes, ha="right", fontsize=6, color=INK2)

# (b) margin_rate against log Var(ΔQ): almost the same number
for bt, (lab, col, mk) in BATCH.items():
    t = tr[tr.batch == bt]
    b.scatter(t.dq_log_var, t.margin_rate, color=col, marker=mk, s=10, lw=0, label=lab)
r = np.corrcoef(tr.margin_rate, tr.dq_log_var)[0, 1]
b.set(xlabel=r"$\log_{10}\mathrm{Var}(\Delta Q)$", ylabel="margin_rate")
b.legend(fontsize=6, loc="upper right", handletextpad=0.1, markerscale=1.3)
b.set_title("(b) margin_rate vs. variance", loc="left", fontsize=7)
b.text(0.04, 0.04, f"r = {r:+.3f}", transform=b.transAxes, ha="left", fontsize=6, color=INK2)

# (c) early capacity rise, three cells (9-cycle rolling median)
for cid, col in (("b1c11", "#2a78d6"), ("b2c29", "#e8a33d"), ("b1c44", "#6250d6")):
    s = cells[cid]["summary"]; cyc = s["cycle"]; q = s["QD_clean"]
    m = (cyc >= 2) & (cyc <= 100) & np.isfinite(q)
    sm = pd.Series(q[m]).rolling(SMOOTH_WIN, center=True, min_periods=4).median().to_numpy()
    c.plot(cyc[m], 1000 * (sm - np.nanmedian(q[m][cyc[m] <= 4])), color=col, lw=1.1,
           label=f"{cid}: {df.loc[cid, 'overhang_after10']:.1f} mAh")
c.axvline(10, color=INK2, lw=0.6, ls=":")
c.set(xlabel="cycle", ylabel=r"$Q_d - Q_d(2\text{--}4)$ (mAh)", ylim=(-10, 7))
c.legend(fontsize=5.5, loc="lower left", handlelength=1.2, title="rise after 10", title_fontsize=5.5)
c.set_title("(c) early capacity rise", loc="left", fontsize=7)

# (d) life relative to a ΔQ-only fit within each batch vs overhang_after10
for bt, (lab, col, mk) in BATCH.items():
    t = tr[tr.batch == bt]; yy = np.log10(t.cycle_life.to_numpy())
    s1, i1 = np.polyfit(t.dq_log_var, yy, 1)
    res = 100 * (10 ** (yy - (i1 + s1 * t.dq_log_var.to_numpy())) - 1)
    d.scatter(t.overhang_after10, res, color=col, marker=mk, s=10, lw=0, label=lab)
d.axhline(0, color=INK2, lw=0.6)
d.set(xlabel="rise after cycle 10 (mAh)", ylabel=r"life vs. $\Delta Q$ fit (%)")
d.set_title("(d) residual vs. late rise", loc="left", fontsize=7)

fig.tight_layout(w_pad=0.5, h_pad=0.6)
out = ROOT / "report" / "figures" / "fig_rq3_physics.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out, f"r(margin_rate, logVar) = {r:.3f}")

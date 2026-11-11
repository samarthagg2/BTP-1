"""P2 figures: parity plots (cf. paper Fig. 3), ΔQ(V), feature correlations, elastic-net CV, b1c18 fault.
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


# ---------------------------------------------------------------------------
# Additional P2 figures (same content as notebooks/02_reproduction.ipynb)
import pickle  # noqa: E402
import warnings  # noqa: E402

import h5py  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from config import BATCHES, PROC_DIR, RAW_DIR, VDLIN  # noqa: E402
from src.data import curve, load_dataset  # noqa: E402
from src.features import ALL_CANDIDATES, FEATURE_SETS, delta_q  # noqa: E402
from src.reproduce import fit_one  # noqa: E402

C1, C2, BAD = "#2a78d6", "#eb6834", "#d03b3b"
BLUES = LinearSegmentedColormap.from_list("blues", ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
mpl.rcParams["axes.axisbelow"] = True
FIG = ROOT / "report" / "figures"
cells = load_dataset()["cells"]
train = df[df.split == "train"]


def _lefttitle(ax, s):
    ax.set_title(s, loc="left", fontsize=8.5)


# --- fig_p2_dq: why ΔQ(V)
LONG, SHORT = "b1c1", "b2c0"
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
for cid, col in [(LONG, C1), (SHORT, C2)]:
    c = cells[cid]
    q10, q100 = curve(c, "Qdlin", 10), curve(c, "Qdlin", 100)
    axes[0].plot(q10, VDLIN, color=col, lw=1.4, label=f"{cid}, cycle 10")
    axes[0].plot(q100, VDLIN, color=col, lw=1.4, ls="--", label=f"{cid}, cycle 100")
    axes[1].plot(1000 * (q100 - q10), VDLIN, color=col, lw=1.8, label=f"{cid} (life {c['cycle_life']:.0f})")
    s = c["summary"]; m = (s["cycle"] >= 2) & (s["cycle"] <= 300)
    axes[2].plot(s["cycle"][m], s["QD_clean"][m], color=col, lw=1.4, label=cid)
axes[0].set(xlabel="discharge capacity (Ah)", ylabel="voltage (V)"); _lefttitle(axes[0], "(a) Discharge curves at 4C")
axes[0].legend(fontsize=6.5)
axes[1].axvline(0, color=INK2, lw=0.7)
axes[1].set(xlabel=r"$\Delta Q_{100-10}$ (mAh)"); _lefttitle(axes[1], r"(b) $\Delta Q_{100-10}(V)$")
axes[1].legend(fontsize=6.5)
axes[2].axvspan(2, 100, color="#f3f2ee")
axes[2].set(xlabel="cycle", ylabel="QD (Ah)"); _lefttitle(axes[2], "(c) Capacity: cycles 2-100 shaded")
axes[2].legend(fontsize=6.5)
norm = LogNorm(train.cycle_life.min(), train.cycle_life.max())
for cid, row in train.sort_values("cycle_life", ascending=False).iterrows():
    axes[3].plot(1000 * delta_q(cells[cid]), VDLIN, color=BLUES(norm(row.cycle_life)), lw=0.8)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=BLUES), ax=axes[3], label="cycle life")
axes[3].set(xlabel=r"$\Delta Q_{100-10}$ (mAh)"); _lefttitle(axes[3], "(d) All 41 training cells")
fig.tight_layout(); fig.savefig(FIG / "fig_p2_dq.png", bbox_inches="tight"); plt.close(fig)

# --- fig_p2_corr: feature-life correlation on training cells
y = np.log10(train.cycle_life)
r = train[ALL_CANDIDATES].apply(lambda col: np.corrcoef(col.fillna(col.median()), y)[0, 1]).sort_values()
used = set(FEATURE_SETS["discharge"]) | set(FEATURE_SETS["full"])
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.barh(r.index, r.values,
        color=[C1 if k in used else "#c9c8c3" for k in r.index], height=0.65)
ax.axvline(0, color=INK2, lw=0.7); ax.set_xlim(-1, 1); ax.tick_params(axis="y", labelsize=7)
ax.set_xlabel(r"Pearson $r$ with $\log_{10}$(cycle life), 41 training cells")
fig.tight_layout(); fig.savefig(FIG / "fig_p2_corr.png", bbox_inches="tight"); plt.close(fig)

# --- fig_p2_cv: elastic-net CV curves
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for ax, name in zip(axes, ("discharge", "full")):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _, info = fit_one(df, name)
    en = model[-1]; mse = en.mse_path_.mean(axis=-1)
    for i, l1 in enumerate(en.l1_ratio):
        ax.plot(en.alphas_[i], np.sqrt(mse[i]), lw=0.9, color=BLUES(0.25 + 0.75 * i / (len(en.l1_ratio) - 1)),
                label=rf"$\ell_1$ ratio {l1}" if i in (0, len(en.l1_ratio) - 1) else None)
    ax.scatter([en.alpha_], [np.sqrt(mse.min())], color=C2, s=30, zorder=5,
               label=rf"chosen $\alpha$={en.alpha_:.3f}, $\ell_1$={en.l1_ratio_}")
    ax.set_xscale("log"); ax.set_xlabel(r"$\alpha$"); ax.set_ylim(0.04, 0.17)
    _lefttitle(ax, f"{name}: {info['n_selected']}/{len(FEATURE_SETS[name])} features kept")
    ax.legend(fontsize=6.5)
axes[0].set_ylabel(r"CV RMSE of $\log_{10}$ life")
fig.tight_layout(); fig.savefig(FIG / "fig_p2_cv.png", bbox_inches="tight"); plt.close(fig)

# --- fig_p2_b1c18: the data fault
raw = pickle.load(open(PROC_DIR / "b1_raw.pkl", "rb"))["cells"]["b1c18"]["summary"]
cl = cells["b1c18"]["summary"]
ext = pickle.load(open(PROC_DIR / "extras.pkl", "rb"))["b1c18"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
m = (raw["cycle"] >= 2) & (raw["cycle"] <= 100)
axes[0].plot(raw["cycle"][m], 1000 * np.where(raw["IR"] > 0, raw["IR"], np.nan)[m], color=BAD, lw=1.1, label="raw")
mc = (cl["cycle"] >= 2) & (cl["cycle"] <= 100)
axes[0].plot(cl["cycle"][mc], 1000 * cl["IR"][mc], color=C1, lw=1.8, label="after rule B9")
axes[0].set(xlabel="cycle", ylabel=r"IR (m$\Omega$)"); _lefttitle(axes[0], "(a) IR logger stuck, cycles 2-53")
axes[0].legend(fontsize=6.5)
mm = (ext["cycle"] >= 2) & (ext["cycle"] <= 100)
axes[1].bar(ext["cycle"][mm], ext["t_gap"][mm] / 60, color=BAD, width=0.9)
axes[1].set(xlabel="cycle", ylabel="time-stamp jump (h)"); _lefttitle(axes[1], "(b) Jump inside each record (rule B10)")
fig.tight_layout(); fig.savefig(FIG / "fig_p2_b1c18.png", bbox_inches="tight"); plt.close(fig)

# --- numbers quoted in the text (printed so they can be checked)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mvar = fit_one(df, "variance")[0]; mfull = fit_one(df, "full")[0]
imp, sc, ols = mvar
b = ols.coef_[0] / sc.scale_[0]; a = ols.intercept_ - b * sc.mean_[0]
print(f"variance fit: log10(life) = {a:.3f} {b:+.3f} log10 Var")
print("train r:", r.round(2).to_dict())
cols = FEATURE_SETS["full"]; trf = train[~train.temp_suspect]
bad = df.loc[["b1c18"]].copy()
irr = np.where(raw["IR"] > 0, raw["IR"], np.nan)
bad["ir_min_2_100"] = np.nanmin(irr[m]); bad["ir_100_minus_2"] = irr[raw["cycle"] == 100][0] - irr[raw["cycle"] == 2][0]
with h5py.File(RAW_DIR / BATCHES["b1"], "r") as f:
    g = f[f["batch"]["cycles"][cells["b1c18"]["index_in_batch"], 0]]
    tot = 0.0
    for n in range(2, 101):
        t = np.asarray(f[g["t"][n - 1, 0]]).ravel(); T = np.asarray(f[g["T"][n - 1, 0]]).ravel()
        ok = np.isfinite(t) & np.isfinite(T); tot += np.trapz(T[ok], t[ok])
bad["temp_integral_2_100"] = tot
zf = lambda v, k: (v - trf[k].mean()) / trf[k].std()
print(f"b1c18 unfixed: z(Tint) {zf(tot, 'temp_integral_2_100'):.0f}, z(IRdiff) {zf(bad.ir_100_minus_2.iloc[0], 'ir_100_minus_2'):.1f}, "
      f"pred {10 ** mfull.predict(bad[cols].values)[0]:.1f}; fixed pred {10 ** mfull.predict(df.loc[['b1c18'], cols].values)[0]:.0f}, "
      f"observed {df.loc['b1c18', 'cycle_life']:.0f}")
print("wrote fig_p2_dq / fig_p2_corr / fig_p2_cv / fig_p2_b1c18")

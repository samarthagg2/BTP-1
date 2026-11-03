"""EDA figures for the 2018-02-20 (unseen, `u1`) batch, from the lightweight cache
data/processed/u1_raw.pkl (raw traces / late cycles are read from the .mat).

Run from the BTP-1 root:
    python3 report/scripts/make_figures.py

Outputs:
    report/figures/*.pdf   vector figures used by main.tex
    report/figures/stats.json   numbers quoted in the report text
"""
import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[2]
PKL = ROOT / "data" / "processed" / "u1_raw.pkl"
import sys
sys.path.insert(0, str(ROOT))
from src.convert import read_raw_cycle  # noqa: E402
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_STD, C_NEW, C_3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
EOL = 0.88  # Ah, 80 % of 1.1 Ah nominal
COL_W, FULL_W = 3.5, 7.16  # IEEE column / page width in inches

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.0, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def lifebar(fig, ax, sm):
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("Cycle life (cycles)")
    cb.outline.set_visible(False)
    cb.set_ticks([400, 600, 800, 1000])
    cb.set_ticklabels(["400", "600", "800", "1000"])
    cb.ax.minorticks_off()
    return cb


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- data
print("loading pickle (~1-2 min)...")
with open(PKL, "rb") as fp:
    bat = pickle.load(fp)["cells"]
for v in bat.values():
    v["charge_policy"] = v["policy"]

VDLIN = np.linspace(3.5, 2.0, 1000)  # grid used for Qdlin/Tdlin/dQdV


def life(c):
    return float(np.ravel(c["cycle_life"])[0])


valid = {k: v for k, v in bat.items() if not np.isnan(life(v))}
is_new = {k: "newstructure" in v["charge_policy"] for k, v in valid.items()}
lives = {k: life(v) for k, v in valid.items()}
norm = LogNorm(vmin=min(lives.values()), vmax=max(lives.values()))

stats = {
    "n_cells_total": len(bat),
    "n_cells_valid": len(valid),
    "excluded": {k: v["charge_policy"] for k, v in bat.items() if k not in valid},
    "life_min": min(lives.values()), "life_max": max(lives.values()),
    "life_mean": float(np.mean(list(lives.values()))),
    "life_median": float(np.median(list(lives.values()))),
    "n_newstructure": int(sum(is_new.values())),
}
std_l = [lives[k] for k in valid if not is_new[k]]
new_l = [lives[k] for k in valid if is_new[k]]
stats["life_std_range"] = [min(std_l), max(std_l)]
stats["life_new_range"] = [min(new_l), max(new_l)]
stats["life_std_mean"] = float(np.mean(std_l))
stats["life_new_mean"] = float(np.mean(new_l))

# ---------------------------------------------------------------- Fig 1: one cycle
c = read_raw_cycle("u1", 0, 1)  # record 1 == cycle number 2
fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 2.9), sharex=True)
a1.plot(c["t"], c["I"], color=C_STD)
a1.axhline(0, color=INK2, lw=0.5)
a1.set_ylabel("Current (C-rate)")
a2.plot(c["t"], c["V"], color=C_STD)
a2.set_ylabel("Voltage (V)")
a2.set_xlabel("Time within cycle (min)")
for ax in (a1, a2):
    for x0, x1 in [(0, 10.2), (15, 41), (43, 66)]:
        ax.axvspan(x0, x1, color="#f0efec", lw=0, zorder=0)
for x, lab in [(5, "fast\ncharge"), (28, "1C CC-CV\nto 3.6 V"), (54.5, "4C discharge,\nCV at 2.0 V")]:
    a1.text(x, 6.3, lab, ha="center", va="bottom", fontsize=6.5, color=INK2)
a1.set_ylim(-5, 8.5)
a1.text(3.3, 3.6, "5.2C", ha="center", fontsize=6.5, color=INK)
a1.text(8.4, 2.5, "4C", ha="center", fontsize=6.5, color=INK)
a1.text(20, 1.4, "1C", ha="center", fontsize=6.5, color=INK)
a1.text(50, -3.55, "4C", ha="center", fontsize=6.5, color=INK)
save(fig, "fig_cycle_protocol")
stats["b1c0_cycle2_duration_min"] = float(c["t"].max())

# ---------------------------------------------------------------- Fig 2: policies
policies = sorted({v["charge_policy"].replace("-newstructure", "") for v in valid.values()})
stats["policies"] = policies


def parse(p):  # '5.2C(58%)-4C' -> (5.2, 0.58, 4.0)
    a, b = p.split("-")
    c1, soc = a.split("C(")
    return float(c1), float(soc.rstrip("%)")) / 100, float(b.rstrip("C"))


rows = []
for p in policies:
    c1, s, c2 = parse(p)
    t1 = s / c1 * 60
    t2 = (0.8 - s) / c2 * 60
    cells = [k for k, v in valid.items() if v["charge_policy"].replace("-newstructure", "") == p]
    std = [lives[k] for k in cells if not is_new[k]]
    new = [lives[k] for k in cells if is_new[k]]
    rows.append(dict(policy=p, c1=c1, soc=s, c2=c2, t1=round(t1, 2), t2=round(t2, 2),
                     t80=round(t1 + t2, 2), n_std=len(std), n_new=len(new),
                     life_std=round(float(np.mean(std)), 0) if std else None,
                     life_new=round(float(np.mean(new)), 0) if new else None))
stats["policy_table"] = rows

n = len(policies)
ncol = 3
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(COL_W, 0.85 * nrow + 0.3), sharex=True, sharey=True)
for ax, r in zip(axes.flat, rows):
    x = [0, r["soc"] * 100, r["soc"] * 100, 80]
    y = [r["c1"], r["c1"], r["c2"], r["c2"]]
    ax.fill_between(x, 0, y, step=None, color="#b7d3f6", lw=0)
    ax.plot(x, y, color=C_STD, lw=1.2)
    ax.set_title(r["policy"], fontsize=6.5, pad=2)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 6.6)
    ax.set_xticks([0, 40, 80])
    ax.set_yticks([0, 3, 6])
for ax in list(axes.flat)[n:]:
    ax.axis("off")
fig.supxlabel("State of charge (%)", fontsize=8, y=0.02)
fig.supylabel("Current (C-rate)", fontsize=8, x=0.02)
fig.tight_layout(h_pad=0.4, w_pad=0.3)
save(fig, "fig_policies")

# ---------------------------------------------------------------- Fig 3: capacity fade
fig, ax = plt.subplots(figsize=(COL_W, 2.4))
for k in sorted(valid, key=lambda k: -lives[k]):
    s = valid[k]["summary"]
    qd, cyc = np.asarray(s["QD"]), np.asarray(s["cycle"])
    m = (qd > 0.7) & (qd < 1.15)
    ax.plot(cyc[m], qd[m], color=BLUES(norm(lives[k])), lw=0.7)
ax.axhline(EOL, color=INK2, ls="--", lw=0.7)
ax.text(15, EOL + 0.004, "EOL: 0.88 Ah", ha="left", va="bottom", fontsize=6.5, color=INK2)
ax.set_xlabel("Cycle number")
ax.set_ylabel("Discharge capacity (Ah)")
ax.set_ylim(0.8, 1.12)
ax.set_xlim(0, 1260)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=BLUES)
lifebar(fig, ax, sm)
save(fig, "fig_capacity_fade")

# ---------------------------------------------------------------- Fig 4: life by policy
order = sorted(policies, key=lambda p: np.mean([lives[k] for k in valid
                                                  if valid[k]["charge_policy"].replace("-newstructure", "") == p]))
fig, ax = plt.subplots(figsize=(COL_W, 2.5))
rng = np.random.default_rng(0)
for i, p in enumerate(order):
    for k, v in valid.items():
        if v["charge_policy"].replace("-newstructure", "") != p:
            continue
        col = C_NEW if is_new[k] else C_STD
        mk = "s" if is_new[k] else "o"
        ax.scatter(i + rng.uniform(-0.15, 0.15), lives[k], s=14, color=col, marker=mk,
                   edgecolor="white", linewidth=0.5, zorder=3)
ax.scatter([], [], s=14, color=C_STD, marker="o", label="standard")
ax.scatter([], [], s=14, color=C_NEW, marker="s", label="-newstructure")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=55, ha="right", fontsize=6)
ax.set_ylabel("Cycle life (cycles)")
ax.legend(loc="upper left")
ax.grid(axis="x", visible=False)
save(fig, "fig_life_by_policy")

# ---------------------------------------------------------------- Fig 5: summary signals
picks = sorted([k for k in valid if not is_new[k]], key=lambda k: lives[k])
picks = [picks[0], picks[len(picks) // 2], max(valid, key=lambda k: lives[k])]
cols = [C_STD, C_NEW, C_3]
fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.9))
for k, col in zip(picks, cols):
    s = valid[k]["summary"]
    cyc = np.asarray(s["cycle"])
    ir, tm, ct = (np.asarray(s[f]) for f in ("IR", "Tmax", "chargetime"))
    lab = f"{k} ({int(lives[k])} cyc.)"
    m = (ir > 0.01) & (ir < 0.03)
    axes[0].plot(cyc[m], ir[m] * 1e3, color=col, lw=0.8, label=lab)
    m = (tm > 25) & (tm < 50)
    axes[1].plot(cyc[m], tm[m], color=col, lw=0.8)
    m = (ct > 5) & (ct < 25)
    axes[2].plot(cyc[m], ct[m], color=col, lw=0.8)
axes[0].set_ylabel(r"Internal resistance (m$\Omega$)")
axes[1].set_ylabel(r"Max. temperature ($^\circ$C)")
axes[2].set_ylabel("Charge time to 80% (min)")
for ax in axes:
    ax.set_xlabel("Cycle number")
axes[0].legend(loc="upper left", fontsize=6)
fig.tight_layout(w_pad=1.2)
save(fig, "fig_summary_signals")

# ---------------------------------------------------------------- Fig 6: Q(V) and dQ/dV
k = "u1c0"
sel = [9, 99, 199, 299, 399, 475]  # records -> cycle numbers 10, 100, ...
cyc_raw = {j: read_raw_cycle("u1", bat[k]["index_in_batch"], j) for j in sel}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.1))
seq = BLUES(np.linspace(0.05, 1, len(sel)))
for j, col in zip(sel, seq):
    q = cyc_raw[j]["Qdlin"]
    d = cyc_raw[j]["dQdV"]
    a1.plot(q, VDLIN, color=col, lw=0.9, label=f"cycle {j + 1}")
    a2.plot(VDLIN, d, color=col, lw=0.9)
a1.set_xlabel("Discharge capacity Q (Ah)")
a1.set_ylabel("Voltage (V)")
a1.legend(loc="lower left", fontsize=6)
a1.set_title(f"(a) Discharge curve Q(V), {k}", loc="left")
a2.set_xlabel("Voltage (V)")
a2.set_ylabel(r"dQ/dV (Ah V$^{-1}$)")
a2.set_xlim(2.0, 3.5)
a2.set_title("(b) Differential capacity dQ/dV", loc="left")
fig.tight_layout(w_pad=1.5)
save(fig, "fig_qv_dqdv")

# ---------------------------------------------------------------- Fig 7/8: Delta Q(V)
feat = {}
fig, ax = plt.subplots(figsize=(COL_W, 2.4))
for k in sorted(valid, key=lambda k: -lives[k]):
    Q = valid[k]["curves"]["Qdlin"].astype(float)
    dq = Q[99] - Q[9]   # records 99/9 == cycle numbers 100/10
    feat[k] = dq
    ax.plot(dq, VDLIN, color=BLUES(norm(lives[k])), lw=0.6)
ax.axvline(0, color=INK2, lw=0.5)
ax.set_xlabel(r"$\Delta Q_{100-10}(V)$ (Ah)")
ax.set_ylabel("Voltage (V)")
lifebar(fig, ax, sm)
save(fig, "fig_delta_q")

logvar = {k: np.log10(np.var(dq)) for k, dq in feat.items()}
loglife = {k: np.log10(lives[k]) for k in valid}
fig, ax = plt.subplots(figsize=(COL_W, 2.3))
for grp, col, mk, lab in [(False, C_STD, "o", "standard"), (True, C_NEW, "s", "-newstructure")]:
    ks = [k for k in valid if is_new[k] == grp]
    ax.scatter([logvar[k] for k in ks], [lives[k] for k in ks], s=16, color=col, marker=mk,
               edgecolor="white", linewidth=0.5, label=lab, zorder=3)
ax.set_yscale("log")
ax.set_yticks([400, 600, 800, 1000])
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_xlabel(r"$\log_{10}\,\mathrm{Var}(\Delta Q_{100-10}(V))$")
ax.set_ylabel("Cycle life (log scale)")
ax.legend(loc="upper right")
save(fig, "fig_var_vs_life")


def pearson(ks):
    x = np.array([logvar[k] for k in ks])
    y = np.array([loglife[k] for k in ks])
    return float(np.corrcoef(x, y)[0, 1])


stats["pearson_logvar_loglife_all"] = pearson(list(valid))
stats["pearson_logvar_loglife_std"] = pearson([k for k in valid if not is_new[k]])
stats["pearson_logvar_loglife_new"] = pearson([k for k in valid if is_new[k]])

# ---------------------------------------------------------------- Fig 9: life histogram
fig, ax = plt.subplots(figsize=(COL_W, 1.8))
bins = np.arange(350, 1250, 50)
ax.hist([std_l, new_l], bins=bins, stacked=True, color=[C_STD, C_NEW],
        edgecolor="white", linewidth=0.6, label=["standard", "-newstructure"])
ax.set_xlabel("Cycle life (cycles)")
ax.set_ylabel("Number of cells")
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
ax.legend(loc="upper right")
ax.grid(axis="x", visible=False)
save(fig, "fig_life_hist")

with open(OUT / "stats.json", "w") as fp:
    json.dump(stats, fp, indent=2)
print(json.dumps({k: v for k, v in stats.items() if k != "policy_table"}, indent=1))
for r in rows:
    print(r)

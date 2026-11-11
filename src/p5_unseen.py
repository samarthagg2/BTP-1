"""P5 (RQ5): frozen models applied to the unseen 2018-02-20 batch (u1).

No model is refitted on u1 or re-tuned: every model is fitted on the 41 training
cells with exactly the settings of Phases 2-4 (deterministic seeds) and applied
once to the 39 labelled u1 cells (30 standard, 9 -newstructure).

PRE-REGISTERED HYPOTHESES (fixed 2026-11-11, before any u1 life was used; decision
rules are evaluated mechanically in `verdicts`):
  H1  Models separate standard from newstructure cells but rank cells poorly
      within each group.  Supported if, for at least 3 of the 4 main models,
      Spearman rho(pred, life) >= 0.7 over all u1 cells and < 0.5 within each group.
  H2  Overhang: u1 standard cells have a late, large overhang rise (z = +2.6), so
      their ΔQ understates degradation. Supported if EN-F2's mean signed % error on
      standard cells is > 0 (over-prediction), its log residual correlates
      positively with overhang_after10 within the standard cells (Spearman > 0),
      and EN-F4 and GP-F4 have a smaller |mean signed % error| than EN-F2 there.
  H3  Procedure: u1 newstructure cells (5-s rests, like b3) are mispredicted in the
      same direction as the secondary set. Supported if the sign of the mean signed
      % error on u1 newstructure equals that on the secondary set for >= 3 of the 4
      main models.
  H4  The full model's temperature integral (z = +2.9 on standard cells) lowers
      their predictions. Supported if its mean contribution to log10 life on the
      standard cells is below -0.05 (more than ~11% shorter).
  H5  GP-F4 is overconfident on u1. Supported if its 95% interval covers < 90% of
      the u1 cells.
MAIN MODELS: EN-F2 (discharge), EN-F3 (full), EN-F4 (physics), GP-F4.

Usage: python3 -m src.p5_unseen      -> results/p5_unseen.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from config import RESULTS_DIR
from src.evaluate import constant_baseline, policy_baseline, scores
from src.features import FEATURE_SETS, load_features
from src.models import NEW_MODELS, gp_predict_std, make_regressor
from src.pls import design_matrix, make_pls
from src.reproduce import fit_one, uses_temperature

MAIN = {"EN-F2": "en_discharge", "EN-F3": "en_full", "EN-F4": "en_physics", "GP-F4": "gp_physics"}
EN_SETS = ("variance", "discharge", "full", "physics", "full_physics")
N_BOOT = 5000


def fit_all(df):
    """Fit every Phase 2-4 model on the training cells (same settings as before)."""
    tr = df[df.split == "train"]
    y = np.log10(tr.cycle_life.to_numpy())
    models = {}
    for fs in EN_SETS:
        models[f"en_{fs}"] = (fit_one(df, fs)[0], FEATURE_SETS[fs])
    for fs in ("discharge", "physics"):
        cols = FEATURE_SETS[fs]
        for m in NEW_MODELS:
            models[f"{m}_{fs}"] = (make_regressor(m, len(cols)).fit(tr[cols].to_numpy(), y), cols)
    for v in ("asinh", "f4_shape"):
        models[f"pls_{v}"] = (make_pls(v).fit(design_matrix(tr, v), y), v)
    return models


def predict(models, d):
    out = {}
    for name, (m, cols) in models.items():
        X = design_matrix(d, cols) if name.startswith("pls_") else d[cols].to_numpy()
        out[name] = pd.Series(10 ** np.ravel(m.predict(X)), index=d.index)
    return out


def group_scores(y, p, groups):
    res = {}
    for g in ("all", "standard", "newstructure"):
        m = np.ones(len(y), bool) if g == "all" else (groups == g)
        yy, pp = y[m], p[m]
        s = scores(yy, pp)
        s["signed_pct"] = float(np.mean((pp - yy) / yy) * 100)
        s["median_abs_pct"] = float(np.median(np.abs(pp - yy) / yy) * 100)
        s["spearman"] = float(spearmanr(pp, yy)[0])
        res[g] = s
    return res


def boot_pairs(y, preds, groups, pairs, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for g in ("all", "standard", "newstructure"):
        m = np.ones(len(y), bool) if g == "all" else (groups == g)
        yy = y[m]
        idx = rng.integers(0, m.sum(), size=(N_BOOT, m.sum()))
        out[g] = {}
        for a, b in pairs:
            ea, eb = preds[a][m] - yy, preds[b][m] - yy
            d = np.sqrt((ea[idx] ** 2).mean(1)) - np.sqrt((eb[idx] ** 2).mean(1))
            out[g][f"{a}-{b}"] = {"diff": float(d.mean()), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out


def en_contributions(model, d, cols):
    """Contribution of each feature to log10 life (coef x standardised value) for an elastic-net pipeline."""
    imp, sc, en = model[0], model[1], model[-1]
    Z = sc.transform(imp.transform(d[cols].to_numpy()))
    return pd.DataFrame(Z * en.coef_, index=d.index, columns=cols)


def run(verbose=True):
    df = load_features()
    u = df[df.split == "unseen"].copy()
    u["group"] = np.where(u.policy.str.contains("newstructure"), "newstructure", "standard")
    tr = df[df.split == "train"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models = fit_all(df)
        preds_u = predict(models, u)
        sec = df[df.split == "secondary_test"]
        pri = df[df.split == "primary_test"]
        preds_s, preds_p = predict(models, sec), predict(models, pri)
        # baselines
        preds_u["constant"] = pd.Series(constant_baseline(tr.cycle_life, 1)[0], index=u.index)
        preds_u["policy_mean"] = pd.Series(policy_baseline(tr.policy, tr.cycle_life, u.policy), index=u.index)
        y = u.cycle_life.to_numpy()
        g = u.group.to_numpy()
        res = {k: group_scores(y, p.to_numpy(), g) for k, p in preds_u.items()}
        # GP intervals on u1
        cov = {}
        for fs in ("discharge", "physics"):
            mu, sd = gp_predict_std(models[f"gp_{fs}"][0], u[FEATURE_SETS[fs]].to_numpy())
            inside = np.abs(np.log10(y) - mu) <= 1.96 * sd
            cov[f"gp_{fs}"] = {grp: float(inside[(g == grp) if grp != "all" else np.ones(len(g), bool)].mean())
                               for grp in ("all", "standard", "newstructure")}
            cov[f"gp_{fs}"]["median_interval_factor"] = float(np.median(10 ** (1.96 * sd)))
        # bootstrap of the main comparisons
        pu = {k: v.to_numpy() for k, v in preds_u.items()}
        boot = boot_pairs(y, pu, g, [("en_physics", "en_discharge"), ("gp_physics", "en_physics"),
                                    ("en_full", "en_discharge"), ("en_discharge", "policy_mean")])
        # contributions of the elastic-net features (log10 life), mean per group / set
        contrib = {}
        for fs in ("discharge", "full", "physics"):
            m, cols = models[f"en_{fs}"]
            parts = {"u1_standard": u[u.group == "standard"], "u1_newstructure": u[u.group == "newstructure"],
                     "secondary": sec, "primary": pri, "train": tr}
            contrib[fs] = {k: en_contributions(m, d, cols).mean().round(4).to_dict() for k, d in parts.items()}
        # permutation importance (RMSE of log10 life) on each test set
        perm = {}
        for name in ("en_discharge", "en_physics", "gp_physics"):
            m, cols = models[name]
            perm[name] = {}
            for k, d in (("primary", pri), ("secondary", sec), ("u1", u)):
                r = permutation_importance(m, d[cols].to_numpy(), np.log10(d.cycle_life.to_numpy()), n_repeats=50,
                                           random_state=0, scoring="neg_root_mean_squared_error")
                perm[name][k] = dict(zip(cols, np.round(r.importances_mean, 5).tolist()))
        # inputs for the hypotheses
        resid_f2 = np.log10(pu["en_discharge"]) - np.log10(y)
        std = g == "standard"
        sec_signed = {k: float(np.mean((preds_s[v].to_numpy() - sec.cycle_life.to_numpy()) / sec.cycle_life.to_numpy()) * 100)
                      for k, v in MAIN.items()}
        h = {
            "H1": {k: {"rho_all": res[v]["all"]["spearman"], "rho_standard": res[v]["standard"]["spearman"],
                       "rho_newstructure": res[v]["newstructure"]["spearman"]} for k, v in MAIN.items()},
            "H2": {"enf2_signed_pct_standard": res["en_discharge"]["standard"]["signed_pct"],
                   "rho_resid_enf2_vs_overhang_standard": float(spearmanr(resid_f2[std], u.overhang_after10.to_numpy()[std])[0]),
                   "abs_signed_pct_standard": {k: abs(res[v]["standard"]["signed_pct"]) for k, v in MAIN.items()}},
            "H3": {k: {"u1_newstructure": res[v]["newstructure"]["signed_pct"], "secondary": sec_signed[k]} for k, v in MAIN.items()},
            "H4": {"temp_integral_contribution_standard": contrib["full"]["u1_standard"]["temp_integral_2_100"]},
            "H5": {"gp_f4_coverage_u1": cov["gp_physics"]["all"]},
        }
    out = {"n": {"all": int(len(u)), "standard": int(std.sum()), "newstructure": int((~std).sum())},
           "scores": res, "gp_coverage": cov, "bootstrap": boot, "contributions": contrib,
           "permutation_importance": perm, "hypotheses": h, "verdicts": verdicts(h),
           "predictions": {k: v.round(1).to_dict() for k, v in preds_u.items()},
           "secondary_signed_pct": sec_signed}
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "p5_unseen.json", "w") as fp:
        json.dump(out, fp, indent=2)
    if verbose:
        report(out)
    return out


def verdicts(h):
    v = {}
    h1 = [x["rho_all"] >= 0.7 and x["rho_standard"] < 0.5 and x["rho_newstructure"] < 0.5 for x in h["H1"].values()]
    v["H1"] = bool(sum(h1) >= 3)
    a = h["H2"]["abs_signed_pct_standard"]
    v["H2"] = bool(h["H2"]["enf2_signed_pct_standard"] > 0 and h["H2"]["rho_resid_enf2_vs_overhang_standard"] > 0
                   and a["EN-F4"] < a["EN-F2"] and a["GP-F4"] < a["EN-F2"])
    same = [np.sign(x["u1_newstructure"]) == np.sign(x["secondary"]) for x in h["H3"].values()]
    v["H3"] = bool(sum(same) >= 3)
    v["H4"] = bool(h["H4"]["temp_integral_contribution_standard"] < -0.05)
    v["H5"] = bool(h["H5"]["gp_f4_coverage_u1"] < 0.90)
    return v


def report(out):
    print(f"u1: {out['n']}")
    print(f"{'model':16s} " + "  ".join(f"{g:>34s}" for g in ("all", "standard", "newstructure")))
    for k, s in out["scores"].items():
        print(f"{k:16s} " + "  ".join(f"{s[g]['rmse']:5.0f} / {s[g]['mape']:5.1f}% / bias {s[g]['signed_pct']:+6.1f}% / ρ {s[g]['spearman']:+.2f}"
                                      for g in ("all", "standard", "newstructure")))
    print("GP coverage:", {k: {g: round(x, 2) for g, x in v.items()} for k, v in out["gp_coverage"].items()})
    for g, b in out["bootstrap"].items():
        print(f"bootstrap {g}: " + "  ".join(f"{p} {w['diff']:+.0f} [{w['ci'][0]:+.0f},{w['ci'][1]:+.0f}]" for p, w in b.items()))
    print("hypotheses:", json.dumps(out["hypotheses"], indent=1, default=float)[:1800])
    print("VERDICTS:", out["verdicts"])


if __name__ == "__main__":
    run()

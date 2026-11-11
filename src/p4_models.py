"""P4: Gaussian process, SVR and additive spline model vs the paper's elastic net.

Pre-registered before any test evaluation: models gp / svr / gam (src/models.py)
x feature sets F2 'discharge' and F4 'physics' = 6 configurations, all evaluated
once on the paper's splits and all reported. References: elastic net on the same
sets. A screened extra feature (voltage sag at fixed charge, ΔV(Q)) was rejected
on the training cells (rho = 0.99 with dq_log_var, no gain) and is not used.

Usage: python3 -m src.p4_models     -> results/p4_models.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from config import RESULTS_DIR
from src.evaluate import scores
from src.features import FEATURE_SETS, load_features
from src.models import NEW_MODELS, gp_predict_std, make_regressor
from src.p3_evaluate import bootstrap, predictions, robust
from src.reproduce import fit_one, make_model

SETS = ("discharge", "physics")


def nested_cv(df, n_repeats=10, seed=7):
    tr = df[df.split == "train"]
    y = np.log10(tr.cycle_life.to_numpy())
    outer = list(RepeatedKFold(n_splits=4, n_repeats=n_repeats, random_state=seed).split(y))
    out = {}
    for fs in SETS:
        X = tr[FEATURE_SETS[fs]].to_numpy()
        makers = {f"en_{fs}": lambda: make_model(fs)}
        makers.update({f"{m}_{fs}": (lambda m=m: make_regressor(m, X.shape[1])) for m in NEW_MODELS})
        for name, make in makers.items():
            e = []
            for i, (a, b) in enumerate(outer):
                mdl = make().fit(X[a], y[a])
                e.append(float(np.sqrt(np.mean((np.ravel(mdl.predict(X[b])) - y[b]) ** 2))))
            out[name] = {"rmse_log": float(np.mean(e)), "folds": e}
    ref = {fs: np.array(out[f"en_{fs}"]["folds"]) for fs in SETS}
    for name, v in out.items():
        fs = name.split("_", 1)[1]
        v["better_than_en_folds_pct"] = float(100 * np.mean(np.array(v["folds"]) < ref[fs]))
    return out


def test_eval(df, n_seeds=5):
    tr = df[df.split == "train"]
    y = np.log10(tr.cycle_life.to_numpy())
    paper = df[df.split != "unseen"]
    lo, hi = float(tr.cycle_life.min()), float(tr.cycle_life.max())
    parts = {"train": paper.split == "train", "primary": paper.split == "primary_test",
             "primary_excl": (paper.split == "primary_test") & ~paper.anomalous,
             "secondary": paper.split == "secondary_test"}
    preds, res = {}, {}
    for fs in SETS:
        cols = FEATURE_SETS[fs]
        X, Xp = tr[cols].to_numpy(), paper[cols].to_numpy()
        preds[f"en_{fs}"] = predictions(df, fit_one(df, fs)[0], fs)
        for m in NEW_MODELS:
            name = f"{m}_{fs}"
            mdl = make_regressor(m, len(cols)).fit(X, y)
            preds[name] = pd.Series(10 ** np.ravel(mdl.predict(Xp)), index=paper.index)
            extra = {}
            if m == "gp":
                mu, sd = gp_predict_std(mdl, Xp)
                obs = np.log10(paper.cycle_life.to_numpy())
                inside = np.abs(obs - mu) <= 1.96 * sd
                extra["kernel"] = str(mdl[-1].kernel_)
                extra["coverage_95"] = {s: float(inside[msk.to_numpy()].mean()) for s, msk in parts.items()}
                extra["median_interval_factor"] = {s: float(np.median(10 ** (1.96 * sd[msk.to_numpy()]))) for s, msk in parts.items()}
                extra["sd_log"] = dict(zip(paper.index, sd.round(5).tolist()))
            else:
                extra["params"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                   for k, v in mdl.best_params_.items()}
                spread = []
                for s in range(n_seeds):
                    p = 10 ** np.ravel(make_regressor(m, len(cols), seed=s).fit(X, y).predict(Xp))
                    spread.append({k: scores(paper.cycle_life[msk], p[msk.to_numpy()])["rmse"] for k, msk in parts.items()
                                   if k in ("primary", "secondary")})
                extra["seed_range_rmse"] = {k: [min(r[k] for r in spread), max(r[k] for r in spread)] for k in ("primary", "secondary")}
            res[name] = extra
    for name, p in preds.items():
        res.setdefault(name, {})
        res[name]["scores"] = {s: scores(paper.cycle_life[m], p[paper.index[m]]) for s, m in parts.items()}
        res[name]["robust"] = robust(df, p, lo, hi)
    pairs = [(f"{m}_{fs}", f"en_{fs}") for fs in SETS for m in NEW_MODELS]
    res["bootstrap"] = bootstrap(df, preds, lo, hi, pairs)
    res["predictions"] = {k: v.round(1).to_dict() for k, v in preds.items()}
    return res


def run(verbose=True):
    df = load_features()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {"preregistered": {"models": list(NEW_MODELS), "feature_sets": list(SETS)},
               "nested_cv_train": nested_cv(df), "test": test_eval(df)}
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "p4_models.json", "w") as fp:
        json.dump(out, fp, indent=2, default=str)
    if verbose:
        report(out)
    return out


def report(out):
    print("nested CV on training cells (RMSE log10 life; % folds better than EN on same set):")
    for k, v in out["nested_cv_train"].items():
        print(f"   {k:14s} {v['rmse_log']:.4f}  ({v['better_than_en_folds_pct']:.0f}%)")
    print("\ntest (RMSE cycles / % error):")
    t = out["test"]
    for k, v in t.items():
        if k in ("bootstrap", "predictions"):
            continue
        s, r = v["scores"], v["robust"]
        line = f"   {k:14s} " + "  ".join(f"{q} {s[q]['rmse']:.0f}/{s[q]['mape']:.1f}%" for q in ("train", "primary", "primary_excl", "secondary"))
        line += f" | in-range {r['primary']['rmse_in_range']:.0f} | med% {r['primary']['median_pct_err']:.1f}/{r['secondary']['median_pct_err']:.1f}"
        if "coverage_95" in v:
            line += f" | 95% cover prim {v['coverage_95']['primary']:.2f} sec {v['coverage_95']['secondary']:.2f}"
        if "seed_range_rmse" in v:
            line += f" | seeds {[[round(a) for a in x] for x in v['seed_range_rmse'].values()]}"
        print(line)
    for key, b in t["bootstrap"].items():
        print(f"   bootstrap {key}: " + "  ".join(f"{p} {w['rmse_diff']:+.0f} [{w['rmse_ci'][0]:+.0f},{w['rmse_ci'][1]:+.0f}]"
                                                for p, w in b.items() if "-" in p))


if __name__ == "__main__":
    run()

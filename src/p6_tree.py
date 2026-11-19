"""P6 add-on: one tree model (random forest) on F2, against the paper's elastic net.

Fixed before any test evaluation: random forest, 300 trees, on F2 'discharge' (same set as the
other non-linear models); only min_samples_leaf and max_features are tuned, by 4-fold CV repeated 3 times
on the 41 training cells. Nested CV uses the outer folds of src/p4_models.py (seed 7) so it is
directly comparable with the elastic net there. The model is then scored once on the primary,
secondary and unseen (u1) sets. Note: these test sets had already been used for other models.

Usage: python3 -m src.p6_tree     -> results/p6_tree.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.pipeline import make_pipeline

from config import RESULTS_DIR
from src.evaluate import scores
from src.features import FEATURE_SETS, load_features
from src.p3_evaluate import bootstrap, predictions, robust
from src.reproduce import fit_one

FS = "discharge"
GRID = {"randomforestregressor__min_samples_leaf": [2, 3, 5, 8],
        "randomforestregressor__max_features": [0.33, 0.67, 1.0]}


def make_rf(seed=42):
    pipe = make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1))
    return GridSearchCV(pipe, GRID, cv=RepeatedKFold(n_splits=4, n_repeats=3, random_state=seed),
                        scoring="neg_root_mean_squared_error", n_jobs=-1)


def nested_cv(tr, n_repeats=10, seed=7):
    y = np.log10(tr.cycle_life.to_numpy())
    X = tr[FEATURE_SETS[FS]].to_numpy()
    e = []
    for a, b in RepeatedKFold(n_splits=4, n_repeats=n_repeats, random_state=seed).split(y):
        m = make_rf().fit(X[a], y[a])
        e.append(float(np.sqrt(np.mean((m.predict(X[b]) - y[b]) ** 2))))
    en = json.load(open(RESULTS_DIR / "p4_models.json"))["nested_cv_train"][f"en_{FS}"]
    return {"rf": float(np.mean(e)), "en": en["rmse_log"],
            "rf_better_folds_pct": float(100 * np.mean(np.array(e) < np.array(en["folds"]))), "folds": e}


def u1_eval(u, rf_pred):
    en = pd.Series(json.load(open(RESULTS_DIR / "p5_unseen.json"))["predictions"]["en_discharge"])
    ns = u.policy.str.contains("newstructure").to_numpy()
    y = u.cycle_life.to_numpy()
    p = {"rf": rf_pred[u.index].to_numpy(), "en": en[u.index].to_numpy()}
    out = {}
    rng = np.random.default_rng(0)
    for g, m in (("all", np.ones(len(y), bool)), ("standard", ~ns), ("newstructure", ns)):
        s = {k: {"rmse": float(np.sqrt(np.mean((v[m] - y[m]) ** 2))),
                 "signed_pct": float(np.mean(100 * (v[m] - y[m]) / y[m]))} for k, v in p.items()}
        idx = rng.integers(0, m.sum(), size=(5000, m.sum()))
        r = {k: np.sqrt(((v[m] - y[m])[idx] ** 2).mean(1)) for k, v in p.items()}
        d = r["rf"] - r["en"]
        s["rf_minus_en"] = [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        out[g] = s
    return out


def run(verbose=True):
    df = load_features()
    tr = df[df.split == "train"]
    lo, hi = float(tr.cycle_life.min()), float(tr.cycle_life.max())
    cols = FEATURE_SETS[FS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = nested_cv(tr)
        rf = make_rf().fit(tr[cols].to_numpy(), np.log10(tr.cycle_life.to_numpy()))
        en_pred = predictions(df, fit_one(df, FS)[0], FS)
        seeds = []
        for s in range(5):
            m = make_rf(seed=s).fit(tr[cols].to_numpy(), np.log10(tr.cycle_life.to_numpy()))
            seeds.append(m)
    pred = pd.Series(10 ** rf.predict(df[cols].to_numpy()), index=df.index)
    paper = df[df.split != "unseen"]
    parts = {"train": paper.split == "train", "primary": paper.split == "primary_test",
             "primary_excl": (paper.split == "primary_test") & ~paper.anomalous,
             "secondary": paper.split == "secondary_test"}
    res = {"grid": {k.split("__")[1]: v for k, v in GRID.items()},
           "best_params": {k.split("__")[1]: v for k, v in rf.best_params_.items()},
           "nested_cv_train": cv,
           "scores": {s: scores(paper.cycle_life[m], pred[paper.index[m]]) for s, m in parts.items()},
           "robust": robust(df, pred[paper.index], lo, hi),
           "bootstrap_vs_en": bootstrap(df, {"rf": pred[paper.index], "en": en_pred}, lo, hi, [("rf", "en")]),
           "seed_range_rmse": {s: [float(f(scores(paper.cycle_life[m], 10 ** sm.predict(paper[cols].to_numpy())[m.to_numpy()])["rmse"]
                                            for sm in seeds)) for f in (min, max)]
                               for s, m in parts.items() if s in ("primary", "secondary")},
           "out_of_range": {c: {"life": float(df.cycle_life[c]), "rf": float(pred[c]), "en": float(en_pred[c])}
                            for c in ("b1c2", "b2c1")},
           "prediction_range": [float(pred[paper.index].min()), float(pred[paper.index].max())],
           "u1": u1_eval(df[df.split == "unseen"], pred),
           "predictions": pred.round(1).to_dict()}
    with open(RESULTS_DIR / "p6_tree.json", "w") as fp:
        json.dump(res, fp, indent=2, default=str)
    if verbose:
        report(res)
    return res


def report(r):
    c = r["nested_cv_train"]
    print(f"best params {r['best_params']}")
    print(f"nested CV (log10): RF {c['rf']:.4f}  EN {c['en']:.4f}  RF better in {c['rf_better_folds_pct']:.0f}% of folds")
    print("  ".join(f"{k} {v['rmse']:.0f}/{v['mape']:.1f}%" for k, v in r["scores"].items()))
    print(f"in-range primary {r['robust']['primary']['rmse_in_range']:.0f}; seed range {r['seed_range_rmse']}")
    for k, b in r["bootstrap_vs_en"].items():
        v = b["rf-en"]
        print(f"bootstrap {k} (in range): rf-en {v['rmse_diff']:+.0f} [{v['rmse_ci'][0]:+.0f}, {v['rmse_ci'][1]:+.0f}]")
    print("out of range", r["out_of_range"], "prediction range", [round(x) for x in r["prediction_range"]])
    for g, s in r["u1"].items():
        d = s["rf_minus_en"]
        print(f"u1 {g:12s} RF {s['rf']['rmse']:.0f} ({s['rf']['signed_pct']:+.0f}%)  EN {s['en']['rmse']:.0f} ({s['en']['signed_pct']:+.0f}%)"
              f"  rf-en {d[0]:+.0f} [{d[1]:+.0f}, {d[2]:+.0f}]")


if __name__ == "__main__":
    run()

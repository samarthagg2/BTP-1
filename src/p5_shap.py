"""P5: SHAP explanations of the main models on every test set.

Models (fitted on the training cells exactly as in src/p5_unseen.py): EN-F2, EN-F3,
EN-F4, GP-F4 and EN F3+F4. Exact SHAP values (shap.explainers.Exact) of the
predicted log10 life, with the model's own training cells as background
(independent masker). A SHAP value of +0.1 raises the prediction by a factor
10**0.1 = 1.26 relative to the mean training prediction.
Caveat: the independent masker treats features as independent; dq_log_min and
dq_log_var are almost collinear (rho ~0.99), so their split of credit is arbitrary
and they should be read together.

Usage: python3 -m src.p5_shap      -> results/p5_shap.json (+ per-cell values .csv)
"""
import json
import warnings

import numpy as np
import pandas as pd
import shap

from config import RESULTS_DIR
from src.features import FEATURE_SETS, load_features
from src.models import make_regressor
from src.reproduce import fit_one, uses_temperature

MODELS = {"EN-F2": ("en", "discharge"), "EN-F3": ("en", "full"), "EN-F4": ("en", "physics"),
          "GP-F4": ("gp", "physics"), "EN-F3+F4": ("en", "full_physics")}
CELLS = ["b1c2", "b2c1", "u1c44", "u1c9", "u1c7", "u1c3"]


def sets(df):
    u = df[df.split == "unseen"]
    ns = u.policy.str.contains("newstructure")
    return {"train": df[df.split == "train"], "primary": df[df.split == "primary_test"],
            "secondary": df[df.split == "secondary_test"], "u1_standard": u[~ns], "u1_newstructure": u[ns]}


def fit(df, kind, fs):
    tr = df[df.split == "train"]
    if uses_temperature(fs):
        tr = tr[~tr.temp_suspect]
    cols = FEATURE_SETS[fs]
    if kind == "en":
        model = fit_one(df, fs)[0]
    else:
        model = make_regressor(kind, len(cols)).fit(tr[cols].to_numpy(), np.log10(tr.cycle_life.to_numpy()))
    return model, cols, tr


def run(verbose=True):
    df = load_features()
    parts = sets(df)
    out, per_cell = {}, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (kind, fs) in MODELS.items():
            model, cols, tr = fit(df, kind, fs)
            bg = tr[cols].to_numpy()
            bg = np.where(np.isnan(bg), np.nanmedian(bg, axis=0), bg)
            f = lambda X, m=model: np.ravel(m.predict(X))
            explainer = shap.explainers.Exact(f, shap.maskers.Independent(bg, max_samples=len(bg)))
            res = {"features": cols, "base_value": None, "sets": {}}
            for key, d in parts.items():
                d = d[~d.temp_suspect] if uses_temperature(fs) else d
                sv = explainer(d[cols].to_numpy())
                res["base_value"] = float(np.ravel(sv.base_values)[0])
                vals = pd.DataFrame(sv.values, index=d.index, columns=cols)
                res["sets"][key] = {"mean_abs": vals.abs().mean().round(5).to_dict(),
                                    "mean": vals.mean().round(5).to_dict(), "n": int(len(d))}
                v = vals.copy()
                v.insert(0, "set", key); v.insert(0, "model", name)
                v["feature_values"] = [dict(zip(cols, np.round(r, 6))) for r in d[cols].to_numpy()]
                per_cell.append(v)
            out[name] = res
    cells = pd.concat(per_cell)
    cells.to_csv(RESULTS_DIR / "p5_shap_values.csv")
    out["examples"] = {}
    for c in CELLS:
        out["examples"][c] = {m: {k: float(x) for k, x in cells[(cells.index == c) & (cells.model == m)][MODELS_FEATS(m)].iloc[0].items()}
                              for m in MODELS if ((cells.index == c) & (cells.model == m)).any()}
    with open(RESULTS_DIR / "p5_shap.json", "w") as fp:
        json.dump(out, fp, indent=2)
    if verbose:
        report(out)
    return out


def MODELS_FEATS(m):
    return FEATURE_SETS[MODELS[m][1]]


def report(out):
    for name in MODELS:
        r = out[name]
        print(f"\n{name}  (base value log10 life {r['base_value']:.3f} = {10 ** r['base_value']:.0f} cycles)")
        t = pd.DataFrame({k: v["mean_abs"] for k, v in r["sets"].items()})
        s = pd.DataFrame({k: v["mean"] for k, v in r["sets"].items()})
        print("  mean |SHAP| (importance):"); print(t.round(3).to_string())
        print("  mean SHAP (direction):"); print(s.round(3).to_string())


if __name__ == "__main__":
    run()

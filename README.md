# BTP-1: Early Prediction of Li-ion Battery Cycle Life

B.Tech Project at IIT Bombay: a comparative study of ML models and physics-informed features for predicting lithium-ion cycle life from the first 100 cycles. It builds on:

> K. A. Severson, P. M. Attia, et al. *Data-driven prediction of battery cycle life before capacity degradation.* Nature Energy 4, 383–391 (2019). [doi:10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8)

The problem statement and research questions are in `report/sections/01b_problem.tex`. The execution plan with checkpoints is in [`PLAN.md`](PLAN.md).

## Repository layout

```
config.py              paths, constants, batch names (b1, b2, b3, u1)
src/convert.py         raw .mat (MATLAB v7.3) → lightweight cache (~70 MB per batch)
src/clean.py           cleaning rules (authors' + ours) → dataset.pkl, splits.json
src/checks.py          verifies the data against the paper's Supplementary Table 9
src/quality.py         per-batch data-quality report
src/data.py            load_dataset(), curve(cell, field, n): always use this to read curves
src/extras.py          per-cycle duration + temperature integral from raw traces
src/features.py        the paper's 20 early-cycle features (SI Table 1) -> features.csv
src/evaluate.py        metrics (RMSE, MAPE on cycles) and baselines
src/reproduce.py       P2: reproduce the paper's Table 1 -> results/p2_reproduction.json
src/screen_physics.py  P3: screen physics-informed candidates -> results/p3_screening.json
src/p3_evaluate.py     P3: F4 with the paper's model on the test sets -> results/p3_test_eval.json
src/pls.py, src/models.py  P4: PLS on the ΔQ curve; GP / SVR / additive-spline models
src/p4_pls.py, src/p4_models.py  P4: model comparison -> results/p4_pls.json, p4_models.json
src/horizon.py         P4: error vs number of early cycles N -> results/horizon.json
src/p5_unseen.py       P5: frozen models on the unseen batch, pre-registered hypotheses -> results/p5_unseen.json
src/p5_shap.py         P5: SHAP explanations -> results/p5_shap.json
src/p5_gp_full.py      P5 (exploratory): GP on F3+F4 -> results/p5_gp_full.json
tests/                 sanity tests (pytest)
notebooks/01_data_explorer.ipynb   visualise any cell / batch / flagged cell
notebooks/02_reproduction.ipynb    Phase 2 walkthrough: ΔQ, features, models, results vs paper
notebooks/03_physics_features.ipynb  Phase 3: physics feature screening, F4
notebooks/04_model_comparison.ipynb  Phase 4: PLS, GP, SVR, GAM; cycles N
notebooks/05_unseen_batch.ipynb      Phase 5: unseen batch, hypotheses, SHAP
data/processed/        caches (gitignored) + splits.json (tracked)
data/reference/        ground truth parsed from the paper's SI (tracked)
report/                IEEE-format LaTeX report (Overleaf); see report/README.md
```

## Data

Raw files come from https://data.matr.io/1. Put them in `data mat/` (gitignored).

| Key | File | Role |
|---|---|---|
| `b1` | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | train / primary test |
| `b2` | `2017-06-30_batchdata_updated_struct_errorcorrect.mat` | train / primary test |
| `b3` | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | secondary test |
| `u1` | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | unseen-batch test |

## Rebuild everything

```bash
pip install -r requirements.txt
python3 -m src.convert all     # .mat → data/processed/*_raw.pkl
python3 -m src.clean           # → data/processed/dataset.pkl + splits.json
python3 -m src.checks          # must print: ALL CHECKS PASSED
python3 -m src.extras          # raw traces -> data/processed/extras.pkl
python3 -m src.features        # -> data/processed/features.csv
python3 -m src.reproduce       # P2 reproduction table vs the paper
python3 -m src.screen_physics  # P3 physics-feature screening (~3 min)
python3 -m src.p3_evaluate     # P3 F4 test-set evaluation (F4 frozen)
python3 -m src.p4_pls          # P4 PLS (~2 min)
python3 -m src.p4_models       # P4 GP / SVR / GAM (~10 min)
python3 -m src.horizon         # P4 error vs number of cycles N (~1 min)
python3 -m src.p5_unseen       # P5 unseen batch (frozen models)
python3 -m src.p5_shap         # P5 SHAP
python3 -m pytest -q tests/
```

Splits: 41 train / 43 primary test / 40 secondary test (identical to the paper), plus 39 unseen cells.

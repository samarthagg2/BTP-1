# BTP-1 Execution Plan

**Project:** Early Prediction of Li-ion Battery Cycle Life: A Comparative Study of ML Models and Physics-Informed Features
**Problem statement:** `report/sections/01b_problem.tex` (RQ1–RQ5)
**Duration:** ~10 weeks · **Started:** 2026-11-03

> Rule of thumb: every phase ends with a **checkpoint** (what must be true before moving on). If a checkpoint fails, fix it before continuing. Most ML projects fail silently from bad data or leakage, not from bad models.

---

## 0. Ground rules (apply to every phase)

| Rule | Why | How |
|---|---|---|
| **A sample is a cell, never a cycle** | Otherwise the same cell ends up in train and test, so results look great but mean nothing | All splits, CV folds and metrics work on cell IDs |
| **Test sets are touched once** | Tuning on the test set inflates results | Primary/secondary/unseen-batch sets are used only in `evaluate.py`, never during tuning |
| **Features use only cycles ≤ N** | Using later cycles leaks the answer | Every feature function takes an `N` argument and slices `[:N]` |
| **Cycle indexing: `cycles['j']` = cycle j+1** | Verified on our data | Cycle 10 = `cycles['9']`, cycle 100 = `cycles['99']`. Write a helper `get_cycle(cell, n)` and never index directly |
| **Fixed seeds, saved splits** | Results must be reproducible | `SEED = 42` in `config.py`; splits saved once to `data/processed/splits.json` |
| **Scripts over notebooks for the pipeline** | Notebooks hide state and run cells out of order | Logic in `src/*.py`; notebooks only *call* it, for exploring and plotting |
| **Commit at every checkpoint** | Lets you roll back if something breaks | `git commit` after each ✅ checkpoint |
| **Log every result** | So the report table is built automatically, not copied by hand | Every run appends a row to `results/runs.csv` |

### Disk-space constraint (only ~17 GB free)
- Raw `.mat` files are several GB each. **Don't create full 1.9 GB pickles for the new batches.** Convert `.mat` directly to the lightweight cache (cycles 1–100 + summaries) and keep only that.
- Once the cache for the 2018-02-20 batch is built and verified, delete `data pkl/batch1.pkl` (1.9 GB). The `.mat` file can always rebuild it.
- Download batches **one at a time**: download, cache, verify, then optionally move the raw `.mat` to external storage or delete it.

---

## 1. Repository structure (set up in week 1)

```
BTP-1/
├── PLAN.md                  ← this file
├── README.md
├── requirements.txt         ← pinned versions
├── config.py                ← paths, SEED, N, voltage grid, EOL threshold
├── data mat/                ← raw .mat (gitignored)
├── data/processed/          ← lightweight caches + splits.json (gitignored except splits.json)
├── src/
│   ├── convert.py           ← .mat → lightweight cache (one function, all batches)
│   ├── clean.py             ← authors' cleaning rules + our rules (IR=0, glitches)
│   ├── features.py          ← F1, F2, F3, F4 feature functions, parameterised by N
│   ├── models.py            ← model factory + hyperparameter grids
│   ├── evaluate.py          ← metrics, baselines, results logging
│   └── run_experiments.py   ← the full grid (feature sets × models × seeds × N)
├── notebooks/
│   ├── 01_data_checks.ipynb
│   ├── 02_feature_eda.ipynb
│   └── 03_results.ipynb     ← reads results/runs.csv → tables & figures
├── results/
│   ├── runs.csv
│   └── figures/
├── tests/                   ← small sanity tests (see §8)
└── report/                  ← LaTeX (Overleaf)
```

**Environment (already installed, Python 3.12):** numpy 1.26, pandas, scipy, h5py 3.11, scikit-learn 1.4, xgboost 2.0, lightgbm 4.6, shap 0.48, matplotlib 3.8. Pin these in `requirements.txt`.

---

## 2. Phase P1 — Data (weeks 1–2)

**Goal:** a clean, verified, lightweight dataset of 124 + 39 cells with fixed splits.

### Tasks
- [x] **P1.1 Get the authors' processing code.** Clone `github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation` (link from the paper's *Code availability* section). Read its data-loading notebook and **write down the exact cleaning rules** in `report/notes.md`.
- [x] **P1.2 Get the Supplementary Information PDF** from the paper's Nature Energy page. Copy **Supplementary Table 1** (the exact feature definitions) into `notes.md`. F1–F3 must match it exactly.
- [x] **P1.3 Download the three batches** from https://data.matr.io/1 (`2017-05-12_…`, `2017-06-30_…`, `2018-04-12_…_batchdata_updated_struct_errorcorrect.mat`), **one at a time** (see the disk constraint above).
- [x] **P1.4 Write `src/convert.py`.** Base it on `BuildPkl_Batch1.ipynb` (the version that already works with h5py 3: `[()]`, not `.value`). For each cell, store:
  - `cycle_life`, `policy`, `barcode`, `channel_id`
  - full `summary` arrays (all cycles; they're small)
  - for cycles 1–100 only: `Qdlin`, `Tdlin`, `dQdV` (1000 points each), plus the raw `I, V, T, t, Qc, Qd` (optional, for the dQ/dV and relaxation features)
  - the `Vdlin` grid, once per batch
  - Output: `data/processed/<batch>.pkl`, expected to be **tens of MB, not GB**.
- [x] **P1.5 Write `src/clean.py`** with the authors' rules. Rules **as we recall them from their code (verify in P1.1):**
  - Batch 2017-05-12: remove cells that never reach 80% capacity.
  - Batch 2017-06-30: some cells are **continuations of 2017-05-12 cells**. Join their summary arrays onto the original cells (shifting cycle numbers) and drop the duplicates.
  - Batch 2018-04-12: remove cells with noisy or faulty channels.
  - **Expected result: 41 + 43 + 40 = 124 cells.** If the count differs, stop and recheck.
  - Our own rules: IR = 0 → NaN; flag single-cycle QD glitches (|QD − rolling median| > 0.02 Ah) → NaN; `chargetime` at the 20-min cap → NaN.
- [x] **P1.6 Build the splits** exactly as in the paper: 41 train / 43 primary test / 40 secondary test. Per the paper's Methods, the first two batches are split into two halves, each spanning the full range of cycle lives, and the third batch is the secondary test set. **Copy the exact indexing from the authors' code; don't reinvent it.** Save to `data/processed/splits.json` as lists of cell IDs. Add `unseen` = the 39 labelled cells of 2018-02-20.
- [x] **P1.7 Note the special cells** from the paper: (a) one **anomalous primary-test cell** that fails very early; the paper reports results with and without it. (b) **4 cells excluded from the full model** because temperature probes lost contact. Identify both by cell ID and record them in `splits.json`.

### Checkpoint ✅ P1 — PASSED 2026-11-03 (`python3 -m src.checks`: all 124 cells match SI Table 9 exactly)
- [x] Cell counts: train 41, primary 43, secondary 40, unseen 39.
- [x] Cycle-life range of the 124 cells ≈ 150–2300.
- [x] Spot-plot 5 random cells per batch (QD vs cycle; Qdlin at cycles 10 & 100). Nothing looks wrong.
- [x] `get_cycle(cell, 100)` returns the cycle whose summary `cycle == 100`.
- [x] No NaN in `Qdlin` for cycles 1–100 of any cell (or NaNs documented).
- [x] Cache total < 1 GB; commit.

---

## 3. Phase P2 — Baselines & reproduction (week 2)

**Goal:** answer RQ1, i.e. reproduce the paper's results.

### Tasks
- [ ] **P2.1 Implement F1–F3 in `features.py`**, exactly per Supplementary Table 1. Expected (verify against the table):
  - **F1 Variance (1):** log₁₀ |Var(ΔQ₁₀₀₋₁₀(V))|
  - **F2 Discharge (6):** log₁₀ of |min|, Var, skewness, kurtosis of ΔQ; QD at cycle 2; max QD − QD at cycle 2
  - **F3 Full (9):** log |min| and log Var of ΔQ; slope and intercept of a linear fit to QD over cycles 2–100; QD at cycle 2; mean charge time over cycles 2–6; integral of temperature over cycles 2–100; min IR (cycles 2–100); IR(100) − IR(2)
  - ⚠️ The paper's figure labels ΔQ as "Q₁₀₁ − Q₁₀", so its indexing may be off by one as well. Check against their code and use the same convention.
- [ ] **P2.2 Baselines in `evaluate.py`:** (a) predict the training-set mean of log life; (b) predict the training mean for that cell's charging policy (falling back to (a) if the policy isn't seen in training).
- [ ] **P2.3 Elastic net:** `StandardScaler` + `ElasticNetCV` inside one `Pipeline`. The paper used 4-fold CV + Monte Carlo sampling; use `RepeatedKFold(n_splits=4, n_repeats=10, random_state=SEED)`, `l1_ratio ∈ {0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1}`.
- [ ] **P2.4 Metrics:** RMSE (cycles) and MAPE (%) computed on **cycles** (10^prediction), plus R² on log scale.

### Target numbers (paper, Table 1)

| Model | RMSE train | RMSE primary (excl. anomaly) | RMSE secondary | %err train | %err primary (excl.) | %err secondary |
|---|---|---|---|---|---|---|
| Variance | 103 | 138 (138) | 196 | 14.1 | 14.7 (13.2) | 11.4 |
| Discharge | 76 | 91 (86) | 173 | 9.8 | 13.0 (10.1) | 8.6 |
| Full | 51 | 118 (100) | 214 | 5.6 | 14.1 (7.5) | 10.7 |

### Checkpoint ✅ P2
- [ ] Our numbers are in the **same ballpark** (within about ±20% relative, and the ranking between models is similar). Exact match isn't expected: the paper used MATLAB and its own random sampling.
- [ ] Both baselines are clearly worse than the variance model. If not, there's a bug.
- [ ] If the numbers are far off: check (1) the ΔQ cycle indexing, (2) the log of the *absolute* variance, (3) the split membership, (4) cleaning, before anything else.
- [ ] Commit; add a "Reproduction" subsection to the report.

---

## 4. Phase P3 — Physics-informed features F4 (weeks 3–4)

**Goal:** a 4th feature set grounded in degradation physics.

| Feature | Exact definition | Mechanism |
|---|---|---|
| `dqdv_peak_h_delta` | Smooth dQ/dV (Savitzky–Golay, window 51, order 3) on the `Vdlin` grid; within **3.0–3.3 V** find the main peak (min, since dQ/dV < 0); height = \|peak\|; feature = h(cycle 100) − h(cycle 10) | Loss of active material / lithium |
| `dqdv_peak_v_delta` | Voltage of that peak, cycle 100 − cycle 10 | Rise in resistance (overpotential) |
| `dqdv_peak_area_delta` | ∫ \|dQ/dV\| dV over 3.0–3.3 V, cycle 100 − cycle 10 | Capacity in that phase |
| `ir_sqrt_slope` | Least-squares slope of IR against √N, cycles 2–100, NaNs dropped | SEI growth ∝ √t |
| `temp_rise` | Mean of (Tmax − Tmin) over cycles 2–100 *or* the slope of Tmax over cycles 2–100 (pick one, document it) | Joule heating ∝ resistance |

**F4 = F3 + these 5 features** (14 total).

### Tasks
- [ ] Implement each feature as its own function with a docstring giving its definition.
- [ ] Plot each feature against log cycle life on the **training set only** (Spearman ρ).
- [ ] Check correlations among F4 features; drop or merge any pair with \|r\| > 0.95.

### Checkpoint ✅ P3
- [ ] No NaN/inf in F4 for any cell (or a documented imputation rule: training median).
- [ ] Peak detection checked visually on ~10 cells (the peak marker lands on the real peak).
- [ ] Commit; add a "Physics-informed features" subsection to the report.

---

## 5. Phase P4 — Model comparison (weeks 5–7)

**Goal:** answer RQ2–RQ4.

### Hyperparameter search (on the training set only, `RepeatedKFold(4, 5)`)

| Model | Library | Search space |
|---|---|---|
| Elastic net | sklearn `ElasticNet` | α: logspace(−4, 1, 30); l1_ratio: {0.1, .5, .7, .9, .95, .99, 1} |
| PLS | sklearn `PLSRegression` | n_components: 1 … min(p, 8) |
| SVR | sklearn `SVR(kernel='rbf')` | C: logspace(−2, 3, 12); γ: logspace(−4, 1, 12); ε: {0.01, 0.05, 0.1} |
| Gaussian process | sklearn `GaussianProcessRegressor` | Kernel `C·RBF(ARD) + WhiteKernel`; `n_restarts_optimizer=10`; `normalize_y=True` (fit by marginal likelihood, no grid) |
| Random forest | sklearn `RandomForestRegressor(n_estimators=500)` | max_depth {None, 3, 5, 8}; min_samples_leaf {1, 2, 4, 8}; max_features {sqrt, 0.5, 1.0} |
| XGBoost | `xgboost.XGBRegressor` | `RandomizedSearchCV`, 60 iterations: n_estimators 100–800; max_depth 2–4; learning_rate 0.01–0.2; subsample 0.6–1; colsample 0.6–1; reg_lambda 0.1–10 |
| MLP | sklearn `MLPRegressor(solver='lbfgs', max_iter=5000)` | hidden {(8,), (16,), (32,), (16, 16)}; alpha logspace(−4, 1, 6) |

- **Every model** sits inside `Pipeline([StandardScaler(), model])`, so scaling is learned on the training folds only.
- Target: log₁₀(cycle life) for all models.

### Tasks
- [ ] **P4.1 `run_experiments.py`:** loop over {F1, F2, F3, F4} × 7 models × seeds {0…9} (seeds only matter for RF, XGB and MLP; the others run once). Log each run to `results/runs.csv` with columns:
  `feature_set, model, seed, N, split, rmse, mape, r2, best_params, timestamp`
- [ ] **P4.2 Results table:** mean ± std over seeds for each (model, feature set) on primary and secondary. Bold the best result per column.
- [ ] **P4.3 Early-horizon study (RQ4):** recompute features for N ∈ {20, 40, 60, 80, 100} (ΔQ = Q_N − Q_10; for N = 20, also try Q_20 − Q_5) and rerun the **best 3 models** on the **best 2 feature sets**. Plot MAPE against N.
- [ ] **P4.4 Optional, only if ahead of schedule:** (a) XGBoost with `monotone_constraints` on the ΔQ variance (more variance ⇒ shorter life); (b) PLS or a 1D CNN on the raw 1000-point ΔQ curve.

### Checkpoint ✅ P4
- [ ] Every (model, feature set) beats the **policy-mean baseline**; anything that doesn't is reported as such.
- [ ] Training RMSE ≪ test RMSE for a model? → it's overfitting; note it in the report (expected for RF/MLP on F4).
- [ ] The results table is generated from `runs.csv` by `03_results.ipynb`, with no hand-copied numbers.
- [ ] Commit; tables and figures are in the report.

---

## 6. Phase P5 — Generalisation & interpretation (week 8)

### Tasks
- [ ] **P5.1 Unseen batch (RQ5):** apply the *already-trained* models (no refitting) to the 39 cells of 2018-02-20. Report RMSE/MAPE **overall, standard-only and `-newstructure`-only**. Expect worse results, and explain them with the protocol difference and the narrow life range.
- [ ] **P5.2 Interpretability:** for the best linear model, report the coefficients. For the best tree model, use SHAP (`shap.TreeExplainer`). For all models, use permutation importance on the primary test set. Does any F4 feature rank in the top 5?
- [ ] **P5.3 Error analysis:** residual against true life, residual by charging policy. Where do models fail: long-lived cells, or particular policies?
- [ ] **P5.4 Optional:** GP predictive intervals; coverage of the 95% interval on the test sets.

### Checkpoint ✅ P5
- [ ] Every RQ (1–5) has a figure or table that answers it.
- [ ] Commit.

---

## 7. Phase P6 — Write-up (weeks 9–10)

| Report section | Content | Source |
|---|---|---|
| Methods: data | Batches, cleaning, splits | P1 |
| Methods: features | F1–F4 definitions (table) | P2, P3 |
| Methods: models & protocol | Table of models, CV, metrics, baselines | P4 |
| Results: RQ1 | Reproduction table vs paper Table 1 | P2 |
| Results: RQ2/RQ3 | Main results table (models × feature sets); parity plots | P4 |
| Results: RQ4 | MAPE against N | P4.3 |
| Results: RQ5 | Unseen-batch table by group | P5.1 |
| Discussion | Importance plots; failure modes; limitations | P5 |
| Conclusion | Answers to RQ1–5 in one line each | all |

- [ ] Fill the red `[TODO]`s (department, roll no., guide, dates).
- [ ] Final figure regeneration from scripts; zip → Overleaf.
- [ ] Presentation: 10–12 slides following the same RQ structure.

---

## 8. Sanity tests (`tests/`, run before every commit)

- `test_counts`: split sizes are 41/43/40/39 and the splits don't overlap.
- `test_indexing`: `get_cycle(c, n)` matches `summary['cycle'] == n`.
- `test_no_future`: features computed with N = 50 don't change when cycles 51+ are deleted from the cell.
- `test_no_leak`: no test cell ID appears in any training fold.
- `test_features_finite`: every feature is finite for every training cell.

---

## 9. Risks & fallbacks

| Risk | Signal | Fallback |
|---|---|---|
| Not enough disk space | Download fails / <5 GB free | One batch at a time; delete `batch1.pkl`; use external drive |
| Can't reproduce the paper | Errors >2× paper | Recheck indexing/split/cleaning; if still off, report the honest difference and continue (RQ1 answered "partially") |
| Peak detection unstable | Peak marker misplaced on many cells | Wider smoothing window; or replace with area-only features |
| Non-linear models overfit | Train ≪ test error | Stronger regularisation; smaller grids; this is itself a finding for RQ2 |
| Running behind schedule | Checkpoint slips >1 week | Drop the optional items (P4.4, P5.4); cut the N study to {50, 100}; never cut P1 checks |

---

## 10. Weekly checkpoints with guide

| Week | Show | Decision needed |
|---|---|---|
| 2 | Data counts + reproduction table | OK to proceed with the paper's split? |
| 4 | F4 feature plots | Feature set frozen |
| 7 | Main results table + MAPE-vs-N | Which results go in the main text |
| 8 | Unseen-batch + importance | Final story |
| 10 | Draft report | Submit |

---

## Definition of done
1. `python src/run_experiments.py` regenerates `results/runs.csv` from the caches.
2. `03_results.ipynb` regenerates every table and figure in the report from `runs.csv`.
3. The report answers RQ1–RQ5, each with evidence.
4. Repo pushed; README explains how to reproduce.

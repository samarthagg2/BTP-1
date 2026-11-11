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
- [x] **P2.1 F1–F3 in `src/features.py`**, exactly per SI Table 1 / Note 1 (all 20 candidates computed; `FEATURE_SETS` picks the paper's subsets). Indexing resolved: SI Note 1 defines ΔQ = Q₁₀₀ − Q₁₀ (the "Q₁₀₁" label belongs to a different experiment). Temperature integral needs raw traces → `src/extras.py`.
- [x] **P2.2 Baselines in `src/evaluate.py`:** constant = training **geometric** mean (reproduces SI Table 3 exactly; the arithmetic mean does not); policy mean.
- [x] **P2.3 Elastic net (`src/reproduce.py`):** median imputer + `StandardScaler` + `ElasticNetCV`, `RepeatedKFold(4, 10)`; variance model = OLS; full model excludes the 4 temp-suspect cells.
- [x] **P2.4 Metrics:** RMSE and MAPE on cycles, R² on log scale. Stability over 10 CV seeds.
- [x] Found + fixed two data faults on the way (cleaning rules **B9** stuck IR logger, **B10** time-stamp gaps; both hit `b1c18`).

### Result ✅ P2 PASSED (2026-11-04) — `results/p2_reproduction.json`, report §"Reproduction"

| Model | RMSE train | RMSE primary (excl.) | RMSE secondary | %err train | %err primary (excl.) | %err secondary |
|---|---|---|---|---|---|---|
| Constant | 327 (327) | 401 (401) / 399 (399) | 511 (511) | 29.6 (29.6) | 35.0 / 28.2 | 36.1 |
| Variance | 104 (103) | 138 (138) / 138 (138) | 196 (196) | 14.1 | 14.7 / 13.2 | 11.4 |
| Discharge | 69 (76) | 90 (91) / 85 (86) | 167 (173) | 9.2 | 12.6 / 9.4 | 9.9 |
| Full | 67 (51) | 125 (118) / 122 (100) | 220 (214) | 6.8 | 11.0 / 8.0 | 11.2 |

Ours (paper). Variance model exact; ranking identical on every split; all 9 full-model coefficient signs match SI Fig. 5. Remaining full-model gap is mostly one cell (b1c2, 2237 cycles → 67 % of primary squared error).

### Target numbers (paper, Table 1)

| Model | RMSE train | RMSE primary (excl. anomaly) | RMSE secondary | %err train | %err primary (excl.) | %err secondary |
|---|---|---|---|---|---|---|
| Variance | 103 | 138 (138) | 196 | 14.1 | 14.7 (13.2) | 11.4 |
| Discharge | 76 | 91 (86) | 173 | 9.8 | 13.0 (10.1) | 8.6 |
| Full | 51 | 118 (100) | 214 | 5.6 | 14.1 (7.5) | 10.7 |

### Checkpoint ✅ P2
- [x] Our numbers are in the **same ballpark** (within about ±20% relative, and the ranking between models is similar).
- [x] Both baselines are clearly worse than the variance model.
- [x] Commit; add a "Reproduction" subsection to the report.

---

## 4. Phase P3 — Physics-informed features F4 (weeks 3–4)
**Goal:** a 4th feature set grounded in degradation physics, judged on quality, not quantity (RQ3, part 1).
### Selection rules (training cells only; test sets untouched)
1. physical mechanism; 2. information beyond `dq_log_var` (within-batch t given ΔQ, paired CV gain);
3. same sign within b1 and b2 (otherwise a batch proxy); 4. gain beyond a batch label; 5. measurable within 100 cycles.
### Tasks
- [x] P3.1 Screening module `src/screen_physics.py` → `results/p3_screening.json/.csv` (19 candidates, ~3 min, reads raw traces).
- [x] P3.2 F4 implemented in `src/features.py` (`margin_rate`, `overhang_after10`; `FEATURE_SETS["physics"]`).
- [x] P3.3 Within-batch correlation table for every candidate; robustness of the overhang feature to the smoothing window and to the b1 test pause (cycles 12–13).
- [x] P3.4 Nested CV on training cells: F1/F2/F3/F4/F3+F4 with the paper's elastic-net protocol.
- [x] P3.5 Notebook `03_physics_features.ipynb`; report section "Physics-informed features"; figures `report/scripts/make_p3_figures.py`.
- [x] P3.6 Tests: F4 finite for all 163 cells; margin_rate identity; no data after cycle 100 used.
### Result ✅ P3 DONE (2026-11-07)
| Finding | Evidence (training cells) |
|---|---|
| Mechanism features (dQ/dV peaks, CV-hold growth, ΔQ(2 V), IR √N, heating) add nothing beyond ΔQ | all negative CV gain on top of `dq_log_var` |
| Coulombic efficiency not measurable | median QD/QC slightly > 1 |
| Fade curvature, policy plating index, fast-charge temperature, cycle duration = batch proxies | sign flips or vanishes within batch |
| ΔQ amplitude grows linearly in cycles 10–100 | power-law exponent median 1.01 (IQR 0.98–1.11) |
| `margin_rate` = log10((QD2 − 0.88)/sd ΔQ) | beats `dq_log_var` in 96% of folds (−9% error); strong in b1 (t +5.3), none in b2 |
| `overhang_after10` = capacity rise cycle 10 → early peak | t −3.9 (b1), −1.6 (b2) given ΔQ; +11% beyond batch label; robust to window; not a pause artefact |
| **F4 = [margin_rate, overhang_after10]** | nested CV: RMSE log 0.0509 / 77 cycles vs F3 (9 features) 0.0510 / 89; F3+F4 no better |
| `-newstructure` = 5-s rests (as b3) | step breakdown; rest time / fast-charge temperature kept as hypotheses for P5 |
**Test sets (used once, F4 frozen; `python3 -m src.p3_evaluate`):** RMSE/%err primary · secondary —
discharge 90/12.6% · 167/9.9%; full 125/11.0% · 220/11.2%; **F4 114/9.9% · 170/11.1%** (seed range 113–116 · 170–171).
F4 beats full on both test sets, ties discharge on secondary, worse RMSE than discharge on primary because of b1c2
(2237 cycles, outside training range; 58% of F4's squared error). In training range (b1c2, b2c1 removed): full 71, F4 75,
discharge 86. Paired bootstrap: only full-vs-discharge on secondary is clearly resolved (+48, CI +7 to +79).
F3+F4 (11 features): primary 122, secondary 205 — slightly better than F3, still far worse than F4 alone on secondary: the elastic net keeps the batch-proxy features (temperature integral, charge time) and shrinks overhang_after10 to ~0.
Caveats: 19 candidates on 41 cells → optimistic (training CV showed F4 > discharge; tests do not); b2 evidence weak (narrow lives); overhang behaviour batch-specific (u1 rises late) → check in P4/P5.
---
## 5. Phase P4 — Model comparison (RQ2, RQ4)
**Scope (decided 2026-11-11):** 2–3 model types different from the paper's; no trees or neural networks (41 cells → memorisation; trees cannot extrapolate beyond the training life range).
### Protocol
Variants and settings chosen by nested CV on the training cells only; every registered configuration evaluated **once** on the test sets, all reported; u1 untouched (P5). Scores: official RMSE/%, in-range RMSE (test cells inside 300–2160 cycles), paired bootstrap vs elastic net on the same features.
### Tasks
- [x] P4.1 PLS on the whole ΔQ curve (`src/pls.py`, `src/p4_pls.py`): raw / asinh / amplitude+shape / F4+shape (block-scaled). Pre-registered: asinh (primary), F4+shape (secondary).
- [x] P4.2 GP (linear + Matern ARD + noise; bounded: length-scale ≥ 1 SD, noise floor), SVR (RBF), additive splines (GAM) on F2 and F4 (`src/models.py`, `src/p4_models.py`).
- [x] P4.3 Screened extra feature ΔV(Q) (voltage drop at fixed charge): ρ 0.99 with dq_log_var, no gain → rejected.
- [x] P4.4 Horizon study N = 20…150 (`src/horizon.py`; cache extended to 160 cycles, features at N = 100 unchanged) incl. GP-F4.
- [x] P4.5 Notebook `04_model_comparison.ipynb`; report Section IX; figures `make_p4_figures.py`; tests `tests/test_models.py`.
### Result ✅ P4 DONE (2026-11-11)
| Config | Primary | In range | Secondary | Bootstrap vs EN (prim / sec) |
|---|---|---|---|---|
| EN F2 / EN F4 | 90 / 114 | 86 / 75 | **167** / 170 | — |
| GP F2 / **GP F4** | **85** / 103 | 78 / **64** | 174 / 180 | −7 n.s. / **−11 [−21,−3]** ; +8, +9 n.s. |
| SVR F2 / F4 | 176 / 107 | 65 / 76 | 169 / 190 | **−20 [−49,−3]** / 0 ; n.s. |
| GAM F2 / F4 | 242 / 92 | 79 / 66 | 169 / 207 | n.s. |
| PLS curve / F4+shape | 138 / 151 | 104 / 117 | **3431** / 285 | secondary significantly worse |
- RQ2: no model beats the linear model on the new batch; non-linear better within training batches; SVR/GAM break outside training range (b1c2: 1186 / 742); PLS fails on new batch (narrow weight at the 3.15 V plateau edge, which shifts in b3).
- GP-F4: calibrated 95% intervals on primary (95%), overconfident on secondary (85%).
- RQ4: ≤30 cycles only policy/batch info helps; 40–80 ≈ half the baseline error; 100 clearly better; >100 no gain. Paper's log|·| features blow up at N = 50 (b1c0 → 13,393 cycles).
- Caveat: 8 new configurations + horizon sweep on the same test sets → only bootstrap-resolved differences count.
---
## 6. Phase P5 — Unseen batch & interpretation (RQ5)
### Protocol
Frozen P2–P4 models (no refit/re-tune) applied once to u1 (39 cells: 30 standard, 9 newstructure). Five hypotheses with decision rules written into `src/p5_unseen.py` **before** any u1 life was used.
### Tasks
- [x] P5.1 Input drift (no lives): standard cells temp integral z +2.9, QD2 +2.6, overhang +2.6; ~1/3 of cells beyond 3 SD.
- [x] P5.2 `src/p5_unseen.py` → `results/p5_unseen.json` (all models, per group, bootstrap, GP coverage, verdicts).
- [x] P5.3 SHAP (`src/p5_shap.py`, exact, training background) → `results/p5_shap.json`, `p5_shap_values.csv`.
- [x] P5.4 Exploratory (not pre-registered): GP on F3+F4 (`src/p5_gp_full.py`).
- [x] P5.5 Notebook `05_unseen_batch.ipynb`; report Section X; figures `make_p5_figures.py`; tests `tests/test_unseen.py`.
### Result ✅ P5 DONE (2026-11-11)
| Model | u1 all | standard | newstructure |
|---|---|---|---|
| constant | 226 | 173 | 351 |
| EN-F2 | 260 (+27% bias) | 104 | 506 |
| EN-F3 | 138 | 51 | 271 |
| EN-F4 / GP-F4 | 178 / 158 | 97 / 49 | 325 / 316 |
| EN-F1 / EN-F3+F4 | **119** / 124 | 77 / **28** | **204** / 254 |
Verdicts: H1 no, H2 no (overhang mechanism not confirmed at cell level), H3 yes (weak), H4 yes, H5 yes (GP-F4 coverage 87%: 97% std, 56% new).
SHAP: F2 over-predicts standard cells via high initial capacity; F3 right via temperature integral (cycle duration); GP-F4 beats EN-F4 because it learned a saturating overhang effect; newstructure over-prediction comes from ΔQ itself (cause open). GP on F3+F4: memorises (train 19), no gain.
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

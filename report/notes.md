# Report scratch notes

Dump raw findings, numbers, ideas and figure ideas here; move them into `sections/` once they're ready.

## Key numbers (from figures/stats.json)
- 47 cells, 39 with valid cycle life (8 VarCharge/SLOWCYCLE removed)
- Cycle life: 392–1186 (mean 566, median 472)
- Standard (30): 392–514, mean 453 | -newstructure (9): 777–1186, mean 942
- All 9 base policies reach 80 % SOC in ~10 min
- Pearson(log Var ΔQ, log life): all −0.92 | standard −0.38 | newstructure −0.27
- Early IR (median cycles 2–6): newstructure ~15.0 mΩ vs standard ~16.8 mΩ (ranges overlap); 6 cells have IR = 0 early (missing)

## Open questions
- What exactly does `-newstructure` change in the test procedure?
- Are b1c22/b1c23 continued in `2018-04-03_varcharge_...mat`?
- What is inside `finaldata_4.mat`, `finaldata_6and8.mat`, `initialdata_all.mat` (MATLAB objects; need MATLAB/Octave to open)?

## To add to the report
- [ ] Department, roll no., guide name
- [ ] Timeline / milestones
- [ ] Results from the paper's three batches once converted
- [ ] Feature table + model results

## Problem statement (finalised 2026-11-03)
Comparative study: 7 models (EN, PLS, SVR, GPR, RF, XGBoost, MLP) × 4 feature sets (F1 variance, F2 discharge, F3 full, F4 full+physics)
on Severson 124 cells (41/43/40 split), plus 2018-02-20 batch as unseen-batch test. RQ1–RQ5 in sections/01b_problem.tex.
Trajectory prediction dropped (too complex for timeline).

## P1 — Authors' cleaning rules (from external/severson_code, verified 2026-11-03)
Python `Load Data.ipynb` (we follow this one):
- b1 (2017-05-12): drop b1c8, b1c10, b1c12, b1c13, b1c22 (don't reach 80 %)
- b2 (2017-06-30): b2c7, b2c8, b2c9, b2c15, b2c16 continue b1c0..b1c4 → append summaries (cycle offset = len(b1 summary)), cycle_life += [662, 981, 1060, 208, 482]; drop from b2
- b3 (2018-04-12): drop b3c37, b3c2, b3c23, b3c32, b3c42, b3c43 (noisy channels)
- Split on combined order (b1, b2, b3): test = arange(0, nb1+nb2, 2) ∪ {83}; train = arange(1, nb1+nb2−1, 2); secondary = last nb3
MATLAB `LoadData.m` differs slightly: add_len one smaller ([661, 980, 1059, 207, 481]); b3 removes channel 46 + cells whose final QD > 0.885 + noisy [3, 40, 41]; label recomputed as first cycle with QD < 0.88. We keep `life_from_qd` alongside the stored cycle_life to compare.
Modelling code is NOT public (academic licence from Braatz) → feature definitions must come from Supplementary Table 1.

## Download links (data.matr.io/1, verified)
- 2017-05-12: https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download (3.03 GB)
- 2017-06-30: https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download (2.01 GB)
- 2018-04-12: https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download (3.24 GB)

# BTP-1: Battery Cycle Life Prediction

B.Tech Project at IIT Bombay on data-driven prediction of lithium-ion battery cycle life, based on:

> K. A. Severson, P. M. Attia, et al. *Data-driven prediction of battery cycle life before capacity degradation.* Nature Energy 4, 383–391 (2019). [doi:10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8)

## Contents

| File | Description |
|---|---|
| `BuildPkl_Batch1.ipynb` | Converts `2018-02-20_batchdata_updated_struct_errorcorrect.mat` to `batch1.pkl` |
| `BuildPkl_Batch2.ipynb` | Same for the 2017-06-30 batch |
| `BuildPkl_Batch3.ipynb` | Same for the 2018-04-12 batch |
| `data.ipynb` | Exploration of batch data: structure, filtering, capacity-fade plots |

## Data

The raw data is not included in this repo (files are several GB). Download the `.mat` batch files from the
[Toyota Research Institute / Stanford battery dataset](https://data.matr.io/1/) and place them in:

```
data mat/   # raw .mat files
data pkl/   # generated .pkl files (output of BuildPkl_*.ipynb)
```

Each pickle is a dict keyed by cell ID (e.g. `b1c0`) with `cycle_life`, `charge_policy`,
`summary` (per-cycle IR, QC, QD, Tavg, Tmin, Tmax, chargetime, cycle) and `cycles`
(per-cycle traces I, Qc, Qd, Qdlin, T, Tdlin, V, dQdV, t).

## Requirements

`numpy`, `pandas`, `matplotlib`, `h5py`, `scipy`

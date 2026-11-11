"""Project-wide constants and paths. Import from here; never hard-code."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data mat"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

SEED = 42
NOMINAL_CAPACITY_AH = 1.1
EOL_AH = 0.88                  # 80 % of nominal
N_EARLY = 100                  # default number of early cycles for features
N_CACHE = 160                  # cycle records cached per cell (covers cycle 150 with margin; RQ4 horizon study)
VDLIN = np.linspace(3.5, 2.0, 1000)   # voltage grid of Qdlin / Tdlin / dQdV

# Raw batch files (data.matr.io/1) and the short names used throughout
BATCHES = {
    "b1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",   # Severson batch 1
    "b2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",   # Severson batch 2
    "b3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",   # Severson batch 3
    "u1": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",   # unseen-batch test
}

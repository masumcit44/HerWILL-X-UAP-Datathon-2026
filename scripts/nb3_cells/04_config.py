# Notebook cell 04 - code: configuration, imports, paths
# --- configuration & paths (relative paths only, no Drive, no personal paths)
import os
import re
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

SEED = 42
np.random.seed(SEED)

N_SPLITS = 5
CLASSES = [0, 1, 2]
RUN_GPU_RETRAIN = False      # Mode B: set True to retrain the GPU sources
USE_DUP_OVERRIDE = True      # duplicate-text override (Submissions 1/2 parity)
SIMPLEX_STEP = 0.05

# predeclared similarity-weighted-prior config (NOT tuned by score)
SID_CFG = dict(KW=120, k=180, beta=10.0, gamma=1.0)

WORD_KW = dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60000,
               sublinear_tf=True, dtype=np.float32)
CHAR_KW = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60000,
               sublinear_tf=True, dtype=np.float32)
LR_KW = dict(solver="lbfgs", C=4.0, class_weight="balanced", max_iter=2000,
             random_state=SEED)

HERE = Path.cwd()
BUNDLE_CANDIDATES = [HERE / "artifacts_submission_3",
                     HERE.parent / "artifacts_submission_3",
                     HERE.parent / "uap_submission" / "reproducible" / "artifacts_submission_3"]
BUNDLE = next((p for p in BUNDLE_CANDIDATES if p.is_dir()), None)
assert BUNDLE is not None, "artifacts_submission_3/ not found next to this notebook"
print("bundle dir:", BUNDLE)


def find_data(name):
    """Locate a competition CSV using RELATIVE paths only (no hardcoded drives)."""
    for base in (Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent):
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "{} not found - place it next to this notebook (cwd: {})".format(
            name, Path.cwd()))


TRAIN_CSV = find_data("train.csv")
TEST_CSV = find_data("test.csv")
SAMPLE_CSV = find_data("sample_submission.csv")
print("data dir:", TRAIN_CSV.parent)

CONFIG = json.load(open(BUNDLE / "config_submission_3.json"))
print("config:", {k: CONFIG[k] for k in
                  ("source_order", "final_weights", "use_dup_override", "mode")})

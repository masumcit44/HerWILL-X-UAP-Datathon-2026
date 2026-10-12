"""Fast smoke test of the notebook's tail cells (10, 11, 12) using the stack
arrays already produced by the reproduction, so we do not pay for a second full
training pass just to catch a formatting bug."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir("repro_s4")
nb = json.load(open("../uap_submission/reproducible/submission_reproducible_file_4.ipynb",
                   encoding="utf-8"))
codes = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
pick = {}
for s in codes:
    for m in ["10. text-only ensemble", "11. test prediction", "12. final summary"]:
        if m in s:
            pick[m] = s
print("tail cells found:", sorted(pick))

S = (42, 7, 2026)
o = np.mean([np.load("stack_all_400_s%d.npz" % s)["oof"] for s in S], axis=0)
t = np.mean([np.load("stack_all_400_s%d.npz" % s)["test"] for s in S], axis=0)
tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")
y = tr["y"].to_numpy(int)
folds = np.load("final_sources/fold_ids.npy").astype(np.int64)
CFG = json.load(open("../uap_submission/reproducible/artifacts_submission_4/"
                    "config_submission_4.json"))
FOLD_OFFSETS = CFG["class_correction"]["fold_offsets"]
FINAL_OFFSET = CFG["class_correction"]["final_offset"]


def macro_f1(a, b):
    cm = np.bincount(a * 3 + b, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    p = tp / np.maximum(cm.sum(0), 1)
    r = tp / np.maximum(cm.sum(1), 1)
    return float((2 * p * r / np.maximum(p + r, 1e-12)).mean())


lo = np.log(np.clip(o, 1e-300, None))
c = lo.copy()
for k in range(5):
    c[folds == k] += np.asarray(FOLD_OFFSETS[k], float)
pred = c.argmax(1)
cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
tp = np.diag(cm).astype(float)
p = tp / np.maximum(cm.sum(0), 1)
r = tp / np.maximum(cm.sum(1), 1)

G = dict(np=np, pd=pd, y=y, tids=te["id"].to_numpy(np.int64), folds=folds,
         n_tr=len(y), n_te=len(te), CLASSES=[0, 1, 2], CFG=CFG,
         POOLED=macro_f1(y, pred), per_class=2 * p * r / np.maximum(p + r, 1e-12),
         ACC=float((pred == y).mean()),
         folds_scores=[macro_f1(y[folds == k], pred[folds == k]) for k in range(5)],
         SEED_SCORES={"42": 0.693895, "7": 0.694031, "2026": 0.692881},
         FOLD_OFFSETS=FOLD_OFFSETS, FINAL_OFFSET=FINAL_OFFSET,
         log_tst=np.log(np.clip(t, 1e-300, None)),
         Ftr=pd.read_parquet("Ftr.parquet"),
         sample=pd.read_csv("sample_submission.csv"),
         HERE=Path.cwd(), BUNDLE=Path("../uap_submission/reproducible/artifacts_submission_4"),
         macro_f1=macro_f1, sha256_of=lambda p: "smoke")
G["SRC"] = {k: "cache/v7_text/%s" % v for k, v in {
    "tfidf": "text_tfidf_final.npz", "knn": "text_knn_final.npz", "sid": "sid.npz",
    "pid": "plain_id.npz"}.items()}
G["SRC"]["e5"] = "final_sources/emb_e5-base_final.npz"
G["SRC"]["idh"] = "final_sources/xlmr_base_idhead_final.npz"

# cell 8 supplies the override functions; run it for real
exec(pick["10. text-only ensemble"].split("# --- 10.")[0] if False else "", G)
import re as _re
G["re"] = _re
G["train"] = tr
G["test"] = te
src8 = next(s for s in codes if "8. duplicate-text override" in s)
exec(src8, G)
# cell 9 is the expensive ablation - stub it out
for m in ["10. text-only ensemble", "11. test prediction", "12. final summary"]:
    print("--- executing", m, "---")
    exec(pick[m], G)
print("SMOKE OK")

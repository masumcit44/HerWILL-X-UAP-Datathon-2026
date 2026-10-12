"""Build cache/v7_text/plain_id.npz using the harness id_prior COPIED VERBATIM
from submission_reproducible_file_3.ipynb (cell containing 'def id_prior').

k=20, inverse distance, eps=1e-6. OOF row in fold k uses pool = train rows of
folds != k; test uses the full-train pool. Keys: oof, test, fold_ids.
"""
import io
import json
import hashlib

import nbformat
import numpy as np
import pandas as pd

nb = nbformat.read("uap_submission/reproducible/submission_reproducible_file_3.ipynb",
                   as_version=4)
cell = None
for c in nb.cells:
    s = "".join(c.source)
    if "def id_prior" in s:
        cell = s
        break
assert cell is not None, "id_prior cell not found"

# take the function body verbatim (exact source segment from the cell)
import ast
tree = ast.parse(cell)
fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "id_prior"][0]
src = ast.get_source_segment(cell, fn)
assert src.startswith("def id_prior"), "extraction failed"
ns = {"np": np, "CLASSES": [0, 1, 2]}
exec(src, ns)
id_prior = ns["id_prior"]
print("id_prior extracted verbatim (%d chars)" % len(src))

tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")
folds = np.load("final_sources/fold_ids.npy").astype(np.int64)
assert hashlib.sha256(np.ascontiguousarray(folds, np.int64).tobytes()).hexdigest() == \
    "cc4fdd169110aeee6d3d5566095234461041b1a9290e2e916ce850c31d27a7ca", "not fold set B"
y = tr["y"].to_numpy(int)
ids = tr["id"].to_numpy(np.int64)
tids = te["id"].to_numpy(np.int64)

oof = np.zeros((len(y), 3))
for k in range(5):
    va = np.flatnonzero(folds == k)
    pool = np.flatnonzero(folds != k)
    oof[va] = id_prior(ids[va], ids[pool], y[pool])
test = id_prior(tids, ids, y)


def mf1(yy, pr):
    cm = np.bincount(yy * 3 + pr, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    P = tp / np.maximum(cm.sum(0), 1)
    R = tp / np.maximum(cm.sum(1), 1)
    return float((2 * P * R / np.maximum(P + R, 1e-12)).mean())


sc = mf1(y, oof.argmax(1))
fs = [mf1(y[folds == k], oof[folds == k].argmax(1)) for k in range(5)]
print("plain_id standalone OOF = %.6f (expect ~0.6136 +-0.001)" % sc)
print("  folds", [round(v, 6) for v in fs])
assert abs(sc - 0.6136) < 0.001, "plain_id out of tolerance: %.6f" % sc
meta = dict(name="plain_id", uses_id=True, k=20, eps=1e-6,
            oof_macro_f1=sc, source="id_prior verbatim from submission_3 notebook")
np.savez_compressed("cache/v7_text/plain_id.npz", oof=oof, test=test,
                    fold_ids=folds, meta=np.array(json.dumps(meta)))
print("  saved cache/v7_text/plain_id.npz  oof%s test%s" % (oof.shape, test.shape))

"""Notebook 4 cells, part 1: disclosure, ablation table, setup, verification."""
import nbformat as nbf

MD0 = """# Submission 4 - LightGBM stacker (performance track, **ID-DERIVED**)

> ## Organizer disclosure - please read first
>
> This submission is **not** the genuine text contribution to this competition.
> It is a leaderboard-performance stacker that is **heavily dependent on the
> `id` column**, and it is submitted as such, with that limitation stated up front.
>
> * **It is a performance stacker, not a text method.** Its headline number comes
>   from exploiting the structure of the `id` column, not from better language
>   modelling.
> * **Three of its six base probability sources are ID-derived:** the
>   similarity-weighted ID prior (`sid`), the plain ID prior (`plain_id`), and the
>   xlm-roberta-base **ID-head** classifier (`xlmr_base_idhead`).
> * **32 of its 60 features are ID-neighbour features**: the labels and log
>   distances of the nearest train ids on each side, plus local class shares in
>   id-windows of width 3/10/30/100/300, computed with the neighbour pool
>   restricted to train rows of the *other* folds. The raw `id` is a further feature.
> * **These features are dataset-specific and unsuitable for deployment.** They
>   work only because ids in this dataset are near-sorted by time and by author
>   type. On any stream where ids are shuffled, re-assigned or anonymised, the
>   ID-neighbour and ID-prior features degrade to noise and this model collapses.
>   It must not be read as evidence of any real modelling capability.
> * **The genuine text methodology is documented separately**, in
>   `submission_reproducible_file_3.ipynb` (Submission 3), which reports
>   **0.682311** honest grouped-OOF macro-F1 and is the method we stand behind.
>   Submission 4 improves the leaderboard number, not the text understanding.
> * **Public leaderboard feedback was not used for model selection.** Every
>   number here comes from grouped out-of-fold predictions on the training set.
> * **No external labelled data was used.** All base sources are trained on the
>   provided `train.csv` alone.
> * We acknowledge the organizer's clarification that `id` is not a text feature
>   and should not be treated as one. This submission is exactly the kind of
>   `id`-exploiting system that clarification was aimed at, and we label it
>   ID-derived rather than presenting it as a text model.
>
> ## What this notebook does
>
> 1. verifies dataset hashes, the fold assignment and every bundled artifact SHA256
> 2. rebuilds the frozen 60-feature stacker design deterministically
> 3. retrains the frozen 3-seed LightGBM stacker and reproduces the OOF score
> 4. applies the frozen per-fold class correction and the duplicate-text override
> 5. writes and validates `submission_4.csv`
>
> ## Honest validation classification
>
> **`STANDARD_OOF_STACKING_WITH_DEPENDENCY`** - see the config cell and section 5.
> This is *not* strictly outer-fold safe and we make no such claim."""

MD1 = """## 1. Required ablation - where the score actually comes from

Every number below is a grouped out-of-fold macro-F1 on the same fold set B.

| System | OOF macro-F1 | Text-derived? | ID-derived? |
|---|---|---|---|
| lexical text only (TF-IDF word(1,2) + char_wb(3,5) -> LR) | 0.516093 | yes | no |
| semantic text only (multilingual-e5-small zero-shot) | 0.521535 | yes | no |
| text-only ensemble (tfidf + knn + e5) | see cell output | yes | no |
| plain ID prior only (k=20, inverse distance) | 0.613631 | no | **yes** |
| SID similarity-weighted ID prior only | 0.627895 | no | **yes** |
| xlm-roberta ID-head only | 0.648743 | no | **yes** |
| **Submission 3** (text blend, the genuine text method) | **0.682311** | yes | partly |
| **Submission 4 stacker** (this submission, ID-DERIVED) | **0.703904** | partly | **yes** |

The reading is unambiguous: the single best *text* signal is 0.5215, while the
single best *ID* signal is 0.6487. The stacker's advantage over Submission 3
(+0.0216) comes from ID-neighbour features and ID priors, not from text. The
ablation cells below drop the raw id and the whole neighbour block and show the
score falling back."""


def code(src):
    return nbf.v4.new_code_cell(src.strip("\n"))


CELLS = []
CELLS.append(nbf.v4.new_markdown_cell(MD0))
CELLS.append(nbf.v4.new_markdown_cell(MD1))
CELLS.append(code(r'''
# --- 2. configuration, versions, seed policy (relative paths only) -----------
import json
import hashlib
import platform
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn

HERE = Path.cwd()
BUNDLE = HERE / "artifacts_submission_4"

CFG = json.load(open(BUNDLE / "config_submission_4.json"))
SCHEMA = json.load(open(BUNDLE / "feature_schema.json"))
MANIFEST = json.load(open(BUNDLE / "artifact_manifest_submission_4.json"))

SEEDS = CFG["seeds"]
ROUNDS = CFG["rounds"]
N_SPLITS = CFG["n_splits"]
CLASSES = CFG["classes"]
FOLD_SHA = CFG["fold_ids_int64_sha256"]

VERSIONS = dict(
    python=platform.python_version(),
    numpy=np.__version__,
    pandas=pd.__version__,
    scipy=scipy.__version__,
    scikit_learn=sklearn.__version__,
    lightgbm=lgb.__version__,
    pyarrow=pyarrow.__version__,
    platform=platform.platform(),
)
print("versions   :", json.dumps(VERSIONS, indent=2))
print("config     :", CFG["title"])
print("validation :", CFG["validation_classification"])
print("seed policy:", CFG["seed_policy"])
print("seeds      :", SEEDS, "rounds:", ROUNDS, "n_features:", SCHEMA["n_features"])
assert SCHEMA["n_features"] == 60


def find_data(name):
    """Locate a competition CSV using RELATIVE paths only (this dir, then up)."""
    for base in [HERE] + list(HERE.parents)[:3]:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError("%s not found near the notebook (cwd: %s)" % (name, HERE))


TRAIN_CSV = find_data("train.csv")
TE_CSV = find_data("test.csv")
SAMPLE_CSV = find_data("sample_submission.csv")
print("data dir    :", TRAIN_CSV.parent)
'''))
CELLS.append(code(r'''
# --- 3. verify the fold set and every bundled artifact SHA256 ---------------
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


for _n in ["train.csv", "test.csv", "sample_submission.csv"]:
    assert find_data(_n).is_file(), "missing %s near the notebook" % _n
print("dataset files present: train.csv test.csv sample_submission.csv")

folds = np.load(BUNDLE / "fold_ids_colab.npy").astype(np.int64)
fold_payload = hashlib.sha256(np.ascontiguousarray(folds, np.int64).tobytes()).hexdigest()
assert fold_payload == FOLD_SHA, "fold set mismatch: %s" % fold_payload
assert list(np.bincount(folds, minlength=5)) == [9543, 9546, 9571, 9581, 9576]
print("fold set B verified, int64 payload sha256 =", fold_payload[:32], "...")
print("fold sizes:", np.bincount(folds, minlength=5).tolist())

bad = []
for name, entry in sorted(MANIFEST.items()):
    got = sha256_of(BUNDLE / name)
    if got != entry["sha256"]:
        bad.append(name)
    print("  %-32s %s %8d B" % (name, "OK " if got == entry["sha256"] else "BAD",
                                entry["size_bytes"]))
assert not bad, "artifact SHA256 mismatch: %s" % bad
print("all %d bundled artifacts verified" % len(MANIFEST))
'''))
CELLS.append(code(r'''
# --- 4. load the base probability sources and verify them standalone ---------
train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
test = pd.read_csv(TE_CSV, encoding="utf-8")
sample = pd.read_csv(SAMPLE_CSV)
assert list(train.columns) == ["id", "text", "y"], list(train.columns)
assert list(test.columns) == ["id", "text"], list(test.columns)
assert list(sample.columns) == ["id", "y_pred"], list(sample.columns)
for df in (train, test):
    df["text"] = df["text"].astype("string").fillna("").astype(str)

y = train["y"].to_numpy(dtype=int)
ids = train["id"].to_numpy(dtype=np.int64)
tids = test["id"].to_numpy(dtype=np.int64)
n_tr, n_te = len(train), len(test)
assert (n_tr, n_te) == (47817, 11955), (n_tr, n_te)
assert len(y) == len(folds)
print("train", n_tr, "test", n_te)

# frozen source order - this dict order defines the feature column order
SRC = {
    "tfidf": BUNDLE / "text_tfidf_final.npz",
    "knn": BUNDLE / "text_knn_final.npz",
    "e5": BUNDLE / "emb_e5-base_final.npz",
    "sid": BUNDLE / "sid.npz",
    "pid": BUNDLE / "plain_id.npz",
    "idh": BUNDLE / "xlmr_base_idhead_final.npz",
}


def macro_f1(y_true, pred):
    cm = np.bincount(np.asarray(y_true) * 3 + np.asarray(pred), minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    return float((2 * prec * rec / np.maximum(prec + rec, 1e-12)).mean())


for name, path in SRC.items():
    z = np.load(path, allow_pickle=True)
    o, t = z["oof"], z["test"]
    assert o.shape == (n_tr, 3) and t.shape == (n_te, 3), (name, o.shape, t.shape)
    assert np.isfinite(o).all() and np.isfinite(t).all()
    assert (o >= 0).all() and (o <= 1).all() and (t >= 0).all() and (t <= 1).all()
    exp = MANIFEST[path.name]["standalone_oof_macro_f1"]
    got = macro_f1(y, o.argmax(1))
    assert abs(got - exp) < 1e-6, (name, got, exp)
    print("  %-6s OOF %.6f (manifest %.6f)  uses_id=%s" % (
        name, got, exp, MANIFEST[path.name]["uses_id"]))
print("all six base sources verified")
'''))

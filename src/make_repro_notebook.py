"""
src/make_repro_notebook.py -- builds the self-contained competition notebook

Output: uap_submission/reproducible/submission_reproducible_file_1.ipynb

The notebook is fully self-contained (no imports from src/, no downloads, no
hardcoded paths): it reads train.csv / test.csv / sample_submission.csv from its
own directory (relative paths only) and writes ./submission/submission_1.csv.

Run:  python src/make_repro_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "uap_submission" / "reproducible"
OUT_IPYNB = OUT_DIR / "submission_reproducible_file_1.ipynb"

# Numbers documented from PART 1 (reports/revalidation_grouped_metrics.json).
PART1_ALPHA = 0.40
PART1_OOF = 0.639579
PART1_FOLDS = [0.642898, 0.632324, 0.641935, 0.639295, 0.641384]
PART1_TEXT = 0.512943
PART1_ID = 0.614135
PART1_PER_CLASS = (0.757270, 0.620188, 0.541279)

cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


MD_INTRO = f"""
# HerWILL X UAP Datathon 2026 -- reproducible submission #1

**Model:** word + character TF-IDF with balanced Logistic Regression (C = 4.0),
blended with an **ID-locality prior** at **alpha = {PART1_ALPHA:.2f}**, plus a
duplicate-text override rule.

This notebook is **fully self-contained**: no Google Drive mount, no hardcoded
personal paths, no external downloads, no pretrained weights, no network access.

**How to run it**

1. Place `train.csv`, `test.csv` and `sample_submission.csv` in the *same
   directory as this notebook*.
2. Restart the kernel and run all cells top to bottom (`Restart & Run All`).
3. The notebook writes `./submission/submission_1.csv` (relative path).

**Seeds:** `SEED = 42` is applied to `random`, `numpy`, scikit-learn's estimators
and the cross-validation splitter, so repeated runs are reproducible.

### Data overview

| file | rows | columns | notes |
| --- | --- | --- | --- |
| `train.csv` | 47,817 | `id`, `text`, `y` | 0 = Explicitly Toxic/Hateful, 1 = Subtly Toxic, 2 = Non-toxic/Neutral |
| `test.csv` | 11,955 | `id`, `text` | no labels; ids continue the train id range |
| `sample_submission.csv` | 11,955 | `id`, `y_pred` | defines the required row order and schema |

Train + test ids together form the **contiguous range 0..59771 with zero
overlap** (verified during re-validation), and the text is bilingual (Bangla,
English, Banglish/mixed) with emoji, punctuation and hashtags preserved.

Class balance in `train.csv`: 0 -> 23,756 (49.68%), 1 -> 18,163 (37.98%),
2 -> 5,898 (12.33%). Primary metric: **Macro F1**.
"""

MD_ARCH = f"""
### Model architecture

Two TF-IDF views of the **cleaned** text are hstacked into one sparse matrix:

| block | settings |
| --- | --- |
| word TF-IDF | `analyzer="word"`, `ngram_range=(1, 2)`, `min_df=2`, `max_features=60000`, `sublinear_tf=True` |
| char TF-IDF | `analyzer="char_wb"`, `ngram_range=(3, 5)`, `min_df=2`, `max_features=60000`, `sublinear_tf=True` |

Cleaning is deliberately minimal - only URL and mention masking: the literal
`URL -> " URL "` and `@mention -> " USER "`, then whitespace collapsed. Nothing
else is changed: no lowercasing, no stemming, and Bangla script / emoji /
hashtags / repeated characters / punctuation are preserved.

Classifier: `LogisticRegression(solver="lbfgs", C=4.0, class_weight="balanced",
max_iter=2000, random_state=42)` - native multinomial `lbfgs` with balanced
class weights handles the 3-class problem and the class imbalance directly.
Per-class probabilities come from `predict_proba`.

### Leakage control

The TF-IDF vocabulary is fitted **inside each CV fold on that fold's training
split only**, then once more on the full training set for the final model.
`dtype=float32` is used for the sparse matrices (memory only; it does not change
the model).
"""


MD_PRIOR = """
### ID-locality prior (transductive, competition-specific -- not for production)

**Formula.** For a query id `q` and a pool of labelled training ids `T` with
labels `y`:

1. take the `k = 20` nearest ids of `q` by `|t - q|` (ties broken by the smaller
   id, so the rule is deterministic),
2. weight each neighbour by `w = 1 / (distance + 1e-6)`,
3. sum the weights per class and normalise:

```
P(class = c | q) = sum_{j in kNN(q)} w_j * 1[y_j = c]  /  sum_{j in kNN(q)} w_j
```

Fewer than `k` available neighbours -> use whatever exists; a query with no
usable neighbour falls back to the uniform `1/3` prior.

**Why it works here.** `train.csv` and `test.csv` ids are contiguous and
non-overlapping, so nearby ids come from the same scrape order (same threads /
bursts) and labels are *locally correlated along the id axis*. That makes the id
axis a predictive signal **for this file only**.

**Why it is NOT a production feature.** It is *transductive*: it exploits how the
competition files were interleaved. On new data, or if ids were randomised, the
signal disappears. It must never be reused as a general-purpose feature outside
this competition.

**Leak-safety.** The prior only ever receives training ids and training labels,
so validation/test labels cannot enter the pool; the re-validation asserted
`val_ids (intersection) pool = empty` for all five folds.
"""

MD_CV = """
### Validation protocol (Part 1 re-validation)

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` where the
**group is the cleaned + lowercased text**. Every duplicate text therefore stays
in a single fold, removing the duplicate-text train/val leakage that plain
`StratifiedKFold` allowed (previously disclosed as an optimism source).

Part-1 independent re-validation (everything recomputed from scratch):

| metric | value |
| --- | --- |
| text-only OOF Macro F1 | **{text}** |
| id-prior-only OOF Macro F1 | **{idv}** |
| blend @ alpha = {alpha:.2f} OOF Macro F1 | **{oof}** |
| per-fold blend Macro F1 | {folds} |
| per-class F1 (blend) | {pc} |
| groups / duplicate groups / groups crossing folds | 47,157 / 480 / 0 |

The blend sits slightly below the earlier plain-stratified number (0.642185),
which is expected with leakage-free folds and was **not** chased: alpha was
re-searched on the new folds and **{alpha:.2f} remained the argmax**, so 0.40 is
the final chosen alpha.
""".format(text=f"{PART1_TEXT:.6f}", idv=f"{PART1_ID:.6f}", alpha=PART1_ALPHA,
           oof=f"{PART1_OOF:.6f}",
           folds=", ".join(f"{f:.6f}" for f in PART1_FOLDS),
           pc=" / ".join(f"{v:.6f}" for v in PART1_PER_CLASS))


# --------------------------------------------------------------------------- #
# Code cells (raw strings: they contain regexes)
# --------------------------------------------------------------------------- #
CODE_ENV = r"""
import random, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
print("python:", sys.version.split()[0], "| numpy:", np.__version__,
      "| pandas:", pd.__version__)
print("SEED =", SEED, "-> random.seed, np.random.seed, sklearn random_state=SEED")
"""

CODE_MODEL = r"""
URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def clean_text(text):
    # URL -> ' URL ', @mention -> ' USER ', collapse whitespace. Nothing else.
    s = "" if text is None else str(text)
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    return WS_RE.sub(" ", s).strip()


WORD_KW = dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60000,
               sublinear_tf=True, dtype=np.float32)
CHAR_KW = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60000,
               sublinear_tf=True, dtype=np.float32)
LR_KW = dict(solver="lbfgs", C=4.0, class_weight="balanced", max_iter=2000,
             random_state=SEED)
ALPHA_GRID = [round(0.05 * i, 2) for i in range(21)]
K_NEIGHBORS, ID_EPS = 20, 1e-6
CLASSES = (0, 1, 2)


def build_matrix(word, char, texts):
    return sp.hstack([word.transform(texts), char.transform(texts)],
                     format="csr").astype(np.float32)


def fit_text_model(texts, y, tag="model"):
    t0 = time.time()
    word = TfidfVectorizer(**WORD_KW)
    word.fit(texts)
    char = TfidfVectorizer(**CHAR_KW)
    char.fit(texts)
    X = build_matrix(word, char, texts)
    clf = LogisticRegression(**LR_KW)
    clf.fit(X, y)
    print(f"{tag}: X={X.shape} nnz={X.nnz:,} word={len(word.vocabulary_):,} "
          f"char={len(char.vocabulary_):,} fit={time.time() - t0:.1f}s")
    return word, char, clf


def blend(p_text, p_id, alpha):
    p = alpha * np.asarray(p_text) + (1.0 - alpha) * np.asarray(p_id)
    return p / p.sum(axis=1, keepdims=True)


def macro_f1(y_true, proba):
    return float(f1_score(y_true, np.asarray(proba).argmax(axis=1), average="macro",
                          labels=list(CLASSES), zero_division=0))


def unanimous_lookup(texts, labels):
    # cleaned text -> label for groups where EVERY occurrence agrees (train only).
    df = pd.DataFrame({"t": pd.Series(list(texts), dtype="object"),
                       "y": np.asarray(labels)})
    g = df.groupby("t")["y"].agg(["nunique", "first"])
    uni = g[g["nunique"] == 1]
    lookup = {str(t): int(v) for t, v in uni["first"].items()}
    n_conflicting = int((g["nunique"] > 1).sum())
    return lookup, n_conflicting


def apply_override(texts, pred, lookup):
    out = np.asarray(pred).copy()
    hits = 0
    for i, t in enumerate(texts):
        v = lookup.get(str(t))
        if v is not None:
            out[i] = v
            hits += 1
    return out, hits
"""

CODE_DATA = r"""
def find_data(name):
    # Locate a competition CSV using RELATIVE paths only (no hardcoded drives).
    for base in (Path.cwd(), Path.cwd().parent):
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{name} not found - place train.csv / test.csv / sample_submission.csv "
        f"next to this notebook (current directory: {Path.cwd()})")


TRAIN_CSV = find_data("train.csv")
TEST_CSV = find_data("test.csv")
SAMPLE_CSV = find_data("sample_submission.csv")
print("data directory:", TRAIN_CSV.parent)

train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
test = pd.read_csv(TEST_CSV, encoding="utf-8")
sample = pd.read_csv(SAMPLE_CSV, encoding="utf-8")
assert list(train.columns) == ["id", "text", "y"], list(train.columns)
assert list(test.columns) == ["id", "text"], list(test.columns)
assert list(sample.columns) == ["id", "y_pred"], list(sample.columns)

for df in (train, test):
    df["text"] = df["text"].astype("string").fillna("").astype(str)
train["clean"] = train["text"].map(clean_text)
test["clean"] = test["text"].map(clean_text)

y = train["y"].to_numpy(dtype=int)
ids = train["id"].to_numpy(dtype=np.int64)
test_ids = test["id"].to_numpy(dtype=np.int64)
clean_texts = train["clean"].to_numpy()     # OBJECT array (never np.asarray(list))
test_clean = test["clean"].to_numpy()
assert set(map(int, np.unique(y))) <= {0, 1, 2}
assert len(set(ids) & set(test_ids)) == 0, "train/test ids must not overlap"
assert len(sample) == len(test), "sample_submission must match test row count"
print("train", train.shape, "| test", test.shape, "| sample", sample.shape)
print("class counts:", {int(c): int((y == c).sum()) for c in CLASSES})
"""

CODE_PRIOR = r"""
def id_prior(query_ids, train_ids, train_labels, k=K_NEIGHBORS, eps=ID_EPS,
             n_classes=3, tag="prior"):
    # k-nearest-id label prior (vectorised). Only TRAIN ids/labels are passed in,
    # so a validation/test row's own label can never influence its own prediction.
    q = np.asarray(query_ids, dtype=np.int64).ravel()
    t = np.asarray(train_ids, dtype=np.int64).ravel()
    yy = np.asarray(train_labels, dtype=np.int64).ravel()
    assert t.size == yy.size and t.size == np.unique(t).size
    order = np.argsort(t, kind="stable")           # ascending id -> deterministic ties
    s_ids, s_y = t[order], yy[order]
    n, kk = len(s_ids), min(k, len(s_ids))

    pos = np.searchsorted(s_ids, q, side="left")
    cand = pos[:, None] + np.arange(-k, k + 1)[None, :]      # k left + k right candidates
    valid = (cand >= 0) & (cand < n)
    cc = np.clip(cand, 0, n - 1)
    d = np.abs(s_ids[cc] - q[:, None]).astype(np.float64)
    d = np.where(valid & (s_ids[cc] != q[:, None]), d, np.inf)   # no self-match
    sel = np.argsort(d, axis=1, kind="stable")[:, :kk]           # exact k nearest
    dsel = np.take_along_axis(d, sel, axis=1)
    ysel = s_y[cc[np.arange(len(q))[:, None], sel]]

    good = np.isfinite(dsel)
    w = np.where(good, 1.0 / (np.where(good, dsel, 0.0) + eps), 0.0)
    P = np.stack([(w * (ysel == c)).sum(axis=1) for c in range(n_classes)], axis=1)
    tot = P.sum(axis=1, keepdims=True)
    P = P / np.where(tot > 0, tot, 1.0)
    P[tot.ravel() <= 0] = 1.0 / n_classes
    overlap = int(np.isin(q, s_ids).sum())
    if overlap:
        print(f"WARNING {tag}: {overlap} query ids also in pool -> self-matches excluded")
    print(f"{tag}: queries={len(q)} pool={n} k={k} pool_disjoint_from_queries={overlap == 0}")
    assert overlap == 0, "leak: query ids present in the prior pool"
    return P
"""

CODE_CV = r"""
# ---- duplicate-safe folds: group = cleaned + lowercased text ----
keys = pd.Index([str(t).lower().strip() for t in clean_texts], dtype="object")
groups = pd.factorize(keys, sort=False)[0]
dup_groups = int((pd.Series(groups).value_counts() > 1).sum())
print("groups:", len(np.unique(groups)), "| duplicate groups:", dup_groups)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_ids = np.full(len(y), -1, dtype=int)
for k, (_tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
    fold_ids[va] = k
crossing = pd.DataFrame({"g": groups, "f": fold_ids}).groupby("g")["f"].nunique()
print("groups spanning >1 fold:", int((crossing > 1).sum()))
assert int((crossing > 1).sum()) == 0, "duplicate text leaked across folds"

oof_text = np.zeros((len(y), 3))
oof_id = np.zeros((len(y), 3))
oof_override = np.zeros(len(y), dtype=int)
fold_rows = []
for k in range(5):
    va = np.flatnonzero(fold_ids == k)
    tr = np.flatnonzero(fold_ids != k)
    assert int(np.intersect1d(ids[va], ids[tr]).size) == 0  # prior pool vs val ids

    word, char, clf = fit_text_model(clean_texts[tr], y[tr], tag=f"fold{k}")
    p_text = clf.predict_proba(build_matrix(word, char, clean_texts[va]))
    p_id = id_prior(ids[va], ids[tr], y[tr], tag=f"fold{k} prior")
    p_blend = blend(p_text, p_id, 0.40)

    # leak-free override: lookup built ONLY from this fold's training split
    lookup, n_conflict = unanimous_lookup(clean_texts[tr], y[tr])
    assert set(lookup).issubset(set(clean_texts[tr].tolist())), "override leak"
    ov, hits = apply_override(clean_texts[va], p_blend.argmax(1), lookup)

    m_t, m_i, m_b = (macro_f1(y[va], p_text), macro_f1(y[va], p_id),
                     macro_f1(y[va], p_blend))
    m_o = float(f1_score(y[va], ov, average="macro", labels=list(CLASSES),
                         zero_division=0))
    oof_text[va] = p_text
    oof_id[va] = p_id
    oof_override[va] = ov
    fold_rows.append({"fold": k, "n_val": len(va), "text": round(m_t, 6),
                      "id_prior": round(m_i, 6), "blend_0.40": round(m_b, 6),
                      "blend_override": round(m_o, 6), "override_rows": hits})
    print(f"fold{k}: text={m_t:.6f} id={m_i:.6f} blend@0.40={m_b:.6f} | "
          f"override_rows={hits} ({m_b:.6f} -> {m_o:.6f})")

print(pd.DataFrame(fold_rows).to_string(index=False))
print(f"pooled OOF: text={macro_f1(y, oof_text):.6f} "
      f"id={macro_f1(y, oof_id):.6f} "
      f"blend@0.40={macro_f1(y, blend(oof_text, oof_id, 0.40)):.6f}")

# ---- fresh alpha grid on pooled OOF probabilities (0.00-1.00, step 0.05) ----
grid = [(a, macro_f1(y, blend(oof_text, oof_id, a))) for a in ALPHA_GRID]
chosen_alpha, best_f1 = max(grid, key=lambda t: t[1])
print("alpha grid:", {f"{a:.2f}": round(f, 6) for a, f in grid})
print(f"CHOSEN ALPHA = {chosen_alpha:.2f} | OOF Macro F1 = {best_f1:.6f}")
assert abs(chosen_alpha - 0.40) < 1e-12, "alpha changed - investigate before submitting"

base = blend(oof_text, oof_id, chosen_alpha).argmax(1)
ov_rows = int((oof_override != base).sum())
ov_after = float(f1_score(y, oof_override, average="macro", labels=list(CLASSES),
                          zero_division=0))
print(f"override on OOF: {ov_rows} rows, {best_f1:.6f} -> {ov_after:.6f} "
      f"(delta {ov_after - best_f1:+.6f}) - expected 0 with grouped folds")
"""

CODE_FINAL = r"""
print("=== final model: refit on the FULL training set ===")
word_f, char_f, clf_f = fit_text_model(clean_texts, y, tag="full")
p_text_test = clf_f.predict_proba(build_matrix(word_f, char_f, test_clean))
p_id_test = id_prior(test_ids, ids, y, tag="test prior")   # pool = ALL train ids
p_test = blend(p_text_test, p_id_test, chosen_alpha)
pred_pure = p_test.argmax(axis=1)

# duplicate-text override: lookup built from TRAIN only, conflicting groups skipped
lookup_full, n_conflict = unanimous_lookup(clean_texts, y)
pred_final, n_hits = apply_override(test_clean, pred_pure, lookup_full)
print(f"final alpha={chosen_alpha:.2f} | mean max prob={p_test.max(axis=1).mean():.4f}")
print(f"override: {n_hits}/{len(test_ids)} test rows matched a unanimous training "
      f"text | {int((pred_final != pred_pure).sum())} labels changed | "
      f"conflicting groups skipped={n_conflict}")

# bias sanity check: predicted test share vs train share, flag drift above 10pp
train_share = {int(c): round(100.0 * float((y == c).mean()), 2) for c in CLASSES}
pred_share = {int(c): round(100.0 * float((pred_final == c).mean()), 2) for c in CLASSES}
drift = {int(c): round(pred_share[c] - train_share[c], 2) for c in CLASSES}
flagged = [int(c) for c in CLASSES if abs(drift[c]) > 10.0]
print("train %:", train_share)
print("pred  %:", pred_share)
print("drift pp:", drift, "-> FLAGGED >10pp" if flagged else "-> all within 10pp")
assert not flagged, f"class share drift exceeds 10pp: {flagged}"
"""

CODE_WRITE = r"""
out_dir = Path("submission")            # RELATIVE path: ./submission/
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "submission_1.csv"

sub = pd.DataFrame({"id": test_ids, "y_pred": pred_final.astype(int)})[["id", "y_pred"]]


def validate_frame(df, where):
    problems = []
    if list(df.columns) != ["id", "y_pred"]:
        problems.append(f"columns={list(df.columns)}")
    if len(df) != 11955:
        problems.append(f"rows={len(df)} (expected 11955)")
    if not (df["id"].to_numpy() == sample["id"].to_numpy()).all():
        problems.append("id order differs from sample_submission.csv")
    if int(df.isna().to_numpy().sum()) > 0:
        problems.append("nulls present")
    vals = df["y_pred"].to_numpy()
    if not set(map(int, pd.unique(vals))) <= {0, 1, 2}:
        problems.append(f"labels outside 0/1/2: {sorted(pd.unique(vals))}")
    if vals.dtype.kind not in "iu":
        problems.append(f"dtype not integer: {df['y_pred'].dtype}")
    print(f"VALIDATION {where}:", "OK" if not problems else problems)
    assert not problems, problems
    return True


validate_frame(sub, "BEFORE writing")
sub.to_csv(out_path, index=False, encoding="utf-8")

reread = pd.read_csv(out_path, encoding="utf-8")
validate_frame(reread, "AFTER writing (re-read from disk)")
print("wrote:", out_path.as_posix(), "(relative) ->", out_path.resolve())
print("label counts:", {int(c): int((reread["y_pred"] == c).sum()) for c in CLASSES})
print(f"DONE: alpha={chosen_alpha:.2f}, OOF Macro F1={best_f1:.6f}, "
      f"submission rows={len(reread)}")
"""
# --------------------------------------------------------------------------- #
# Markdown cells that reference the generated cell inventory (kept after main)
# --------------------------------------------------------------------------- #


DUP_MARKER_NOTE = "kept-after-main-final"


MD_OVERRIDE_KEPT_FINAL = f"""
### Duplicate-text override rule

After blending, one *conservative* override is applied:

* group **training** rows by cleaned text,
* keep only groups whose label is **unanimous** across all occurrences
  (conflicting groups are skipped entirely -- 53 such groups in this dataset),
* any **test** row whose cleaned text exactly matches one of those keys takes
  that unanimous label; everything else keeps `argmax(blend)`.

**Leak-safety.** For the OOF measurement the lookup is rebuilt from each fold's
*training split only*, so a validation row can never be overridden using its own
label. Because folds are duplicate-grouped, a validation row's exact twin always
shares its fold, so the override affects **0 OOF rows** under this protocol
(Macro F1 0.639579 -> 0.639579, delta +0.000000). It still affects the test set
(272 rows matched, 39 labels changed), where no fold constraint applies.

### Final chosen alpha (Part-1 re-validation)

**alpha = {PART1_ALPHA:.2f}** on the text model (the id prior gets
`1 - alpha = {1 - PART1_ALPHA:.2f}`), confirmed as the argmax of a 0.00-1.00
grid (step 0.05) over grouped OOF probabilities, with

**OOF Macro F1 = {PART1_OOF:.6f}**
(5-fold `StratifiedGroupKFold`, grouped by cleaned+lowercased text).

The alpha grid is recomputed below, so the printed value is a fresh
reproduction rather than a stored number.
"""



def main() -> None:
    parts = [("md", MD_INTRO), ("code", CODE_ENV), ("md", MD_ARCH),
             ("code", CODE_MODEL), ("code", CODE_DATA), ("md", MD_PRIOR),
             ("code", CODE_PRIOR), ("md", MD_CV), ("code", CODE_CV),
             ("md", MD_OVERRIDE_KEPT_FINAL), ("code", CODE_FINAL), ("code", CODE_WRITE)]
    nb_cells = []
    for i, (kind, src) in enumerate(parts, start=1):
        src = src.strip("\n")
        cell = (nbf.v4.new_markdown_cell(src) if kind == "md"
                else nbf.v4.new_code_cell(src))
        cell["id"] = f"cell-{i:02d}"
        nb_cells.append(cell)
    nb = nbf.v4.new_notebook(cells=nb_cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0],
                          "mimetype": "text/x-python",
                          "file_extension": ".py"},
        "title": "HerWILL X UAP Datathon 2026 -- reproducible submission #1",
    })
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(OUT_IPYNB))
    n_code = sum(1 for kind, _ in parts if kind == "code")
    print(f"wrote {OUT_IPYNB} | cells={len(parts)} (markdown={len(parts) - n_code}, "
          f"code={n_code}) | execution_count set to None (fresh on Run All)")


if __name__ == "__main__":
    main()
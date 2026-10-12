"""
src/experiment_v2_harness.py
============================

Shared harness for the `src/experiment_v2_*.py` family. Goal: try to beat the
locked, validated baseline (pooled grouped-fold leak-free OOF Macro F1 =
0.639579, alpha = 0.40, k = 20) WITHOUT touching

    uap_submission/reproducible/submission_reproducible_file_1.ipynb
    uap_submission/reproducible/submission/submission_1.csv

Hard rules honoured here (carried over verbatim from the baseline):

* Validation protocol is fixed: StratifiedGroupKFold(5, shuffle=True,
  random_state=42) with group = cleaned + lowercased text. Never plain
  StratifiedKFold.
* Any ID-locality feature is leak-free: a query row's own label is never in the
  prior pool. This is asserted per fold (val ids disjoint from train ids) and
  the prior function is independently re-implemented and diffed.
* The reported metric is always pooled OOF Macro F1 over the 5 grouped folds,
  directly comparable to 0.639579.
* CPU only, no network, no external data, no pretrained weights.
* SEED = 42 everywhere.

Nothing in this file is imported by the locked notebook; the baseline is left
byte-identical. This module only *re-implements* the baseline primitives so a
new experiment can reuse them, and caches the per-fold TF-IDF matrices so
several experiments share one vectorisation pass.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Constants (identical to the locked baseline)
# --------------------------------------------------------------------------- #
SEED = 42
N_SPLITS = 5
ALPHA_GRID = [round(0.05 * i, 2) for i in range(21)]
ALPHA_BASELINE = 0.40
K_NEIGHBORS, ID_EPS = 20, 1e-6
CLASSES = (0, 1, 2)

BASELINE_OOF_F1 = 0.639579          # the number every experiment is compared to

WORD_KW = dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60000,
               sublinear_tf=True, dtype=np.float32)
CHAR_KW = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60000,
               sublinear_tf=True, dtype=np.float32)
LR_KW = dict(solver="lbfgs", C=4.0, class_weight="balanced", max_iter=2000,
             random_state=SEED)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache" / "v2"
PRED_DIR = ROOT / "predictions" / "v2"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+", flags=re.UNICODE)
REPEAT_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE)
HASH_RE = re.compile(r"#\w+", flags=re.UNICODE)

BN_LO, BN_HI = 0x0980, 0x09FF
EMOJI_RANGES = (
    (0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x1F000, 0x1F2FF),
    (0xFE00, 0xFE0F), (0x1F1E6, 0x1F1FF), (0x2190, 0x21FF),
)
PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


# --------------------------------------------------------------------------- #
# Text cleaning -- byte-identical logic to the baseline notebook
# --------------------------------------------------------------------------- #
def clean_text(text):
    # URL -> ' URL ', @mention -> ' USER ', collapse whitespace. Nothing else.
    s = "" if text is None else str(text)
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    return WS_RE.sub(" ", s).strip()


def group_key(clean):
    # Grouping key: cleaned text, lowercased and stripped.
    return str(clean).lower().strip()


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data(verbose: bool = True):
    train = pd.read_csv(ROOT / "train.csv", encoding="utf-8")
    test = pd.read_csv(ROOT / "test.csv", encoding="utf-8")
    sample = pd.read_csv(ROOT / "sample_submission.csv", encoding="utf-8")
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
    clean_texts = train["clean"].to_numpy()      # OBJECT array, never asarray(list)
    test_clean = test["clean"].to_numpy()

    assert set(map(int, np.unique(y))) <= set(CLASSES)
    assert len(set(ids) & set(test_ids)) == 0, "train/test ids must not overlap"
    assert len(sample) == len(test)
    if verbose:
        print(f"train {train.shape} | test {test.shape} | sample {sample.shape}")
        print("class counts:", {int(c): int((y == c).sum()) for c in CLASSES})
    return dict(train=train, test=test, sample=sample, y=y, ids=ids,
                test_ids=test_ids, clean_texts=clean_texts, test_clean=test_clean)


def sha1_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def input_checksums() -> dict:
    return {name: {"sha1": sha1_file(ROOT / name),
                   "bytes": (ROOT / name).stat().st_size}
            for name in ("train.csv", "test.csv", "sample_submission.csv")}


# --------------------------------------------------------------------------- #
# Fold construction -- StratifiedGroupKFold(5, shuffle=True, random_state=42)
# --------------------------------------------------------------------------- #
def build_folds(clean_texts, y, verbose: bool = True) -> np.ndarray:
    keys = pd.Index([group_key(t) for t in clean_texts], dtype="object")
    groups = pd.factorize(keys, sort=False)[0]
    sizes = pd.Series(groups).value_counts()
    n_groups, n_dup = int(len(sizes)), int((sizes > 1).sum())
    rows_in_dup = int(sizes[sizes > 1].sum())

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_ids = np.full(len(y), -1, dtype=int)
    for k, (_tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold_ids[va] = k
    assert (fold_ids >= 0).all(), "some rows received no fold id"

    crossing = pd.DataFrame({"g": groups, "f": fold_ids}).groupby("g")["f"].nunique()
    n_crossing = int((crossing > 1).sum())
    assert n_crossing == 0, f"{n_crossing} text groups span >1 fold -> leakage"
    if verbose:
        print(f"groups: {n_groups} | duplicate groups: {n_dup} | "
              f"rows in duplicate groups: {rows_in_dup} | groups spanning >1 fold: 0")
    return fold_ids


def fold_split(fold_ids: np.ndarray, k: int):
    va = np.flatnonzero(fold_ids == k)
    tr = np.flatnonzero(fold_ids != k)
    return tr, va


# --------------------------------------------------------------------------- #
# ID-locality prior -- VERBATIM copy of the baseline notebook function
# --------------------------------------------------------------------------- #
def id_prior(query_ids, train_ids, train_labels, k=K_NEIGHBORS, eps=ID_EPS,
             n_classes=3, tag="prior", assert_disjoint=True):
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
    cand = pos[:, None] + np.arange(-k, k + 1)[None, :]      # k left + k right
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
    if assert_disjoint:
        assert overlap == 0, "leak: query ids present in the prior pool"
    return P


def id_prior_diagnostics(query_ids, train_ids, train_labels, k=K_NEIGHBORS,
                          eps=ID_EPS, tag="prior") -> dict:
    q = np.asarray(query_ids, dtype=np.int64).ravel()
    t = np.asarray(train_ids, dtype=np.int64).ravel()
    yy = np.asarray(train_labels, dtype=np.int64).ravel()
    order = np.argsort(t, kind="stable")
    s_ids, s_y = t[order], yy[order]
    kk = int(min(k, len(s_ids)))
    pos = np.searchsorted(s_ids, q, side="left")
    cand = pos[:, None] + np.arange(-k, k + 1)[None, :]
    valid = (cand >= 0) & (cand < len(s_ids))
    cc = np.clip(cand, 0, len(s_ids) - 1)
    d = np.abs(s_ids[cc] - q[:, None]).astype(np.float64)
    d = np.where(valid & (s_ids[cc] != q[:, None]), d, np.inf)
    sel = np.argsort(d, axis=1, kind="stable")[:, :kk]
    ds = np.take_along_axis(d, sel, axis=1)
    used = np.isfinite(ds)
    return {
        "tag": tag, "n_queries": int(len(q)), "n_pool": int(len(s_ids)), "k": int(k),
        "weights": f"1/(distance+{eps})",
        "pool_ids_disjoint_from_queries": bool(np.isin(q, s_ids).sum() == 0),
        "self_matches_excluded": int(used.sum()),
        "mean_neighbours_used": round(float(used.sum(axis=1).mean()), 4),
        "queries_with_fewer_than_k": int((used.sum(axis=1) < kk).sum()),
        "queries_with_no_neighbour": int((used.sum(axis=1) == 0).sum()),
        "mean_kth_distance": (round(float(np.sort(np.where(used, ds, np.inf), axis=1)
                                    [:, -1].mean()), 4)
                             if len(q) else None),
    }


# --------------------------------------------------------------------------- #
# INDEPENDENT re-implementation used only to DIFF the prior above.
# Deliberately a different algorithm: full distance matrix + full argsort, no
# searchsorted window, no self-exclusion-by-index -- processed in blocks.
# (Same approach the Part-1 re-validation used.)
# --------------------------------------------------------------------------- #
def independent_knn_id_prior(query_ids, train_ids, train_labels, k=K_NEIGHBORS,
                             eps=ID_EPS, n_classes=3, block=300) -> np.ndarray:
    q = np.asarray(query_ids, dtype=np.int64).ravel()
    t = np.asarray(train_ids, dtype=np.int64).ravel()
    y = np.asarray(train_labels, dtype=np.int64).ravel()
    order = np.argsort(t, kind="stable")
    s_ids, s_y = t[order], y[order]
    kk = int(min(k, len(s_ids)))
    out = np.empty((len(q), n_classes), dtype=float)
    inf = np.iinfo(np.int64).max
    for start in range(0, len(q), block):
        chunk = q[start:start + block]
        d = np.abs(chunk[:, None] - s_ids[None, :])
        d = np.where(chunk[:, None] == s_ids[None, :], inf, d)   # never pick self
        idx = np.argsort(d, axis=1, kind="stable")[:, :kk]       # naive full sort
        ds = np.take_along_axis(d, idx, axis=1).astype(np.float64)
        ys = s_y[idx]
        good = ds < inf
        w = np.where(good, 1.0 / (np.where(good, ds, 0.0) + eps), 0.0)
        blk = np.stack([(w * (ys == c)).sum(axis=1) for c in range(n_classes)], axis=1)
        tot = blk.sum(axis=1, keepdims=True)
        blk = blk / np.where(tot > 0, tot, 1.0)
        blk[tot.ravel() <= 0] = 1.0 / n_classes
        out[start:start + len(chunk)] = blk
    return out


def prove_id_prior_leak_free(clean_texts, y, ids, fold_ids, tol=1e-10,
                             verbose=True):
    """Per fold: assert val ids disjoint from the prior pool, and diff the
    production prior against the independent re-implementation."""
    rows = []
    for k in range(N_SPLITS):
        tr, va = fold_split(fold_ids, k)
        overlap = int(np.intersect1d(ids[va], ids[tr]).size)
        assert overlap == 0, f"fold{k}: val ids present in prior pool"
        p_prod = id_prior(ids[va], ids[tr], y[tr], tag=f"fold{k} prior")
        p_ind = independent_knn_id_prior(ids[va], ids[tr], y[tr], k=K_NEIGHBORS)
        max_diff = float(np.abs(p_prod - p_ind).max())
        assert max_diff <= tol, f"fold{k}: prior differs from independent impl {max_diff}"
        diag = id_prior_diagnostics(ids[va], ids[tr], y[tr], tag=f"fold{k}_val")
        diag.update({"fold": k, "train_val_id_overlap": overlap,
                     "max_abs_diff_vs_independent": max_diff})
        rows.append(diag)
        if verbose:
            print(f"fold{k}: val&pool={overlap} "
                  f"max|prod-ind|={max_diff:.2e} "
                  f"mean_neighbours={diag['mean_neighbours_used']}")
    return rows


# --------------------------------------------------------------------------- #
# TF-IDF: per-fold vectorisers, cached to disk so experiments share one pass
# --------------------------------------------------------------------------- #
def _cache_path(k: int, part: str, which: str) -> Path:
    return CACHE_DIR / f"fold{k}_{part}_{which}.npz"


def fold_tfidf(clean_texts, fold_ids, k, verbose=True, use_cache=True):
    """Fit word+char TF-IDF on fold k's TRAINING rows only and transform both
    splits. Returns (Xtr, Xva) as a single CSR float32 matrix (word|char)."""
    if use_cache:
        ctr, cva = _cache_path(k, "tfidf", "tr"), _cache_path(k, "tfidf", "va")
        if ctr.exists() and cva.exists():
            Xtr = sp.load_npz(ctr)
            Xva = sp.load_npz(cva)
            if verbose:
                print(f"fold{k}: tfidf from cache Xtr={Xtr.shape} Xva={Xva.shape}")
            return Xtr, Xva

    tr, va = fold_split(fold_ids, k)
    t0 = time.time()
    word = TfidfVectorizer(**WORD_KW)
    word.fit(clean_texts[tr])
    char = TfidfVectorizer(**CHAR_KW)
    char.fit(clean_texts[tr])
    Xtr = sp.hstack([word.transform(clean_texts[tr]),
                     char.transform(clean_texts[tr])], format="csr").astype(np.float32)
    Xva = sp.hstack([word.transform(clean_texts[va]),
                     char.transform(clean_texts[va])], format="csr").astype(np.float32)
    del word, char
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        sp.save_npz(_cache_path(k, "tfidf", "tr"), Xtr, compressed=False)
        sp.save_npz(_cache_path(k, "tfidf", "va"), Xva, compressed=False)
    if verbose:
        print(f"fold{k}: tfidf fit {time.time() - t0:.1f}s "
              f"Xtr={Xtr.shape} nnz={Xtr.nnz:,} Xva={Xva.shape} nnz={Xva.nnz:,}")
    return Xtr, Xva


def fold_tfidf_word_only(clean_texts, fold_ids, k, verbose=True, use_cache=True):
    """Word TF-IDF only (needed by ComplementNB in experiment 1)."""
    if use_cache:
        ctr, cva = _cache_path(k, "word", "tr"), _cache_path(k, "word", "va")
        if ctr.exists() and cva.exists():
            return sp.load_npz(ctr), sp.load_npz(cva)
    tr, va = fold_split(fold_ids, k)
    t0 = time.time()
    word = TfidfVectorizer(**WORD_KW)
    word.fit(clean_texts[tr])
    Xtr = word.transform(clean_texts[tr]).astype(np.float32)
    Xva = word.transform(clean_texts[va]).astype(np.float32)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        sp.save_npz(_cache_path(k, "word", "tr"), Xtr, compressed=False)
        sp.save_npz(_cache_path(k, "word", "va"), Xva, compressed=False)
    if verbose:
        print(f"fold{k}: word-only tfidf fit {time.time() - t0:.1f}s "
              f"Xtr={Xtr.shape} Xva={Xva.shape}")
    return Xtr, Xva


def fit_vectorizers(texts):
    """Fit the word + char TF-IDF pair on an arbitrary row subset."""
    word = TfidfVectorizer(**WORD_KW)
    word.fit(texts)
    char = TfidfVectorizer(**CHAR_KW)
    char.fit(texts)
    return word, char


def matrix_from(word, char, texts):
    return sp.hstack([word.transform(texts), char.transform(texts)],
                     format="csr").astype(np.float32)


def fit_text_lr(X, y, **kw):
    """The baseline text classifier: LogisticRegression(C=4.0, balanced)."""
    from sklearn.linear_model import LogisticRegression
    params = dict(LR_KW)
    params.update(kw)
    return LogisticRegression(**params).fit(X, y)


def full_tfidf(clean_texts, test_clean, verbose=True, use_cache=True):
    """Word+char TF-IDF fitted on the FULL training set, applied to train+test."""
    if use_cache:
        ctr = _cache_path("full", "tfidf", "tr")
        cte = _cache_path("full", "tfidf", "te")
        if ctr.exists() and cte.exists():
            if verbose:
                print("full: tfidf from cache")
            return sp.load_npz(ctr), sp.load_npz(cte)
    t0 = time.time()
    word = TfidfVectorizer(**WORD_KW)
    word.fit(clean_texts)
    char = TfidfVectorizer(**CHAR_KW)
    char.fit(clean_texts)
    Xtr = sp.hstack([word.transform(clean_texts), char.transform(clean_texts)],
                    format="csr").astype(np.float32)
    Xte = sp.hstack([word.transform(test_clean), char.transform(test_clean)],
                    format="csr").astype(np.float32)
    n_w, n_c = len(word.vocabulary_), len(char.vocabulary_)
    del word, char
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        sp.save_npz(ctr, Xtr, compressed=False)
        sp.save_npz(cte, Xte, compressed=False)
    if verbose:
        print(f"full: tfidf fit {time.time() - t0:.1f}s Xtr={Xtr.shape} "
              f"nnz={Xtr.nnz:,} word={n_w:,} char={n_c:,}")
    return Xtr, Xte


# --------------------------------------------------------------------------- #
# Experiment 2: hand-crafted dense features (computed on the RAW text)
# --------------------------------------------------------------------------- #
HAND_FEATURE_NAMES = [
    "bangla_char_ratio", "latin_char_ratio", "digit_ratio",
    "punctuation_density", "repeated_char_run_max", "emoji_count",
    "has_url", "has_mention", "text_length",
]


def _is_emoji(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in EMOJI_RANGES)


def _row_hand_features(s: str) -> list:
    n = len(s)
    if n == 0:
        return [0.0] * len(HAND_FEATURE_NAMES)
    n_bn = n_lat = n_dig = n_pun = n_emo = 0
    for ch in s:
        cp = ord(ch)
        if BN_LO <= cp <= BN_HI:
            n_bn += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            n_lat += 1
        elif "0" <= ch <= "9":
            n_dig += 1
        elif ch in PUNCT:
            n_pun += 1
        if _is_emoji(cp):
            n_emo += 1
    runs = [len(m.group(0)) for m in REPEAT_RE.finditer(s)]
    max_run = int(max(runs)) if runs else 0
    return [
        n_bn / n,
        n_lat / n,
        n_dig / n,
        n_pun / n,
        float(max_run),
        float(n_emo),
        1.0 if URL_RE.search(s) else 0.0,
        1.0 if MENTION_RE.search(s) else 0.0,
        float(n),
    ]


def hand_feature_matrix(raw_texts) -> np.ndarray:
    """Unscaled 9-column hand-crafted block (row-local: no cross-row info)."""
    return np.asarray([_row_hand_features("" if t is None else str(t))
                       for t in raw_texts], dtype=np.float64)


def hand_features(raw_texts, scaler=None, fit: bool = False):
    """9 hand-crafted row features -> float32 dense block (StandardScaler'd).

    `fit=True` fits a fresh StandardScaler on these rows (used on the training
    split of a fold only, so the scaler never sees validation rows).
    """
    D = hand_feature_matrix(raw_texts)
    if fit or scaler is None:
        scaler = StandardScaler().fit(D)
    Z = scaler.transform(D).astype(np.float32)
    return Z, scaler


# --------------------------------------------------------------------------- #
# Metrics / blending
# --------------------------------------------------------------------------- #
def macro_f1(y_true, proba) -> float:
    return float(f1_score(y_true, np.asarray(proba).argmax(axis=1),
                          average="macro", labels=list(CLASSES), zero_division=0))


def per_class_f1(y_true, pred) -> dict:
    pred = np.asarray(pred)
    vals = f1_score(np.asarray(y_true), pred, average=None,
                    labels=list(CLASSES), zero_division=0)
    return {f"f1_class{int(c)}": round(float(v), 6) for c, v in zip(CLASSES, vals)}


def accuracy(y_true, proba) -> float:
    return float((np.asarray(proba).argmax(axis=1) == np.asarray(y_true)).mean())


def blend(p_text, p_id, alpha):
    p = alpha * np.asarray(p_text, dtype=float) + (1.0 - alpha) * np.asarray(p_id, dtype=float)
    return p / p.sum(axis=1, keepdims=True)


def blend_per_class(p_text, p_id, w):
    """Per-class convex blend: col c uses w[c] for text and 1 - w[c] for id."""
    p_text = np.asarray(p_text, dtype=float)
    p_id = np.asarray(p_id, dtype=float)
    w = np.asarray(w, dtype=float).reshape(1, -1)
    p = w * p_text + (1.0 - w) * p_id
    return p / p.sum(axis=1, keepdims=True)


def alpha_grid(y, p_text, p_id):
    return [(a, macro_f1(y, blend(p_text, p_id, a))) for a in ALPHA_GRID]


# --------------------------------------------------------------------------- #
# Duplicate-text override (verbatim baseline behaviour, for parity checks)
# --------------------------------------------------------------------------- #
def unanimous_lookup(texts, labels):
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


# --------------------------------------------------------------------------- #
# Result IO
# --------------------------------------------------------------------------- #
COMPARISON_CSV = REPORTS_DIR / "v2_comparison_table.csv"
CONFIGS_DIR = REPORTS_DIR / "v2_configs"
COMPARISON_COLUMNS = [
    "experiment", "description", "oof_macro_f1", "delta_vs_baseline",
    "per_class_f1", "per_fold_std", "config_hyperparams",
]

# The live, best-known-good submission. READ-ONLY: never touched, never written.
PROTECTED = {
    "uap_submission/reproducible/submission_reproducible_file_1.ipynb":
        "fa0e055fc226b20f6e0c366d9909ce4e813e33bb",
    "uap_submission/reproducible/submission/submission_1.csv":
        "cd623b33101873a3e22a19e7ad44197d791d1b4f",
    "train.csv": "66c6b109c8fd187f4c2c474a09159905d25978cd",
    "test.csv": "9caf35ded5a6fe4b2765cdc3340f192d0994d072",
    "sample_submission.csv": "73345c6143ad4e155bac35de1bf8d5d81ed1d039",
}


def check_protected(verbose: bool = True) -> dict:
    """Assert the locked baseline notebook/submission and the three input CSVs
    are byte-identical to their recorded sha1 (they are never written by any
    experiment_v2_* file)."""
    out = {}
    for rel, want in PROTECTED.items():
        p = ROOT / rel
        got = sha1_file(p) if p.exists() else "MISSING"
        out[rel] = {"expected_sha1": want, "actual_sha1": got, "match": got == want}
    bad = [k for k, v in out.items() if not v["match"]]
    if verbose:
        for k, v in out.items():
            print(f"protected {k}: {'unchanged' if v['match'] else 'CHANGED!'}")
    assert not bad, f"protected files modified: {bad}"
    return out


def save_result(name: str, payload: dict):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"v2_{name}_metrics.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    oof = payload.pop("_oof_arrays", None)
    if oof:
        np.savez_compressed(PRED_DIR / f"v2_{name}_oof.npz", **oof)
    return payload


def save_config(name: str, config: dict):
    """Persist the fitted config so a winner can be re-materialised later
    without repeating the search."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIGS_DIR / f"{name}.json"
    path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    return path


def record_comparison(experiment: str, description: str, oof_macro_f1: float,
                      per_class_f1, per_fold_std, config: dict,
                      note: str = "") -> dict:
    """Upsert one row into the single running reports/v2_comparison_table.csv and
    drop the config JSON next to it. Keyed on `experiment`, so re-running is
    idempotent and no history is lost."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    full = dict(config)
    if note:
        full["evaluation_note"] = note
    row = {
        "experiment": experiment,
        "description": description,
        "oof_macro_f1": round(float(oof_macro_f1), 6),
        "delta_vs_baseline": round(float(oof_macro_f1) - BASELINE_OOF_F1, 6),
        "per_class_f1": json.dumps(per_class_f1, sort_keys=True),
        "per_fold_std": round(float(per_fold_std), 6),
        "config_hyperparams": json.dumps(config, sort_keys=True, default=str),
    }
    rows = []
    if COMPARISON_CSV.exists():
        rows = pd.read_csv(COMPARISON_CSV, dtype=str).to_dict("records")
    rows = [r for r in rows if r.get("experiment") != experiment]
    rows.append(row)
    tmp = COMPARISON_CSV.with_suffix(".csv.tmp")
    pd.DataFrame(rows, columns=COMPARISON_COLUMNS).to_csv(tmp, index=False)
    tmp.replace(COMPARISON_CSV)
    cfg_path = save_config(experiment, config)
    print(f"[comparison] {experiment}: OOF {oof_macro_f1:.6f} "
          f"(delta {row['delta_vs_baseline']:+.6f}) -> {COMPARISON_CSV.name}, "
          f"config -> {cfg_path.relative_to(ROOT).as_posix()}")
    return row


def read_comparison() -> pd.DataFrame:
    if not COMPARISON_CSV.exists():
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    return pd.read_csv(COMPARISON_CSV, dtype=str)


def banner(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

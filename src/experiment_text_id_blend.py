"""
src/experiment_text_id_blend.py -- Merged "text + ID-locality-prior" classifier
for the HerWILL X UAP Datathon 2026.

WHY THIS MODEL
--------------
`train.csv` (47,817 rows) and `test.csv` (11,955 rows) ids together cover the
contiguous range 0..59771 with zero overlap. Neighbouring ids are transductively
correlated with the label (same scrape/thread order), so a k-nearest-id prior
over *training ids only* is a genuine dataset-specific signal. It is combined
with a text TF-IDF + LogisticRegression model:

    p_final = alpha * p_text + (1 - alpha) * p_id_prior   (alpha searched on OOF)

This signal is dataset-specific and must not be reused as a generic feature
outside this competition.

RIGOUR (same conventions as src/experiment_small_tfidf.py)
---------------------------------------------------------
* Reads ONLY train.csv / test.csv / sample_submission.csv, opened read-only.
* No external data, no downloads, no pretrained weights, no network access.
* Atomic writes (temp file + os.replace) for every artifact.
* Per-fold artifacts saved immediately -> interrupted runs resume.
* Invariant logging for every fold, plus hard STOP conditions.
* No GPU assumed; parallelism kept conservative.

STOP CONDITIONS (raise, log, do NOT proceed) -- see ``check_fold_expectations``:
  1. any fold's macro F1 outside the externally-verified range by more than the
     allowed tolerance (symptoms of leakage / indexing bugs),
  2. any chance that the id-prior read a validation-fold label,
  3. the final submission failing any Step-5 validation check (validated BEFORE
     the file is written).

Run:  python -u src/experiment_text_id_blend.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
SAMPLE_CSV = ROOT / "sample_submission.csv"

DATA_CHECKS_DIR = ROOT / "data_checks"
MODELS_DIR = ROOT / "models" / "text_id_blend"
OOF_DIR = ROOT / "predictions" / "oof"
TEST_PRED_DIR = ROOT / "predictions" / "test"
REPORTS_DIR = ROOT / "reports"
SUBMISSIONS_DIR = ROOT / "submissions"
LOGS_DIR = ROOT / "logs"

FOLD_ASSIGN_CSV = DATA_CHECKS_DIR / "text_id_blend_fold_assignments.csv"
OOF_BLEND_OUT = OOF_DIR / "oof_text_id_blend.csv"
OOF_TEXT_OUT = OOF_DIR / "oof_text_only_lbfgs.csv"
OOF_ID_OUT = OOF_DIR / "oof_id_prior_only.csv"
TEST_BLEND_OUT = TEST_PRED_DIR / "test_text_id_blend.csv"
TEST_TEXT_OUT = TEST_PRED_DIR / "test_text_only_lbfgs.csv"
TEST_ID_OUT = TEST_PRED_DIR / "test_id_prior_only.csv"
SUBMISSION_OUT = SUBMISSIONS_DIR / "submission_text_id_blend.csv"
FOLD_SCORES_OUT = REPORTS_DIR / "text_id_blend_fold_scores.csv"
METRICS_OUT = REPORTS_DIR / "text_id_blend_metrics.json"
SUMMARY_OUT = REPORTS_DIR / "text_id_blend_summary.md"
LOG_PATH = LOGS_DIR / "experiment_text_id_blend.log"

MODEL_NAME = "blend_text_id"
CLASSES = (0, 1, 2)
CLASS_LABELS = {0: "explicitly_toxic", 1: "subtly_toxic", 2: "non_toxic"}
PROB_COLS = ("prob_0", "prob_1", "prob_2")
OOF_COLUMNS = ["id", "y_true", "prob_0", "prob_1", "prob_2", "y_pred", "fold", "model_name"]
TEST_PROB_COLUMNS = ["id", "prob_0", "prob_1", "prob_2", "y_pred", "model_name"]
SUBMISSION_COLUMNS = ["id", "y_pred"]

N_SPLITS = 5
SEED = 42
K_NEIGHBORS = 20
ID_EPS = 1e-6
ALPHA_GRID = [round(0.05 * i, 2) for i in range(21)]      # 0.00, 0.05 ... 1.00
ALPHA_REFERENCE = 0.40                                    # alpha * text + (1-alpha) * id
PROB_SUM_TOL = 1e-6
BIAS_WARN_PP = 10.0                                       # class-share drift warning

WORD_KW = dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60_000,
               sublinear_tf=True, dtype=np.float32)
CHAR_KW = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60_000,
               sublinear_tf=True, dtype=np.float32)
LR_KW = dict(solver="lbfgs", max_iter=2000, class_weight="balanced", C=4.0)

# Externally verified reference values (+- tolerance) -> STOP condition #1.
EXPECTED = {
    "text_only": {"center": 0.52, "tol": 0.03},
    "id_prior": {"center": 0.615, "tol": 0.035},
    "blend_0.40": {"center": 0.64, "tol": 0.03},
}

RESOURCES: dict = {}


class StopCondition(RuntimeError):
    """Raised when a hard stop condition of this experiment triggers."""


# --------------------------------------------------------------------------- #
# Logging / resources / atomic writers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def fmt_secs(seconds: float) -> str:
    return f"{seconds:.1f}s ({seconds / 60:.2f} min)"


def detect_resources() -> dict:
    cpu = os.cpu_count() or 4
    total_gb = free_gb = float("nan")
    try:
        import psutil

        vm = psutil.virtual_memory()
        total_gb, free_gb = vm.total / 1e9, vm.available / 1e9
    except Exception as exc:
        log(f"WARN psutil unavailable ({exc}); RAM unknown -> conservative defaults")
    n_jobs = int(max(1, min(4, cpu // 2)))
    if not np.isnan(free_gb) and free_gb < 3.0:
        n_jobs = min(n_jobs, 2)
    res = {"cpu_count": int(cpu), "ram_total_gb": round(total_gb, 2),
           "ram_free_gb": round(free_gb, 2), "n_jobs": int(n_jobs), "gpu_used": False}
    log(f"resources: cpus={res['cpu_count']} ram_total={res['ram_total_gb']}GB "
        f"ram_free={res['ram_free_gb']}GB n_jobs={res['n_jobs']} gpu_used=False")
    return res


def ram_free_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().available / 1e9, 2)
    except Exception:
        return float("nan")


def report_sparse(name: str, X) -> dict:
    nnz = int(getattr(X, "nnz", 0))
    info = {"name": name, "shape": list(X.shape), "nnz": nnz,
            "est_gb": round(nnz * 8 / 1e9, 3)}
    log(f"matrix[{name}]: shape={X.shape} nnz={nnz:,} dtype={X.dtype} est~{info['est_gb']}GB")
    return info


def atomic_write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write CSV via temp file + os.replace; the three inputs can never be touched."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for protected in (TRAIN_CSV, TEST_CSV, SAMPLE_CSV):
        if path.resolve() == protected.resolve():
            raise PermissionError(f"refusing to overwrite protected input: {protected}")
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_json(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# --------------------------------------------------------------------------- #
# Step 1: text cleaning (URL/mention masking only -- no lowercasing, no stemming,
# Bangla script and emoji are preserved untouched)
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def clean_text(text) -> str:
    """Mask URLs and mentions, collapse whitespace. Nothing else is altered."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text)
    s = _URL_RE.sub(" URL ", s)
    s = _MENTION_RE.sub(" USER ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in (TRAIN_CSV, TEST_CSV, SAMPLE_CSV):
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")
    train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    test = pd.read_csv(TEST_CSV, encoding="utf-8")
    sample = pd.read_csv(SAMPLE_CSV, encoding="utf-8")
    assert list(train.columns) == ["id", "text", "y"], list(train.columns)
    assert list(test.columns) == ["id", "text"], list(test.columns)
    assert list(sample.columns) == ["id", "y_pred"], list(sample.columns)

    train["text"] = train["text"].astype("string").fillna("").astype(str)
    test["text"] = test["text"].astype("string").fillna("").astype(str)
    train["clean"] = train["text"].map(clean_text)
    test["clean"] = test["text"].map(clean_text)

    # id-range invariants claimed in the task background, re-verified here.
    ids = np.concatenate([train["id"].to_numpy(), test["id"].to_numpy()])
    all_ids = np.unique(ids)
    contiguous = bool(all_ids.min() == 0 and all_ids.max() == len(all_ids) - 1)
    log(f"loaded train={train.shape} test={test.shape} sample={sample.shape}")
    log(f"id structure: overlap={len(set(train['id']) & set(test['id']))} "
        f"union={len(all_ids)} span={all_ids.max() - all_ids.min() + 1} "
        f"contiguous_0..N={contiguous} dup_ids={len(ids) - len(all_ids)}")
    log(f"clean_text sanity: train chars raw={int(train['text'].str.len().sum()):,} "
        f"clean={int(train['clean'].str.len().sum()):,}; "
        f"URL tags={int(train['clean'].str.count(' URL ').sum())} "
        f"USER tags={int(train['clean'].str.count(' USER ').sum())}")
    return train, test, sample


# --------------------------------------------------------------------------- #
# Step 2: ID-locality prior (TRAIN labels only -- structurally leak-proof)
# --------------------------------------------------------------------------- #
def id_prior_probs(query_ids, train_ids, train_labels, k: int = K_NEIGHBORS,
                   eps: float = ID_EPS, n_classes: int = len(CLASSES),
                   tag: str = "id_prior") -> tuple[np.ndarray, dict]:
    """k-nearest-id label prior. Vectorised with np.searchsorted (no Python loop).

    LEAK-PROOFING BY SIGNATURE: this function only receives *training* ids and
    *training* labels as the neighbour source, so a validation/test row's own
    label can never enter the prior. Any query id that also appears as a training
    id (impossible for this dataset: train/test ids are disjoint and unique) has
    its exact self-match excluded, which is logged loudly.

    Weighting: w = 1 / (distance + eps). Neighbours are summed per class and
    normalised into a 3-class probability vector. If fewer than k neighbours are
    available, whatever is available is used; a query with no usable neighbour
    falls back to the uniform 1/3 prior (logged).
    """
    q = np.asarray(query_ids, dtype=np.int64).ravel()
    t = np.asarray(train_ids, dtype=np.int64).ravel()
    y = np.asarray(train_labels, dtype=np.int64).ravel()
    if len(t) != len(y):
        raise ValueError("train_ids and train_labels length mismatch")
    if len(t) == 0:
        raise StopCondition("id_prior_probs received an empty neighbour pool")
    if t.size != np.unique(t).size:
        raise StopCondition("neighbour pool contains duplicate training ids")
    if not np.isin(y, list(CLASSES)).all():
        raise StopCondition("neighbour labels contain values outside {0,1,2}")

    order = np.argsort(t, kind="mergesort")
    s_ids, s_y = t[order], y[order]
    n = len(s_ids)

    self_match = np.isin(q, s_ids)
    if self_match.any():
        log(f"WARN id_prior[{tag}]: {int(self_match.sum())} query ids also occur in the "
            f"neighbour pool; their exact self-match is excluded from the neighbourhood")

    pos = np.searchsorted(s_ids, q, side="left")
    offsets = np.arange(-k, k + 1, dtype=np.int64)          # 2k+1 candidates per query
    cand = pos[:, None] + offsets[None, :]
    valid = (cand >= 0) & (cand < n)
    cand_c = np.clip(cand, 0, n - 1)
    cand_ids = s_ids[cand_c]
    dist = np.abs(cand_ids - q[:, None]).astype(np.float64)
    # +inf marks unusable candidates so the k-smallest selection skips them
    dist = np.where(valid & (cand_ids != q[:, None]), dist, np.inf)

    kk = int(min(k, n))
    # Deterministic k-smallest selection: a stable sort keeps the lowest candidate
    # index first when distances tie (very common with integer ids), so a tie can
    # never resolve differently between runs or between implementations.
    order = np.argsort(dist, axis=1, kind="stable")[:, :kk]
    d_sel = np.take_along_axis(dist, order, axis=1)
    y_sel = s_y[cand_c[np.arange(len(q))[:, None], order]]

    finite = np.isfinite(d_sel)
    w = np.where(finite, 1.0 / (np.where(finite, d_sel, 0.0) + eps), 0.0)
    P = np.zeros((len(q), n_classes), dtype=np.float64)
    for c in range(n_classes):
        P[:, c] = (w * (y_sel == c)).sum(axis=1)

    tot = P.sum(axis=1, keepdims=True)
    n_empty = int((tot.ravel() <= 0).sum())
    P = P / np.where(tot > 0, tot, 1.0)
    if n_empty:
        P[tot.ravel() <= 0] = 1.0 / n_classes
        log(f"WARN id_prior[{tag}]: {n_empty} queries had no usable neighbour -> uniform prior")

    eff = finite.sum(axis=1)
    # mean distance to the k-th (farthest selected) neighbour, NaN-safe
    kth = np.where(finite, d_sel, -np.inf).max(axis=1)
    kth = np.where(np.isfinite(kth), kth, np.nan)
    mean_kth = float(np.nanmean(kth)) if np.isfinite(kth).any() else float("nan")
    diag = {
        "tag": tag, "n_queries": int(len(q)), "n_pool": int(n), "k": int(k),
        "weights": "1/(distance+1e-6)", "self_matches_excluded": int(self_match.sum()),
        "queries_with_fewer_than_k": int((eff < kk).sum()),
        "queries_with_no_neighbour": n_empty,
        "mean_neighbours_used": round(float(eff.mean()), 3),
        "mean_kth_distance": round(mean_kth, 3),
        "pool_ids_disjoint_from_queries": bool(not self_match.any()),
        "labels_are_train_only": True,
    }
    log(f"id_prior[{tag}]: queries={diag['n_queries']} pool={diag['n_pool']} k={k} "
        f"mean_neighbours={diag['mean_neighbours_used']} "
        f"mean_kth_dist={diag['mean_kth_distance']} "
        f"pool_disjoint_from_queries={diag['pool_ids_disjoint_from_queries']}")
    return P, diag


def blend_probs(p_text, p_id, alpha: float) -> np.ndarray:
    """alpha * p_text + (1 - alpha) * p_id, renormalised."""
    P = alpha * np.asarray(p_text, dtype=float) + (1.0 - alpha) * np.asarray(p_id, dtype=float)
    return P / P.sum(axis=1, keepdims=True)


def metric_block(y_true, proba) -> dict:
    pred = np.asarray(proba).argmax(axis=1)
    per_class = f1_score(y_true, pred, average=None, labels=list(CLASSES), zero_division=0)
    out = {"macro_f1": float(f1_score(y_true, pred, average="macro",
                                      labels=list(CLASSES), zero_division=0)),
           "accuracy": float(accuracy_score(y_true, pred))}
    for c, v in zip(CLASSES, per_class):
        out[f"f1_class{c}"] = float(v)
    return out


# --------------------------------------------------------------------------- #
# Step 3a: StratifiedKFold(5) folds + fold-assignment artifact
# --------------------------------------------------------------------------- #
def build_folds(train: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """StratifiedKFold(5, shuffle=True, random_state=42) on y, fully verified."""
    y = train["y"].to_numpy(dtype=int)
    ids = train["id"].to_numpy(dtype=np.int64)
    clean = train["clean"].to_numpy()
    fold_ids = np.full(len(train), -1, dtype=int)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for k, (_tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_ids[va] = k
    if (fold_ids < 0).any():
        raise StopCondition("some training rows did not receive a fold id")

    rows = []
    for k in range(N_SPLITS):
        va = np.flatnonzero(fold_ids == k)
        tr = np.flatnonzero(fold_ids != k)
        id_overlap = np.intersect1d(ids[tr], ids[va])
        if id_overlap.size:                     # STOP condition #2 precondition
            raise StopCondition(f"fold {k}: {id_overlap.size} ids appear in both "
                                "train and validation sides")
        tr_texts = set(clean[tr])
        dup_leak = int(sum(1 for t in clean[va] if t in tr_texts))
        row = {"fold": k, "n_train": int(len(tr)), "n_val": int(len(va)),
               "train_val_id_overlap": int(id_overlap.size),
               "val_rows_with_text_also_in_train": dup_leak,
               "val_dup_text_share_pct": round(100.0 * dup_leak / max(1, len(va)), 3)}
        for c in CLASSES:
            row[f"train_c{c}"] = int((y[tr] == c).sum())
            row[f"val_c{c}"] = int((y[va] == c).sum())
        rows.append(row)
        log(f"fold[{k}]: n_train={row['n_train']} n_val={row['n_val']} "
            f"val_counts=({row['val_c0']},{row['val_c1']},{row['val_c2']}) "
            f"id_overlap={row['train_val_id_overlap']} "
            f"val_rows_with_dup_text_in_train={dup_leak} "
            f"({row['val_dup_text_share_pct']}%)")

    fold_info = {
        "strategy": "StratifiedKFold", "n_splits": N_SPLITS, "shuffle": True,
        "random_state": SEED, "metric": "macro_f1", "fold_rows": rows,
        "note": ("Plain stratified folds are used as specified. Duplicate-text rows "
                 "can therefore straddle folds; the per-fold duplicate-share column "
                 "quantifies that optimism and is reported, not hidden."),
    }
    atomic_write_csv(pd.DataFrame({"id": ids, "y": y, "fold": fold_ids}), FOLD_ASSIGN_CSV)
    log(f"wrote {FOLD_ASSIGN_CSV.relative_to(ROOT)} rows={len(train)}")
    return fold_ids, fold_info


# --------------------------------------------------------------------------- #
# Step 4a: unanimous duplicate-text lookup (conflicting groups are skipped)
# --------------------------------------------------------------------------- #
def unanimous_lookup(texts, labels) -> tuple[dict, dict]:
    """Map cleaned text -> label for groups where every occurrence agrees."""
    df = pd.DataFrame({"t": np.asarray(texts), "y": np.asarray(labels)})
    g = df.groupby("t")["y"].agg(["nunique", "size", "first"])
    uni = g[g["nunique"] == 1]
    conflicting = g[g["nunique"] > 1]
    lookup = {str(t): int(v) for t, v in uni["first"].items()}
    stats = {
        "n_distinct_texts": int(len(g)),
        "n_unanimous_groups": int(len(uni)),
        "n_conflicting_groups_skipped": int(len(conflicting)),
        "rows_in_unanimous_groups": int(uni["size"].sum()),
        "rows_in_conflicting_groups": int(conflicting["size"].sum()),
    }
    log(f"override lookup: distinct_texts={stats['n_distinct_texts']} "
        f"unanimous_groups={stats['n_unanimous_groups']} "
        f"conflicting_groups_skipped={stats['n_conflicting_groups_skipped']} "
        f"(rows in conflicting groups={stats['rows_in_conflicting_groups']})")
    return lookup, stats


def apply_override(texts, pred, lookup) -> tuple[np.ndarray, int]:
    """Replace predictions whose cleaned text matches a unanimous training text."""
    out = np.asarray(pred).copy()
    hits = 0
    for i, t in enumerate(texts):
        v = lookup.get(str(t))
        if v is not None:
            out[i] = v
            hits += 1
    return out, hits


# --------------------------------------------------------------------------- #
# Step 1b: text model fitting (per fold + full refit) with resume
# --------------------------------------------------------------------------- #
def fold_model_dir(k: int) -> Path:
    return MODELS_DIR / f"fold{k}"


def fold_val_prob_path(k: int) -> Path:
    return OOF_DIR / f"_textid_fold{k}_val_text_probs.csv"


def build_text_matrix(word, char, texts):
    """hstack of word + char_wb TF-IDF blocks (CSR, float32)."""
    return sp.hstack([word.transform(texts), char.transform(texts)],
                     format="csr").astype(np.float32)


def fit_text_model(texts_tr, y_tr, tag: str):
    """Fit both vectorizers + multinomial LogisticRegression on the given texts."""
    t0 = time.time()
    word = TfidfVectorizer(**WORD_KW)
    word.fit(texts_tr)
    char = TfidfVectorizer(**CHAR_KW)
    char.fit(texts_tr)
    X = build_text_matrix(word, char, texts_tr)
    report_sparse(f"{tag}_train_matrix", X)
    clf = LogisticRegression(random_state=SEED, **LR_KW)
    clf.fit(X, y_tr)
    meta = {"n_features_word": int(len(word.vocabulary_)),
            "n_features_char": int(len(char.vocabulary_)),
            "matrix_shape": list(X.shape), "matrix_nnz": int(X.nnz),
            "fit_seconds": round(time.time() - t0, 2),
            "lr_n_iter": [int(v) for v in np.atleast_1d(getattr(clf, "n_iter_", []))]}
    log(f"{tag}: text model fitted in {fmt_secs(meta['fit_seconds'])} "
        f"(word={meta['n_features_word']:,} char={meta['n_features_char']:,} "
        f"nnz={meta['matrix_nnz']:,} lbfgs_iter={meta['lr_n_iter']})")
    return word, char, clf, meta


def _load_prob_csv(path: Path, expected_ids):
    """Load and validate a saved probability CSV against expected ids."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8")
        if len(df) != len(expected_ids) or not np.array_equal(df["id"].to_numpy(), expected_ids):
            return None
        P = df[list(PROB_COLS)].to_numpy(dtype=float)
        if not (np.isfinite(P).all() and (P >= 0).all() and (P <= 1).all()):
            return None
        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
            return None
        return df
    except Exception as exc:
        log(f"WARN could not reuse {path.name} ({exc})")
        return None


def fold_text_probs(k: int, texts_tr, y_tr, texts_va, val_ids, resume: bool = True):
    """Per-fold text probabilities, with immediate artifact saves (resume-safe)."""
    d = fold_model_dir(k)
    d.mkdir(parents=True, exist_ok=True)
    wpath, cpath, lpath = (d / "word_vectorizer.joblib", d / "char_vectorizer.joblib",
                           d / "lr.joblib")
    vpath = fold_val_prob_path(k)
    val_ids = np.asarray(val_ids)
    models_present = lpath.exists() and wpath.exists() and cpath.exists()

    if resume and models_present:
        saved = _load_prob_csv(vpath, val_ids)
        word, char = joblib.load(wpath), joblib.load(cpath)
        if saved is not None:
            log(f"fold[{k}]: RESUME -> reused {vpath.name} + saved fold model (no refit); "
                f"free_ram={ram_free_gb()}GB")
            return saved[list(PROB_COLS)].to_numpy(dtype=float), {
                "resumed": True, "fit_seconds": 0.0, "model_path": str(lpath.relative_to(ROOT)),
                "lr_n_iter": None, "n_features_word": int(len(word.vocabulary_)),
                "n_features_char": int(len(char.vocabulary_)), "matrix_nnz": None}
        log(f"fold[{k}]: saved fold model found but val probabilities missing/stale -> "
            f"recomputing them (no refit)")
        clf = joblib.load(lpath)
        P = np.asarray(clf.predict_proba(build_text_matrix(word, char, texts_va)), dtype=float)
        atomic_write_csv(pd.DataFrame({"id": val_ids,
                                      **{c: P[:, i] for i, c in enumerate(PROB_COLS)}}), vpath)
        return P, {"resumed": True, "fit_seconds": 0.0, "model_path": str(lpath.relative_to(ROOT)),
                   "lr_n_iter": None, "n_features_word": int(len(word.vocabulary_)),
                   "n_features_char": int(len(char.vocabulary_)), "matrix_nnz": None}

    word, char, clf, meta = fit_text_model(texts_tr, y_tr, tag=f"fold{k}")
    joblib.dump(word, wpath)
    joblib.dump(char, cpath)
    joblib.dump(clf, lpath)
    P = np.asarray(clf.predict_proba(build_text_matrix(word, char, texts_va)), dtype=float)
    atomic_write_csv(pd.DataFrame({"id": val_ids,
                                  **{c: P[:, i] for i, c in enumerate(PROB_COLS)}}), vpath)
    log(f"fold[{k}]: saved {vpath.name} + fold model; free_ram={ram_free_gb()}GB")
    return P, {"resumed": False, "fit_seconds": meta["fit_seconds"],
               "model_path": str(lpath.relative_to(ROOT)), "lr_n_iter": meta["lr_n_iter"],
               "n_features_word": meta["n_features_word"],
               "n_features_char": meta["n_features_char"], "matrix_nnz": meta["matrix_nnz"]}


# --------------------------------------------------------------------------- #
# STOP condition #1: per-fold scores must sit in the externally verified ranges
# --------------------------------------------------------------------------- #
EXPECTED_COLUMN = {"text_only": "macro_f1_text", "id_prior": "macro_f1_id",
                   "blend_0.40": "macro_f1_blend_0.40"}


def check_fold_expectations(fold_rows: list[dict]) -> list[dict]:
    """Raise StopCondition when any fold deviates > tolerance from the reference."""
    issues = []
    for row in fold_rows:
        for name, col in EXPECTED_COLUMN.items():
            exp = EXPECTED[name]
            val = float(row[col])
            if abs(val - exp["center"]) > exp["tol"]:
                issues.append({"fold": int(row["fold"]), "quantity": name, "value": round(val, 6),
                               "expected_center": exp["center"], "tolerance": exp["tol"],
                               "deviation": round(val - exp["center"], 6)})
    if issues:
        for it in issues:
            log(f"STOP CONDITION #1 VIOLATED: fold{it['fold']} {it['quantity']}= "
                f"{it['value']:.6f} vs expected {it['expected_center']:.3f} "
                f"+-{it['tolerance']:.3f} (deviation {it['deviation']:+.6f})")
        raise StopCondition(f"{len(issues)} per-fold score(s) outside the externally "
                            f"verified range: {issues}")
    log("fold expectation check PASSED: every fold's text/id/blend macro F1 sits inside "
        "the externally verified range")
    return issues


# --------------------------------------------------------------------------- #
# Alpha grid search on pooled OOF probabilities
# --------------------------------------------------------------------------- #
def alpha_grid_search(y_true, p_text, p_id) -> tuple[float, pd.DataFrame]:
    """Search alpha in ALPHA_GRID on pooled OOF; alpha weights the TEXT model."""
    rows = []
    for a in ALPHA_GRID:
        m = metric_block(y_true, blend_probs(p_text, p_id, a))
        rows.append({"alpha_text_weight": a, "alpha_id_weight": round(1.0 - a, 2),
                     **{k: round(v, 6) for k, v in m.items()}})
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["macro_f1"].idxmax()]
    ref = float(grid.loc[grid["alpha_text_weight"] == ALPHA_REFERENCE, "macro_f1"].iloc[0])
    log(f"alpha grid (pooled OOF): best alpha={best['alpha_text_weight']:.2f} "
        f"-> macro_f1={best['macro_f1']:.6f} | alpha=0.40 -> {ref:.6f} | "
        f"text-only(1.0) -> {float(grid.loc[grid['alpha_text_weight'] == 1.0, 'macro_f1'].iloc[0]):.6f} | "
        f"id-only(0.0) -> {float(grid.loc[grid['alpha_text_weight'] == 0.0, 'macro_f1'].iloc[0]):.6f}")
    return float(best["alpha_text_weight"]), grid


def bias_check(pred_test, y_train) -> dict:
    """Compare predicted class shares with train shares; warn on >10pp drift."""
    shares_train = {int(c): float((y_train == c).mean()) for c in CLASSES}
    shares_pred = {int(c): float((np.asarray(pred_test) == c).mean()) for c in CLASSES}
    drift = {int(c): round(100.0 * (shares_pred[c] - shares_train[c]), 2) for c in CLASSES}
    warns = [f"class {c} share drift {drift[c]:+.2f}pp (train {100 * shares_train[c]:.2f}% "
             f"vs pred {100 * shares_pred[c]:.2f}%)"
             for c in CLASSES if abs(drift[c]) > BIAS_WARN_PP]
    if warns:
        for w in warns:
            log(f"WARN BIAS SANITY CHECK: {w} -> exceeds the {BIAS_WARN_PP:.0f}pp threshold")
    else:
        log(f"bias sanity check PASSED: every class share is within {BIAS_WARN_PP:.0f}pp "
            "of the training distribution")
    return {"train_share_pct": {c: round(100.0 * shares_train[c], 2) for c in CLASSES},
            "pred_share_pct": {c: round(100.0 * shares_pred[c], 2) for c in CLASSES},
            "drift_pp": drift, "warnings": warns, "threshold_pp": BIAS_WARN_PP,
            "passed": not warns}


# --------------------------------------------------------------------------- #
# Invariants (OOF / test probs / submission)
# --------------------------------------------------------------------------- #
def run_invariants(oof: pd.DataFrame, test_probs: pd.DataFrame, sub, train: pd.DataFrame,
                   test: pd.DataFrame, sample: pd.DataFrame, fold_info: dict) -> list[dict]:
    """Every invariant required by the task, logged PASS/FAIL."""
    checks: list[dict] = []

    def add(name: str, ok, detail: str = "") -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
        log(f"CHECK {'PASS' if ok else 'FAIL'}: {name} :: {detail}")

    def prob_checks(df: pd.DataFrame, tag: str) -> None:
        P = df[list(PROB_COLS)].to_numpy(dtype=float)
        add(f"{tag}_probs_finite", bool(np.isfinite(P).all()),
            f"non_finite={int((~np.isfinite(P)).sum())}")
        add(f"{tag}_probs_in_0_1", bool((P >= 0).all() and (P <= 1).all()),
            f"min={P.min():.6g} max={P.max():.6g}")
        dev = float(np.max(np.abs(P.sum(axis=1) - 1.0)))
        add(f"{tag}_probs_rows_sum_to_1", bool(dev <= PROB_SUM_TOL),
            f"max_abs_dev={dev:.3g} tol={PROB_SUM_TOL:g}")
        add(f"{tag}_pred_in_012", bool(np.isin(df["y_pred"].to_numpy(), list(CLASSES)).all()),
            f"unique={sorted(pd.unique(df['y_pred']))[:6]}")
        add(f"{tag}_pred_equals_argmax",
            bool((P.argmax(axis=1) == df["y_pred"].to_numpy()).all()),
            f"mismatch={int((P.argmax(axis=1) != df['y_pred'].to_numpy()).sum())}")
        add(f"{tag}_no_missing", int(df.isna().to_numpy().sum()) == 0,
            f"n_missing={int(df.isna().to_numpy().sum())}")

    train_ids, y = train["id"].to_numpy(), train["y"].to_numpy()

    # --- OOF ---------------------------------------------------------------
    add("oof_columns_exact", list(oof.columns) == OOF_COLUMNS, f"got={list(oof.columns)}")
    add("oof_rows_equal_train_rows", len(oof) == len(train), f"oof={len(oof)} train={len(train)}")
    add("oof_ids_unique", bool(oof["id"].is_unique), f"nunique={int(oof['id'].nunique())}")
    add("oof_ids_match_train", set(oof["id"]) == set(train_ids), "symmetric set equality")
    add("oof_labels_subset_of_012",
        bool(np.isin(oof["y_true"].to_numpy(), list(CLASSES)).all()),
        f"unique={sorted(pd.unique(oof['y_true']))[:6]}")
    m = oof[["id", "y_true"]].merge(train[["id", "y"]], on="id", how="left")
    add("oof_y_true_matches_train", bool((m["y_true"].to_numpy() == m["y"].to_numpy()).all()),
        f"mismatch={int((m['y_true'] != m['y']).sum())}")
    prob_checks(oof, "oof")
    fit = fold_info["fold_rows"]
    add("fold_sizes_sum_to_train_rows", int(sum(r["n_val"] for r in fit)) == len(train),
        f"sum(n_val)={int(sum(r['n_val'] for r in fit))} fold sizes={[r['n_val'] for r in fit]}")
    add("fold_ids_cover_all_rows",
        bool(oof["fold"].notna().all() and set(pd.unique(oof["fold"])) == set(range(N_SPLITS))),
        f"folds={sorted(pd.unique(oof['fold']))}")
    add("train_val_id_overlap_zero_in_every_fold",
        all(int(r["train_val_id_overlap"]) == 0 for r in fit),
        "; ".join(f"fold{r['fold']}={r['train_val_id_overlap']}" for r in fit))
    add("id_prior_labels_train_only",
        all(bool(d.get("labels_are_train_only")) for d in fold_info["id_prior_diags"]),
        "prior built exclusively from each fold's training ids/labels")
    dist_oof = {int(c): int((oof["y_pred"].to_numpy() == c).sum()) for c in CLASSES}
    add("oof_pred_class_distribution_logged", True, str(dist_oof))

    # --- test probability file ---------------------------------------------
    add("test_prob_columns_exact", list(test_probs.columns) == TEST_PROB_COLUMNS,
        f"got={list(test_probs.columns)}")
    add("test_prob_rows_equal_test_rows", len(test_probs) == len(test),
        f"probs={len(test_probs)} test={len(test)}")
    add("test_prob_id_order_equals_test_csv",
        bool((test_probs["id"].to_numpy() == test["id"].to_numpy()).all()), "row order comparison")
    prob_checks(test_probs, "test_prob")

    # --- submission ---------------------------------------------------------
    if sub is not None:
        add("submission_columns_exact", list(sub.columns) == SUBMISSION_COLUMNS,
            f"got={list(sub.columns)}")
        add("submission_rows_equal_sample", len(sub) == len(sample),
            f"sub={len(sub)} sample={len(sample)}")
        add("submission_id_order_equals_sample",
            bool((sub["id"].to_numpy() == sample["id"].to_numpy()).all()), "row order comparison")
        add("submission_id_set_equals_sample", set(sub["id"]) == set(sample["id"]),
            "symmetric set equality")
        add("submission_ids_unique", bool(sub["id"].is_unique),
            f"nunique={int(sub['id'].nunique())}")
        add("submission_no_nulls", int(sub.isna().to_numpy().sum()) == 0,
            f"n_missing={int(sub.isna().to_numpy().sum())}")
        vals = sub["y_pred"].to_numpy()
        add("submission_values_in_012", bool(np.isin(vals, list(CLASSES)).all()),
            f"unique={sorted(pd.unique(vals))[:6]}")
        add("submission_values_integer", bool(np.all(vals == vals.astype(int))),
            f"dtype={sub['y_pred'].dtype}")
        dist_sub = {int(c): int((vals == c).sum()) for c in CLASSES}
        add("submission_pred_class_distribution_logged", True, str(dist_sub))
    return checks


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    body = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _summary_part1(metrics: dict, fold_rows: pd.DataFrame, grid: pd.DataFrame) -> str:
    fs = fold_rows[["fold", "n_val", "macro_f1_text", "macro_f1_id", "macro_f1_blend_0.40",
                    "accuracy_blend_0.40", "val_rows_with_text_also_in_train"]].copy()
    for c in fs.columns:
        if c not in ("fold", "n_val", "val_rows_with_text_also_in_train"):
            fs[c] = fs[c].map(lambda v: f"{float(v):.6f}")
    g = grid[["alpha_text_weight", "alpha_id_weight", "macro_f1", "accuracy",
              "f1_class0", "f1_class1", "f1_class2"]].copy()
    for c in g.columns:
        if c not in ("alpha_text_weight", "alpha_id_weight"):
            g[c] = g[c].map(lambda v: f"{float(v):.6f}")
    o, ov, best = metrics["oof_overall"], metrics["override"], metrics["chosen_alpha"]
    cm = pd.DataFrame(metrics["oof_confusion_matrix"]["matrix"],
                      index=[f"true_{c}" for c in CLASSES],
                      columns=[f"pred_{c}" for c in CLASSES])
    cm = cm.reset_index().rename(columns={"index": "true/pred"})
    return f"""# Text + ID-locality-prior blend -- local summary

**Model:** `{metrics['model_name']}` -- word(1-2) + char_wb(3-5) TF-IDF (60k each, min_df=2,
sublinear, URL/USER masking only) with `LogisticRegression(solver='lbfgs', C=4.0,
class_weight='balanced', max_iter=2000)`, blended with a k={K_NEIGHBORS} nearest-id label prior
(weight `1/(distance+1e-6)`) built from **training ids only**.

`p = alpha * p_text + (1 - alpha) * p_id`, **alpha = {best:.2f}** (text weight) chosen on pooled
OOF only. Validation: StratifiedKFold(5, shuffle=True, random_state=42).
**Generated:** {metrics['timestamp']}. **GPU used:** False.

## 1. Per-fold Macro F1

{md_table(fs)}

Weakest fold: text-only {metrics['fold_min']['text_only']:.6f}, id-only
{metrics['fold_min']['id_prior']:.6f}, blend@0.40 {metrics['fold_min']['blend_0.40']:.6f}.

## 2. Pooled OOF comparison (47,817 rows)

| Model | Macro F1 | Accuracy | F1 c0 | F1 c1 | F1 c2 |
| --- | --- | --- | --- | --- | --- |
| text-only (lbfgs LR) | {o['text_only']['macro_f1']:.6f} | {o['text_only']['accuracy']:.6f} | {o['text_only']['f1_class0']:.6f} | {o['text_only']['f1_class1']:.6f} | {o['text_only']['f1_class2']:.6f} |
| id-prior only (k={K_NEIGHBORS}) | {o['id_prior_only']['macro_f1']:.6f} | {o['id_prior_only']['accuracy']:.6f} | {o['id_prior_only']['f1_class0']:.6f} | {o['id_prior_only']['f1_class1']:.6f} | {o['id_prior_only']['f1_class2']:.6f} |
| blend @ alpha=0.40 | {o['blend_0.40']['macro_f1']:.6f} | {o['blend_0.40']['accuracy']:.6f} | {o['blend_0.40']['f1_class0']:.6f} | {o['blend_0.40']['f1_class1']:.6f} | {o['blend_0.40']['f1_class2']:.6f} |
| blend @ alpha={best:.2f} (chosen) | {o['blend_best']['macro_f1']:.6f} | {o['blend_best']['accuracy']:.6f} | {o['blend_best']['f1_class0']:.6f} | {o['blend_best']['f1_class1']:.6f} | {o['blend_best']['f1_class2']:.6f} |
| blend @ alpha={best:.2f} + text override (leak-free OOF) | {ov['oof_macro_f1_after_override']:.6f} | - | - | - | - |

Confusion matrix (OOF, blend @ chosen alpha):

{md_table(cm)}

## 3. Alpha grid (pooled OOF; alpha = text weight)

{md_table(g)}

## 4. Comparison with the externally verified targets

| Quantity | External reference | This run | Delta |
| --- | --- | --- | --- |
| text-only macro F1 | 0.52 | {o['text_only']['macro_f1']:.6f} | {o['text_only']['macro_f1'] - 0.52:+.6f} |
| id-prior-only macro F1 | 0.61-0.62 | {o['id_prior_only']['macro_f1']:.6f} | {o['id_prior_only']['macro_f1'] - 0.615:+.6f} |
| blend @ 0.40 macro F1 | 0.64 | {o['blend_0.40']['macro_f1']:.6f} | {o['blend_0.40']['macro_f1'] - 0.64:+.6f} |
| best alpha found | 0.35-0.45 | {best:.2f} | - |
| best blend macro F1 | ~0.64 | {o['blend_best']['macro_f1']:.6f} | {o['blend_best']['macro_f1'] - 0.64:+.6f} |

**Stop condition #1 (per-fold scores inside the verified ranges): PASSED**
"""


def _summary_part2(metrics: dict, checks: list[dict]) -> str:
    ov, b = metrics["override"], metrics["bias_check"]
    ck = pd.DataFrame(checks)
    warn_line = ("WARNINGS: " + "; ".join(b["warnings"])) if b["warnings"] else \
        "No class drifts by more than the threshold."
    return f"""
## 5. Duplicate-text override

* Unanimous cleaned-text groups only; conflicting groups skipped:
  {ov['lookup_stats']['n_unanimous_groups']:,} usable texts,
  {ov['lookup_stats']['n_conflicting_groups_skipped']} conflicting groups skipped
  ({ov['lookup_stats']['rows_in_conflicting_groups']} rows).
* Test rows overridden: **{ov['n_test_overridden']:,}** of {ov['n_test_rows']:,}
  ({ov['test_override_share_pct']:.3f}%).
* Leak-free OOF effect (per-fold lookups built from that fold's training split only):
  {ov['oof_rows_overridden']:,} OOF rows overridden, macro F1
  {ov['oof_macro_f1_before_override']:.6f} -> {ov['oof_macro_f1_after_override']:.6f}
  ({ov['oof_macro_f1_delta']:+.6f}).

## 6. Bias sanity check (threshold {b['threshold_pp']:.0f}pp)

| class | train share % | predicted share % | drift (pp) |
| --- | --- | --- | --- |
| 0 | {b['train_share_pct'][0]} | {b['pred_share_pct'][0]} | {b['drift_pp'][0]:+} |
| 1 | {b['train_share_pct'][1]} | {b['pred_share_pct'][1]} | {b['drift_pp'][1]:+} |
| 2 | {b['train_share_pct'][2]} | {b['pred_share_pct'][2]} | {b['drift_pp'][2]:+} |

{warn_line}

## 7. Invariants

{md_table(ck)}

All invariants passed: **{metrics['invariants_all_passed']}**
(submission validated BEFORE it was written; see scope notes.)

## 8. Outputs

* Submission: `{metrics['outputs']['submission']}` ({metrics['submission_rows']:,} rows, written: {metrics['submission_written']})
* OOF blend: `{metrics['outputs']['oof_blend']}` | OOF text: `{metrics['outputs']['oof_text']}` | OOF id-prior: `{metrics['outputs']['oof_id']}`
* Test probs: `{metrics['outputs']['test_blend']}` (+ `{metrics['outputs']['test_text']}`, `{metrics['outputs']['test_id']}`)
* Fold scores: `{metrics['outputs']['fold_scores']}` | Metrics: `{metrics['outputs']['metrics']}`
* Fold assignments: `{metrics['outputs']['fold_assignments']}` | Models: `{metrics['outputs']['models_dir']}`
* Run log: `logs/experiment_text_id_blend.log`
* Runtime: {metrics['runtime']['total_seconds']}s (folds {metrics['runtime']['fold_seconds']}s, final fit {metrics['runtime']['final_fit_seconds']}s)

## 9. Scope notes

{chr(10).join('- ' + n for n in metrics['scope_notes'])}
"""


def _write_summary(metrics: dict, fold_rows: pd.DataFrame, grid: pd.DataFrame,
                   checks: list[dict]) -> None:
    text = _summary_part1(metrics, fold_rows, grid) + _summary_part2(metrics, checks)
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = SUMMARY_OUT.with_name(SUMMARY_OUT.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, SUMMARY_OUT)
    log(f"wrote {SUMMARY_OUT.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    t_start = time.time()
    for d in (DATA_CHECKS_DIR, MODELS_DIR, OOF_DIR, TEST_PRED_DIR, REPORTS_DIR,
              SUBMISSIONS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log("=" * 78)
    log("EXPERIMENT text_id_blend -- text TF-IDF + ID-locality prior -- start")

    global RESOURCES
    RESOURCES = detect_resources()

    train, test, sample = load_data()
    y = train["y"].to_numpy(dtype=int)
    train_ids = train["id"].to_numpy(dtype=np.int64)
    test_ids = test["id"].to_numpy(dtype=np.int64)
    train_clean = train["clean"].to_numpy()
    test_clean = test["clean"].to_numpy()

    fold_ids, fold_info = build_folds(train)
    fold_info["id_prior_diags"] = []

    # ---------------- Step 3: cross-validation -----------------------------
    t_folds = time.time()
    oof_text = np.zeros((len(train), len(CLASSES)), dtype=float)
    oof_id = np.zeros((len(train), len(CLASSES)), dtype=float)
    fold_rows, fold_va_idx, fold_lookups = [], {}, {}
    for k in range(N_SPLITS):
        va = np.flatnonzero(fold_ids == k)
        tr = np.flatnonzero(fold_ids != k)
        fold_va_idx[k] = va

        p_text, tmeta = fold_text_probs(k, train_clean[tr], y[tr], train_clean[va],
                                        train_ids[va])
        p_id, idiag = id_prior_probs(train_ids[va], train_ids[tr], y[tr], tag=f"fold{k}_val")
        fold_info["id_prior_diags"].append(idiag)
        oof_text[va] = p_text
        oof_id[va] = p_id

        m_text = metric_block(y[va], p_text)
        m_id = metric_block(y[va], p_id)
        m_blend = metric_block(y[va], blend_probs(p_text, p_id, ALPHA_REFERENCE))

        # leak-free override for this fold (lookup from the fold's TRAIN split only)
        lookup_k, ostats = unanimous_lookup(train_clean[tr], y[tr])
        fold_lookups[k] = lookup_k
        fold_rows.append({
            "fold": k, "n_train": int(len(tr)), "n_val": int(len(va)),
            "n_val_after_grouping": int(len(va)),
            "macro_f1_text": round(m_text["macro_f1"], 6),
            "accuracy_text": round(m_text["accuracy"], 6),
            "macro_f1_id": round(m_id["macro_f1"], 6),
            "accuracy_id": round(m_id["accuracy"], 6),
            "macro_f1_blend_0.40": round(m_blend["macro_f1"], 6),
            "accuracy_blend_0.40": round(m_blend["accuracy"], 6),
            "f1_class0_blend": round(m_blend["f1_class0"], 6),
            "f1_class1_blend": round(m_blend["f1_class1"], 6),
            "f1_class2_blend": round(m_blend["f1_class2"], 6),
            "text_fit_seconds": tmeta["fit_seconds"], "text_model_resumed": tmeta["resumed"],
            "lr_n_iter": str(tmeta["lr_n_iter"]),
            "text_model_path": tmeta["model_path"],
            "id_prior_mean_neighbours": idiag["mean_neighbours_used"],
            "id_prior_mean_kth_distance": idiag["mean_kth_distance"],
            "fold_override_unanimous_texts": ostats["n_unanimous_groups"],
            "val_rows_with_text_also_in_train": int(
                next(r["val_rows_with_text_also_in_train"]
                     for r in fold_info["fold_rows"] if r["fold"] == k)),
        })
        log(f"fold[{k}]: macro_f1 text={m_text['macro_f1']:.6f} "
            f"id={m_id['macro_f1']:.6f} blend@0.40={m_blend['macro_f1']:.6f} | "
            f"acc text={m_text['accuracy']:.6f} blend={m_blend['accuracy']:.6f} | "
            f"free_ram={ram_free_gb()}GB")
    fold_seconds = round(time.time() - t_folds, 2)
    log(f"cross-validation finished in {fmt_secs(fold_seconds)}")

    # ---------------- STOP CONDITION #1 ------------------------------------
    check_fold_expectations(fold_rows)

    return finalize(t_start, train, test, sample, y, train_ids, test_ids, train_clean,
                    test_clean, fold_ids, fold_info, fold_rows, fold_va_idx, fold_lookups,
                    oof_text, oof_id, fold_seconds)


def finalize(t_start, train, test, sample, y, train_ids, test_ids, train_clean, test_clean,
             fold_ids, fold_info, fold_rows, fold_va_idx, fold_lookups, oof_text, oof_id,
             fold_seconds) -> int:
    """Pooled OOF metrics, alpha choice, OOF artifacts, then the final fit/submission."""
    fold_scores = pd.DataFrame(fold_rows)
    fold_stats = {name: {"mean": round(float(fold_scores[col].mean()), 6),
                         "std": round(float(fold_scores[col].std(ddof=0)), 6),
                         "min": round(float(fold_scores[col].min()), 6),
                         "max": round(float(fold_scores[col].max()), 6)}
                  for name, col in EXPECTED_COLUMN.items()}

    oof_overall = {
        "text_only": metric_block(y, oof_text),
        "id_prior_only": metric_block(y, oof_id),
        "blend_0.40": metric_block(y, blend_probs(oof_text, oof_id, ALPHA_REFERENCE)),
    }
    chosen_alpha, grid = alpha_grid_search(y, oof_text, oof_id)
    oof_blend_best = blend_probs(oof_text, oof_id, chosen_alpha)
    oof_overall["blend_best"] = metric_block(y, oof_blend_best)
    best_pred = oof_blend_best.argmax(axis=1)
    cm = confusion_matrix(y, best_pred, labels=list(CLASSES)).tolist()
    overall_before = oof_overall["blend_best"]["macro_f1"]

    # leak-free OOF estimate of the duplicate-text override at the chosen alpha
    ov_pred = np.zeros(len(y), dtype=int)
    ov_hits_total = 0
    for k in range(N_SPLITS):
        va = fold_va_idx[k]
        p = blend_probs(oof_text[va], oof_id[va], chosen_alpha)
        pred_k, hits_k = apply_override(train_clean[va], p.argmax(axis=1), fold_lookups[k])
        ov_pred[va] = pred_k
        ov_hits_total += hits_k
    after = metric_block(y, np.eye(len(CLASSES))[ov_pred])
    log(f"leak-free OOF override effect: {ov_hits_total:,} rows overridden, "
        f"macro F1 {overall_before:.6f} -> {after['macro_f1']:.6f} "
        f"({after['macro_f1'] - overall_before:+.6f})")

    # ---------------- OOF artifacts ----------------------------------------
    oof_df = pd.DataFrame({
        "id": train_ids, "y_true": y,
        **{c: oof_blend_best[:, i] for i, c in enumerate(PROB_COLS)},
        "y_pred": best_pred, "fold": fold_ids,
        "model_name": f"{MODEL_NAME}_a{chosen_alpha:.2f}",
    })[OOF_COLUMNS]
    atomic_write_csv(oof_df, OOF_BLEND_OUT)
    atomic_write_csv(pd.DataFrame({
        "id": train_ids, "y_true": y,
        **{c: oof_text[:, i] for i, c in enumerate(PROB_COLS)},
        "y_pred": oof_text.argmax(axis=1), "fold": fold_ids,
        "model_name": "text_lbfgs_tfidf"})[OOF_COLUMNS], OOF_TEXT_OUT)
    atomic_write_csv(pd.DataFrame({
        "id": train_ids, "y_true": y,
        **{c: oof_id[:, i] for i, c in enumerate(PROB_COLS)},
        "y_pred": oof_id.argmax(axis=1), "fold": fold_ids,
        "model_name": f"id_prior_k{K_NEIGHBORS}"})[OOF_COLUMNS], OOF_ID_OUT)
    log(f"wrote OOF artifacts: {OOF_BLEND_OUT.name}, {OOF_TEXT_OUT.name}, {OOF_ID_OUT.name}")

    return _final_stage(t_start, train, test, sample, y, train_ids, test_ids, train_clean,
                        test_clean, fold_info, fold_scores, fold_stats, oof_df, oof_overall,
                        chosen_alpha, grid, cm, overall_before, after, ov_hits_total,
                        fold_seconds)


def _final_stage(t_start, train, test, sample, y, train_ids, test_ids, train_clean, test_clean,
                 fold_info, fold_scores, fold_stats, oof_df, oof_overall, chosen_alpha, grid,
                 cm, overall_before, after, ov_hits_total, fold_seconds) -> int:
    """Step 5: full refit, test priors, blend, override, validated submission, reports."""
    # ---------------- Step 5a: full refit ----------------------------------
    t_final = time.time()
    log("final fit: refitting TF-IDF + LogisticRegression on the FULL training set")
    word, char, clf, fmeta = fit_text_model(train_clean, y, tag="full")
    fdir = MODELS_DIR / "full"
    fdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(word, fdir / "word_vectorizer.joblib")
    joblib.dump(char, fdir / "char_vectorizer.joblib")
    joblib.dump(clf, fdir / "lr.joblib")
    p_text_test = np.asarray(clf.predict_proba(build_text_matrix(word, char, test_clean)),
                             dtype=float)

    # ---------------- Step 5b: id prior for the full test set --------------
    p_id_test, idiag_test = id_prior_probs(test_ids, train_ids, y, tag="full_test")
    fold_info["id_prior_diags"].append(idiag_test)

    # ---------------- Step 5c: blend at the OOF-chosen alpha ---------------
    p_test_blend = blend_probs(p_text_test, p_id_test, chosen_alpha)
    test_pred_pure = p_test_blend.argmax(axis=1)
    log(f"test blend computed at alpha={chosen_alpha:.2f}; "
        f"mean max prob={p_test_blend.max(axis=1).mean():.4f}")

    # ---------------- Step 4b: duplicate-text override for test ------------
    lookup_full, ostats_full = unanimous_lookup(train_clean, y)
    test_pred_override, n_test_override = apply_override(test_clean, test_pred_pure, lookup_full)
    test_override_share = round(100.0 * n_test_override / len(test), 4)
    changed = int((test_pred_override != test_pred_pure).sum())
    log(f"test override: {n_test_override:,} / {len(test):,} rows "
        f"({test_override_share:.3f}%) matched a unanimous training text; "
        f"label changed on {changed:,} rows")

    # ---------------- test probability artifacts (pre-override argmax) -----
    test_probs_df = pd.DataFrame({
        "id": test_ids, **{c: p_test_blend[:, i] for i, c in enumerate(PROB_COLS)},
        "y_pred": test_pred_pure, "model_name": f"{MODEL_NAME}_a{chosen_alpha:.2f}",
    })[TEST_PROB_COLUMNS]
    atomic_write_csv(test_probs_df, TEST_BLEND_OUT)
    atomic_write_csv(pd.DataFrame({
        "id": test_ids, **{c: p_text_test[:, i] for i, c in enumerate(PROB_COLS)},
        "y_pred": p_text_test.argmax(axis=1), "model_name": "text_lbfgs_tfidf"
    })[TEST_PROB_COLUMNS], TEST_TEXT_OUT)
    atomic_write_csv(pd.DataFrame({
        "id": test_ids, **{c: p_id_test[:, i] for i, c in enumerate(PROB_COLS)},
        "y_pred": p_id_test.argmax(axis=1), "model_name": f"id_prior_k{K_NEIGHBORS}"
    })[TEST_PROB_COLUMNS], TEST_ID_OUT)
    log(f"wrote test probability artifacts: {TEST_BLEND_OUT.name}, {TEST_TEXT_OUT.name}, "
        f"{TEST_ID_OUT.name}")
    return _submit_and_report(t_start, train, test, sample, y, train_ids, fold_info,
                              fold_scores, fold_stats, oof_df, oof_overall, chosen_alpha, grid,
                              cm, overall_before, after, ov_hits_total, ostats_full,
                              n_test_override, test_override_share, changed,
                              test_probs_df, test_pred_override, test_ids, fmeta, idiag_test,
                              t_final, fold_seconds)


def _submit_and_report(t_start, train, test, sample, y, train_ids, fold_info, fold_scores,
                       fold_stats, oof_df, oof_overall, chosen_alpha, grid, cm, overall_before,
                       after, ov_hits_total, ostats_full, n_test_override, test_override_share,
                       changed, test_probs_df, test_pred_override, test_ids, fmeta, idiag_test,
                       t_final, fold_seconds) -> int:
    """Bias check -> validate -> write submission -> metrics JSON -> summary."""
    sub = pd.DataFrame({"id": test_ids, "y_pred": test_pred_override})[SUBMISSION_COLUMNS]
    bias = bias_check(test_pred_override, y)

    # ---------------- STOP CONDITION #3: validate BEFORE writing -----------
    checks = run_invariants(oof_df, test_probs_df, sub, train, test, sample, fold_info)
    failed = [c for c in checks if c["status"] != "PASS"]
    if failed:
        log("STOP CONDITION #3 VIOLATED: submission validation failed -> submission NOT written")
        for c in failed:
            log(f"   FAILED CHECK: {c['check']} :: {c['detail']}")
        write_json({"experiment": "text_id_blend", "model_name": MODEL_NAME,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "resources": RESOURCES,
                    "chosen_alpha": chosen_alpha, "oof_overall": oof_overall, "stopped": True,
                    "stop_reason": "submission validation failed", "failed_checks": failed,
                    "invariants": checks}, METRICS_OUT)
        raise StopCondition("submission validation failed; nothing was written")

    atomic_write_csv(sub, SUBMISSION_OUT)
    log(f"wrote {SUBMISSION_OUT.relative_to(ROOT)} rows={len(sub)} (validated before writing)")

    reread = pd.read_csv(SUBMISSION_OUT, encoding="utf-8")
    ok_reread = (list(reread.columns) == SUBMISSION_COLUMNS and len(reread) == len(sample)
                 and bool((reread["id"].to_numpy() == sample["id"].to_numpy()).all())
                 and bool(np.isin(reread["y_pred"].to_numpy(), list(CLASSES)).all())
                 and int(reread.isna().to_numpy().sum()) == 0)
    checks.append({"check": "submission_re_read_from_disk_is_valid",
                   "status": "PASS" if ok_reread else "FAIL",
                   "detail": f"rows={len(reread)} cols={list(reread.columns)}"})
    log(f"CHECK {'PASS' if ok_reread else 'FAIL'}: submission_re_read_from_disk_is_valid :: "
        f"rows={len(reread)}")
    if not ok_reread:
        raise StopCondition("the submission re-read from disk failed validation")

    # ---------------- metrics JSON + reports --------------------------------
    return _emit_metrics(t_start, train, test, sample, y, fold_info, fold_scores, fold_stats,
                         oof_df, oof_overall, chosen_alpha, grid, cm, overall_before, after,
                         ov_hits_total, ostats_full, n_test_override, test_override_share,
                         changed, test_probs_df, sub, bias, checks, fmeta, idiag_test,
                         t_final, fold_seconds)


def _emit_metrics(t_start, train, test, sample, y, fold_info, fold_scores, fold_stats, oof_df,
                  oof_overall, chosen_alpha, grid, cm, overall_before, after, ov_hits_total,
                  ostats_full, n_test_override, test_override_share, changed, test_probs_df,
                  sub, bias, checks, fmeta, idiag_test, t_final, fold_seconds) -> int:
    """Assemble the metrics JSON, fold-score CSV and summary, then print the result block."""
    override_stats = {
        "lookup_stats": ostats_full,
        "n_test_overridden": int(n_test_override),
        "n_test_rows": int(len(test_probs_df)),
        "test_override_share_pct": float(test_override_share),
        "test_labels_changed_by_override": int(changed),
        "oof_rows_overridden": int(ov_hits_total),
        "oof_macro_f1_before_override": round(overall_before, 6),
        "oof_macro_f1_after_override": round(after["macro_f1"], 6),
        "oof_macro_f1_delta": round(after["macro_f1"] - overall_before, 6),
        "oof_method": "per-fold lookup built only from that fold's training split (leak-free)",
    }
    metrics = {
        "experiment": "text_id_blend",
        "model_name": f"{MODEL_NAME}_a{chosen_alpha:.2f}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resources": RESOURCES,
        "validation": {"method": "StratifiedKFold", "n_splits": N_SPLITS, "shuffle": True,
                       "random_state": SEED, "metric": "macro_f1",
                       "fold_rows": fold_scores.to_dict(orient="records"),
                       "fold_assignment_rows": fold_info["fold_rows"],
                       "duplicate_text_note": fold_info["note"]},
        "id_prior": {"k": K_NEIGHBORS, "weights": "1/(distance+1e-6)",
                     "labels": "training ids/labels only (never validation or test labels)",
                     "diagnostics": fold_info["id_prior_diags"], "final_test_prior": idiag_test},
        "text_model": {
            "features": "word(1-2) + char_wb(3-5) TF-IDF, 60k each, min_df=2, sublinear_tf, "
                        "float32, hstacked with scipy.sparse.hstack",
            "clean_text": "URL -> ' URL ', @mention -> ' USER ', whitespace collapsed; no "
                          "lowercasing, no stemming, Bangla script + emoji preserved",
            "lr": {**{k: str(v) for k, v in LR_KW.items()}, "random_state": SEED},
            "full_fit": fmeta},
        "chosen_alpha": chosen_alpha,
        "alpha_grid": grid.to_dict(orient="records"),
        "fold_macro_f1_stats": fold_stats,
        "fold_min": {k: v["min"] for k, v in fold_stats.items()},
        "oof_overall": {k: {kk: round(vv, 6) for kk, vv in v.items()}
                        for k, v in oof_overall.items()},
        "oof_confusion_matrix": {"labels": [f"{c}_{CLASS_LABELS[c]}" for c in CLASSES],
                                 "matrix": cm},
        "oof_pred_class_distribution": {int(c): int((oof_df["y_pred"] == c).sum())
                                        for c in CLASSES},
        "test_pred_class_distribution_pre_override": {
            int(c): int((test_probs_df["y_pred"] == c).sum()) for c in CLASSES},
        "submission_pred_class_distribution": {int(c): int((sub["y_pred"] == c).sum())
                                               for c in CLASSES},
        "override": override_stats,
    }
    return _emit_metrics_tail(metrics, t_start, fold_info, fold_scores, grid, checks, bias,
                              override_stats, oof_overall, fold_stats, cm, sub, fmeta,
                              idiag_test, t_final, fold_seconds)


def _emit_metrics_tail(metrics, t_start, fold_info, fold_scores, grid, checks, bias,
                       override_stats, oof_overall, fold_stats, cm, sub, fmeta, idiag_test,
                       t_final, fold_seconds) -> int:
    """Finish the metrics dict, write JSON + CSV + summary, print the result block."""
    metrics.update({
        "bias_check": bias,
        "invariants": checks,
        "invariants_all_passed": bool(all(c["status"] == "PASS" for c in checks)),
        "submission_rows": int(len(sub)),
        "submission_written": True,
        "runtime": {"total_seconds": round(time.time() - t_start, 2),
                    "fold_seconds": fold_seconds,
                    "final_fit_seconds": round(time.time() - t_final, 2),
                    "folds_reused_from_disk": int(fold_scores["text_model_resumed"].sum())},
        "outputs": {
            "submission": str(SUBMISSION_OUT.relative_to(ROOT)),
            "oof_blend": str(OOF_BLEND_OUT.relative_to(ROOT)),
            "oof_text": str(OOF_TEXT_OUT.relative_to(ROOT)),
            "oof_id": str(OOF_ID_OUT.relative_to(ROOT)),
            "test_blend": str(TEST_BLEND_OUT.relative_to(ROOT)),
            "test_text": str(TEST_TEXT_OUT.relative_to(ROOT)),
            "test_id": str(TEST_ID_OUT.relative_to(ROOT)),
            "fold_scores": str(FOLD_SCORES_OUT.relative_to(ROOT)),
            "metrics": str(METRICS_OUT.relative_to(ROOT)),
            "summary": str(SUMMARY_OUT.relative_to(ROOT)),
            "fold_assignments": str(FOLD_ASSIGN_CSV.relative_to(ROOT)),
            "models_dir": str(MODELS_DIR.relative_to(ROOT)),
        },
        "scope_notes": [
            "The id-locality signal is dataset-specific (contiguous non-overlapping train/test "
            "ids from one scrape order); do not reuse it as a generic feature elsewhere.",
            "The prior never sees validation/test labels: the neighbour pool is each fold's "
            "training split only and the function signature excludes any query labels.",
            "Plain StratifiedKFold is used as specified, so duplicate texts may straddle folds; "
            "val_rows_with_text_also_in_train per fold quantifies that optimism.",
            "test_text_id_blend.csv keeps y_pred = argmax(blended probs) (pure model); the "
            "duplicate-text override is applied only in the submission file.",
            "Submission validated before writing and re-read from disk afterwards; alpha was "
            "selected on pooled OOF only, never on leaderboard information.",
        ],
    })
    write_json(metrics, METRICS_OUT)
    atomic_write_csv(fold_scores, FOLD_SCORES_OUT)
    log(f"wrote {METRICS_OUT.relative_to(ROOT)} and {FOLD_SCORES_OUT.relative_to(ROOT)}")
    _write_summary(metrics, fold_scores, grid, checks)
    _print_final_block(metrics, oof_overall, fold_stats, cm, bias, override_stats)
    log(f"EXPERIMENT finished in {fmt_secs(metrics['runtime']['total_seconds'])}; "
        f"submission written and validated; chosen alpha={metrics['chosen_alpha']:.2f}")
    return 0


def _print_final_block(metrics, oof_overall, fold_stats, cm, bias, override_stats) -> None:
    """Print the final result block (also captured in the run log)."""
    lines = ["", "============ TEXT + ID-PRIOR BLEND -- VERIFIED RESULTS ============",
             "Validation: StratifiedKFold(5, shuffle=True, random_state=42); metric: Macro F1",
             "1. Per-fold Macro F1 (text | id-prior | blend@0.40):"]
    for r in metrics["validation"]["fold_rows"]:
        lines.append(f"   fold{r['fold']}: {r['macro_f1_text']:.6f} | {r['macro_f1_id']:.6f} | "
                     f"{r['macro_f1_blend_0.40']:.6f}  (n_val={r['n_val']})")
    lines += [
        f"2. Fold mean+-std: text {fold_stats['text_only']['mean']:.6f}+-"
        f"{fold_stats['text_only']['std']:.6f} | id {fold_stats['id_prior']['mean']:.6f}+-"
        f"{fold_stats['id_prior']['std']:.6f} | blend@0.40 "
        f"{fold_stats['blend_0.40']['mean']:.6f}+-{fold_stats['blend_0.40']['std']:.6f}",
        f"3. Pooled OOF Macro F1: text-only {oof_overall['text_only']['macro_f1']:.6f} | "
        f"id-only {oof_overall['id_prior_only']['macro_f1']:.6f} | "
        f"blend@0.40 {oof_overall['blend_0.40']['macro_f1']:.6f} | "
        f"blend@{metrics['chosen_alpha']:.2f} {oof_overall['blend_best']['macro_f1']:.6f}",
        f"4. Per-class F1 (OOF, blend@chosen): class0 "
        f"{oof_overall['blend_best']['f1_class0']:.6f}, class1 "
        f"{oof_overall['blend_best']['f1_class1']:.6f}, class2 "
        f"{oof_overall['blend_best']['f1_class2']:.6f}",
        f"5. OOF accuracy (blend@chosen): {oof_overall['blend_best']['accuracy']:.6f}",
        "6. Confusion matrix (OOF, rows=true, cols=pred, order [0,1,2]):",
        f"   {cm[0]}", f"   {cm[1]}", f"   {cm[2]}",
        f"7. Chosen alpha (text weight) on OOF = {metrics['chosen_alpha']:.2f}; "
        f"reference alpha 0.40 -> {oof_overall['blend_0.40']['macro_f1']:.6f}",
        f"8. Override: {override_stats['n_test_overridden']:,}/{override_stats['n_test_rows']:,} "
        f"test rows ({override_stats['test_override_share_pct']:.3f}%); "
        f"{override_stats['test_labels_changed_by_override']:,} labels changed; leak-free OOF "
        f"{override_stats['oof_macro_f1_before_override']:.6f} -> "
        f"{override_stats['oof_macro_f1_after_override']:.6f} "
        f"({override_stats['oof_macro_f1_delta']:+.6f}); conflicting groups skipped="
        f"{override_stats['lookup_stats']['n_conflicting_groups_skipped']}",
        f"9. Bias check: pred shares {bias['pred_share_pct']} vs train {bias['train_share_pct']} "
        f"(drift pp {bias['drift_pp']}, threshold {bias['threshold_pp']:.0f}pp, "
        f"passed={bias['passed']})",
        f"10. Invariants: {sum(1 for c in metrics['invariants'] if c['status'] == 'PASS')}"
        f"/{len(metrics['invariants'])} passed; submission {metrics['outputs']['submission']} "
        f"(rows={metrics['submission_rows']:,})",
        "    External references: text ~0.52 | id-prior ~0.61-0.62 | blend ~0.64",
        "=================================================================",
    ]
    for line in lines:
        log(line)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StopCondition:
        log("EXPERIMENT STOPPED by a hard stop condition (see the lines above).")
        raise
    except Exception:
        import traceback

        log("EXPERIMENT FAILED -- traceback follows")
        log(traceback.format_exc())
        log("Completed fold artifacts on disk will be reused on the next run (resume).")
        raise

















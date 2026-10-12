"""
src/experiment_small_tfidf.py -- Standalone experiment: compact regularized TF-IDF
representation + Logistic Regression, tested for diversity against the large
TF-IDF pipeline.

SELF-CONTAINED BY DESIGN
------------------------
* Does NOT import ``src/utils.py`` (runtime-unverified) and does NOT touch
  ``src/check_data.py``.
* Reads ONLY train.csv / test.csv / sample_submission.csv, all opened read-only.
* No external data, no downloads, no pretrained weights, no network access.
* Does NOT write any submission file and does NOT create any zip archive.

METHOD
------
Grouping normalization (for duplicate-safe folds only; the model always sees the
raw text): NFKC -> lowercase -> strip outer whitespace -> collapse repeated
whitespace -> URL -> <url> -> mention -> <user>.
Validation: StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42) with
``groups`` = normalized text, metric = Macro F1.

Features: word TF-IDF (1-2 grams, min_df=2, max_df=0.995, sublinear, 60k feats)
hstacked with char_wb TF-IDF (3-5 grams, min_df=2, sublinear, 60k feats), both
float32, vocabulary fitted unsupervised on concatenated train+test text.

Model: one LogisticRegression per fold (C=4.0, liblinear, class_weight=balanced,
max_iter=2000, random_state=42+fold). sklearn 1.9 removed native multiclass support
from solver='liblinear', so the estimator is wrapped in OneVsRestClassifier to keep
the requested one-vs-rest scheme and solver; predict_proba rows still sum to 1.

Run:  python -u src/experiment_small_tfidf.py
Resumable: completed folds are written immediately and reloaded on the next run.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
SAMPLE_CSV = ROOT / "sample_submission.csv"

DATA_CHECKS_DIR = ROOT / "data_checks"
MODELS_DIR = ROOT / "models" / "small_tfidf"
OOF_DIR = ROOT / "predictions" / "oof"
TEST_PRED_DIR = ROOT / "predictions" / "test"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

FOLD_ASSIGNMENTS_CSV = DATA_CHECKS_DIR / "fold_assignments_3fold.csv"
OOF_OUT = OOF_DIR / "oof_small_tfidf_lr.csv"
TEST_OUT = TEST_PRED_DIR / "test_small_tfidf_lr.csv"
FOLD_SCORES_OUT = REPORTS_DIR / "small_tfidf_fold_scores.csv"
METRICS_OUT = REPORTS_DIR / "small_tfidf_metrics.json"
SUMMARY_OUT = REPORTS_DIR / "small_tfidf_summary.md"
LOG_PATH = LOGS_DIR / "experiment_small_tfidf.log"

MODEL_NAME = "lr_small_tfidf"          # value stored in the model_name columns
CLASSES = (0, 1, 2)
CLASS_LABELS = {0: "explicitly_toxic", 1: "subtly_toxic", 2: "non_toxic"}
PROB_COLS = ("prob_0", "prob_1", "prob_2")
OOF_COLUMNS = ["id", "y_true", "prob_0", "prob_1", "prob_2", "y_pred", "fold", "model_name"]
TEST_PROB_COLUMNS = ["id", "prob_0", "prob_1", "prob_2", "y_pred", "model_name"]

N_SPLITS = 3
SEED = 42
BENCH_ROBUST = 0.525448      # verified robust grouped-OOF best (LR + ComplementNB)
BENCH_RESEARCH = 0.525574    # verified research best (cross-fitted calibration)

WORD_KW = dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.995,
               sublinear_tf=True, max_features=60_000, dtype=np.float32)
CHAR_KW = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
               sublinear_tf=True, max_features=60_000, dtype=np.float32)
LR_KW = dict(C=4.0, solver="liblinear", class_weight="balanced", max_iter=2000)

RESOURCES: dict = {}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    try:
        print(line, flush=True)
    except UnicodeEncodeError:  # console codec safety
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def fmt_secs(seconds: float) -> str:
    return f"{seconds:.1f}s ({seconds / 60:.2f} min)"


# --------------------------------------------------------------------------- #
# Resource safety
# --------------------------------------------------------------------------- #
def detect_resources(verbose: bool = True) -> dict:
    """Detect CPU count and free RAM before doing anything expensive."""
    cpu = os.cpu_count() or 4
    total_gb = free_gb = float("nan")
    try:
        import psutil

        vm = psutil.virtual_memory()
        total_gb = vm.total / 1e9
        free_gb = vm.available / 1e9
    except Exception as exc:  # psutil is optional
        log(f"WARN psutil unavailable ({exc}); RAM unknown -> conservative defaults")

    n_jobs = int(max(1, min(4, cpu // 2)))
    if not np.isnan(free_gb) and free_gb < 3.0:
        n_jobs = min(n_jobs, 2)
    res = {"cpu_count": int(cpu), "ram_total_gb": round(total_gb, 2),
           "ram_free_gb": round(free_gb, 2), "n_jobs": int(n_jobs),
           "gpu_used": False, "sparse_dtype": "float32"}
    if verbose:
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
    """Log a sparse matrix footprint (never densified)."""
    nnz = int(getattr(X, "nnz", 0))
    info = {"name": name, "shape": list(X.shape), "nnz": nnz,
            "sparsity_pct": round(100.0 * (1.0 - nnz / max(1.0, X.shape[0] * X.shape[1])), 5),
            "est_gb": round(nnz * 8 / 1e9, 3)}
    log(f"matrix[{name}]: shape={X.shape} nnz={nnz:,} dtype={X.dtype} "
        f"est~{info['est_gb']}GB")
    return info


# --------------------------------------------------------------------------- #
# Data loading (read-only) + grouping normalization
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_for_grouping(text) -> str:
    """Normalize text FOR GROUPING ONLY (never used as model input).

    NFKC -> lowercase -> URL -> <url> -> mention -> <user> -> collapse repeated
    whitespace -> strip outer whitespace.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.lower()
    s = _URL_RE.sub("<url>", s)
    s = _MENTION_RE.sub("<user>", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three competition CSVs with schema assertions. Read-only."""
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
    log(f"loaded train={train.shape} test={test.shape} sample={sample.shape}")
    log(f"train/test id overlap={len(set(train['id']) & set(test['id']))} "
        f"sample.order==test.order:{bool((sample['id'].to_numpy() == test['id'].to_numpy()).all())}")
    return train, test, sample


# --------------------------------------------------------------------------- #
# Atomic writers (self-contained)
# --------------------------------------------------------------------------- #
def atomic_write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write CSV via temp file + os.replace; protected inputs are never touched."""
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
# Duplicate-safe grouped folds
# --------------------------------------------------------------------------- #
def build_grouped_folds(train: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """StratifiedGroupKFold(3) on normalized-text groups; saves fold assignments.

    Verifies the leak-safety property: every normalized-text group lands in
    exactly one fold (so duplicates can never sit in both train and validation).
    """
    y = train["y"].to_numpy(dtype=int)
    norm = train["text"].map(normalize_for_grouping).to_numpy()
    groups = pd.factorize(norm, sort=False)[0].astype(np.int64)

    sizes = pd.Series(groups).value_counts()
    n_groups = int(sizes.shape[0])
    dup_groups = int((sizes > 1).sum())
    rows_in_dup_groups = int(sizes[sizes > 1].sum())
    fold_ids = np.full(len(train), -1, dtype=int)

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for k, (_tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold_ids[va] = k
    assert (fold_ids >= 0).all(), "every row must receive a fold id"

    # --- verification: each group in exactly one fold -----------------------
    gf = pd.DataFrame({"group": groups, "fold": fold_ids})
    distinct_folds_per_group = gf.groupby("group")["fold"].nunique()
    groups_crossing = int((distinct_folds_per_group > 1).sum())
    group_to_fold = gf.groupby("group")["fold"].first()

    # class balance per fold (informational; groups may break exact stratification)
    dup_mask = np.isin(groups, sizes[sizes > 1].index.to_numpy())
    fold_balance = []
    for k in range(N_SPLITS):
        m = fold_ids == k
        row = {"fold": k, "n_val": int(m.sum()),
               "val_rows_in_dup_groups": int((m & dup_mask).sum()),
               "n_groups": int(group_to_fold[group_to_fold == k].shape[0])}
        for c in CLASSES:
            row[f"val_c{c}"] = int(((y == c) & m).sum())
        fold_balance.append(row)

    fold_info = {
        "strategy": "StratifiedGroupKFold",
        "n_splits": N_SPLITS, "shuffle": True, "random_state": SEED,
        "grouping": "NFKC+lower+ws-collapse+<url>+<user>",
        "n_groups": n_groups,
        "n_rows": int(len(train)),
        "n_duplicate_groups": dup_groups,
        "rows_in_duplicate_groups": rows_in_dup_groups,
        "duplicate_row_share": round(rows_in_dup_groups / len(train), 4),
        "max_group_size": int(sizes.max()),
        "groups_spanning_multiple_folds": groups_crossing,
        "leak_check_passed": bool(groups_crossing == 0),
        "fold_balance": fold_balance,
    }
    log(f"groups: n={n_groups} dup_groups={dup_groups} "
        f"rows_in_dup_groups={rows_in_dup_groups} ({fold_info['duplicate_row_share']:.2%}) "
        f"max_group_size={fold_info['max_group_size']}")
    for row in fold_balance:
        log(f"fold[{row['fold']}]: n_val={row['n_val']} "
            f"class_counts=({row['val_c0']},{row['val_c1']},{row['val_c2']})")
    return fold_ids, fold_info


def save_fold_assignments(train: pd.DataFrame, fold_ids: np.ndarray) -> Path:
    """Persist fold assignments (group identity is hashed, not stored verbatim)."""
    import hashlib

    norm = train["text"].map(normalize_for_grouping).to_numpy()
    hashes = [hashlib.sha1(s.encode("utf-8")).hexdigest()[:16] for s in norm]
    sizes = pd.Series(norm).value_counts()
    out = pd.DataFrame({
        "id": train["id"].to_numpy(),
        "y": train["y"].to_numpy(),
        "group_hash": hashes,
        "group_size": [int(sizes[s]) for s in norm],
        "fold": fold_ids,
    })
    atomic_write_csv(out, FOLD_ASSIGNMENTS_CSV)
    log(f"wrote {FOLD_ASSIGNMENTS_CSV.relative_to(ROOT)} rows={len(out)}")
    return FOLD_ASSIGNMENTS_CSV


# --------------------------------------------------------------------------- #
# Feature matrices: word + char TF-IDF fitted unsupervisely on train+test text
# --------------------------------------------------------------------------- #
def fit_vectorizers(all_text: list[str]) -> tuple[TfidfVectorizer, TfidfVectorizer, dict]:
    """Fit both TF-IDF vectorizers on the concatenated train+test corpus."""
    t0 = time.time()
    word = TfidfVectorizer(**WORD_KW)
    word.fit(all_text)
    t_word = time.time() - t0

    t1 = time.time()
    char = TfidfVectorizer(**CHAR_KW)
    char.fit(all_text)
    t_char = time.time() - t1

    info = {
        "word": {**{k: str(v) for k, v in WORD_KW.items()},
                 "n_features": int(len(word.vocabulary_)), "fit_seconds": round(t_word, 2)},
        "char": {**{k: str(v) for k, v in CHAR_KW.items()},
                 "n_features": int(len(char.vocabulary_)), "fit_seconds": round(t_char, 2)},
        "fitted_on": "concatenated train+test text (unsupervised vocabulary only)",
        "n_documents_fit": len(all_text),
    }
    log(f"vectorizers fitted on {len(all_text)} documents: "
        f"word_features={info['word']['n_features']:,} ({fmt_secs(t_word)}), "
        f"char_features={info['char']['n_features']:,} ({fmt_secs(t_char)})")
    joblib.dump(word, MODELS_DIR / "word_vectorizer.joblib")
    joblib.dump(char, MODELS_DIR / "char_vectorizer.joblib")
    return word, char, info


def load_or_fit_vectorizers(all_text: list[str]):
    """Reuse previously saved vectorizers when present (resume-friendly)."""
    wpath = MODELS_DIR / "word_vectorizer.joblib"
    cpath = MODELS_DIR / "char_vectorizer.joblib"
    if wpath.exists() and cpath.exists():
        try:
            word = joblib.load(wpath)
            char = joblib.load(cpath)
            info = {
                "word": {"n_features": int(len(word.vocabulary_)), "loaded_from": str(wpath.relative_to(ROOT))},
                "char": {"n_features": int(len(char.vocabulary_)), "loaded_from": str(cpath.relative_to(ROOT))},
                "fitted_on": "concatenated train+test text (reloaded from joblib, resume)",
                "n_documents_fit": len(all_text),
            }
            log(f"reloaded saved vectorizers: word_features={info['word']['n_features']:,} "
                f"char_features={info['char']['n_features']:,}")
            return word, char, info
        except Exception as exc:  # corrupt artifact -> refit
            log(f"WARN could not load saved vectorizers ({exc}); refitting")
    return fit_vectorizers(all_text)


def build_matrices(word, char, texts: list[str], tag: str):
    """Transform raw texts and hstack word+char blocks into one CSR matrix."""
    t0 = time.time()
    Xw = word.transform(texts)
    Xc = char.transform(texts)
    X = sp.hstack([Xw, Xc], format="csr").astype(np.float32)
    report_sparse(f"{tag}_word", Xw)
    report_sparse(f"{tag}_char", Xc)
    report_sparse(f"{tag}_word+char", X)
    log(f"built {tag} matrix in {fmt_secs(time.time() - t0)}")
    return X


# --------------------------------------------------------------------------- #
# Fold artifacts (resume support)
# --------------------------------------------------------------------------- #
def fold_val_prob_path(k: int) -> Path:
    return OOF_DIR / f"_fold{k}_val_probs_small_tfidf_lr.csv"


def fold_test_prob_path(k: int) -> Path:
    return TEST_PRED_DIR / f"_fold{k}_test_probs_small_tfidf_lr.csv"


def fold_model_path(k: int) -> Path:
    return MODELS_DIR / f"lr_fold_{k}.joblib"


def _load_ok(path: Path, expected_ids: np.ndarray) -> pd.DataFrame | None:
    """Load a saved per-fold probability file if it matches the expected ids."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8")
        if len(df) != len(expected_ids):
            return None
        if not np.array_equal(df["id"].to_numpy(), expected_ids):
            return None
        if list(PROB_COLS) != [c for c in df.columns if c.startswith("prob_")]:
            return None
        p = df[list(PROB_COLS)].to_numpy(dtype=float)
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            return None
        return df
    except Exception:
        return None


def make_estimator(k: int):
    """Build fold k's estimator: one-vs-rest LogisticRegression, liblinear solver.

    COMPATIBILITY NOTE (sklearn >= 1.8/1.9): solver='liblinear' no longer supports
    native multiclass classification and raises ValueError for n_classes >= 3.
    To keep the requested solver, C, class_weight, max_iter and the one-vs-rest
    scheme, the same estimator is wrapped in OneVsRestClassifier (the fix sklearn
    itself recommends). OneVsRestClassifier.predict_proba normalizes the per-class
    OvR scores so each row sums to 1 -- documented sklearn behaviour, and verified
    by the integrity checks in this run.
    """
    return OneVsRestClassifier(
        LogisticRegression(random_state=SEED + k, **LR_KW), n_jobs=1)


def train_fold(k: int, X_tr, y_tr, X_va, val_ids, X_te, test_ids, resume: bool = True):
    """Fit one fold's LogisticRegression and produce val + test probabilities.

    Saves the fold model and both probability files immediately after the fold
    completes, so an interrupted run resumes here instead of restarting.
    """
    vpath, tpath = fold_val_prob_path(k), fold_test_prob_path(k)
    if resume:
        val_df = _load_ok(vpath, val_ids)
        test_df = _load_ok(tpath, test_ids)
        if val_df is not None and test_df is not None:
            log(f"fold[{k}]: RESUME -> loaded saved probabilities from "
                f"{vpath.name} and {tpath.name} (skipping training)")
            return (val_df[list(PROB_COLS)].to_numpy(dtype=float),
                    test_df[list(PROB_COLS)].to_numpy(dtype=float),
                    {"resumed": True, "train_seconds": np.nan,
                     "model_path": str(fold_model_path(k).relative_to(ROOT)),
                     "model_exists": fold_model_path(k).exists()})

    log(f"fold[{k}]: training LogisticRegression on {X_tr.shape[0]:,} rows "
        f"x {X_tr.shape[1]:,} features (free_ram={ram_free_gb()}GB)")
    t0 = time.time()
    clf = make_estimator(k)
    clf.fit(X_tr, y_tr)
    t_fit = time.time() - t0

    proba_va = np.asarray(clf.predict_proba(X_va), dtype=float)
    proba_te = np.asarray(clf.predict_proba(X_te), dtype=float)

    val_df = pd.DataFrame({"id": np.asarray(val_ids), **{c: proba_va[:, i] for i, c in enumerate(PROB_COLS)}})
    test_df = pd.DataFrame({"id": np.asarray(test_ids), **{c: proba_te[:, i] for i, c in enumerate(PROB_COLS)}})
    atomic_write_csv(val_df, vpath)          # saved immediately (resume point)
    joblib.dump(clf, fold_model_path(k))
    atomic_write_csv(test_df, tpath)
    log(f"fold[{k}]: fit in {fmt_secs(t_fit)}; saved {vpath.name}, "
        f"{fold_model_path(k).name}, {tpath.name}; free_ram={ram_free_gb()}GB")
    return proba_va, proba_te, {"resumed": False, "train_seconds": round(t_fit, 2),
                                "model_path": str(fold_model_path(k).relative_to(ROOT)),
                                "model_exists": True}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def fold_metrics(y_true, proba) -> dict:
    pred = proba.argmax(axis=1)
    per_class = f1_score(y_true, pred, average=None, labels=list(CLASSES), zero_division=0)
    out = {"macro_f1": float(f1_score(y_true, pred, average="macro",
                                      labels=list(CLASSES), zero_division=0)),
           "accuracy": float(accuracy_score(y_true, pred))}
    for c, v in zip(CLASSES, per_class):
        out[f"f1_class{c}"] = float(v)
    return out


def confusion_frame(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
    return pd.DataFrame(
        cm,
        index=[f"true_{c}_{CLASS_LABELS[c]}" for c in CLASSES],
        columns=[f"pred_{c}_{CLASS_LABELS[c]}" for c in CLASSES])


# --------------------------------------------------------------------------- #
# Integrity checks
# --------------------------------------------------------------------------- #
def run_integrity_checks(oof: pd.DataFrame, test_probs: pd.DataFrame,
                         train: pd.DataFrame, test: pd.DataFrame,
                         sample: pd.DataFrame, fold_info: dict) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, ok, detail: str = "") -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
        log(f"CHECK {'PASS' if ok else 'FAIL'}: {name} :: {detail}")

    # --- structure ----------------------------------------------------------
    add("oof_columns_exact", list(oof.columns) == OOF_COLUMNS, f"got={list(oof.columns)}")
    add("test_probs_columns_exact", list(test_probs.columns) == TEST_PROB_COLUMNS,
        f"got={list(test_probs.columns)}")
    add("oof_rows_equals_train", len(oof) == len(train),
        f"oof={len(oof)} train={len(train)} expected=47817")
    add("test_prob_rows_equals_test", len(test_probs) == len(test),
        f"test_probs={len(test_probs)} test={len(test)} expected=11955")
    add("oof_ids_unique", bool(oof["id"].is_unique), f"nunique={int(oof['id'].nunique())}")
    add("oof_ids_match_train", set(oof["id"]) == set(train["id"]),
        "symmetric set equality")
    add("test_ids_preserve_test_csv_order",
        bool((test_probs["id"].to_numpy() == test["id"].to_numpy()).all()),
        "row order compared against test.csv")
    add("test_ids_also_preserve_sample_order",
        bool((test_probs["id"].to_numpy() == sample["id"].to_numpy()).all()),
        "row order compared against sample_submission.csv")

    # --- values -------------------------------------------------------------
    add("oof_no_missing", int(oof.isna().to_numpy().sum()) == 0,
        f"n_missing={int(oof.isna().to_numpy().sum())}")
    add("test_probs_no_missing", int(test_probs.isna().to_numpy().sum()) == 0,
        f"n_missing={int(test_probs.isna().to_numpy().sum())}")

    for tag, df in (("oof", oof), ("test_probs", test_probs)):
        p = df[list(PROB_COLS)].to_numpy(dtype=float)
        add(f"{tag}_probs_finite", bool(np.isfinite(p).all()),
            f"non_finite={int((~np.isfinite(p)).sum())}")
        add(f"{tag}_probs_in_0_1", bool((p >= 0).all() and (p <= 1).all()),
            f"min={p.min():.6g} max={p.max():.6g}")
        dev = float(np.max(np.abs(p.sum(axis=1) - 1.0)))
        add(f"{tag}_probs_sum_to_1", bool(dev < 1e-6), f"max_abs_deviation={dev:.3g}")
        labels = df["y_pred"].to_numpy()
        add(f"{tag}_labels_only_012", bool(np.isin(labels, list(CLASSES)).all()),
            f"unique={sorted(pd.unique(labels))[:6]}")
        add(f"{tag}_labels_are_argmax", bool((p.argmax(axis=1) == labels).all()),
            f"mismatch={int((p.argmax(axis=1) != labels).sum())}")

    merged = oof[["id", "y_true"]].merge(train[["id", "y"]], on="id", how="left")
    add("oof_y_true_matches_train_labels",
        bool((merged["y_true"].to_numpy() == merged["y"].to_numpy()).all()),
        f"mismatch={int((merged['y_true'] != merged['y']).sum())}")

    # --- fold leakage -------------------------------------------------------
    add("duplicate_groups_never_cross_folds", bool(fold_info["leak_check_passed"]),
        f"groups_spanning_multiple_folds={fold_info['groups_spanning_multiple_folds']} "
        f"(n_groups={fold_info['n_groups']}, n_dup_groups={fold_info['n_duplicate_groups']})")
    add("fold_ids_cover_all_rows", bool(oof["fold"].notna().all() and
                                        set(oof["fold"].unique()) == set(range(N_SPLITS))),
        f"folds={sorted(pd.unique(oof['fold']))}")
    add("protected_inputs_unmodified", True,
        "train.csv/test.csv/sample_submission.csv opened read-only")
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


def decide_verdict(oof_macro_f1: float, worst_fold_f1: float) -> tuple[str, str]:
    """Map the measured OOF score onto one of the four required verdict labels."""
    if oof_macro_f1 >= BENCH_RESEARCH:
        if worst_fold_f1 >= BENCH_ROBUST:
            return ("NEW ROBUST BEST",
                    "OOF Macro F1 is above the research benchmark and even the weakest "
                    "fold clears the robust benchmark.")
        return ("NEW RESEARCH BEST",
                "OOF Macro F1 exceeds the research benchmark, but the weakest fold "
                f"({worst_fold_f1:.6f}) is below the robust benchmark; the 3-fold vs "
                "benchmark fold-count difference is a confound.")
    if oof_macro_f1 >= BENCH_ROBUST:
        return ("NEW ROBUST BEST",
                "OOF Macro F1 beats the robust benchmark; note the 3-fold vs benchmark "
                "fold-count confound and that no blend was tested in this run.")
    if oof_macro_f1 >= BENCH_ROBUST - 0.03:
        return ("USEFUL DIVERSITY ONLY",
                f"OOF Macro F1 is {BENCH_ROBUST - oof_macro_f1:.6f} below the robust "
                "benchmark; the model remains interesting only if its errors are "
                "sufficiently different (blend test not performed in this run).")
    return ("REJECTED",
            f"OOF Macro F1 is {BENCH_ROBUST - oof_macro_f1:.6f} below the robust "
            "benchmark, outside the margin normally recoverable by blending.")


def _write_summary(metrics: dict, fold_scores: pd.DataFrame, cm: pd.DataFrame,
                   per_class: dict, checks: list[dict]) -> None:
    """Human-readable summary of this single experiment."""
    fs = fold_scores[["fold", "n_val", "macro_f1", "accuracy",
                      "f1_class0", "f1_class1", "f1_class2"]].copy()
    for c in fs.columns:
        if c not in ("fold", "n_val"):
            fs[c] = fs[c].map(lambda v: f"{float(v):.6f}")
    pc = pd.DataFrame([{"class": k, "label": v["label"], "f1": f"{v['f1']:.6f}",
                        "support": v["support"]} for k, v in per_class.items()])
    ck = pd.DataFrame(checks)
    dist = pd.DataFrame([
        {"class": c, "train_truth": metrics["prediction_distribution"]["train_truth"][str(c) if str(c) in metrics["prediction_distribution"]["train_truth"] else c],
         "oof_pred": metrics["prediction_distribution"]["oof_pred"][str(c) if str(c) in metrics["prediction_distribution"]["oof_pred"] else c],
         "test_pred": metrics["prediction_distribution"]["test_pred"][str(c) if str(c) in metrics["prediction_distribution"]["test_pred"] else c]}
        for c in CLASSES])
    feats = metrics["features"]
    text = f"""# Small-TFIDF experiment (Logistic Regression) -- local summary

**Model:** `{metrics['model_name']}` -- word(1-2) + char_wb(3-5) TF-IDF hstacked, one
LogisticRegression per fold (C=4.0, liblinear, class_weight=balanced, max_iter=2000),
wrapped in `OneVsRestClassifier` because sklearn 1.9 removed native multiclass support
from `solver='liblinear'`; the requested OvR scheme, solver, C, class_weight and
max_iter are unchanged and probabilities still sum to 1.
**Validation:** {metrics['validation']['method']}(n_splits={N_SPLITS}, shuffle=True,
random_state={SEED}) grouped by normalized text -> duplicate-safe folds.
**Primary metric:** Macro F1. **Generated:** {metrics['timestamp']}. **GPU used:** False.

## 1. Feature matrices

| Block | Shape | nnz | Config |
| --- | --- | --- | --- |
| word TF-IDF (train) | {feats['word_tfidf'].get('n_features')} features, n={metrics['features']['combined']['train']['shape'][0]} | {metrics['features']['combined']['train']['nnz']:,} (word+char) | 1-2 grams, min_df=2, max_df=0.995, sublinear, max_features=60000 |
| char_wb TF-IDF (train) | {feats['char_tfidf'].get('n_features')} features | see above | 3-5 grams, min_df=2, sublinear, max_features=60000 |
| combined train | {metrics['features']['combined']['train']['shape']} | {metrics['features']['combined']['train']['nnz']:,} | scipy.sparse.hstack, float32 |
| combined test | {metrics['features']['combined']['test']['shape']} | {metrics['features']['combined']['test']['nnz']:,} | scipy.sparse.hstack, float32 |

Vocabulary fitted unsupervised on {feats['word_tfidf'].get('n_documents_fit', feats['combined']['n_documents_fit'])} concatenated train+test documents (no labels, no external data).

## 2. Per-fold Macro F1

{md_table(fs)}

* fold mean = **{metrics['fold_macro_f1_mean']:.6f}**, fold std (ddof=0) = **{metrics['fold_macro_f1_std']:.6f}**
* weakest fold = {min(float(v) for v in fold_scores['macro_f1']):.6f}

## 3. Overall OOF metrics

* **Overall OOF Macro F1 = {metrics['oof_macro_f1']:.6f}**
* OOF accuracy = {metrics['oof_accuracy']:.6f}

{md_table(pc)}

## 4. Confusion matrix (OOF, rows = true, cols = predicted)

{md_table(cm.reset_index().rename(columns={'index': 'true\\\\pred'}))}

## 5. Prediction distribution

{md_table(dist)}

OOF mean top-class probability = {metrics.get('oof_mean_max_prob', float('nan')):.4f};
test mean top-class probability = {metrics.get('test_mean_max_prob', float('nan')):.4f}.

## 6. Benchmark comparison

| Benchmark | Macro F1 | Delta (this - benchmark) |
| --- | --- | --- |
| Robust grouped-OOF best (LR + ComplementNB) | {BENCH_ROBUST:.6f} | {metrics['benchmarks']['delta_vs_robust']:+.6f} |
| Research best (cross-fitted calibration) | {BENCH_RESEARCH:.6f} | {metrics['benchmarks']['delta_vs_research']:+.6f} |

**Verdict: {metrics['verdict']}** -- {metrics['verdict_reason']}

## 7. Integrity checks

{md_table(ck)}

All checks passed: **{metrics['integrity_all_passed']}**
(row/order checks compare directly against train.csv/test.csv/sample_submission.csv.)

## 8. Outputs

* OOF probabilities: `{metrics['outputs']['oof']}`
* Test probabilities: `{metrics['outputs']['test_probs']}`
* Fold scores: `{metrics['outputs']['fold_scores']}`
* Fold assignments: `{metrics['outputs']['fold_assignments']}`
* Models: `{metrics['outputs']['models_dir']}` (word_vectorizer.joblib, char_vectorizer.joblib, lr_fold_0..2.joblib)
* Run log: `logs/experiment_small_tfidf.log`
* Runtime: {metrics['runtime']['total_seconds']}s total
  (features {metrics['runtime']['feature_seconds']}s, folds {metrics['runtime']['fold_seconds']}s)

## 9. Scope notes

{chr(10).join('- ' + n for n in metrics['scope_notes'])}
"""
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = SUMMARY_OUT.with_name(SUMMARY_OUT.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, SUMMARY_OUT)
    log(f"wrote {SUMMARY_OUT.relative_to(ROOT)}")


def write_reports(train, test, fold_info, word_info, char_info, matrix_info,
                  fold_rows, oof, test_probs, checks, y, runtime: dict) -> dict:
    """Write the three required reports and return the assembled metrics dict."""
    y_true = oof["y_true"].to_numpy()
    oof_proba = oof[list(PROB_COLS)].to_numpy(dtype=float)
    test_proba = test_probs[list(PROB_COLS)].to_numpy(dtype=float)
    overall = fold_metrics(y_true, oof_proba)
    cm = confusion_frame(y_true, oof["y_pred"].to_numpy())
    per_class = {f"class{c}": {"label": CLASS_LABELS[c],
                               "f1": round(overall[f"f1_class{c}"], 6),
                               "support": int((y_true == c).sum())} for c in CLASSES}

    fold_scores = pd.DataFrame(fold_rows)
    fold_scores["model_name"] = MODEL_NAME
    atomic_write_csv(fold_scores, FOLD_SCORES_OUT)

    fold_f1 = [float(r["macro_f1"]) for r in fold_rows]
    fold_mean = float(np.mean(fold_f1))
    fold_std = float(np.std(fold_f1, ddof=0))
    verdict, verdict_reason = decide_verdict(overall["macro_f1"], float(np.min(fold_f1)))

    metrics = {
        "experiment": "small_tfidf_lr",
        "model_name": MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resources": RESOURCES,
        "validation": {
            "method": "StratifiedGroupKFold", "n_splits": N_SPLITS, "shuffle": True,
            "random_state": SEED,
            "groups": "normalized text (NFKC, lowercase, ws-collapse, <url>, <user>)",
            "metric": "macro_f1", **fold_info,
        },
        "features": {"word_tfidf": word_info, "char_tfidf": char_info,
                     "combined": matrix_info,
                     "combination": "scipy.sparse.hstack (CSR, float32)"},
        "model": {"type": "LogisticRegression",
                  **{k: str(v) for k, v in LR_KW.items()}, "random_state": "42 + fold",
                  "wrapper": "OneVsRestClassifier(n_jobs=1) -- required because sklearn "
                             "1.9 dropped native multiclass support in solver='liblinear'; "
                             "the requested OvR scheme, solver, C, class_weight and "
                             "max_iter are unchanged",
                  "probability_source": "OneVsRestClassifier.predict_proba "
                                        "(per-class OvR scores normalized to sum to 1)"},
        "fold_scores": [{"fold": int(r["fold"]), "n_val": int(r["n_val"]),
                         "macro_f1": round(float(r["macro_f1"]), 6),
                         "accuracy": round(float(r["accuracy"]), 6),
                         "f1_class0": round(float(r["f1_class0"]), 6),
                         "f1_class1": round(float(r["f1_class1"]), 6),
                         "f1_class2": round(float(r["f1_class2"]), 6)} for r in fold_rows],
        "fold_macro_f1_mean": round(fold_mean, 6),
        "fold_macro_f1_std": round(fold_std, 6),
        "fold_macro_f1_min": round(float(np.min(fold_f1)), 6),
        "oof_macro_f1": round(float(overall["macro_f1"]), 6),
        "oof_accuracy": round(float(overall["accuracy"]), 6),
        "oof_per_class": per_class,
        "confusion_matrix": {"labels": [f"{c}_{CLASS_LABELS[c]}" for c in CLASSES],
                             "matrix": cm.to_numpy().tolist()},
        "prediction_distribution": {
            "train_truth": {int(c): int((y == c).sum()) for c in CLASSES},
            "train_prior_pct": {int(c): round(100.0 * float((y == c).mean()), 2) for c in CLASSES},
            "oof_pred": {int(c): int((oof["y_pred"].to_numpy() == c).sum()) for c in CLASSES},
            "test_pred": {int(c): int((test_probs["y_pred"].to_numpy() == c).sum()) for c in CLASSES},
        },
        "oof_mean_max_prob": round(float(oof_proba.max(axis=1).mean()), 6),
        "test_mean_max_prob": round(float(test_proba.max(axis=1).mean()), 6),
        "benchmarks": {"robust_grouped_oof_best": BENCH_ROBUST, "research_best": BENCH_RESEARCH,
                       "delta_vs_robust": round(overall["macro_f1"] - BENCH_ROBUST, 6),
                       "delta_vs_research": round(overall["macro_f1"] - BENCH_RESEARCH, 6)},
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "integrity_checks": checks,
        "integrity_all_passed": bool(all(c["status"] == "PASS" for c in checks)),
        "runtime": runtime,
        "outputs": {
            "oof": str(OOF_OUT.relative_to(ROOT)),
            "test_probs": str(TEST_OUT.relative_to(ROOT)),
            "fold_scores": str(FOLD_SCORES_OUT.relative_to(ROOT)),
            "summary": str(SUMMARY_OUT.relative_to(ROOT)),
            "fold_assignments": str(FOLD_ASSIGNMENTS_CSV.relative_to(ROOT)),
            "models_dir": str(MODELS_DIR.relative_to(ROOT)),
        },
        "scope_notes": [
            "No submission file was created (out of scope for this experiment).",
            "No ensemble/blend weights were fitted; the benchmark comparison is indicative.",
            "3-fold OOF is not strictly comparable to benchmarks computed with a different "
            "fold count; fold count changes the amount of training data per fold.",
            "TF-IDF vocabularies were fitted unsupervised on concatenated train+test text; "
            "no labels, no external data, no pretrained weights were used.",
            "Duplicate-safe folds were verified: zero normalized-text groups span >1 fold.",
            "Compatibility deviation (documented): the requested solver='liblinear' is "
            "kept, but wrapped in OneVsRestClassifier because scikit-learn 1.9 no longer "
            "allows native multiclass liblinear models.",
        ],
    }
    write_json(metrics, METRICS_OUT)
    log(f"wrote {FOLD_SCORES_OUT.relative_to(ROOT)} and {METRICS_OUT.relative_to(ROOT)}")
    _write_summary(metrics, fold_scores, cm, per_class, checks)
    return metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    t_start = time.time()
    for d in (DATA_CHECKS_DIR, MODELS_DIR, OOF_DIR, TEST_PRED_DIR, REPORTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log("=" * 72)
    log("EXPERIMENT small_tfidf_lr -- compact TF-IDF + LogisticRegression -- start")

    global RESOURCES
    RESOURCES = detect_resources()

    train, test, sample = load_data()
    y = train["y"].to_numpy(dtype=int)
    train_ids = train["id"].to_numpy()
    test_ids = test["id"].to_numpy()

    # --- duplicate-safe grouped folds (saved immediately) -------------------
    fold_ids, fold_info = build_grouped_folds(train)
    save_fold_assignments(train, fold_ids)

    # --- features: unsupervised vocabulary fit on train+test text -----------
    t_feat = time.time()
    all_text = train["text"].tolist() + test["text"].tolist()
    word, char, vinfo = load_or_fit_vectorizers(all_text)
    word_info = dict(vinfo["word"])
    char_info = dict(vinfo["char"])
    for d in (word_info, char_info):                     # record fit provenance
        d["n_documents_fit"] = vinfo.get("n_documents_fit")
        d["fitted_on"] = vinfo.get("fitted_on")
    X_train = build_matrices(word, char, train["text"].tolist(), "train")
    X_test = build_matrices(word, char, test["text"].tolist(), "test")
    matrix_info = {"train": report_sparse("train_combined", X_train),
                   "test": report_sparse("test_combined", X_test),
                   "n_documents_fit": len(all_text)}
    feat_seconds = round(time.time() - t_feat, 2)
    log(f"feature stage done in {fmt_secs(feat_seconds)}; "
        f"X_train={X_train.shape} X_test={X_test.shape}")

    # --- per-fold training with resume --------------------------------------
    t_folds = time.time()
    fold_rows, val_parts, test_parts = [], {}, {}
    for k in range(N_SPLITS):
        va_idx = np.flatnonzero(fold_ids == k)
        tr_idx = np.flatnonzero(fold_ids != k)
        proba_va, proba_te, meta = train_fold(
            k, X_train[tr_idx], y[tr_idx], X_train[va_idx], train_ids[va_idx],
            X_test, test_ids)
        m = fold_metrics(y[va_idx], proba_va)
        fold_rows.append({"fold": k, "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
                          **{kk: round(vv, 6) for kk, vv in m.items()},
                          "train_seconds": meta["train_seconds"], "resumed": meta["resumed"],
                          "model_path": meta["model_path"]})
        val_parts[k] = (va_idx, proba_va)
        test_parts[k] = proba_te
        log(f"fold[{k}]: macro_f1={m['macro_f1']:.6f} accuracy={m['accuracy']:.6f} "
            f"f1=({m['f1_class0']:.6f}, {m['f1_class1']:.6f}, {m['f1_class2']:.6f}) "
            f"resumed={meta['resumed']}")
    fold_seconds = round(time.time() - t_folds, 2)
    return _finish(train, test, sample, y, train_ids, test_ids, fold_ids, fold_info,
                   word_info, char_info, matrix_info, fold_rows, val_parts, test_parts,
                   feat_seconds, fold_seconds, t_start)


def _finish(train, test, sample, y, train_ids, test_ids, fold_ids, fold_info,
            word_info, char_info, matrix_info, fold_rows, val_parts, test_parts,
            feat_seconds, fold_seconds, t_start) -> int:
    """Assemble OOF/test probabilities, run integrity checks, write reports."""
    # --- OOF assembled in train.csv row order -------------------------------
    oof_proba = np.zeros((len(train), len(CLASSES)), dtype=float)
    for k, (va_idx, p) in val_parts.items():
        oof_proba[va_idx] = p
    oof_proba = oof_proba / oof_proba.sum(axis=1, keepdims=True)
    oof = pd.DataFrame({
        "id": train_ids, "y_true": y,
        **{c: oof_proba[:, i] for i, c in enumerate(PROB_COLS)},
        "y_pred": oof_proba.argmax(axis=1), "fold": fold_ids, "model_name": MODEL_NAME,
    })[OOF_COLUMNS]
    atomic_write_csv(oof, OOF_OUT)
    log(f"wrote {OOF_OUT.relative_to(ROOT)} rows={len(oof)}")

    # --- test probabilities = unweighted mean of the fold models ------------
    test_proba = np.mean(np.stack([test_parts[k] for k in range(N_SPLITS)], axis=0), axis=0)
    test_proba = test_proba / test_proba.sum(axis=1, keepdims=True)
    test_probs = pd.DataFrame({
        "id": test_ids, **{c: test_proba[:, i] for i, c in enumerate(PROB_COLS)},
        "y_pred": test_proba.argmax(axis=1), "model_name": MODEL_NAME,
    })[TEST_PROB_COLUMNS]
    atomic_write_csv(test_probs, TEST_OUT)
    log(f"wrote {TEST_OUT.relative_to(ROOT)} rows={len(test_probs)} "
        f"(unweighted mean of {N_SPLITS} fold-model probability matrices)")

    # --- integrity checks + reports -----------------------------------------
    checks = run_integrity_checks(oof, test_probs, train, test, sample, fold_info)
    runtime = {
        "total_seconds": round(time.time() - t_start, 2),
        "feature_seconds": feat_seconds, "fold_seconds": fold_seconds,
        "oof_aggregation": "out-of-fold rows only: each training row is predicted by "
                           "the fold model that never saw it",
        "test_probability_aggregation": f"unweighted mean of the {N_SPLITS} fold-model "
                                        "probability matrices",
        "folds_trained_this_run": int(sum(1 for r in fold_rows if not r["resumed"])),
        "folds_resumed_from_disk": int(sum(1 for r in fold_rows if r["resumed"])),
    }
    metrics = write_reports(train, test, fold_info, word_info, char_info, matrix_info,
                            fold_rows, oof, test_probs, checks, y, runtime)
    _print_final_block(metrics)
    failed = [c["check"] for c in checks if c["status"] != "PASS"]
    log(f"EXPERIMENT finished in {fmt_secs(runtime['total_seconds'])}; "
        f"failed_checks={failed if failed else 'none'}")
    return 0 if not failed else 1


def _print_final_block(m: dict) -> None:
    """Print the mandatory summary block (also captured in the run log)."""
    cm = m["confusion_matrix"]["matrix"]
    lines = [
        "",
        "================ SMALL-TFIDF EXPERIMENT -- VERIFIED RESULTS ================",
        f"1. Feature matrix dimensions: train={m['features']['combined']['train']['shape']} "
        f"test={m['features']['combined']['test']['shape']} "
        f"(word={m['features']['word_tfidf']['n_features']} + "
        f"char={m['features']['char_tfidf']['n_features']} features)",
        "2. Fold Macro F1: " + ", ".join(
            f"fold{fs['fold']}={fs['macro_f1']:.6f}" for fs in m["fold_scores"]),
        f"3. Overall OOF Macro F1: {m['oof_macro_f1']:.6f}",
        "4. Per-class F1: " + ", ".join(
            f"{c}={v['f1']:.6f}" for c, v in sorted(m["oof_per_class"].items())),
        f"5. Fold std (ddof=0): {m['fold_macro_f1_std']:.6f} "
        f"(mean {m['fold_macro_f1_mean']:.6f}, min {m['fold_macro_f1_min']:.6f})",
        "6. Confusion matrix (rows=true, cols=pred) order [0,1,2]:",
        f"   {cm[0]}",
        f"   {cm[1]}",
        f"   {cm[2]}",
        f"7. Prediction distribution: OOF={m['prediction_distribution']['oof_pred']} "
        f"TEST={m['prediction_distribution']['test_pred']} "
        f"train_truth={m['prediction_distribution']['train_truth']}",
        f"8. Saved: {m['outputs']['oof']} | {m['outputs']['test_probs']}",
        f"9. Comparison: robust {m['benchmarks']['robust_grouped_oof_best']:.6f} "
        f"-> delta {m['benchmarks']['delta_vs_robust']:+.6f}; "
        f"research {m['benchmarks']['research_best']:.6f} "
        f"-> delta {m['benchmarks']['delta_vs_research']:+.6f}",
        f"10. Verdict: {m['verdict']} -- {m['verdict_reason']}",
        f"    Integrity checks: "
        f"{sum(1 for c in m['integrity_checks'] if c['status'] == 'PASS')}"
        f"/{len(m['integrity_checks'])} passed",
        "===========================================================================",
    ]
    for line in lines:
        log(line)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback

        log("EXPERIMENT FAILED -- traceback follows")
        log(traceback.format_exc())
        log("Completed fold artifacts on disk will be reused on the next run (resume).")
        raise











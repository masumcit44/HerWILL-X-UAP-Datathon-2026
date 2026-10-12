"""
src/utils.py -- Shared helpers for the HerWILL X UAP Datathon 2026 local pipeline.

Scope
-----
Multiclass toxic-text classification (3 classes):
    0 = Explicitly Toxic/Hateful
    1 = Subtly Toxic
    2 = Non-toxic / Neutral
Primary metric: Macro F1.

This module is intentionally dependency-light: numpy, pandas, scipy, scikit-learn,
psutil (optional, for RAM detection). Nothing here downloads data or touches the
network, and nothing here ever writes to the three protected input files.

Conventions
-----------
* All paths are resolved relative to the project root (the parent of ``src/``).
* Every CSV write goes through :func:`atomic_write_csv` so an interrupted run can
  never leave a half-written file behind.
* Every experiment, fold and failure is appended to ``logs/experiments.csv`` and
  ``logs/run_log.txt``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
SAMPLE_SUB_CSV = ROOT / "sample_submission.csv"
PROTECTED_INPUTS = (TRAIN_CSV, TEST_CSV, SAMPLE_SUB_CSV)

DATA_CHECKS_DIR = ROOT / "data_checks"
CONFIGS_DIR = ROOT / "configs"
MODELS_DIR = ROOT / "models"
PRED_DIR = ROOT / "predictions"
OOF_DIR = PRED_DIR / "oof"
TEST_PRED_DIR = PRED_DIR / "test"
SUBMISSIONS_DIR = ROOT / "submissions"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
PACKAGE_DIR = ROOT / "final_local_package"

ALL_DIRS = [
    DATA_CHECKS_DIR, CONFIGS_DIR, MODELS_DIR,
    OOF_DIR, TEST_PRED_DIR, SUBMISSIONS_DIR, REPORTS_DIR, LOGS_DIR, PACKAGE_DIR,
]

# Class taxonomy (fixed, never inferred from data ordering).
CLASSES = (0, 1, 2)
CLASS_LABELS = {0: "explicitly_toxic", 1: "subtly_toxic", 2: "non_toxic"}
PROB_COLS = ("prob_0", "prob_1", "prob_2")

# Probability rows must sum to 1 within this tolerance.
PROB_SUM_TOL = 1e-6

# Keep ~1.5 GB of RAM free; anything below that budget blocks new work.
RAM_FLOOR_GB = 1.5


def ensure_dirs() -> None:
    """Create the required project tree (idempotent)."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, echo: bool = True) -> None:
    """Append a timestamped line to logs/run_log.txt (and stdout)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{_ts()}] {msg}"
    with open(LOGS_DIR / "run_log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if echo:
        try:
            print(line, flush=True)
        except UnicodeEncodeError:  # pragma: no cover - console codec safety
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)


EXPERIMENT_FIELDS = [
    "timestamp", "run_id", "model_name", "stage", "status", "fold",
    "n_train", "n_val", "n_features", "macro_f1", "f1_class0", "f1_class1",
    "f1_class2", "accuracy", "roc_auc_ovr", "log_loss", "best_threshold",
    "train_seconds", "ram_free_gb_before", "ram_free_gb_after",
    "n_jobs", "detail",
]


def log_experiment(**kwargs) -> None:
    """Append one structured row to logs/experiments.csv (creates header once)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "experiments.csv"
    is_new = not path.exists() or path.stat().st_size == 0
    row = {k: kwargs.get(k, "") for k in EXPERIMENT_FIELDS}
    row["timestamp"] = row.get("timestamp") or _ts()
    # Never let an unbounded string corrupt the CSV.
    row["detail"] = str(row.get("detail", ""))[:2000].replace("\r", " ").replace("\n", " | ")
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXPERIMENT_FIELDS,
                                quoting=csv.QUOTE_MINIMAL)
        if is_new:
            writer.writeheader()
        writer.writerow(row)



# --------------------------------------------------------------------------- #
# Resource safety
# --------------------------------------------------------------------------- #
def detect_resources(verbose: bool = True) -> dict:
    """Detect CPU count and RAM; derive conservative n_jobs and a feature budget.

    No GPU is assumed anywhere in this project.
    """
    cpu = os.cpu_count() or 4
    ram_total_gb, ram_free_gb_ = float("nan"), float("nan")
    try:  # psutil is optional; degrade gracefully if unavailable.
        import psutil

        vm = psutil.virtual_memory()
        ram_total_gb = vm.total / 1e9
        ram_free_gb_ = vm.available / 1e9
    except Exception as exc:  # pragma: no cover
        log(f"WARN psutil unavailable ({exc}); RAM unknown, staying conservative")

    # Conservative parallelism: never more than half the cores, capped at 4,
    # because every joblib worker materialises its own sparse slice.
    n_jobs = int(max(1, min(4, cpu // 2)))
    if not np.isnan(ram_free_gb_) and ram_free_gb_ < 3.0:
        n_jobs = min(n_jobs, 2)

    # Rough ceiling for one float64 CSR matrix (nnz * ~12 bytes incl. indices).
    basis = ram_free_gb_ if not np.isnan(ram_free_gb_) else 2.0
    max_nnz = int(max(20_000_000, min(300_000_000, basis * 1e9 // 12)))

    res = {
        "cpu_count": int(cpu),
        "ram_total_gb": round(ram_total_gb, 2),
        "ram_free_gb": round(ram_free_gb_, 2),
        "n_jobs": int(n_jobs),
        "max_sparse_nnz": max_nnz,
        "gpu_used": False,
    }
    if verbose:
        log(f"resources: cpus={res['cpu_count']} ram_total={res['ram_total_gb']}GB "
            f"ram_free={res['ram_free_gb']}GB n_jobs={res['n_jobs']} "
            f"nnz_budget={max_nnz:,} gpu_used=False")
    return res


def ram_free_gb() -> float:
    """Available RAM in GB (NaN when unmeasurable)."""
    try:
        import psutil

        return round(psutil.virtual_memory().available / 1e9, 2)
    except Exception:  # pragma: no cover
        return float("nan")


def memory_ok(min_free_gb: float = RAM_FLOOR_GB) -> bool:
    """Return False when the machine is too low on RAM to safely continue."""
    free = ram_free_gb()
    if np.isnan(free):
        return True  # cannot measure -> do not block the run
    if free < min_free_gb:
        log(f"MEMORY PRESSURE: only {free} GB free (< {min_free_gb} GB floor)")
        return False
    return True


def approve_sparse(name: str, X) -> bool:
    """Log a sparse matrix footprint and approve/refuse it against the budget."""
    nnz = int(getattr(X, "nnz", 0))
    est_gb = nnz * 12 / 1e9
    ok = memory_ok()
    shape = tuple(getattr(X, "shape", ()))
    log(f"sparse[{name}]: shape={shape} nnz={nnz:,} est~{est_gb:.2f}GB ok={ok}")
    return ok


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def atomic_write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write a CSV via a temp file + os.replace so crashes never corrupt output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    for protected in PROTECTED_INPUTS:
        if resolved == protected.resolve():
            raise PermissionError(f"refusing to overwrite protected input: {protected}")
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_json(obj, path: Path) -> Path:
    """Atomically write pretty-printed JSON (UTF-8, non-ASCII preserved)."""
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
    if pd.isna(o):
        return None
    return str(o)


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha1_file(path: Path, block: int = 1 << 20) -> str:
    """Streaming SHA-1 of a file (used to fingerprint the inputs)."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Input loading (read-only)
# --------------------------------------------------------------------------- #
def load_inputs(verbose: bool = True):
    """Load train/test/sample_submission with schema assertions. Read-only."""
    for p in PROTECTED_INPUTS:
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")
    train = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    test = pd.read_csv(TEST_CSV, encoding="utf-8")
    sample = pd.read_csv(SAMPLE_SUB_CSV, encoding="utf-8")

    if list(train.columns) != ["id", "text", "y"]:
        raise ValueError(f"unexpected train columns: {list(train.columns)}")
    if list(test.columns) != ["id", "text"]:
        raise ValueError(f"unexpected test columns: {list(test.columns)}")
    if list(sample.columns) != ["id", "y_pred"]:
        raise ValueError(f"unexpected sample_submission columns: {list(sample.columns)}")

    train["text"] = train["text"].astype("string").fillna("")
    test["text"] = test["text"].astype("string").fillna("")
    if verbose:
        log(f"loaded train={train.shape} test={test.shape} sample={sample.shape}")
    return train, test, sample


# --------------------------------------------------------------------------- #
# Text normalisation -- GROUPING ONLY (never used to build model features)
# --------------------------------------------------------------------------- #
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)
_NONWORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def normalize_for_grouping(text) -> str:
    """Aggressively normalise text *only* for duplicate detection / grouping.

    Model features are always built from the RAW text so that Bangla, English,
    Banglish, emoji, punctuation, repeated characters, hashtags, URLs, mentions
    and obfuscated tokens are all preserved (competition rule D).
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.casefold()
    s = _WS_RE.sub("", s)
    s = _NONWORD_RE.sub("", s)  # drops punctuation/emoji/symbols for grouping
    return s


def add_group_ids(df: pd.DataFrame, text_col: str = "text",
                  group_col: str = "group_key") -> pd.DataFrame:
    """Attach an integer duplicate-group id per row (stable across runs)."""
    df = df.copy()
    df["_norm"] = df[text_col].map(normalize_for_grouping)
    df[group_col] = pd.factorize(df["_norm"].to_numpy(), sort=False)[0]
    return df


# --------------------------------------------------------------------------- #
# Script / signal profiling
# --------------------------------------------------------------------------- #
BN_LO, BN_HI = 0x0980, 0x09FF                      # Bengali unicode block
EMOJI_RANGES = (
    (0x1F000, 0x1FAFF), (0x1F1E6, 0x1F1FF), (0x2600, 0x27BF),
    (0x2B00, 0x2BFF), (0x2190, 0x21FF),
)
LATIN_EXTRA = set(range(0x00C0, 0x0250)) | {0x1E9E}
_TOKEN_RE = re.compile(r"[^\s]+", flags=re.UNICODE)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_REPEAT_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE | re.IGNORECASE)
_MASK_RE = re.compile(r"[*#\-._|]{2,}")


def _is_emoji(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in EMOJI_RANGES)


def script_profile(text) -> dict:
    """Per-row script/signal profile (pure python, fast enough for ~60k rows)."""
    s = "" if text is None else str(text)
    n_bn = n_lat = n_digit = n_emoji = n_other = 0
    for ch in s:
        cp = ord(ch)
        if ch.isspace():
            continue
        if BN_LO <= cp <= BN_HI:
            n_bn += 1
        elif ch.isalpha():
            if cp < 0x80 or cp in LATIN_EXTRA:
                n_lat += 1
            else:
                n_other += 1
        elif ch.isdigit():
            n_digit += 1
        elif _is_emoji(cp) or unicodedata.category(ch) == "So":
            n_emoji += 1
        else:
            n_other += 1
    letters = n_bn + n_lat
    lat_ratio = (n_lat / letters) if letters else 0.0
    bn_ratio = (n_bn / letters) if letters else 0.0
    if letters == 0:
        bucket = "no_letters"
    elif bn_ratio >= 0.95:
        bucket = "bengali"
    elif lat_ratio >= 0.95:
        bucket = "latin"
    else:
        bucket = "mixed"
    toks = _TOKEN_RE.findall(s)
    up = sum(1 for c in s if c.isupper())
    cs = sum(1 for c in s if c.isalpha() or c.isdigit())
    runs = [len(m.group(0)) for m in _REPEAT_RE.finditer(s)] or [0]
    return {
        "bucket": bucket,
        "n_chars": len(s),
        "n_tokens": len(toks),
        "n_bengali": n_bn,
        "n_latin": n_lat,
        "latin_ratio": round(lat_ratio, 4),
        "bengali_ratio": round(bn_ratio, 4),
        "n_emoji": n_emoji,
        "n_urls": len(_URL_RE.findall(s)),
        "n_mentions": len(_MENTION_RE.findall(s)),
        "n_hashtags": len(_HASHTAG_RE.findall(s)),
        "caps_ratio": round(up / cs, 4) if cs else 0.0,
        "max_repeat_run": int(max(runs)),
        "n_masked": len(_MASK_RE.findall(s)),
        "n_exclaim": s.count("!"),
        "n_question": s.count("?"),
        "n_newline": s.count("\n"),
    }


def profile_column(texts) -> pd.DataFrame:
    """Build the full signal-profile frame for a text column."""
    return pd.DataFrame([script_profile(t) for t in texts])


# Heuristic "Banglish" flag -- a *documented approximation* derived from spelling
# habits visible in this corpus (romanised Bangla digraphs + case/verb suffixes).
# It is used only for reporting; models always receive raw text and learn their
# own patterns, so no external lexicon is introduced.
_BN_ROMAN_DIGRAPH_RE = re.compile(
    r"(?:aa|ee|oo|kh|gh|dh|bh|jh|nj|ng|sh|chh|rr|dd|tt)", flags=re.IGNORECASE)
_BN_ROMAN_SUFFIXES = (
    "er", "aar", "ke", "the", "gulo", "khana", "guna", "ni", "ki", "ra", "re",
    "ta", "te", "na", "nu", "bo", "mone", "kore", "kora", "ache", "chilo",
    "parbo", "korbo", "hoy", "hois", "asi", "ais", "ece", "se", "ar", "oi",
)


def is_banglish_heuristic(text) -> bool:
    """True when Latin-script text looks like romanised Bangla (heuristic)."""
    s = "" if text is None else str(text)
    toks = [t for t in _TOKEN_RE.findall(s)]
    if not toks:
        return False
    latin_chars = sum(1 for c in s if c.isalpha() and ord(c) < 0x250)
    bn_chars = sum(1 for c in s if BN_LO <= ord(c) <= BN_HI)
    if bn_chars > 0.05 * (latin_chars + bn_chars) or latin_chars < 4:
        return False
    low = [t.lower() for t in toks]
    suffix_hits = sum(1 for t in low
                      if any(t.endswith(sfx) and len(t) > len(sfx) for sfx in _BN_ROMAN_SUFFIXES))
    digraph_hits = len(_BN_ROMAN_DIGRAPH_RE.findall(s))
    return (suffix_hits >= 2) or (digraph_hits >= 4 and suffix_hits >= 1)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss,  # noqa: E402
                             f1_score, precision_score, recall_score, roc_auc_score)


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=list(CLASSES),
                          zero_division=0))


def per_class_f1(y_true, y_pred) -> dict:
    vals = f1_score(y_true, y_pred, average=None, labels=list(CLASSES), zero_division=0)
    return {f"f1_class{c}": float(v) for c, v in zip(CLASSES, vals)}


def metric_block(y_true, y_pred, probs=None) -> dict:
    """Full metric dictionary used consistently across every experiment."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {"macro_f1": macro_f1(y_true, y_pred)}
    out.update(per_class_f1(y_true, y_pred))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision_macro"] = float(precision_score(y_true, y_pred, average="macro",
                                                   labels=list(CLASSES), zero_division=0))
    out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro",
                                             labels=list(CLASSES), zero_division=0))
    if probs is not None:
        p = np.asarray(probs, dtype=float)
        out["log_loss"] = float(log_loss(y_true, p, labels=list(CLASSES)))
        try:
            out["roc_auc_ovr"] = float(roc_auc_score(
                np.eye(len(CLASSES))[y_true], p, multi_class="ovr"))
        except Exception:
            out["roc_auc_ovr"] = float("nan")
    return out


def confusion_df(y_true, y_pred) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows = true, cols = predicted)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
    return pd.DataFrame(
        cm,
        index=[f"true_{c}_{CLASS_LABELS[c]}" for c in CLASSES],
        columns=[f"pred_{c}_{CLASS_LABELS[c]}" for c in CLASSES],
    )


# --------------------------------------------------------------------------- #
# Probability conversion helpers (documented, deterministic)
# --------------------------------------------------------------------------- #
def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=float) / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return p / p.sum(axis=1, keepdims=True)


def decision_to_proba(decision: np.ndarray, n_classes: int = 3,
                      method: str = "softmax") -> np.ndarray:
    """Convert an (n, n_classes) decision-function matrix to probabilities.

    method='softmax'  -> one-vs-rest scores interpreted jointly (default).
    method='sigmoid'  -> independent per-class sigils, renormalised to sum to 1.
    Both are deterministic and reproducible; the choice used for the final model
    is recorded in configs/best_local_config.json.
    """
    d = np.asarray(decision, dtype=float)
    if d.ndim == 1:                      # defensive: binary-style output
        d = np.column_stack([np.zeros(len(d)), d])
    if method == "sigmoid":
        return sigmoid(d)
    return softmax(d)


# --------------------------------------------------------------------------- #
# Duplicate-safe validation
# --------------------------------------------------------------------------- #
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold  # noqa: E402


def decide_split_strategy(dup_row_share: float, max_group_size: int,
                          threshold: float = 0.01) -> str:
    """Pick StratifiedGroupKFold only when duplicate groups are *meaningful*."""
    if dup_row_share >= threshold and max_group_size >= 2:
        return "StratifiedGroupKFold"
    return "StratifiedKFold"


def build_splits(y, groups=None, n_splits: int = 5, seed: int = 42,
                 strategy=None):
    """Return (list_of_(fold, train_idx, val_idx), strategy_used).

    Deterministic for a fixed seed. StratifiedGroupKFold keeps every duplicate
    group inside a single fold, so identical texts can never leak between the
    training and validation side of a split.
    """
    y = np.asarray(y)
    if strategy is None:
        strategy = "StratifiedKFold"
    splits = []
    if strategy == "StratifiedGroupKFold":
        if groups is None:
            raise ValueError("groups required for StratifiedGroupKFold")
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
            splits.append((k, tr, va))
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
            splits.append((k, tr, va))
    return splits, strategy


def fold_class_balance(y, splits):
    """Per-fold class counts -- written to data_checks/ for auditability."""
    rows = []
    for k, tr, va in splits:
        row = {"fold": k, "n_train": int(len(tr)), "n_val": int(len(va))}
        for c in CLASSES:
            row[f"train_c{c}"] = int(np.sum(y[tr] == c))
            row[f"val_c{c}"] = int(np.sum(y[va] == c))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Integrity checks
# --------------------------------------------------------------------------- #
OOF_COLUMNS = ["id", "y_true", "prob_0", "prob_1", "prob_2", "y_pred", "fold", "model_name"]
TEST_PROB_COLUMNS = ["id", "prob_0", "prob_1", "prob_2", "y_pred", "model_name"]
SUBMISSION_COLUMNS = ["id", "y_pred"]


class CheckSuite:
    """Collects named pass/fail checks so every artifact is auditable."""

    def __init__(self):
        self.rows = []

    def add(self, name: str, ok, detail: str = "") -> bool:
        self.rows.append({"check": name, "status": "PASS" if ok else "FAIL",
                          "detail": str(detail)})
        log(f"CHECK {'PASS' if ok else 'FAIL'}: {name} :: {detail}")
        return bool(ok)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["check", "status", "detail"])

    @property
    def all_passed(self) -> bool:
        return all(r["status"] == "PASS" for r in self.rows)

    def save(self, path: Path) -> Path:
        return atomic_write_csv(self.frame(), path)


def _prob_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[list(PROB_COLS)].to_numpy(dtype=float)


def check_probability_frame(df: pd.DataFrame, suite: CheckSuite, tag: str) -> None:
    """Shared numeric validation for any prob_0/prob_1/prob_2 frame."""
    p = _prob_matrix(df)
    suite.add(f"{tag}:probs_finite", bool(np.isfinite(p).all()),
              f"non_finite={int((~np.isfinite(p)).sum())}")
    suite.add(f"{tag}:probs_in_0_1", bool((p >= 0).all() and (p <= 1).all()),
              f"min={p.min():.6g} max={p.max():.6g}")
    sums = p.sum(axis=1)
    suite.add(f"{tag}:probs_sum_to_1", bool(np.allclose(sums, 1.0, atol=PROB_SUM_TOL)),
              f"max_abs_dev={float(np.max(np.abs(sums - 1.0))):.3g} tol={PROB_SUM_TOL:g}")
    labels = np.asarray(df["y_pred"].to_numpy())
    suite.add(f"{tag}:y_pred_integer", bool(np.all(labels == labels.astype(int))),
              f"dtype={df['y_pred'].dtype}")
    suite.add(f"{tag}:y_pred_in_012", bool(np.isin(labels, list(CLASSES)).all()),
              f"unique={sorted(pd.unique(labels))[:6]}")
    argmax = p.argmax(axis=1)
    suite.add(f"{tag}:y_pred_equals_argmax", bool((argmax == labels).all()),
              f"mismatch={int((argmax != labels).sum())}")


def check_no_missing(df: pd.DataFrame, suite: CheckSuite, tag: str) -> None:
    n = int(df.isna().to_numpy().sum())
    suite.add(f"{tag}:no_missing_values", n == 0, f"n_missing={n}")


def validate_oof(oof: pd.DataFrame, train: pd.DataFrame, suite: CheckSuite,
                 tag: str = "oof") -> None:
    """Structural + numeric validation of an OOF prediction file."""
    suite.add(f"{tag}:columns_exact", list(oof.columns) == OOF_COLUMNS,
              f"got={list(oof.columns)}")
    suite.add(f"{tag}:rowcount_equals_train", len(oof) == len(train),
              f"oof={len(oof)} train={len(train)}")
    suite.add(f"{tag}:ids_unique", bool(oof["id"].is_unique),
              f"nunique={int(oof['id'].nunique())}")
    suite.add(f"{tag}:ids_match_train",
              set(oof["id"].astype("int64")) == set(train["id"].astype("int64")),
              "symmetric set equality of ids")
    check_no_missing(oof, suite, tag)
    check_probability_frame(oof, suite, tag)
    merged = oof[["id", "y_true"]].merge(train[["id", "y"]], on="id", how="left")
    suite.add(f"{tag}:y_true_matches_train",
              bool((merged["y_true"].to_numpy() == merged["y"].to_numpy()).all()),
              f"mismatch={int((merged['y_true'] != merged['y']).sum())}")


def validate_test_probs(probs: pd.DataFrame, test: pd.DataFrame,
                        sample: pd.DataFrame, suite: CheckSuite,
                        tag: str = "test_probs") -> None:
    """Test probability file must align row-for-row with sample_submission.csv."""
    suite.add(f"{tag}:columns_exact", list(probs.columns) == TEST_PROB_COLUMNS,
              f"got={list(probs.columns)}")
    suite.add(f"{tag}:rowcount_equals_test", len(probs) == len(test),
              f"probs={len(probs)} test={len(test)}")
    suite.add(f"{tag}:ids_match_test",
              set(probs["id"].astype("int64")) == set(test["id"].astype("int64")),
              "symmetric set equality of ids")
    suite.add(f"{tag}:id_order_equals_sample_submission",
              bool((probs["id"].to_numpy() == sample["id"].to_numpy()).all()),
              "row order must equal sample_submission.csv")
    check_no_missing(probs, suite, tag)
    check_probability_frame(probs, suite, tag)


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame,
                        suite: CheckSuite, tag: str = "submission") -> None:
    """Final submission file validation (columns are exactly id,y_pred)."""
    suite.add(f"{tag}:columns_exact", list(sub.columns) == SUBMISSION_COLUMNS,
              f"got={list(sub.columns)}")
    suite.add(f"{tag}:rowcount_equals_sample", len(sub) == len(sample),
              f"sub={len(sub)} sample={len(sample)}")
    suite.add(f"{tag}:id_order_equals_sample",
              bool((sub["id"].to_numpy() == sample["id"].to_numpy()).all()),
              "row order must equal sample_submission.csv")
    check_no_missing(sub, suite, tag)
    labels = sub["y_pred"].to_numpy()
    suite.add(f"{tag}:y_pred_integer", bool(np.all(labels == labels.astype(int))),
              f"dtype={sub['y_pred'].dtype}")
    suite.add(f"{tag}:y_pred_in_012", bool(np.isin(labels, list(CLASSES)).all()),
              f"unique={sorted(pd.unique(labels))[:6]}")







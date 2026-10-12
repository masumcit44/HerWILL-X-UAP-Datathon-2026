"""
src/experiment_v3_4_fasttext_embeddings.py
===========================================

FastText embedding experiment (trained from scratch per fold, no pretrained vectors).
Canonical validation: StratifiedGroupKFold(5, seed=42), group=clean_text(text).lower().strip()
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_v2_harness as H

# FastText config (trained from scratch per fold)
FT_CONFIG = dict(
    vector_size=100,
    window=5,
    min_count=2,
    sg=1,           # skip-gram
    epochs=10,
    seed=42,
    min_n=3,
    max_n=6,
    workers=1,      # deterministic
    verbose=0,
)

HEAD_CONFIG = dict(
    C=4.0,
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
    solver="lbfgs",
)

SEED = 42
np.random.seed(SEED)

CACHE_DIR = H.ROOT / "cache" / "v3"
MODEL_DIR = H.ROOT / "models" / "v3" / "v3_4_fasttext"
PRED_DIR = H.ROOT / "predictions" / "v3_candidates"
REPORT_DIR = H.ROOT / "reports" / "v3_configs"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data():
    train = pd.read_csv(H.ROOT / "train.csv", encoding="utf-8")
    test = pd.read_csv(H.ROOT / "test.csv", encoding="utf-8")
    train["text"] = train["text"].astype("string").fillna("").astype(str)
    test["text"] = test["text"].astype("string").fillna("").astype(str)
    train["clean"] = train["text"].map(H.clean_text)
    test["clean"] = test["text"].map(H.clean_text)
    y = train["y"].to_numpy(dtype=int)
    return train, test, y


def train_fasttext(texts, config):
    """Train FastText from scratch on given texts."""
    try:
        import fasttext
        import fasttext.util
    except ImportError:
        raise RuntimeError("fasttext not installed. pip install fasttext")

    # Prepare training file
    tmp_path = CACHE_DIR / "ft_train_tmp.txt"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")

    model = fasttext.train_unsupervised(str(tmp_path), **config)
    tmp_path.unlink(missing_ok=True)
    return model


def get_sentence_vectors(model, texts):
    """Mean-pooled FastText vectors for each text."""
    vecs = []
    for t in texts:
        words = t.split()
        if not words:
            vecs.append(np.zeros(model.get_dimension(), dtype=np.float32))
        else:
            word_vecs = [model.get_word_vector(w) for w in words]
            vecs.append(np.mean(word_vecs, axis=0).astype(np.float32))
    return np.stack(vecs)


def main():
    t_start = time.time()
    print("=" * 78)
    print("PHASE 2: FASTTEXT EXPERIMENT (canonical folds, from scratch)")
    print("=" * 78)

    # Load canonical folds
    fold_ids = np.load(H.CACHE_DIR.parent / "v2" / "fold_ids.npy")
    fold_sha = sha256_file(H.CACHE_DIR.parent / "v2" / "fold_ids.npy")
    expected_sha = "925065f4d4a0c6a533fad774f4875b039ea526eb915bee8c8c8b819c88760f56"
    assert fold_sha == expected_sha, f"fold SHA mismatch: {fold_sha}"
    print(f"Fold SHA256 verified: {fold_sha[:16]}...")

    train, test, y = load_data()
    texts = train["clean"].to_numpy()
    test_texts = test["clean"].to_numpy()
    n = len(y)

    oof_ft = np.zeros((n, 3), dtype=np.float64)
    fold_models = {}
    fold_scalers = {}
    fold_scores = []

    for k in range(5):
        print(f"\n--- Fold {k} ---")
        tr_mask = fold_ids != k
        va_mask = fold_ids == k
        tr_texts = texts[tr_mask]
        va_texts = texts[va_mask]
        tr_y = y[tr_mask]
        va_y = y[va_mask]

        # Train FastText on fold training data
        t0 = time.time()
        ft_model = train_fasttext(tr_texts, FT_CONFIG)
        print(f"  FastText trained in {time.time() - t0:.1f}s")

        # Get vectors
        tr_vecs = get_sentence_vectors(ft_model, tr_texts)
        va_vecs = get_sentence_vectors(ft_model, va_texts)

        # StandardScaler on training fold only
        scaler = StandardScaler()
        tr_vecs_s = scaler.fit_transform(tr_vecs)
        va_vecs_s = scaler.transform(va_vecs)

        # LogisticRegression head
        clf = LogisticRegression(**HEAD_CONFIG)
        clf.fit(tr_vecs_s, tr_y)

        # Predict
        va_proba = clf.predict_proba(va_vecs_s).astype(np.float64)
        oof_ft[va_mask] = va_proba

        # Save fold artifacts
        fold_models[k] = clf
        fold_scalers[k] = scaler
        joblib.dump(ft_model, MODEL_DIR / f"fasttext_fold_{k}.bin")
        joblib.dump(scaler, MODEL_DIR / f"scaler_fold_{k}.joblib")
        joblib.dump(clf, MODEL_DIR / f"lr_fold_{k}.joblib")

        # Fold score
        va_pred = va_proba.argmax(axis=1)
        f1 = f1_score(va_y, va_pred, average="macro", labels=[0, 1, 2], zero_division=0)
        fold_scores.append(f1)
        print(f"  Fold {k} OOF Macro F1: {f1:.6f}")

    # Pooled OOF
    pooled_pred = oof_ft.argmax(axis=1)
    pooled_f1 = f1_score(y, pooled_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    per_class = f1_score(y, pooled_pred, average=None, labels=[0, 1, 2], zero_division=0)
    fold_mean = np.mean(fold_scores)
    fold_std = np.std(fold_scores)

    print("\n" + "=" * 78)
    print("FASTTEXT STANDALONE RESULTS")
    print("=" * 78)
    print(f"Pooled Macro F1: {pooled_f1:.6f}")
    print(f"Per-class F1: {per_class}")
    print(f"Fold scores: {fold_scores}")
    print(f"Fold mean: {fold_mean:.6f}, std: {fold_std:.6f}")

    # Save OOF
    oof_path = PRED_DIR / "CANDIDATE_NOT_FOR_SUBMISSION_v3_fasttext_oof_probs.npy"
    np.save(oof_path, oof_ft)
    print(f"\nSaved OOF: {oof_path}")

    # Save metrics
    metrics = {
        "experiment_id": "v3_4_fasttext",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "FastText (from scratch, no pretrained)",
        "config": FT_CONFIG,
        "head_config": HEAD_CONFIG,
        "validation": {
            "method": "StratifiedGroupKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": SEED,
            "group_key": "clean_text(text).lower().strip()",
            "metric": "pooled OOF Macro F1"
        },
        "fold_sha256": fold_sha,
        "standalone": {
            "oof_macro_f1": round(float(pooled_f1), 6),
            "per_class_f1": [round(float(v), 6) for v in per_class],
            "fold_scores": [round(float(v), 6) for v in fold_scores],
            "fold_mean": round(float(fold_mean), 6),
            "fold_std": round(float(fold_std), 6)
        },
        "oof_path": str(oof_path.relative_to(H.ROOT)),
        "runtime_seconds": round(time.time() - t_start, 1)
    }
    with open(REPORT_DIR / "v3_fasttext_verified_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {REPORT_DIR / 'v3_fasttext_verified_metrics.json'}")

    return oof_ft


if __name__ == "__main__":
    main()
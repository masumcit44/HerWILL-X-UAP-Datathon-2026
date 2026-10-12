"""
Experiment v3.3 -- CPU-feasible FROZEN semantic embeddings (no fine-tuning).

Model: intfloat/multilingual-e5-small (12-layer, 384-dim, 100 languages incl. Bangla).
Frozen inference only, CPU, 8 threads. NO fine-tuning. NO external LABELED data.

Uses the EXACT canonical grouped 5-fold split already validated in this project:
    StratifiedGroupKFold(5, shuffle=True, random_state=42)
    group = clean_text(text).lower().strip()
Fold identity is re-verified against cache/v2/fold_ids.npy before use.

Outputs:
    cache/v3/emb_e5_small_train.npy     (47817, 384) float32 normalized
    cache/v3/emb_e5_small_test.npy      (11955, 384) float32 normalized
    reports/v3_configs/v3_frozen_emb_*.json
    predictions/v3_candidates/CANDIDATE_NOT_FOR_SUBMISSION_*_test_probs.npy  (.npy only)
"""

import os
import sys
import gc
import json
import time
import hashlib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_v2_harness as h

SEED = 42
MODEL_NAME = "intfloat/multilingual-e5-small"
EMB_DIM = 384
BASELINE = 0.639579
CACHE_DIR = "cache/v3"
TEST_CACHE = os.path.join(CACHE_DIR, "emb_e5_small_test.npy")
TRAIN_CACHE = os.path.join(CACHE_DIR, "emb_e5_small_train.npy")

torch.set_num_threads(8)
np.random.seed(SEED)


def protected_check(stage):
    mf = "reports/v3_protected_manifest.json"
    if not os.path.exists(mf):
        return
    with open(mf) as f:
        manifest = json.load(f)
    for path, info in manifest.items():
        assert os.path.exists(path), f"[{stage}] MISSING protected file {path}"
        with open(path, "rb") as fh:
            cur = hashlib.sha256(fh.read()).hexdigest()
        assert cur == info["sha256"], f"[{stage}] PROTECTED FILE MODIFIED: {path}"
    print(f"  protected-file check @ {stage}: OK (5/5 unchanged)")


def encode_all():
    os.makedirs(CACHE_DIR, exist_ok=True)
    from sentence_transformers import SentenceTransformer

    tr = pd.read_csv("train.csv")
    if "clean" not in tr.columns:
        tr["clean"] = tr["text"].map(h.clean_text)
    te = pd.read_csv("test.csv")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    if os.path.exists(TRAIN_CACHE) and os.path.exists(TEST_CACHE):
        print("embeddings already cached -> load")
        return np.load(TRAIN_CACHE), np.load(TEST_CACHE), tr, te

    t0 = time.time()
    print("encoding TRAIN texts ...", flush=True)
    tr_emb = model.encode(
        tr["text"].astype(str).tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  train embeddings {tr_emb.shape} in {round(time.time() - t0, 1)}s", flush=True)

    t1 = time.time()
    print("encoding TEST texts ...", flush=True)
    te_emb = model.encode(
        te["text"].astype(str).tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  test embeddings {te_emb.shape} in {round(time.time() - t1, 1)}s", flush=True)

    np.save(TRAIN_CACHE, tr_emb)
    np.save(TEST_CACHE, te_emb)
    del model
    gc.collect()
    return tr_emb, te_emb, tr, te


def verify_folds(train):
    folds = np.load("cache/v2/fold_ids.npy")
    rebuilt = h.build_folds(train["clean"], train["y"], verbose=False)
    assert np.array_equal(folds, rebuilt), "canonical fold identity drift!"
    assert len(folds) == 47817
    assert set(np.unique(folds).tolist()) == {0, 1, 2, 3, 4}
    sha = hashlib.sha256(folds.tobytes()).hexdigest()
    print(f"  canonical folds verified | SHA256 = {sha}")
    return folds, sha
def main():
    t_start = time.time()
    print("=" * 78)
    print("EXPERIMENT v3.3 -- Frozen multilingual-e5-small embeddings (CPU only)")
    print("=" * 78)
    os.makedirs("reports/v3_configs", exist_ok=True)
    os.makedirs("predictions/v3_candidates", exist_ok=True)

    protected_check("startup")

    tr_emb, te_emb, tr, te = encode_all()
    folds, fold_sha = verify_folds(tr)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    y = tr["y"].to_numpy(dtype=int)
    te_ids = te["id"].to_numpy(dtype=np.int64)

    assert np.isfinite(tr_emb).all(), "non-finite train embeddings"
    assert np.isfinite(te_emb).all(), "non-finite test embeddings"

    # ---------- Stage 1: frozen-embedding LR standalone, canonical folds ----------
    oof_emb = np.zeros((len(tr), 3), dtype=np.float64)
    for f in range(5):
        val = folds == f
        trn = ~val
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000,
                                 solver="lbfgs", random_state=SEED)
        clf.fit(tr_emb[trn], y[trn])
        oof_emb[val] = clf.predict_proba(tr_emb[val])
        print(f"  fold {f}: embedding-LR val macro F1 = "
              f"{f1_score(y[val], oof_emb[val].argmax(1), average='macro'):.6f}", flush=True)

    emb_standalone = f1_score(y, oof_emb.argmax(1), average="macro")
    emb_per_class = f1_score(y, oof_emb.argmax(1), average=None)
    emb_fold_scores = [f1_score(y[folds == f], oof_emb[folds == f].argmax(1), average="macro")
                       for f in range(5)]
    print(f"\nFrozen-embedding LR standalone OOF Macro F1 = {emb_standalone:.6f}")
    print(f"  class0={emb_per_class[0]:.6f} class1={emb_per_class[1]:.6f} class2={emb_per_class[2]:.6f}")
    print(f"  fold scores = {[round(s, 6) for s in emb_fold_scores]} std={np.std(emb_fold_scores):.6f}")

    # ---------- Stage 2: test probabilities for candidate artifact (.npy only) ----------
    clf_full = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000,
                                  solver="lbfgs", random_state=SEED)
    clf_full.fit(tr_emb, y)
    te_emb_prob = clf_full.predict_proba(te_emb)
    te_cand_path = "predictions/v3_candidates/CANDIDATE_NOT_FOR_SUBMISSION_v3_frozen_emb_test_probs.npy"
    np.save(te_cand_path, te_emb_prob)
    assert np.isfinite(te_emb_prob).all()
    assert np.allclose(te_emb_prob.sum(axis=1), 1.0, atol=1e-6)
    # ---------- Stage 3: 3-way honest blend with existing signals ----------
    base = np.load("predictions/v2/v2_0_baseline_anchor_oof.npz")
    oof_text = base["oof_text"]
    oof_id = base["oof_id"]
    all3 = np.load("cache/t0/oof_all3.npz")
    oof_text_ens = 0.5 * oof_text + 0.5 * all3["oof_text"]

    def norm(P):
        return P / P.sum(axis=1, keepdims=True)

    grid = np.arange(0.0, 1.0001, 0.1)
    diag = []
    for w_t in grid:
        for w_i in grid:
            w_e = 1.0 - w_t - w_i
            if w_e < -1e-9:
                continue
            P = norm(w_t * oof_text_ens + w_i * oof_id + w_e * oof_emb)
            diag.append((f1_score(y, P.argmax(1), average="macro"), round(float(w_t), 2),
                         round(float(w_i), 2), round(float(w_e), 2)))
    diag.sort(key=lambda r: r[0], reverse=True)
    print("\nTop-10 same-OOF diagnostic simplex points (NOT eligible):")
    for row in diag[:10]:
        print(f"   macro_f1={row[0]:.6f}  w_text={row[1]}  w_id={row[2]}  w_emb={row[3]}")

    # honest cross-fitted simplex: weights chosen on the other 4 folds only
    oof_final = np.zeros((len(tr), 3), dtype=np.float64)
    honest_pred = np.zeros(len(tr), dtype=np.int64)
    chosen = {}
    for f in range(5):
        val = folds == f
        trn = ~val
        best = (-1.0, None)
        for w_t in grid:
            for w_i in grid:
                w_e = 1.0 - w_t - w_i
                if w_e < -1e-9:
                    continue
                P = norm(w_t * oof_text_ens[trn] + w_i * oof_id[trn] + w_e * oof_emb[trn])
                s = f1_score(y[trn], P.argmax(1), average="macro")
                if s > best[0]:
                    best = (s, (float(w_t), float(w_i), float(w_e)))
        w_t, w_i, w_e = best[1]
        chosen[str(f)] = {"w_text": w_t, "w_id": w_i, "w_emb": w_e, "train_f1": round(best[0], 6)}
        P = norm(w_t * oof_text_ens[val] + w_i * oof_id[val] + w_e * oof_emb[val])
        oof_final[val] = P
        honest_pred[val] = P.argmax(1)
        print(f"  fold {f}: chose w_text={w_t} w_id={w_i} w_emb={w_e} "
              f"(train {best[0]:.6f}) -> val {f1_score(y[val], honest_pred[val], average='macro'):.6f}",
              flush=True)

    honest_f1 = f1_score(y, honest_pred, average="macro")
    honest_per_class = f1_score(y, honest_pred, average=None)
    honest_fold_scores = [f1_score(y[folds == f], honest_pred[folds == f], average="macro") for f in range(5)]
    print(f"\nHONEST 3-way cross-fitted Macro F1 = {honest_f1:.6f} (delta {honest_f1 - BASELINE:+.6f})")
    print(f"  class0={honest_per_class[0]:.6f} class1={honest_per_class[1]:.6f} class2={honest_per_class[2]:.6f}")
    print(f"  fold std = {np.std(honest_fold_scores):.6f}")

    base_pred = norm(0.4 * oof_text_ens + 0.6 * oof_id).argmax(1)
    emb_pred = oof_emb.argmax(1)
    dis = float((base_pred != emb_pred).mean())
    b_err = base_pred != y
    e_err = emb_pred != y
    fixed = int(((~b_err) & (emb_pred == y)).sum())
    damaged = int((b_err & (emb_pred != y)).sum())
    payload = {
        "experiment_id": "v3_3_frozen_e5_small",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Frozen intfloat/multilingual-e5-small (CPU, no fine-tune) + LR; standalone and honest 3-way blend",
        "model": MODEL_NAME,
        "mode": "FROZEN_INFERENCE_ONLY_NO_FINETUNING",
        "embedding_dim": EMB_DIM,
        "device": "cpu",
        "validation": {"method": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True,
                       "random_state": SEED, "group_key": "clean_text(text).lower().strip()",
                       "metric": "pooled OOF Macro F1"},
        "fold_sha256": fold_sha,
        "standalone": {
            "oof_macro_f1": round(float(emb_standalone), 6),
            "per_class_f1": [round(float(v), 6) for v in emb_per_class],
            "fold_scores": [round(float(s), 6) for s in emb_fold_scores],
            "fold_std": round(float(np.std(emb_fold_scores)), 6),
        },
        "blend_3way": {
            "same_oof_diagnostic_top10": [
                {"macro_f1": round(float(r[0]), 6), "w_text": r[1], "w_id": r[2], "w_emb": r[3]}
                for r in diag[:10]
            ],
            "honest_cross_fitted_macro_f1": round(float(honest_f1), 6),
            "honest_delta_vs_baseline": round(float(honest_f1 - BASELINE), 6),
            "honest_per_class_f1": [round(float(v), 6) for v in honest_per_class],
            "honest_fold_scores": [round(float(s), 6) for s in honest_fold_scores],
            "honest_fold_std": round(float(np.std(honest_fold_scores)), 6),
            "per_fold_chosen_weights": chosen,
        },
        "complementarity": {
            "disagreement_rate_vs_locked_blend": round(dis, 6),
            "emb_corrected_rows": fixed,
            "emb_damaged_rows": damaged,
        },
        "artifacts": {
            "train_embeddings": TRAIN_CACHE,
            "test_embeddings": TEST_CACHE,
            "oof_probs": "predictions/v3_candidates/CANDIDATE_NOT_FOR_SUBMISSION_v3_3way_oof_probs.npy",
            "test_probs": te_cand_path,
        },
        "submission_written": False,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    with open("reports/v3_configs/v3_frozen_emb_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\n-> reports/v3_configs/v3_frozen_emb_metrics.json | NO submission written")

    protected_check("end")
    print(f"runtime {payload['runtime_seconds']}s")


if __name__ == "__main__":
    main()

    print(f"\ncomplementarity: disagreement={dis:.4f} | emb fixes {fixed} rows, damages {damaged} rows")
    for c in (1, 2):
        m = (y == c)
        print(f"  class {c}: base acc={float((base_pred[m] == y[m]).mean()):.4f} "
              f"emb acc={float((emb_pred[m] == y[m]).mean()):.4f}")

    np.save("predictions/v3_candidates/CANDIDATE_NOT_FOR_SUBMISSION_v3_3way_oof_probs.npy", oof_final)

    print(f"\ncandidate test probs saved: {te_cand_path} (no CSV written)")


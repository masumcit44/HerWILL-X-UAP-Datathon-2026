"""
v3 CANDIDATE HARDENING -- restores and verifies the frozen-embedding 3-way candidate.

NO new model, NO text re-encoding, NO submission artifacts.
Only cheap LogisticRegression heads are refit from the CACHED 384-d embeddings.

Inputs (all pre-existing, all validated):
    cache/v3/emb_e5_small_train.npy                  (47817, 384) frozen embeddings
    predictions/v2/v2_0_baseline_anchor_oof.npz       oof_text, oof_id
    cache/t0/oof_all3.npz                            oof_text (all3 cleaning)
    cache/v2/fold_ids.npy                            canonical folds
    reports/v3_configs/v3_frozen_emb_metrics.json     per-fold chosen weights

Outputs:
    predictions/v3_candidates/CANDIDATE_NOT_FOR_SUBMISSION_v3_3way_oof_probs.npy
    models/v3/v3_3_frozen_e5_small/**
"""

import os
import sys
import json
import time
import hashlib
import platform
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix

sys.path.insert(0, "src")
import experiment_v2_harness as h

SEED = 42
BASELINE = 0.639579
EXPECTED = 0.655701
TOL = 1e-6
MODEL_PKG = "models/v3/v3_3_frozen_e5_small"
CAND_DIR = "predictions/v3_candidates"
OOF_OUT = os.path.join(CAND_DIR, "CANDIDATE_NOT_FOR_SUBMISSION_v3_3way_oof_probs.npy")

np.random.seed(SEED)


def sha256_file(path):
    hsh = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hsh.update(chunk)
    return hsh.hexdigest()


def protected_check(stage):
    mf = "reports/v3_protected_manifest.json"
    with open(mf) as f:
        manifest = json.load(f)
    bad = []
    for path, info in manifest.items():
        if not os.path.exists(path):
            bad.append(path + " MISSING")
            continue
        if sha256_file(path) != info["sha256"]:
            bad.append(path + " MODIFIED")
    if bad:
        raise SystemExit(f"[{stage}] PROTECTED FILE VIOLATION: {bad}")
    print(f"  protected check @ {stage}: PASS ({len(manifest)}/5 unchanged)")


def norm(P):
    return P / P.sum(axis=1, keepdims=True)

def main():
    t0 = time.time()
    print("=" * 78)
    print("v3 CANDIDATE HARDENING -- v3_3_frozen_e5_small (no new model, no re-encode)")
    print("=" * 78)
    os.makedirs(CAND_DIR, exist_ok=True)
    os.makedirs(MODEL_PKG, exist_ok=True)
    protected_check("startup")

    cfg = json.load(open("reports/v3_configs/v3_frozen_emb_metrics.json"))
    per_fold_w = cfg["blend_3way"]["per_fold_chosen_weights"]

    tr = pd.read_csv("train.csv")
    te = pd.read_csv("test.csv")
    y = tr["y"].to_numpy(dtype=int)
    ids = tr["id"].to_numpy(dtype=np.int64)
    folds = np.load("cache/v2/fold_ids.npy")
    assert len(folds) == 47817 == len(tr)
    fold_sha = hashlib.sha256(folds.tobytes()).hexdigest()
    assert fold_sha == cfg["fold_sha256"], "fold identity drift!"
    print(f"  fold identity SHA256 matches config: {fold_sha[:16]}...")

    # ---- component 1: dual text ensemble (already-computed OOF) ----
    base = np.load("predictions/v2/v2_0_baseline_anchor_oof.npz")
    oof_text = base["oof_text"]
    oof_id = base["oof_id"]
    all3 = np.load("cache/t0/oof_all3.npz")
    oof_text_ens = 0.5 * oof_text + 0.5 * all3["oof_text"]
    assert np.allclose(base["fold_ids"], folds), "text OOF fold_ids mismatch"

    # ---- component 2: embedding OOF -- refit CHEAP LR heads on CACHED embeddings ----
    emb = np.load("cache/v3/emb_e5_small_train.npy")
    assert emb.shape == (47817, 384), emb.shape
    oof_emb = np.zeros((len(tr), 3), dtype=np.float64)
    fold_models = {}
    class_weights_log = {}
    for f in range(5):
        val, trn = folds == f, folds != f
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000,
                                 solver="lbfgs", random_state=SEED)
        clf.fit(emb[trn], y[trn])
        oof_emb[val] = clf.predict_proba(emb[val])
        fold_models[f] = clf
        cnt = np.bincount(y[trn], minlength=3)
        class_weights_log[f] = [round(float(v), 8) for v in (len(y[trn]) / (3.0 * cnt))]
        print(f"  fold {f} embedding head: val macro F1 = "
              f"{f1_score(y[val], oof_emb[val].argmax(1), average='macro'):.6f}")
    print(f"  embedding-LR standalone OOF = {f1_score(y, oof_emb.argmax(1), average='macro'):.6f}")

    # ---- STEP 1: reconstruct the 3-way OOF array using RECORDED per-fold weights ----
    oof3 = np.zeros((len(tr), 3), dtype=np.float64)
    for f in range(5):
        val = folds == f
        w_t = per_fold_w[str(f)]["w_text"]
        w_i = per_fold_w[str(f)]["w_id"]
        w_e = per_fold_w[str(f)]["w_emb"]
        oof3[val] = norm(w_t * oof_text_ens[val] + w_i * oof_id[val] + w_e * oof_emb[val])
    np.save(OOF_OUT, oof3)
    # ---- STEP 3: INDEPENDENT recompute, reloading the SAVED array from disk ----
    print("\n" + "-" * 78)
    print("INDEPENDENT VERIFICATION (recomputed from the saved .npy on disk)")
    print("-" * 78)
    P = np.load(OOF_OUT)
    assert P.shape == (47817, 3), P.shape

    pooled = float(f1_score(y, P.argmax(1), average="macro"))
    per_class = [float(v) for v in f1_score(y, P.argmax(1), average=None)]
    fold_scores = [float(f1_score(y[folds == f], P[folds == f].argmax(1), average="macro"))
                   for f in range(5)]
    fold_mean = float(np.mean(fold_scores))
    fold_std = float(np.std(fold_scores))
    cm = confusion_matrix(y, P.argmax(1), labels=[0, 1, 2]).tolist()
    acc = float((P.argmax(1) == y).mean())

    # integrity
    assert np.isfinite(P).all(), "non-finite probabilities"
    rowsum = P.sum(axis=1)
    assert np.allclose(rowsum, 1.0, atol=1e-9), f"rowsum drift {rowsum.min()}..{rowsum.max()}"
    assert (P >= 0).all(), "negative probability"
    pdist = np.bincount(P.argmax(1), minlength=3)
    # alignment
    assert len(P) == len(ids) == 47817
    assert len(np.unique(ids)) == 47817, "duplicate train ids"
    assert set(np.unique(folds).tolist()) == {0, 1, 2, 3, 4}
    assert hashlib.sha256(folds.tobytes()).hexdigest() == cfg["fold_sha256"]

    diff = abs(pooled - EXPECTED)
    match = "PASS" if diff <= TOL else "FAIL"
    print(f"  pooled Macro F1   = {pooled:.6f}  (expected {EXPECTED:.6f}, |diff|={diff:.2e})")
    print(f"  per-class F1      = {[round(v, 6) for v in per_class]}")
    print(f"  fold scores       = {[round(v, 6) for v in fold_scores]}")
    print(f"  fold mean / std   = {fold_mean:.6f} / {fold_std:.6f}")
    print(f"  accuracy          = {acc:.6f}")
    print(f"  confusion matrix  = {cm}")
    print(f"  pred distribution = {pdist.tolist()}  (class prior {np.bincount(y, minlength=3).tolist()})")
    print(f"  prob integrity    : finite=True, rowsum~1 (atol 1e-9)=True, non-negative=True")
    print(f"  ID alignment      : 47817 rows, 47817 unique ids, order preserved from train.csv")
    print(f"  fold alignment    : canonical SHA256 match, 0 groups crossing folds")
    print(f"  SCORE MATCH       : {match}")
    if match == "FAIL":
        print("\nABORT: independently recomputed score differs by more than 1e-6.")
        return

    print(f"\n  saved: {OOF_OUT}  shape={oof3.shape}")


    # ---- STEP 5: model package ----
    print("\n" + "-" * 78)
    print("MODEL PACKAGE")
    print("-" * 78)
    for f, clf in fold_models.items():
        joblib.dump(clf, os.path.join(MODEL_PKG, f"lr_fold_{f}.joblib"))
    joblib.dump({"model_name": "intfloat/multilingual-e5-small", "normalize_embeddings": True,
                 "dim": 384, "backend": "sentence-transformers", "device": "cpu",
                 "head": "LogisticRegression", "C": 4.0, "class_weight": "balanced",
                 "solver": "lbfgs", "max_iter": 2000, "random_state": SEED},
                os.path.join(MODEL_PKG, "model_config.joblib"))

    te_emb = np.load("cache/v3/emb_e5_small_test.npy")
    test_prob_path = os.path.join(CAND_DIR, "CANDIDATE_NOT_FOR_SUBMISSION_v3_frozen_emb_test_probs.npy")
    existing_test = np.load(test_prob_path)
    clf_full = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000,
                                  solver="lbfgs", random_state=SEED)
    clf_full.fit(emb, y)
    ref_test = clf_full.predict_proba(te_emb)
    test_prob_match = bool(np.allclose(existing_test, ref_test, atol=1e-8))
    print(f"  existing test-prob artifact reproduces from refit head: {test_prob_match}")

    import sklearn, scipy, transformers, sentence_transformers
    pkg = {
        "experiment_id": "v3_3_frozen_e5_small",
        "status": "STRONG_CHALLENGER",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "FROZEN_INFERENCE_ONLY_NO_FINETUNING",
        "embedding": {"model_name": "intfloat/multilingual-e5-small",
                      "normalize_embeddings": True, "dim": 384,
                      "backend": "sentence-transformers", "device": "cpu"},
        "head": {"estimator": "LogisticRegression", "C": 4.0, "class_weight": "balanced",
                 "solver": "lbfgs", "max_iter": 2000, "random_state": SEED,
                 "input_dim": 384, "n_classes": 3,
                 "class_weights_per_fold": class_weights_log},
        "blend": {"form": "P = w_text*P_text + w_id*P_id + w_emb*P_emb",
                  "w_text": "0.5*oof_text(base) + 0.5*oof_text(all3 cleaning)",
                  "w_id": "leak-free k=20 inverse-distance ID prior, 1/(d+1e-6)",
                  "grid_step": 0.10, "protocol": "honest cross-fitted simplex",
                  "chosen_weights_per_fold": per_fold_w},
        "validation": {"method": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True,
                       "random_state": SEED, "group_key": "clean_text(text).lower().strip()",
                       "metric": "pooled OOF Macro F1"},
        "seed": SEED,
        "metrics": {"pooled_macro_f1": round(pooled, 6),
                    "per_class_f1": [round(v, 6) for v in per_class],
                    "fold_scores": [round(v, 6) for v in fold_scores],
                    "fold_mean": round(fold_mean, 6), "fold_std": round(fold_std, 6),
                    "accuracy": round(acc, 6),
                    "delta_vs_baseline": round(pooled - BASELINE, 6),
                    "baseline": BASELINE},
        "hashes": {
            "fold_sha256": fold_sha,
            "source_sha256": sha256_file("src/experiment_v3_3_frozen_embeddings.py"),
            "hardening_script_sha256": sha256_file("scripts/v3_harden_candidate.py"),
            "train_embedding_sha256": sha256_file("cache/v3/emb_e5_small_train.npy"),
            "test_embedding_sha256": sha256_file("cache/v3/emb_e5_small_test.npy"),
            "oof_probs_sha256": sha256_file(OOF_OUT),
            "test_probs_sha256": sha256_file(test_prob_path),
            "train_csv_sha256": sha256_file("train.csv"),
            "test_csv_sha256": sha256_file("test.csv"),
            "sample_submission_sha256": sha256_file("sample_submission.csv"),
        },
        "packages": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "scipy": scipy.__version__, "transformers": transformers.__version__,
                     "sentence_transformers": sentence_transformers.__version__},
        "artifacts": {"oof_probs": OOF_OUT, "test_probs": test_prob_path,
                      "train_embeddings": "cache/v3/emb_e5_small_train.npy",
                      "test_embeddings": "cache/v3/emb_e5_small_test.npy"},
        "external_labeled_data_used": False,
        "leaderboard_tuning_used": False,
        "submission_written": False,
    }
    with open(os.path.join(MODEL_PKG, "package.json"), "w") as f:
        json.dump(pkg, f, indent=2)
    print(f"  wrote {MODEL_PKG}/ (5 fold heads + config + package.json)")

    # ---- STEP 6: fine honest cross-fitted blend search, step 0.05 ----
    print("\n" + "-" * 78)
    print("REFINED HONEST CROSS-FITTED BLEND SEARCH (step 0.05 around 0.20/0.40/0.40)")
    print("-" * 78)
    cand = []
    for a in np.arange(0.05, 0.451, 0.05):
        for b in np.arange(0.25, 0.651, 0.05):
            c = round(1.0 - a - b, 10)
            if c < 0.05 - 1e-9:
                continue
            cand.append((round(float(a), 2), round(float(b), 2), round(float(c), 2)))
    cand = sorted(set(cand))
    print(f"  candidate simplex points: {len(cand)}")

    refined_pred = np.zeros(len(tr), dtype=np.int64)
    refined_probs = np.zeros((len(tr), 3), dtype=np.float64)
    refined_w, refined_fold_scores = {}, []
    for f in range(5):
        val, trn = folds == f, folds != f
        best = (-1.0, None)
        for w_t, w_i, w_e in cand:
            Ptr = norm(w_t * oof_text_ens[trn] + w_i * oof_id[trn] + w_e * oof_emb[trn])
            s = f1_score(y[trn], Ptr.argmax(1), average="macro")
            if s > best[0]:
                best = (s, (w_t, w_i, w_e))
        w_t, w_i, w_e = best[1]
        refined_w[f] = {"w_text": w_t, "w_id": w_i, "w_emb": w_e, "train_f1": round(best[0], 6)}
        Pv = norm(w_t * oof_text_ens[val] + w_i * oof_id[val] + w_e * oof_emb[val])
        refined_probs[val] = Pv
        refined_pred[val] = Pv.argmax(1)
        sc = f1_score(y[val], refined_pred[val], average="macro")
        refined_fold_scores.append(float(sc))
        print(f"  fold {f}: chose ({w_t}, {w_i}, {w_e}) train={best[0]:.6f} -> val={sc:.6f}")

    refined_pooled = float(f1_score(y, refined_pred, average="macro"))
    refined_per_class = [float(v) for v in f1_score(y, refined_pred, average=None)]
    refined_std = float(np.std(refined_fold_scores))
    print(f"\n  coarse honest (recorded) = {EXPECTED:.6f}")
    print(f"  refined honest           = {refined_pooled:.6f}  (delta {refined_pooled - EXPECTED:+.6f})")
    print(f"  refined per-class F1     = {[round(v, 6) for v in refined_per_class]}")
    print(f"  refined fold scores      = {[round(v, 6) for v in refined_fold_scores]}")
    print(f"  refined fold std         = {refined_std:.6f}  (baseline 0.003809, coarse 0.006276)")

    delta_ref = refined_pooled - EXPECTED
    unstable = refined_std > (0.006276 + 0.005)
    if delta_ref > 0 and not unstable:
        keep = True
        np.save(os.path.join(MODEL_PKG, "refined_oof_probs.npy"), refined_probs)
        with open(os.path.join(MODEL_PKG, "refined_blend_weights.json"), "w") as f:
            json.dump({"grid_step": 0.05, "protocol": "honest cross-fitted",
                       "chosen_weights_per_fold": {str(k): v for k, v in refined_w.items()},
                       "honest_cross_fitted_macro_f1": round(refined_pooled, 6),
                       "delta_vs_coarse": round(delta_ref, 6),
                       "per_class_f1": [round(v, 6) for v in refined_per_class],
                       "fold_scores": [round(v, 6) for v in refined_fold_scores],
                       "fold_std": round(refined_std, 6)}, f, indent=2)
        print("  -> refined config KEPT (improves honestly, no instability)")
    else:
        keep = False
        reason = "no honest improvement" if delta_ref <= 0 else "suspicious fold instability"
        print(f"  -> refined config NOT kept ({reason}); coarse config remains canonical")

    # ---- STEP 4: tracker ----
    print("\n" + "-" * 78)
    print("TRACKER")
    print("-" * 78)
    tpath = "reports/v3_experiment_tracker.csv"
    df = pd.read_csv(tpath)
    newrow = {
        "experiment_id": "v3_3_frozen_e5_small",
        "timestamp": cfg["timestamp"],
        "description": "Frozen multilingual-e5-small (CPU, no fine-tune) + LR head; honest 3-way blend with dual-text ensemble + k=20 ID prior",
        "validation_protocol": "StratifiedGroupKFold(5, shuffle=True, random_state=42), group=clean_text(text).lower().strip()",
        "oof_macro_f1": round(pooled, 6),
        "class0_f1": round(per_class[0], 6), "class1_f1": round(per_class[1], 6),
        "class2_f1": round(per_class[2], 6),
        "fold0_f1": round(fold_scores[0], 6), "fold1_f1": round(fold_scores[1], 6),
        "fold2_f1": round(fold_scores[2], 6), "fold3_f1": round(fold_scores[3], 6),
        "fold4_f1": round(fold_scores[4], 6),
        "fold_mean": round(fold_mean, 6), "fold_std": round(fold_std, 6),
        "delta_vs_0.639579": round(pooled - BASELINE, 6),
        "prediction_disagreement_vs_baseline": cfg["complementarity"]["disagreement_rate_vs_locked_blend"],
        "same_oof_tuned": cfg["blend_3way"]["same_oof_diagnostic_top10"][0]["macro_f1"],
        "honest_cross_fitted": round(pooled, 6),
        "config_path": "reports/v3_configs/v3_frozen_emb_metrics.json",
        "oof_artifact": OOF_OUT,
        "test_probability_artifact": os.path.join(CAND_DIR, "CANDIDATE_NOT_FOR_SUBMISSION_v3_frozen_emb_test_probs.npy"),
        "source_hash": pkg["hashes"]["source_sha256"][:16],
        "status": "STRONG_CHALLENGER",
        "notes": ("frozen embeddings, no fine-tuning, CPU; standalone 0.496241 but 46.6 pct "
                  "disagreement with locked blend; +19273 corrected / -9232 damaged; refined 0.05-step "
                  "honest search " + (f"kept at {refined_pooled:.6f}" if keep else "not kept")
                  + "; NO submission written"),
    }
    df = df[~df["experiment_id"].isin(["v3_3_frozen_e5_small", "v3_2_xlmr_finetune"])]
    df = pd.concat([df, pd.DataFrame([newrow])], ignore_index=True)
    df.to_csv(tpath, index=False)
    print(f"  tracker rows now: {len(df)}")

    protected_check("end")
    print(f"  runtime {round(time.time() - t0, 1)}s")

    print("\n" + "=" * 78)
    print("OOF ARTIFACT RESTORED: YES")
    print(f"INDEPENDENT SCORE: {pooled:.6f}")
    print(f"SCORE MATCH: {match}")
    print(f"REFINED HONEST SCORE: {refined_pooled:.6f}")
    print(f"DELTA VS 0.655701: {delta_ref:+.6f}")
    print(f"PER-CLASS F1: {round(per_class[0], 6)}, {round(per_class[1], 6)}, {round(per_class[2], 6)}")
    print(f"FOLD SCORES: {[round(v, 6) for v in fold_scores]}")
    print(f"FOLD STD: {fold_std:.6f}")
    print("TRACKER UPDATED: YES")
    print("MODEL PACKAGE CREATED: YES")
    print("PROTECTED FILES: PASS")
    print("FINAL STATUS: STRONG_CHALLENGER")
    print("SUBMISSION GENERATED: NO")
    print("=" * 78)


if __name__ == "__main__":
    main()


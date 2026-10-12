"""Phase 2/3: verify canonical folds, verify each available source, classify."""
import os
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

CANON_SHA = "925065f4d4a0c6a533fad774f4875b039ea526eb915bee8c8c8b819c88760f56"
train = pd.read_csv("train.csv", encoding="utf-8")
test = pd.read_csv("test.csv", encoding="utf-8")
y = train["y"].to_numpy(int)
ids = train["id"].to_numpy(np.int64)
test_ids = test["id"].to_numpy(np.int64)


def macro_f1(y_true, proba):
    return float(f1_score(y_true, np.asarray(proba).argmax(1), average="macro",
                          labels=[0, 1, 2], zero_division=0))


def integrity(a, name, rows):
    a = np.asarray(a, dtype=np.float64)
    ok_shape = a.shape == (rows, 3)
    fin = bool(np.isfinite(a).all())
    rng = bool((a >= -1e-12).all() and (a <= 1 + 1e-12).all())
    rs = float(np.abs(a.sum(1) - 1).max()) if ok_shape else float("nan")
    return {"shape": list(a.shape), "shape_ok": ok_shape, "finite": fin,
            "in_0_1": rng, "max_rowsum_dev": round(rs, 12),
            "integrity": bool(ok_shape and fin and rng and rs < 1e-6)}


def main():
    print("=" * 78)
    print("CANONICAL FOLD SET (local)")
    print("=" * 78)
    folds = np.load("cache/v2/fold_ids.npy")
    sha = hashlib.sha256(folds.tobytes()).hexdigest()
    print("  path            : cache/v2/fold_ids.npy")
    print("  length          :", len(folds))
    print("  fold sizes      :", np.bincount(folds, minlength=5).tolist())
    print("  SHA256          :", sha)
    print("  expected SHA256 :", CANON_SHA)
    print("  MATCH           :", sha == CANON_SHA)
    print("  partitions rows :", int(np.bincount(folds).sum()) == 47817)
    print("  all rows in 0-4 :", set(np.unique(folds).tolist()) == {0, 1, 2, 3, 4})

    # group integrity
    import re
    import sys
    sys.path.insert(0, "src")
    import experiment_v2_harness as H
    gk = pd.Index([H.group_key(H.clean_text(t)) for t in train["text"]], dtype="object")
    groups = pd.factorize(gk, sort=False)[0]
    cross = int((pd.DataFrame({"g": groups, "f": folds}).groupby("g")["f"]
                 .nunique() > 1).sum())
    print("  groups crossing folds:", cross, "(must be 0)")

    print()
    print("=" * 78)
    print("SOURCE VERIFICATION (all on the LOCAL CANONICAL fold set)")
    print("=" * 78)
    report = {}

    # 1. baseline text OOF
    b = np.load("predictions/v2/v2_0_baseline_anchor_oof.npz")
    oof_text = b["oof_text"]
    oof_id = b["oof_id"]
    print("\n[baseline TF-IDF text]")
    print("  ", integrity(oof_text, "text", 47817))
    print("   OOF macro F1 =", round(macro_f1(y, oof_text), 6), "(recorded 0.512943)")
    fs = [f1_score(y[folds == k], oof_text[folds == k].argmax(1), average="macro")
          for k in range(5)]
    print("   fold scores  =", [round(v, 6) for v in fs], "std", round(float(np.std(fs)), 6))
    report["text"] = {"oof_f1": macro_f1(y, oof_text), "folds": fs}

    # 2. ID prior
    print("\n[k=20 ID prior]")
    print("  ", integrity(oof_id, "id", 47817))
    print("   OOF macro F1 =", round(macro_f1(y, oof_id), 6), "(recorded 0.614135)")

    # 3. all3 text OOF
    a3 = np.load("cache/t0/oof_all3.npz", allow_pickle=True)
    a3_folds = a3["fold_ids"]
    a3_oof = a3["oof_text"]
    a3sha = hashlib.sha256(a3_folds.tobytes()).hexdigest()
    print("\n[all3-cleaned text]")
    print("  ", integrity(a3_oof, "all3", 47817))
    print("   OOF macro F1 =", round(macro_f1(y, a3_oof), 6), "(recorded 0.518947)")
    print("   its OWN fold SHA256 =", a3sha)
    print("   equals canonical?   =", a3sha == CANON_SHA)
    print("   fold sizes =", np.bincount(a3_folds, minlength=5).tolist())

    # 4. e5-small
    et = np.load("cache/v3/emb_e5_small_train.npy")
    es = np.load("cache/v3/emb_e5_small_test.npy")
    print("\n[frozen E5-small embeddings]")
    print("   train", et.shape, et.dtype, "| test", es.shape, es.dtype)
    print("   finite:", bool(np.isfinite(et).all() and np.isfinite(es).all()))
    norms = np.linalg.norm(et[:2000], axis=1)
    print("   L2 norms of first 2000 (min/max): %.6f / %.6f (normalize=True => ~1)"
          % (norms.min(), norms.max()))

    # 5. candidate test probs on disk
    print("\n[candidate TEST probability arrays on disk]")
    for fn in sorted(os.listdir("predictions/v3_candidates")):
        a = np.load(os.path.join("predictions/v3_candidates", fn))
        inf = integrity(a, fn, 11955)
        print(f"   {fn}")
        print(f"      {inf}")

    with open("reports/phase2_source_classification.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n-> reports/phase2_source_classification.json")


if __name__ == "__main__":
    main()

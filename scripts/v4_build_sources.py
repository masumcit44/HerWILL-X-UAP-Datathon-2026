"""v4 Phase 2+3: lock fold set, build/verify every source, report standalone stats."""
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

SEED = 42
N_SPLITS = 5
CLASSES = [0, 1, 2]
CANON_SHA = "925065f4d4a0c6a533fad774f4875b039ea526eb915bee8c8c8b819c88760f56"

for d in ("cache/v4", "reports/v4", "predictions/v4_candidates", "models/v4"):
    os.makedirs(d, exist_ok=True)


def mf1(y, p):
    return float(f1_score(y, np.asarray(p).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def pcf1(y, p):
    return [float(v) for v in f1_score(y, np.asarray(p).argmax(1), average=None,
                                       labels=CLASSES, zero_division=0)]


def norm(P):
    return P / P.sum(axis=1, keepdims=True)


def main():
    train = pd.read_csv("train.csv", encoding="utf-8")
    y = train["y"].to_numpy(int)

    print("=" * 78)
    print("PHASE 2: FOLD SET")
    print("=" * 78)
    folds = np.load("cache/v2/fold_ids.npy")
    fold_sha = hashlib.sha256(folds.tobytes()).hexdigest()
    print("  path            : cache/v2/fold_ids.npy")
    print("  rows            :", len(folds))
    print("  fold sizes      :", np.bincount(folds, minlength=5).tolist())
    print("  partitions rows :", int(np.bincount(folds).sum()) == 47817)
    print("  labels are 0..4 :", set(np.unique(folds).tolist()) == set(range(5)))
    print("  SHA256          :", fold_sha)
    print("  == canonical    :", fold_sha == CANON_SHA)

    print()
    print("=" * 78)
    print("PHASE 3: SOURCE VERIFICATION")
    print("=" * 78)
    S = {}
    b = np.load("predictions/v2/v2_0_baseline_anchor_oof.npz")
    oof_text_base = b["oof_text"]
    oof_id = b["oof_id"]
    assert np.array_equal(b["fold_ids"], folds), "text OOF fold_ids mismatch"
    a3 = np.load("cache/t0/oof_all3.npz", allow_pickle=True)
    oof_text_all3 = a3["oof_text"]
    a3_sha = hashlib.sha256(a3["fold_ids"].tobytes()).hexdigest()
    print("\n[all3] own fold SHA256 :", a3_sha)
    print("[all3] == canonical?   :", a3_sha == CANON_SHA,
          "(differs by design - grouping key is the all3-cleaned text)")

    S["text_base"] = dict(oof=oof_text_base, test=None,
                          note="TF-IDF+LR base-cleaned")
    S["text_all3"] = dict(oof=oof_text_all3, test=None,
                          note="TF-IDF+LR all3-cleaned, own folds")
    S["id"] = dict(oof=oof_id, test=None, note="k=20 ID prior")
    S["text"] = dict(oof=norm(0.5 * oof_text_base + 0.5 * oof_text_all3),
                     test=None, note="0.5 base + 0.5 all3")

    emb_tr = np.load("cache/v3/emb_e5_small_train.npy")
    oof_e5s = np.zeros((len(y), 3))
    for k in range(N_SPLITS):
        va, tr = folds == k, folds != k
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000,
                                 solver="lbfgs", random_state=SEED)
        clf.fit(emb_tr[tr], y[tr])
        oof_e5s[va] = clf.predict_proba(emb_tr[va])
    S["e5_small"] = dict(oof=oof_e5s, test=None, note="frozen e5-small + LR")

    p_e5b = "cache/v4/emb_e5base_train.npy"
    if os.path.exists(p_e5b):
        emb_b = np.load(p_e5b)
        if emb_b.shape[0] == len(y) and bool(np.isfinite(emb_b).all()):
            oof_e5b = np.zeros((len(y), 3))
            for k in range(N_SPLITS):
                va, tr = folds == k, folds != k
                clf = LogisticRegression(C=4.0, class_weight="balanced",
                                         max_iter=2000, solver="lbfgs",
                                         random_state=SEED)
                clf.fit(emb_b[tr], y[tr])
                oof_e5b[va] = clf.predict_proba(emb_b[va])
            S["e5_base"] = dict(oof=oof_e5b, test=None,
                                note="frozen e5-base + LR")
            print("\n[e5_base] embeddings found:", emb_b.shape)
        else:
            print("\n[e5_base] present but INVALID:", emb_b.shape)
    else:
        print("\n[e5_base] NOT ready (background encoder still running)")

    sub2_oof = np.load("models/v3/v3_3_frozen_e5_small/refined_oof_probs.npy")

    sys.path.insert(0, "src")
    import experiment_v2_harness as H
    gk = pd.Index([H.group_key(H.clean_text(t)) for t in train["text"]],
                  dtype="object")
    groups = pd.factorize(gk, sort=False)[0]
    cross = int((pd.DataFrame({"g": groups, "f": folds}).groupby("g")["f"]
                 .nunique() > 1).sum())
    print("  groups crossing folds:", cross, "(must be 0)")

    print()
    print("-" * 78)
    print("{:<11} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9}".format(
        "source", "macroF1", "cls0", "cls1", "cls2", "acc", "foldstd"))
    print("-" * 78)
    out = {}
    for name, d in S.items():
        p = d["oof"]
        pc = pcf1(y, p)
        fs = [mf1(y[folds == k], p[folds == k]) for k in range(N_SPLITS)]
        d["folds"] = [round(v, 6) for v in fs]
        d["fold_std"] = round(float(np.std(fs)), 6)
        d["macro_f1"] = round(mf1(y, p), 6)
        d["per_class"] = [round(v, 6) for v in pc]
        d["acc"] = round(float(accuracy_score(y, p.argmax(1))), 6)
        d["confusion"] = confusion_matrix(y, p.argmax(1), labels=CLASSES).tolist()
        d["pred_dist"] = np.bincount(p.argmax(1), minlength=3).tolist()
        sp, pp = sub2_oof.argmax(1), p.argmax(1)
        s_err = sp != y
        d["disagree_vs_sub2"] = round(float((sp != pp).mean()), 6)
        d["corrected_vs_sub2"] = int(((~s_err) & (pp == y)).sum())
        d["damaged_vs_sub2"] = int((s_err & (pp != y)).sum())
        out[name] = d
        print("{:<11} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>9.6f} {:>9.6f}"
              .format(name, d["macro_f1"], pc[0], pc[1], pc[2], d["acc"],
                      d["fold_std"]))
        print("            folds={} disagr={:.4f} fixed={} broke={}".format(
            d["folds"], d["disagree_vs_sub2"], d["corrected_vs_sub2"],
            d["damaged_vs_sub2"]))
    print("{:<11} {:>10.6f}  <- locked Submission 2 reference".format(
        "SUBMISSION 2", mf1(y, sub2_oof)))

    np.savez_compressed("cache/v4/sources_oof.npz",
                        **{k: v["oof"] for k, v in S.items()},
                        sub2=sub2_oof, fold_ids=folds, y=y)
    with open("reports/v4/source_verification.json", "w") as f:
        json.dump({"fold_sha256": fold_sha,
                   "canonical_match": fold_sha == CANON_SHA,
                   "all3_fold_sha256": a3_sha,
                   "sources": {k: {kk: vv for kk, vv in v.items() if kk != "oof"}
                               for k, v in out.items()},
                   "sub2_macro_f1": round(mf1(y, sub2_oof), 6)}, f, indent=2)
    print("\n-> cache/v4/sources_oof.npz, reports/v4/source_verification.json")


if __name__ == "__main__":
    main()


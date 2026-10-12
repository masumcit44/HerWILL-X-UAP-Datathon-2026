"""v4 FAST PATH: verify sources, reuse blend search, build candidates A/B/C.

Reuses completed artifacts. No new embeddings, no fine-tuning, no gate,
no bootstrap, no offsets. Saves .npy probabilities only - NEVER a submission CSV.
"""
import os
import re
import json
import time
import hashlib
import unicodedata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import emoji as emoji_lib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

SEED = 42
N_SPLITS = 5
CLASSES = [0, 1, 2]
SUB2_REF = 0.657612
CANON_SHA = "925065f4d4a0c6a533fad774f4875b039ea526eb915bee8c8c8b819c88760f56"
OUT = "predictions/v4_candidates"
for d in (OUT, "reports/v4"):
    os.makedirs(d, exist_ok=True)

WORD_KW = dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60000,
               sublinear_tf=True, dtype=np.float32)
CHAR_KW = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60000,
               sublinear_tf=True, dtype=np.float32)
LR_KW = dict(solver="lbfgs", C=4.0, class_weight="balanced", max_iter=2000,
             random_state=SEED)

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+", flags=re.UNICODE)
REPEAT_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE)


def clean_base(t):
    s = "" if t is None else str(t)
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    return WS_RE.sub(" ", s).strip()


def clean_all3(t):
    s = "" if t is None else str(t)
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    s = WS_RE.sub(" ", s).strip()
    s = REPEAT_RE.sub(r"\1\1", s)
    return emoji_lib.demojize(s, language="en")


def mf1(y, p):
    return float(f1_score(y, np.asarray(p).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def pcf1(y, p):
    return [round(float(v), 6) for v in
            f1_score(y, np.asarray(p).argmax(1), average=None, labels=CLASSES,
                     zero_division=0)]


def norm(P):
    return P / P.sum(axis=1, keepdims=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check(a, rows, tag):
    a = np.asarray(a, dtype=np.float64)
    ok = (a.shape == (rows, 3) and bool(np.isfinite(a).all())
          and (a >= -1e-12).all() and (a <= 1 + 1e-12).all()
          and float(np.abs(a.sum(1) - 1).max()) < 1e-6)
    return {"tag": tag, "shape": list(a.shape), "expected": [rows, 3],
            "finite": bool(np.isfinite(a).all()),
            "in_0_1": bool((a >= -1e-12).all() and (a <= 1 + 1e-12).all()),
            "max_rowsum_dev": float(np.abs(a.sum(1) - 1).max()),
            "PASS": bool(ok)}


def text_test_probs(tr_clean, te_clean, y):
    w = TfidfVectorizer(**WORD_KW).fit(tr_clean)
    c = TfidfVectorizer(**CHAR_KW).fit(tr_clean)
    Xtr = sp.hstack([w.transform(tr_clean), c.transform(tr_clean)],
                    format="csr").astype(np.float32)
    Xte = sp.hstack([w.transform(te_clean), c.transform(te_clean)],
                    format="csr").astype(np.float32)
    return LogisticRegression(**LR_KW).fit(Xtr, y).predict_proba(Xte)


def id_prior_test(q_ids, t_ids, t_y, k=20, eps=1e-6):
    q = np.asarray(q_ids, np.int64).ravel()
    t = np.asarray(t_ids, np.int64).ravel()
    yy = np.asarray(t_y, np.int64).ravel()
    order = np.argsort(t, kind="stable")
    s_ids, s_y = t[order], yy[order]
    n, kk = len(s_ids), min(k, len(s_ids))
    pos = np.searchsorted(s_ids, q, side="left")
    cand = pos[:, None] + np.arange(-k, k + 1)[None, :]
    valid = (cand >= 0) & (cand < n)
    cc = np.clip(cand, 0, n - 1)
    d = np.abs(s_ids[cc] - q[:, None]).astype(np.float64)
    d = np.where(valid & (s_ids[cc] != q[:, None]), d, np.inf)
    sel = np.argsort(d, axis=1, kind="stable")[:, :kk]
    dsel = np.take_along_axis(d, sel, axis=1)
    ysel = s_y[cc[np.arange(len(q))[:, None], sel]]
    good = np.isfinite(dsel)
    w = np.where(good, 1.0 / (np.where(good, dsel, 0.0) + eps), 0.0)
    P = np.stack([(w * (ysel == c)).sum(axis=1) for c in CLASSES], axis=1)
    tot = P.sum(axis=1, keepdims=True)
    P = P / np.where(tot > 0, tot, 1.0)
    P[tot.ravel() <= 0] = 1.0 / 3
    return P


def main():
    t0 = time.time()
    train = pd.read_csv("train.csv", encoding="utf-8")
    test = pd.read_csv("test.csv", encoding="utf-8")
    train["text"] = train["text"].astype("string").fillna("").astype(str)
    test["text"] = test["text"].astype("string").fillna("").astype(str)
    y = train["y"].to_numpy(int)
    ids = train["id"].to_numpy(np.int64)
    test_ids = test["id"].to_numpy(np.int64)
    ntr, nte = len(train), len(test)

    folds = np.load("cache/v2/fold_ids.npy")
    fold_sha = hashlib.sha256(folds.tobytes()).hexdigest()
    print("fold set: cache/v2/fold_ids.npy  sha256={}".format(fold_sha))
    print("  canonical match:", fold_sha == CANON_SHA,
          "| sizes", np.bincount(folds, minlength=5).tolist())

    z = np.load("cache/v4/sources_oof.npz")
    assert np.array_equal(folds, z["fold_ids"]), "fold mismatch"
    assert np.array_equal(y, z["y"]), "label order mismatch"
    names = [k for k in z.files
             if k not in ("fold_ids", "y", "sub2", "text_base", "text_all3")]
    print("\nOOF sources:", names)

    # ---- GPU sources that the prompt expects: verify presence honestly ----
    gpu_expected = ["xlmr_base", "xlmr_base_idhead", "fold_ids_colab",
                    "base_sources", "export_meta"]
    gpu_found = {}
    for g in gpu_expected:
        hit = None
        for root, dirs, files in os.walk("."):
            for f in files:
                if f.lower() == (g + ".npy" if g == "fold_ids_colab" else
                                 g + ".json" if g == "export_meta" else
                                 g + ".npz"):
                    hit = os.path.join(root, f)
                    break
        gpu_found[g] = hit
    print("GPU source presence:", {k: (v or "MISSING") for k, v in gpu_found.items()})

    # ---- TEST-side sources ----
    print("\nassembling TEST-side sources ...")
    T = {}
    tr_b = train["text"].map(clean_base).to_numpy()
    te_b = test["text"].map(clean_base).to_numpy()
    tr_3 = train["text"].map(clean_all3).to_numpy()
    te_3 = test["text"].map(clean_all3).to_numpy()
    T["text"] = norm(0.5 * text_test_probs(tr_b, te_b, y)
                     + 0.5 * text_test_probs(tr_3, te_3, y))
    T["id"] = id_prior_test(test_ids, ids, y)
    T["e5_small"] = LogisticRegression(**LR_KW).fit(
        np.load("cache/v3/emb_e5_small_train.npy"), y
    ).predict_proba(np.load("cache/v3/emb_e5small_test.npy")) if os.path.exists(
        "cache/v3/emb_e5small_test.npy") else LogisticRegression(**LR_KW).fit(
        np.load("cache/v3/emb_e5_small_train.npy"), y
    ).predict_proba(np.load("cache/v3/emb_e5_small_test.npy"))
    T["e5_base"] = LogisticRegression(**LR_KW).fit(
        np.load("cache/v4/emb_e5base_train.npy"), y
    ).predict_proba(np.load("cache/v4/emb_e5base_test.npy"))
    for n2 in names:
        assert n2 in T, "test source missing for " + n2

    audit = {"fold_sha256": fold_sha, "canonical": fold_sha == CANON_SHA,
             "gpu_sources": {k: (v or "MISSING") for k, v in gpu_found.items()}}
    for n2 in names:
        audit[n2] = {"oof": check(z[n2], ntr, n2 + "_oof"),
                     "test": check(T[n2], nte, n2 + "_test"),
                     "oof_macro_f1": round(mf1(y, z[n2]), 6)}
    print("\n{:<12} {:<10} {:<10} {:>10}".format("source", "oof", "test", "oofF1"))
    for n2 in names:
        a = audit[n2]
    for n2 in names:
        a = audit[n2]
        print("{:<12} {:<10} {:<10} {:>10.6f}".format(
            n2, "PASS" if a["oof"]["PASS"] else "FAIL",
            "PASS" if a["test"]["PASS"] else "FAIL", a["oof_macro_f1"]))
    with open("reports/v4/fast_path_source_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    return train, test, y, ids, test_ids, ntr, nte, folds, z, names, T, t0


def build(y, folds, ntr, nte, z, names, T):
    S = np.stack([z[n] for n in names])
    Tb = np.stack([T[n] for n in names])
    bj = json.load(open("reports/v4/blend_search.json"))

    # verify the stored refined artifact independently reproduces its metric
    a_oof = np.load("cache/v4/blend_refined_oof.npy")
    rep = mf1(y, a_oof)
    stored = bj["refined"]["honest_f1"]
    print("\nstored refined honest_f1 = {:.6f} | recomputed from artifact = {:.6f}"
          .format(stored, rep), "| match:", abs(stored - rep) < 1e-6)
    reuse = abs(stored - rep) < 1e-6

    w_fold = [np.array(bj["refined"]["chosen"][str(k)]["weights"])
              for k in range(5)]
    w_med = np.median(np.array(w_fold), axis=0)
    w_coarse = np.array(bj["coarse"]["chosen"]["0"]["weights"])

    b_oof = norm(np.tensordot(w_med, S, 1))
    c_oof = norm(np.tensordot(w_coarse, S, 1))
    if not reuse:
        print("  !! stored artifact invalid -> rebuild A from median weights")
        a_oof = b_oof

    a_te = norm(np.tensordot(w_fold[0] * 0 + np.array(
        bj["refined"]["opt_w"]), Tb, 1))
    b_te = norm(np.tensordot(w_med, Tb, 1))
    c_te = norm(np.tensordot(w_coarse, Tb, 1))

    res = {}
    for tag, Po, Pt in (("A", a_oof, a_te), ("B", b_oof, b_te), ("C", c_oof, c_te)):
        fs = [mf1(y[folds == k], Po[folds == k]) for k in range(5)]
        res[tag] = {
            "oof_macro_f1": round(mf1(y, Po), 6),
            "per_class_f1": pcf1(y, Po),
            "accuracy": round(float(accuracy_score(y, Po.argmax(1))), 6),
            "fold_scores": [round(v, 6) for v in fs],
            "fold_mean": round(float(np.mean(fs)), 6),
            "fold_std": round(float(np.std(fs)), 6),
            "worst_fold": round(float(np.min(fs)), 6),
            "confusion": confusion_matrix(y, Po.argmax(1), labels=CLASSES).tolist(),
            "oof_pred_dist": np.bincount(Po.argmax(1), minlength=3).tolist(),
            "test_pred_dist": np.bincount(Pt.argmax(1), minlength=3).tolist(),
            "test_pred_pct": [round(100.0 * float(v), 2) for v in
                              np.bincount(Pt.argmax(1), minlength=3) / len(Pt)],
            "delta_vs_sub2": round(mf1(y, Po) - SUB2_REF, 6),
            "oof_check": check(Po, len(y), tag + "_oof"),
            "test_check": check(Pt, len(Pt), tag + "_test"),
        }
    res["_weights"] = {"per_fold": [list(map(float, w)) for w in w_fold],
                       "median_consensus": [float(v) for v in w_med],
                       "coarse_alt": [float(v) for v in w_coarse],
                       "refined_opt": list(map(float, bj["refined"]["opt_w"]))}
    return a_oof, b_oof, c_oof, a_te, b_te, c_te, res, names


def main_run():
    (train, test, y, ids, test_ids, ntr, nte,
     folds, z, names, T, t0) = main()
    a_oof, b_oof, c_oof, a_te, b_te, c_te, res, names = build(
        y, folds, ntr, nte, z, names, T)

    print("\n{:<4} {:>10} {:>9} {:>9} {:>9} {:>9} {:>10}".format(
        "cand", "macroF1", "cls0", "cls1", "cls2", "acc", "foldstd"))
    print("-" * 74)
    for c in ("A", "B", "C"):
        r = res[c]
        print("{:<4} {:>10.6f} {:>9.6f} {:>9.6f} {:>9.6f} {:>9.6f} {:>10.6f}".format(
            c, r["oof_macro_f1"], *r["per_class_f1"], r["accuracy"], r["fold_std"]))

    dab = float((a_te.argmax(1) != b_te.argmax(1)).mean())
    dac = float((a_te.argmax(1) != c_te.argmax(1)).mean())
    maxdiff = float(np.abs(a_te - b_te).max())
    print("\ntest disagreement A-B {:.4f} | A-C {:.4f}".format(dab, dac))
    print("max |A_test - B_test| = {:.6f}".format(maxdiff))
    print("test pred dist A={} B={} C={}".format(
        res["A"]["test_pred_dist"], res["B"]["test_pred_dist"],
        res["C"]["test_pred_dist"]))

    c_ok = (res["A"]["oof_macro_f1"] - res["C"]["oof_macro_f1"] <= 0.003
            and dac >= 0.01)
    print("Candidate C rule: within 0.003 of A and >=1%% test disagreement ->",
          c_ok)

    np.save(OUT + "/candidate_A_oof_probs.npy", a_oof)
    np.save(OUT + "/CANDIDATE_NOT_FOR_SUBMISSION_A_test_probs.npy", a_te)
    np.save(OUT + "/candidate_B_oof_probs.npy", b_oof)
    np.save(OUT + "/CANDIDATE_NOT_FOR_SUBMISSION_B_test_probs.npy", b_te)
    for n2, v in T.items():
        np.save(OUT + "/CANDIDATE_NOT_FOR_SUBMISSION_source_{}_test_probs.npy"
                .format(n2), v)
    if c_ok:
        np.save(OUT + "/candidate_C_oof_probs.npy", c_oof)
        np.save(OUT + "/CANDIDATE_NOT_FOR_SUBMISSION_C_test_probs.npy", c_te)
    else:
        for p in (OUT + "/candidate_C_oof_probs.npy",
                  OUT + "/CANDIDATE_NOT_FOR_SUBMISSION_C_test_probs.npy"):
            if os.path.exists(p):
                os.remove(p)

    res["_test_disagreement"] = {"A_B": round(dab, 6), "A_C": round(dac, 6),
                                 "max_abs_diff_A_B": round(maxdiff, 6)}
    res["_candidate_C_created"] = bool(c_ok)
    res["_sources"] = names
    res["_runtime_s"] = round(time.time() - t0, 1)
    res["_hashes"] = {}
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".npy") and ("A_" in f or "B_" in f or "C_" in f
                                    or "_A_" in f or "_B_" in f or "_C_" in f):
            res["_hashes"][f] = sha(os.path.join(OUT, f))
    with open("reports/v4/fast_path_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\n-> reports/v4/fast_path_metrics.json")
    print("artifacts:")
    for f in sorted(os.listdir(OUT)):
        print("   ", f)
    return res


if __name__ == "__main__":
    main_run()

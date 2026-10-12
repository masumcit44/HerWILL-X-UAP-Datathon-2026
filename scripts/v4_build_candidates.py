"""v4 Phase 7+9: build the three-candidate portfolio and save all artifacts.

TEST probabilities are saved ONLY as CANDIDATE_NOT_FOR_SUBMISSION_*.npy.
No submission CSV is created. No Kaggle upload.
"""
import os
import re
import sys
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
for d in ("reports/v4", "predictions/v4_candidates", "models/v4", "cache/v4"):
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


def text_test_probs(train_clean, test_clean, y):
    w = TfidfVectorizer(**WORD_KW).fit(train_clean)
    c = TfidfVectorizer(**CHAR_KW).fit(train_clean)
    Xtr = sp.hstack([w.transform(train_clean), c.transform(train_clean)],
                    format="csr").astype(np.float32)
    Xte = sp.hstack([w.transform(test_clean), c.transform(test_clean)],
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


def main():
    t0 = time.time()
    train = pd.read_csv("train.csv", encoding="utf-8")
    test = pd.read_csv("test.csv", encoding="utf-8")
    train["text"] = train["text"].astype("string").fillna("").astype(str)
    test["text"] = test["text"].astype("string").fillna("").astype(str)
    y = train["y"].to_numpy(int)
    ids = train["id"].to_numpy(np.int64)
    test_ids = test["id"].to_numpy(np.int64)

    z = np.load("cache/v4/sources_oof.npz")
    folds, sub2 = z["fold_ids"], z["sub2"]
    names = [k for k in z.files
             if k not in ("fold_ids", "y", "sub2", "text_base", "text_all3")]
    print("OOF sources available:", names)

    print("\nbuilding TEST-side sources ...")
    tc_base = test["text"].map(clean_base).to_numpy()
    tc_all3 = test["text"].map(clean_all3).to_numpy()
    tr_base = train["text"].map(clean_base).to_numpy()
    tr_all3 = train["text"].map(clean_all3).to_numpy()
    t_text = norm(0.5 * text_test_probs(tr_base, tc_base, y)
                  + 0.5 * text_test_probs(tr_all3, tc_all3, y))
    t_id = id_prior_test(test_ids, ids, y)
    T = {"text": t_text, "id": t_id}

    emb_s_tr = np.load("cache/v3/emb_e5_small_train.npy")
    emb_s_te = np.load("cache/v3/emb_e5_small_test.npy")
    T["e5_small"] = LogisticRegression(**LR_KW).fit(emb_s_tr, y).predict_proba(emb_s_te)
    if "e5_base" in names:
        T["e5_base"] = LogisticRegression(**LR_KW).fit(
            np.load("cache/v4/emb_e5base_train.npy"), y
        ).predict_proba(np.load("cache/v4/emb_e5base_test.npy"))
    for k2, v in T.items():
        assert v.shape == (len(test), 3) and np.isfinite(v).all()
        assert np.allclose(v.sum(1), 1.0, atol=1e-6)
        np.save("predictions/v4_candidates/CANDIDATE_NOT_FOR_SUBMISSION_"
                "source_{}_test_probs.npy".format(k2), v)
    print("  test sources:", {k2: v.shape for k2, v in T.items()})

    S = {n: z[n] for n in names}
    Sb = np.stack([S[n] for n in names])
    Tb = np.stack([T[n] for n in names])
    bj = json.load(open("reports/v4/blend_search.json"))
    oj = json.load(open("reports/v4/logit_offsets.json")) if os.path.exists(
        "reports/v4/logit_offsets.json") else None
    w_med = np.median(np.array([bj["refined"]["chosen"][str(k)]["weights"]
                                for k in range(5)]), axis=0)
    w_coarse = np.array(bj["coarse"]["chosen"]["0"]["weights"])
    w_fine = np.array(bj["refined"]["opt_w"])

    A_oof = np.load("predictions/v4_candidates/"
                    "candidate_logit_offset_oof_probs.npy") if (
        oj and oj.get("keep")) else np.load(
        "models/v3/v3_3_frozen_e5_small/refined_oof_probs.npy")
    A_desc = ("refined blend + per-class logit offsets" if (oj and oj.get("keep"))
              else "refined blend (offsets rejected)")
    B_oof = norm(np.tensordot(w_med, Sb, 1))
    C_oof = norm(np.tensordot(w_coarse, Sb, 1))
    cands = {"A": (A_oof, A_desc),
             "B": (B_oof, "consensus median fold weights "
                  + str([round(float(v), 3) for v in w_med])),
             "C": (C_oof, "coarse-step weights "
                  + str([round(float(v), 3) for v in w_coarse]) + " (alt simplex cell)")}

    print("\n{:<4} {:>10} {:>9} {:>9} {:>9} {:>9}  {}".format(
        "cand", "macroF1", "cls0", "cls1", "cls2", "foldstd", "descr"))
    print("-" * 100)
    summary = {}
    for c, (P, desc) in cands.items():
        fs = [mf1(y[folds == k], P[folds == k]) for k in range(N_SPLITS)]
        print("{:<4} {:>10.6f} {:>9.6f} {:>9.6f} {:>9.6f} {:>9.6f}  {}".format(
            c, mf1(y, P), *pcf1(y, P), float(np.std(fs)), desc))
        summary[c] = {"macro_f1": round(mf1(y, P), 6),
                      "delta_vs_sub2": round(mf1(y, P) - SUB2_REF, 6),
                      "per_class_f1": pcf1(y, P),
                      "fold_scores": [round(v, 6) for v in fs],
                      "fold_mean": round(float(np.mean(fs)), 6),
                      "fold_std": round(float(np.std(fs)), 6),
                      "worst_fold": round(float(np.min(fs)), 6),
                      "accuracy": round(float(accuracy_score(y, P.argmax(1))), 6),
                      "confusion": confusion_matrix(y, P.argmax(1),
                                                    labels=CLASSES).tolist(),
                      "pred_dist": np.bincount(P.argmax(1), minlength=3).tolist(),
                      "description": desc}
        np.save("predictions/v4_candidates/candidate_{}_oof_probs.npy".format(c), P)

    for c in ("A", "B", "C"):
        if c == "A":
            Tc = norm(np.tensordot(w_fine, Tb, 1))
            if oj and oj.get("keep"):
                b1 = float(np.median([oj["per_fold_offsets"][str(k)]["b1"]
                                      for k in range(5)]))
                b2 = float(np.median([oj["per_fold_offsets"][str(k)]["b2"]
                                      for k in range(5)]))
                Tc = apply_bias(Tc, b1, b2)
        elif c == "B":
            Tc = norm(np.tensordot(w_med, Tb, 1))
        else:
            Tc = norm(np.tensordot(w_coarse, Tb, 1))
        assert Tc.shape == (len(test), 3) and np.isfinite(Tc).all()
        assert np.allclose(Tc.sum(1), 1.0, atol=1e-6)
        np.save("predictions/v4_candidates/CANDIDATE_NOT_FOR_SUBMISSION_"
                "{}_test_probs.npy".format(c), Tc)

    ta = np.load("predictions/v4_candidates/"
                 "CANDIDATE_NOT_FOR_SUBMISSION_A_test_probs.npy")
    tbb = np.load("predictions/v4_candidates/"
                  "CANDIDATE_NOT_FOR_SUBMISSION_B_test_probs.npy")
    tc = np.load("predictions/v4_candidates/"
                 "CANDIDATE_NOT_FOR_SUBMISSION_C_test_probs.npy")
    dab = float((ta.argmax(1) != tbb.argmax(1)).mean())
    dac = float((ta.argmax(1) != tc.argmax(1)).mean())
    dbc = float((tbb.argmax(1) != tc.argmax(1)).mean())
    print("\nTEST disagreement: A-B {:.4f} | A-C {:.4f} | B-C {:.4f}"
          .format(dab, dac, dbc))
    print("test pred dist: A={} B={} C={}".format(
        np.bincount(ta.argmax(1), minlength=3).tolist(),
        np.bincount(tbb.argmax(1), minlength=3).tolist(),
        np.bincount(tc.argmax(1), minlength=3).tolist()))
    summary["_test_disagreement"] = {"A_B": round(dab, 6), "A_C": round(dac, 6),
                                     "B_C": round(dbc, 6)}

    with open("reports/v4/candidate_portfolio_metrics.json", "w") as f:
        json.dump({"sources": names, "sub2_ref": SUB2_REF,
                   "weights": {"median_consensus": [float(v) for v in w_med],
                               "refined_opt": [float(v) for v in w_fine],
                               "coarse_alt": [float(v) for v in w_coarse]},
                   "candidates": summary,
                   "runtime_s": round(time.time() - t0, 1)}, f, indent=2)
    with open("reports/v4/candidate_weight_stability.json", "w") as f:
        json.dump({"per_fold_refined_weights":
                   {str(k): bj["refined"]["chosen"][str(k)]["weights"]
                    for k in range(5)},
                   "median_consensus": [float(v) for v in w_med],
                   "coarse_alt": [float(v) for v in w_coarse]}, f, indent=2)
    print("\n-> reports/v4/candidate_portfolio_metrics.json")


if __name__ == "__main__":
    main()

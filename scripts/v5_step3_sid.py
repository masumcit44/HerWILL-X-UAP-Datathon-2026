"""v5 Step 3: similarity-weighted ID prior ("sid").

Predeclared PRIMARY config A: KW=120, k=180, beta=10, gamma=1.0.
Sensitivity configs B and C are computed for reporting only; never selected by score.
"""
import os
import re
import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, confusion_matrix

SEED = 42
N_SPLITS = 5
CLASSES = [0, 1, 2]
WORK = "cache/v5"
for d in (WORK, "reports/v5", "predictions/v5_candidates", "models/v5"):
    os.makedirs(d, exist_ok=True)

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+", flags=re.UNICODE)

CONFIG_A = dict(KW=120, k=180, beta=10.0, gamma=1.0)
CONFIG_B = dict(KW=120, k=90, beta=8.0, gamma=0.75)
CONFIG_C = dict(KW=120, k=120, beta=10.0, gamma=1.0)


def clean_text(t):
    s = "" if t is None else str(t)
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    return WS_RE.sub(" ", s).strip()


def mf1(y, p):
    return float(f1_score(y, np.asarray(p).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def pcf1(y, p):
    return [round(float(v), 6) for v in
            f1_score(y, np.asarray(p).argmax(1), average=None, labels=CLASSES,
                     zero_division=0)]


def cos_pairs(Xq_sub, Xtr_n, n_idx):
    """Row-wise cosine between Xq_sub[i] and Xtr_n[n_idx[i, :]] -> (m, k)."""
    m, k = n_idx.shape
    gathered = Xtr_n[n_idx.ravel()]
    rep = Xq_sub[np.repeat(np.arange(m), k)]
    prod = gathered.multiply(rep)
    s = np.asarray(prod.sum(axis=1)).ravel().reshape(m, k)
    return s


def sid(q_ids, Xq, pool_rows, ids, y, KW=120, k=180, beta=10.0, gamma=1.0,
        chunk=800, sub=100):
    """Similarity-weighted k-NN ID prior."""
    global _XTRAIN_N
    Xtr_n = _XTRAIN_N
    o = np.argsort(ids[pool_rows], kind="stable")
    pid = ids[pool_rows][o]
    prow = pool_rows[o]
    py = y[prow]
    npool = len(pid)
    out = np.zeros((len(q_ids), 3), dtype=np.float64)
    for cs in range(0, len(q_ids), chunk):
        ce = min(cs + chunk, len(q_ids))
        qid = q_ids[cs:ce]
        pos = np.searchsorted(pid, qid)
        cand = pos[:, None] + np.arange(-KW, KW)[None, :]
        valid = (cand >= 0) & (cand < npool)
        cc = np.clip(cand, 0, npool - 1)
        d = np.abs(pid[cc] - qid[:, None]).astype(np.float64)
        d = np.where(valid, d, np.inf)
        order = np.argsort(d, axis=1, kind="stable")[:, :k]
        dsel = np.take_along_axis(d, order, axis=1)
        nrow = prow[np.take_along_axis(cc, order, axis=1)]
        sim = np.zeros_like(dsel)
        for ss in range(0, ce - cs, sub):
            se = min(ss + sub, ce - cs)
            sim[ss:se] = cos_pairs(Xq[cs + ss:cs + se], Xtr_n, nrow[ss:se])
        good = np.isfinite(dsel)
        w = np.where(good, 1.0 / np.power(np.where(good, dsel, 0.0) + 1e-6,
                                          gamma), 0.0) * np.exp(beta * sim)
        w = np.where(good, w, 0.0)
        ny = y[nrow]
        P = np.stack([(w * (ny == c)).sum(axis=1) for c in CLASSES], axis=1)
        tot = P.sum(axis=1, keepdims=True)
        P = P / np.where(tot > 0, tot, 1.0)
        P[tot.ravel() <= 0] = 1.0 / 3
        out[cs:ce] = P
    return out


def main():
    global _XTRAIN_N
    t0 = time.time()
    d12 = np.load(WORK + "/step12.npz", allow_pickle=True)
    y = d12["y"]
    ids = d12["ids"]
    test_ids = d12["test_ids"]
    folds = d12["fold_ids"]
    train = pd.read_csv("train.csv", encoding="utf-8")
    test = pd.read_csv("test.csv", encoding="utf-8")
    train["text"] = train["text"].astype("string").fillna("").astype(str)
    test["text"] = test["text"].astype("string").fillna("").astype(str)
    clean = train["text"].map(clean_text).to_numpy()
    clean_te = test["text"].map(clean_text).to_numpy()

    print("fitting label-free SIM_VEC (char_wb 2-4, 80k) on train_clean ...")
    sv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                         sublinear_tf=True, max_features=80000,
                         dtype=np.float32)
    sv.fit(clean)
    Xtr_n = normalize(sv.transform(clean)).tocsr()
    Xte_n = normalize(sv.transform(clean_te)).tocsr()
    _XTRAIN_N = Xtr_n
    print("  train sim matrix", Xtr_n.shape, "| test", Xte_n.shape,
          "| vocab", len(sv.vocabulary_))

    results = {}
    store = {}
    for tag, cfg in (("A", CONFIG_A), ("B", CONFIG_B), ("C", CONFIG_C)):
        ts = time.time()
        oof = np.zeros((len(y), 3))
        for k in range(N_SPLITS):
            va = np.flatnonzero(folds == k)
            tr = np.flatnonzero(folds != k)
            assert set(ids[va].tolist()).isdisjoint(ids[tr].tolist())
            oof[va] = sid(ids[va], Xtr_n, tr, ids, y, **cfg)
        te = sid(test_ids, Xte_n, np.arange(len(y)), ids, y, **cfg)
        s = mf1(y, oof)
        print("\n[sid config {}] {} -> ID-only OOF macro F1 = {:.6f}  ({:.0f}s)".format(
            tag, cfg, s, time.time() - ts))
        print("   per-class {}".format(pcf1(y, oof)))
        print("   fold scores  {}".format(
            [round(mf1(y[folds == k], oof[folds == k]), 6) for k in range(5)]))
        results[tag] = {"cfg": cfg, "id_only_oof": round(s, 6),
                        "per_class": pcf1(y, oof),
                        "folds": [round(mf1(y[folds == k], oof[folds == k]), 6)
                                  for k in range(5)]}
        if tag == "A":
            store["oof"] = oof
            store["test"] = te
        np.save(WORK + "/sid_{}_oof.npy".format(tag), oof)
        np.save(WORK + "/sid_{}_test.npy".format(tag), te)

    a = results["A"]["id_only_oof"]
    print("\nconfig A expectation 0.6280 +-0.001 -> diff {:.6f}".format(abs(a - 0.6280)))
    ok = abs(a - 0.6280) <= 0.001
    print("SID CONFIG A GATE:", "PASS" if ok else "FAIL")
    if not ok:
        print("*** STOP: predeclared config A did not reach the expected band ***")
        return

    np.savez_compressed(WORK + "/sid.npz", oof=store["oof"], test=store["test"],
                        fold_ids=folds, y=y, ids=ids, test_ids=test_ids)
    with open("reports/v5/sid.json", "w") as f:
        json.dump({"configs": results, "config_a_gate_pass": bool(ok),
                   "runtime_s": round(time.time() - t0, 1)}, f, indent=2)
    print("\n-> cache/v5/sid.npz, reports/v5/sid.json")


if __name__ == "__main__":
    main()

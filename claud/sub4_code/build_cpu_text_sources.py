"""Build the two CPU text-only sources on the locked Colab fold set (fold set B).

  text_tfidf : word (1,2) + char_wb (3,5) TF-IDF -> balanced LogisticRegression
               (same recipe as Submissions 1-3; standalone OOF ~0.516)
  text_knn   : soft k-nearest-neighbour vote in char_wb (2,4) TF-IDF space.
               Neighbours come from fold-train rows only, weight exp(10 * cosine), k=50.
               The vectoriser is fitted on train text without labels. Standalone OOF ~0.487;
               it adds about +0.012 to a no-ID text+E5 blend.

Neither source reads `id`. Output format matches the Colab export:
  oof (N_train,3), test (N_test,3), fold_ids, coverage_train, coverage_test, meta (json string)

Usage (from the project root):
  python build_cpu_text_sources.py --data-dir . --folds final_sources/fold_ids.npy --out cache/v7_text
Runtime on a laptop CPU: about 5 min (tfidf) + 2 min (knn).
"""
import argparse, hashlib, json, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

EXPECTED_FOLD_INT64 = "cc4fdd169110aeee6d3d5566095234461041b1a9290e2e916ce850c31d27a7ca"
EXPECTED_TRAIN = "a269ad841f824c6e60930cfda8cb25bdce58a6a1ee01fc1fe663d23b19e34449"
EXPECTED_TEST = "cca36f5416abfb8a41b1c84e5fa83aec18ffbff32fecd277c668a3f73312a38f"
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")


def clean_text(t):
    return WS_RE.sub(" ", MENTION_RE.sub(" USER ", URL_RE.sub(" URL ", str(t)))).strip()


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def macro_f1(y, pred):
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    p, r = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    return float((2 * p * r / np.maximum(p + r, 1e-12)).mean())


def tfidf_source(X, XT, y, folds):
    wk = dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60000,
              sublinear_tf=True, dtype=np.float32)
    ck = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60000,
              sublinear_tf=True, dtype=np.float32)
    lk = dict(solver="lbfgs", C=4.0, class_weight="balanced", max_iter=2000, random_state=42)

    def feats(fit, *apply):
        w, c = TfidfVectorizer(**wk).fit(fit), TfidfVectorizer(**ck).fit(fit)
        return [sp.hstack([w.transform(a), c.transform(a)], format="csr") for a in apply]

    oof = np.zeros((len(y), 3))
    for k in range(5):
        tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        A, B = feats(X[tr], X[tr], X[va])
        oof[va] = LogisticRegression(**lk).fit(A, y[tr]).predict_proba(B)
        print(f"  tfidf fold {k} done", flush=True)
    A, B = feats(X, X, XT)
    return oof, LogisticRegression(**lk).fit(A, y).predict_proba(B)


def knn_source(X, XT, y, folds, K=50, beta=10.0, chunk=2000):
    lower = np.array([s.lower() for s in X], dtype=object)
    lower_t = np.array([s.lower() for s in XT], dtype=object)
    V = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True,
                        max_features=150000, dtype=np.float32).fit(lower)   # label-free
    A, AT = normalize(V.transform(lower)).tocsr(), normalize(V.transform(lower_t)).tocsr()

    def vote(Q, P, yp):
        out = np.zeros((Q.shape[0], 3))
        for s in range(0, Q.shape[0], chunk):
            S = (Q[s:s + chunk] @ P.T).toarray()
            idx = np.argpartition(-S, K, axis=1)[:, :K]
            w = np.exp(beta * np.take_along_axis(S, idx, 1))
            yy = yp[idx]
            pr = np.stack([(w * (yy == c)).sum(1) for c in range(3)], 1)
            out[s:s + chunk] = pr / pr.sum(1, keepdims=True)
        return out

    oof = np.zeros((len(y), 3))
    for k in range(5):
        tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        oof[va] = vote(A[va], A[tr], y[tr])       # pool = fold-train rows only
        print(f"  knn fold {k} done", flush=True)
    return oof, vote(AT, A, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--folds", default="final_sources/fold_ids.npy")
    ap.add_argument("--out", default="cache/v7_text")
    a = ap.parse_args()
    d, out = Path(a.data_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert sha256(d / "train.csv") == EXPECTED_TRAIN, "train.csv hash mismatch"
    assert sha256(d / "test.csv") == EXPECTED_TEST, "test.csv hash mismatch"
    folds = np.load(a.folds).astype(np.int64)
    assert hashlib.sha256(folds.tobytes()).hexdigest() == EXPECTED_FOLD_INT64, "wrong fold set"
    tr, te = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    X = tr["text"].astype(str).map(clean_text).to_numpy()
    XT = te["text"].astype(str).map(clean_text).to_numpy()
    y = tr["y"].to_numpy(np.int64)

    for name, fn in (("text_tfidf", tfidf_source), ("text_knn", knn_source)):
        t0 = time.time()
        oof, test = fn(X, XT, y, folds)
        assert oof.shape == (len(y), 3) and test.shape == (len(te), 3)
        assert np.isfinite(oof).all() and np.isfinite(test).all()
        assert np.abs(oof.sum(1) - 1).max() < 1e-6 and np.abs(test.sum(1) - 1).max() < 1e-6
        score = macro_f1(y, oof.argmax(1))
        meta = dict(name=name, uses_id=False, fold_int64_sha256=EXPECTED_FOLD_INT64,
                    oof_macro_f1=score, seconds=round(time.time() - t0))
        np.savez(out / f"{name}_final.npz", oof=oof, test=test, fold_ids=folds,
                 coverage_train=np.ones(len(y), bool), coverage_test=np.ones(len(te), bool),
                 meta=np.array(json.dumps(meta)))
        print(f"{name}: OOF macro-F1 {score:.6f} -> {out / (name + '_final.npz')}")


if __name__ == "__main__":
    main()

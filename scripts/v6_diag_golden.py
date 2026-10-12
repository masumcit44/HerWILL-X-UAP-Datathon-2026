"""Diagnose the golden-test diff: float32 vs float64 similarity accumulation."""
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def clean_text(t):
    s = "" if t is None else str(t)
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    return WS_RE.sub(" ", s).strip()


GOLDEN = np.array([
    [0.915507, 0.084493, 0.000000],
    [0.823588, 0.160618, 0.015794],
    [0.722480, 0.254924, 0.022597],
    [0.850096, 0.095842, 0.054063],
    [0.695871, 0.278242, 0.025887]])
ROWS = [1, 4, 8, 9, 13]


def sid_golden(q_ids, Xq, pool_rows, ids, y, Xpool, KW=120, k=180, beta=10.0, gamma=1.0):
    o = np.argsort(ids[pool_rows], kind="stable")
    pid = ids[pool_rows][o]
    prow = pool_rows[o]
    assert len(np.intersect1d(q_ids, pid)) == 0
    qi = q_ids
    xq = Xq
    pos = np.searchsorted(pid, qi)
    cand = pos[:, None] + np.arange(-KW, KW)[None, :]
    valid = (cand >= 0) & (cand < len(pid))
    cc = np.clip(cand, 0, len(pid) - 1)
    d = np.abs(pid[cc] - qi[:, None]).astype(float)
    d[~valid] = np.inf
    order = np.argsort(d, axis=1, kind="stable")[:, :k]
    dsel = np.take_along_axis(d, order, 1)
    nrow = prow[np.take_along_axis(cc, order, 1)]
    qrep = np.repeat(np.arange(xq.shape[0]), nrow.shape[1])
    sim = np.asarray(xq[qrep].multiply(Xpool[nrow.ravel()]).sum(1)).reshape(nrow.shape)
    w = np.where(np.isfinite(dsel), 1.0 / ((np.where(np.isfinite(dsel), dsel, 1.0) + 1e-6) ** gamma), 0.0) * np.exp(beta * sim)
    yy = y[nrow]
    P = np.stack([(w * (yy == c)).sum(1) for c in range(3)], 1)
    tot = P.sum(1, keepdims=True)
    return np.where(tot > 0, P / np.where(tot > 0, tot, 1), 1 / 3), sim


train = pd.read_csv("train.csv", encoding="utf-8")
train["text"] = train["text"].astype("string").fillna("").astype(str)
y = train["y"].to_numpy(int)
ids = train["id"].to_numpy(np.int64)
clean = train["text"].map(clean_text).to_numpy()
import io
import zipfile
folds = np.load(io.BytesIO(zipfile.ZipFile(
    "uap_complete_export_20260928_064140.zip").read("final_sources/fold_ids.npy")))
tr0 = np.flatnonzero(folds != 0)
g_ids = ids[ROWS]
print("rows", ROWS, "ids", g_ids.tolist(), "folds", folds[ROWS].tolist())

# vocabulary cap check: if max_features=80000 actually truncates, the retained
# vocabulary depends on term-frequency tie-breaking, which varies across versions.
V0 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                     sublinear_tf=True, max_features=80000, dtype=np.float32)
V0.fit(clean)
full = len(V0.vocabulary_)
print("vocabulary retained with max_features=80000:", full, "(CAPPED)" if full >= 80000
      else "(not capped)")
Vraw = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                       sublinear_tf=True, dtype=np.float32)
Xraw = Vraw.fit_transform(clean)
df = np.asarray((Xraw > 0).sum(axis=0)).ravel()
srt = np.sort(df)[::-1]
print("doc-freq at rank 80000:", int(srt[79999]), "| rank 80001:", int(srt[80000]),
      "| tie at boundary:", int(srt[79999] == srt[80000]),
      "| terms tied at that freq:", int((df == srt[80000]).sum()))


for dt in (np.float32, np.float64):
    V = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                        sublinear_tf=True, max_features=80000, dtype=dt).fit(clean)
    Xn = normalize(V.transform(clean)).tocsr()
    P, sim = sid_golden(g_ids, Xn[ROWS], tr0, ids, y, Xn)
    print("\ndtype={} max|diff|={:.3e}".format(np.dtype(dt).name, np.abs(P - GOLDEN).max()))
    print(np.array2string(P, precision=6, suppress_small=True))
    print("  per-row max diff:", np.round(np.abs(P - GOLDEN).max(1), 7).tolist())

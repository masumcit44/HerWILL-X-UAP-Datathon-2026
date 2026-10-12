"""v6 Step 5 (OPTIONAL, time-boxed): sid_e5 - sid_prior with E5-base embedding similarity.

Predeclared adoption rule: adopt ONLY if honest OOF improves >= +0.002 over config A
AND at least 4 folds improve. Otherwise keep config A.
"""
import os
import json
import time
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score

CLASSES = [0, 1, 2]
N_SPLITS = 5
WORK = "cache/v6"


def mf1(y, p):
    return float(f1_score(y, np.asarray(p).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def sid_prior_e5(q_ids, Xq, pool_rows, ids, y, Xpool, KW=120, k=180, beta=10.0,
                 gamma=1.0, chunk=200):
    """SAME logic as the reference sid_prior, but similarity = dot product of E5
    embeddings. Xq / Xpool are L2-normalised embedding matrices. chunk<=200 to bound
    the intermediate gather."""
    o = np.argsort(ids[pool_rows], kind="stable")
    pid = ids[pool_rows][o]
    prow = pool_rows[o]
    assert len(np.intersect1d(q_ids, pid)) == 0, "query ids must not be in the pool"
    out = np.zeros((len(q_ids), 3))
    for s in range(0, len(q_ids), chunk):
        qi = q_ids[s:s + chunk]
        xq = np.asarray(Xq[s:s + chunk])
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
        sim = np.einsum("ij,ij->i", xq[qrep], Xpool[nrow.ravel()]).reshape(nrow.shape)
        w = np.where(np.isfinite(dsel),
                     1.0 / ((np.where(np.isfinite(dsel), dsel, 1.0) + 1e-6) ** gamma),
                     0.0) * np.exp(beta * sim)
        yy = y[nrow]
        P = np.stack([(w * (yy == c)).sum(1) for c in range(3)], 1)
        tot = P.sum(1, keepdims=True)
        out[s:s + chunk] = np.where(tot > 0, P / np.where(tot > 0, tot, 1), 1 / 3)
    return out


def main():
    d = np.load(WORK + "/step123.npz", allow_pickle=True)
    y, folds = d["y"], d["fold_ids"]
    ids, test_ids = d["ids"], d["test_ids"]
    Etr = normalize(np.load("cache/v4/emb_e5base_train.npy")).astype(np.float32)
    Ete = normalize(np.load("cache/v4/emb_e5base_test.npy")).astype(np.float32)
    print("E5 train {} test {} (label-free, row order = train.csv / test.csv)".format(
        Etr.shape, Ete.shape))
    assert Etr.shape[0] == len(y) and Ete.shape[0] == len(test_ids)

    out = {}
    for beta in (4.0, 8.0, 12.0):
        for k in (90, 180):
            t0 = time.time()
            oof = np.zeros((len(y), 3))
            for f in range(N_SPLITS):
                va = np.flatnonzero(folds == f)
                tr = np.flatnonzero(folds != f)
                oof[va] = sid_prior_e5(ids[va], Etr[va], tr, ids, y, Etr,
                                      KW=120, k=k, beta=beta, gamma=1.0)
            s = mf1(y, oof)
            fs = [mf1(y[folds == f], oof[folds == f]) for f in range(N_SPLITS)]
            out["beta{}_k{}".format(beta, k)] = {
                "id_only_oof": round(s, 6), "folds": [round(v, 6) for v in fs],
                "oof": None}
            np.save(WORK + "/sidE5_beta{}_k{}_oof.npy".format(beta, k), oof)
            print("  beta={:<5} k={:<4} ID-only OOF {:.6f}  ({:.0f}s)".format(
                beta, k, s, time.time() - t0))
    with open("reports/v6/sid_e5.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-> reports/v6/sid_e5.json, cache/v6/sidE5_*_oof.npy")


if __name__ == "__main__":
    main()

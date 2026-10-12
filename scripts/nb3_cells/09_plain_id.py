# Notebook cell 09 - code: plain ID prior
# --- 4. plain ID prior (k=20, inverse distance, fold-train pool only) --------
def id_prior(q_ids, pool_ids, pool_y, k=20, eps=1e-6):
    """Weighted vote over the k nearest ids on each side of each query id.

    The pool contains ONLY fold-training ids, so no validation/test label is
    ever visible. The assert below is a hard guard against self-inclusion.
    """
    q = np.asarray(q_ids, np.int64).ravel()
    t = np.asarray(pool_ids, np.int64).ravel()
    yy = np.asarray(pool_y, np.int64).ravel()
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
    wt = np.where(good, 1.0 / (np.where(good, dsel, 0.0) + eps), 0.0)
    P = np.stack([(wt * (ysel == c)).sum(axis=1) for c in CLASSES], axis=1)
    tot = P.sum(axis=1, keepdims=True)
    P = P / np.where(tot > 0, tot, 1.0)
    P[tot.ravel() <= 0] = 1.0 / 3
    assert int(np.isin(q, s_ids).sum()) == 0, "leak: a query id is inside the pool"
    return P


oof_plain = np.zeros((n_tr, 3))
for k in range(N_SPLITS):
    va = np.flatnonzero(folds == k)
    tr = np.flatnonzero(folds != k)
    oof_plain[va] = id_prior(ids[va], ids[tr], y[tr])
plain_only = macro_f1(y, oof_plain)
print("plain-ID OOF macro-F1 = {:.6f} (expected 0.613631)".format(plain_only))
assert abs(plain_only - 0.613631) <= 1e-5

test_plain = id_prior(test_ids, ids, y)
print("test plain-ID probs", test_plain.shape)

# Notebook cell 11 - code: sid prior functions
# --- 5. similarity-weighted ID prior -----------------------------------------
def build_sim_matrix(train_clean):
    """Label-free char TF-IDF, L2-normalised rows. Fit on ALL train text (no labels)."""
    V = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                        sublinear_tf=True, max_features=80000,
                        dtype=np.float32).fit(train_clean)
    return V, normalize(V.transform(train_clean)).tocsr()


def sid_prior(q_ids, Xq, pool_rows, ids, y, Xpool, KW=120, k=180, beta=10.0,
              gamma=1.0, chunk=800):
    """q_ids: ids of the query rows; Xq: L2-normalised TF-IDF rows of the SAME
    queries in the SAME ORDER; pool_rows: global train-row indices allowed as
    neighbours."""
    o = np.argsort(ids[pool_rows], kind="stable")
    pid = ids[pool_rows][o]
    prow = pool_rows[o]
    assert len(np.intersect1d(q_ids, pid)) == 0, "query ids must not be in the pool"
    out = np.zeros((len(q_ids), 3))
    for s in range(0, len(q_ids), chunk):
        qi = q_ids[s:s + chunk]
        xq = Xq[s:s + chunk]
        pos = np.searchsorted(pid, qi)
        cand = pos[:, None] + np.arange(-KW, KW)[None, :]
        valid = (cand >= 0) & (cand < len(pid))
        cc = np.clip(cand, 0, len(pid) - 1)
        d = np.abs(pid[cc] - qi[:, None]).astype(float)
        d[~valid] = np.inf
        order = np.argsort(d, axis=1, kind="stable")[:, :k]
        dsel = np.take_along_axis(d, order, 1)
        nrow = prow[np.take_along_axis(cc, order, 1)]
        # row i of xq <-> its own neighbours: repeat keeps the pairing explicit
        qrep = np.repeat(np.arange(xq.shape[0]), nrow.shape[1])
        sim = np.asarray(xq[qrep].multiply(Xpool[nrow.ravel()]).sum(1)).reshape(nrow.shape)
        w = np.where(np.isfinite(dsel),
                     1.0 / ((np.where(np.isfinite(dsel), dsel, 1.0) + 1e-6) ** gamma),
                     0.0) * np.exp(beta * sim)
        yy = y[nrow]
        P = np.stack([(w * (yy == c)).sum(1) for c in range(3)], 1)
        tot = P.sum(1, keepdims=True)
        out[s:s + chunk] = np.where(tot > 0, P / np.where(tot > 0, tot, 1), 1 / 3)
    return out


V, Xn = build_sim_matrix(clean_tr)
print("similarity matrix", Xn.shape)

# Notebook cell 14 - code: honest cross-fitted blend
# --- 6. honest cross-fitted blend over [text, sid, e5_base, xlmr_idhead] -----
def norm_rows(P):
    return P / P.sum(axis=1, keepdims=True)


def simplex_vectors(n, t):
    """All integer compositions of t into n parts, lexicographic order."""
    if n == 1:
        yield (t,)
        return
    for i in range(t + 1):
        for rest in simplex_vectors(n - 1, t - i):
            yield (i,) + rest


T = int(round(1.0 / SIMPLEX_STEP))
GRID = [tuple(np.array(c, dtype=float) / T) for c in simplex_vectors(4, T)]
print("simplex candidates:", len(GRID))


def honest_blend(S, y, folds, grid):
    """Per outer fold: pick weights on the other folds, apply to the held-out fold."""
    out = np.zeros((len(y), 3))
    chosen, scores = {}, []
    for k in range(N_SPLITS):
        va, tr = folds == k, folds != k
        Sr, yr = S[:, tr], y[tr]
        best = (-1.0, None)
        for w in grid:
            sc = macro_f1(yr, norm_rows(np.tensordot(w, Sr, 1)))
            if sc > best[0]:
                best = (sc, w)
        chosen[str(k)] = {"w": [float(v) for v in best[1]], "train_f1": round(best[0], 6)}
        out[va] = norm_rows(np.tensordot(best[1], S[:, va], 1))
        scores.append(macro_f1(y[va], out[va]))
    opt = (-1.0, None)
    for w in grid:
        sc = macro_f1(y, norm_rows(np.tensordot(w, S, 1)))
        if sc > opt[0]:
            opt = (sc, w)
    return out, macro_f1(y, out), scores, chosen, opt


S = np.stack([oof_text, oof_sid, e5_oof, xlmr_oof])
blend_oof, blend_f1, blend_folds, blend_w, optimistic = honest_blend(S, y, folds, GRID)
blend_std = float(np.std(blend_folds))
worst = float(np.min(blend_folds))
print("HONEST blend OOF macro-F1 = {:.6f} (expected ~0.6827 +-0.003)".format(blend_f1))
print("  per-class F1", per_class_f1(y, blend_oof))
print("  accuracy    {:.6f}".format(accuracy_score(y, blend_oof.argmax(1))))
print("  confusion (rows = true 0/1/2)")
print(confusion_matrix(y, blend_oof.argmax(1), labels=CLASSES))
print("  fold scores", [round(v, 6) for v in blend_folds])
print("  fold std    {:.6f} | worst {:.6f}".format(blend_std, worst))
print("  per-fold w ", {k: v["w"] for k, v in blend_w.items()})
print("  OPTIMISTIC (same-OOF, not a performance claim) = {:.6f}".format(optimistic[0]))
assert abs(blend_f1 - CONFIG["honest_blend_oof_macro_f1"]) <= 0.002, \
    "recomputed honest OOF drifted from the recorded value"

# comparison blend: plain ID prior in place of sid
Sp = np.stack([oof_text, oof_plain, e5_oof, xlmr_oof])
_, plain_blend_f1, plain_folds, _, _ = honest_blend(Sp, y, folds, GRID)
print("plain-ID blend OOF macro-F1 = {:.6f}".format(plain_blend_f1))

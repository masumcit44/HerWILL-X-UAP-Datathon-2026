"""v4 Phase 3b: honest cross-fitted simplex blend search over verified sources."""
import os
import json
import hashlib
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

SEED = 42
N_SPLITS = 5
CLASSES = [0, 1, 2]
SUB2_REF = 0.657612
for d in ("reports/v4", "cache/v4"):
    os.makedirs(d, exist_ok=True)


def mf1(y, p):
    return float(f1_score(y, np.asarray(p).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def pcf1(y, p):
    return [float(v) for v in f1_score(y, np.asarray(p).argmax(1), average=None,
                                       labels=CLASSES, zero_division=0)]


def norm(P):
    return P / P.sum(axis=1, keepdims=True)


def simplex(n, t):
    if n == 1:
        yield (t,)
        return
    for i in range(t + 1):
        for rest in simplex(n - 1, t - i):
            yield (i,) + rest




def grid_from(weights, span, step):
    steps = np.arange(0, 1.0001, step)
    dims = []
    for a, w in zip(weights, span):
        lo = max(0.0, w - a)
        hi = min(1.0, w + a)
        dims.append(steps[(steps >= lo - 1e-9) & (steps <= hi + 1e-9)])
    if any(len(d) == 0 for d in dims):
        return []
    out = set()
    mesh = np.array(np.meshgrid(*dims, indexing="ij"))
    for combo in mesh.reshape(len(dims), -1).T:
        w = np.asarray(combo, dtype=float)
        if (w >= -1e-9).all() and abs(w.sum() - 1.0) <= 2 * step:
            out.add(tuple(np.round(w / w.sum(), 4)))
    return sorted(out)


def honest_search(S, y, folds, grid, tag):
    nsrc = S.shape[0]
    honest = np.zeros((len(y), 3))
    chosen, fold_scores = {}, []
    for k in range(N_SPLITS):
        va, tr = folds == k, folds != k
        Sr, yr = S[:, tr], y[tr]
        best = (-1.0, None)
        for w in grid:
            s = mf1(yr, np.tensordot(w, Sr, 1))
            if s > best[0]:
                best = (s, w)
        w = best[1]
        chosen[str(k)] = {"weights": [float(v) for v in w],
                          "train_f1": round(best[0], 6)}
        P = norm(np.tensordot(w, S[:, va], 1))
        honest[va] = P
        fold_scores.append(mf1(y[va], P))
    opt = (-1.0, None)
    for w in grid:
        s = mf1(y, np.tensordot(w, S, 1))
        if s > opt[0]:
            opt = (s, w)
    hf = mf1(y, honest)
    print("\n[{}] {} grid points".format(tag, len(grid)))
    print("  optimistic same-OOF : {:.6f}  w={}".format(
        opt[0], [round(float(v), 3) for v in opt[1]]))
    print("  HONEST cross-fitted : {:.6f}   delta vs 0.657612 = {:+.6f}".format(
        hf, hf - SUB2_REF))
    for k in sorted(chosen):
        print("    fold {}: {}  (train {:.6f})".format(
            k, [round(v, 3) for v in chosen[k]["weights"]], chosen[k]["train_f1"]))
    print("  fold scores: {}".format([round(v, 6) for v in fold_scores]))
    print("  fold std   : {:.6f}".format(float(np.std(fold_scores))))
    print("  per-class  : {}".format([round(v, 6) for v in pcf1(y, honest)]))
    print("  accuracy   : {:.6f}".format(float(accuracy_score(y, honest.argmax(1)))))
    res = {"sources": None, "step_grid_points": len(grid),
           "opt_f1": opt[0], "opt_w": [float(v) for v in opt[1]],
           "honest_f1": hf, "honest_delta": hf - SUB2_REF,
           "chosen": chosen, "fold_scores": fold_scores,
           "fold_std": float(np.std(fold_scores)),
           "per_class": [round(v, 6) for v in pcf1(y, honest)],
           "acc": round(float(accuracy_score(y, honest.argmax(1))), 6),
           "confusion": confusion_matrix(y, honest.argmax(1), labels=CLASSES).tolist(),
           "pred_dist": np.bincount(honest.argmax(1), minlength=3).tolist()}
    return honest, res


def main():
    z = np.load("cache/v4/sources_oof.npz")
    folds, y, sub2 = z["fold_ids"], z["y"], z["sub2"]
    names = [k for k in z.files
             if k not in ("fold_ids", "y", "sub2", "text_base", "text_all3")]
    print("blend sources:", names)
    S = np.stack([z[n] for n in names])
    print("SUBMISSION 2 reference OOF: {:.6f}".format(mf1(y, sub2)))

    T = 10
    coarse_grid = [tuple(np.array(c) / T) for c in simplex(len(names), T)]
    oof_c, res_c = honest_search(S, y, folds, coarse_grid, "coarse step 0.10")
    res_c["sources"] = names

    ref_grid = grid_from(res_c["opt_w"], np.repeat(0.20, len(names)), 0.05)
    oof_r, res_r = honest_search(S, y, folds, ref_grid, "refined step 0.05")
    res_r["sources"] = names
    res_r["neighbourhood_of"] = res_c["opt_w"]

    np.save("cache/v4/blend_coarse_oof.npy", oof_c)
    np.save("cache/v4/blend_refined_oof.npy", oof_r)
    with open("reports/v4/blend_search.json", "w") as f:
        json.dump({"coarse": res_c, "refined": res_r}, f, indent=2)
    print("\n-> cache/v4/blend_{coarse,refined}_oof.npy, reports/v4/blend_search.json")


if __name__ == "__main__":
    main()

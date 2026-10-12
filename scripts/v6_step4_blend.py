"""v6 Step 4: honest cross-fitted 4-source blends - sid variant vs plain-ID variant."""
import os
import json
import time
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

N_SPLITS = 5
CLASSES = [0, 1, 2]
STEP = 0.05
WORK = "cache/v6"
os.makedirs("reports/v6", exist_ok=True)


def mf1(y, p):
    return float(f1_score(y, np.asarray(p).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def pcf1(y, p):
    return [round(float(v), 6) for v in
            f1_score(y, np.asarray(p).argmax(1), average=None, labels=CLASSES,
                     zero_division=0)]


def norm(P):
    return P / P.sum(axis=1, keepdims=True)


def simplex_vectors(n, t):
    """Lexicographic composition order, first max wins."""
    if n == 1:
        yield (t,)
        return
    for i in range(t + 1):
        for rest in simplex_vectors(n - 1, t - i):
            yield (i,) + rest


def honest_blend(S, y, folds, grid):
    out = np.zeros((len(y), 3))
    chosen, fs = {}, []
    for k in range(N_SPLITS):
        va, tr = folds == k, folds != k
        Sr, yr = S[:, tr], y[tr]
        best = (-1.0, None)
        for w in grid:
            sc = mf1(yr, norm(np.tensordot(w, Sr, 1)))
            if sc > best[0]:
                best = (sc, w)
        w = best[1]
        chosen[str(k)] = {"w": [float(v) for v in w], "train_f1": round(best[0], 6)}
        out[va] = norm(np.tensordot(w, S[:, va], 1))
        fs.append(mf1(y[va], out[va]))
    opt = (-1.0, None)
    for w in grid:
        sc = mf1(y, norm(np.tensordot(w, S, 1)))
        if sc > opt[0]:
            opt = (sc, w)
    return out, mf1(y, out), fs, chosen, opt


def main():
    d12 = np.load(WORK + "/step123.npz", allow_pickle=True)
    y, folds = d12["y"], d12["fold_ids"]
    text, plain_id = d12["oof_text"], d12["oof_plain"]
    sid = d12["oof_sid"]
    e5 = d12["e5_oof"]
    xlmr = d12["xlmr_oof"]

    T = int(round(1.0 / STEP))
    grid = [tuple(np.array(c, dtype=float) / T) for c in simplex_vectors(4, T)]
    print("simplex vectors: {} (step {})".format(len(grid), STEP))

    variants = {
        "sid_variant":   [text, sid, e5, xlmr],
        "plain_id_variant": [text, plain_id, e5, xlmr],
    }
    res = {}
    for nm, arrs in variants.items():
        S = np.stack(arrs)
        t0 = time.time()
        out, f1, fs, chosen, opt = honest_blend(S, y, folds, grid)
        med = np.median(np.array([chosen[str(k)]["w"] for k in range(N_SPLITS)]),
                        axis=0)
        med = med / med.sum()
        res[nm] = {
            "honest_oof": round(f1, 6), "fold_scores": [round(v, 6) for v in fs],
            "fold_mean": round(float(np.mean(fs)), 6),
            "fold_std": round(float(np.std(fs)), 6),
            "worst_fold": round(float(np.min(fs)), 6),
            "per_class": pcf1(y, out),
            "accuracy": round(float(accuracy_score(y, out.argmax(1))), 6),
            "confusion": confusion_matrix(y, out.argmax(1),
                                          labels=CLASSES).tolist(),
            "per_fold_weights": chosen,
            "median_weights": [float(v) for v in med],
            "same_oof_OPTIMISTIC": round(opt[0], 6),
            "same_oof_opt_weights": [float(v) for v in opt[1]],
            "min_fold": round(float(np.min(fs)), 6),
        }
        np.save(WORK + "/blend_{}_oof.npy".format(nm), out)
        print("\n=== {} ===".format(nm))
        print("  honest OOF      = {:.6f}".format(f1))
        print("  per-class F1    = {}".format(res[nm]["per_class"]))
        print("  fold scores     = {}".format(res[nm]["fold_scores"]))
        print("  fold std        = {:.6f} | worst {:.6f}".format(
            res[nm]["fold_std"], res[nm]["worst_fold"]))
        print("  per-fold w      = {}".format(
            {k: v["w"] for k, v in chosen.items()}))
        print("  median weights  = {}".format([round(float(v), 4) for v in med]))
        print("  OPTIMISTIC      = {:.6f}".format(opt[0]))
        print("  ({:.0f}s)".format(time.time() - t0))

    a = res["sid_variant"]["honest_oof"]
    b = res["plain_id_variant"]["honest_oof"]
    print("\n" + "=" * 70)
    print("GATES")
    print("  G2a honest OOF >= 0.678           : {:.6f} -> {}".format(
        a, "PASS" if a >= 0.678 else "FAIL"))
    print("  G2b sid >= plain-ID + 0.003        : {:.6f} vs {:.6f} (diff {:+.6f}) -> {}".format(
        a, b, a - b, "PASS" if a - b >= 0.003 else "FAIL"))
    print("  G3  every fold >= 0.675           : min {:.6f} -> {}".format(
        res["sid_variant"]["min_fold"],
        "PASS" if res["sid_variant"]["min_fold"] >= 0.675 else "FAIL"))
    with open("reports/v6/blend.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\n-> reports/v6/blend.json, cache/v6/blend_*_oof.npy")


if __name__ == "__main__":
    main()

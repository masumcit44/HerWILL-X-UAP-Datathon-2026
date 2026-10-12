"""v4 Phase 5: honest cross-fitted per-class logit offsets on a fixed blend.

bias(class0) = 0 by construction. b1, b2 searched on the OTHER four folds only.
Keep rule: honest gain >= +0.0010, >=4 folds not materially regressing, std sane.
"""
import os
import sys
import json
import itertools
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

N_SPLITS = 5
CLASSES = [0, 1, 2]
SUB2_REF = 0.657612
os.makedirs("reports/v4", exist_ok=True)
os.makedirs("predictions/v4_candidates", exist_ok=True)

B1 = np.round(np.arange(-0.30, 0.3001, 0.025), 4)
B2 = np.round(np.arange(-0.30, 0.3001, 0.025), 4)


def mf1(y, p):
    return float(f1_score(y, np.asarray(p).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def pcf1(y, p):
    return [float(v) for v in f1_score(y, np.asarray(p).argmax(1), average=None,
                                       labels=CLASSES, zero_division=0)]


def apply_bias(P, b1, b2):
    eps = 1e-12
    L = np.log(np.clip(P, eps, 1.0))
    L = L + np.array([0.0, b1, b2])[None, :]
    L = L - L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


def main():
    y = np.load("cache/v4/sources_oof.npz")["y"]
    folds = np.load("cache/v4/sources_oof.npz")["fold_ids"]
    P = np.load(sys.argv[1] if len(sys.argv) > 1
                else "cache/v4/blend_refined_oof.npy")
    print("base blend:", sys.argv[1] if len(sys.argv) > 1
          else "cache/v4/blend_refined_oof.npy")
    base_f1 = mf1(y, P)
    base_folds = [mf1(y[folds == k], P[folds == k]) for k in range(N_SPLITS)]
    print("base blend (Submission 2 weights) honest OOF: {:.6f}".format(base_f1))
    print("  per-class {}  fold std {:.6f}".format(
        [round(v, 6) for v in pcf1(y, P)], float(np.std(base_folds))))

    grid = [(float(b1), float(b2)) for b1 in B1 for b2 in B2]
    print("bias grid points: {}".format(len(grid)))

    honest = np.zeros_like(P)
    chosen, fold_scores = {}, []
    for k in range(N_SPLITS):
        va, tr = folds == k, folds != k
        best = (-1.0, None)
        for b1, b2 in grid:
            s = mf1(y[tr], apply_bias(P[tr], b1, b2))
            if s > best[0]:
                best = (s, (b1, b2))
        b1, b2 = best[1]
        chosen[str(k)] = {"b1": b1, "b2": b2, "train_f1": round(best[0], 6)}
        honest[va] = apply_bias(P[va], b1, b2)
        fold_scores.append(mf1(y[va], honest[va]))
        print("  fold {}: b1={:+.3f} b2={:+.3f} (train {:.6f}) -> held-out {:.6f}"
              .format(k, b1, b2, best[0], fold_scores[-1]))

    hf = mf1(y, honest)
    opt = max(grid, key=lambda bb: mf1(y, apply_bias(P, bb[0], bb[1])))
    print("\noptimistic same-OOF offsets: b1={:+.3f} b2={:+.3f} -> {:.6f}".format(
        opt[0], opt[1], mf1(y, apply_bias(P, opt[0], opt[1]))))
    print("HONEST cross-fitted offsets : {:.6f}  (delta vs base = {:+.6f})"
          .format(hf, hf - base_f1))
    print("  per-class  : {}  (base {})".format(
        [round(v, 6) for v in pcf1(y, honest)],
        [round(v, 6) for v in pcf1(y, P)]))
    print("  fold scores: {} (base {})".format(
        [round(v, 6) for v in fold_scores], [round(v, 6) for v in base_folds]))
    print("  fold std   : {:.6f} (base {:.6f})".format(
        float(np.std(fold_scores)), float(np.std(base_folds))))
    print("  accuracy   : {:.6f}".format(float(accuracy_score(y, honest.argmax(1)))))
    nch = int((honest.argmax(1) != P.argmax(1)).sum())
    print("  OOF predictions changed by offsets: {} ({:.4f}%)".format(
        nch, 100.0 * nch / len(y)))
    regress = sum(1 for k in range(N_SPLITS)
                  if fold_scores[k] < base_folds[k] - 0.0005)
    print("  folds regressing >0.0005: {} (must be <= 1)".format(regress))

    keep = (hf - base_f1) >= 0.0010 and regress <= 1
    print("\nKEEP OFFSETS:", keep, "(rule: gain>=+0.0010 and >=4 folds non-regressing)")
    print("DECISION BAND vs 0.657612:", end=" ")
    d = hf - SUB2_REF
    if d < 0:
        print("REJECT")
    elif d < 0.005388:
        print("SMALL_INCREMENT")
    elif d < 0.012388:
        print("STRONG_INCREMENT")
    elif d < 0.022388:
        print("MAJOR_CHALLENGER")
    else:
        print("NUMBER1_LEVEL_LOCAL_CANDIDATE")

    np.save("predictions/v4_candidates/candidate_logit_offset_oof_probs.npy", honest)
    with open("reports/v4/logit_offsets.json", "w") as f:
        json.dump({"base_honest": base_f1, "base_per_class": pcf1(y, P),
                   "base_folds": base_folds,
                   "optimistic_offsets": list(opt),
                   "optimistic_f1": mf1(y, apply_bias(P, opt[0], opt[1])),
                   "honest_f1": hf, "honest_delta_vs_base": hf - base_f1,
                   "honest_per_class": pcf1(y, honest),
                   "honest_folds": fold_scores,
                   "honest_fold_std": float(np.std(fold_scores)),
                   "per_fold_offsets": chosen,
                   "n_changed": nch, "folds_regressing": regress,
                   "keep": bool(keep)}, f, indent=2)
    print("-> reports/v4/logit_offsets.json")


if __name__ == "__main__":
    main()

"""PERFORMANCE TRACK - uses ID-locality; not for judge-facing track.

Honest cross-fitted blending for the leaderboard-performance track.

This is a SEPARATE entry point from ``text_blend_eval.py``. That script blocks
every id-family source from its T1/T2/T3 candidates, which is correct for the
judge-facing track. Here, id-family sources (plain ID prior, SID, ID-head
models) are explicitly ALLOWED, because the performance track is the one that
reproduces Submission 3.

Its id-block guard is deliberately NOT loosened.

Protocol
--------
* Fold set B (the recovered Colab folds) only. Any other fold file is refused.
  No fold set A (canonical local) OOF array may be mixed in.
* For each held-out fold k, blend weights are selected on the other four folds
  and applied to fold k. The weight for fold k never sees fold k.
* Candidate weights are all points of the 4-simplex at step 0.05 (1771 vectors),
  enumerated in lexicographic composition order; the first maximum wins, which
  makes selection deterministic.
* The same-OOF optimum is reported separately and always labelled OPTIMISTIC.

Usage
-----
    python scripts/perf_blend_eval.py
    python scripts/perf_blend_eval.py --src id:sid=<path> ...
    python scripts/perf_blend_eval.py --group-idhead

No CSV is ever written. Probability arrays are the only optional output.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

HEADER = "PERFORMANCE TRACK - uses ID-locality; not for judge-facing track"
FOLD_PATH = "final_sources/fold_ids.npy"
FOLD_INT64_SHA = "cc4fdd169110aeee6d3d5566095234461041b1a9290e2e916ce850c31d27a7ca"
FOLD_SIZES = [9543, 9546, 9571, 9581, 9576]
FOLD_FILE_SHA = "5ab5cfe96a72b04c8bd30adc45c1eeeee477707b0c382a19d179d8635295fbbd"
N_SPLITS = 5
CLASSES = [0, 1, 2]
STEP = 0.05
REFERENCE_OOF = 0.682311
REFERENCE_FOLDS = [0.682243, 0.685245, 0.680023, 0.680668, 0.683172]
REFERENCE_STD = 0.001859
REFERENCE_WORST = 0.680023
REFERENCE_PER_CLASS = [0.782202, 0.640594, 0.624137]
FAMILIES = ("lexical", "semantic", "id", "finetuned")
TRAIN_SHARE = [0.4968, 0.3798, 0.1233]

# Submission 3 sources (the four that produced 0.682311)
DEFAULT_SOURCES = [
    "lexical:text=cache/v6/step123.npz::oof_text::test_text",
    "id:sid=cache/v6/step123.npz::oof_sid::test_sid",
    "semantic:e5=final_sources/emb_e5-base_final.npz::oof::test",
    "id:xlmr_idhead=final_sources/xlmr_base_idhead_final.npz::oof::test",
]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def macro_f1(y, pr):
    return float(f1_score(y, np.asarray(pr).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def per_class(y, pr):
    return [round(float(v), 6) for v in
            f1_score(y, np.asarray(pr).argmax(1), average=None, labels=CLASSES,
                     zero_division=0)]


def norm(P):
    return P / P.sum(axis=1, keepdims=True)


def integrity(P, n, tag):
    probs = []
    if P.shape != (n, 3):
        probs.append("shape %s" % (P.shape,))
    if not np.isfinite(P).all():
        probs.append("non-finite")
    if P.min() < -1e-6 or P.max() > 1 + 1e-6:
        probs.append("out of [0,1]")
    rs = float(np.abs(P.sum(1) - 1).max())
    if rs > 1e-4:
        probs.append("rowsum %.2e" % rs)
    return (not probs), probs, rs


def load_folds(path):
    if os.path.abspath(path) != os.path.abspath(FOLD_PATH):
        raise SystemExit("refusing fold file %r: performance track is fold set B only (%s)"
                         % (path, FOLD_PATH))
    raw = open(path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != FOLD_FILE_SHA:
        raise SystemExit("fold FILE sha256 mismatch - refusing")
    f = np.load(path).astype(np.int64)
    got = hashlib.sha256(np.ascontiguousarray(f, np.int64).tobytes()).hexdigest()
    if got != FOLD_INT64_SHA:
        raise SystemExit("fold int64 sha256 mismatch (%s) - refusing" % got[:16])
    if np.bincount(f, minlength=N_SPLITS).tolist() != FOLD_SIZES:
        raise SystemExit("fold sizes mismatch - refusing")
    return f


def load_source(spec, folds, n_tr, n_te):
    """spec = family:name=path[:oofkey[:testkey]]  (defaults oof/test)"""
    fam, rest = spec.split(":", 1)
    if fam not in FAMILIES:
        raise SystemExit("unknown family %r (allowed: %s)" % (fam, ",".join(FAMILIES)))
    name, path = rest.split("=", 1)
    parts = path.split("::")
    path = parts[0]
    ok = [k for k in parts[1:]]
    z = np.load(path)
    oof = np.asarray(z[ok[0] if ok else "oof"], dtype=np.float64)
    test = np.asarray(z[ok[1] if len(ok) > 1 else "test"], dtype=np.float64)
    if "fold_ids" in z:
        if not np.array_equal(np.asarray(z["fold_ids"], np.int64), folds):
            raise SystemExit("%s: fold set mismatch (NOT fold set B) - refusing" % name)
    ok1, why1, rs1 = integrity(oof, n_tr, name)
    ok2, why2, rs2 = integrity(test, n_te, name)
    if not (ok1 and ok2):
        raise SystemExit("%s integrity failure: %s %s" % (name, why1, why2))
    return dict(name=name, family=fam, oof=oof, test=test, path=path,
                sha256=sha256_file(path), oof_rowsum_dev=rs1, test_rowsum_dev=rs2)


def simplex(n, t):
    if n == 1:
        yield (t,)
        return
    for i in range(t + 1):
        for r in simplex(n - 1, t - i):
            yield (i,) + r


def weight_grid(n_src, step=STEP):
    T = int(round(1.0 / step))
    return [np.array(c, float) / T for c in simplex(n_src, T)]


def honest_blend(S, y, folds):
    """Select weights on 4 folds, apply to the held-out 5th."""
    grid = weight_grid(S.shape[0])
    out = np.zeros((len(y), 3))
    chosen, fold_scores = {}, []
    for k in range(N_SPLITS):
        va, tr = folds == k, folds != k
        Sr, yr = S[:, tr], y[tr]
        best = (-1.0, None)
        for w in grid:                       # lexicographic; first max wins
            sc = macro_f1(yr, norm(np.tensordot(w, Sr, 1)))
            if sc > best[0]:
                best = (sc, w)
        w = best[1]
        chosen[str(k)] = [float(v) for v in w]
        out[va] = norm(np.tensordot(w, S[:, va], 1))
        fold_scores.append(macro_f1(y[va], out[va]))
    opt = (-1.0, None)
    for w in grid:
        sc = macro_f1(y, norm(np.tensordot(w, S, 1)))
        if sc > opt[0]:
            opt = (sc, [float(v) for v in w])
    return out, macro_f1(y, out), fold_scores, chosen, opt


def paired_delta(a, b):
    """a,b = per-fold scores. Returns mean delta, SE, t (paired, df=4)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    se = sd / np.sqrt(len(d)) if sd > 0 else float("inf")
    # Degenerate case (reference compared against itself): mean and SD are float
    # noise, so any t-ratio would be meaningless. Report 0 rather than a
    # spurious large value.
    if sd < 1e-12 or abs(m) < 1e-12:
        t, se = 0.0, 0.0
    else:
        t = m / se
    return m, se, t, [round(float(v), 6) for v in d]


def gates(pooled, fs, pc, std, worst, test_probs, y):
    g = {}
    g["G1_delta_ge_0_0015"] = (pooled - REFERENCE_OOF) >= 0.0015


    g["G2_4folds_not_worse"] = sum(1 for a, b in zip(fs, REFERENCE_FOLDS)
                                   if a >= b - 0.0005) >= 4
    g["G3_no_class_drop_gt_0_005"] = all((pc[i] - REFERENCE_PER_CLASS[i]) >= -0.005
                                         for i in range(3))
    g["G4_std_le_1_5x"] = std <= 1.5 * REFERENCE_STD
    g["G5_worst_fold"] = worst >= (REFERENCE_WORST - 0.001)
    ok, why, rs = integrity(test_probs, test_probs.shape[0], "test")
    g["G6_test_integrity"] = ok
    share = [float((test_probs.argmax(1) == c).mean()) for c in CLASSES]
    drift = [100 * abs(share[i] - float((y == CLASSES[i]).mean())) for i in range(3)]
    g["G7_test_share_within_6pp"] = all(v <= 6.0 for v in drift)
    return g, share, drift, rs



def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=HEADER)
    ap.add_argument("--folds", default=FOLD_PATH)
    ap.add_argument("--src", action="append", default=None,
                    help="family:name=path[:oofkey[:testkey]]; repeatable")
    ap.add_argument("--group-idhead", action="store_true",
                    help="mean of all *_idhead sources -> one 'idhead_ens' source")
    ap.add_argument("--out", default="reports/performance_v7_gpu")
    ap.add_argument("--label", default="run")
    ap.add_argument("--reproduce-check", action="store_true",
                    help="require pooled OOF within 1e-4 of 0.682311, else stop")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("=" * 72)
    print(HEADER)
    print("=" * 72)
    folds = load_folds(a.folds)
    import pandas as pd
    tr = pd.read_csv("train.csv", usecols=["y"])["y"].to_numpy(int)
    n_tr, n_te = len(tr), 11955
    print("fold set B locked: int64 %s | sizes %s" % (FOLD_INT64_SHA[:16], FOLD_SIZES))
    S = {}
    for spec in (a.src if a.src else DEFAULT_SOURCES):
        s = load_source(spec, folds, n_tr, n_te)
        S[s["name"]] = s
        print("  loaded %-9s %-16s oof%s test%s  %s"
              % (s["family"], s["name"], s["oof"].shape, s["test"].shape, s["path"]))
    if a.group_idhead:
        heads = [n for n in S if n.endswith("_idhead")]
        if len(heads) < 2:
            raise SystemExit("--group-idhead needs >= 2 *_idhead sources, got %s" % heads)
        keep = [n for n in S if n not in heads]
        eo, et = norm(np.mean([S[n]["oof"] for n in heads], 0)), norm(np.mean([S[n]["test"] for n in heads], 0))
        ok1, why1, _ = integrity(eo, n_tr, "idhead_ens")
        ok2, why2, _ = integrity(et, n_te, "idhead_ens")
        if not (ok1 and ok2):
            raise SystemExit("idhead_ens integrity failure: %s %s" % (why1, why2))
        S = {n: S[n] for n in keep}
        S["idhead_ens"] = dict(name="idhead_ens", family="id", oof=eo, test=et,
                               path="<mean of %s>" % heads, sha256="",
                               oof_rowsum_dev=0.0, test_rowsum_dev=0.0)
        print("  GROUPED %s -> 'idhead_ens' (weight count unchanged)" % heads)
    order = sorted(S)
    Sarr = np.stack([S[n]["oof"] for n in order])
    out, pooled, fs, chosen, opt = honest_blend(Sarr, tr, folds)
    pc = per_class(tr, out)
    acc = float(accuracy_score(tr, out.argmax(1)))
    std, worst = float(np.std(fs)), float(min(fs))
    mean_d, se, tstat, deltas = paired_delta(fs, REFERENCE_FOLDS)
    med = np.median(np.array([chosen[str(k)] for k in range(N_SPLITS)]), axis=0)
    med = med / med.sum()
    test_probs = norm(sum(float(w) * S[n]["test"] for w, n in zip(med, order)))
    g, share, drift, _ = gates(pooled, fs, pc, std, worst, test_probs, tr)
    print("\n--- result (honest cross-fitted, fold set B) ---")
    print("  label            : %s" % a.label)
    print("  sources (%d)      : %s" % (len(order), order))
    print("  pooled OOF       : %.6f   (Submission 3 ref %.6f)" % (pooled, REFERENCE_OOF))
    print("  delta vs ref     : %+.6f" % (pooled - REFERENCE_OOF))
    print("  per-class F1     : %s" % pc)
    print("  accuracy         : %.6f" % acc)
    print("  fold scores      : %s" % [round(v, 6) for v in fs])
    print("  fold std / worst : %.6f / %.6f" % (std, worst))
    print("  paired per-fold  : mean %+.6f  SE %.6f  t %.4f (df=4)" % (mean_d, se, tstat))
    print("                      per-fold %s" % deltas)
    print("  OPTIMISTIC same-OOF (not a result): %.6f  w=%s"
          % (opt[0], [round(v, 2) for v in opt[1]]))
    print("  median weights   : %s" % {n: round(float(v), 4) for n, v in zip(order, med)})
    print("  weights per fold : %s" % {k: [round(v, 2) for v in chosen[k]] for k in chosen})
    print("  test share/drift : %s  %s" % ([round(v, 4) for v in share],
                                           [round(v, 2) for v in drift]))
    print("  gates            : %s" % {k: ("PASS" if v else "FAIL") for k, v in g.items()})
    print("  ALL GATES PASS   : %s" % all(g.values()))

    rep = dict(header=HEADER, label=a.label, fold_int64_sha256=FOLD_INT64_SHA,
               fold_sizes=FOLD_SIZES, order=order, grouped_idhead=bool(a.group_idhead),
               sources={n: dict(family=S[n]["family"], path=S[n]["path"], sha256=S[n]["sha256"],
                                oof_rowsum_dev=S[n]["oof_rowsum_dev"],
                                test_rowsum_dev=S[n]["test_rowsum_dev"]) for n in order},
               pooled_oof=round(pooled, 6),
               delta_vs_submission3=round(pooled - REFERENCE_OOF, 6),
               per_class_f1=pc, accuracy=round(acc, 6),
               fold_scores=[round(v, 6) for v in fs], fold_std=round(std, 6),
               worst_fold=round(worst, 6),
               paired_delta=dict(mean=round(mean_d, 6), se=round(se, 6),
                                 t=round(tstat, 4), per_fold=deltas),
               same_oof_OPTIMISTIC=round(opt[0], 6), same_oof_weights=opt[1],
               fold_weights=chosen,
               median_weights={n: float(v) for n, v in zip(order, med)},
               reference=dict(pooled_oof=REFERENCE_OOF, fold_scores=REFERENCE_FOLDS,
                              fold_std=REFERENCE_STD, worst_fold=REFERENCE_WORST,
                              per_class=REFERENCE_PER_CLASS),
               test_class_share=[round(v, 4) for v in share],
               test_share_drift_pp=[round(v, 3) for v in drift],
               gates=g, all_gates_pass=bool(all(g.values())),
               submission_generated=False, elapsed_seconds=round(time.time() - t0, 1))
    with open(os.path.join(a.out, "perf_blend_%s.json" % a.label), "w") as f:
        json.dump(rep, f, indent=2)
    print("\n  report: %s/perf_blend_%s.json  (no CSV, no P-file)" % (a.out, a.label))
    print("  elapsed %.1fs" % (time.time() - t0))
    if a.reproduce_check and abs(pooled - REFERENCE_OOF) > 1e-4:
        print("\nSTOP: reference not reproduced (%.6f vs %.6f)" % (pooled, REFERENCE_OOF))
        raise SystemExit(2)
    return rep


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

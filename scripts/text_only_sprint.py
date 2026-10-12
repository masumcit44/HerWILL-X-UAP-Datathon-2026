"""15-minute TEXT-ONLY improvement sprint. Wall-clock guarded, CPU only.

Text-derived sources only. No ID, SID, row-order, XLM-R ID-head, transformer
training, embedding, or leaderboard feedback. No submission CSV/notebook.
"""
import os
import json
import time
import hashlib
import platform
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

T_START = time.time()
BUDGET, STOP_NEW = 15 * 60.0, 13 * 60.0
OUT_P, OUT_R = "predictions/text_only_fast", "reports/text_only_fast"
for d in (OUT_P, OUT_R):
    os.makedirs(d, exist_ok=True)


def el():
    return time.time() - T_START


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_int64(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()


CLASSES = [0, 1, 2]
PROTECTED = [
    "train.csv", "test.csv", "sample_submission.csv",
    "uap_submission/reproducible/submission/submission_1.csv",
    "uap_submission/submission/submission_2.csv",
    "uap_submission/submission/submission_3.csv",
    "uap_submission/reproducible/submission_reproducible_file_1.ipynb",
    "uap_submission/reproducible/submission_reproducible_file_2.ipynb",
    "uap_submission/reproducible/submission_reproducible_file_3.ipynb",
    "reports/final_claude_handoff.md",
]
PROT_BEFORE = {p: (sha256_file(p) if os.path.exists(p) else "MISSING")
               for p in PROTECTED}


def mf1(y, pr):
    return float(f1_score(y, np.asarray(pr).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def pcf1(y, pr):
    return [round(float(v), 6) for v in
            f1_score(y, np.asarray(pr).argmax(1), average=None, labels=CLASSES,
                     zero_division=0)]


def norm(P):
    return P / P.sum(axis=1, keepdims=True)


def integrity(P, nrows):
    probs = []
    if P.shape != (nrows, 3):
        probs.append("shape %s" % (P.shape,))
    if not np.isfinite(P).all():
        probs.append("non-finite")
    if P.min() < -1e-6 or P.max() > 1 + 1e-6:
        probs.append("out of [0,1]")
    rs = float(np.abs(P.sum(1) - 1).max())
    if rs > 1e-4:
        probs.append("rowsum %.2e" % rs)
    return (len(probs) == 0), probs, rs


CAND = {
    "text_base": ("cache/v4/sources_oof.npz", "text_base", None),
    "text_all3": ("cache/v4/sources_oof.npz", "text_all3", None),
    "text": ("cache/v4/sources_oof.npz", "text",
             "predictions/v4_candidates/CANDIDATE_NOT_FOR_SUBMISSION_source_text_test_probs.npy"),
    "e5_small": ("cache/v4/sources_oof.npz", "e5_small",
                 "predictions/v4_candidates/CANDIDATE_NOT_FOR_SUBMISSION_source_e5_small_test_probs.npy"),
    "e5_base": ("cache/v4/sources_oof.npz", "e5_base",
                "predictions/v4_candidates/CANDIDATE_NOT_FOR_SUBMISSION_source_e5_base_test_probs.npy"),
}
FOLD_PATH = "cache/v2/fold_ids.npy"
folds = np.load(FOLD_PATH).astype(np.int64)
_d = np.load("cache/v4/sources_oof.npz")
y = _d["y"].astype(int)
assert np.array_equal(folds, _d["fold_ids"].astype(np.int64)), "fold mismatch"
N_FOLDS = int(folds.max()) + 1
FOLD_SIZES = np.bincount(folds, minlength=N_FOLDS).tolist()
FOLD_FILE_SHA, FOLD_INT_SHA = sha256_file(FOLD_PATH), sha256_int64(folds)


# ---------------------------------------------------------------- Phase 1
print("PHASE 1 - ARTIFACT INVENTORY  [t=%.1fs]" % el(), flush=True)
print("fold set A sizes", FOLD_SIZES, flush=True)
print("fold FILE sha256 ", FOLD_FILE_SHA, flush=True)
print("fold INT64 sha256", FOLD_INT_SHA, flush=True)

audit, usable = {}, {}
for name, (op, key, tp) in CAND.items():
    d = np.load(op)
    P = d[key].astype(np.float64)
    ok, why, rs = integrity(P, 47817)
    a = {"oof_path": op, "oof_key": key, "test_path": tp,
         "oof_sha256": sha256_file(op), "oof_rowsum_dev": rs,
         "oof_ok": ok, "oof_problems": why}
    if tp and os.path.exists(tp):
        T = np.load(tp).astype(np.float64)
        tok, twhy, trs = integrity(T, 11955)
        a.update({"test_sha256": sha256_file(tp), "test_rowsum_dev": trs,
                  "test_ok": tok, "test_problems": twhy})
    else:
        T, tok = None, False
        a["test_problems"] = ["no cached test array"]
    a["status"] = "FOUND_AND_ALIGNED" if (ok and tok) else ("OOF_ONLY" if ok else "REJECTED")
    audit[name] = a
    if ok:
        usable[name] = (P, T)
    print("  %-10s %s%s" % (name, a["status"], "" if tok else " (oof only)"), flush=True)

other = {}
try:
    d6 = np.load("cache/v6/step123.npz")
    for nm, k in (("text", "oof_text"), ("e5_base", "e5_oof")):
        P = d6[k].astype(np.float64)
        ok, _, _ = integrity(P, 47817)
        other[nm] = {"fold_set": "B (Colab)", "integrity_ok": ok}
    other["xlmr_idhead"] = {"fold_set": "B (Colab)", "excluded": "ID-head forbidden"}
    other["plain_xlmr_base"] = {"fold_set": "-", "status": "MISSING",
                                "note": "xlmr_base_final.npz lost; 0.544479 historical only"}
except Exception as e:
    other["error"] = str(e)[:80]
print("  fold-set-B text sources (NOT mixed):", sorted(other), flush=True)
print("PHASE 1 done [t=%.1fs]" % el(), flush=True)

# ---------------------------------------------------------------- Phase 2
print("\nPHASE 2 - VERIFY TEXT SOURCES  [t=%.1fs]" % el(), flush=True)
NAMES = [n for n in CAND if n in usable]
standalone = {}
for n in NAMES:
    P, T = usable[n]
    fs = [mf1(y[folds == k], P[folds == k]) for k in range(N_FOLDS)]
    standalone[n] = {
        "pooled_macro_f1": round(mf1(y, P), 6), "per_class_f1": pcf1(y, P),
        "accuracy": round(float(accuracy_score(y, P.argmax(1))), 6),
        "fold_scores": [round(v, 6) for v in fs],
        "fold_std": round(float(np.std(fs)), 6),
        "pred_dist": np.bincount(P.argmax(1), minlength=3).tolist(),
        "test_dist": np.bincount(T.argmax(1), minlength=3).tolist() if T is not None else None,
    }
    print("  %-10s pooled %.6f acc %.6f std %.6f folds %s"
          % (n, standalone[n]["pooled_macro_f1"], standalone[n]["accuracy"],
             standalone[n]["fold_std"], standalone[n]["fold_scores"]), flush=True)
for i, a in enumerate(NAMES):
    for b in NAMES[i + 1:]:
        dis = float((usable[a][0].argmax(1) != usable[b][0].argmax(1)).mean())
        standalone[a].setdefault("disagreement", {})[b] = round(dis, 6)
        standalone[b].setdefault("disagreement", {})[a] = round(dis, 6)
        print("  disagree %-10s vs %-10s %.4f" % (a, b, dis), flush=True)
print("PHASE 2 done [t=%.1fs]" % el(), flush=True)


# ---------------------------------------------------------------- Phase 3
print("\nPHASE 3 - HONEST CROSS-FITTED BLEND  [t=%.1fs]" % el(), flush=True)
# priority order: plain XLM-R (missing) > E5 Base > E5 Small > lexical
PRIORITY = ["plain_xlmr_base", "e5_base", "e5_small", "text", "text_all3", "text_base"]
BLEND = [n for n in PRIORITY if n in usable][:4]
# The saved test artifact needs sources that have cached test arrays, so a
# secondary 3-source "test-capable" blend is computed for the test side only.
BLEND_TEST = [n for n in PRIORITY if n in usable and usable[n][1] is not None][:4]
print("  blend set (%d): %s" % (len(BLEND), BLEND), flush=True)
print("  test-capable set (%d): %s" % (len(BLEND_TEST), BLEND_TEST), flush=True)
STEP = 0.10
T_GRID = int(round(1.0 / STEP))


def simplex(n, t):
    if n == 1:
        yield (t,)
        return
    for i in range(t + 1):
        for r in simplex(n - 1, t - i):
            yield (i,) + r


GRID = [tuple(np.array(c, float) / T_GRID) for c in simplex(len(BLEND), T_GRID)]
print("  simplex candidates (step %.2f): %d" % (STEP, len(GRID)), flush=True)
S = np.stack([usable[n][0] for n in BLEND])


def honest(S, off=None):
    """Cross-fitted: weights chosen on the other four folds, applied to the held-out one."""
    out = np.zeros((len(y), 3))
    chosen, fs = {}, []
    for k in range(N_FOLDS):
        va, tr = folds == k, folds != k
        Sr, yr = S[:, tr], y[tr]
        best = (-1.0, None)
        for w in GRID:
            P = norm(np.tensordot(w, Sr, 1))
            if off is not None:
                P = apply_off(P, off)
            sc = mf1(yr, P)
            if sc > best[0]:          # first max wins -> deterministic lexicographic
                best = (sc, w)
        chosen[k] = best[1]
        P = norm(np.tensordot(best[1], S[:, va], 1))
        if off is not None:
            P = apply_off(P, off)
        out[va] = P
        fs.append(mf1(y[va], out[va]))
    return out, mf1(y, out), fs, chosen


out, honest_f1, fold_scores, chosen = honest(S)
opt = max(((mf1(y, norm(np.tensordot(w, S, 1))), w) for w in GRID), key=lambda t: t[0])
fold_std = float(np.std(fold_scores))
best_single = max(standalone[n]["pooled_macro_f1"] for n in BLEND)
best_single_name = max(BLEND, key=lambda n: standalone[n]["pooled_macro_f1"])
print("  BEST HONEST TEXT-ONLY OOF = %.6f" % honest_f1, flush=True)
print("  per-class", pcf1(y, out), "acc %.6f" % accuracy_score(y, out.argmax(1)), flush=True)
print("  folds", [round(v, 6) for v in fold_scores], "std %.6f" % fold_std, flush=True)
print("  OPTIMISTIC_DIAGNOSTIC_ONLY = %.6f  w=%s" % (opt[0], [round(v, 2) for v in opt[1]]), flush=True)
print("  delta from best single (%s %.6f) = %+.6f"
      % (best_single_name, best_single, honest_f1 - best_single), flush=True)
med = np.median(np.array([chosen[k] for k in range(N_FOLDS)]), axis=0)
med = med / med.sum()
print("  median final weights:", dict(zip(BLEND, [round(float(v), 4) for v in med])), flush=True)
print("PHASE 3 done [t=%.1fs]" % el(), flush=True)

# ---------------------------------------------------------------- Phase 4
print("\nPHASE 4 - CLASS-LOGIT OFFSET TEST  [t=%.1fs]" % el(), flush=True)
OFFSETS_TESTED, offsets_adopted = [], None
off_res = None
if BUDGET - el() > 120:
    GRID_OFF = [(a, b) for a in (-0.10, 0.0, 0.10) for b in (-0.10, 0.0, 0.10)]

    def apply_off(P, off):
        if off == (0.0, 0.0):
            return P
        z = np.log(np.clip(P, 1e-12, None))
        z[:, 1] += off[0]
        z[:, 2] += off[1]
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    base_pc = pcf1(y, out)
    for off in GRID_OFF:
        o2, f2, fs2, _ = honest(S, off)
        pc2 = pcf1(y, o2)
        nbet = sum(1 for a, b in zip(fs2, fold_scores) if a >= b - 1e-12)
        worst_drop = max(base_pc[i] - pc2[i] for i in range(3))
        gain = f2 - honest_f1
        rec = {"offsets": list(off), "honest_oof": round(f2, 6), "gain": round(gain, 6),
               "per_class_f1": pc2, "folds": [round(v, 6) for v in fs2],
               "folds_not_regressed": nbet, "worst_class_drop": round(worst_drop, 6),
               "fold_std": round(float(np.std(fs2)), 6)}
        OFFSETS_TESTED.append(rec)
        keep = (gain >= 0.001 and nbet >= 4 and worst_drop <= 0.005
                and float(np.std(fs2)) <= fold_std * 1.25)
        rec["adopt"] = bool(keep)
        if keep:
            off_res = (f2, off, o2, fs2, pc2)
        print("   off %+.2f,%+.2f honest %.6f gain %+.6f nbet %d worstdrop %.6f -> %s"
              % (off[0], off[1], f2, gain, nbet, worst_drop, "KEEP" if keep else "drop"), flush=True)
    if off_res:
        offsets_adopted = list(off_res[1])
        print("  ADOPTED offsets", off_res[1], flush=True)
    else:
        print("  no offsets met the adoption criteria", flush=True)
else:
    print("  skipped - under 2 minutes remaining", flush=True)
print("PHASE 4 done [t=%.1fs]" % el(), flush=True)

# ---------------------------------------------------------------- Phase 5
print("\nPHASE 5 - SAVE RESEARCH ARTIFACTS  [t=%.1fs]" % el(), flush=True)
final_oof = off_res[2] if off_res else out
final_honest = off_res[0] if off_res else honest_f1
final_folds = off_res[3] if off_res else fold_scores
final_pc = off_res[4] if off_res else pcf1(y, out)
has_test = all(usable[n][1] is not None for n in BLEND)
test_probs = None
test_honest = None
if BLEND_TEST:
    St = np.stack([usable[n][0] for n in BLEND_TEST])
    globals()["GRID"] = [tuple(np.array(c, float) / T_GRID) for c in simplex(len(BLEND_TEST), T_GRID)]
    out_t, honest_t, folds_t, chosen_t = honest(St)
    test_honest = honest_t
    med_t = np.median(np.array([chosen_t[k] for k in range(N_FOLDS)]), axis=0)
    med_t = med_t / med_t.sum()
    TM = {n: usable[n][1] for n in BLEND_TEST}
    test_probs = norm(sum(float(w) * TM[n] for w, n in zip(med_t, BLEND_TEST)))
    test_dist = np.bincount(test_probs.argmax(1), minlength=3).tolist()
    test_share = [round(v / len(test_probs), 4) for v in test_dist]
    np.save(OUT_P + "/CANDIDATE_NOT_FOR_SUBMISSION_text_fast_test_probs.npy",
            test_probs.astype(np.float32))
    print("  test-capable blend %s -> honest %.6f" % (BLEND_TEST, test_honest), flush=True)
    print("  test-capable median weights: %s"
          % {k: round(float(v), 4) for k, v in zip(BLEND_TEST, med_t)}, flush=True)
else:
    test_dist, test_share = None, None
np.save(OUT_P + "/candidate_text_fast_oof_probs.npy", final_oof.astype(np.float32))
oof_sha = sha256_file(OUT_P + "/candidate_text_fast_oof_probs.npy")
test_sha = (sha256_file(OUT_P + "/CANDIDATE_NOT_FOR_SUBMISSION_text_fast_test_probs.npy")
            if test_probs is not None else None)
print("  saved OOF", oof_sha[:16], flush=True)
print("  saved TEST", (test_sha or "-")[:16], flush=True)

time_limit = el() > BUDGET
prot_after = {p: (sha256_file(p) if os.path.exists(p) else "MISSING") for p in PROTECTED}

metrics = {
    "experiment": "text_only_fast_sprint", "text_only": True,
    "uses_id_or_sid": False, "uses_xlmr_idhead": False,
    "fold_set": "A (local canonical) cache/v2/fold_ids.npy",
    "fold_file_sha256": FOLD_FILE_SHA, "fold_int64_sha256": FOLD_INT_SHA,
    "fold_sizes": FOLD_SIZES, "n_folds": N_FOLDS,
    "blend_sources": BLEND, "simplex_step": STEP, "n_simplex_candidates": len(GRID),
    "source_standalone": standalone, "source_audit": audit,
    "other_fold_set_text_sources": other,
    "honest_oof_macro_f1": round(final_honest, 6), "per_class_f1": final_pc,
    "accuracy": round(float(accuracy_score(y, final_oof.argmax(1))), 6),
    "confusion": confusion_matrix(y, final_oof.argmax(1), labels=CLASSES).tolist(),
    "fold_scores": [round(v, 6) for v in final_folds],
    "fold_std": round(float(np.std(final_folds)), 6),
    "per_fold_weights": {str(k): [float(v) for v in chosen[k]] for k in range(N_FOLDS)},
    "final_median_weights": dict(zip(BLEND, [float(v) for v in med])),
    "same_oof_OPTIMISTIC_DIAGNOSTIC_ONLY": round(opt[0], 6),
    "offsets_tested": OFFSETS_TESTED, "offsets_adopted": offsets_adopted,
    "best_single_source": best_single_name, "best_single_macro_f1": best_single,
    "delta_from_best_single": round(final_honest - best_single, 6),
    "test_pred_dist": test_dist, "test_class_share": test_share,
    "train_class_share": [round(float((y == c).mean()), 4) for c in CLASSES],
    "artifacts": {
        "oof": OUT_P + "/candidate_text_fast_oof_probs.npy", "oof_sha256": oof_sha,
        "test": (OUT_P + "/CANDIDATE_NOT_FOR_SUBMISSION_text_fast_test_probs.npy"
                 if test_probs is not None else None), "test_sha256": test_sha,
        "test_capable_blend": BLEND_TEST, "test_capable_honest_oof":
            (round(test_honest, 6) if test_honest is not None else None)},
    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                 "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__},
    "elapsed_seconds": round(el(), 1), "stopped_due_to_time_limit": bool(time_limit),
    "submission_generated": False, "protected_files_unchanged": bool(prot_ok),
}
with open(OUT_R + "/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open(OUT_R + "/source_audit.json", "w") as f:
    json.dump({"audit": audit, "other_fold_set": other,
               "fold_file_sha256": FOLD_FILE_SHA, "fold_int64_sha256": FOLD_INT_SHA,
               "fold_sizes": FOLD_SIZES}, f, indent=2)
STATUS = ("EXCELLENT_TEXT_ONLY_CANDIDATE" if final_honest >= 0.680 else
          "MAJOR_TEXT_CANDIDATE" if final_honest >= 0.670 else
          "STRONG_TEXT_CANDIDATE" if final_honest >= 0.650 else
          "CREDIBLE_TEXT_CANDIDATE" if final_honest >= 0.630 else
          "BASELINE_TEXT_CANDIDATE" if final_honest >= 0.600 else "WEAK")
md = ["# Text-only fast sprint - summary", "",
      "Text-derived sources ONLY. No ID, SID, row order, XLM-R ID-head.", "",
      "## Headline", "",
      "- Honest cross-fitted OOF macro F1: **%.6f**" % final_honest,
      "- Per-class F1: %s" % final_pc,
      "- Accuracy: %.6f" % metrics["accuracy"],
      "- Fold scores: %s (std %.6f)" % (metrics["fold_scores"], metrics["fold_std"]),
      "- Same-OOF OPTIMISTIC_DIAGNOSTIC_ONLY: %.6f (not a performance claim)" % opt[0],
      "- Best single text source: %s %.6f (delta %+.6f)"
      % (best_single_name, best_single, final_honest - best_single),
      "- Final status: **%s**" % STATUS, "",
      "## Blend", "",
      "- Sources: %s" % ", ".join(BLEND),
      "- Simplex step %.2f, %d candidates, first max in lexicographic order" % (STEP, len(GRID)),
      "- Median final weights: %s" % {k: round(v, 4) for k, v in zip(BLEND, med)},
      "- Class-logit offsets adopted: %s" % offsets_adopted, "",
      "## Fold set", "",
      "- Set A (local canonical), sizes %s" % FOLD_SIZES,
      "- file sha256 `%s`" % FOLD_FILE_SHA,
      "- int64-bytes sha256 `%s`" % FOLD_INT_SHA,
      "- Fold set B (Colab) sources were NOT mixed in.", "",
      "## Methodological note", "",
      "This score is NOT directly comparable to Submission 3 (0.682311), which",
      "places 0.4211 of its weight on SID plus an ID-head transformer. This",
      "candidate uses comment text only, by design, to satisfy the organizer",
      "requirement for meaningful use of the text.", "",
      "## Provenance", "",
      "- elapsed %.1fs, stopped_due_to_time_limit=%s" % (el(), bool(time_limit)),
      "- submission_generated: false",
      "- protected files unchanged: %s" % prot_ok, ""]
with open(OUT_R + "/summary.md", "w") as f:
    f.write("\n".join(md))
print("PHASE 5 done [t=%.1fs]" % el(), flush=True)

print("\n" + "=" * 70, flush=True)
print("ELAPSED TIME: %.1fs" % el(), flush=True)
print("FOLD SET: A (local canonical) cache/v2/fold_ids.npy", flush=True)
print("FOLD SHA256: file %s | int64 %s" % (FOLD_FILE_SHA, FOLD_INT_SHA), flush=True)
print("USABLE TEXT SOURCES: %s" % ", ".join(BLEND), flush=True)
print("REJECTED OR MISSING SOURCES: plain_xlmr_base MISSING (artifact lost, 0.544479 "
      "historical only); xlmr_idhead EXCLUDED (ID-head, forbidden); fold-set-B "
      "(Colab) text sources not mixed", flush=True)
print("SOURCE STANDALONE SCORES:", flush=True)
for n in BLEND:
    print("   %-10s %.6f" % (n, standalone[n]["pooled_macro_f1"]), flush=True)
print("BEST HONEST TEXT-ONLY OOF: %.6f" % final_honest, flush=True)
print("PER-CLASS F1: %s" % final_pc, flush=True)
print("ACCURACY: %.6f" % metrics["accuracy"], flush=True)
print("FOLD SCORES: %s" % metrics["fold_scores"], flush=True)
print("FOLD STD: %.6f" % metrics["fold_std"], flush=True)
print("FOLD WEIGHTS: %s"
      % {k: [round(float(v), 2) for v in chosen[k]] for k in range(N_FOLDS)}, flush=True)
print("OPTIMISTIC DIAGNOSTIC SCORE: %.6f" % opt[0], flush=True)
print("OFFSETS TESTED: %d combos" % len(OFFSETS_TESTED), flush=True)
print("OFFSETS ADOPTED: %s" % (offsets_adopted if offsets_adopted else "NONE"), flush=True)
print("DELTA FROM BEST SINGLE TEXT SOURCE: %+.6f (vs %s %.6f)"
      % (final_honest - best_single, best_single_name, best_single), flush=True)
print("TEST CLASS DISTRIBUTION: %s" % (test_share if test_share else "no test probs"), flush=True)
print("OOF ARTIFACT: %s/candidate_text_fast_oof_probs.npy sha256=%s" % (OUT_P, oof_sha), flush=True)
print("TEST PROBABILITY ARTIFACT: %s/CANDIDATE_NOT_FOR_SUBMISSION_text_fast_test_probs.npy "
      "sha256=%s" % (OUT_P, test_sha), flush=True)
print("REPORT FILES: %s/{metrics.json,source_audit.json,summary.md}" % OUT_R, flush=True)
print("PROTECTED FILES: %s" % ("UNCHANGED" if prot_ok else "CHANGED - INVESTIGATE"), flush=True)
print("TIME LIMIT REACHED: %s" % ("YES" if time_limit else "NO"), flush=True)
print("FINAL STATUS: %s" % STATUS, flush=True)
print("SUBMISSION GENERATED: NO", flush=True)

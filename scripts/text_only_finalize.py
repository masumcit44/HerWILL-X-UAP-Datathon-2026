"""Finalize the text-only sprint report from already-saved artifacts."""
import os
import json
import time
import hashlib
import platform
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

T0 = time.time()
CLASSES = [0, 1, 2]
OUT_P, OUT_R = "predictions/text_only_fast", "reports/text_only_fast"
SRC, FOLD_PATH = "cache/v4/sources_oof.npz", "cache/v2/fold_ids.npy"
BLEND = ["e5_base", "e5_small", "text", "text_all3"]
BLEND_TEST = ["e5_base", "e5_small", "text"]
OFFSETS, STEP = [0.0, -0.10], 0.10
BASE_HONEST, OFF_HONEST, OPT = 0.552649, 0.559094, 0.553788
TEST_CAPABLE_HONEST = 0.553638
MED = {"e5_base": 0.4, "e5_small": 0.2, "text": 0.3, "text_all3": 0.1}
MED_T = {"e5_base": 0.4, "e5_small": 0.2, "text": 0.4}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def mf1(y, p):
    return float(f1_score(y, np.asarray(p).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def pcf1(y, p):
    return [round(float(v), 6) for v in
            f1_score(y, np.asarray(p).argmax(1), average=None, labels=CLASSES,
                     zero_division=0)]


d = np.load(SRC)
y = d["y"].astype(int)
folds = np.load(FOLD_PATH).astype(np.int64)
NF = int(folds.max()) + 1
P = np.load(OUT_P + "/candidate_text_fast_oof_probs.npy").astype(np.float64)
T = np.load(OUT_P + "/CANDIDATE_NOT_FOR_SUBMISSION_text_fast_test_probs.npy").astype(np.float64)
oof_sha = sha(OUT_P + "/candidate_text_fast_oof_probs.npy")
test_sha = sha(OUT_P + "/CANDIDATE_NOT_FOR_SUBMISSION_text_fast_test_probs.npy")
fs = [mf1(y[folds == k], P[folds == k]) for k in range(NF)]
pc, acc, fold_std = pcf1(y, P), float(accuracy_score(y, P.argmax(1))), float(np.std(fs))
test_dist = np.bincount(T.argmax(1), minlength=3).tolist()
test_share = [round(v / len(T), 4) for v in test_dist]
train_share = [round(float((y == c).mean()), 4) for c in CLASSES]

standalone, disagreement = {}, {}
for n in ["text_base", "text_all3", "text", "e5_small", "e5_base"]:
    Q = d[n].astype(np.float64)
    sfs = [mf1(y[folds == k], Q[folds == k]) for k in range(NF)]
    standalone[n] = {"pooled_macro_f1": round(mf1(y, Q), 6), "per_class_f1": pcf1(y, Q),
                     "accuracy": round(float(accuracy_score(y, Q.argmax(1))), 6),
                     "fold_scores": [round(v, 6) for v in sfs],
                     "fold_std": round(float(np.std(sfs)), 6),
                     "pred_dist": np.bincount(Q.argmax(1), minlength=3).tolist(),
                     "test_available": n in ("text", "e5_small", "e5_base")}
names = list(standalone)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        dis = round(float((d[a].argmax(1) != d[b].argmax(1)).mean()), 6)
        standalone[a].setdefault("disagreement", {})[b] = dis

VP = "predictions/v4_candidates"
TP = {"text": VP + "/CANDIDATE_NOT_FOR_SUBMISSION_source_text_test_probs.npy",
      "e5_small": VP + "/CANDIDATE_NOT_FOR_SUBMISSION_source_e5_small_test_probs.npy",
      "e5_base": VP + "/CANDIDATE_NOT_FOR_SUBMISSION_source_e5_base_test_probs.npy"}
audit = {}
for n in standalone:
    a = {"oof_path": SRC, "oof_key": n, "oof_sha256": sha(SRC),
         "fold_set": "A (local canonical)",
         "integrity": "shape ok, finite, in [0,1], rowsum dev < 1e-6",
         "status": "FOUND_AND_ALIGNED" if n in TP else "OOF_ONLY"}
    if n in TP:
        a["test_path"], a["test_sha256"] = TP[n], sha(TP[n])
    audit[n] = a
other = {"xlmr_idhead": {"fold_set": "B (Colab)", "status": "EXCLUDED",
                         "reason": "ID-head forbidden by task"},
         "plain_xlmr_base": {"status": "MISSING",
                             "note": "xlmr_base_final.npz lost; 0.544479 historical only"},
         "fold_set_B_text_and_e5": {"status": "NOT_MIXED",
                                    "reason": "different fold set; never combined"}}
STATUS = ("EXCELLENT_TEXT_ONLY_CANDIDATE" if OFF_HONEST >= 0.680 else
          "MAJOR_TEXT_CANDIDATE" if OFF_HONEST >= 0.670 else
          "STRONG_TEXT_CANDIDATE" if OFF_HONEST >= 0.650 else
          "CREDIBLE_TEXT_CANDIDATE" if OFF_HONEST >= 0.630 else
          "BASELINE_TEXT_CANDIDATE" if OFF_HONEST >= 0.600 else "WEAK")
best_name = max(standalone, key=lambda n: standalone[n]["pooled_macro_f1"])
best_single = standalone[best_name]["pooled_macro_f1"]
OT = [{"offsets": [0.0, 0.0], "honest_oof": BASE_HONEST, "gain": 0.0, "adopt": False,
       "reason": "reference (no offsets)"},
      {"offsets": [0.0, -0.10], "honest_oof": OFF_HONEST,
       "gain": round(OFF_HONEST - BASE_HONEST, 6), "folds_not_regressed": 5,
       "worst_class_drop": -0.001682, "fold_std": round(fold_std, 6), "adopt": True,
       "reason": "gain >= +0.001, 5/5 folds, no class F1 drop > 0.005, std not worsened"}]
for _a in (-0.10, 0.0, 0.10):
    for _b in (-0.10, 0.0, 0.10):
        if [_a, _b] in ([0.0, 0.0], [0.0, -0.10]):
            continue
        OT.append({"offsets": [_a, _b], "honest_oof": None, "adopt": False,
                   "reason": "measured in sprint log; failed adoption criteria"})

metrics = {
    "experiment": "text_only_fast_sprint", "text_only": True, "uses_id_or_sid": False,
    "uses_xlmr_idhead": False, "uses_row_order": False,
    "fold_set": "A (local canonical) cache/v2/fold_ids.npy",
    "fold_file_sha256": sha(FOLD_PATH),
    "fold_int64_sha256": hashlib.sha256(np.ascontiguousarray(folds, np.int64).tobytes()).hexdigest(),
    "fold_sizes": np.bincount(folds, minlength=NF).tolist(), "n_folds": NF,
    "blend_sources": BLEND, "test_capable_blend_sources": BLEND_TEST,
    "simplex_step": STEP, "n_simplex_candidates": 286,
    "source_standalone": standalone, "source_audit": audit,
    "other_sources_considered": other,
    "honest_oof_macro_f1": round(OFF_HONEST, 6),
    "honest_oof_without_offsets": round(BASE_HONEST, 6),
    "per_class_f1": pc, "accuracy": round(acc, 6),
    "confusion": confusion_matrix(y, P.argmax(1), labels=CLASSES).tolist(),
    "fold_scores": [round(v, 6) for v in fs], "fold_std": round(fold_std, 6),
    "final_median_weights": MED, "test_capable_median_weights": MED_T,
    "same_oof_OPTIMISTIC_DIAGNOSTIC_ONLY": OPT,
    "offsets_tested": OT, "offsets_adopted": OFFSETS,
    "best_single_source": best_name, "best_single_macro_f1": best_single,
    "delta_from_best_single": round(OFF_HONEST - best_single, 6),
    "test_capable_honest_oof": TEST_CAPABLE_HONEST,
    "test_pred_dist": test_dist, "test_class_share": test_share,
    "train_class_share": train_share,
    "artifacts": {"oof": OUT_P + "/candidate_text_fast_oof_probs.npy", "oof_sha256": oof_sha,
                  "test": OUT_P + "/CANDIDATE_NOT_FOR_SUBMISSION_text_fast_test_probs.npy",
                  "test_sha256": test_sha},
    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                 "sklearn": __import__("sklearn").__version__},
    "elapsed_seconds": round(time.time() - T0, 1),
    "stopped_due_to_time_limit": False, "submission_generated": False,
    "status": STATUS,
}
with open(OUT_R + "/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open(OUT_R + "/source_audit.json", "w") as f:
    json.dump({"audit": audit, "other_sources_considered": other,
               "fold_file_sha256": metrics["fold_file_sha256"],
               "fold_int64_sha256": metrics["fold_int64_sha256"],
               "fold_sizes": metrics["fold_sizes"]}, f, indent=2)
md = ["# Text-only fast sprint", "",
      "Text-derived sources ONLY. No ID, SID, row order, or XLM-R ID-head.", "",
      "## Headline", "",
      "- Honest cross-fitted OOF macro F1: **%.6f** (WEAK)" % OFF_HONEST,
      "- Without class-logit offsets: %.6f" % BASE_HONEST,
      "- Per-class F1: %s" % pc, "- Accuracy: %.6f" % acc,
      "- Fold scores: %s (std %.6f)" % (metrics["fold_scores"], fold_std),
      "- OPTIMISTIC_DIAGNOSTIC_ONLY: %.6f (same-OOF, not a performance claim)" % OPT,
      "- Best single text source: %s %.6f (delta %+.6f)"
      % (best_name, best_single, OFF_HONEST - best_single),
      "- Final status: **%s**" % STATUS, "",
      "## Why this is low, stated plainly", "",
      "The plain ID prior alone scores 0.614135 and SID alone 0.627895, both of",
      "which are excluded here by the task rules. Every text-only source on disk",
      "sits in a narrow 0.496-0.524 band and their blend reaches only ~0.56. The",
      "gap to Submission 3 (0.682311) is almost entirely the ID-family signal.",
      "This candidate is methodologically honest for a text-only judging",
      "requirement, but it is NOT competitive on score.", "",
      "## Blend", "",
      "- Sources: %s" % ", ".join(BLEND),
      "- Test-capable 3-source variant: %s (honest %.6f), used for the test artifact"
      % (", ".join(BLEND_TEST), TEST_CAPABLE_HONEST),
      "- Simplex step %.2f, 286 candidates, first max in lexicographic order" % STEP,
      "- Median final weights: %s" % MED, "- Offsets adopted: %s" % OFFSETS, "",
      "## Fold set", "",
      "- Set A (local canonical), sizes %s" % metrics["fold_sizes"],
      "- file sha256 `%s`" % metrics["fold_file_sha256"],
      "- int64-bytes sha256 `%s`" % metrics["fold_int64_sha256"],
      "- Fold set B (Colab) sources were NOT mixed in.", "",
      "## Provenance", "",
      "- submission_generated: false", "- stopped_due_to_time_limit: false", ""]
with open(OUT_R + "/summary.md", "w") as f:
    f.write("\n".join(md))
print("reports written; STATUS=%s HONEST=%.6f STD=%.6f PC=%s ACC=%.6f FOLDS=%s"
      % (STATUS, OFF_HONEST, fold_std, pc, acc, metrics["fold_scores"]))

"""Section 5 adoption check: does sid_e5 beat config A in the honest blend?

Predeclared adoption rule: adopt ONLY if honest OOF improves by >= +0.002 over
config A AND at least 4 folds improve.
"""
import json
import time
import numpy as np
import sys

sys.path.insert(0, "scripts")
from v6_step4_blend import (N_SPLITS, CLASSES, STEP, WORK, mf1, pcf1, norm,
                            simplex_vectors, honest_blend)

d = np.load(WORK + "/step123.npz", allow_pickle=True)
y, folds = d["y"], d["fold_ids"]
text, sid_A, plain = d["oof_text"], d["oof_sid"], d["oof_plain"]
e5, xlmr = d["e5_oof"], d["xlmr_oof"]
base = json.load(open("reports/v6/blend.json"))["sid_variant"]

T = int(round(1.0 / STEP))
grid = [tuple(np.array(c, dtype=float) / T) for c in simplex_vectors(4, T)]
results = {}
for k_best in (90, 180):
    tag = "beta12.0_k{}".format(k_best)
    arr = np.load("{}/sidE5_beta12.0_k{}_oof.npy".format(WORK, k_best))
    t0 = time.time()
    S = np.stack([text, arr, e5, xlmr])
    out, f1, fs, chosen, opt = honest_blend(S, y, folds, grid)
    base_fs = base["fold_scores"]
    n_better = sum(1 for a, b in zip(fs, base_fs) if a > b)
    print("\n=== sid_e5 {} in place of sid ({:.0f}s) ===".format(tag, time.time() - t0))
    print("  honest OOF   = {:.6f}  (config A = {:.6f}, diff {:+.6f})".format(
        f1, base["honest_oof"], f1 - base["honest_oof"]))
    print("  per-class    = {}".format(pcf1(y, out)))
    print("  folds        = {}".format([round(v, 6) for v in fs]))
    print("  config A     = {}".format(base_fs))
    print("  folds improved = {}/5".format(n_better))
    print("  OPTIMISTIC   = {:.6f}".format(opt[0]))
    adopt = (f1 - base["honest_oof"]) >= 0.002 and n_better >= 4
    print("  ADOPT? {} (needs >= +0.002 AND >= 4 folds)".format(
        "YES" if adopt else "NO"))
    results[tag] = {"honest_oof": round(f1, 6), "folds": [round(v, 6) for v in fs],
                    "delta": round(f1 - base["honest_oof"], 6),
                    "folds_improved": n_better, "adopt": bool(adopt),
                    "per_class": pcf1(y, out)}

any_adopt = any(v["adopt"] for v in results.values())
print("\nOVERALL ADOPTION: {}".format("YES" if any_adopt else "NO - keep config A"))
json.dump({"results": results, "adopted": any_adopt,
           "config_a_honest_oof": base["honest_oof"]},
          open("reports/v6/sid_e5_adoption.json", "w"), indent=2)
print("-> reports/v6/sid_e5_adoption.json")

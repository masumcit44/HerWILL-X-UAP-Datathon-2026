"""v4 Phase 6: robustness checks + grouped bootstrap vs Submission 2."""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

SEED = 42
CLASSES = [0, 1, 2]
SUB2_REF = 0.657612
N_BOOT = 1000
os.makedirs("reports/v4", exist_ok=True)


def mf1(y, p):
    return float(f1_score(y, np.asarray(p).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def pcf1(y, p):
    return [float(v) for v in f1_score(y, np.asarray(p).argmax(1), average=None,
                                       labels=CLASSES, zero_division=0)]


def describe(name, y, P, folds, sub2_pred, groups):
    pred = P.argmax(1)
    fs = [mf1(y[folds == k], P[folds == k]) for k in range(5)]
    per_class = pcf1(y, P)
    disagree = int((pred != sub2_pred).sum())
    fixed = int(((sub2_pred != y) & (pred == y)).sum())
    broke = int(((sub2_pred == y) & (pred != y)).sum())
    conf = P.max(axis=1)
    srt = np.sort(P, axis=1)
    margin = srt[:, 2] - srt[:, 1]
    out = {
        "name": name,
        "macro_f1": round(mf1(y, P), 6),
        "per_class_f1": [round(v, 6) for v in per_class],
        "accuracy": round(float(accuracy_score(y, pred)), 6),
        "fold_scores": [round(v, 6) for v in fs],
        "fold_mean": round(float(np.mean(fs)), 6),
        "fold_std": round(float(np.std(fs)), 6),
        "worst_fold": round(float(np.min(fs)), 6),
        "confusion": confusion_matrix(y, pred, labels=CLASSES).tolist(),
        "pred_dist": np.bincount(pred, minlength=3).tolist(),
        "conf_vs_sub2": round(float((pred != sub2_pred).mean()), 6),
        "rows_changed_vs_sub2": disagree,
        "rows_corrected_vs_sub2": fixed,
        "rows_damaged_vs_sub2": broke,
        "mean_confidence": round(float(conf.mean()), 6),
        "low_margin_count_lt_0.10": int((margin < 0.10).sum()),
        "low_margin_frac": round(float((margin < 0.10).mean()), 6),
    }
    return out, pred


def bootstrap(y, P, pred_sub2, groups, n_boot=N_BOOT, seed=SEED):
    """Grouped (by cleaned text) bootstrap of Macro-F1 difference vs Submission 2."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    gidx = {g: np.flatnonzero(groups == g) for g in ug}
    deltas = np.empty(n_boot)
    base_pred = P.argmax(1)
    for b in range(n_boot):
        pick = rng.integers(0, len(ug), size=len(ug))
        sel = np.concatenate([gidx[ug[i]] for i in pick])
        d = (f1_score(y[sel], base_pred[sel], average="macro", labels=CLASSES,
                      zero_division=0)
             - f1_score(y[sel], pred_sub2[sel], average="macro", labels=CLASSES,
                        zero_division=0))
        deltas[b] = d
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_boot": n_boot, "seed": seed,
        "observed_delta": round(mf1(y, P) - mf1(y, pred_sub2), 6),
        "median_delta": round(float(np.median(deltas)), 6),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "frac_boot_delta_gt_0": round(float((deltas > 0).mean()), 4),
        "significant_at_95": bool(lo > 0),
    }


def main():
    sys_path = "src"
    import sys
    sys.path.insert(0, sys_path)
    import experiment_v2_harness as H
    train = pd.read_csv("train.csv", encoding="utf-8")
    y = train["y"].to_numpy(int)
    folds = np.load("cache/v2/fold_ids.npy")
    groups = pd.factorize(pd.Index(
        [H.group_key(H.clean_text(t)) for t in train["text"]], dtype="object"),
        sort=False)[0]

    sub2 = np.load("models/v3/v3_3_frozen_e5_small/refined_oof_probs.npy")
    sub2_pred = sub2.argmax(1)
    cands = {"submission2_reference": sub2}
    p_off = "predictions/v4_candidates/candidate_logit_offset_oof_probs.npy"
    if os.path.exists(p_off):
        cands["logit_offset"] = np.load(p_off)
    p_gate = "predictions/v4_candidates/candidate_nested_gate_oof_probs.npy"
    if os.path.exists(p_gate):
        cands["nested_gate"] = np.load(p_gate)
    p_bc = "predictions/v4_candidates/candidate_B_oof_probs.npy"
    if os.path.exists(p_bc):
        cands["consensus_B"] = np.load(p_bc)
    p_a = "predictions/v4_candidates/candidate_A_oof_probs.npy"
    if os.path.exists(p_a):
        cands["candidate_A"] = np.load(p_a)
    p_c = "predictions/v4_candidates/candidate_C_oof_probs.npy"
    if os.path.exists(p_c):
        cands["candidate_C"] = np.load(p_c)

    report = {}
    print("{:<24} {:>10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "candidate", "macroF1", "cls0", "cls1", "cls2", "worstF", "foldstd",
        "chg_vs_s2"))
    print("-" * 96)
    for nm, P in cands.items():
        d, _ = describe(nm, y, P, folds, sub2_pred, groups)
        report[nm] = d
        print("{:<24} {:>10.6f} {:>9.6f} {:>9.6f} {:>9.6f} {:>9.6f} {:>9.6f} {:>9.4f}"
              .format(nm, d["macro_f1"], *d["per_class_f1"], d["worst_fold"],
                      d["fold_std"], d["conf_vs_sub2"]))

    print()
    boots = {}
    for nm, P in cands.items():
        if nm == "submission2_reference":
            continue
        boots[nm] = bootstrap(y, P, sub2_pred, groups)
        b = boots[nm]
        print("bootstrap {:<22} median_delta={:+.6f}  CI95=[{:+.6f}, {:+.6f}]  "
              "P(delta>0)={:.3f}  significant={}".format(
                  nm, b["median_delta"], b["ci95"][0], b["ci95"][1],
                  b["frac_boot_delta_gt_0"], b["significant_at_95"]))

    with open("reports/v4/robustness_bootstrap.json", "w") as f:
        json.dump({"candidates": report, "bootstrap": boots,
                   "sub2_ref": SUB2_REF}, f, indent=2)
    print("\n-> reports/v4/robustness_bootstrap.json")


if __name__ == "__main__":
    main()

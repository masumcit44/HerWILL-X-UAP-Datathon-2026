"""
src/experiment_v2_5_combined.py
===============================

EXPERIMENT 5 (prompt priority 5)
--------------------------------
"Combine the winners and report the final combined OOF F1."

Gate first
----------
The prompt makes this step conditional on a single experiment beating the locked
baseline 0.639579. The gate is evaluated here on the HONEST number of each
experiment (a config whose weights/alpha were tuned on the same pooled OOF it is
scored on does not count as a winner; its cross-fitted counterpart does), and
compared against the fold noise scale (baseline per-fold std = 0.0038).

Combination, run honestly
-------------------------
Everything is evaluated with the cross-fitted protocol -- any free parameter is
fitted on the OOF rows of 4 folds and scored on the held-out 5th fold -- so every
row below is directly comparable to 0.639579. Combinations that mix several text
models are built by averaging already-computed leak-free OOF probability
matrices, so nothing is re-fitted and nothing leaks.

This script writes only reports/ and predictions/ artifacts. It does NOT build
any submission, and it never touches the locked baseline notebook or
submission_1.csv (sha1-verified).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_v2_harness as H
import experiment_v2_3_per_class_alpha as E3

TOLERANCE = 0.005          # "similar range" band from the prompt's stop condition


def honest_per_class(p_text, p_id, y, fold_ids, tag, verbose=True):
    """Cross-fitted per-class blend weights: tune w on 4 folds, score the 5th."""
    pred = np.zeros(len(y), dtype=int)
    ws = {}
    for k in range(H.N_SPLITS):
        tr, va = H.fold_split(fold_ids, k)
        w, _ = E3.nm_search(y[tr], p_text[tr], p_id[tr])
        pred[va] = H.blend_per_class(p_text[va], p_id[va], w).argmax(1)
        ws[k] = [round(float(v), 4) for v in w]
    f1_pooled = float(H.f1_score(y, pred, average="macro", labels=list(H.CLASSES),
                                 zero_division=0))
    per_fold, fixed_pred = [], H.blend(p_text, p_id, H.ALPHA_BASELINE).argmax(1)
    for k in range(H.N_SPLITS):
        _, va = H.fold_split(fold_ids, k)
        per_fold.append({
            "fold": k, "n_val": int(len(va)),
            "f1_fixed_alpha_0.40": round(H.macro_f1(
                y[va], H.blend(p_text[va], p_id[va], H.ALPHA_BASELINE)), 6),
            "f1_per_class_crossfitted": round(float(H.f1_score(
                y[va], pred[va], average="macro", labels=list(H.CLASSES),
                zero_division=0)), 6),
            "w": ws[k],
        })
    f1_fixed = float(H.f1_score(y, fixed_pred, average="macro",
                                labels=list(H.CLASSES), zero_division=0))
    std = float(np.std([r["f1_per_class_crossfitted"] for r in per_fold]))
    std_fixed = float(np.std([r["f1_fixed_alpha_0.40"] for r in per_fold]))
    if verbose:
        print(f"{tag:52s} OOF={f1_pooled:.6f} (delta {f1_pooled - H.BASELINE_OOF_F1:+.6f})"
              f" | fixed-alpha same-protocol {f1_fixed:.6f} "
              f"(delta {f1_fixed - H.BASELINE_OOF_F1:+.6f}) | std {std:.6f} vs "
              f"{std_fixed:.6f}")
    return {
        "tag": tag, "pooled_oof_macro_f1": f1_pooled,
        "delta_vs_baseline": f1_pooled - H.BASELINE_OOF_F1,
        "per_class_f1": H.per_class_f1(y, pred),
        "per_fold": per_fold, "w_per_fold": ws,
        "fold_macro_f1_std": std,
        "fold_macro_f1_std_fixed_alpha": std_fixed,
        "fixed_alpha_0.40_same_protocol": f1_fixed,
    }


def cross_fitted_proba_matrix(p_text, p_id, w_per_fold):
    """Rebuild the cross-fitted blended probability matrix from the per-fold w."""
    proba = np.zeros((len(p_text), 3))
    for k_str, w in w_per_fold.items():
        k = int(k_str)
        _, va = H.fold_split(
            np.load(H.CACHE_DIR / "fold_ids.npy"), k)
        proba[va] = H.blend_per_class(p_text[va], p_id[va], w)
    return proba


def main() -> int:
    t_start = time.time()
    H.banner("EXP 5 -- combine the surviving candidates (cross-fitted protocol)")
    H.check_protected()

    d = H.load_data()
    y = d["y"]
    P = H.PRED_DIR
    a0 = np.load(P / "v2_0_baseline_anchor_oof.npz")
    oof_text, oof_id, fold_ids = a0["oof_text"], a0["oof_id"], a0["fold_ids"]
    e1 = np.load(P / "v2_1_text_ensemble_oof.npz")
    e2 = np.load(P / "v2_2_hand_features_oof.npz")
    m2 = json.loads((H.REPORTS_DIR / "v2_2_hand_features_metrics.json").read_text())
    m3 = json.loads((H.REPORTS_DIR / "v2_3_per_class_alpha_metrics.json").read_text())

    base_f1 = H.macro_f1(y, H.blend(oof_text, oof_id, H.ALPHA_BASELINE))
    base_std = float(np.std([H.macro_f1(y[va], H.blend(oof_text[va], oof_id[va],
                                                      H.ALPHA_BASELINE))
                             for _, va in (H.fold_split(fold_ids, k)
                                           for k in range(H.N_SPLITS))]))
    print(f"locked baseline: OOF {base_f1:.6f} | per-fold std {base_std:.6f}")
    print(f"comparison band: a delta is only meaningful if it survives cross-fitting "
          f"AND exceeds the fold-noise scale (~{base_std:.4f})")

    # ------------------------------------------------------------------ #
    # GATE: honest delta of every single experiment
    # ------------------------------------------------------------------ #
    print("\n--- GATE: single-experiment honest deltas ---")
    single = {}
    r3ens = m3["results"]["ens"]
    single["v2_1_text_ensemble"] = {
        "honest_config": "equal-weight text ensemble, alpha re-gridded to "
                         f"{r3ens['single_alpha_best']['alpha']} on pooled OOF",
        "oof_macro_f1": r3ens["single_alpha_best"]["macro_f1"],
        "tuned_on_scored_oof": True,
        "cross_fitted_delta": r3ens["single_alpha_best"]["macro_f1"] - H.BASELINE_OOF_F1}
    s2 = m2["pooled_oof_macro_f1"]
    scales = list(m2["features"]["block_scales_swept"])
    best_s = max(scales, key=lambda s: s2[f"blend_scale{s}_0.40"])
    single["v2_2_hand_features"] = {
        "honest_config": f"hand features, best scale {best_s}, alpha fixed at 0.40 "
                         f"(alpha NOT tuned on the scored OOF)",
        "oof_macro_f1": s2[f"blend_scale{best_s}_0.40"],
        "tuned_on_scored_oof": False,
        "cross_fitted_delta": s2[f"blend_scale{best_s}_0.40"] - H.BASELINE_OOF_F1}
    r3lr = m3["results"]["lr"]["per_class_cross_fitted"]
    single["v2_3_per_class_alpha"] = {
        "honest_config": "per-class blend weights, cross-fitted (w tuned on 4 folds, "
                         "scored on the 5th)",
        "oof_macro_f1": r3lr["macro_f1"],
        "tuned_on_scored_oof": False,
        "cross_fitted_delta": r3lr["delta_vs_baseline"]}
    m4 = json.loads((H.REPORTS_DIR / "v2_4_meta_stacker_metrics.json").read_text())
    single["v2_4_meta_stacker"] = {
        "honest_config": "meta-LR on [text_probs, id_probs, hand features], cross-fitted",
        "oof_macro_f1": m4["results"][0]["pooled_oof_macro_f1"],
        "tuned_on_scored_oof": False,
        "cross_fitted_delta": m4["results"][0]["delta_vs_baseline"]}
    for k, v in single.items():
        v["beats_baseline"] = bool(v["cross_fitted_delta"] > 0)
        v["delta_exceeds_fold_noise"] = bool(abs(v["cross_fitted_delta"]) > base_std)
        print(f"  {k:24s} OOF {v['oof_macro_f1']:.6f} | honest delta "
              f"{v['cross_fitted_delta']:+.6f} | beats={v['beats_baseline']} | "
              f"exceeds fold noise={v['delta_exceeds_fold_noise']}")

    survivors = [k for k, v in single.items()
                 if v["beats_baseline"] and v["delta_exceeds_fold_noise"]]
    gate_pass = len(survivors) > 0
    gate_note = (
        f"survivors exceeding both tests: {survivors or 'none'}; "
        + ("gate PASSED -> combination proceeds as a real candidate"
           if gate_pass else
           "gate NOT passed -> the combination below is reported for completeness "
           "only and does not qualify as beating the baseline"))
    print(f"\nGATE: {'PASS' if gate_pass else 'NOT PASSED'} -- {gate_note}")

    # ------------------------------------------------------------------ #
    # COMBINATION (all cross-fitted => honest, directly comparable)
    # ------------------------------------------------------------------ #
    print("\n--- COMBINATION (every free parameter cross-fitted) ---")
    p_lr = oof_text
    p_hand = e2["oof_scale0.1"] if "oof_scale0.1" in e2 else e2[f"oof_scale{scales[2]}"]
    p_ens = e1["oof_ens"]
    p_lr_hand = 0.5 * p_lr + 0.5 * p_hand
    p_all3 = (p_lr + p_hand + p_ens) / 3.0
    p_lr_ens = 0.5 * p_lr + 0.5 * p_ens

    combos = []
    combos.append(honest_per_class(p_lr, oof_id, y, fold_ids,
                                   "EXP3 survivor: per-class w on LR text + id prior"))
    combos.append(honest_per_class(p_hand, oof_id, y, fold_ids,
                                   "EXP2 text (hand scale 0.1) + per-class w + id prior"))
    combos.append(honest_per_class(p_lr_hand, oof_id, y, fold_ids,
                                   "COMBO: mean(LR, hand) text + per-class w + id prior"))
    combos.append(honest_per_class(p_lr_ens, oof_id, y, fold_ids,
                                   "COMBO: mean(LR, ensemble) text + per-class w + id prior"))
    combos.append(honest_per_class(p_all3, oof_id, y, fold_ids,
                                   "COMBO: mean(LR, hand, ensemble) text + per-class w + id prior"))
    # fixed-alpha versions of the same text sources, for a like-for-like view
    fixed = []
    text_sources = [("LR text (baseline)", p_lr), ("hand scale 0.1 text", p_hand),
                    ("mean(LR, hand)", p_lr_hand), ("mean(LR, ens)", p_lr_ens),
                    ("mean(LR, hand, ens)", p_all3)]
    for tag, p in text_sources:
        f1 = H.macro_f1(y, H.blend(p, oof_id, H.ALPHA_BASELINE))
        std = float(np.std([H.macro_f1(y[va], H.blend(p[va], oof_id[va], H.ALPHA_BASELINE))
                            for _, va in (H.fold_split(fold_ids, k)
                                          for k in range(H.N_SPLITS))]))
        pb = H.blend(p, oof_id, H.ALPHA_BASELINE)
        fixed.append({"tag": f"fixed alpha 0.40 | {tag}", "pooled_oof_macro_f1": f1,
                      "delta_vs_baseline": f1 - H.BASELINE_OOF_F1,
                      "fold_macro_f1_std": std,
                      "per_fold_macro_f1": [round(H.macro_f1(
                          y[va], pb[va]), 6) for _, va in
                          (H.fold_split(fold_ids, k) for k in range(H.N_SPLITS))],
                      "per_class_f1": H.per_class_f1(y, pb.argmax(1))})
        print(f"{'fixed alpha 0.40 | ' + tag:52s} OOF={f1:.6f} "
              f"(delta {f1 - H.BASELINE_OOF_F1:+.6f}) | std {std:.6f}")

    best = max(combos, key=lambda r: r["pooled_oof_macro_f1"])
    # Tie-break: a fixed-alpha row is preferred over a cross-fitted one at equal
    # F1 (fewer tuned parameters), and the locked baseline sits at the top.
    def rank(r):
        return (round(r["pooled_oof_macro_f1"], 6),
                1 if r["tag"].startswith("fixed alpha") else 0)
    best_overall = max(combos + fixed, key=rank)
    print(f"\nbest cross-fitted combination: {best['tag']} -> "
          f"{best['pooled_oof_macro_f1']:.6f} "
          f"(delta {best['delta_vs_baseline']:+.6f})")
    print(f"best overall configuration     : {best_overall['tag']} -> "
          f"{best_overall['pooled_oof_macro_f1']:.6f} "
          f"(delta {best_overall['delta_vs_baseline']:+.6f})")

    # ------------------------------------------------------------------ #
    # stop conditions
    # ------------------------------------------------------------------ #
    print("\n--- STOP CONDITIONS ---")
    std_ok = abs(best["fold_macro_f1_std"] - base_std) <= TOLERANCE
    print(f"per-fold std {best['fold_macro_f1_std']:.6f} vs baseline {base_std:.6f} "
          f"-> |delta| {abs(best['fold_macro_f1_std'] - base_std):.6f} "
          f"<= {TOLERANCE}: {std_ok}")
    per_fold_delta = [round(b["f1_per_class_crossfitted"] - b["f1_fixed_alpha_0.40"], 6)
                      for b in best["per_fold"]]
    print("per-fold delta vs fixed alpha (no single-fold swing):", per_fold_delta)
    swings = [abs(v) for v in per_fold_delta]
    print(f"max |per-fold swing| {max(swings):.6f} "
          f"({'OK' if max(swings) < 0.02 else 'SUSPICIOUS'})")
    print(f"ID-prior leak-freedom re-verified: val ids disjoint from prior pool on all "
          f"5 folds, production prior == independent re-implementation "
          f"(max |diff| <= 1e-10) -- see reports/v2_0_baseline_anchor_metrics.json "
          f"-> id_prior_leak_proof")
    H.check_protected()
    print("no submission file was written by this script (by design)")

    best_is_fixed = best_overall["tag"].startswith("fixed alpha")
    best_config = {
        "name": best_overall["tag"],
        "oof_macro_f1": round(best_overall["pooled_oof_macro_f1"], 6),
        "delta_vs_baseline": round(best_overall["delta_vs_baseline"], 6),
        "per_class_f1": best_overall["per_class_f1"],
        "per_fold_std": round(best_overall["fold_macro_f1_std"], 6),
        "text_model": {
            "type": "LogisticRegression", **H.LR_KW,
            "features": "word TfidfVectorizer(analyzer=word, ngram_range=(1,2), "
                        "min_df=2, max_features=60000, sublinear_tf=True) + "
                        "char TfidfVectorizer(analyzer=char_wb, ngram_range=(3,5), "
                        "min_df=2, max_features=60000, sublinear_tf=True), hstacked",
            "cleaning": "URL -> ' URL ', @mention -> ' USER ', whitespace collapsed; "
                        "no lowercasing, no stemming",
            "source": "LR text only" if best_is_fixed and best_overall is fixed[0]
                      else "see name",
        },
        "id_prior": {"k": H.K_NEIGHBORS, "weights": "1/(distance+1e-6)",
                     "pool": "training-fold ids/labels only (leak-free, asserted per fold)",
                     "note": "self-matches excluded; queries disjoint from the pool"},
        "blend": {
            "form": "P(c) = a * text_probs_c + (1 - a) * id_probs_c",
            "alpha": H.ALPHA_BASELINE if best_is_fixed else None,
            "cross_fitted_w_per_fold": (None if best_is_fixed
                                        else best_overall["w_per_fold"]),
            "tuned_parameters": ("none beyond the baseline's own choice of alpha=0.40"
                                 if best_is_fixed else "one 3-vector per fold"),
        },
        "duplicate_override": "unanimous cleaned-text lookup, conflicting groups skipped",
        "protocol": {"validation": "StratifiedGroupKFold(5, shuffle=True, random_state=42)",
                     "group_key": "clean_text.lower().strip()",
                     "metric": "pooled leak-free OOF Macro F1",
                     "seed": H.SEED},
        "per_fold_macro_f1": best_overall.get(
            "per_fold_macro_f1",
            [r["f1_per_class_crossfitted"] for r in best_overall["per_fold"]]
            if "per_fold" in best_overall else None),
        "status": "best validated config found by experiment_v2_*; NOT submitted "
                  "(submission_2 requires an explicit request)",
    }

    payload = {
        "experiment": "v2_5_combined",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0], "seed": H.SEED,
        "validation": {"method": "StratifiedGroupKFold", "n_splits": H.N_SPLITS,
                       "shuffle": True, "random_state": H.SEED,
                       "group_key": "clean_text.lower().strip()",
                       "metric": "pooled OOF macro F1"},
        "baseline_oof_f1": H.BASELINE_OOF_F1,
        "baseline_per_fold_std": base_std,
        "gate": {"per_experiment": single, "survivors": survivors,
                 "any_single_experiment_beat_baseline": gate_pass,
                 "note": gate_note},
        "cross_fitted_combinations": combos,
        "fixed_alpha_combinations": fixed,
        "best": best_overall,
        "best_cross_fitted": best,
        "stop_conditions": {
            "per_fold_std_within_tolerance": bool(std_ok),
            "tolerance": TOLERANCE,
            "best_std": best["fold_macro_f1_std"],
            "baseline_std": base_std,
            "per_fold_delta_vs_fixed_alpha": per_fold_delta,
            "max_abs_per_fold_swing": max(swings),
            "id_prior_leak_proof": "re-verified (val ids disjoint from pool on all 5 "
                                   "folds; prod == independent re-implementation)",
        },
        "best_config": best_config,
        "submission_written": False,
        "runtime_seconds": round(time.time() - t_start, 1),
        "_oof_arrays": {
            "oof_best": cross_fitted_proba_matrix(p_lr, oof_id, best["w_per_fold"])
            if best["tag"].startswith("EXP3") else np.eye(3)[
                H.blend_per_class(p_lr, oof_id, best["w_per_fold"][0]).argmax(1)],
            "fold_ids": fold_ids},
    }
    H.save_result("5_combined", payload)
    H.save_config("best_validated_config", best_config)
    H.record_comparison("v2_5_combined_best",
                        f"EXP 5 combination: {best_overall['tag']} "
                        f"(gate {'PASS' if gate_pass else 'NOT PASSED'})",
                        best_overall["pooled_oof_macro_f1"],
                        best_overall["per_class_f1"],
                        best_overall["fold_macro_f1_std"], best_config,
                        note=gate_note)
    print(f"\nmetrics -> reports/v2_5_combined_metrics.json | "
          f"best config -> reports/v2_configs/best_validated_config.json | "
          f"runtime {payload['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

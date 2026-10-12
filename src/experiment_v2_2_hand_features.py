"""
src/experiment_v2_2_hand_features.py
====================================

EXPERIMENT 2 (prompt priority 2)
--------------------------------
Hand-crafted features for each row:

    bangla_char_ratio, latin_char_ratio, digit_ratio, punctuation_density,
    repeated_char_run_max, emoji_count, has_url, has_mention, text_length

appended as a dense block (StandardScaler'd) to the sparse TF-IDF matrix via
hstack; the text model (LogisticRegression C=4.0, class_weight='balanced') is
re-fitted on the augmented matrix. Reported alone and blended.

Leak discipline
---------------
* features are computed on the RAW text (has_url / has_mention are masked away
  by clean_text, so they are only informative pre-cleaning),
* the StandardScaler is fitted on the TRAINING rows of each fold only, so no
  validation row contributes to the scaling,
* the grouped folds and the leak-free ID prior are exactly the EXP 0 ones
  (asserted, not recomputed differently).

Extra, clearly-labelled diagnostic
----------------------------------
A StandardScaler'd 9-column dense block has L2 norm ~3 per row while each
L2-normalised TF-IDF row has norm 1, so at C=4.0 the dense block would dominate
the linear model. The block is therefore additionally reported at a few global
scales (DENSE_SCALES); scale=1.0 is the literal prompt specification, the rest
are a scale/regularisation sweep. Any scale chosen by looking at pooled OOF F1
is marked optimistic in the report.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_v2_harness as H
from sklearn.linear_model import LogisticRegression

DENSE_SCALES = (1.0, 0.3, 0.1, 0.03)
LITERAL_SCALE = 1.0


def main() -> int:
    t_start = time.time()
    H.banner("EXP 2 -- hand-crafted dense features hstacked onto TF-IDF")

    d = H.load_data()
    y, ids, clean_texts, raw_train = d["y"], d["ids"], d["clean_texts"], d["train"]
    fold_ids = np.load(H.CACHE_DIR / "fold_ids.npy")

    anchor = np.load(H.PRED_DIR / "v2_0_baseline_anchor_oof.npz")
    oof_id = anchor["oof_id"]
    oof_text_base = anchor["oof_text"]
    assert np.array_equal(anchor["fold_ids"], fold_ids), "fold ids differ from EXP 0"
    print(f"EXP 0 anchor: blend@0.40 = "
          f"{H.macro_f1(y, H.blend(oof_text_base, oof_id, H.ALPHA_BASELINE)):.6f}")

    raw = raw_train["text"].to_numpy()
    Hraw = H.hand_feature_matrix(raw)        # row-local: computed once, no leak
    print(f"hand-crafted block: {Hraw.shape} names={H.HAND_FEATURE_NAMES}")
    print("  column means (train):", np.round(Hraw.mean(axis=0), 4).tolist())
    print("  column stds  (train):", np.round(Hraw.std(axis=0), 4).tolist())

    n = len(y)
    oof_by_scale = {s: np.zeros((n, 3)) for s in DENSE_SCALES}
    oof_dense_only = np.zeros((n, 3))
    per_fold = []

    for k in range(H.N_SPLITS):
        tr, va = H.fold_split(fold_ids, k)
        Xtr, Xva = H.fold_tfidf(clean_texts, fold_ids, k, verbose=False)
        assert int(np.intersect1d(ids[va], ids[tr]).size) == 0, "prior pool leak"

        Ztr, scaler = H.hand_features(Hraw[tr], fit=True)     # train rows only
        Zva, _ = H.hand_features(Hraw[va], scaler=scaler)     # val uses train scaler
        assert Ztr.shape[1] == len(H.HAND_FEATURE_NAMES)

        row = {"fold": k, "n_val": int(len(va))}
        for s in DENSE_SCALES:
            A = sp.hstack([Xtr, Ztr * s], format="csr").astype(np.float32)
            B = sp.hstack([Xva, Zva * s], format="csr").astype(np.float32)
            t0 = time.time()
            clf = LogisticRegression(**H.LR_KW).fit(A, y[tr])
            p = clf.predict_proba(B)
            oof_by_scale[s][va] = p
            row[f"f1_text_scale{s}"] = round(H.macro_f1(y[va], p), 6)
            row[f"f1_blend_scale{s}_0.40"] = round(
                H.macro_f1(y[va], H.blend(p, oof_id[va], H.ALPHA_BASELINE)), 6)
            row[f"sec_scale{s}"] = round(time.time() - t0, 1)
            del A, B, clf

        # diagnostic: dense block on its own (does it carry any signal at all?)
        dclf = LogisticRegression(**dict(H.LR_KW)).fit(Ztr, y[tr])
        oof_dense_only[va] = dclf.predict_proba(Zva)
        row["f1_dense_only"] = round(H.macro_f1(y[va], oof_dense_only[va]), 6)
        per_fold.append(row)
        print(f"fold{k}: " + " | ".join(
            f"s={s}: text={row[f'f1_text_scale{s}']:.6f} "
            f"blend={row[f'f1_blend_scale{s}_0.40']:.6f}" for s in DENSE_SCALES)
            + f" | dense_only={row['f1_dense_only']:.6f}")

    print(pd.DataFrame(per_fold).to_string(index=False))

    pooled = {"dense_only_lr": H.macro_f1(y, oof_dense_only)}
    for s in DENSE_SCALES:
        pooled[f"text_scale{s}"] = H.macro_f1(y, oof_by_scale[s])
        pooled[f"blend_scale{s}_0.40"] = H.macro_f1(
            y, H.blend(oof_by_scale[s], oof_id, H.ALPHA_BASELINE))
    pooled["id_prior_only"] = H.macro_f1(y, oof_id)
    pooled["baseline_text_lr"] = H.macro_f1(y, oof_text_base)
    pooled["baseline_blend_0.40"] = H.macro_f1(
        y, H.blend(oof_text_base, oof_id, H.ALPHA_BASELINE))

    print("\npooled OOF Macro F1 (grouped, leak-free):")
    for kk, v in pooled.items():
        print(f"  {kk:26s} {v:.6f}")

    grids = {s: H.alpha_grid(y, oof_by_scale[s], oof_id) for s in DENSE_SCALES}
    bests = {s: max(g, key=lambda t: t[1]) for s, g in grids.items()}
    for s in DENSE_SCALES:
        print(f"  scale {s}: argmax alpha = {bests[s][0]:.2f} -> {bests[s][1]:.6f}")

    literal = pooled[f"blend_scale{LITERAL_SCALE}_0.40"]
    delta_literal = literal - H.BASELINE_OOF_F1
    best_scale = max(DENSE_SCALES,
                     key=lambda s: pooled[f"blend_scale{s}_0.40"])
    delta_best = pooled[f"blend_scale{best_scale}_0.40"] - H.BASELINE_OOF_F1
    print(f"\nvs baseline {H.BASELINE_OOF_F1:.6f}: "
          f"literal spec (scale={LITERAL_SCALE}, alpha=0.40) -> {delta_literal:+.6f} | "
          f"best scale {best_scale} (tuned on OOF, optimistic) -> {delta_best:+.6f}")

    payload = {
        "experiment": "v2_2_hand_features",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0], "seed": H.SEED,
        "validation": {"method": "StratifiedGroupKFold", "n_splits": H.N_SPLITS,
                       "shuffle": True, "random_state": H.SEED,
                       "group_key": "clean_text.lower().strip()",
                       "metric": "pooled OOF macro F1"},
        "features": {"names": H.HAND_FEATURE_NAMES, "computed_on": "raw text",
                     "scaler": "StandardScaler fitted on each fold's TRAIN rows only",
                     "block_scales_swept": list(DENSE_SCALES),
                     "literal_prompt_scale": LITERAL_SCALE},
        "per_fold": per_fold,
        "pooled_oof_macro_f1": pooled,
        "per_class_f1_literal_scale": H.per_class_f1(
            y, H.blend(oof_by_scale[LITERAL_SCALE], oof_id, H.ALPHA_BASELINE).argmax(1)),
        "alpha_grid_by_scale": {str(s): [{"alpha": a, "macro_f1": f} for a, f in g]
                                for s, g in grids.items()},
        "chosen_alpha_by_scale": {str(s): bests[s][0] for s in DENSE_SCALES},
        "baseline_oof_f1": H.BASELINE_OOF_F1,
        "delta_vs_baseline_literal": delta_literal,
        "delta_vs_baseline_best_scale_optimistic": delta_best,
        "beats_baseline_literal": bool(literal > H.BASELINE_OOF_F1),
        "fold_macro_f1_std": {
            f"blend_scale{s}_0.40": float(np.std([r[f"f1_blend_scale{s}_0.40"]
                                                 for r in per_fold]))
            for s in DENSE_SCALES},
        "runtime_seconds": round(time.time() - t_start, 1),
        "_oof_arrays": {"oof_id": oof_id, "fold_ids": fold_ids,
                        "oof_dense_only": oof_dense_only,
                        **{f"oof_scale{s}": oof_by_scale[s] for s in DENSE_SCALES}},
    }
    H.save_result("2_hand_features", payload)
    print(f"\nbeats baseline (literal spec): {payload['beats_baseline_literal']} "
          f"(delta {delta_literal:+.6f}) | runtime {payload['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

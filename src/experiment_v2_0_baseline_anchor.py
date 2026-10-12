"""
src/experiment_v2_0_baseline_anchor.py
=====================================

EXP 0 (not in the prompt's list, run first as a sanity anchor): independently
reproduce the locked baseline's grouped-fold OOF numbers inside the new
experiment_v2 harness.

Purpose
-------
* prove the harness's folds == the baseline's folds,
* prove the ID prior is leak-free (per-fold val&pool assertion + independent
  re-implementation diff),
* reproduce pooled OOF Macro F1 = 0.639579 at alpha = 0.40,
* cache the OOF text/id probability matrices so experiments 1-4 reuse one
  vectorisation + one LR pass instead of repeating them.

Nothing outside cache/ and reports/v2_* and predictions/v2_* is written.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_v2_harness as H
from sklearn.linear_model import LogisticRegression


def main() -> int:
    t_start = time.time()
    H.banner("EXP 0 -- baseline anchor: reproduce OOF Macro F1 = 0.639579")

    print("input checksums:", H.input_checksums())
    d = H.load_data()
    y, ids, clean_texts = d["y"], d["ids"], d["clean_texts"]

    fold_ids = H.build_folds(clean_texts, y)
    H.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(H.CACHE_DIR / "fold_ids.npy", fold_ids)

    fold_rows = H.prove_id_prior_leak_free(clean_texts, y, ids, fold_ids)
    print("ID-prior leak proof: PASS (val&pool = 0 on all folds, "
          "prod == independent re-implementation)")

    oof_text = np.zeros((len(y), 3))
    oof_id = np.zeros((len(y), 3))
    oof_override = np.zeros(len(y), dtype=int)
    per_fold = []

    for k in range(H.N_SPLITS):
        tr, va = H.fold_split(fold_ids, k)
        Xtr, Xva = H.fold_tfidf(clean_texts, fold_ids, k)
        t0 = time.time()
        clf = LogisticRegression(**H.LR_KW)
        clf.fit(Xtr, y[tr])
        p_text = clf.predict_proba(Xva)
        p_id = H.id_prior(ids[va], ids[tr], y[tr], tag=f"fold{k} prior")
        p_blend = H.blend(p_text, p_id, H.ALPHA_BASELINE)
        lookup, n_conflict = H.unanimous_lookup(clean_texts[tr], y[tr])
        assert set(lookup).issubset(set(clean_texts[tr].tolist())), "override leak"
        ov, hits = H.apply_override(clean_texts[va], p_blend.argmax(1), lookup)

        oof_text[va], oof_id[va], oof_override[va] = p_text, p_id, ov
        row = {
            "fold": k, "n_train": int(len(tr)), "n_val": int(len(va)),
            "macro_f1_text": round(H.macro_f1(y[va], p_text), 6),
            "macro_f1_id": round(H.macro_f1(y[va], p_id), 6),
            "macro_f1_blend_0.40": round(H.macro_f1(y[va], p_blend), 6),
            "macro_f1_blend_0.40_override": round(
                float(H.f1_score(y[va], ov, average="macro",
                                 labels=list(H.CLASSES), zero_division=0)), 6),
            "override_rows": int(hits), "override_conflicting_groups_skipped": n_conflict,
            "seconds": round(time.time() - t0, 1),
        }
        per_fold.append(row)
        print(f"fold{k}: text={row['macro_f1_text']:.6f} id={row['macro_f1_id']:.6f} "
              f"blend@0.40={row['macro_f1_blend_0.40']:.6f} "
              f"override={row['macro_f1_blend_0.40_override']:.6f}")

    import pandas as pd
    print(pd.DataFrame(per_fold).to_string(index=False))

    pooled = {
        "text_only": H.macro_f1(y, oof_text),
        "id_prior_only": H.macro_f1(y, oof_id),
        "blend_0.40": H.macro_f1(y, H.blend(oof_text, oof_id, H.ALPHA_BASELINE)),
    }
    grid = H.alpha_grid(y, oof_text, oof_id)
    chosen_alpha, best_f1 = max(grid, key=lambda t: t[1])
    f1s = np.array([f for _, f in grid])

    print("\npooled OOF (grouped, leak-free):")
    for kk, v in pooled.items():
        print(f"  {kk:14s} {v:.6f}")
    print("alpha grid:", {f"{a:.2f}": round(f, 6) for a, f in grid})
    print(f"argmax alpha = {chosen_alpha:.2f} -> {best_f1:.6f}")

    delta = pooled["blend_0.40"] - H.BASELINE_OOF_F1
    anchor_ok = abs(pooled["blend_0.40"] - H.BASELINE_OOF_F1) < 1e-6
    print(f"baseline anchor = {H.BASELINE_OOF_F1:.6f} | reproduced = "
          f"{pooled['blend_0.40']:.6f} | delta = {delta:+.6f} | "
          f"{'MATCH' if anchor_ok else 'MISMATCH -- investigate'}")

    blend_pred = H.blend(oof_text, oof_id, H.ALPHA_BASELINE).argmax(1)
    payload = {
        "experiment": "v2_0_baseline_anchor",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "seed": H.SEED,
        "validation": {"method": "StratifiedGroupKFold", "n_splits": H.N_SPLITS,
                       "shuffle": True, "random_state": H.SEED,
                       "group_key": "clean_text.lower().strip()",
                       "metric": "pooled OOF macro F1"},
        "input_checksums": H.input_checksums(),
        "per_fold": per_fold,
        "id_prior_leak_proof": fold_rows,
        "pooled_oof_macro_f1": pooled,
        "per_class_f1_blend_0.40": H.per_class_f1(y, blend_pred),
        "alpha_grid": [{"alpha": a, "macro_f1": f} for a, f in grid],
        "chosen_alpha": chosen_alpha, "chosen_macro_f1": best_f1,
        "baseline_oof_f1": H.BASELINE_OOF_F1,
        "delta_vs_baseline": delta,
        "anchor_reproduced": bool(anchor_ok),
        "fold_macro_f1_std": {
            "text_only": float(np.std([r["macro_f1_text"] for r in per_fold], ddof=0)),
            "id_prior_only": float(np.std([r["macro_f1_id"] for r in per_fold], ddof=0)),
            "blend_0.40": float(np.std([r["macro_f1_blend_0.40"] for r in per_fold], ddof=0)),
        },
        "runtime_seconds": round(time.time() - t_start, 1),
        "_oof_arrays": {"oof_text": oof_text, "oof_id": oof_id,
                        "fold_ids": fold_ids, "oof_override": oof_override},
    }
    H.save_result("0_baseline_anchor", payload)
    print(f"\nanchor reproduced: {anchor_ok} | runtime {payload['runtime_seconds']}s")
    return 0 if anchor_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

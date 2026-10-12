"""
src/experiment_v2_1_text_ensemble.py
=====================================

EXPERIMENT 1 (prompt priority 1)
--------------------------------
Replace the single LogisticRegression(C=4.0, class_weight='balanced') text model
with an average of predicted probabilities from

    1. LogisticRegression(C=4.0, class_weight='balanced')  on word+char TF-IDF
    2. CalibratedClassifierCV(LinearSVC(class_weight='balanced')) on word+char TF-IDF
    3. ComplementNB() on word TF-IDF only

then re-run the full grouped leak-free OOF blend with this ensemble in place of
the old text_probs.

Reported for this experiment
----------------------------
* text-ENSEMBLE-ONLY pooled OOF Macro F1,
* new blend OOF Macro F1 at the baseline alpha = 0.40 (comparable to 0.639579),
* re-grid-searched alpha for the new text model (the optimum may shift),
* per-class F1, per-fold scores, delta vs 0.639579.

The same grouped OOF ID-prior matrix as EXP 0 is reused (identical folds, so
identical leak-free priors) -- it is reloaded, not recomputed, and the equality
is asserted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_v2_harness as H
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

SVC_KW = dict(class_weight="balanced", C=1.0, max_iter=5000, random_state=H.SEED)
CAL_KW = dict(method="sigmoid", cv=5)     # cv=5 = CalibratedClassifierCV default
NB_KW = dict()


def main() -> int:
    t_start = time.time()
    H.banner("EXP 1 -- text ensemble: LR + calibrated LinearSVC + ComplementNB")

    d = H.load_data()
    y, ids, clean_texts = d["y"], d["ids"], d["clean_texts"]
    fold_ids = np.load(H.CACHE_DIR / "fold_ids.npy")

    anchor = np.load(H.PRED_DIR / "v2_0_baseline_anchor_oof.npz")
    oof_id = anchor["oof_id"]
    assert np.array_equal(anchor["fold_ids"], fold_ids), "fold ids differ from EXP 0"
    print(f"reused leak-free OOF id-prior from EXP 0: pooled id-only F1 = "
          f"{H.macro_f1(y, oof_id):.6f}")

    n = len(y)
    oof_lr = np.zeros((n, 3))
    oof_svc = np.zeros((n, 3))
    oof_nb = np.zeros((n, 3))
    oof_ens = np.zeros((n, 3))
    per_fold = []

    for k in range(H.N_SPLITS):
        tr, va = H.fold_split(fold_ids, k)
        Xtr, Xva = H.fold_tfidf(clean_texts, fold_ids, k, verbose=False)
        Wtr, Wva = H.fold_tfidf_word_only(clean_texts, fold_ids, k, verbose=False)
        assert int(np.intersect1d(ids[va], ids[tr]).size) == 0, "prior pool leak"
        secs = {}

        t0 = time.time()
        lr = LogisticRegression(**H.LR_KW).fit(Xtr, y[tr])
        oof_lr[va] = lr.predict_proba(Xva)
        secs["lr"] = round(time.time() - t0, 1)

        t0 = time.time()
        cal = CalibratedClassifierCV(LinearSVC(**SVC_KW), **CAL_KW)
        cal.fit(Xtr, y[tr])
        oof_svc[va] = cal.predict_proba(Xva)
        secs["cal_svc"] = round(time.time() - t0, 1)

        t0 = time.time()
        nb = ComplementNB(**NB_KW).fit(Wtr, y[tr])
        oof_nb[va] = nb.predict_proba(Wva)
        secs["cnb"] = round(time.time() - t0, 1)

        oof_ens[va] = (oof_lr[va] + oof_svc[va] + oof_nb[va]) / 3.0
        row = {
            "fold": k, "n_val": int(len(va)),
            "f1_lr": round(H.macro_f1(y[va], oof_lr[va]), 6),
            "f1_cal_svc": round(H.macro_f1(y[va], oof_svc[va]), 6),
            "f1_cnb_word": round(H.macro_f1(y[va], oof_nb[va]), 6),
            "f1_ensemble_text": round(H.macro_f1(y[va], oof_ens[va]), 6),
            "f1_blend_ens_0.40": round(
                H.macro_f1(y[va], H.blend(oof_ens[va], oof_id[va], H.ALPHA_BASELINE)), 6),
            "f1_blend_lr_0.40": round(  # baseline text model, same fold
                H.macro_f1(y[va], H.blend(oof_lr[va], oof_id[va], H.ALPHA_BASELINE)), 6),
            **secs,
        }
        per_fold.append(row)
        print(f"fold{k}: lr={row['f1_lr']:.6f} svc={row['f1_cal_svc']:.6f} "
              f"cnb={row['f1_cnb_word']:.6f} | ens={row['f1_ensemble_text']:.6f} "
              f"blend@0.40={row['f1_blend_ens_0.40']:.6f} "
              f"(lr baseline {row['f1_blend_lr_0.40']:.6f})  {secs}")

    print(pd.DataFrame(per_fold).to_string(index=False))

    # ---------------- pooled OOF ----------------
    pooled = {
        "text_lr": H.macro_f1(y, oof_lr),
        "text_cal_svc": H.macro_f1(y, oof_svc),
        "text_cnb_word": H.macro_f1(y, oof_nb),
        "text_ensemble": H.macro_f1(y, oof_ens),
        "id_prior_only": H.macro_f1(y, oof_id),
        "blend_ensemble_0.40": H.macro_f1(y, H.blend(oof_ens, oof_id, H.ALPHA_BASELINE)),
        "blend_lr_0.40": H.macro_f1(y, H.blend(oof_lr, oof_id, H.ALPHA_BASELINE)),
    }
    print("\npooled OOF Macro F1 (grouped, leak-free):")
    for kk, v in pooled.items():
        print(f"  {kk:22s} {v:.6f}")

    grid = H.alpha_grid(y, oof_ens, oof_id)
    chosen_alpha, best_f1 = max(grid, key=lambda t: t[1])
    print("alpha grid (ensemble text):",
          {f"{a:.2f}": round(f, 6) for a, f in grid})
    print(f"argmax alpha = {chosen_alpha:.2f} -> {best_f1:.6f}")

    delta_fixed = pooled["blend_ensemble_0.40"] - H.BASELINE_OOF_F1
    delta_best = best_f1 - H.BASELINE_OOF_F1
    print(f"\nvs baseline {H.BASELINE_OOF_F1:.6f}: "
          f"alpha=0.40 -> {delta_fixed:+.6f} | best alpha -> {delta_best:+.6f}")

    # -------- diagnostic: coarse simplex weight search (tuned ON the OOF, so
    # this number is optimistic and is NOT the headline result) --------
    diag = []
    for wl in np.arange(0, 1.01, 0.1):
        for ws in np.arange(0, 1.01 - wl + 1e-9, 0.1):
            wn = 1.0 - wl - ws
            p = wl * oof_lr + ws * oof_svc + wn * oof_nb
            diag.append((round(float(wl), 2), round(float(ws), 2), round(float(wn), 2),
                         H.macro_f1(y, H.blend(p, oof_id, chosen_alpha))))
    diag.sort(key=lambda t: -t[3])
    print("diagnostic top-5 simplex weights (w_lr, w_svc, w_nb) -> blended F1 "
          "[optimistic, tuned on OOF]:")
    for r in diag[:5]:
        print(f"  {r[0]:.2f} {r[1]:.2f} {r[2]:.2f} -> {r[3]:.6f}")
    w_lr, w_svc, w_nb = diag[0][0], diag[0][1], diag[0][2]
    oof_ens_w = w_lr * oof_lr + w_svc * oof_svc + w_nb * oof_nb
    simplex_fixed_alpha = H.macro_f1(y, H.blend(oof_ens_w, oof_id, H.ALPHA_BASELINE))

    pred = H.blend(oof_ens, oof_id, chosen_alpha).argmax(1)
    payload = {
        "experiment": "v2_1_text_ensemble",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0], "seed": H.SEED,
        "validation": {"method": "StratifiedGroupKFold", "n_splits": H.N_SPLITS,
                       "shuffle": True, "random_state": H.SEED,
                       "group_key": "clean_text.lower().strip()",
                       "metric": "pooled OOF macro F1"},
        "text_model": {
            "members": ["LogisticRegression(C=4.0, class_weight=balanced) on word+char",
                        "CalibratedClassifierCV(LinearSVC(class_weight=balanced), "
                        "method=sigmoid, cv=5) on word+char",
                        "ComplementNB() on word TF-IDF only"],
            "combination": "equal-weight average of predicted probabilities",
            "linear_svc": SVC_KW, "calibration": CAL_KW,
        },
        "per_fold": per_fold,
        "pooled_oof_macro_f1": pooled,
        "per_class_f1_blend_ensemble": H.per_class_f1(y, pred),
        "alpha_grid": [{"alpha": a, "macro_f1": f} for a, f in grid],
        "chosen_alpha": chosen_alpha, "chosen_macro_f1": best_f1,
        "baseline_oof_f1": H.BASELINE_OOF_F1,
        "delta_vs_baseline_alpha_0.40": delta_fixed,
        "delta_vs_baseline_best_alpha": delta_best,
        "beats_baseline": bool(pooled["blend_ensemble_0.40"] > H.BASELINE_OOF_F1),
        "diagnostic_simplex_best": {
            "w_lr": w_lr, "w_svc": w_svc, "w_nb": w_nb,
            "blended_f1_at_chosen_alpha_optimistic": simplex_fixed_alpha,
            "note": "weights tuned on the same pooled OOF -> optimistic, diagnostic only",
        },
        "fold_macro_f1_std": {
            "text_ensemble": float(np.std([r["f1_ensemble_text"] for r in per_fold])),
            "blend_ensemble_0.40": float(np.std([r["f1_blend_ens_0.40"] for r in per_fold])),
        },
        "runtime_seconds": round(time.time() - t_start, 1),
        "_oof_arrays": {"oof_lr": oof_lr, "oof_svc": oof_svc, "oof_nb": oof_nb,
                        "oof_ens": oof_ens, "oof_id": oof_id, "fold_ids": fold_ids,
                        "oof_ens_w": oof_ens_w},
    }
    H.save_result("1_text_ensemble", payload)
    print(f"\nbeats baseline at alpha=0.40: {payload['beats_baseline']} "
          f"(delta {delta_fixed:+.6f}) | runtime {payload['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
src/experiment_v2_3_per_class_alpha.py
======================================

EXPERIMENT 3 (prompt priority 3)
--------------------------------
Per-class alpha instead of one alpha for all three classes: search a 3-vector of
blend weights w = (w0, w1, w2) maximising pooled OOF Macro F1 directly, with

    P(c | row) = w_c * text_probs[c] + (1 - w_c) * id_probs[c]

* Nelder-Mead (scipy.optimize.minimize) from several starts, as suggested,
* plus a coarse 0.05-step grid over [0,1]^3 as an independent check that the
  optimiser did not fall into a local optimum,
* plus a CROSS-FITTED variant: w is tuned on the OOF rows of 4 folds and scored
  on the held-out 5th fold, so the reported gain is not the product of tuning on
  the same rows that are scored (this is the honest number).

Both text sources are handled: the baseline LR (EXP 0) and the EXP 1 ensemble if
its OOF arrays are present. Reported number is always pooled OOF Macro F1.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_v2_harness as H
from scipy.optimize import minimize

GRID_STEP = 0.05
STARTS = [(0.40, 0.40, 0.40), (0.30, 0.40, 0.50), (0.50, 0.40, 0.30),
          (0.40, 0.30, 0.50), (0.20, 0.60, 0.60)]


def make_objective(y, p_text, p_id):
    def neg_f1(w):
        w = np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
        return -H.macro_f1(y, H.blend_per_class(p_text, p_id, w))
    return neg_f1


def nm_search(y, p_text, p_id, starts=STARTS):
    obj = make_objective(y, p_text, p_id)
    best_w, best_f1 = None, -np.inf
    for s0 in starts:
        res = minimize(obj, np.array(s0, dtype=float), method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000,
                                "maxfev": 2000})
        w = np.clip(res.x, 0.0, 1.0)
        f1 = -obj(w)
        if f1 > best_f1:
            best_w, best_f1 = w, f1
    return best_w, float(best_f1)


def grid_search(y, p_text, p_id, step=GRID_STEP):
    vals = np.round(np.arange(0.0, 1.0 + step / 2, step), 2)
    best_w, best_f1, n = None, -np.inf, 0
    for w in itertools.product(vals, repeat=3):
        n += 1
        f1 = H.macro_f1(y, H.blend_per_class(p_text, p_id, w))
        if f1 > best_f1:
            best_w, best_f1 = np.array(w, dtype=float), f1
    return best_w, float(best_f1), n


def local_refine(y, p_text, p_id, w0, half=0.05, step=0.01):
    """Fine local grid around w0 -- confirms NM landed on a local maximum."""
    base = np.asarray(w0, dtype=float)
    n_off = int(half / step)
    off = np.arange(-n_off, n_off + 1) * step
    grid_axes = [np.round(np.clip(base[i] + off, 0.0, 1.0), 4) for i in range(3)]
    best_w, best_f1 = base, -np.inf
    for w in itertools.product(*grid_axes):
        f1 = H.macro_f1(y, H.blend_per_class(p_text, p_id, w))
        if f1 > best_f1:
            best_w, best_f1 = np.array(w, dtype=float), f1
    return best_w, float(best_f1)


def cross_fitted(y, p_text, p_id, fold_ids, tuner):
    """Tune w on 4 folds of OOF rows, score the held-out fold. Honest."""
    pred = np.full(len(y), -1, dtype=int)
    ws = {}
    for k in range(H.N_SPLITS):
        tr, va = H.fold_split(fold_ids, k)
        w, _ = tuner(tr)
        p = H.blend_per_class(p_text[va], p_id[va], w)
        pred[va] = p.argmax(1)
        ws[k] = [round(float(v), 4) for v in w]
    return pred, ws


def run(y, p_text, p_id, fold_ids, label: str) -> dict:
    t0 = time.time()
    print(f"\n--- text source: {label} ---")

    base_fixed = H.macro_f1(y, H.blend(p_text, p_id, H.ALPHA_BASELINE))
    grid = H.alpha_grid(y, p_text, p_id)
    a_best, f1_a_best = max(grid, key=lambda t: t[1])
    print(f"single alpha @0.40        : {base_fixed:.6f}")
    print(f"single alpha best ({a_best:.2f}) : {f1_a_best:.6f}")

    w_nm, f1_nm = nm_search(y, p_text, p_id)
    w_gr, f1_gr, n_gr = grid_search(y, p_text, p_id)
    w_fine, f1_fine = local_refine(y, p_text, p_id, w_nm, half=0.05, step=0.01)
    print(f"per-class w, Nelder-Mead   : {np.round(w_nm, 4).tolist()} -> {f1_nm:.6f}")
    print(f"per-class w, {GRID_STEP}-grid ({n_gr} pts): "
          f"{np.round(w_gr, 4).tolist()} -> {f1_gr:.6f}")
    print(f"per-class w, local 0.01 grid: {np.round(w_fine, 4).tolist()} -> {f1_fine:.6f}")
    if f1_nm > f1_gr + 1e-9:
        print("  note: NM beat the 0.05 grid (grid is coarse); the local 0.01 grid "
              f"confirms the optimum ({f1_fine:.6f})")
    best_w, f1_tuned = (w_fine, f1_fine) if f1_fine >= f1_nm else (w_nm, f1_nm)

    # honest: cross-fitted per-class weights
    pred_cf, ws = cross_fitted(
        y, p_text, p_id, fold_ids,
        lambda rows: nm_search(y[rows], p_text[rows], p_id[rows]))
    f1_cf = float(H.f1_score(y, pred_cf, average="macro", labels=list(H.CLASSES),
                             zero_division=0))
    pred_cf_fixed = H.blend(p_text, p_id, H.ALPHA_BASELINE).argmax(1)
    f1_cf_fixed = float(H.f1_score(y, pred_cf_fixed, average="macro",
                                    labels=list(H.CLASSES), zero_division=0))
    print(f"cross-fitted per-class w   : {f1_cf:.6f} "
          f"(same-protocol fixed alpha 0.40: {f1_cf_fixed:.6f}, "
          f"delta {f1_cf - f1_cf_fixed:+.6f})")
    print(f"  per-fold w: {ws}")

    per_fold = []
    for k in range(H.N_SPLITS):
        _, va = H.fold_split(fold_ids, k)
        per_fold.append({
            "fold": k, "n_val": int(len(va)),
            "f1_single_alpha_0.40": round(
                H.macro_f1(y[va], H.blend(p_text[va], p_id[va], H.ALPHA_BASELINE)), 6),
            "f1_per_class_tuned_on_4folds": round(float(H.f1_score(
                y[va], pred_cf[va], average="macro", labels=list(H.CLASSES),
                zero_division=0)), 6),
            "w": ws[k],
        })
    for r in per_fold:
        print(f"  fold{r['fold']}: fixed={r['f1_single_alpha_0.40']:.6f} "
              f"per-class(cross-fitted)={r['f1_per_class_tuned_on_4folds']:.6f} "
              f"w={r['w']}")

    std_cf = float(np.std([r["f1_per_class_tuned_on_4folds"] for r in per_fold]))
    std_fx = float(np.std([r["f1_single_alpha_0.40"] for r in per_fold]))

    return {
        "text_source": label,
        "single_alpha_0.40": base_fixed,
        "single_alpha_best": {"alpha": a_best, "macro_f1": f1_a_best},
        "per_class_nelder_mead": {"w": [float(v) for v in w_nm], "macro_f1": f1_nm,
                                  "delta_vs_baseline_alpha_0.40":
                                      f1_nm - H.BASELINE_OOF_F1,
                                  "note": "tuned on the same pooled OOF -> optimistic"},
        "per_class_grid": {"w": [float(v) for v in w_gr], "macro_f1": f1_gr,
                           "step": GRID_STEP, "n_points": n_gr},
        "per_class_tuned_headline": {
            "w": [float(v) for v in best_w], "macro_f1": f1_tuned,
            "delta_vs_baseline": f1_tuned - H.BASELINE_OOF_F1,
            "method": "Nelder-Mead + local 0.01 grid refinement",
            "note": "tuned on the same pooled OOF -> optimistic"},
        "per_class_local_fine_grid": {"w": [float(v) for v in w_fine],
                                      "macro_f1": f1_fine, "step": 0.01},
        "per_class_cross_fitted": {"macro_f1": f1_cf,
                                   "fixed_alpha_0.40_same_protocol": f1_cf_fixed,
                                   "delta_vs_fixed_alpha": f1_cf - f1_cf_fixed,
                                   "delta_vs_baseline": f1_cf - H.BASELINE_OOF_F1,
                                   "per_fold_w": ws,
                                   "note": "w tuned on 4 folds, scored on the 5th"},
        "per_fold": per_fold,
        "fold_macro_f1_std": {"single_alpha_0.40": std_fx,
                              "per_class_cross_fitted": std_cf},
        "per_class_f1_cross_fitted": H.per_class_f1(y, pred_cf),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="lr,ens",
                    help="comma list of text OOF sources: lr, ens, ens_w")
    args = ap.parse_args()

    H.banner("EXP 3 -- per-class blend weights (3-vector alpha)")
    d = H.load_data()
    y = d["y"]

    anchor = np.load(H.PRED_DIR / "v2_0_baseline_anchor_oof.npz")
    oof_id, fold_ids = anchor["oof_id"], anchor["fold_ids"]
    sources = {"lr": anchor["oof_text"]}

    ens_path = H.PRED_DIR / "v2_1_text_ensemble_oof.npz"
    if ens_path.exists():
        e = np.load(ens_path)
        sources["ens"] = e["oof_ens"]
        if "oof_ens_w" in e:
            sources["ens_w"] = e["oof_ens_w"]
    missing = [s for s in args.sources.split(",") if s not in sources]
    if missing:
        print("requested but unavailable text sources:", missing,
              "| available:", list(sources))

    results = {label: run(y, sources[label], oof_id, fold_ids, label)
               for label in args.sources.split(",") if label in sources}

    payload = {
        "experiment": "v2_3_per_class_alpha",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0], "seed": H.SEED,
        "validation": {"method": "StratifiedGroupKFold", "n_splits": H.N_SPLITS,
                       "shuffle": True, "random_state": H.SEED,
                       "group_key": "clean_text.lower().strip()",
                       "metric": "pooled OOF macro F1"},
        "method": {"parametrisation": "P(c) = w_c * text_c + (1 - w_c) * id_c, w in [0,1]^3",
                   "optimiser": "scipy Nelder-Mead (multi-start) + 0.05 grid cross-check",
                   "starts": [list(s) for s in STARTS]},
        "baseline_oof_f1": H.BASELINE_OOF_F1,
        "results": results,
    }
    H.save_result("3_per_class_alpha", payload)
    print("\nsaved reports/v2_3_per_class_alpha_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

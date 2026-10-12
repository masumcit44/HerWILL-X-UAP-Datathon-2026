"""
src/revalidate_grouped_cv.py -- PART 1: independent re-validation of the
text + ID-locality-prior blend pipeline (HerWILL X UAP Datathon 2026).

Nothing from a previous run's summary or artifact is used as ground truth: every
number below is recomputed from the three CSVs in a fresh interpreter. The only
values carried over from before are
  (a) the recorded SHA-1 checksums of the protected inputs (Step 1 must PASS),
  (b) the previously chosen alpha = 0.40, which is RE-CONFIRMED, not assumed.

Steps -- all must PASS before Part 2 packaging runs:
  1. byte-identical checksums of train.csv / test.csv / sample_submission.csv
  2. full 5-fold StratifiedGroupKFold CV, group = cleaned + lowercased text
     (this removes the duplicate-text train/val leakage disclosed earlier),
     reporting per-fold and pooled OOF Macro F1 for text-only, id-prior-only and
     the blend at alpha = 0.40, plus a fresh alpha grid (0.00-1.00, step 0.05)
  3. id-prior leak-free proof: per-fold val ids vs prior-pool ids disjoint, and
     an independent naive k=20 nearest-id implementation diffed against the
     production one (must be < 1e-10)
  4. duplicate-text override leak-free: each fold's lookup built ONLY from that
     fold's train split; rows overridden and before/after Macro F1 reported
  5. bias check: predicted test class shares vs train class shares (>10pp flag)
  6. PASS/FAIL summary -> data_checks/part1_revalidation_status.json

Run:  python -u src/revalidate_grouped_cv.py
Exit code 0 = every step PASS (Part 2 allowed); 2 = any FAIL (Part 2 forbidden).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

# Production module: freshly imported in this process (no cached run state).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_text_id_blend as prod  # noqa: E402

ROOT = prod.ROOT
LOG_PATH = ROOT / "logs" / "revalidation_grouped.log"
prod.LOG_PATH = LOG_PATH                    # redirect the shared logger to our log

FOLD_SCORES_OUT = ROOT / "reports" / "revalidation_grouped_fold_scores.csv"
METRICS_OUT = ROOT / "reports" / "revalidation_grouped_metrics.json"
OOF_OUT = ROOT / "predictions" / "oof" / "oof_revalidation_grouped.csv"
STATUS_OUT = ROOT / "data_checks" / "part1_revalidation_status.json"

# Recorded checksums (from the earlier verified audit) -- Step 1 compares these.
RECORDED = {
    "train.csv": ("66c6b109c8fd187f4c2c474a09159905d25978cd", 5357729),
    "test.csv": ("9caf35ded5a6fe4b2765cdc3340f192d0994d072", 1290735),
    "sample_submission.csv": ("73345c6143ad4e155bac35de1bf8d5d81ed1d039", 105398),
}

SEED = 42
N_SPLITS = 5
ALPHA_GRID = [round(0.05 * i, 2) for i in range(21)]
ALPHA_PREVIOUS = 0.40
K = 20
EPS = 1e-6
BIAS_THRESHOLD_PP = 10.0
PROB_TOL = 1e-6
DIFF_TOL = 1e-10

CLASSES = (0, 1, 2)


def log(msg: str) -> None:
    prod.log(msg)          # writes to LOG_PATH (redirected above) and echoes


def group_key(text) -> str:
    """Grouping key: cleaned text, lowercased and stripped (grouping ONLY)."""
    return prod.clean_text(text).lower().strip()


# --------------------------------------------------------------------------- #
# Step 3 helper: INDEPENDENT naive k=20 nearest-id implementation.
# Deliberately different algorithm from production: full distance matrix + full
# argsort (no insertion-position window, no partition), processed in blocks.
# --------------------------------------------------------------------------- #
def independent_knn_id_prior(query_ids, train_ids, train_labels, k: int = K,
                            eps: float = EPS, n_classes: int = 3,
                            block: int = 300) -> np.ndarray:
    q = np.asarray(query_ids, dtype=np.int64).ravel()
    t = np.asarray(train_ids, dtype=np.int64).ravel()
    y = np.asarray(train_labels, dtype=np.int64).ravel()
    order = np.argsort(t, kind="stable")          # ascending id (documented tie rule)
    s_ids, s_y = t[order], y[order]
    kk = int(min(k, len(s_ids)))
    out = np.empty((len(q), n_classes), dtype=float)
    inf = np.iinfo(np.int64).max
    for start in range(0, len(q), block):
        chunk = q[start:start + block]
        d = np.abs(chunk[:, None] - s_ids[None, :])
        d = np.where(chunk[:, None] == s_ids[None, :], inf, d)   # never pick self
        idx = np.argsort(d, axis=1, kind="stable")[:, :kk]       # naive full sort
        ds = np.take_along_axis(d, idx, axis=1).astype(np.float64)
        ys = s_y[idx]
        good = ds < inf
        w = np.where(good, 1.0 / (np.where(good, ds, 0.0) + eps), 0.0)
        blk = np.stack([(w * (ys == c)).sum(axis=1) for c in range(n_classes)], axis=1)
        tot = blk.sum(axis=1, keepdims=True)
        blk = blk / np.where(tot > 0, tot, 1.0)
        blk[tot.ravel() <= 0] = 1.0 / n_classes
        out[start:start + len(chunk)] = blk
    return out


def macro_f1(y_true, proba) -> float:
    return float(f1_score(y_true, np.asarray(proba).argmax(axis=1), average="macro",
                          labels=list(CLASSES), zero_division=0))


# --------------------------------------------------------------------------- #
# STEP 1 -- protected inputs must be byte-identical to the recorded checksums
# --------------------------------------------------------------------------- #
def step1_checksums() -> tuple[bool, dict]:
    import hashlib

    detail, ok = {}, True
    for name, (sha, size) in RECORDED.items():
        raw = (ROOT / name).read_bytes()
        got_sha, got_size = hashlib.sha1(raw).hexdigest(), len(raw)
        match = (got_sha == sha and got_size == size)
        ok = ok and match
        detail[name] = {"recorded_sha1": sha, "actual_sha1": got_sha,
                        "recorded_bytes": size, "actual_bytes": got_size,
                        "match": bool(match)}
        log(f"STEP1 {name}: sha1_match={got_sha == sha} size_match={got_size == size} "
            f"({got_sha})")
    log(f"STEP1 overall: {'PASS' if ok else 'FAIL'}")
    return ok, detail


# --------------------------------------------------------------------------- #
# STEP 2 + 3 + 4 -- grouped CV, id-prior proof, override proof
# --------------------------------------------------------------------------- #
def step2_grouped_cv(train, clean_texts, y, ids, keys) -> dict:
    # Factorise the group keys as an OBJECT index: a plain np.asarray() on this
    # list would allocate a 7 GiB fixed-width unicode array (see main()).
    groups = pd.factorize(pd.Index(keys, dtype="object"), sort=False)[0]
    sizes = pd.Series(groups).value_counts()
    n_dup_groups = int((sizes > 1).sum())
    rows_in_dups = int(sizes[sizes > 1].sum())
    log(f"groups: n={len(sizes)} duplicate_groups={n_dup_groups} "
        f"rows_in_duplicate_groups={rows_in_dups} ({rows_in_dups / len(y):.2%}) "
        f"max_group_size={int(sizes.max())}")

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_ids = np.full(len(y), -1, dtype=int)
    for k, (_tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold_ids[va] = k
    if (fold_ids < 0).any():
        raise SystemExit("STEP2 FAIL: some rows received no fold id")
    gf = pd.DataFrame({"g": groups, "f": fold_ids})
    crossing = int((gf.groupby("g")["f"].nunique() > 1).sum())
    log(f"STEP2 group purity: groups spanning >1 fold = {crossing} "
        f"({'PASS' if crossing == 0 else 'FAIL'})")
    if crossing:
        raise SystemExit(f"STEP2 FAIL: {crossing} text groups span multiple folds")

    oof_text = np.zeros((len(y), 3), dtype=float)
    oof_id = np.zeros((len(y), 3), dtype=float)
    oof_override = np.zeros(len(y), dtype=np.int64)
    fold_rows = []
    for k in range(N_SPLITS):
        va = np.flatnonzero(fold_ids == k)
        tr = np.flatnonzero(fold_ids != k)
        row, p_text, p_id, ov_pred = _one_fold(clean_texts, y, ids, k, va, tr)
        oof_text[va] = p_text
        oof_id[va] = p_id
        oof_override[va] = ov_pred
        fold_rows.append(row)
    return {"fold_ids": fold_ids, "fold_rows": fold_rows, "oof_text": oof_text,
            "oof_id": oof_id, "oof_override": oof_override,
            "groups_crossing_folds": crossing, "n_groups": int(len(sizes)),
            "n_duplicate_groups": n_dup_groups, "rows_in_duplicate_groups": rows_in_dups}


def _one_fold(clean_texts, y, ids, k: int, va, tr):
    """Train + evaluate a single grouped-CV fold; returns its metrics row."""
    va_ids, tr_ids = ids[va], ids[tr]

    # --- Step 3, first half: leak assertions -------------------------------
    overlap = int(np.intersect1d(va_ids, tr_ids).size)
    if overlap:
        raise SystemExit(f"STEP3 FAIL: fold {k}: {overlap} validation ids appear in the "
                         f"prior pool")
    log(f"STEP3 fold {k}: val ids ∩ prior pool = {overlap} "
        f"(pool={len(tr_ids)}, val={len(va_ids)}) -> PASS")

    # --- text model (fold-train only); pass OBJECT arrays, never str lists ---
    word, char, clf, meta = prod.fit_text_model(clean_texts[tr], y[tr], tag=f"rv_fold{k}")
    p_text = np.asarray(clf.predict_proba(
        prod.build_text_matrix(word, char, clean_texts[va])), dtype=float)

    # --- id prior: production vs independent re-implementation -------------
    p_id, idiag = prod.id_prior_probs(va_ids, tr_ids, y[tr], k=K, tag=f"rv_fold{k}")
    p_ind = independent_knn_id_prior(va_ids, tr_ids, y[tr], k=K)
    diff = float(np.abs(p_id - p_ind).max())
    log(f"STEP3 fold {k}: production vs independent k-20 prior max_abs_diff="
        f"{diff:.3g} ({'PASS' if diff < DIFF_TOL else 'FAIL'}, tol={DIFF_TOL:g}); "
        f"mean_neighbours={idiag['mean_neighbours_used']}")
    if diff >= DIFF_TOL:
        raise SystemExit(f"STEP3 FAIL: fold {k} prior diff {diff:.3g} >= {DIFF_TOL:g}")

    # --- metrics -----------------------------------------------------------
    p_blend = prod.blend_probs(p_text, p_id, ALPHA_PREVIOUS)
    m_text = prod.metric_block(y[va], p_text)
    m_id = prod.metric_block(y[va], p_id)
    m_bl = prod.metric_block(y[va], p_blend)

    # --- Step 4: override lookup from THIS fold's train split only ---------
    lookup, ostats = prod.unanimous_lookup(clean_texts[tr], y[tr])
    # Direct proof that the lookup is train-only: every key must come from a
    # training row of this fold (validation/test texts are never inserted).
    train_text_set = set(clean_texts[tr].tolist())
    if not set(lookup).issubset(train_text_set):
        raise SystemExit(f"STEP4 FAIL: fold {k} override lookup contains a text that "
                         f"is not in this fold's train split")
    log(f"STEP4 fold {k}: lookup ⊆ fold train texts "
        f"({len(lookup):,} keys, all present in the {len(tr):,} train rows) -> PASS; "
        f"conflicting groups skipped={ostats['n_conflicting_groups_skipped']}")
    ov_pred, ov_hits = prod.apply_override(clean_texts[va], p_blend.argmax(axis=1), lookup)
    m_ov = float(f1_score(y[va], ov_pred, average="macro", labels=list(CLASSES),
                          zero_division=0))
    log(f"STEP4 fold {k}: override lookup built from {len(tr):,} train rows only -> "
        f"{ov_hits} val rows overridden; macro F1 {m_bl['macro_f1']:.6f} -> {m_ov:.6f} "
        f"({m_ov - m_bl['macro_f1']:+.6f})")
    log(f"STEP2 fold {k}: text={m_text['macro_f1']:.6f} id={m_id['macro_f1']:.6f} "
        f"blend@0.40={m_bl['macro_f1']:.6f} | acc text={m_text['accuracy']:.6f} "
        f"blend={m_bl['accuracy']:.6f} | free_ram={prod.ram_free_gb()}GB")

    row = {
        "fold": k, "n_train": int(len(tr)), "n_val": int(len(va)),
        "train_val_id_overlap": overlap,
        "prior_diff_vs_independent": diff,
        "macro_f1_text": round(m_text["macro_f1"], 6),
        "macro_f1_id": round(m_id["macro_f1"], 6),
        "macro_f1_blend_0.40": round(m_bl["macro_f1"], 6),
        "macro_f1_blend_0.40_override": round(m_ov, 6),
        "accuracy_text": round(m_text["accuracy"], 6),
        "accuracy_id": round(m_id["accuracy"], 6),
        "accuracy_blend_0.40": round(m_bl["accuracy"], 6),
        "override_rows": int(ov_hits),
        "override_lookup_texts": int(ostats["n_unanimous_groups"]),
        "override_conflicting_groups_skipped": int(ostats["n_conflicting_groups_skipped"]),
        "text_fit_seconds": meta["fit_seconds"],
    }
    return row, p_text, p_id, ov_pred


# --------------------------------------------------------------------------- #
# STEP 2 (pooled) -- OOF metrics + fresh alpha grid
# --------------------------------------------------------------------------- #
def pooled_and_grid(y, oof_text, oof_id, oof_override) -> dict:
    eye = np.eye(len(CLASSES))
    pooled = {
        "text_only": prod.metric_block(y, oof_text),
        "id_prior_only": prod.metric_block(y, oof_id),
        "blend_0.40": prod.metric_block(y, prod.blend_probs(oof_text, oof_id,
                                                            ALPHA_PREVIOUS)),
        "blend_0.40_override": prod.metric_block(y, eye[oof_override]),
    }
    grid_rows = []
    for a in ALPHA_GRID:
        m = prod.metric_block(y, prod.blend_probs(oof_text, oof_id, a))
        grid_rows.append({"alpha_text_weight": a, "alpha_id_weight": round(1 - a, 2),
                          **{k: round(v, 6) for k, v in m.items()}})
    grid = pd.DataFrame(grid_rows)
    best_a = float(grid.loc[grid["macro_f1"].idxmax(), "alpha_text_weight"])
    best_f1 = float(grid["macro_f1"].max())
    at_040 = float(grid.loc[grid["alpha_text_weight"] == ALPHA_PREVIOUS,
                            "macro_f1"].iloc[0])
    confirmed = bool(abs(best_a - ALPHA_PREVIOUS) < 1e-12)
    log(f"STEP2 pooled OOF (grouped): text={pooled['text_only']['macro_f1']:.6f} "
        f"id={pooled['id_prior_only']['macro_f1']:.6f} "
        f"blend@0.40={pooled['blend_0.40']['macro_f1']:.6f} "
        f"blend@0.40+override={pooled['blend_0.40_override']['macro_f1']:.6f}")
    log(f"STEP2 alpha grid argmax = {best_a:.2f} (macro F1 {best_f1:.6f}); "
        f"alpha 0.40 -> {at_040:.6f}; 0.40 still argmax = {confirmed}")
    delta = float(pooled["blend_0.40_override"]["macro_f1"]
                  - pooled["blend_0.40"]["macro_f1"])
    base040 = prod.blend_probs(oof_text, oof_id, ALPHA_PREVIOUS).argmax(axis=1)
    changed = int((oof_override != base040).sum())
    log(f"STEP4 pooled OOF override: {changed:,} of {len(y):,} OOF rows overridden; "
        f"macro F1 {pooled['blend_0.40']['macro_f1']:.6f} -> "
        f"{pooled['blend_0.40_override']['macro_f1']:.6f} ({delta:+.6f})")
    return {"pooled": pooled, "grid": grid, "best_alpha": best_a, "best_f1": best_f1,
            "alpha_0.40_macro_f1": at_040, "alpha_040_confirmed_as_argmax": confirmed,
            "override_rows_oof": changed, "override_delta": round(delta, 6)}


# --------------------------------------------------------------------------- #
# STEP 5 -- bias check on the predicted test-class distribution
# --------------------------------------------------------------------------- #
def step5_bias(clean_texts, test_clean, y, train_ids, test_ids, alpha) -> dict:
    log("STEP5: full refit on ALL training rows, then test predictions for the bias check")
    word, char, clf, meta = prod.fit_text_model(clean_texts, y, tag="rv_full")
    p_text = np.asarray(clf.predict_proba(prod.build_text_matrix(word, char, test_clean)),
                        dtype=float)
    p_id, idiag = prod.id_prior_probs(test_ids, train_ids, y, k=K, tag="rv_full_test")
    overlap = int(np.intersect1d(test_ids, train_ids).size)
    if overlap:
        raise SystemExit(f"STEP5 FAIL: {overlap} test ids appear in the id-prior pool")
    log(f"STEP5 id-prior: test ids ∩ prior pool = {overlap} "
        f"(pool={len(train_ids)}, test={len(test_ids)}) -> PASS")

    p = prod.blend_probs(p_text, p_id, alpha)
    pre = p.argmax(axis=1)
    lookup, stats = prod.unanimous_lookup(clean_texts, y)
    post, n_hits = prod.apply_override(test_clean, pre, lookup)

    def shares(v):
        return {int(c): round(100.0 * float((np.asarray(v) == c).mean()), 2)
                for c in CLASSES}

    train_share = shares(y)
    pre_share, post_share = shares(pre), shares(post)
    pre_drift = {int(c): round(pre_share[c] - train_share[c], 2) for c in CLASSES}
    post_drift = {int(c): round(post_share[c] - train_share[c], 2) for c in CLASSES}
    flags_pre = [int(c) for c in CLASSES if abs(pre_drift[c]) > BIAS_THRESHOLD_PP]
    flags_post = [int(c) for c in CLASSES if abs(post_drift[c]) > BIAS_THRESHOLD_PP]

    log(f"STEP5 bias (pre-override):  pred%={pre_share} train%={train_share} "
        f"drift_pp={pre_drift} flagged={flags_pre}")
    log(f"STEP5 bias (post-override): pred%={post_share} train%={train_share} "
        f"drift_pp={post_drift} flagged={flags_post} "
        f"-> {'FAIL' if flags_post else 'PASS'} (threshold {BIAS_THRESHOLD_PP:.0f}pp)")
    log(f"STEP5 test override: {n_hits:,}/{len(test_ids):,} rows matched a unanimous "
        f"training text ({100.0 * n_hits / len(test_ids):.3f}%); "
        f"{int((post != pre).sum()):,} labels changed")
    return {"threshold_pp": BIAS_THRESHOLD_PP, "alpha_used": alpha,
            "train_share_pct": train_share, "test_share_pre_override_pct": pre_share,
            "test_share_post_override_pct": post_share,
            "drift_pre_override_pp": pre_drift, "drift_post_override_pp": post_drift,
            "flagged_pre_override": flags_pre, "flagged_post_override": flags_post,
            "passed": not flags_post,
            "test_rows_overridden": int(n_hits),
            "test_labels_changed": int((post != pre).sum()),
            "lookup_stats": stats, "prior_diag": idiag,
            "mean_max_prob": round(float(p.max(axis=1).mean()), 6),
            "text_full_fit_seconds": meta["fit_seconds"]}


# --------------------------------------------------------------------------- #
# Status gate + main
# --------------------------------------------------------------------------- #
def write_status(steps: dict, all_pass: bool, exit_code: int) -> None:
    prod.write_json({"script": "src/revalidate_grouped_cv.py",
                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "python": sys.version.split()[0], "pid": os.getpid(),
                     "steps": steps, "all_pass": bool(all_pass),
                     "exit_code": int(exit_code)}, STATUS_OUT)
    log(f"wrote {STATUS_OUT.relative_to(ROOT)} all_pass={all_pass} exit_code={exit_code}")


def main() -> int:
    t0 = time.time()
    for d in (prod.REPORTS_DIR, prod.OOF_DIR, prod.DATA_CHECKS_DIR, prod.LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log("=" * 78)
    log("PART 1 INDEPENDENT RE-VALIDATION -- fresh process, everything recomputed")
    log(f"python={sys.version.split()[0]} numpy={np.__version__} pid={os.getpid()}")

    # ---------------- STEP 1: checksums (abort on failure) -----------------
    ok1, detail1 = step1_checksums()
    if not ok1:
        write_status({"1_checksums": {"status": "FAIL", "detail": detail1}}, False, 2)
        log("ABORT: protected inputs differ from recorded checksums -> Part 2 FORBIDDEN")
        return 2
    steps = {"1_checksums": {"status": "PASS", "detail": detail1}}

    # ---------------- fresh load -------------------------------------------
    # NOTE: keep texts as a pandas object column -> numpy OBJECT array. Converting
    # a python list of strings with np.asarray would allocate a giant fixed-width
    # <U<len> array (~7 GiB here) and die. Object dtype stores each string alone.
    train, test, sample = prod.load_data()
    y = train["y"].to_numpy(dtype=int)
    ids = train["id"].to_numpy(np.int64)
    test_ids = test["id"].to_numpy(np.int64)
    clean_texts = train["clean"].to_numpy()
    test_clean = test["clean"].to_numpy()
    keys = pd.Index([str(t).lower().strip() for t in clean_texts], dtype="object")

    # ---------------- STEPS 2/3/4: grouped CV ------------------------------
    cv = step2_grouped_cv(train, clean_texts, y, ids, keys)
    pg = pooled_and_grid(y, cv["oof_text"], cv["oof_id"], cv["oof_override"])
    fold_df = pd.DataFrame(cv["fold_rows"])

    step2_ok = (len(cv["fold_rows"]) == N_SPLITS and cv["groups_crossing_folds"] == 0
                and pg["best_f1"] > 0)
    steps["2_grouped_cv"] = {
        "status": "PASS" if step2_ok else "FAIL",
        "strategy": "StratifiedGroupKFold", "n_splits": N_SPLITS, "shuffle": True,
        "random_state": SEED, "group_key": "clean_text(text).lower().strip()",
        "n_groups": cv["n_groups"], "duplicate_groups": cv["n_duplicate_groups"],
        "rows_in_duplicate_groups": cv["rows_in_duplicate_groups"],
        "groups_spanning_folds": cv["groups_crossing_folds"],
        "per_fold": cv["fold_rows"],
        "pooled_macro_f1": {k: round(v["macro_f1"], 6) for k, v in pg["pooled"].items()},
        "alpha_grid": pg["grid"].to_dict(orient="records"),
        "best_alpha": pg["best_alpha"], "best_macro_f1": round(pg["best_f1"], 6),
        "alpha_0.40_macro_f1": round(pg["alpha_0.40_macro_f1"], 6),
        "alpha_040_confirmed_as_argmax": pg["alpha_040_confirmed_as_argmax"],
    }

    step3_ok = (all(int(r["train_val_id_overlap"]) == 0 for r in cv["fold_rows"])
                and all(float(r["prior_diff_vs_independent"]) < DIFF_TOL
                        for r in cv["fold_rows"]))
    steps["3_id_prior_leak_free"] = {
        "status": "PASS" if step3_ok else "FAIL",
        "per_fold_val_id_overlap": [int(r["train_val_id_overlap"])
                                    for r in cv["fold_rows"]],
        "per_fold_max_abs_diff_vs_independent": [
            float(r["prior_diff_vs_independent"]) for r in cv["fold_rows"]],
        "tolerance": DIFF_TOL,
        "independent_implementation": "full distance matrix + full stable argsort "
                                      "(no insertion window, no partition)",
        "k": K, "weights": f"1/(distance+{EPS})",
        "final_test_prior_overlap_checked_in_step5": True,
    }

    steps["4_override_leak_free"] = {
        "status": "PASS",
        "lookup_source": "each fold's train split ONLY (asserted subset of that fold's "
                         "train texts inside _one_fold); test data never used",
        "per_fold_rows_overridden": [int(r["override_rows"]) for r in cv["fold_rows"]],
        "per_fold_conflicting_groups_skipped": [
            int(r["override_conflicting_groups_skipped"]) for r in cv["fold_rows"]],
        "oof_rows_overridden": int(pg["override_rows_oof"]),
        "oof_macro_f1_before": round(pg["pooled"]["blend_0.40"]["macro_f1"], 6),
        "oof_macro_f1_after": round(pg["pooled"]["blend_0.40_override"]["macro_f1"], 6),
        "oof_macro_f1_delta": pg["override_delta"],
    }

    # ---------------- STEP 5: bias check -----------------------------------
    chosen_alpha = pg["best_alpha"]
    bias = step5_bias(clean_texts, test_clean, y, ids, test_ids, chosen_alpha)
    steps["5_bias"] = {"status": "PASS" if bias["passed"] else "FAIL", **bias}

    all_pass = all(s["status"] == "PASS" for s in steps.values())
    return _finish(cv, pg, fold_df, steps, all_pass, bias, chosen_alpha, train, test,
                   sample, y, ids, t0)


def _finish(cv, pg, fold_df, steps, all_pass, bias, chosen_alpha, train, test, sample,
            y, ids, t0) -> int:
    """Write artifacts + status, print the PASS/FAIL summary, return the exit code."""
    prod.atomic_write_csv(fold_df, FOLD_SCORES_OUT)
    log(f"wrote {FOLD_SCORES_OUT.relative_to(ROOT)} rows={len(fold_df)}")

    oof = pd.DataFrame({
        "id": train["id"].to_numpy(), "y_true": y,
        **{c: v for c, v in zip(prod.PROB_COLS,
                                prod.blend_probs(cv["oof_text"], cv["oof_id"],
                                                 chosen_alpha).T)},
        "y_pred": prod.blend_probs(cv["oof_text"], cv["oof_id"],
                                   chosen_alpha).argmax(axis=1),
        "fold": cv["fold_ids"],
        "model_name": f"blend_text_id_a{chosen_alpha:.2f}_groupedcv",
    })[prod.OOF_COLUMNS]
    prod.atomic_write_csv(oof, OOF_OUT)
    log(f"wrote {OOF_OUT.relative_to(ROOT)} rows={len(oof)}")

    metrics = {
        "experiment": "revalidate_grouped_cv",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0], "numpy": np.__version__, "pid": os.getpid(),
        "recorded_checksums": steps["1_checksums"]["detail"],
        "validation": {"strategy": "StratifiedGroupKFold", "n_splits": N_SPLITS,
                       "shuffle": True, "random_state": SEED,
                       "group_key": "clean_text(text).lower().strip()"},
        "fold_scores": cv["fold_rows"],
        "pooled_oof_macro_f1": {k: round(v["macro_f1"], 6) for k, v in pg["pooled"].items()},
        "pooled_oof_accuracy": {k: round(v["accuracy"], 6) for k, v in pg["pooled"].items()},
        "pooled_oof_per_class": {k: {f"f1_class{c}": round(v[f"f1_class{c}"], 6)
                                     for c in CLASSES}
                                 for k, v in pg["pooled"].items()},
        "alpha_grid": pg["grid"].to_dict(orient="records"),
        "chosen_alpha": chosen_alpha, "chosen_macro_f1": round(pg["best_f1"], 6),
        "alpha_0.40_macro_f1": round(pg["alpha_0.40_macro_f1"], 6),
        "alpha_040_confirmed_as_argmax": pg["alpha_040_confirmed_as_argmax"],
        "bias_check": bias,
        "steps": steps, "all_pass": bool(all_pass),
        "runtime_seconds": round(time.time() - t0, 2),
        "outputs": {"fold_scores": str(FOLD_SCORES_OUT.relative_to(ROOT)),
                    "oof": str(OOF_OUT.relative_to(ROOT)),
                    "metrics": str(METRICS_OUT.relative_to(ROOT)),
                    "status": str(STATUS_OUT.relative_to(ROOT))},
    }
    prod.write_json(metrics, METRICS_OUT)
    log(f"wrote {METRICS_OUT.relative_to(ROOT)}")
    write_status(steps, all_pass, 0 if all_pass else 2)

    # ---------------- PASS/FAIL summary -------------------------------------
    lines = ["", "=" * 76,
             "PART 1 -- FINAL PASS/FAIL SUMMARY (all values recomputed this run)",
             "=" * 76]
    for name, s in steps.items():
        lines.append(f"  {name}: {s['status']}")
    lines += [
        f"  1 checksums   : " + "; ".join(
            f"{k}={'OK' if v['match'] else 'DIFF'}" for k, v in
            steps["1_checksums"]["detail"].items()),
        f"  2 grouped CV  : folds=5 groups={steps['2_grouped_cv']['n_groups']} "
        f"dup_groups={steps['2_grouped_cv']['duplicate_groups']} "
        f"crossing={steps['2_grouped_cv']['groups_spanning_folds']} | pooled OOF "
        f"text={pg['pooled']['text_only']['macro_f1']:.6f} "
        f"id={pg['pooled']['id_prior_only']['macro_f1']:.6f} "
        f"blend@0.40={pg['pooled']['blend_0.40']['macro_f1']:.6f} "
        f"blend+override={pg['pooled']['blend_0.40_override']['macro_f1']:.6f}",
        f"      alpha grid argmax={pg['best_alpha']:.2f} "
        f"(macro F1 {pg['best_f1']:.6f}); 0.40 confirmed as argmax="
        f"{pg['alpha_040_confirmed_as_argmax']}",
        f"  3 id-prior    : val∩pool={steps['3_id_prior_leak_free']['per_fold_val_id_overlap']} "
        f"max_diffs={steps['3_id_prior_leak_free']['per_fold_max_abs_diff_vs_independent']} "
        f"(tol={DIFF_TOL:g})",
        f"  4 override    : OOF rows overridden="
        f"{steps['4_override_leak_free']['oof_rows_overridden']:,} "
        f"macro F1 {steps['4_override_leak_free']['oof_macro_f1_before']:.6f} -> "
        f"{steps['4_override_leak_free']['oof_macro_f1_after']:.6f} "
        f"(delta {steps['4_override_leak_free']['oof_macro_f1_delta']:+.6f}); "
        f"lookup=train-split-only",
        f"  5 bias        : train%={bias['train_share_pct']} "
        f"test%={bias['test_share_post_override_pct']} "
        f"drift_pp={bias['drift_post_override_pp']} "
        f"flagged={bias['flagged_post_override']} (>{BIAS_THRESHOLD_PP:.0f}pp)",
        f"  RESULT: {'ALL PASS -> Part 2 may proceed' if all_pass else 'FAILURE -> Part 2 FORBIDDEN'}",
        f"  chosen_alpha={chosen_alpha:.2f} | grouped-OOF macro F1="
        f"{pg['best_f1']:.6f} | runtime={metrics['runtime_seconds']}s",
        "=" * 76,
    ]
    for line in lines:
        log(line)
    return 0 if all_pass else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        log("PART 1 FAILED -- traceback follows")
        log(traceback.format_exc())
        write_status({"harness": {"status": "FAIL",
                                  "error": traceback.format_exc()[-800:]}}, False, 2)
        raise







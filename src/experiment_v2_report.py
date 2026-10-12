"""
src/experiment_v2_report.py
===========================

Rebuilds the single running comparison table `reports/v2_comparison_table.csv`
and the per-experiment config registry `reports/v2_configs/*.json` from the
artifacts already on disk:

* `reports/v2_*_metrics.json`      -- every experiment's full result payload
* `predictions/v2/*_oof.npz`       -- every experiment's OOF probability matrices
* `cache/v2/strict_base_probs.npz` -- EXP 4's double-nested base probabilities

Rows are keyed on `experiment`, so re-running this script is idempotent: nothing
is duplicated and the full history of what was tried (wins AND losses) is
preserved. Every number is recomputed here from the cached OOF matrices, so the
table is consistent by construction rather than hand-copied.

Every row is scored on the SAME protocol: pooled leak-free OOF Macro F1 over
StratifiedGroupKFold(5, shuffle=True, random_state=42), group =
clean_text.lower().strip(), compared against the locked baseline 0.639579.
Rows whose weights/alpha were chosen by looking at the same pooled OOF are
labelled `optimistic` -- their honest cross-fitted counterpart is a separate row.

This script never writes a submission and never touches the locked baseline
notebook / submission_1.csv (checked via sha1 at the end).

Usage:  python src/experiment_v2_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_v2_harness as H

METRICS = {
    "v2_0_baseline_anchor": "baseline anchor: LR(C=4,balanced) text + k=20 ID prior, "
                            "blend alpha=0.40, grouped leak-free OOF (reproduces "
                            "0.639579 exactly)",
    "v2_1_text_ensemble": "text model replaced by equal-weight prob average of "
                          "LR + CalibratedClassifierCV(LinearSVC) + ComplementNB(word)",
    "v2_2_hand_features": "9 hand-crafted dense features (StandardScaler'd, fitted on "
                          "train rows only) hstacked onto word+char TF-IDF, LR refitted",
    "v2_3_per_class_alpha": "per-class 3-vector blend weights instead of a single alpha "
                            "(Nelder-Mead + local grid)",
    "v2_4_meta_stacker": "OOF meta-stacker: LR on [text_probs, id_probs, hand features], "
                         "cross-fitted with the same grouped 5-fold split",
    "v2_5_combined": "combination of the surviving candidates, cross-fitted",
}

PROTOCOL = {"validation": "StratifiedGroupKFold(5, shuffle=True, random_state=42)",
            "group_key": "clean_text.lower().strip()",
            "metric": "pooled leak-free OOF Macro F1",
            "baseline_oof_macro_f1": H.BASELINE_OOF_F1}


def pooled(y, proba):
    return H.macro_f1(y, proba)


def fold_scores(y, proba, fold_ids):
    vals = [H.macro_f1(y[va], proba[va]) for _, va in
            (H.fold_split(fold_ids, k) for k in range(H.N_SPLITS))]
    return vals, float(np.std(vals))


def add(experiment, description, y, proba, fold_ids, config, note=""):
    vals, std = fold_scores(y, proba, fold_ids)
    f1 = pooled(y, proba)
    full = dict(config)
    full.update(PROTOCOL)
    if note:
        full["evaluation_note"] = note
    full["per_fold_macro_f1"] = [round(v, 6) for v in vals]
    H.record_comparison(experiment, description, f1,
                        H.per_class_f1(y, proba.argmax(1)), std, full)
    return f1, std, vals


def main() -> int:
    H.banner("v2 comparison table rebuild (from cached OOF matrices + metrics JSON)")
    H.check_protected()

    d = H.load_data()
    y = d["y"]

    P = H.PRED_DIR
    need = ["v2_0_baseline_anchor_oof.npz", "v2_1_text_ensemble_oof.npz",
            "v2_2_hand_features_oof.npz", "v2_4_meta_stacker",
            "strict_base_probs.npz"]
    missing = [n for n in need[:-1]
               if not (P / (n if n.endswith(".npz") else f"{n}_oof.npz")).exists()]
    if missing:
        print("WARNING missing OOF artifacts:", missing)

    a0 = np.load(P / "v2_0_baseline_anchor_oof.npz")
    oof_text, oof_id, fold_ids = a0["oof_text"], a0["oof_id"], a0["fold_ids"]

    m0 = json.loads((H.REPORTS_DIR / "v2_0_baseline_anchor_metrics.json").read_text())
    m1 = json.loads((H.REPORTS_DIR / "v2_1_text_ensemble_metrics.json").read_text())
    m2 = json.loads((H.REPORTS_DIR / "v2_2_hand_features_metrics.json").read_text())
    m3 = json.loads((H.REPORTS_DIR / "v2_3_per_class_alpha_metrics.json").read_text())

    # ---------------- reference rows ----------------
    add("ref_id_prior_only", "reference: ID-locality prior alone (k=20, 1/(d+1e-6))",
        y, oof_id, fold_ids,
        {"k": H.K_NEIGHBORS, "weights": "1/(distance+1e-6)", "pool": "fold-train ids only"})
    add("ref_text_lr_only", "reference: baseline LR text model alone (no ID prior)",
        y, oof_text, fold_ids,
        {"model": "LogisticRegression", **H.LR_KW,
         "features": "word(1-2) + char_wb(3-5) TF-IDF, 60k each, min_df=2"})
    add("ref_baseline_blend_alpha0.40",
        "LOCKED BASELINE (best-known-good): LR text + k=20 ID prior, blend alpha=0.40",
        y, H.blend(oof_text, oof_id, H.ALPHA_BASELINE), fold_ids,
        {"alpha": H.ALPHA_BASELINE, "k": H.K_NEIGHBORS, "model": "LogisticRegression",
         **H.LR_KW,
         "features": "word(1-2) + char_wb(3-5) TF-IDF, 60k each, min_df=2"},
        note="reproduced exactly by v2_0 (delta 0.000000)")

    # ---------------- EXP 1 ----------------
    e1 = np.load(P / "v2_1_text_ensemble_oof.npz")
    ens = e1["oof_ens"]
    t1 = m1["text_model"]
    base_cfg_1 = {
        "text_members": t1["members"], "combination": t1["combination"],
        "linear_svc": t1["linear_svc"], "calibration": t1["calibration"],
        "id_prior": {"k": H.K_NEIGHBORS, "weights": "1/(distance+1e-6)"},
    }
    add("v2_1_ens_equal_alpha0.40", f"{METRICS['v2_1_text_ensemble']} | alpha=0.40",
        y, H.blend(ens, oof_id, 0.40), fold_ids, {**base_cfg_1, "alpha": 0.40})
    add("v2_1_ens_equal_alpha_best0.45", f"{METRICS['v2_1_text_ensemble']} | alpha=0.45 "
                                         f"(re-gridded)", y,
        H.blend(ens, oof_id, m1["chosen_alpha"]), fold_ids,
        {**base_cfg_1, "alpha": m1["chosen_alpha"]},
        note="alpha re-gridded on pooled OOF; mild optimism")
    dw = m1["diagnostic_simplex_best"]
    add("v2_1_ens_simplex_tuned_alpha0.45",
        f"{METRICS['v2_1_text_ensemble']} | member weights tuned on OOF "
        f"(LR={dw['w_lr']}, SVC={dw['w_svc']}, CNB={dw['w_nb']})",
        y, H.blend(e1["oof_ens_w"], oof_id, m1["chosen_alpha"]), fold_ids,
        {**base_cfg_1, "alpha": m1["chosen_alpha"], "member_weights":
            [dw["w_lr"], dw["w_svc"], dw["w_nb"]]},
        note="OPTIMISTIC: member weights selected on the same pooled OOF")

    # ---------------- EXP 2 ----------------
    e2 = np.load(P / "v2_2_hand_features_oof.npz")
    f2 = m2["features"]
    for s in f2["block_scales_swept"]:
        arr = e2[f"oof_scale{s}"]
        add(f"v2_2_hand_scale{s}_alpha0.40",
            f"{METRICS['v2_2_hand_features']} | dense block scale={s}, alpha=0.40",
            y, H.blend(arr, oof_id, 0.40), fold_ids,
            {"dense_block_scale": s, "alpha": 0.40, "features": f2["names"],
             "scaler": f2["scaler"], "computed_on": f2["computed_on"],
             "model": "LogisticRegression", **H.LR_KW,
             "id_prior": {"k": H.K_NEIGHBORS}},
            note=("literal prompt specification" if s == 1.0 else "scale sweep"))
    s_best = max(f2["block_scales_swept"],
                 key=lambda s: m2["pooled_oof_macro_f1"][f"blend_scale{s}_0.40"])
    a_best = m2["chosen_alpha_by_scale"][str(s_best)]
    add(f"v2_2_hand_scale{s_best}_alpha{a_best}",
        f"{METRICS['v2_2_hand_features']} | best dense scale={s_best} + re-gridded alpha",
        y, H.blend(e2[f"oof_scale{s_best}"], oof_id, a_best), fold_ids,
        {"dense_block_scale": s_best, "alpha": a_best, "features": f2["names"],
         "scaler": f2["scaler"], "computed_on": f2["computed_on"],
         "model": "LogisticRegression", **H.LR_KW,
         "id_prior": {"k": H.K_NEIGHBORS}},
        note="OPTIMISTIC: scale and alpha both chosen on the same pooled OOF")

    # ---------------- EXP 3 ----------------
    def cross_fitted_proba(p_text, p_id, w_folds):
        proba = np.zeros((len(y), 3))
        for k_str, w in w_folds.items():
            k = int(k_str)
            _, va = H.fold_split(fold_ids, k)
            proba[va] = H.blend_per_class(p_text[va], p_id[va], w)
        return proba

    r3 = m3["results"]
    for label, key in (("lr", "lr"), ("ens", "ens")):
        res = r3.get(label)
        if res is None:
            continue
        base_p = ens if label == "ens" else oof_text
        cfg3 = {"text_source": label, "id_prior": {"k": H.K_NEIGHBORS},
                "parametrisation": "P(c)=w_c*text_c+(1-w_c)*id_c, w in [0,1]^3",
                "optimiser": "scipy Nelder-Mead multi-start + local 0.01 grid"}
        head = res["per_class_tuned_headline"]
        add(f"v2_3_perclass_{label}_tuned_on_oof",
            f"{METRICS['v2_3_per_class_alpha']} | text={label} | w={np.round(head['w'],4).tolist()}",
            y, H.blend_per_class(base_p, oof_id, head["w"]), fold_ids,
            {**cfg3, "w": head["w"]},
            note="OPTIMISTIC: w tuned on the same pooled OOF that is scored")
        cf = res["per_class_cross_fitted"]
        add(f"v2_3_perclass_{label}_crossfitted",
            f"{METRICS['v2_3_per_class_alpha']} | text={label} | HONEST cross-fitted w",
            y, cross_fitted_proba(base_p, oof_id, cf["per_fold_w"]), fold_ids,
            {**cfg3, "w_per_fold": cf["per_fold_w"]},
            note="w tuned on 4 folds of OOF rows, scored on the held-out fold")

    # ---------------- EXP 4 ----------------
    m4 = json.loads((H.REPORTS_DIR / "v2_4_meta_stacker_metrics.json").read_text())
    feat_names = m4["meta_learner"]["features"]
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    raw = d["train"]["text"].to_numpy()
    Hraw = H.hand_feature_matrix(raw)
    mkw = dict(m4["meta_learner"])
    mkw.pop("features", None)
    mkw.pop("model", None)
    mkw.pop("scaler", None)
    mkw.pop("note", None)
    spec_blocks = {
        "text+id+hand": np.hstack([oof_text, oof_id, Hraw]),
        "text+id (no hand)": np.hstack([oof_text, oof_id]),
        "text+hand (no id)": np.hstack([oof_text, Hraw]),
        "id+hand (no text)": np.hstack([oof_id, Hraw]),
    }
    for res in m4["results"]:
        label = res["tag"].replace("[spec] ", "").replace(" (spec)", "")
        block = spec_blocks[label]
        pred = np.zeros(len(y), dtype=int)
        for k in range(H.N_SPLITS):
            tr, va = H.fold_split(fold_ids, k)
            sc = StandardScaler().fit(block[tr])
            clf = LogisticRegression(**mkw).fit(sc.transform(block[tr]), y[tr])
            pred[va] = clf.predict(sc.transform(block[va]))
        add(f"v2_4_meta_{label.replace(' ', '_').replace('+', '')}",
            f"{METRICS['v2_4_meta_stacker']} | {label}",
            y, np.eye(3)[pred], fold_ids,
            {"meta_learner": "LogisticRegression", **mkw,
             "features": [n for n in feat_names][: block.shape[1]],
             "base": "grouped OOF text_probs + id_probs",
             "inner_cv": "same grouped 5-fold split (asserted identical)"},
            note="meta-learner cross-fitted: trained on 4 folds, scored on the 5th")
    if m4.get("strict_nested"):
        sp_path = H.CACHE_DIR / "strict_base_probs.npz"
        if sp_path.exists():
            s = np.load(sp_path)
            pred = np.zeros(len(y), dtype=int)
            block = np.hstack([s["oof_text"], s["oof_id"], Hraw])
            for k in range(H.N_SPLITS):
                tr, va = H.fold_split(fold_ids, k)
                sc = StandardScaler().fit(block[tr])
                clf = LogisticRegression(**mkw).fit(sc.transform(block[tr]), y[tr])
                pred[va] = clf.predict(sc.transform(block[va]))
            add("v2_4_meta_strict_text_id_hand",
                f"{METRICS['v2_4_meta_stacker']} | STRICT double-nested base features",
                y, np.eye(3)[pred], fold_ids,
                {"meta_learner": "LogisticRegression", **mkw,
                 "features": feat_names,
                 "base": "double-nested: meta-train base models never saw the "
                         "meta-eval fold either"},
                note="base models here see 3/5 of the data, so this row is NOT "
                     "directly comparable to 0.639579; the in-protocol reference "
                     f"(plain blend on the same strict base) is "
                     f"{m4['strict_nested']['reference_blend_0.40_on_strict_base']}")

    # ---------------- EXP 5 (if present) ----------------
    m5_path = H.REPORTS_DIR / "v2_5_combined_metrics.json"
    if m5_path.exists():
        m5 = json.loads(m5_path.read_text())
        e5 = np.load(P / "v2_5_combined_oof.npz")
        add("v2_5_combined_best",
            f"{METRICS['v2_5_combined']} | gate={m5['gate']['any_single_experiment_beat_baseline']}",
            y, e5["oof_best"], fold_ids, m5["best_config"],
            note=m5["gate"]["note"])

    # ---------------- print ----------------
    tbl = H.read_comparison()
    cols = ["experiment", "oof_macro_f1", "delta_vs_baseline", "per_fold_std"]
    winners = tbl["delta_vs_baseline"].astype(float) > 0
    print("\n=== reports/v2_comparison_table.csv (pooled leak-free OOF Macro F1) ===")
    print(tbl[cols].to_string(index=False))
    print(f"\nbaseline = {H.BASELINE_OOF_F1:.6f} | rows with delta > 0: "
          f"{tbl.loc[winners, 'experiment'].tolist()}")
    H.check_protected()
    print(f"\ntable -> {H.COMPARISON_CSV.relative_to(H.ROOT).as_posix()}  "
          f"({len(tbl)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

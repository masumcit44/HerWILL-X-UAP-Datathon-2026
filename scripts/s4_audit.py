"""Build reports/submission_4_audit.json and reports/submission_4_final_report.md
from the EXECUTED notebooks, and evaluate gates G1-G9. No model logic here.
"""
import ast
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

PKG_NB = "uap_submission/reproducible/submission_reproducible_file_4.ipynb"
FRESH_NB = "repro_s4/fresh/submission_reproducible_file_4.ipynb"
ART = "uap_submission/reproducible/artifacts_submission_4"
CSV = "uap_submission/submission/submission_4.csv"
CFG = json.load(open(os.path.join(ART, "config_submission_4.json")))
MANIFEST = json.load(open(os.path.join(ART, "artifact_manifest_submission_4.json")))

# Submission 3 recomputed in this same environment (reference for G3/G4)
SUB3 = dict(pooled=0.682311, folds=[0.682243, 0.685245, 0.680023, 0.680668, 0.683172],
            per_class=[0.782202, 0.640594, 0.624137])


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def outputs(nb_path):
    nb = json.load(open(nb_path, encoding="utf-8"))
    txt = []
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            txt.append("".join(o.get("text", [])))
            if "data" in o and "text/plain" in o["data"]:
                txt.append("".join(o["data"]["text/plain"]))
    return "\n".join(txt)


def grab(txt, key, cast=float):
    m = re.search(r"^%s\s*:\s*(.+)$" % re.escape(key), txt, re.M)
    if not m:
        return None
    v = m.group(1).strip()
    try:
        return cast(v)
    except ValueError:
        return v


def err_count(nb_path):
    nb = json.load(open(nb_path, encoding="utf-8"))
    return sum(1 for c in nb["cells"] for o in c.get("outputs", [])
               if o.get("output_type") == "error")


pkg = outputs(PKG_NB)
fresh = outputs(FRESH_NB)
PKG_ERRS = err_count(PKG_NB)
FRESH_ERRS = err_count(FRESH_NB)

res = dict(
    pooled=grab(pkg, "reproduced pooled OOF"),
    target=CFG["reproduced"]["recorded_target"],
    per_class=json.loads(grab(pkg, "per-class F1")),
    accuracy=grab(pkg, "accuracy"),
    folds=json.loads(grab(pkg, "fold scores")),
    fold_std=grab(pkg, "fold std"),
    worst_fold=grab(pkg, "worst fold", str),
    seed_scores=ast.literal_eval(grab(pkg, "seed scores (no corr.)", str)),
    test_dist=json.loads(grab(pkg, "test class distribution")),
    csv_sha=grab(pkg, "csv sha256", str),
    csv_bytes=grab(pkg, "csv bytes", int),
    fresh_pooled=grab(fresh, "reproduced pooled OOF"),
    fresh_csv_sha=grab(fresh, "csv sha256", str),
    override=grab(pkg, "override", str),
    pre_write_ok="PRE-WRITE validation OK" in pkg,
    post_read_ok="POST-READ validation OK" in pkg,
    drift=[float(x) for x in grab(pkg, "drift (pp)").strip("[]").split(",")],
    share_before=json.loads(grab(pkg, "test share before override")),
    train_share=json.loads(grab(pkg, "train class share")),
    ablation=re.findall(r"^\s+(?:full 60 features|drop raw id \(rid\)|drop all 32 ID-neighbour)\s*:\s*([\d.]+)$", pkg, re.M),
)
res["difference"] = res["pooled"] - res["target"]

# confusion matrix + prediction distribution, parsed from the notebook output
cm_txt = re.search(r"confusion matrix[^\[]*(\[\[.*?\]\])", pkg, re.S)
res["confusion"] = ([list(map(int, re.findall(r"\d+", r)))
                     for r in re.findall(r"\[[^\[\]]*\]", cm_txt.group(1))]
                    if cm_txt else None)
pd_txt = re.search(r"prediction distribution:\s*(\[[\d, ]+\])", pkg)
res["prediction_distribution"] = json.loads(pd_txt.group(1)) if pd_txt else None
print("confusion parsed:", res["confusion"])
print("prediction distribution:", res["prediction_distribution"])


# ---- independent G2 array checks straight from the bundled artifacts --------
ART_CHECKS = {}
folds = np.load(os.path.join(ART, "fold_ids_colab.npy")).astype(np.int64)
fold_payload = hashlib.sha256(np.ascontiguousarray(folds, np.int64).tobytes()).hexdigest()
ART_CHECKS["fold_set_b_alignment"] = (fold_payload == CFG["fold_ids_int64_sha256"])
for name in ["text_tfidf_final.npz", "text_knn_final.npz", "emb_e5-base_final.npz",
             "sid.npz", "plain_id.npz", "xlmr_base_idhead_final.npz"]:
    z = np.load(os.path.join(ART, name), allow_pickle=True)
    o, t = z["oof"], z["test"]
    ART_CHECKS[name] = dict(
        shape_ok=(o.shape == (47817, 3) and t.shape == (11955, 3)),
        finite=bool(np.isfinite(o).all() and np.isfinite(t).all()),
        in_range=bool((o >= 0).all() and (o <= 1).all() and (t >= 0).all() and (t <= 1).all()),
        rowsum_max_dev=float(max(abs(o.sum(1) - 1).max(), abs(t.sum(1) - 1).max())),
        fold_ids_match=bool(np.array_equal(z["fold_ids"].astype(np.int64), folds))
        if "fold_ids" in z else None)
for name, c in ART_CHECKS.items():
    if isinstance(c, dict):
        print("%-30s shape %s finite %s range %s rowsum %.2e" % (
            name, c["shape_ok"], c["finite"], c["in_range"], c["rowsum_max_dev"]))

# ---- protected files, before vs after --------------------------------------
before = json.load(open("repro_s4/_protected_before.json"))
after = {}
for p in before:
    after[p] = sha256(p) if os.path.exists(p) else "ABSENT"
changed = [p for p in before if before[p] != after[p]]
print("protected files changed:", changed if changed else "NONE")

# ---- packaged vs fresh CSV --------------------------------------------------
fresh_csv = "repro_s4/fresh/submission_4.csv"
csv_exists = os.path.exists(CSV)
fresh_exists = os.path.exists(fresh_csv)
disk_sha = sha256(CSV) if csv_exists else None
fresh_disk_sha = sha256(fresh_csv) if fresh_exists else None
print("packaged csv sha:", disk_sha)
print("fresh    csv sha:", fresh_disk_sha)

sub3 = pd.read_csv("uap_submission/submission/submission_3.csv")
sub4 = pd.read_csv(CSV) if csv_exists else None
DISAGREE = (100.0 * float((sub3["y_pred"].to_numpy() != sub4["y_pred"].to_numpy()).mean())
            if sub4 is not None else None)
counts = np.bincount(sub4["y_pred"], minlength=3).tolist() if sub4 is not None else None

# ---- gates ------------------------------------------------------------------
g = {}
g["G1_deterministic_oof_within_0.001"] = bool(abs(res["difference"]) <= 0.001)
g["G2_array_integrity"] = bool(
    ART_CHECKS["fold_set_b_alignment"] and all(
        c["shape_ok"] and c["finite"] and c["in_range"] and c["rowsum_max_dev"] < 1e-5
        for k, c in ART_CHECKS.items() if isinstance(c, dict)))
n_regress = sum(1 for a, b in zip(res["folds"], SUB3["folds"]) if a >= b - 0.004)
g["G3_at_least_4_folds_no_material_regression"] = bool(n_regress >= 4)
worst_cls = min(a - b for a, b in zip(res["per_class"], SUB3["per_class"]))
g["G4_no_class_f1_regress_over_0.005"] = bool(worst_cls >= -0.005)
g["G5_test_shares_within_6pp"] = bool(all(v <= 6.0 for v in res["drift"]))
g["G6_csv_pre_and_post_validation"] = bool(res["pre_write_ok"] and res["post_read_ok"] and csv_exists)
g["G7_fresh_notebook_execution"] = bool(FRESH_ERRS == 0 and res["fresh_pooled"] is not None
                                         and abs(res["fresh_pooled"] - 0.703904) <= 0.001)
g["G8_fresh_and_packaged_csv_identical"] = bool(
    fresh_exists and csv_exists and fresh_disk_sha == disk_sha)
g["G9_protected_files_unchanged"] = bool(not changed)

audit = dict(
    submission=4,
    validation_classification=CFG["validation_classification"],
    authoritative_oof=res["pooled"], recorded_target=res["target"],
    score_difference=res["difference"],
    per_class_f1=res["per_class"], accuracy=res["accuracy"],
    confusion=res["confusion"], folds=res["folds"], fold_std=res["fold_std"],
    worst_fold=res["worst_fold"], prediction_distribution=res["prediction_distribution"],
    seed_scores=res["seed_scores"], fold_offsets=CFG["class_correction"]["fold_offsets"],
    final_offset=CFG["class_correction"]["final_offset"],
    test_class_distribution=res["test_dist"], test_share_before_override=res["share_before"],
    train_class_share=res["train_share"], drift_pp=res["drift"],
    class_counts=counts, disagreement_vs_submission_3_pct=DISAGREE,
    csv_path=CSV, csv_sha256=disk_sha, csv_bytes=res["csv_bytes"],
    notebook_path=PKG_NB, notebook_sha256=sha256(PKG_NB),
    fresh_csv_sha256=fresh_disk_sha, fresh_oof=res["fresh_pooled"],
    fresh_cell_errors=FRESH_ERRS, package_cell_errors=PKG_ERRS,
    override=res["override"], ablations_id_features=res["ablation"],
    package_versions=json.load(open(os.path.join(ART, "package_versions.json"))),
    artifact_manifest_path=os.path.join(ART, "artifact_manifest_submission_4.json"),
    artifact_manifest=MANIFEST,
    submission_3_reference=SUB3,
    folds_not_regressing=n_regress, worst_class_delta=worst_cls,
    protected_files_before=before, protected_files_after=after,
    protected_files_changed=changed,
    array_checks=ART_CHECKS, gates=g, all_gates_pass=all(g.values()),
)
json.dump(audit, open("reports/submission_4_audit.json", "w"), indent=2)
print("gates:", json.dumps(g, indent=2))
print("ALL GATES PASS:", all(g.values()))

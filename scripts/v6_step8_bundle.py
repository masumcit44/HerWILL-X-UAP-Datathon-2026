"""v6 Section 8a: extract bundle artifacts next to the notebook + manifest."""
import os
import json
import zipfile
import hashlib

ZIP = "uap_complete_export_20260928_064140.zip"
NB_DIR = "uap_submission/reproducible"
BUNDLE = NB_DIR + "/artifacts_submission_3"
os.makedirs(BUNDLE, exist_ok=True)


def sha(b):
    return hashlib.sha256(b).hexdigest()


zf = zipfile.ZipFile(ZIP)
manifest = {}
fb = zf.read("final_sources/fold_ids.npy")
with open(BUNDLE + "/fold_ids_colab.npy", "wb") as f:
    f.write(fb)
manifest["fold_ids_colab.npy"] = {
    "sha256": sha(fb), "bytes": len(fb),
    "int64_bytes_sha256": sha(fb[128:]),
    "source_in_zip": "final_sources/fold_ids.npy",
    "note": "Colab fold ids. FILE sha256 covers the .npy container; "
            "int64_bytes_sha256 covers the raw int64 array payload."}

names = ["final_sources/emb_e5-base_final.npz",
         "final_sources/xlmr_base_idhead_final.npz"] + \
        ["per_fold_sources/xlmr_base_idhead_fold{}.npz".format(k) for k in range(5)]
for n in names:
    b = zf.read(n)
    out = os.path.basename(n)
    with open(BUNDLE + "/" + out, "wb") as f:
        f.write(b)
    manifest[out] = {"sha256": sha(b), "bytes": len(b), "source_in_zip": n}
zf.close()

cfg = json.load(open("reports/v6/final.json"))
blend = json.load(open("reports/v6/blend.json"))
s123 = json.load(open("reports/v6/step123.json"))
config = {
    "seed": 42,
    "n_splits": 5,
    "classes": [0, 1, 2],
    "metric": "macro_f1",
    "use_dup_override": True,
    "source_order": cfg["source_order"],
    "source_meaning": {
        "text": "TF-IDF word(1,2) + char_wb(3,5) -> LogisticRegression (CPU)",
        "sid": "similarity-weighted id prior, KW=120 k=180 beta=10 gamma=1.0",
        "e5_base": "intfloat/e5-base-v2 zero-shot logits (verified export, GPU-trained)",
        "xlmr_idhead": "xlm-roberta-base + id-head classifier (verified export, GPU-trained)",
    },
    "final_weights": cfg["final_weights"],
    "honest_blend_oof_macro_f1": blend["sid_variant"]["honest_oof"],
    "honest_blend_per_class_f1": blend["sid_variant"]["per_class"],
    "honest_blend_fold_scores": blend["sid_variant"]["fold_scores"],
    "plain_id_blend_oof_macro_f1": blend["plain_id_variant"]["honest_oof"],
    "sid_id_only_oof_macro_f1": s123["sid"]["A"]["id_only_oof"],
    "text_only_oof_macro_f1": s123["text_oof"],
    "plain_id_only_oof_macro_f1": s123["plain_id_oof"],
    "golden_test_max_abs_diff": s123["golden_max_diff"],
    "simplex_step": 0.05,
    "override_rows_matched": cfg["override_rows_matched"],
    "override_labels_changed": cfg["override_labels_changed"],
    "override_conflicting_groups": cfg["override_conflicting_groups"],
    "test_class_share": cfg["test_class_share"],
    "train_class_share": cfg["train_class_share"],
    "run_gpu_retrain": False,
    "mode": "A",
}
with open(BUNDLE + "/config_submission_3.json", "w") as f:
    json.dump(config, f, indent=2)
manifest["config_submission_3.json"] = {
    "sha256": sha(open(BUNDLE + "/config_submission_3.json", "rb").read()),
    "bytes": os.path.getsize(BUNDLE + "/config_submission_3.json"),
    "source_in_zip": "generated"}
with open(BUNDLE + "/manifest_submission_3.json", "w") as f:
    json.dump(manifest, f, indent=2)
for k in sorted(manifest):
    print("  {:<44} {}".format(k, manifest[k]["sha256"][:16]))
print("-> bundle written to", BUNDLE)


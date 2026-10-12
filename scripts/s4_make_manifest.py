"""Write artifact_manifest_submission_4.json for every bundled artifact."""
import hashlib
import json
import os

import numpy as np

ART = "uap_submission/reproducible/artifacts_submission_4"
cfg = json.load(open(os.path.join(ART, "config_submission_4.json")))
src = json.load(open("repro_s4/_srcinfo.json"))["sources"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


PURPOSE = {
    "fold_ids_colab.npy": "fold assignment (fold set B); int64 payload is the fold identity",
    "config_submission_4.json": "frozen model configuration, validation class, class correction, seed policy",
    "feature_schema.json": "the 60 feature columns, grouped by type, in exact build order",
    "package_versions.json": "package version record for the reproducing environment",
    "text_tfidf_final.npz": src["text_tfidf_final.npz"]["purpose"],
    "text_knn_final.npz": src["text_knn_final.npz"]["purpose"],
    "emb_e5-base_final.npz": src["emb_e5-base_final.npz"]["purpose"],
    "sid.npz": src["sid.npz"]["purpose"],
    "plain_id.npz": src["plain_id.npz"]["purpose"],
    "xlmr_base_idhead_final.npz": src["xlmr_base_idhead_final.npz"]["purpose"],
}

man = {}
for name in sorted(os.listdir(ART)):
    p = os.path.join(ART, name)
    if not os.path.isfile(p) or name.endswith("manifest_submission_4.json"):
        continue
    e = dict(path="artifacts_submission_4/" + name, size_bytes=os.path.getsize(p),
             sha256=sha256(p), purpose=PURPOSE[name], fold_set="B")
    if name in src:
        e.update(oof_shape=src[name]["oof_shape"], test_shape=src[name]["test_shape"],
                 standalone_oof_macro_f1=src[name]["standalone_oof_macro_f1"],
                 derivation=src[name]["derivation"], uses_id=src[name]["uses_id"])
    else:
        e.update(derivation="n/a (configuration/fold metadata)", uses_id=False)
    if name == "fold_ids_colab.npy":
        f = np.load(p).astype(np.int64)
        e["int64_bytes_sha256"] = hashlib.sha256(
            np.ascontiguousarray(f, np.int64).tobytes()).hexdigest()
        e["fold_sizes"] = np.bincount(f, minlength=5).tolist()
        e["uses_id"] = True
    man[name] = e

json.dump(man, open(os.path.join(ART, "artifact_manifest_submission_4.json"), "w"), indent=2)
print("wrote artifact_manifest_submission_4.json with %d entries" % len(man))
for k, v in man.items():
    print("  %-32s %9d B  %s  uses_id=%s" % (k, v["size_bytes"], v["sha256"][:16], v["uses_id"]))
assert cfg["validation_classification"] == "STANDARD_OOF_STACKING_WITH_DEPENDENCY"

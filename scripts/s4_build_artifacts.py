"""Build artifacts_submission_4: copy the frozen base probability sources and the
fold ids, then write the frozen config / feature schema / package version record.
Reads protected inputs only; writes solely under the artifact directory.
"""
import hashlib
import json
import os
import platform
import shutil

import numpy as np
import pandas as pd

ART = "uap_submission/reproducible/artifacts_submission_4"
os.makedirs(ART, exist_ok=True)

SOURCES = [
    ("final_sources/fold_ids_colab.npy", "fold_ids_colab.npy", "fold assignment (fold set B)", None),
    ("cache/v7_text/text_tfidf_final.npz", "text_tfidf_final.npz",
     "TF-IDF word(1,2)+char_wb(3,5) -> LogisticRegression, CPU", "text-derived"),
    ("cache/v7_text/text_knn_final.npz", "text_knn_final.npz",
     "TF-IDF cosine kNN text neighbour, CPU", "text-derived"),
    ("final_sources/emb_e5-base_final.npz", "emb_e5-base_final.npz",
     "intfloat/multilingual-e5-small zero-shot logits (verified export)", "text-derived"),
    ("cache/v7_text/sid.npz", "sid.npz",
     "similarity-weighted ID prior, config A (KW=120 k=180 beta=10 gamma=1)", "ID-derived"),
    ("cache/v7_text/plain_id.npz", "plain_id.npz",
     "plain ID prior, k=20 inverse distance eps=1e-6", "ID-derived"),
    ("final_sources/xlmr_base_idhead_final.npz", "xlmr_base_idhead_final.npz",
     "xlm-roberta-base + ID-head classifier (verified export)", "ID-derived"),
]
STANDALONE = {"text_tfidf_final.npz": 0.516093, "text_knn_final.npz": 0.487078,
              "emb_e5-base_final.npz": 0.521535, "sid.npz": 0.627895,
              "plain_id.npz": 0.613631, "xlmr_base_idhead_final.npz": 0.648743}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def mf1(yy, pr):
    cm = np.bincount(yy * 3 + pr, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    P = tp / np.maximum(cm.sum(0), 1)
    R = tp / np.maximum(cm.sum(1), 1)
    return float((2 * P * R / np.maximum(P + R, 1e-12)).mean())


for src, dst, _, _ in SOURCES:
    if not os.path.exists(src):
        alt = "uap_submission/reproducible/artifacts_submission_3/" + dst
        if os.path.exists(alt):
            src = alt
        else:
            raise SystemExit("missing source " + src)
    shutil.copy2(src, os.path.join(ART, dst))
    print("bundled %-32s <- %s" % (dst, src))

folds = np.load(os.path.join(ART, "fold_ids_colab.npy")).astype(np.int64)
fold_sha = hashlib.sha256(np.ascontiguousarray(folds, np.int64).tobytes()).hexdigest()
assert fold_sha == "cc4fdd169110aeee6d3d5566095234461041b1a9290e2e916ce850c31d27a7ca", fold_sha
y = pd.read_csv("train.csv")["y"].to_numpy(int)

srcinfo = {}
for _, dst, purpose, kind in SOURCES[1:]:
    z = np.load(os.path.join(ART, dst), allow_pickle=True)
    o, t = z["oof"], z["test"]
    sc = mf1(y, o.argmax(1))
    assert abs(sc - STANDALONE[dst]) < 0.003, (dst, sc)
    srcinfo[dst] = dict(purpose=purpose, oof_shape=list(o.shape), test_shape=list(t.shape),
                        standalone_oof_macro_f1=round(sc, 6),
                        expected_standalone=STANDALONE[dst], derivation=kind,
                        uses_id=kind == "ID-derived", fold_set="B")
    print("  %-32s oof %-12s test %-11s OOF %.6f  %s" % (dst, o.shape, t.shape, sc, kind))

SRC_ORDER = ["tfidf", "knn", "e5", "sid", "pid", "idh"]
Ftr = pd.read_parquet("repro_s4/Ftr.parquet")
groups = {"source_probs": ["%s_p%d" % (n, k) for n in SRC_ORDER for k in range(3)],
          "id_neighbour": [c for c in Ftr.columns if c.startswith(("prev", "next", "w"))],
          "text_meta": ["len", "nwords", "bn", "lat", "emoji", "caps", "excl", "q", "digits"],
          "raw_id": ["rid"]}
assert set(sum(groups.values(), [])) == set(Ftr.columns), sorted(set(Ftr.columns) - set(sum(groups.values(), [])))
json.dump(dict(n_features=int(Ftr.shape[1]), n_rows=int(Ftr.shape[0]), test_rows=11955,
               groups=groups, column_order=list(Ftr.columns)),
          open(os.path.join(ART, "feature_schema.json"), "w"), indent=2)
print("feature schema:", Ftr.shape[1], "features ->", {k: len(v) for k, v in groups.items()})
json.dump(dict(sources=srcinfo), open("repro_s4/_srcinfo.json", "w"), indent=2)


# ---- frozen config ---------------------------------------------------------
rep = json.load(open("claud/sub4_code/final_report.json"))
FROZEN = dict(
    submission=4,
    title="LightGBM stacker over six base probability sources (ID-DERIVED)",
    validation_classification="STANDARD_OOF_STACKING_WITH_DEPENDENCY",
    validation_note=(
        "NOT strictly outer-fold safe. Base-source OOF arrays are standard stacking "
        "inputs: for a training row of fold j the base model saw folds != j, which "
        "include fold k, so the stacker's training features encode fold-k label "
        "information. The per-fold class offsets are additionally chosen on the other "
        "four folds' stacker OOF (single-level, not double-nested). No claim of strict "
        "nesting is made."),
    metric="macro_f1", classes=[0, 1, 2], n_splits=5, fold_set="B",
    fold_ids_int64_sha256=fold_sha,
    feature_policy=dict(
        n_features=60,
        source_order=list(SRC_ORDER),
        source_derivation={
            "tfidf": srcinfo["text_tfidf_final.npz"]["derivation"],
            "knn": srcinfo["text_knn_final.npz"]["derivation"],
            "e5": srcinfo["emb_e5-base_final.npz"]["derivation"],
            "sid": srcinfo["sid.npz"]["derivation"],
            "pid": srcinfo["plain_id.npz"]["derivation"],
            "idh": srcinfo["xlmr_base_idhead_final.npz"]["derivation"]},
        id_neighbour_windows=[3, 10, 30, 100, 300], id_neighbour_offsets=[1, 2, 3],
        neighbour_pool="train rows of the OTHER folds only",
        raw_id_feature="rid",
        text_meta=["len", "nwords", "bn", "lat", "emoji", "caps", "excl", "q", "digits"]),
    lightgbm_params=dict(objective="multiclass", num_class=3, learning_rate=0.05,
                         num_leaves=31, min_data_in_leaf=40, feature_fraction=0.8,
                         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                         num_threads=2, verbose=-1),
    rounds=400, seeds=[42, 7, 2026],
    seed_policy=("three fixed seeds averaged; each seed also sets bagging_seed and "
                 "feature_fraction_seed; num_threads pinned to 2 for determinism"),
    class_balance="cw=len(y)/(3*bincount(y)); sample weight = cw[y]",
    class_correction=dict(
        kind="per-fold additive logit offset", frozen=True,
        fold_offsets=rep["fold_offsets"], final_offset=rep["final_offset"],
        applied_to="argmax(log p + offset)",
        selection="each fold's offset maximises macro-F1 on the OTHER four folds"),
    duplicate_override=dict(enabled=True, rule="unanimous cleaned-text train groups",
                            rows_matched=272, labels_changed=37,
                            source="submission_reproducible_file_3.ipynb cells 5 and 15 (verbatim)"),
    reproduced=dict(pooled_oof_macro_f1=0.703904, recorded_target=0.704001,
                    difference=-0.000097,
                    folds=[0.709531, 0.705706, 0.699544, 0.700173, 0.704385],
                    per_class_f1=[0.782266, 0.672590, 0.656856],
                    seed_scores=[0.693895, 0.694031, 0.692881]),
    train_rows=47817, test_rows=11955,
    train_class_share=[0.496811, 0.379844, 0.123345],
    leakage_disclosure=("Base-source OOF arrays are standard stacking inputs; see "
                        "validation_note. Folds are grouped by cleaned text so no group "
                        "spans folds."),
)
json.dump(FROZEN, open(os.path.join(ART, "config_submission_4.json"), "w"), indent=2)
print("wrote config_submission_4.json")

import lightgbm
import scipy
import sklearn
json.dump(dict(python=platform.python_version(), numpy=np.__version__,
               pandas=pd.__version__, scipy=scipy.__version__,
               scikit_learn=sklearn.__version__, lightgbm=lightgbm.__version__,
               pyarrow=__import__("pyarrow").__version__, platform=platform.platform(),
               mode="A (CPU only, no GPU, no network)"),
          open(os.path.join(ART, "package_versions.json"), "w"), indent=2)
print("wrote package_versions.json")
print("ARTIFACTS OK")


"""v6 Section 7: final weights, test probs, G4/G5, override, validation, CSV."""
import os
import re
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

CLASSES = [0, 1, 2]
WORK = "cache/v6"
OUT_DIR = "uap_submission/submission"
os.makedirs(OUT_DIR, exist_ok=True)
TRAIN_SHARE = [0.4968, 0.3798, 0.1233]
USE_DUP_OVERRIDE = True

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def clean_text(t):
    s = "" if t is None else str(t)
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    return WS_RE.sub(" ", s).strip()


def unanimous_lookup(texts, labels):
    df = pd.DataFrame({"t": pd.Series(list(texts), dtype="object"),
                       "y": np.asarray(labels)})
    g = df.groupby("t")["y"].agg(["nunique", "first"])
    uni = g[g["nunique"] == 1]
    lookup = {str(t): int(v) for t, v in uni["first"].items()}
    n_conflicting = int((g["nunique"] > 1).sum())
    return lookup, n_conflicting


def apply_override(texts, pred, lookup):
    out = np.asarray(pred).copy()
    hits = 0
    for i, t in enumerate(texts):
        v = lookup.get(str(t))
        if v is not None:
            out[i] = v
            hits += 1
    return out, hits


def validate(df, sample):
    assert list(df.columns) == ["id", "y_pred"], list(df.columns)
    assert len(df) == 11955, len(df)
    assert df["id"].is_unique
    assert np.array_equal(df["id"].to_numpy(), sample["id"].to_numpy())
    assert not df.isna().any().any()
    assert df["y_pred"].dtype.kind in "iu"
    assert set(df["y_pred"].unique()) <= {0, 1, 2}
    return True


def main():
    d = np.load(WORK + "/step123.npz", allow_pickle=True)
    y, ids, test_ids = d["y"], d["ids"], d["test_ids"]
    blend = json.load(open("reports/v6/blend.json"))
    w = np.array(blend["sid_variant"]["median_weights"], dtype=float)
    w = w / w.sum()
    print("final weights (median of 5 fold vectors, renormalised):", w.tolist())

    test_sources = {"text": d["test_text"], "sid": d["test_sid"],
                    "e5_base": d["e5_test"], "xlmr_idhead": d["xlmr_test"]}
    test_mix = sum(float(wi) * test_sources[k] for wi, k in zip(w, test_sources))
    test_mix = test_mix / test_mix.sum(axis=1, keepdims=True)

    sample = pd.read_csv("sample_submission.csv")
    train = pd.read_csv("train.csv", encoding="utf-8")
    test = pd.read_csv("test.csv", encoding="utf-8")
    test["text"] = test["text"].astype("string").fillna("").astype(str)
    clean_te = test["text"].map(clean_text).to_numpy()
    ytr = train["y"].to_numpy(int)
    train_share = [float((ytr == c).mean()) for c in CLASSES]
    print("train class share :", [round(v, 4) for v in train_share])
    print("declared          :", TRAIN_SHARE)

    # G4 predicted test class share within 6 pp of train share
    pred_pure = test_mix.argmax(1)
    share = [float((pred_pure == c).mean()) for c in CLASSES]
    drift_pp = [100.0 * abs(share[c] - train_share[c]) for c in CLASSES]
    g4 = all(v <= 6.0 for v in drift_pp)
    print("test class share  :", [round(v, 4) for v in share])
    print("drift (pp)        :", [round(v, 3) for v in drift_pp])
    print("G4:", "PASS" if g4 else "FAIL")

    # G5 test probs
    g5 = (np.isfinite(test_mix).all() and test_mix.shape == (11955, 3)
          and float(np.abs(test_mix.sum(1) - 1).max()) < 1e-6)
    print("G5: shape {} finite {} max|rowsum-1| {:.2e} order==sample {} -> {}".format(
        test_mix.shape, bool(np.isfinite(test_mix).all()),
        float(np.abs(test_mix.sum(1) - 1).max()),
        bool(np.array_equal(test_ids, sample["id"].to_numpy())),
        "PASS" if g5 else "FAIL"))
    if not (g4 and g5):
        print("*** STOP at G4/G5: no CSV written ***")
        return

    # override
    clean_tr = train["text"].astype("string").fillna("").astype(str).map(clean_text).to_numpy()
    lookup, n_conflict = unanimous_lookup(clean_tr, ytr)
    pred_final, n_hits = apply_override(clean_te, pred_pure, lookup) if USE_DUP_OVERRIDE \
        else (pred_pure.copy(), 0)
    changed = int((pred_final != pred_pure).sum())
    print("override: {} rows matched, {} labels changed, {} conflicting groups skipped".format(
        n_hits, changed, n_conflict))

    sub = pd.DataFrame({"id": test_ids, "y_pred": pred_final.astype(int)})
    validate(sub, sample)
    path = OUT_DIR + "/submission_3.csv"
    sub.to_csv(path, index=False)
    re_df = pd.read_csv(path)
    validate(re_df, sample)
    print("POST-validation OK ->", path)

    s2 = pd.read_csv("uap_submission/submission/submission_2.csv")
    assert np.array_equal(s2["id"].to_numpy(), sub["id"].to_numpy())
    diff = int((s2["y_pred"].to_numpy() != sub["y_pred"].to_numpy()).sum())
    print("labels differing from Submission 2: {} / {} = {:.2f}%".format(
        diff, len(sub), 100.0 * diff / len(sub)))
    print("CSV SHA256:", hashlib.sha256(open(path, "rb").read()).hexdigest())

    np.savez_compressed(WORK + "/final_test.npz", test_mix=test_mix,
                        final_weights=w, pred_pure=pred_pure, pred_final=pred_final)
    json.dump({"final_weights": [float(v) for v in w],
               "source_order": list(test_sources),
               "test_class_share": share, "train_class_share": train_share,
               "drift_pp": drift_pp,
               "override_rows_matched": int(n_hits), "override_labels_changed": changed,
               "override_conflicting_groups": int(n_conflict),
               "diff_vs_submission_2": diff,
               "honest_oof": blend["sid_variant"]["honest_oof"]},
              open("reports/v6/final.json", "w"), indent=2)
    print("-> reports/v6/final.json")


if __name__ == "__main__":
    main()

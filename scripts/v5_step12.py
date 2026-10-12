"""v5 Step 1+2: verify recovered GPU artifacts; rebuild text + plain k=20 ID prior
on the COLAB fold set. Hard gate: plain-ID OOF must equal 0.613631 (+-1e-5).
"""
import os
import io
import re
import sys
import json
import time
import hashlib
import zipfile
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

SEED = 42
N_SPLITS = 5
CLASSES = [0, 1, 2]
ZIP = r"C:\Users\DREAMER\Downloads\uap_export.zip"
WORK = "cache/v5"
os.makedirs(WORK, exist_ok=True)
os.makedirs("reports/v5", exist_ok=True)
for d in ("predictions/v5_candidates", "models/v5"):
    os.makedirs(d, exist_ok=True)

WORD_KW = dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60000,
               sublinear_tf=True, dtype=np.float32)
CHAR_KW = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60000,
               sublinear_tf=True, dtype=np.float32)
LR_KW = dict(solver="lbfgs", C=4.0, class_weight="balanced", max_iter=2000,
             random_state=SEED)
K_NEIGHBORS, ID_EPS = 20, 1e-6

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def clean_text(t):
    s = "" if t is None else str(t)
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    return WS_RE.sub(" ", s).strip()


def group_key(c):
    return str(c).lower().strip()


def mf1(y, p):
    return float(f1_score(y, np.asarray(p).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def pcf1(y, p):
    return [round(float(v), 6) for v in
            f1_score(y, np.asarray(p).argmax(1), average=None, labels=CLASSES,
                     zero_division=0)]


def id_prior(q_ids, pool_ids, pool_y, k=K_NEIGHBORS, eps=ID_EPS):
    """k-nearest-id inverse-distance label prior. Pool must exclude query ids."""
    q = np.asarray(q_ids, np.int64).ravel()
    t = np.asarray(pool_ids, np.int64).ravel()
    yy = np.asarray(pool_y, np.int64).ravel()
    order = np.argsort(t, kind="stable")
    s_ids, s_y = t[order], yy[order]
    n, kk = len(s_ids), min(k, len(s_ids))
    pos = np.searchsorted(s_ids, q, side="left")
    cand = pos[:, None] + np.arange(-k, k + 1)[None, :]
    valid = (cand >= 0) & (cand < n)
    cc = np.clip(cand, 0, n - 1)
    d = np.abs(s_ids[cc] - q[:, None]).astype(np.float64)
    d = np.where(valid & (s_ids[cc] != q[:, None]), d, np.inf)
    sel = np.argsort(d, axis=1, kind="stable")[:, :kk]
    dsel = np.take_along_axis(d, sel, axis=1)
    ysel = s_y[cc[np.arange(len(q))[:, None], sel]]
    good = np.isfinite(dsel)
    wt = np.where(good, 1.0 / (np.where(good, dsel, 0.0) + eps), 0.0)
    P = np.stack([(wt * (ysel == c)).sum(axis=1) for c in CLASSES], axis=1)
    tot = P.sum(axis=1, keepdims=True)
    P = P / np.where(tot > 0, tot, 1.0)
    P[tot.ravel() <= 0] = 1.0 / 3
    assert int(np.isin(q, s_ids).sum()) == 0, "leak: query id in pool"
    return P


def main():
    t0 = time.time()
    train = pd.read_csv("train.csv", encoding="utf-8")
    test = pd.read_csv("test.csv", encoding="utf-8")
    train["text"] = train["text"].astype("string").fillna("").astype(str)
    test["text"] = test["text"].astype("string").fillna("").astype(str)
    y = train["y"].to_numpy(int)
    ids = train["id"].to_numpy(np.int64)
    test_ids = test["id"].to_numpy(np.int64)
    clean = train["text"].map(clean_text).to_numpy()
    clean_te = test["text"].map(clean_text).to_numpy()

    # ---------- STEP 1 ----------
    print("=" * 76)
    print("STEP 1 - ARTIFACT VERIFICATION")
    print("=" * 76)
    zf = zipfile.ZipFile(ZIP)
    print("zip:", ZIP, "| entries:", sorted(zf.namelist()))
    meta = json.loads(zf.read("export_meta.json").decode("utf-8"))
    print("export_meta:", json.dumps(meta))

    train_sha1 = hashlib.sha1(open("train.csv", "rb").read()).hexdigest()
    test_sha1 = hashlib.sha1(open("test.csv", "rb").read()).hexdigest()
    ok_sha1 = (train_sha1 == meta["train_sha1"] and test_sha1 == meta["test_sha1"])
    print("train SHA1 {} | expected {} | match {}".format(
        train_sha1[:16], meta["train_sha1"][:16], train_sha1 == meta["train_sha1"]))
    print("test  SHA1 {} | expected {} | match {}".format(
        test_sha1[:16], meta["test_sha1"][:16], test_sha1 == meta["test_sha1"]))
    assert ok_sha1, "train/test SHA1 mismatch vs export_meta"

    folds = np.load(io.BytesIO(zf.read("fold_ids.npy")))
    fold_sha = hashlib.sha256(folds.tobytes()).hexdigest()
    sizes = np.bincount(folds, minlength=5).tolist()
    print("\nfold_ids: shape {} dtype {} sizes {}".format(folds.shape, folds.dtype, sizes))
    print("fold SHA256:", fold_sha)
    assert folds.shape == (47817,) and sizes == [9543, 9546, 9571, 9581, 9576]
    assert int(np.bincount(folds).sum()) == 47817
    gk = pd.Index([group_key(c) for c in clean], dtype="object")
    g = pd.factorize(gk, sort=False)[0]
    cross = int((pd.DataFrame({"g": g, "f": folds}).groupby("g")["f"].nunique() > 1).sum())
    print("cleaned-text groups crossing folds:", cross, "(must be 0)")
    assert cross == 0

    src = {}
    for n, exp in (("xlmr_base_idhead_final.npz", 0.648743),
                   ("emb_e5-base_final.npz", 0.521535)):
        d = np.load(io.BytesIO(zf.read(n)))
        o, t = d["oof"], d["test"]
        sc = mf1(y, o)
        print("\n{}: oof{} test{}".format(n, o.shape, t.shape))
        print("  finite {} | in[0,1] {} | maxrowsumdev {:.2e}".format(
            bool(np.isfinite(o).all()), bool((o >= 0).all() and (o <= 1).all()),
            float(np.abs(o.sum(1) - 1).max())))
        print("  standalone OOF macro F1 = {:.6f} (expected {:.6f}, diff {:.2e})".format(
            sc, exp, abs(sc - exp)))
        assert o.shape == (47817, 3) and t.shape == (11955, 3)
        assert abs(sc - exp) < 2e-6, "standalone mismatch for " + n
        src[n.split("_final")[0]] = (o, t)
    zf.close()
    print("\nall STEP 1 checks PASSED")

    # ---------- STEP 2 ----------
    print()
    print("=" * 76)
    print("STEP 2 - REBUILD TEXT + PLAIN ID PRIOR ON COLAB FOLDS")
    print("=" * 76)
    ntr, nte = len(train), len(test)
    oof_text = np.zeros((ntr, 3))
    oof_id = np.zeros((ntr, 3))
    for k in range(N_SPLITS):
        va = np.flatnonzero(folds == k)
        tr = np.flatnonzero(folds != k)
        w = TfidfVectorizer(**WORD_KW).fit(clean[tr])
        c = TfidfVectorizer(**CHAR_KW).fit(clean[tr])
        Xtr = sp.hstack([w.transform(clean[tr]), c.transform(clean[tr])],
                        format="csr").astype(np.float32)
        Xva = sp.hstack([w.transform(clean[va]), c.transform(clean[va])],
                        format="csr").astype(np.float32)
        oof_text[va] = LogisticRegression(**LR_KW).fit(Xtr, y[tr]).predict_proba(Xva)
        oof_id[va] = id_prior(ids[va], ids[tr], y[tr])
        print("  fold {}: text {:.6f} | id {:.6f}".format(
            k, mf1(y[va], oof_text[va]), mf1(y[va], oof_id[va])))

    text_f1 = mf1(y, oof_text)
    id_f1 = mf1(y, oof_id)
    print("\ntext-only OOF macro F1 = {:.6f}  (expect ~0.5160 +-0.003)".format(text_f1))
    print("plain-ID OOF macro F1  = {:.6f}  (expect 0.613631 +-1e-5)".format(id_f1))
    assert abs(text_f1 - 0.5160) <= 0.003, "text-only OOF outside 0.5160+-0.003"
    if abs(id_f1 - 0.613631) > 1e-5:
        print("\n*** STOP: plain-ID OOF differs from 0.613631 by more than 1e-5 ***")
        print("    diff = {:.3e}".format(abs(id_f1 - 0.613631)))
        return
    print("plain-ID gate PASSED (diff {:.2e})".format(abs(id_f1 - 0.613631)))

    # full-train text fit -> test
    w = TfidfVectorizer(**WORD_KW).fit(clean)
    c = TfidfVectorizer(**CHAR_KW).fit(clean)
    Xtr = sp.hstack([w.transform(clean), c.transform(clean)],
                    format="csr").astype(np.float32)
    Xte = sp.hstack([w.transform(clean_te), c.transform(clean_te)],
                    format="csr").astype(np.float32)
    test_text = LogisticRegression(**LR_KW).fit(Xtr, y).predict_proba(Xte)
    test_id = id_prior(test_ids, ids, y)
    print("test text {} | test plain-ID {}".format(test_text.shape, test_id.shape))

    np.savez_compressed(WORK + "/step12.npz", oof_text=oof_text, oof_id=oof_id,
                        test_text=test_text, test_id=test_id,
                        fold_ids=folds, y=y, ids=ids, test_ids=test_ids,
                        xlmr_idhead_oof=src["xlmr_base_idhead"][0],
                        xlmr_idhead_test=src["xlmr_base_idhead"][1],
                        e5base_oof=src["emb_e5-base"][0],
                        e5base_test=src["emb_e5-base"][1])
    with open("reports/v5/step12.json", "w") as f:
        json.dump({"fold_sha256": fold_sha, "fold_sizes": sizes,
                   "text_oof": round(text_f1, 6), "id_oof": round(id_f1, 6),
                   "text_per_class": pcf1(y, oof_text),
                   "id_per_class": pcf1(y, oof_id),
                   "train_sha1": train_sha1, "test_sha1": test_sha1,
                   "runtime_s": round(time.time() - t0, 1)}, f, indent=2)
    print("\n-> cache/v5/step12.npz, reports/v5/step12.json  ({:.0f}s)".format(
        time.time() - t0))


if __name__ == "__main__":
    main()

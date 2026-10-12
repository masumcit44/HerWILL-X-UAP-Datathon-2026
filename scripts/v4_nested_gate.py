"""v4 Phase 4: nested cross-fitted class-conditional source-reliability gate.

STRICT nesting: for outer fold k the gate's own training features are produced by
an INNER cross-fit (each inner group predicted by base models trained on the other
three inner groups). The outer fold's features come from base models trained on all
four outer-training folds -- i.e. on MORE data than the gate saw, so the gate is
scored conservatively rather than optimistically.

No generic model is ever trained and scored on the same rows.
"""
import os
import re
import json
import time
import unicodedata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import emoji as emoji_lib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_SPLITS = 5
CLASSES = [0, 1, 2]
SUB2_REF = 0.657612
os.makedirs("reports/v4", exist_ok=True)

WORD_KW = dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60000,
               sublinear_tf=True, dtype=np.float32)
CHAR_KW = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60000,
               sublinear_tf=True, dtype=np.float32)
LR_KW = dict(solver="lbfgs", C=4.0, class_weight="balanced", max_iter=2000,
             random_state=SEED)

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+", flags=re.UNICODE)
REPEAT_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE)


def clean_base(t):
    s = "" if t is None else str(t)
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    return WS_RE.sub(" ", s).strip()


def clean_all3(t):
    s = "" if t is None else str(t)
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    s = WS_RE.sub(" ", s).strip()
    s = REPEAT_RE.sub(r"\1\1", s)
    return emoji_lib.demojize(s, language="en")


def mf1(y, p):
    return float(f1_score(y, np.asarray(p).argmax(1), average="macro",
                          labels=CLASSES, zero_division=0))


def pcf1(y, p):
    return [float(v) for v in f1_score(y, np.asarray(p).argmax(1), average=None,
                                       labels=CLASSES, zero_division=0)]


def norm(P):
    return P / P.sum(axis=1, keepdims=True)


def id_prior(q_ids, t_ids, t_y, k=20, eps=1e-6):
    q = np.asarray(q_ids, np.int64).ravel()
    t = np.asarray(t_ids, np.int64).ravel()
    yy = np.asarray(t_y, np.int64).ravel()
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
    w = np.where(good, 1.0 / (np.where(good, dsel, 0.0) + eps), 0.0)
    P = np.stack([(w * (ysel == c)).sum(axis=1) for c in CLASSES], axis=1)


def build_source_block(va_idx, tr_idx, clean_a, clean_b, ids, y):
    """Recompute every base source for va_idx using ONLY tr_idx."""
    out = {}
    # --- text (base + all3), two separate vectoriser pairs ---
    for tag, cl in (("base", clean_a), ("all3", clean_b)):
        w = TfidfVectorizer(**WORD_KW).fit(cl[tr_idx])
        c = TfidfVectorizer(**CHAR_KW).fit(cl[tr_idx])
        Xtr = sp.hstack([w.transform(cl[tr_idx]), c.transform(cl[tr_idx])],
                        format="csr").astype(np.float32)
        Xva = sp.hstack([w.transform(cl[va_idx]), c.transform(cl[va_idx])],
                        format="csr").astype(np.float32)
        out[tag] = LogisticRegression(**LR_KW).fit(Xtr, y[tr_idx]).predict_proba(Xva)
    out["text"] = norm(0.5 * out["base"] + 0.5 * out["all3"])
    # --- ID prior ---
    out["id"] = id_prior(ids[va_idx], ids[tr_idx], y[tr_idx])
    return out


def build_emb_block(va_idx, tr_idx, y, emb):
    clf = LogisticRegression(**LR_KW).fit(emb[tr_idx], y[tr_idx])
    return clf.predict_proba(emb[va_idx])


def reliability_features(sources, va_idx, ids_sorted, y_sorted):
    """Confidence / agreement / text-shape features for the given rows."""
    names = sorted(sources.keys())
    S = np.stack([sources[n] for n in names])
    feats = [S.reshape(len(names), -1).T]
    top = np.sort(S, axis=2)
    conf = top[:, :, 2]
    margin = top[:, :, 2] - top[:, :, 1]
    ent = -np.sum(S * np.log(np.clip(S, 1e-12, 1.0)), axis=2)
    for a in range(len(names)):
        oh = np.zeros((len(va_idx), 3))
        oh[np.arange(len(va_idx)), S[a].argmax(1)] = 1.0
        feats += [conf[a][:, None], margin[a][:, None], ent[a][:, None], oh]
    preds = S.argmax(2)
    votes = np.stack([(preds == c).sum(axis=0) for c in range(3)], axis=1).astype(float)
    unanimous = (votes.max(axis=1) == len(names)).astype(float)[:, None]
    feats += [votes, unanimous]
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            feats.append((preds[a] != preds[b]).astype(float)[:, None])
    return np.hstack(feats)


def text_shape(va_idx, clean_a):
    s = clean_a[va_idx]
    n = np.maximum(np.array([len(x) for x in s], dtype=float), 1.0)
    out = np.stack([
        n,
        np.array([sum(0x0980 <= ord(ch) <= 0x09FF for ch in x) for x in s]) / n,
        np.array([sum(ord("a") <= ord(ch) <= ord("z") or ord("A") <= ord(ch) <= ord("Z")
                      for ch in x) for x in s]) / n,
        np.array([sum(ord("0") <= ord(ch) <= ord("9") for ch in x) for x in s]) / n,
        np.array([1.0 if " URL " in x else 0.0 for x in s]),
        np.array([1.0 if " USER " in x else 0.0 for x in s]),
    ], axis=1)
    return out

    tot = P.sum(axis=1, keepdims=True)
    P = P / np.where(tot > 0, tot, 1.0)
    P[tot.ravel() <= 0] = 1.0 / 3
    return P


def main():
    train = pd.read_csv("train.csv", encoding="utf-8")
    train["text"] = train["text"].astype("string").fillna("").astype(str)
    y = train["y"].to_numpy(int)
    ids = train["id"].to_numpy(np.int64)
    clean_a = train["text"].map(clean_base).to_numpy()
    clean_b = train["text"].map(clean_all3).to_numpy()

    z = np.load("cache/v4/sources_oof.npz")
    folds = z["fold_ids"]
    assert np.array_equal(folds, np.load("cache/v2/fold_ids.npy"))
    src_names = [k for k in z.files
                 if k not in ("fold_ids", "y", "sub2", "text_base", "text_all3")]

    emb_small = np.load("cache/v3/emb_e5_small_train.npy")
    emb_paths = {"e5_small": emb_small}
    p_big = "cache/v4/emb_e5base_train.npy"
    use_big = os.path.exists(p_big)
    if use_big:
        e = np.load(p_big)
        if e.shape[0] == len(y) and bool(np.isfinite(e).all()):
            emb_paths["e5_base"] = e
            src_names.append("e5_base")
        else:
            use_big = False
    print("gate sources:", src_names, "| e5_base available:", use_big)

    # canonical (outer) OOF sources = the already-verified arrays
    def sources_for(idx, va_idx, tr_idx):
        out = {}
        for nm in src_names:
            if nm in ("text", "id"):
                continue
            out[nm] = build_emb_block(va_idx, tr_idx, y, emb_paths[nm])
        out["text"] = z["text"][idx]
        out["id"] = z["id"][idx]
        return out

    t0 = time.time()
    gate_pred = np.zeros(len(y), dtype=int)
    gate_prob = np.zeros((len(y), 3))
    per_fold = {}
    for k in range(N_SPLITS):
        outer_va = np.flatnonzero(folds == k)
        tr_folds = [f for f in range(N_SPLITS) if f != k]
        # ---- INNER cross-fit over the 4 outer-training folds ----
        Xtr_list, ytr_list = [], []
        for g, gf in enumerate(tr_folds):
            inner_va = np.flatnonzero(folds == gf)
            inner_tr = np.flatnonzero(np.isin(folds, [f for f in tr_folds if f != gf]))
            src = build_source_block(inner_va, inner_tr, clean_a, clean_b, ids, y)
            for nm in src_names:
                if nm in ("text", "id"):
                    continue
                src[nm] = build_emb_block(inner_va, inner_tr, y, emb_paths[nm])
            X = reliability_features(src, inner_va, ids, y)
            X = np.hstack([X, text_shape(inner_va, clean_a)])
            Xtr_list.append(X)
            ytr_list.append(y[inner_va])
        Xtr = np.vstack(Xtr_list)
        ytr = np.concatenate(ytr_list)

        # ---- outer fold features from the canonical verified OOF ----
        src = {nm: z[nm][outer_va] for nm in src_names}
        Xva = reliability_features(src, outer_va, ids, y)
        Xva = np.hstack([Xva, text_shape(outer_va, clean_a)])

        g1 = HistGradientBoostingClassifier(max_depth=4, max_iter=200,
                                           learning_rate=0.06,
                                           random_state=SEED)
        g1.fit(Xtr, ytr)
        p1 = g1.predict_proba(Xva)
        g2 = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        g2.fit(Xtr, ytr)
        p2 = g2.predict_proba(Xva)

        gate_prob[outer_va] = p1
        gate_pred[outer_va] = p1.argmax(1)
        sc1 = mf1(y[outer_va], p1)
        sc2 = mf1(y[outer_va], p2)
        per_fold[str(k)] = {"hgb_fold_f1": round(sc1, 6), "lr_gate_fold_f1": round(sc2, 6),
                            "n_train": int(len(ytr))}
        print("  fold {}: HGB gate {:.6f} | LR gate {:.6f}  (train rows {})"
              .format(k, sc1, sc2, len(ytr)), flush=True)

    f1 = mf1(y, gate_prob)
    fs = [mf1(y[folds == k], gate_prob[folds == k]) for k in range(N_SPLITS)]
    print("\nNESTED GATE honest OOF Macro F1: {:.6f}  (delta vs 0.657612 = {:+.6f})"
          .format(f1, f1 - SUB2_REF))
    print("  per-class : {}".format([round(v, 6) for v in pcf1(y, gate_prob)]))
    print("  fold scores: {}".format([round(v, 6) for v in fs]))
    print("  fold std   : {:.6f}".format(float(np.std(fs))))
    print("  runtime    : {:.0f}s".format(time.time() - t0))
    np.save("predictions/v4_candidates/candidate_nested_gate_oof_probs.npy", gate_prob)
    with open("reports/v4/nested_gate.json", "w") as f:
        json.dump({"sources": src_names, "e5_base_used": use_big,
                   "honest_f1": f1, "honest_delta": f1 - SUB2_REF,
                   "per_class": [round(v, 6) for v in pcf1(y, gate_prob)],
                   "fold_scores": [round(v, 6) for v in fs],
                   "fold_std": float(np.std(fs)),
                   "per_fold": per_fold}, f, indent=2)
    print("-> reports/v4/nested_gate.json")


if __name__ == "__main__":
    main()


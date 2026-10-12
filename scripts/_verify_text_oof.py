"""Verify fresh recomputation of base-text and all3-text OOF matches cached artifacts."""
import re, sys, time, unicodedata
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, "src")
import experiment_v2_harness as H

WORD_KW, CHAR_KW, LR_KW = H.WORD_KW, H.CHAR_KW, H.LR_KW
N_SPLITS, SEED, K = 5, 42, 20

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
    import emoji as _e
    s = "" if t is None else str(t)
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    s = WS_RE.sub(" ", s).strip()
    s = REPEAT_RE.sub(r"\1\1", s)
    return _e.demojize(s, language="en")


def folds_for(cleaned, y):
    keys = pd.Index([str(t).lower().strip() for t in cleaned], dtype="object")
    groups = pd.factorize(keys, sort=False)[0]
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    f = np.full(len(y), -1, dtype=int)
    for k, (_t, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        f[va] = k
    assert (f >= 0).all()
    return f


def text_oof(cleaned, y, fold_ids):
    oof = np.zeros((len(y), 3))
    for k in range(N_SPLITS):
        va = np.flatnonzero(fold_ids == k)
        tr = np.flatnonzero(fold_ids != k)
        w = TfidfVectorizer(**WORD_KW).fit(cleaned[tr])
        c = TfidfVectorizer(**CHAR_KW).fit(cleaned[tr])
        Xtr = sp.hstack([w.transform(cleaned[tr]), c.transform(cleaned[tr])],
                        format="csr").astype(np.float32)
        Xva = sp.hstack([w.transform(cleaned[va]), c.transform(cleaned[va])],
                        format="csr").astype(np.float32)
        oof[va] = LogisticRegression(**LR_KW).fit(Xtr, y[tr]).predict_proba(Xva)
    return oof


tr = pd.read_csv("train.csv", encoding="utf-8")
tr["text"] = tr["text"].astype("string").fillna("").astype(str)
y = tr["y"].to_numpy(int)

t0 = time.time()
base_clean = tr["text"].map(clean_base).to_numpy()
base_folds = folds_for(base_clean, y)
canon = np.load("cache/v2/fold_ids.npy")
print("base folds == canonical:", bool(np.array_equal(base_folds, canon)))
base_oof = text_oof(base_clean, y, base_folds)
ref_base = np.load("predictions/v2/v2_0_baseline_anchor_oof.npz")["oof_text"]
print("base text OOF identical to cache:",
      bool(np.allclose(base_oof, ref_base, atol=1e-9)),
      "| maxdiff", float(np.abs(base_oof - ref_base).max()))
print("base text Macro F1:", f1_score(y, base_oof.argmax(1), average="macro"), "(cached 0.512943)")

a3_clean = tr["text"].map(clean_all3).to_numpy()
a3_folds = folds_for(a3_clean, y)
d3 = np.load("cache/t0/oof_all3.npz", allow_pickle=True)
print("all3 folds == cached all3 folds:", bool(np.array_equal(a3_folds, d3["fold_ids"])))
a3_oof = text_oof(a3_clean, y, a3_folds)
print("all3 text OOF identical to cache:", bool(np.allclose(a3_oof, d3["oof_text"], atol=1e-9)),
      "| maxdiff", float(np.abs(a3_oof - d3["oof_text"]).max()))
print("all3 text Macro F1:", f1_score(y, a3_oof.argmax(1), average="macro"), "(cached 0.518947)")
np.savez("cache/v3/_verify_text_oof.npz", base=base_oof, all3=a3_oof,
         base_folds=base_folds, a3_folds=a3_folds)
print("elapsed", round(time.time() - t0, 1), "s")

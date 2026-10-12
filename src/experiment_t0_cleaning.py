"""
src/experiment_t0_cleaning.py
============================

TIER 0 -- text-cleaning A/B upgrades.

Each cleaning variant is evaluated through the EXACT existing leak-free grouped
protocol (StratifiedGroupKFold(5, shuffle=True, random_state=42), group =
clean_text.lower().strip()), with the same text model as the locked baseline
(word 1-2 + char_wb 3-5 TF-IDF -> LogisticRegression(C=4.0, class_weight=
'balanced')) and the same k=20 ID-locality prior, blend alpha=0.40.

Variants are tested ALONE first, then in combination, so any gain can be
attributed to a specific change.

Anchors this is measured against (from reports/v2_comparison_table.csv):
    text-TFIDF-only  pooled OOF Macro F1 = 0.512943
    id-prior-only    pooled OOF Macro F1 = 0.614135
    blend @ alpha=0.40 (LOCKED)           = 0.639579

No submission is written by this script. The locked notebook and submission_1
are sha1-verified untouched at the end.
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_v2_harness as H  # noqa: E402

SEED = 42
N_SPLITS = 5
ALPHA = 0.40
TEXT_ONLY_ANCHOR = 0.512943
ID_ONLY_ANCHOR = 0.614135
BASELINE = 0.639579
NOISE_BAND = 0.005

WORD_KW = H.WORD_KW
CHAR_KW = H.CHAR_KW
LR_KW = H.LR_KW
K = H.K_NEIGHBORS

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
CACHE = ROOT / "cache" / "t0"
CACHE.mkdir(parents=True, exist_ok=True)

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+", flags=re.UNICODE)
REPEAT_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE)


# --------------------------------------------------------------------------- #
# Cleaning steps (composable)
# --------------------------------------------------------------------------- #
def step_nfkc(s: str) -> str:
    """(1) Unicode NFKC normalisation before masking."""
    return unicodedata.normalize("NFKC", s)


def step_base(s: str) -> str:
    """The locked baseline cleaner: URL -> ' URL ', @mention -> ' USER ', ws."""
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    return WS_RE.sub(" ", s).strip()


def step_repeat(s: str) -> str:
    """(2) Collapse runs of the same char beyond 2 repeats -> exactly 2."""
    return REPEAT_RE.sub(r"\1\1", s)


def step_emoji(s: str) -> str:
    """(3) Demojize to :bracketed_name: so word TF-IDF sees emoji as tokens."""
    import emoji as _emoji
    return _emoji.demojize(s, language="en")


STEPS = {"nfkc": step_nfkc, "repeat": step_repeat, "emoji": step_emoji}


def make_cleaner(toggles: tuple[str, ...]):
    """Build a clean_text function for a set of toggles.

    Order is fixed and documented: NFKC (if on) runs on the raw string BEFORE
    URL/mention masking; the baseline masking then runs; repeat-collapse and
    emoji-demojize run on the masked result.
    """
    def clean(text) -> str:
        s = "" if text is None else str(text)
        if not s:
            return ""
        if "nfkc" in toggles:
            s = step_nfkc(s)
        s = step_base(s)
        if "repeat" in toggles:
            s = step_repeat(s)
        if "emoji" in toggles:
            s = step_emoji(s)
        return s
    return clean


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_raw():
    train = pd.read_csv(ROOT / "train.csv", encoding="utf-8")
    test = pd.read_csv(ROOT / "test.csv", encoding="utf-8")
    for df in (train, test):
        df["text"] = df["text"].astype("string").fillna("").astype(str)
    return train, test


def build_folds(clean_texts, y) -> np.ndarray:
    keys = pd.Index([str(t).lower().strip() for t in clean_texts], dtype="object")
    groups = pd.factorize(keys, sort=False)[0]
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_ids = np.full(len(y), -1, dtype=int)
    for k, (_tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold_ids[va] = k
    assert (fold_ids >= 0).all()
    crossing = pd.DataFrame({"g": groups, "f": fold_ids}).groupby("g")["f"].nunique()
    n_cross = int((crossing > 1).sum())
    assert n_cross == 0, f"{n_cross} groups span >1 fold -> leakage"
    return fold_ids


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def eval_variant(name: str, toggles: tuple[str, ...], train, test, y, ids,
                 test_ids, verbose=True) -> dict:
    t0 = time.time()
    clean = make_cleaner(toggles)
    tr_clean = train["text"].map(clean).to_numpy()
    te_clean = test["text"].map(clean).to_numpy()
    fold_ids = build_folds(tr_clean, y)

    n = len(y)
    oof_text = np.zeros((n, 3))
    oof_id = np.zeros((n, 3))
    per_fold = []

    for k in range(N_SPLITS):
        va = np.flatnonzero(fold_ids == k)
        tr = np.flatnonzero(fold_ids != k)
        assert int(np.intersect1d(ids[va], ids[tr]).size) == 0, "id prior leak"

        word = TfidfVectorizer(**WORD_KW)
        word.fit(tr_clean[tr])
        char = TfidfVectorizer(**CHAR_KW)
        char.fit(tr_clean[tr])
        Xtr = sp.hstack([word.transform(tr_clean[tr]), char.transform(tr_clean[tr])],
                        format="csr").astype(np.float32)
        Xva = sp.hstack([word.transform(tr_clean[va]), char.transform(tr_clean[va])],
                        format="csr").astype(np.float32)
        clf = LogisticRegression(**LR_KW).fit(Xtr, y[tr])
        oof_text[va] = clf.predict_proba(Xva)
        oof_id[va] = H.id_prior(ids[va], ids[tr], y[tr], tag=f"{name}_f{k}")
        del word, char, Xtr, Xva, clf

        per_fold.append({
            "fold": k,
            "f1_text": round(H.macro_f1(y[va], oof_text[va]), 6),
            "f1_id": round(H.macro_f1(y[va], oof_id[va]), 6),
            "f1_blend": round(H.macro_f1(
                y[va], H.blend(oof_text[va], oof_id[va], ALPHA)), 6),
        })

    fold_text = [r["f1_text"] for r in per_fold]
    fold_blend = [r["f1_blend"] for r in per_fold]
    f1_text = H.macro_f1(y, oof_text)
    f1_blend = H.macro_f1(y, H.blend(oof_text, oof_id, ALPHA))
    pc_text = H.per_class_f1(y, oof_text.argmax(1))
    pc_blend = H.per_class_f1(y, H.blend(oof_text, oof_id, ALPHA).argmax(1))

    res = {
        "variant": name, "toggles": list(toggles),
        "n_unique_texts": int(len(set(tr_clean.tolist()))),
        "text_only_macro_f1": round(f1_text, 6),
        "text_only_delta_vs_anchor": round(f1_text - TEXT_ONLY_ANCHOR, 6),
        "text_only_per_class_f1": pc_text,
        "text_only_per_fold": fold_text,
        "text_only_fold_std": round(float(np.std(fold_text)), 6),
        "id_only_macro_f1": round(H.macro_f1(y, oof_id), 6),
        "blend040_macro_f1": round(f1_blend, 6),
        "blend040_delta_vs_baseline": round(f1_blend - BASELINE, 6),
        "blend040_per_class_f1": pc_blend,
        "blend040_per_fold": fold_blend,
        "blend040_fold_std": round(float(np.std(fold_blend)), 6),
        "seconds": round(time.time() - t0, 1),
    }
    np.savez_compressed(CACHE / f"oof_{name}.npz", oof_text=oof_text, oof_id=oof_id,
                        fold_ids=fold_ids, te_clean=te_clean, tr_clean=tr_clean,
                        test_ids=test_ids, ids=ids, y=y)
    if verbose:
        print(f"{name:16s} text={f1_text:.6f} ({f1_text - TEXT_ONLY_ANCHOR:+.6f}) "
              f"blend@0.40={f1_blend:.6f} ({f1_blend - BASELINE:+.6f}) "
              f"| textstd={res['text_only_fold_std']:.6f} "
              f"blendstd={res['blend040_fold_std']:.6f} "
              f"uniq={res['n_unique_texts']:,} | {res['seconds']}s", flush=True)
    return res


def main() -> int:
    t_start = time.time()
    H.banner("TIER 0 -- text-cleaning A/B variants (grouped leak-free OOF)")
    H.check_protected()

    train, test = load_raw()
    y = train["y"].to_numpy(dtype=int)
    ids = train["id"].to_numpy(dtype=np.int64)
    test_ids = test["id"].to_numpy(dtype=np.int64)
    print(f"train={train.shape} test={test.shape} | "
          f"class counts={{{', '.join(f'{c}: {int((y == c).sum())}' for c in H.CLASSES)}}}")

    # sanity: mean character length of the raw posts
    L = train["text"].str.len()
    print(f"raw text length: median={L.median():.0f} mean={L.mean():.0f} "
          f"p90={L.quantile(0.9):.0f} p99={L.quantile(0.99):.0f} max={L.max()}")
    n_emoji = int(train["text"].map(lambda s: any(ord(c) > 0x2190 for c in s)).sum())
    print(f"rows containing an emoji/symbol codepoint: {n_emoji:,}")

    variants = [
        ("base", ()),
        ("nfkc", ("nfkc",)),
        ("repeat", ("repeat",)),
        ("emoji", ("emoji",)),
        ("nfkc_repeat", ("nfkc", "repeat")),
        ("nfkc_emoji", ("nfkc", "emoji")),
        ("repeat_emoji", ("repeat", "emoji")),
        ("all3", ("nfkc", "repeat", "emoji")),
    ]

    results = []
    print("\n--- each change ALONE, then combined ---", flush=True)
    for name, tog in variants:
        try:
            results.append(eval_variant(name, tog, train, test, y, ids, test_ids))
        except MemoryError:
            print(f"{name}: MemoryError, skipped", flush=True)
        except Exception as exc:
            print(f"{name}: FAILED {type(exc).__name__}: {exc}", flush=True)

    df = pd.DataFrame(results)[["variant", "toggles", "text_only_macro_f1",
                               "text_only_delta_vs_anchor", "text_only_fold_std",
                               "blend040_macro_f1", "blend040_delta_vs_baseline",
                               "blend040_fold_std", "n_unique_texts", "seconds"]]
    print("\n" + df.to_string(index=False))

    if results:
        bt = max(results, key=lambda r: r["text_only_macro_f1"])
        bb = max(results, key=lambda r: r["blend040_macro_f1"])
        gate_t = bt["text_only_delta_vs_anchor"] >= NOISE_BAND
        gate_b = bb["blend040_delta_vs_baseline"] >= NOISE_BAND
        print(f"\nbest text-only : {bt['variant']} "
              f"{bt['text_only_macro_f1']:.6f} "
              f"({bt['text_only_delta_vs_anchor']:+.6f}) gate={'PASS' if gate_t else 'FAIL'}")
        print(f"best blend@0.40: {bb['variant']} {bb['blend040_macro_f1']:.6f} "
              f"({bb['blend040_delta_vs_baseline']:+.6f}) gate={'PASS' if gate_b else 'FAIL'}")
        print(f"noise band = +{NOISE_BAND} on pooled OOF Macro F1")
        print("STOP CONDITION: " + (
            "at least one variant clears the band -> carry forward to the full blend"
            if (gate_t or gate_b) else
            "no cleaning variant clears +0.005 -> cleaning is NOT the bottleneck, "
            "Tier 0 done, move to Tier 1"))

    payload = {
        "experiment": "t0_cleaning",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation": {"method": "StratifiedGroupKFold", "n_splits": N_SPLITS,
                       "shuffle": True, "random_state": SEED,
                       "group_key": "variant clean_text.lower().strip()",
                       "metric": "pooled OOF Macro F1"},
        "anchors": {"text_only": TEXT_ONLY_ANCHOR, "id_only": ID_ONLY_ANCHOR,
                    "blend_0.40_locked": BASELINE, "noise_band": NOISE_BAND},
        "results": results,
        "best_text_only": bt["variant"] if results else None,
        "best_blend": bb["variant"] if results else None,
        "gate_pass_text": bool(gate_t) if results else False,
        "gate_pass_blend": bool(gate_b) if results else False,
        "submission_written": False,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    (REPORTS / "t0_cleaning_metrics.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    H.check_protected()
    print(f"\n-> reports/t0_cleaning_metrics.json | runtime "
          f"{payload['runtime_seconds']}s | no submission written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

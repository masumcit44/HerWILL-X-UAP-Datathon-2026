"""Notebook 4 cells, part 3: class correction, override, ablation, CSV, summary."""
import nbformat as nbf

# importing part2 also imports part1; CELLS is a single shared list, so the
# cells accumulate in order part1 -> part2 -> part3
from make_repro_notebook_s4_part2 import CELLS, code  # noqa: F401

MD_CORR = """## 4. The frozen class correction, and how honest it is

Macro-F1 is dominated by class 2 (share 0.123). A per-fold additive logit offset
`[0, b1, b2]` is applied to `log p` before the argmax. The five fold offsets and
the single test offset are **frozen in `config_submission_4.json`** and were
each selected to maximise macro-F1 on the **other four folds** - never on the
fold they are applied to.

**Honest statement about the validation class.** This is
`STANDARD_OOF_STACKING_WITH_DEPENDENCY`, *not* `STRICTLY_OUTER_FOLD_SAFE`:

* the base-source OOF arrays are standard stacking inputs - for a training row of
  fold `j` the base model was trained on folds `!= j`, which **include fold `k`**,
  so the stacker's training features carry fold-`k` label information;
* the offsets are chosen on the other four folds' stacker OOF, which is
  single-level cross-fitting, not double-nested.

We therefore do **not** claim strict nesting, and the 0.7039 figure should be
read as an honest grouped-OOF stacking number with a known dependency, not as a
fully clean outer-fold estimate."""

MD_OVR = """## 5. Duplicate-text override

Identical to Submission 3, and copied verbatim from
`submission_reproducible_file_3.ipynb` (cells 5 and 15): where a cleaned test
text matches a train group in which **every** occurrence carries the same label,
that label is used. Groups with conflicting labels are skipped entirely. The
lookup is built from train rows only, so no test label ever influences it."""

MD_ABL = """## 6. ID ablation - the features that carry the score

Retraining the *same* frozen policy with the raw `id` dropped, and with the whole
ID-neighbour block dropped, shows how much of the score is ID structure. These
are single-seed diagnostics on the frozen configuration; they are not tuned."""


def md(s):
    return nbf.v4.new_markdown_cell(s)


CELLS.append(md(MD_CORR))
CELLS.append(code(r'''
# --- 7. frozen per-fold class correction + full OOF report -------------------
FOLD_OFFSETS = CFG["class_correction"]["fold_offsets"]
FINAL_OFFSET = CFG["class_correction"]["final_offset"]
log_oof = np.log(np.clip(oof_avg, 1e-300, None))
log_tst = np.log(np.clip(tst_avg, 1e-300, None))

corrected = log_oof.copy()
for k in range(N_SPLITS):
    v = folds == k
    corrected[v] = log_oof[v] + np.asarray(FOLD_OFFSETS[k], float)
pred_oof = corrected.argmax(1)

folds_scores = [macro_f1(y[folds == k], pred_oof[folds == k]) for k in range(N_SPLITS)]
cm = np.bincount(y * 3 + pred_oof, minlength=9).reshape(3, 3)
tp = np.diag(cm).astype(float)
prec = tp / np.maximum(cm.sum(0), 1)
rec = tp / np.maximum(cm.sum(1), 1)
per_class = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
POOLED = macro_f1(y, pred_oof)
ACC = float((pred_oof == y).mean())

print("fold offsets (frozen):", FOLD_OFFSETS)
print("test offset  (frozen):", FINAL_OFFSET)
print()
print("POOLED OOF macro-F1 : %.6f" % POOLED)
print("per-class F1        :", [round(float(v), 6) for v in per_class])
print("accuracy            : %.6f" % ACC)
print("fold scores         :", [round(v, 6) for v in folds_scores])
print("fold std            : %.6f" % float(np.std(folds_scores)))
print("worst fold          : %.6f (fold %d)" % (min(folds_scores),
                                                int(np.argmin(folds_scores))))
print("confusion matrix (rows = true, cols = predicted)")
print(cm)
print("prediction distribution:", np.bincount(pred_oof, minlength=3).tolist())
print("                       shares:", [round(float(v), 6) for v in
                                         np.bincount(pred_oof, minlength=3) / n_tr])
print("row sums of oof_avg  : min %.9f max %.9f" % (oof_avg.sum(1).min(),
                                                    oof_avg.sum(1).max()))
'''))
CELLS.append(md(MD_OVR))
CELLS.append(code(r'''
# --- 8. duplicate-text override (verbatim from Submission 3's notebook) ------
URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def clean_text(t):
    """URL -> ' URL ', @mention -> ' USER ', collapse whitespace, strip."""
    s = "" if t is None else str(t)
    s = URL_RE.sub(" URL ", s)
    s = MENTION_RE.sub(" USER ", s)
    return WS_RE.sub(" ", s).strip()


def unanimous_lookup(texts, labels):
    """cleaned text -> label for groups where EVERY occurrence agrees."""
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


clean_tr = train["text"].map(clean_text).to_numpy()
clean_te = test["text"].map(clean_text).to_numpy()
lookup, n_conflict = unanimous_lookup(clean_tr, y)
print("unanimous train groups: %d   conflicting groups skipped: %d"
      % (len(lookup), n_conflict))
'''))


CELLS.append(md(MD_ABL))
CELLS.append(code(r'''
# --- 9. ID ablation on the frozen policy (single seed, diagnostic only) ------
def ablate(drop, seed=42):
    cols = [c for c in Ftr.columns if c not in drop]
    params = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    oof = np.zeros((n_tr, 3))
    for k in range(N_SPLITS):
        trm = folds != k
        va = folds == k
        m = lgb.train(params, lgb.Dataset(Ftr.loc[trm, cols], y[trm], weight=WTS[trm]), ROUNDS)
        oof[va] = m.predict(Ftr.loc[va, cols])
    return macro_f1(y, oof.argmax(1))


print("ablation (seed 42, no class correction, frozen params)")
print("  full 60 features (seed 42) : %.6f" % SEED_SCORES["42"])
print("  drop raw id (rid)          : %.6f" % ablate(["rid"]))
print("  drop all 32 ID-neighbour   : %.6f" % ablate(NB_COLS))
'''))
CELLS.append(code(r'''
# --- 10. text-only ensemble and the per-source table (disclosure) -----------
text_only_cols = ["%s_p%d" % (n, k) for n in ("tfidf", "knn", "e5") for k in range(3)]
text_only = Ftr[text_only_cols].to_numpy().reshape(-1, 3, 3).mean(axis=1)
print("text-only ensemble (tfidf+knn+e5, equal weight) OOF: %.6f"
      % macro_f1(y, text_only.argmax(1)))
for nm, key in [("lexical tfidf", "tfidf"), ("text knn", "knn"),
                ("semantic e5", "e5"), ("plain id prior", "pid"),
                ("SID id prior", "sid"), ("xlmr id-head", "idh")]:
    z = np.load(SRC[key], allow_pickle=True)
    print("  %-16s %.6f" % (nm, macro_f1(y, z["oof"].argmax(1))))
print("  Submission 3 (text blend)  0.682311")
print("  Submission 4 (this)        %.6f" % POOLED)
'''))
CELLS.append(md("""## 7. Test prediction, validation and CSV

The test prediction is `argmax(log(mean test probs over the 3 seeds) + final
offset)`, then the duplicate override. The CSV is written by this notebook and
validated both before writing and after reading it back from disk."""))

CELLS.append(code(r'''
# --- 11. test prediction, pre-write validation, write, post-read validation --
def resolve_out_dir():
    """If this notebook lives inside a uap_submission/reproducible/ tree, write
    to that project's uap_submission/submission/. Otherwise (e.g. a clean-room
    copy) write to the notebook's own directory - never walk up into some other
    project's tree."""
    here = HERE.resolve()
    for base in [here] + list(here.parents):
        cand = base / "uap_submission" / "reproducible"
        if cand.is_dir() and (cand == here or cand in here.parents):
            return base / "uap_submission" / "submission"
    return here


OUT_DIR = resolve_out_dir()
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "submission_4.csv"

test_pred = (log_tst + np.asarray(FINAL_OFFSET, float)).argmax(1)
share_before = [float((test_pred == c).mean()) for c in CLASSES]
train_share = [float((y == c).mean()) for c in CLASSES]
print("test share before override:", [round(v, 4) for v in share_before])
print("train class share          :", [round(v, 4) for v in train_share])
drift_pp = [100.0 * abs(share_before[c] - train_share[c]) for c in CLASSES]
print("drift (pp)                 :", [round(v, 3) for v in drift_pp])
assert all(v <= 6.0 for v in drift_pp), "test class share drifted too far"

test_final, n_hits = apply_override(clean_te, test_pred, lookup)
changed = int((test_final != test_pred).sum())
share_after = [float((test_final == c).mean()) for c in CLASSES]
print("override: %d rows matched, %d labels changed" % (n_hits, changed))
print("test share after override :", [round(v, 4) for v in share_after])

sub = pd.DataFrame({"id": tids, "y_pred": test_final.astype(int)})
# ---- pre-write validation ----
assert list(sub.columns) == ["id", "y_pred"], list(sub.columns)
assert len(sub) == 11955, len(sub)
assert sub["id"].is_unique, "duplicate ids"
assert np.array_equal(sub["id"].to_numpy(), sample["id"].to_numpy()), (
    "id order differs from sample_submission")
assert not sub.isna().any().any(), "null values"
assert sub["y_pred"].dtype.kind in "iu", "y_pred must be integer"
assert set(sub["y_pred"].unique()) <= {0, 1, 2}, "unexpected label"
print("PRE-WRITE validation OK: columns, 11955 rows, unique ids, order matches")
print("sample_submission, no nulls, int dtype, labels in {0,1,2}")

sub.to_csv(OUT_CSV, index=False)

# ---- post-read validation ----
back = pd.read_csv(OUT_CSV)
assert list(back.columns) == ["id", "y_pred"], list(back.columns)
assert len(back) == 11955, len(back)
assert back["id"].is_unique
assert np.array_equal(back["id"].to_numpy(), sample["id"].to_numpy())
assert not back.isna().any().any()
assert back["y_pred"].dtype.kind in "iu"
assert set(back["y_pred"].unique()) <= {0, 1, 2}
assert np.array_equal(back["y_pred"].to_numpy(), sub["y_pred"].to_numpy())
counts = np.bincount(back["y_pred"], minlength=3)
print("POST-READ validation OK: file readable, not truncated, content identical")
print("class counts:", counts.tolist())
print("class shares:", [round(float(v), 4) for v in counts / len(back)])
print("csv bytes   :", OUT_CSV.stat().st_size)
print("csv sha256  :", sha256_of(OUT_CSV))
print("wrote", OUT_CSV)
'''))
CELLS.append(code(r'''
# --- 12. final summary ------------------------------------------------------
TARGET = CFG["reproduced"]["recorded_target"]
print("=" * 66)
print("Submission 4 - final summary (ID-DERIVED performance stacker)")
print("=" * 66)
print("validation classification :", CFG["validation_classification"])
print("reproduced pooled OOF     : %.6f" % POOLED)
print("recorded target           : %.6f" % TARGET)
print("difference                : %+.6f" % (POOLED - TARGET))
print("within 0.001 of target    :", abs(POOLED - TARGET) <= 0.001)
print("per-class F1              :", [round(float(v), 6) for v in per_class])
print("accuracy                  : %.6f" % ACC)
print("fold scores               :", [round(v, 6) for v in folds_scores])
print("fold std                  : %.6f" % float(np.std(folds_scores)))
print("seed scores (no corr.)    :", {k: round(v, 6) for k, v in SEED_SCORES.items()})
print("frozen class correction   : fold offsets", FOLD_OFFSETS)
print("                            test offset ", FINAL_OFFSET)
print("test class distribution   :", [round(v, 4) for v in share_after])
print("=" * 66)
'''))

nb = nbf.v4.new_notebook(cells=CELLS)
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
})
nbf.write(nb, "uap_submission/reproducible/submission_reproducible_file_4.ipynb")
print("wrote submission_reproducible_file_4.ipynb with %d cells" % len(CELLS))
print("markdown:", sum(1 for c in CELLS if c.cell_type == "markdown"),
      "code:", sum(1 for c in CELLS if c.cell_type == "code"))


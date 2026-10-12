# Notebook cell 15 - code: final weights, test blend, override, CSV
# --- 7. final weights, test probabilities, override, CSV ---------------------
# Final weights = element-wise median of the five cross-fitted fold weight
# vectors, renormalised to sum to 1. Using the median of honest fold weights
# (rather than one globally tuned vector) keeps the test-time weights honest.
final_w = np.array(CONFIG["final_weights"], dtype=float)
med = np.median(np.array([blend_w[str(k)]["w"] for k in range(N_SPLITS)]), axis=0)
med = med / med.sum()
print("median of fold weights :", np.round(med, 4).tolist())
print("final weights (config) :", np.round(final_w, 4).tolist(),
      "order", CONFIG["source_order"])
assert np.allclose(final_w, med, atol=1e-9), "bundled final weights != recomputed median"

test_sources = {"text": test_text, "sid": test_sid,
                "e5_base": e5_test, "xlmr_idhead": xlmr_test}
test_mix = sum(float(wi) * test_sources[k]
               for wi, k in zip(final_w, CONFIG["source_order"]))
test_mix = norm_rows(test_mix)
assert test_mix.shape == (11955, 3) and np.isfinite(test_mix).all()
print("test probs: shape {} max|rowsum-1| {:.2e}".format(
    test_mix.shape, float(np.abs(test_mix.sum(1) - 1).max())))
assert float(np.abs(test_mix.sum(1) - 1).max()) < 1e-6

# predicted class share must stay within 6 pp of the train share
train_share = [float((y == c).mean()) for c in CLASSES]
pred_pure = test_mix.argmax(1)
test_share = [float((pred_pure == c).mean()) for c in CLASSES]
drift_pp = [100.0 * abs(test_share[c] - train_share[c]) for c in CLASSES]
print("train class share", [round(v, 4) for v in train_share])
print("test  class share", [round(v, 4) for v in test_share])
print("drift (pp)       ", [round(v, 3) for v in drift_pp])
assert all(v <= 6.0 for v in drift_pp), "test class share drifted too far"


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


# lookup built from TRAIN rows only; conflicting groups skipped entirely
lookup, n_conflict = unanimous_lookup(clean_tr, y)
if USE_DUP_OVERRIDE:
    pred_final, n_hits = apply_override(clean_te, pred_pure, lookup)
else:
    pred_final, n_hits = pred_pure.copy(), 0
changed = int((pred_final != pred_pure).sum())
print("override (USE_DUP_OVERRIDE={}): {} rows matched, {} labels changed, "
      "{} conflicting groups skipped".format(USE_DUP_OVERRIDE, n_hits, changed, n_conflict))


def validate(df):
    assert list(df.columns) == ["id", "y_pred"], list(df.columns)
    assert len(df) == 11955, len(df)
    assert df["id"].is_unique, "duplicate ids"
    assert np.array_equal(df["id"].to_numpy(), sample["id"].to_numpy()), \
        "id order differs from sample_submission"
    assert not df.isna().any().any(), "null values"
    assert df["y_pred"].dtype.kind in "iu", "y_pred must be integer"
    assert set(df["y_pred"].unique()) <= {0, 1, 2}, "unexpected label"
    print("  validation OK: columns, 11955 rows, unique ids, order matches "
          "sample_submission, no nulls, int dtype, labels in {0,1,2}")


sub = pd.DataFrame({"id": test_ids, "y_pred": pred_final.astype(int)})
print("PRE-write validation:")
validate(sub)

# Resolve the output directory by walking UP from the notebook location until a
# directory that already contains uap_submission/ is found. This works whether the
# kernel's cwd is the project root or the notebook folder, and never nests a
# second uap_submission/ inside an existing one.
def resolve_out_dir():
    here = Path.cwd().resolve()
    for base in [here] + list(here.parents):
        cand = base / "uap_submission" / "submission"
        if cand.is_dir() and (base / "uap_submission" / "reproducible").is_dir():
            return cand
    for base in [here] + list(here.parents):
        if (base / "uap_submission").is_dir():
            return base / "uap_submission" / "submission"
    return here / "uap_submission" / "submission"


out_dir = resolve_out_dir()
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / "submission_3.csv"
sub.to_csv(csv_path, index=False)

print("POST-write validation (re-read from disk):")
validate(pd.read_csv(csv_path))
print("wrote", csv_path)
print("CSV SHA256:", hashlib.sha256(csv_path.read_bytes()).hexdigest())

prev = None
for _base in [Path.cwd()] + list(Path.cwd().parents):
    _cand = _base / "uap_submission" / "submission" / "submission_2.csv"
    if _cand.exists():
        prev = _cand
        break
if prev is not None:
    prev_df = pd.read_csv(prev)
    n_diff = int((prev_df["y_pred"].to_numpy() != sub["y_pred"].to_numpy()).sum())
    print("labels differing from Submission 2: {} / {} = {:.2f}%".format(
        n_diff, len(sub), 100.0 * n_diff / len(sub)))

"""Finalize the Submission 4 reproduction: average 3 seeds, apply the reported
per-fold logit offsets, independently re-search offsets, apply the duplicate
override (functions extracted VERBATIM from submission_reproducible_file_3.ipynb)
and write submissions/submission_4_repro.csv.
"""
import ast
import json
import os

import nbformat
import numpy as np
import pandas as pd

SEEDS = [42, 7, 2026]
CLASSES = [0, 1, 2]
REPORT = json.load(open("../claud/sub4_code/final_report.json"))


def mf1(yy, pr):
    cm = np.bincount(np.asarray(yy) * 3 + np.asarray(pr), minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    P = tp / np.maximum(cm.sum(0), 1)
    R = tp / np.maximum(cm.sum(1), 1)
    return float((2 * P * R / np.maximum(P + R, 1e-12)).mean())


def f1s(yy, pr):
    cm = np.bincount(np.asarray(yy) * 3 + np.asarray(pr), minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    P = tp / np.maximum(cm.sum(0), 1)
    R = tp / np.maximum(cm.sum(1), 1)
    return (2 * P * R / np.maximum(P + R, 1e-12))


tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")
y = tr["y"].to_numpy(int)
folds = np.load("final_sources/fold_ids.npy").astype(np.int64)

# ---- (a) average the 3 seeds -------------------------------------------------
oofs, tests = [], []
for s in SEEDS:
    z = np.load("stack_all_400_s%d.npz" % s)
    assert np.array_equal(z["fold_ids"], folds), "seed %s fold mismatch" % s
    oofs.append(z["oof"])
    tests.append(z["test"])
    print("seed %-5s pooled-no-offset %.6f  folds %s" % (
        s, mf1(y, z["oof"].argmax(1)),
        [round(mf1(y[folds == k], z["oof"][folds == k].argmax(1)), 4) for k in range(5)]))
oof = np.mean(oofs, axis=0)
test = np.mean(tests, axis=0)
lo = np.log(np.clip(oof, 1e-300, None))
lt = np.log(np.clip(test, 1e-300, None))

# ---- (b) reported offsets ---------------------------------------------------
rep_off = REPORT["fold_offsets"]


def apply_offsets(lp, offs):
    p = lp.copy()
    for k in range(5):
        v = folds == k
        p[v] = lp[v] + np.asarray(offs[k], float)
    return p.argmax(1)


pred_rep = apply_offsets(lo, rep_off)
fs_rep = [mf1(y[folds == k], pred_rep[folds == k]) for k in range(5)]
pc_rep = f1s(y, pred_rep)
cm = np.bincount(y * 3 + pred_rep, minlength=9).reshape(3, 3)
pooled = mf1(y, pred_rep)
print("\n== reported offsets ==")
print("pooled %.6f (target %.6f, d=%+.6f)" % (
    pooled, REPORT["honest_oof"], pooled - REPORT["honest_oof"]))
print("folds  ", [round(v, 6) for v in fs_rep])
print("target ", REPORT["folds"])
print("percls ", [round(float(v), 6) for v in pc_rep], "target", REPORT["per_class"])
print("confusion\n", cm)

# ---- (c) independent re-search on the OTHER four folds ----------------------
B1 = np.round(np.arange(-0.3, 0.3001, 0.05), 2)
B2 = np.round(np.arange(-1.0, 0.2001, 0.05), 2)
searched, best_scores = [], []
for k in range(5):
    oth = folds != k
    lp = lo[oth]
    yy = y[oth]
    best, bo = -1.0, (0.0, 0.0, 0.0)
    for b1 in B1:
        for b2 in B2:
            sc = mf1(yy, (lp + np.array([0.0, b1, b2])).argmax(1))
            if sc > best:
                best, bo = sc, (0.0, float(b1), float(b2))
    searched.append([bo[0], bo[1], bo[2]])
    best_scores.append(best)
final_med = [0.0] + [float(np.median([s[j] for s in searched])) for j in (1, 2)]
pred_srch = apply_offsets(lo, searched)
pooled_srch = mf1(y, pred_srch)
print("\n== independent offset re-search (selected on the other 4 folds) ==")
for k in range(5):
    print("fold %d  reported %-20s searched %-20s score(other4) %.6f" % (
        k, rep_off[k], searched[k], best_scores[k]))
print("median of searched =", final_med, " report final_offset =", REPORT["final_offset"])
print("pooled with searched offsets %.6f (vs reported %.6f)" % (pooled_srch, pooled))

# ---- (d) test prediction -----------------------------------------------------
test_pred = (lt + np.asarray(REPORT["final_offset"], float)).argmax(1)
share_before = [float((test_pred == c).mean()) for c in CLASSES]
print("\ntest share before override", [round(v, 4) for v in share_before],
      "target", REPORT["test_share"])


# ---- (e) duplicate override, functions VERBATIM from notebook 3 -------------
nb = nbformat.read(
    "../uap_submission/reproducible/submission_reproducible_file_3.ipynb", as_version=4)


def grab(fname, cell_idx):
    src = "".join(nb.cells[cell_idx].source)
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fname][0]
    return ast.get_source_segment(src, fn)


ns = {"np": np, "pd": pd, "CLASSES": CLASSES}
# cell 5 defines the regexes that clean_text closes over; pull them verbatim too
src5 = "".join(nb.cells[5].source)
consts5 = "\n".join(
    ast.get_source_segment(src5, n) for n in ast.parse(src5).body
    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
    and isinstance(n.value.func, ast.Attribute)
    and n.value.func.attr == "compile")
exec("import re\n" + consts5, ns)
exec("import re\n" + grab("clean_text", 5), ns)
exec(grab("unanimous_lookup", 15), ns)
exec(grab("apply_override", 15), ns)

tr2 = tr.copy()
te2 = te.copy()
tr2["text"] = tr2["text"].astype("string").fillna("").astype(str)
te2["text"] = te2["text"].astype("string").fillna("").astype(str)
clean_tr = tr2["text"].map(ns["clean_text"]).to_numpy()
clean_te = te2["text"].map(ns["clean_text"]).to_numpy()
lookup, n_conflict = ns["unanimous_lookup"](clean_tr, y)
test_final, n_hits = ns["apply_override"](clean_te, test_pred, lookup)
changed = int((test_final != test_pred).sum())
print("override: %d rows matched, %d labels changed, %d conflicting groups skipped"
      % (n_hits, changed, n_conflict))
share_after = [float((test_final == c).mean()) for c in CLASSES]
print("test share after override ", [round(v, 4) for v in share_after])

# ---- (f) write + validate ----------------------------------------------------
os.makedirs("../submissions", exist_ok=True)
out = pd.DataFrame({"id": te["id"].to_numpy(), "y_pred": test_final.astype(int)})
assert list(out.columns) == ["id", "y_pred"]
assert len(out) == 11955
assert out["id"].is_unique
assert np.array_equal(out["id"].to_numpy(), sample["id"].to_numpy())
assert not out.isna().any().any()
assert out["y_pred"].dtype.kind in "iu"
assert set(out["y_pred"].unique()) <= {0, 1, 2}
out.to_csv("../submissions/submission_4_repro.csv", index=False)
print("wrote submissions/submission_4_repro.csv  (validation OK)")

ref = "../claud/sub4_code/submission_4_stacker.csv"
if not os.path.exists(ref):
    ref = "submission_4_stacker.csv"
ag = None
if os.path.exists(ref):
    r = pd.read_csv(ref)
    m = r.merge(out, on="id", suffixes=("_ref", "_new"))
    ag = 100.0 * (m["y_pred_ref"].to_numpy() == m["y_pred_new"].to_numpy()).mean()
    print("agreement with %s: %.3f%%  (n=%d)" % (ref, ag, len(m)))
    print("reference class share", [round(float((r["y_pred"] == c).mean()), 4) for c in CLASSES])
else:
    print("NOTE: submission_4_stacker.csv not supplied -> agreement not computed")

res = dict(pooled_reported=pooled, target=REPORT["honest_oof"],
           delta=pooled - REPORT["honest_oof"],
           folds=fs_rep, folds_target=REPORT["folds"],
           per_class=[float(v) for v in pc_rep], per_class_target=REPORT["per_class"],
           confusion=cm.tolist(), std=float(np.std(fs_rep)),
           offsets_reported=rep_off, offsets_searched=searched,
           median_searched=final_med, final_offset=REPORT["final_offset"],
           pooled_with_searched=pooled_srch,
           test_share_before=share_before, test_share_after=share_after,
           test_share_target=REPORT["test_share"],
           override_matched=n_hits, override_changed=changed,
           override_target=[REPORT["override_matched"], REPORT["override_changed"]],
           agreement_pct=ag,
           seeds=dict((str(s), mf1(y, o.argmax(1))) for s, o in zip(SEEDS, oofs)))
json.dump(res, open("repro_s4_results.json", "w"), indent=2)
print("wrote repro_s4/repro_s4_results.json")

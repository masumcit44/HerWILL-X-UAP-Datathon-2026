"""15-minute performance-track sprint: fixed E5-SID beta12/k180 challenger."""
import os, json, time, hashlib
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

T0 = time.time()
BUDGET = 15 * 60.0
OUT_P, OUT_R = "predictions/performance_v7", "reports/performance_v7"
for d in (OUT_P, OUT_R):
    os.makedirs(d, exist_ok=True)
CLS = [0, 1, 2]
CFG = dict(KW=120, k=180, beta=12.0, gamma=1.0)


def el():
    return time.time() - T0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def atomic_save(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def mf1(y, pr):
    return float(f1_score(y, np.asarray(pr).argmax(1), average="macro",
                          labels=CLS, zero_division=0))


def pcf1(y, pr):
    return [round(float(v), 6) for v in f1_score(y, np.asarray(pr).argmax(1),
            average=None, labels=CLS, zero_division=0)]


def norm(P):
    return P / P.sum(axis=1, keepdims=True)


def integrity(P, n):
    p = []
    if P.shape != (n, 3):
        p.append("shape %s" % (P.shape,))
    if not np.isfinite(P).all():
        p.append("nonfinite")
    if P.min() < -1e-6 or P.max() > 1 + 1e-6:
        p.append("range")
    rs = float(np.abs(P.sum(1) - 1).max())
    if rs > 1e-4:
        p.append("rowsum %.1e" % rs)
    return (not p), p, rs


folds = np.load("final_sources/fold_ids.npy").astype(np.int64)
FOLD_INT = hashlib.sha256(np.ascontiguousarray(folds, np.int64).tobytes()).hexdigest()
assert FOLD_INT == "cc4fdd169110aeee6d3d5566095234461041b1a9290e2e916ce850c31d27a7ca"
assert np.bincount(folds, minlength=5).tolist() == [9543, 9546, 9571, 9581, 9576]
z = np.load("cache/v6/step123.npz")
y = z["y"].astype(int); ids = z["ids"].astype(np.int64)
tids = z["test_ids"].astype(np.int64)
assert np.array_equal(z["fold_ids"].astype(np.int64), folds), "fold mismatch"
REF = dict(text=z["oof_text"], sid=z["oof_sid"], e5=z["e5_oof"], xlmr=z["xlmr_oof"])
REF_T = dict(text=z["test_text"], sid=z["test_sid"], e5=z["e5_test"], xlmr=z["xlmr_test"])
audit = {}
for k in REF:
    ok, why, rs = integrity(REF[k].astype(np.float64), 47817)
    okt, whyt, rst = integrity(REF_T[k].astype(np.float64), 11955)
    audit[k] = {"oof_shape": list(REF[k].shape), "test_shape": list(REF_T[k].shape),
                "oof_ok": ok, "oof_problems": why, "oof_rowsum_dev": rs,
                "test_ok": okt, "test_problems": whyt, "test_rowsum_dev": rst,
                "standalone_oof": round(mf1(y, REF[k]), 6)}
    assert ok and okt, (k, why, whyt)
print("SOURCES OK:", {k: v["standalone_oof"] for k, v in audit.items()}, flush=True)
REF_ORDER = ["text", "sid", "e5", "xlmr"]


def simplex(n, t):
    if n == 1:
        yield (t,)
        return
    for i in range(t + 1):
        for r in simplex(n - 1, t - i):
            yield (i,) + r


def grid_of(step):
    T = int(round(1.0 / step))
    return [np.array(c, float) / T for c in simplex(4, T)]


def honest(S, steps=(0.10, 0.05)):
    out = np.zeros((len(y), 3)); chosen = {}; fs = []
    for k in range(5):
        va, tr = folds == k, folds != k
        Sr, yr = S[:, tr], y[tr]
        w = None
        for st in steps:
            best = (-1.0, None)
            for cand in grid_of(st):
                sc = mf1(yr, norm(np.tensordot(cand, Sr, 1)))
                if sc > best[0]:
                    best = (sc, cand)
            w = best[1]
        chosen[k] = [float(v) for v in w]
        out[va] = norm(np.tensordot(w, S[:, va], 1))
        fs.append(mf1(y[va], out[va]))
    opt = (-1.0, None)
    for st in steps:
        for cand in grid_of(st):
            sc = mf1(y, norm(np.tensordot(cand, S, 1)))
            if sc > opt[0]:
                opt = (sc, [float(v) for v in cand])
    return out, mf1(y, out), fs, chosen, opt


S_ref = np.stack([REF[k] for k in REF_ORDER])
ref_oof, ref_f1, ref_fs, ref_w, ref_opt = honest(S_ref)
print("SUB3 REPRODUCED = %.6f (target 0.682311, diff %.2e)"
      % (ref_f1, abs(ref_f1 - 0.682311)), flush=True)
print("  per-class", pcf1(y, ref_oof), "folds", [round(v, 6) for v in ref_fs],
      "std %.6f" % np.std(ref_fs), flush=True)
if abs(ref_f1 - 0.682311) > 1e-4:
    print("STOP: Submission 3 not reproducible", flush=True)
    raise SystemExit(1)

# ---------------- Phase 3: fixed E5-SID beta12/k180
print("\nPHASE 3 - FIXED E5-SID beta12/k180 [t=%.1fs]" % el(), flush=True)
SID_PATH = "cache/v6/sidE5_beta12.0_k180_oof.npy"
sid_e5_oof = np.load(SID_PATH).astype(np.float64)
ok, why, rs = integrity(sid_e5_oof, 47817)
sid_standalone = mf1(y, sid_e5_oof)
audit["sidE5_fixed"] = {"oof_path": SID_PATH, "oof_sha256": sha(SID_PATH), "cfg": CFG,
                        "oof_ok": ok, "oof_problems": why, "oof_rowsum_dev": rs,
                        "standalone_oof": round(sid_standalone, 6),
                        "expected_standalone": 0.628273, "test": "reconstructed below"}
print("  standalone OOF %.6f (expected ~0.628273) | ok=%s %s"
      % (sid_standalone, ok, why), flush=True)


def sid_prior_e5(q_ids, Xq, pool_rows, ids_, y_, Xpool, KW=120, k=180, beta=12.0,
                 gamma=1.0, chunk=200):
    o = np.argsort(ids_[pool_rows], kind="stable")
    pid = ids_[pool_rows][o]; prow = pool_rows[o]
    assert len(np.intersect1d(q_ids, pid)) == 0, "query ids inside pool"
    out = np.zeros((len(q_ids), 3))
    for s in range(0, len(q_ids), chunk):
        qi = q_ids[s:s + chunk]; xq = np.asarray(Xq[s:s + chunk])
        pos = np.searchsorted(pid, qi)
        cand = pos[:, None] + np.arange(-KW, KW)[None, :]
        valid = (cand >= 0) & (cand < len(pid))
        cc = np.clip(cand, 0, len(pid) - 1)
        d = np.abs(pid[cc] - qi[:, None]).astype(float); d[~valid] = np.inf
        order = np.argsort(d, axis=1, kind="stable")[:, :k]
        dsel = np.take_along_axis(d, order, 1)
        nrow = prow[np.take_along_axis(cc, order, 1)]
        qrep = np.repeat(np.arange(xq.shape[0]), nrow.shape[1])
        sim = np.einsum("ij,ij->i", xq[qrep], Xpool[nrow.ravel()]).reshape(nrow.shape)
        w = np.where(np.isfinite(dsel),
                     1.0 / ((np.where(np.isfinite(dsel), dsel, 1.0) + 1e-6) ** gamma),
                     0.0) * np.exp(beta * sim)
        yy = y_[nrow]
        P = np.stack([(w * (yy == c)).sum(1) for c in range(3)], 1)
        tot = P.sum(1, keepdims=True)
        out[s:s + chunk] = np.where(tot > 0, P / np.where(tot > 0, tot, 1), 1 / 3)
    return out


Etr = normalize(np.load("cache/v4/emb_e5base_train.npy")).astype(np.float32)
Ete = normalize(np.load("cache/v4/emb_e5base_test.npy")).astype(np.float32)
assert Etr.shape[0] == len(y) and Ete.shape[0] == len(tids)
sid_e5_test = sid_prior_e5(tids, Ete, np.arange(len(y)), ids, y, Etr, **CFG)
okt, whyt, rst = integrity(sid_e5_test, 11955)
audit["sidE5_fixed"]["test"] = {"reconstructed": True, "ok": okt, "problems": whyt,
                                "rowsum_dev": rst}
atomic_save(os.path.join(OUT_P, "_tmp_sidE5_test_probs.npy"), sid_e5_test.astype(np.float32))
print("  test reconstructed ok=%s %s rowsum %.1e [t=%.1fs]" % (okt, whyt, rst, el()), flush=True)
SID_ALIGNED = ok and okt and abs(sid_standalone - 0.628273) < 0.005
print("  SID ALIGNED/CONSISTENT =", SID_ALIGNED, flush=True)
print("Phase 3 done [t=%.1fs]" % el(), flush=True)

# ---------------- Phase 4: challenger honest blend
print("\nPHASE 4 - HONEST CHALLENGER BLEND [t=%.1fs]" % el(), flush=True)
CH = dict(text=REF["text"], sid=sid_e5_oof, e5=REF["e5"], xlmr=REF["xlmr"])
CH_ORDER = ["text", "sid", "e5", "xlmr"]
S_ch = np.stack([CH[k] for k in CH_ORDER])
ch_oof, ch_f1, ch_fs, ch_w, ch_opt = honest(S_ch, (0.10, 0.05))
ch_pc, ch_acc = pcf1(y, ch_oof), float(accuracy_score(y, ch_oof.argmax(1)))
ch_std, ch_worst = float(np.std(ch_fs)), float(np.min(ch_fs))
print("  CHALLENGER HONEST OOF = %.6f  (expected ~0.684811)" % ch_f1, flush=True)
print("  per-class", ch_pc, "acc %.6f" % ch_acc, flush=True)
print("  folds", [round(v, 6) for v in ch_fs], "std %.6f worst %.6f" % (ch_std, ch_worst), flush=True)

# ---------------- Phase 5: adoption gates
print("\nPHASE 5 - ADOPTION GATES [t=%.1fs]" % el(), flush=True)
SUB3 = 0.682311
ref_std, ref_worst, ref_pc6, ref_fs6 = float(np.std(ref_fs)), float(min(ref_fs)), pcf1(y, ref_oof), ref_fs
g1 = (ch_f1 - SUB3) >= 0.0015
g2 = sum(1 for a, b in zip(ch_fs, ref_fs6) if a >= b - 0.0005) >= 4
g3 = all((ch_pc[i] - ref_pc6[i]) >= -0.005 for i in range(3))
g4 = ch_std <= 1.5 * ref_std
g5 = ch_worst >= (ref_worst - 0.001)
med = np.median(np.array([ch_w[k] for k in range(5)]), axis=0); med = med / med.sum()
CH_T = {"text": REF_T["text"], "sid": sid_e5_test, "e5": REF_T["e5"], "xlmr": REF_T["xlmr"]}
p1_test = norm(sum(float(w) * CH_T[n] for w, n in zip(med, CH_ORDER)))
ok6, why6, rs6 = integrity(p1_test, 11955)
g6 = ok6
train_share = [float((y == c).mean()) for c in CLS]
p1_share = [float((p1_test.argmax(1) == c).mean()) for c in CLS]
drift = [100 * abs(p1_share[i] - train_share[i]) for i in range(3)]
g7 = all(v <= 6.0 for v in drift)
GATES = dict(G1_delta_ge_0_0015=bool(g1), G2_4folds_not_worse=bool(g2),
             G3_no_class_drop_gt_0_005=bool(g3), G4_std_le_1_5x=bool(g4),
             G5_worst_fold=bool(g5), G6_test_integrity=bool(g6),
             G7_test_share_within_6pp=bool(g7))
PASS = all(GATES.values())
print("  G1 %+.6f >= +0.0015 -> %s" % (ch_f1 - SUB3, g1), flush=True)
print("  G2 folds not worse: %d/5 -> %s" % (sum(1 for a, b in zip(ch_fs, ref_fs6) if a >= b - 0.0005), g2), flush=True)
print("  G3 class deltas %s -> %s" % ([round(ch_pc[i] - ref_pc6[i], 6) for i in range(3)], g3), flush=True)
print("  G4 std %.6f vs 1.5x %.6f -> %s" % (ch_std, 1.5 * ref_std, g4), flush=True)
print("  G5 worst %.6f vs ref %.6f -> %s" % (ch_worst, ref_worst, g5), flush=True)
print("  G6 test integrity %s %s -> %s" % (ok6, why6, g6), flush=True)
print("  G7 share %s drift %s -> %s" % ([round(v, 4) for v in p1_share], [round(v, 2) for v in drift], g7), flush=True)
print("  FINAL WEIGHTS", {n: round(float(v), 4) for n, v in zip(CH_ORDER, med)}, flush=True)
print("  ALL GATES PASS =", PASS, flush=True)
print("Phase 5 done [t=%.1fs]" % el(), flush=True)

# ---------------- Phase 6: packages
print("\nPHASE 6 - PACKAGES [t=%.1fs]" % el(), flush=True)
P2_OOF = P2_T = None
p2_f1 = None
dis3 = dis12 = None
if PASS:
    atomic_save(os.path.join(OUT_P, "candidate_P1_oof_probs.npy"), ch_oof.astype(np.float32))
    atomic_save(os.path.join(OUT_P, "CANDIDATE_NOT_FOR_SUBMISSION_P1_test_probs.npy"),
                p1_test.astype(np.float32))
    zero_n = [n for i, n in enumerate(CH_ORDER) if zero[n] >= 3]
    if zero_n:
        keep = [n for n in CH_ORDER if n not in zero_n]
        p2_oof, p2_f1, p2_fs, p2_w, _ = honest(S_ch[[CH_ORDER.index(n) for n in keep]], (0.10, 0.05))
        med2 = np.median(np.array([p2_w[k] for k in range(5)]), axis=0); med2 /= med2.sum()
        P2_OOF = p2_oof
        P2_T = norm(sum(float(w) * CH_T[n] for w, n in zip(med2, keep)))
        print("  P2 dropped %s -> honest %.6f" % (zero_n, p2_f1), flush=True)
    else:
        P2_OOF, P2_T, p2_f1 = ch_oof, p1_test, ch_f1
        print("  P2 = P1 (no source zero in >=3 folds)", flush=True)
    atomic_save(os.path.join(OUT_P, "candidate_P2_oof_probs.npy"), P2_OOF.astype(np.float32))
    atomic_save(os.path.join(OUT_P, "CANDIDATE_NOT_FOR_SUBMISSION_P2_test_probs.npy"),
                P2_T.astype(np.float32))
    sub3 = np.load("uap_submission/submission/submission_3.csv")["y_pred"].to_numpy()

# ---------------- reports
prot_before = json.load(open("cache/_perf_v7_protected_before.json"))
prot_after = {k: sha(k) for k in prot_before}
prot_ok = all(prot_after[k] == v for k, v in prot_before.items())
metrics = {
    "experiment": "perf_v7_fixed_e5sid_k180", "fold_set": "B (recovered Colab)",
    "fold_int64_sha256": FOLD_INT,
    "fold_sizes": np.bincount(folds, minlength=5).tolist(),
    "fixed_e5sid_config": CFG,
    "fixed_e5sid_standalone_oof": round(sid_standalone, 6),
    "sid_aligned_and_consistent": bool(SID_ALIGNED),
    "reference": {"honest_oof": round(ref_f1, 6), "per_class_f1": ref_pc6,
                  "fold_scores": [round(v, 6) for v in ref_fs6],
                  "fold_std": round(ref_std, 6), "worst_fold": round(ref_worst, 6),
                  "order": REF_ORDER, "fold_weights": ref_w,
                  "same_oof_OPTIMISTIC": round(ref_opt[0], 6)},
    "challenger": {"honest_oof": round(ch_f1, 6), "per_class_f1": ch_pc,
                   "accuracy": round(ch_acc, 6),
                   "fold_scores": [round(v, 6) for v in ch_fs],
                   "fold_std": round(ch_std, 6), "worst_fold": round(ch_worst, 6),
                   "order": CH_ORDER, "fold_weights": ch_w, "zero_weight_folds": zero,
                   "same_oof_OPTIMISTIC": round(ch_opt[0], 6),
                   "confusion": confusion_matrix(y, ch_oof.argmax(1), labels=CLS).tolist()},
    "delta_vs_submission_3": round(ch_f1 - SUB3, 6),
    "gates": GATES, "all_gates_pass": bool(PASS),
    "final_weights_P1": {n: float(v) for n, v in zip(CH_ORDER, med)},
    "p2_honest_oof": (round(p2_f1, 6) if p2_f1 is not None else None),
    "train_class_share": [round(v, 4) for v in train_share],
    "p1_test_class_share": [round(v, 4) for v in p1_share],
    "p1_test_share_drift_pp": [round(v, 3) for v in drift],
    "p1_vs_submission3_test_disagreement": (round(dis3, 6) if dis3 is not None else None),
    "p1_vs_p2_test_disagreement": (round(dis12, 6) if dis12 is not None else None),
    "source_audit": audit, "elapsed_seconds": round(el(), 1),
    "submission_generated": False, "protected_files_unchanged": bool(prot_ok),
}
with open(os.path.join(OUT_R, "fixed_e5sid_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
with open(os.path.join(OUT_R, "source_audit.json"), "w") as f:
    json.dump({"audit": audit, "fold_int64_sha256": FOLD_INT,
               "fixed_e5sid_config": CFG, "sources": {
                   "step123_npz": sha("cache/v6/step123.npz"),
                   "e5base_train": sha("cache/v4/emb_e5base_train.npy"),
                   "e5base_test": sha("cache/v4/emb_e5base_test.npy"),
                   "emb_e5_base_final": sha("final_sources/emb_e5-base_final.npz"),
                   "xlmr_idhead_final": sha("final_sources/xlmr_base_idhead_final.npz")}},
              f, indent=2)
st = "ADOPTED (all gates passed)" if PASS else "REJECTED (gate failure)"
md = ["# Performance sprint - fixed E5-SID beta12/k180", "",
      "Track A only. Fold set B, honest cross-fitted grouped OOF macro-F1.", "",
      "## Result", "",
      "- Submission 3 reference reproduced: **%.6f**" % ref_f1,
      "- Fixed E5-SID standalone OOF: **%.6f**" % sid_standalone,
      "- Challenger honest OOF: **%.6f** (delta %+.6f vs 0.682311)"
      % (ch_f1, ch_f1 - SUB3),
      "- Per-class F1: %s | accuracy %.6f" % (ch_pc, ch_acc),
      "- Fold scores: %s | std %.6f | worst %.6f"
      % ([round(v, 6) for v in ch_fs], ch_std, ch_worst),
      "- Same-OOF OPTIMISTIC (not a result): %.6f" % ch_opt[0],
      "- Final weights: %s" % {n: round(float(v), 4) for n, v in zip(CH_ORDER, med)},
      "- **Status: %s**" % st, "", "## Gates", ""]
for k, v in GATES.items():
    md.append("- `%s`: %s" % (k, "PASS" if v else "FAIL"))
md += ["", "## Notes", "",
       "- The six-configuration E5-SID grid was NOT rerun and no other",
       "  configuration was selected. beta=12/k=180/KW=120/gamma=1 was fixed in",
       "  advance; the stored OOF array was audited and only the missing test",
       "  array was reconstructed from cached E5 Base embeddings.",
       "- Protected files unchanged: %s" % prot_ok,
       "- submission_generated: false", ""]
with open(os.path.join(OUT_R, "summary.md"), "w") as f:
    f.write("\n".join(md))
print("\nELAPSED %.1fs" % el(), flush=True)
print("RESULT_JSON=" + json.dumps(dict(
    ref=round(ref_f1, 6), ch=round(ch_f1, 6), delta=round(ch_f1 - SUB3, 6),
    sid=round(sid_standalone, 6), pass_=bool(PASS), std=round(ch_std, 6),
    worst=round(ch_worst, 6), refworst=round(ref_worst, 6), opt=round(ch_opt[0], 6),
    refopt=round(ref_opt[0], 6), pc=ch_pc, acc=round(ch_acc, 6),
    folds=[round(v, 6) for v in ch_fs], reffolds=[round(v, 6) for v in ref_fs6],
    refpc=ref_pc6, w={k: [round(v, 3) for v in ch_w[k]] for k in range(5)},
    finalw={n: round(float(v), 4) for n, v in zip(CH_ORDER, med)},
    gates=GATES, share=[round(v, 4) for v in p1_share], prot=bool(prot_ok),
    dis3=dis3, dis12=dis12, p2=p2_f1, sidok=bool(SID_ALIGNED))), flush=True)

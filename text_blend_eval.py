"""Honest text-only blending, ablation and error analysis on the locked Colab fold set (fold set B).

Sources are passed as FAMILY:NAME=PATH, where FAMILY is one of
    lexical   (TF-IDF, text-kNN)            semantic (frozen embeddings, e.g. E5)
    finetuned (text-only transformers)      id       (plain ID prior, SID, ID-head models)
Every npz must hold `oof` (N_train,3) and `test` (N_test,3); optional `coverage_train`,
`coverage_test` (bool) for subset models such as the Bangla-script BanglaBERT run, whose
uncovered rows are NaN.

Hard rules enforced in code:
  * `id`-family sources and anything whose name/meta suggests ID use are NEVER allowed into the
    text-only candidates (T1/T2/T3). They are used only for the clearly labelled ablation rows.
  * Weights (and optional class offsets) for each outer fold are chosen on the other four folds.
  * Same-OOF optima are printed only with the label OPTIMISTIC.
  * No CSV is written. Outputs are probability arrays named CANDIDATE_NOT_FOR_SUBMISSION_*.

Example:
  python text_blend_eval.py --data-dir . --folds final_sources/fold_ids.npy --out reports/v7 \
     --src lexical:text_tfidf=cache/v7_text/text_tfidf_final.npz \
     --src lexical:text_knn=cache/v7_text/text_knn_final.npz \
     --src semantic:emb_e5_base=final_sources/emb_e5-base_final.npz \
     --src finetuned:xlmr_text=colab_v7/runs/xlmr_text_final.npz \
     --src finetuned:banglabert_bn=colab_v7/runs/banglabert_bn_final.npz \
     --src id:plain_id=cache/v6/plain_id.npz \
     --src id:xlmr_idhead=final_sources/xlmr_base_idhead_final.npz
"""
import argparse, hashlib, itertools, json, re, time
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_FOLD_INT64 = "cc4fdd169110aeee6d3d5566095234461041b1a9290e2e916ce850c31d27a7ca"
SUB3_HONEST = 0.682311
FAMILIES = ("lexical", "semantic", "finetuned", "id")
ID_NAME_RE = re.compile(r"(^|[_\-])(id|sid)([_\-]|$)|idhead|id_prior|idprior", re.I)
N_SPLITS = 5


# ---------------------------------------------------------------- metrics
def conf(y, p):
    return np.bincount(y * 3 + p, minlength=9).reshape(3, 3)


def f1s(y, p):
    cm = conf(y, p)
    tp = np.diag(cm).astype(float)
    pr, rc = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    return 2 * pr * rc / np.maximum(pr + rc, 1e-12)


def mf1(y, p):
    return float(f1s(y, p).mean())


# ---------------------------------------------------------------- loading + checks
def load_sources(specs, n_tr, n_te, folds):
    S = {}
    for spec in specs:
        fam, rest = spec.split(":", 1)
        name, path = rest.split("=", 1)
        assert fam in FAMILIES, f"unknown family {fam}"
        z = np.load(path, allow_pickle=False)
        oof, test = z["oof"].astype(np.float64), z["test"].astype(np.float64)
        cov_tr = z["coverage_train"].astype(bool) if "coverage_train" in z else ~np.isnan(oof).any(1)
        cov_te = z["coverage_test"].astype(bool) if "coverage_test" in z else ~np.isnan(test).any(1)
        assert oof.shape == (n_tr, 3) and test.shape == (n_te, 3), f"{name}: bad shape"
        if "fold_ids" in z:
            assert np.array_equal(z["fold_ids"].astype(np.int64), folds), f"{name}: DIFFERENT FOLD SET"
        for arr, cov, what in ((oof, cov_tr, "oof"), (test, cov_te, "test")):
            a = arr[cov]
            assert np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all(), f"{name} {what}: bad values"
            assert np.abs(a.sum(1) - 1).max() < 1e-3, f"{name} {what}: rows do not sum to 1"
        uses_id = fam == "id" or bool(ID_NAME_RE.search(name))
        if "meta" in z:
            try:
                uses_id = uses_id or bool(json.loads(str(z["meta"])).get("uses_id", False))
            except Exception:
                pass
        if uses_id and fam != "id":
            raise SystemExit(f"{name} looks ID-derived but was given family {fam}. Use id:{name}=...")
        S[name] = dict(family=fam, oof=oof, test=test, cov_tr=cov_tr, cov_te=cov_te, path=path,
                       sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), uses_id=uses_id)
        print(f"  loaded {fam:9s} {name:18s} covered {cov_tr.sum():6d}/{n_tr}  uses_id={uses_id}")
    return S


# ---------------------------------------------------------------- blending
def simplex(n, step):
    T = int(round(1 / step))
    return np.array([c for c in itertools.product(range(T + 1), repeat=n) if sum(c) == T], float) / T


def refine(center, step=0.05, radius=0.1):
    axes = [np.round(np.arange(max(0, c - radius), min(1, c + radius) + 1e-9, step), 4) for c in center]
    cands = [w for w in itertools.product(*axes) if abs(sum(w) - 1) < 1e-9]
    return np.array(cands, float) if cands else center[None]


def best_weights(P, y):
    """P: (n_src, n_rows, 3). Coarse 0.10 simplex, then 0.05 refinement around the winner."""
    def search(G):
        scores = [mf1(y, np.tensordot(w, P, 1).argmax(1)) for w in G]
        i = int(np.argmax(scores))
        return G[i], scores[i]
    w, _ = search(simplex(P.shape[0], 0.10))
    return search(refine(w))


def mix(P, w):
    return np.tensordot(w, P, 1)


def route_masks(route_by_script, bangla):
    return {"bangla": bangla, "other": ~bangla} if route_by_script else {"all": np.ones_like(bangla)}


def usable(S, names, mask, test=False):
    key = "cov_te" if test else "cov_tr"
    return [n for n in names if S[n][key][mask].all()]


def honest_blend(S, names, y, folds, routes, offsets=False):
    """Cross-fitted: for fold k, weights (per route) and optional offsets come from folds != k only."""
    n = len(y)
    probs = np.zeros((n, 3))
    fold_w = []
    for k in range(N_SPLITS):
        trm, vam = folds != k, folds == k
        wk = {}
        for r, rm in routes.items():
            srcs = usable(S, names, rm)
            assert srcs, f"route {r} has no covering source"
            P = np.stack([S[s]["oof"] for s in srcs])
            w, _ = best_weights(P[:, rm & trm], y[rm & trm])
            probs[rm & vam] = mix(P[:, rm & vam], w)
            wk[r] = dict(zip(srcs, map(float, w)))
        fold_w.append(wk)
    pred = probs.argmax(1)
    out = dict(names=names, pred=pred, probs=probs, fold_w=fold_w)
    if offsets:
        # Strictly nested: for outer fold k, build INNER cross-fitted probabilities on the other four
        # folds (inner fold j gets weights chosen on folds not in {j, k}), tune the per-class offset on
        # those, then apply it to fold k's outer probabilities. Fold k's labels never touch its offset.
        pred_o = pred.copy()
        fold_b = []
        grid = [np.array([0, a, b]) for a in np.arange(-1, 1.01, 0.1) for b in np.arange(-1, 1.01, 0.1)]
        for k in range(N_SPLITS):
            outer_tr, vam = folds != k, folds == k
            inner = np.zeros((n, 3))
            for j in range(N_SPLITS):
                if j == k:
                    continue
                fit_m, app_m = (folds != k) & (folds != j), folds == j
                for r, rm in routes.items():
                    srcs = usable(S, names, rm)
                    P = np.stack([S[s]["oof"] for s in srcs])
                    w, _ = best_weights(P[:, rm & fit_m], y[rm & fit_m])
                    inner[rm & app_m] = mix(P[:, rm & app_m], w)
            Li = np.log(np.clip(inner[outer_tr], 1e-9, 1))
            sc = [mf1(y[outer_tr], (Li + b).argmax(1)) for b in grid]
            b = grid[int(np.argmax(sc))]
            pred_o[vam] = (np.log(np.clip(probs[vam], 1e-9, 1)) + b).argmax(1)
            fold_b.append(b.round(2).tolist())
        out.update(pred_offset=pred_o, fold_b=fold_b)
    return out


def summarise(y, pred, folds):
    fs = [mf1(y[folds == k], pred[folds == k]) for k in range(N_SPLITS)]
    return dict(macro_f1=round(mf1(y, pred), 6), per_class=np.round(f1s(y, pred), 6).tolist(),
                accuracy_NOT_macro_f1=round(float((pred == y).mean()), 6),
                confusion=conf(y, pred).tolist(), fold_scores=np.round(fs, 6).tolist(),
                fold_mean=round(float(np.mean(fs)), 6), fold_std=round(float(np.std(fs)), 6),
                worst_fold=round(float(np.min(fs)), 6),
                pred_share=np.round(np.bincount(pred, minlength=3) / len(pred), 4).tolist())


def consensus(fold_w):
    """Median of the five fold weight vectors per route, renormalised."""
    out = {}
    for r in fold_w[0]:
        names = list(fold_w[0][r])
        W = np.array([[fw[r][n] for n in names] for fw in fold_w])
        m = np.median(W, 0)
        m = m / m.sum() if m.sum() > 0 else np.full(len(m), 1 / len(m))
        out[r] = dict(zip(names, map(float, m)))
    return out


def apply_weights(S, wmap, routes_tr, routes_te, key_oof="oof", key_test="test"):
    oof = np.zeros((len(next(iter(routes_tr.values()))), 3))
    test = np.zeros((len(next(iter(routes_te.values()))), 3))
    for r, w in wmap.items():
        for n, v in w.items():
            oof[routes_tr[r]] += v * S[n][key_oof][routes_tr[r]]
            test[routes_te[r]] += v * S[n][key_test][routes_te[r]]
    return oof, test


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", default="reports/v7")
    ap.add_argument("--pred-out", default="predictions/text_v7")
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--no-offsets", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    out, pout = Path(a.out), Path(a.pred_out)
    out.mkdir(parents=True, exist_ok=True)
    pout.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(Path(a.data_dir) / "train.csv")
    test = pd.read_csv(Path(a.data_dir) / "test.csv")
    y = train["y"].to_numpy(np.int64)
    folds = np.load(a.folds).astype(np.int64)
    assert hashlib.sha256(folds.tobytes()).hexdigest() == EXPECTED_FOLD_INT64, "not fold set B"
    bn_re = re.compile("[ঀ-৿]")
    bangla = train["text"].astype(str).map(lambda t: bool(bn_re.search(t))).to_numpy()
    bangla_te = test["text"].astype(str).map(lambda t: bool(bn_re.search(t))).to_numpy()
    length = train["text"].astype(str).str.len().to_numpy()

    print("sources:")
    S = load_sources(a.src, len(train), len(test), folds)
    text_names = [n for n in S if not S[n]["uses_id"]]
    id_names = [n for n in S if S[n]["uses_id"]]
    assert text_names, "no text-only sources"
    partial = any(not S[n]["cov_tr"].all() for n in text_names)
    routes = route_masks(partial, bangla)
    routes_te = route_masks(partial, bangla_te)
    R = dict(sources={n: dict(family=S[n]["family"], path=S[n]["path"], sha256=S[n]["sha256"],
                              uses_id=S[n]["uses_id"], covered=int(S[n]["cov_tr"].sum()))
                      for n in S},
             routing="by script (bangla / other)" if partial else "none", fold_int64_sha256=EXPECTED_FOLD_INT64)

    # ---- standalone
    R["standalone"] = {}
    for n in S:
        c = S[n]["cov_tr"]
        p = S[n]["oof"][c].argmax(1)
        R["standalone"][n] = dict(macro_f1_on_covered=round(mf1(y[c], p), 6),
                                  per_class=np.round(f1s(y[c], p), 4).tolist(), covered=int(c.sum()))

    # ---- T1: all text sources, honest, with nested offsets as a gated option
    print("honest blend over all text-only sources ...", flush=True)
    full = honest_blend(S, text_names, y, folds, routes, offsets=not a.no_offsets)
    base = summarise(y, full["pred"], folds)
    R["T1_no_offsets"] = dict(base, fold_weights=full["fold_w"])
    adopt_offsets = False
    if not a.no_offsets:
        off = summarise(y, full["pred_offset"], folds)
        gain = off["macro_f1"] - base["macro_f1"]
        folds_ok = sum(o >= b - 0.001 for o, b in zip(off["fold_scores"], base["fold_scores"]))
        no_collapse = all(o >= b - 0.02 for o, b in zip(off["per_class"], base["per_class"]))
        stable = off["fold_std"] <= 1.5 * base["fold_std"] + 1e-4
        adopt_offsets = gain >= 0.001 and folds_ok >= 4 and no_collapse and stable
        R["T1_offsets_nested"] = dict(off, fold_offsets=full["fold_b"], gain=round(gain, 6),
                                      gates=dict(gain_ge_0_001=gain >= 0.001, folds_not_regressed=folds_ok,
                                                 no_class_collapse=no_collapse, std_stable=stable),
                                      adopted=adopt_offsets)

    # ---- same-OOF optimum (OPTIMISTIC, reporting only)
    opt_pred = np.zeros(len(y), int)
    for r, rm in routes.items():
        srcs = usable(S, text_names, rm)
        P = np.stack([S[s]["oof"] for s in srcs])
        w, _ = best_weights(P[:, rm], y[rm])
        opt_pred[rm] = mix(P[:, rm], w).argmax(1)
    R["OPTIMISTIC_same_oof_macro_f1_NOT_A_RESULT"] = round(mf1(y, opt_pred), 6)

    # ---- leave-one-source-out: honest contribution, unique corrections and damages
    print("leave-one-out contributions ...", flush=True)
    R["leave_one_out"] = {}
    corr_full = full["pred"] == y
    for n in text_names:
        rest = [m for m in text_names if m != n]
        if not rest or any(not usable(S, rest, rm) for rm in routes.values()):
            continue
        lo = honest_blend(S, rest, y, folds, routes)
        corr_lo = lo["pred"] == y
        R["leave_one_out"][n] = dict(
            blend_without=round(mf1(y, lo["pred"]), 6),
            contribution=round(mf1(y, full["pred"]) - mf1(y, lo["pred"]), 6),
            unique_corrections=int((corr_full & ~corr_lo).sum()),
            unique_damages=int((~corr_full & corr_lo).sum()),
            class2_f1_delta=round(float(f1s(y, full["pred"])[2] - f1s(y, lo["pred"])[2]), 6))

    # ---- disagreement between sources (argmax), on rows all of them cover
    common = np.all([S[n]["cov_tr"] for n in text_names], 0)
    R["disagreement_rate"] = {f"{i}|{j}": round(float((S[i]["oof"][common].argmax(1)
                                                       != S[j]["oof"][common].argmax(1)).mean()), 4)
                              for i, j in itertools.combinations(text_names, 2)}

    # ---- T2: stable consensus - drop sources with zero weight in >=3 folds, re-run, no offsets
    zero_counts = {n: sum(all(fw[r].get(n, 0) == 0 for r in fw) for fw in full["fold_w"])
                   for n in text_names}
    t2_names = [n for n in text_names if zero_counts[n] < 3] or text_names
    t2 = honest_blend(S, t2_names, y, folds, routes)
    R["T2_stable"] = dict(summarise(y, t2["pred"], folds), sources=t2_names, fold_weights=t2["fold_w"])

    # ---- T3: diverse hedge - equal weights, nothing tuned (so its OOF is honest by construction)
    eq = {r: {n: 1 / len(usable(S, text_names, rm)) for n in usable(S, text_names, rm)}
          for r, rm in routes.items()}
    t3_oof, t3_test = apply_weights(S, eq, routes, routes_te)
    R["T3_equal_hedge"] = dict(summarise(y, t3_oof.argmax(1), folds), weights=eq)

    # ---- test-time packages: T1 uses median fold weights (+ median offsets if adopted)
    w1 = consensus(full["fold_w"])
    t1_oof_cons, t1_test = apply_weights(S, w1, routes, routes_te)
    if adopt_offsets:
        b = np.median(np.array(full["fold_b"]), 0)
        t1_test_pred = (np.log(np.clip(t1_test, 1e-9, 1)) + b).argmax(1)
        R["T1_final_offsets"] = b.round(3).tolist()
    else:
        t1_test_pred = t1_test.argmax(1)
    w2 = consensus(t2["fold_w"])
    t2_oof_cons, t2_test = apply_weights(S, w2, routes, routes_te)
    train_share = np.bincount(y, minlength=3) / len(y)
    for tag, wmap, tp, tpred in (("T1", w1, t1_test, t1_test_pred), ("T2", w2, t2_test, t2_test.argmax(1)),
                                 ("T3", eq, t3_test, t3_test.argmax(1))):
        assert tp.shape == (len(test), 3) and np.isfinite(tp).all()
        share = np.bincount(tpred, minlength=3) / len(tpred)
        R[f"{tag}_package"] = dict(final_weights=wmap, test_pred_share=share.round(4).tolist(),
                                   max_share_drift_pp=round(float(100 * np.abs(share - train_share).max()), 2))
        np.save(pout / f"CANDIDATE_NOT_FOR_SUBMISSION_{tag}_test_probs.npy", tp)
    np.save(pout / "CANDIDATE_T1_oof_honest_pred.npy",
            full["pred_offset"] if adopt_offsets else full["pred"])
    np.save(pout / "CANDIDATE_T2_oof_honest_pred.npy", t2["pred"])
    np.save(pout / "CANDIDATE_T3_oof_probs.npy", t3_oof)

    # ---- error analysis on the T1 honest OOF predictions
    p1 = full["pred_offset"] if adopt_offsets else full["pred"]
    groups = {"script=bangla": bangla, "script=other": ~bangla,
              "len<20": length < 20, "len20-49": (length >= 20) & (length < 50),
              "len50-119": (length >= 50) & (length < 120), "len>=120": length >= 120,
              "has_url_or_mention": train["text"].astype(str).str.contains(r"https?://|www\.|@\w").to_numpy()}
    R["error_analysis_T1"] = {g: dict(rows=int(m.sum()), macro_f1=round(mf1(y[m], p1[m]), 4),
                                      per_class=np.round(f1s(y[m], p1[m]), 4).tolist())
                              for g, m in groups.items() if m.sum() > 50}
    cm = conf(y, p1)
    R["error_analysis_T1"]["top_confusions"] = sorted(
        [(f"true {i} -> pred {j}", int(cm[i, j])) for i in range(3) for j in range(3) if i != j],
        key=lambda t: -t[1])

    # ---- ablation table (ID rows clearly labelled)
    def fam(f):
        return [n for n in text_names if S[n]["family"] == f]

    rows = [("lexical text only", fam("lexical")), ("frozen semantic only", fam("semantic")),
            ("fine-tuned transformer text only", fam("finetuned")),
            ("lexical + semantic", fam("lexical") + fam("semantic")),
            ("all text-only (T1, no offsets)", text_names)]
    plain = [n for n in id_names if "plain" in n or n in ("id", "id_prior")]
    if plain:
        rows += [("plain ID only [ID-DERIVED]", plain[:1]),
                 ("all text + plain ID [ID-DERIVED]", text_names + plain[:1])]
    if id_names:
        rows += [("all text + all ID sources [ID-DERIVED]", text_names + id_names)]
    R["ablation"] = []
    print("ablation rows ...", flush=True)
    full_routes = {"all": np.ones(len(y), bool)}
    for label, names in rows:
        if not names:
            continue
        rr = routes if any(not S[n]["cov_tr"].all() for n in names) else full_routes
        res = summarise(y, honest_blend(S, names, y, folds, rr)["pred"], folds)
        R["ablation"].append(dict(row=label, sources=names, text_derived="ID-DERIVED" not in label,
                                  macro_f1=res["macro_f1"], per_class=res["per_class"],
                                  fold_std=res["fold_std"]))
    R["ablation"].append(dict(row="Submission 3 full model [ID-DERIVED, recorded, not recomputed]",
                              sources=["text", "SID", "E5 Base", "XLM-R ID-head"], text_derived=False,
                              macro_f1=SUB3_HONEST, per_class=[0.782202, 0.640594, 0.624137],
                              fold_std=0.001859))

    best_text = max(R["T1_no_offsets"]["macro_f1"],
                    R.get("T1_offsets_nested", {}).get("macro_f1", 0) if adopt_offsets else 0)
    R["verdict"] = dict(best_text_only_honest=best_text, sub3_honest=SUB3_HONEST,
                        beats_sub3_honestly=bool(best_text > SUB3_HONEST),
                        band=("weak" if best_text < 0.66 else "credible" if best_text < 0.67 else
                              "strong" if best_text < 0.68 else "major" if best_text < 0.69 else "excellent"),
                        submission_generated=False, seconds=round(time.time() - t0))
    (out / "text_v7_report.json").write_text(json.dumps(R, indent=2, default=lambda o: o.tolist()))

    # ---- console summary
    print("\n================ TEXT-ONLY RESULT (fold set B, honest cross-fitted) ================")
    for n, s in R["standalone"].items():
        print(f"  standalone {n:18s} {s['macro_f1_on_covered']:.6f} {s['per_class']} (covered {s['covered']})")
    t1 = R["T1_no_offsets"]
    print(f"  T1 honest            {t1['macro_f1']:.6f} per-class {t1['per_class']} folds {t1['fold_scores']} "
          f"std {t1['fold_std']}")
    if "T1_offsets_nested" in R:
        o = R["T1_offsets_nested"]
        print(f"  T1 + nested offsets  {o['macro_f1']:.6f} gain {o['gain']:+.6f} adopted={o['adopted']} {o['gates']}")
    print(f"  T2 stable            {R['T2_stable']['macro_f1']:.6f} sources {R['T2_stable']['sources']}")
    print(f"  T3 equal hedge       {R['T3_equal_hedge']['macro_f1']:.6f}")
    print(f"  OPTIMISTIC same-OOF  {R['OPTIMISTIC_same_oof_macro_f1_NOT_A_RESULT']:.6f}  (not a result)")
    print("  leave-one-out:", {n: v["contribution"] for n, v in R["leave_one_out"].items()})
    print("  ablation:")
    for r in R["ablation"]:
        print(f"    {r['row']:60s} {r['macro_f1']:.6f}  {r['per_class']}")
    print(f"  BEATS 0.682311 HONESTLY: {R['verdict']['beats_sub3_honestly']}   band: {R['verdict']['band']}")
    print(f"  report: {out / 'text_v7_report.json'}   packages: {pout}   SUBMISSION GENERATED: NO")


if __name__ == "__main__":
    main()

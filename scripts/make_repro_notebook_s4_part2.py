"""Notebook 4 cells, part 2: features and the frozen stacker."""
import nbformat as nbf

from make_repro_notebook_s4_part1 import CELLS, code

MD_FEAT = """## 2. Rebuilding the frozen 60-feature design

The feature vector is built deterministically from the bundled probability
arrays plus the raw `id` and `text`. Nothing is fitted here - this is pure
feature construction, so Mode A reproduces it exactly on any CPU.

* **18 source probabilities** - `p0/p1/p2` for each of the six bundled sources.
* **32 ID-neighbour features** - for each offset 1/2/3: the label and
  `log1p` distance to the nearest pool ids on each side, and for each id-window
  width 3/10/30/100/300: the neighbour count and the three class shares.
* **9 text meta features** - length, word count, Bengali/Latin flags, emoji
  count, capital ratio, `!`/`?` counts, digit ratio.
* **1 raw id feature** - `rid`.

The neighbour pool for a fold-`k` training row is the train rows of the **other
folds only**, so a row's own label is never visible to its own features. Test
features are rebuilt per fold with a pool of the same density."""

MD_STACK = """## 3. The frozen stacker policy

Cross-fitted LightGBM: for each fold `k` a model is trained on folds `!= k` and
predicts fold `k`. Every hyperparameter, the 400 rounds, the three seeds and the
`num_threads=2` pin are read from `config_submission_4.json` - nothing is tuned
here. Class weights are the standard inverse-frequency balance.

Test features are rebuilt inside the fold loop with that fold's pool, so the test
neighbour density matches the density seen during training."""

CELLS.append(nbf.v4.new_markdown_cell(MD_FEAT))
CELLS.append(code(r'''
# --- 5. deterministic feature construction (frozen definitions) --------------
def nb_feats(q, pool_ids, pool_y):
    """ID-neighbour labels/distances/window shares. Pool is never the query's fold."""
    o = np.argsort(pool_ids)
    s = pool_ids[o]
    sy = pool_y[o]
    n = len(s)
    pos = np.searchsorted(s, q)
    F = {}
    for i in (1, 2, 3):
        p = pos - i
        ok = p >= 0
        pc = np.clip(p, 0, n - 1)
        F["prev%d_y" % i] = np.where(ok, sy[pc], -1)
        F["prev%d_d" % i] = np.where(ok, np.log1p(np.abs(q - s[pc])), 9.0)
        nx = pos + i - 1
        ok = nx < n
        nc = np.clip(nx, 0, n - 1)
        F["next%d_y" % i] = np.where(ok, sy[nc], -1)
        F["next%d_d" % i] = np.where(ok, np.log1p(np.abs(s[nc] - q)), 9.0)
    cum = np.vstack([np.zeros((1, 3)), np.cumsum(np.eye(3)[sy], 0)])
    for Wn in (3, 10, 30, 100, 300):
        a = np.searchsorted(s, q - Wn, "left")
        b = np.searchsorted(s, q + Wn, "right")
        c = cum[b] - cum[a]
        t = c.sum(1)
        F["w%d_n" % Wn] = t
        for k in range(3):
            F["w%d_p%d" % (Wn, k)] = np.where(t > 0, c[:, k] / np.maximum(t, 1), np.nan)
    return pd.DataFrame(F)


def text_meta(txt):
    t = pd.Series(txt).astype(str)
    L = t.str.len()
    return pd.DataFrame({
        "len": L,
        "nwords": t.str.split().str.len(),
        "bn": t.str.contains("[\u0980-\u09ff]").astype(int),
        "lat": t.str.contains("[A-Za-z]").astype(int),
        "emoji": t.str.count("[\U0001F300-\U0001FAFF\u2600-\u27bf]"),
        "caps": t.str.count("[A-Z]") / L.clip(lower=1),
        "excl": t.str.count("!"),
        "q": t.str.count(r"\?"),
        "digits": t.str.count(r"\d") / L.clip(lower=1),
    })


parts = []
for k in range(N_SPLITS):
    va = np.flatnonzero(folds == k)
    pool = np.flatnonzero(folds != k)
    f = nb_feats(ids[va], ids[pool], y[pool])
    f.index = va
    parts.append(f)
nbtr = pd.concat(parts).sort_index()
nbte = nb_feats(tids, ids, y)


def probs(split):
    out = {}
    for n, p in SRC.items():
        z = np.load(p, allow_pickle=True)
        a = z["oof" if split == "tr" else "test"]
        for k in range(3):
            out["%s_p%d" % (n, k)] = a[:, k]
    return pd.DataFrame(out)


Ftr = pd.concat([probs("tr"), nbtr.reset_index(drop=True), text_meta(train["text"])], axis=1)
Ftr["rid"] = ids
Fte0 = pd.concat([probs("te"), nbte, text_meta(test["text"])], axis=1)
Fte0["rid"] = tids

print("Ftr", Ftr.shape, " Fte", Fte0.shape)
assert Ftr.shape == (47817, 60) and Fte0.shape == (11955, 60)
assert list(Ftr.columns) == list(SCHEMA["column_order"]) == list(Fte0.columns)
print("feature schema matches the bundled feature_schema.json exactly")
print(Ftr[["prev1_d", "next1_d", "w10_n"]].describe().loc[["mean", "50%"]])
'''))
CELLS.append(nbf.v4.new_markdown_cell(MD_STACK))
CELLS.append(code(r'''
# --- 6. frozen cross-fitted 3-seed LightGBM stacker -------------------------
import time
import warnings

warnings.filterwarnings("ignore")

NB_COLS = [c for c in Ftr.columns if c.startswith(("prev", "next", "w"))]
PARAMS = dict(CFG["lightgbm_params"])
cw = len(y) / (3 * np.bincount(y))
WTS = cw[y]


def run_seed(seed):
    params = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    oof = np.zeros((n_tr, 3))
    tst = np.zeros((n_te, 3))
    t0 = time.time()
    for k in range(N_SPLITS):
        trm = folds != k
        va = folds == k
        m = lgb.train(params, lgb.Dataset(Ftr.loc[trm], y[trm], weight=WTS[trm]), ROUNDS)
        oof[va] = m.predict(Ftr.loc[va])
        pool = np.flatnonzero(folds != k)
        Fte = Fte0.copy()
        Fte[NB_COLS] = nb_feats(tids, ids[pool], y[pool])[NB_COLS].values
        tst += m.predict(Fte) / N_SPLITS
        print("   seed %-5d fold %d  %4ds  %.4f" % (
            seed, k, time.time() - t0, macro_f1(y[va], oof[va].argmax(1))), flush=True)
    print("   seed %-5s no-correction OOF %.6f" % (seed, macro_f1(y, oof.argmax(1))), flush=True)
    return oof, tst


OOF, TST, SEED_SCORES = [], [], {}
for s in SEEDS:
    o, t = run_seed(s)
    OOF.append(o)
    TST.append(t)
    SEED_SCORES[str(s)] = macro_f1(y, o.argmax(1))

oof_avg = np.mean(OOF, axis=0)
tst_avg = np.mean(TST, axis=0)
assert oof_avg.shape == (47817, 3) and tst_avg.shape == (11955, 3)
assert np.isfinite(oof_avg).all() and np.isfinite(tst_avg).all()
assert (oof_avg >= 0).all() and (oof_avg <= 1).all()
assert (tst_avg >= 0).all() and (tst_avg <= 1).all()
print("seed scores :", {k: round(v, 6) for k, v in SEED_SCORES.items()})
print("3-seed average, no correction: %.6f" % macro_f1(y, oof_avg.argmax(1)))
'''))


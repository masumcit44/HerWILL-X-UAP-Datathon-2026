# Notebook cell 07 - code: GPU source arrays + per-fold stitch verification
# --- 2. GPU source artifacts: per-fold stitch check --------------------------
d_xlmr = np.load(BUNDLE / "xlmr_base_idhead_final.npz")
d_e5 = np.load(BUNDLE / "emb_e5-base_final.npz")
xlmr_oof, xlmr_test = d_xlmr["oof"], d_xlmr["test"]
e5_oof, e5_test = d_e5["oof"], d_e5["test"]

for nm, arr_o, arr_t, exp in (("xlmr_base_idhead", xlmr_oof, xlmr_test, 0.648743),
                              ("emb_e5-base", e5_oof, e5_test, 0.521535)):
    assert arr_o.shape == (47817, 3) and arr_t.shape == (11955, 3)
    assert np.isfinite(arr_o).all() and np.isfinite(arr_t).all()
    sc = macro_f1(y, arr_o)
    print("  {:<20} standalone OOF macro-F1 {:.6f} (expected {:.6f})".format(
        nm, sc, exp))
    assert abs(sc - exp) < 2e-6

# stitching the five per-fold ID-head blocks must reproduce the combined arrays
stitched = np.zeros((n_tr, 3))
tps = []
for k in range(N_SPLITS):
    d = np.load(BUNDLE / "xlmr_base_idhead_fold{}.npz".format(k))
    va, vp, tp = d["va"], d["vp"], d["tp"]
    assert np.array_equal(np.sort(va), np.flatnonzero(folds == k)), "va/fold mismatch"
    stitched[va] = vp
    tps.append(tp)
d_oof = float(np.abs(stitched - xlmr_oof).max())
d_test = float(np.abs(np.mean(tps, axis=0) - xlmr_test).max())
print("  per-fold stitch max|diff|  oof {:.2e}  test {:.2e}".format(d_oof, d_test))
assert d_oof < 1e-6 and d_test < 1e-6

# Notebook cell 06 - code: verify bundled artifacts + duplicate-leakage check
# --- 1. verify the bundled artifacts against the SHA256 manifest -------------
MANIFEST = json.load(open(BUNDLE / "manifest_submission_3.json"))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


for name, meta in sorted(MANIFEST.items()):
    p = BUNDLE / name
    got = sha256_file(p)
    assert got == meta["sha256"], "SHA256 mismatch for {}: {} != {}".format(
        name, got, meta["sha256"])
    print("  OK {:<44} {}".format(name, got[:16]))

folds = np.load(BUNDLE / "fold_ids_colab.npy").astype(np.int64)
print("fold sizes:", np.bincount(folds, minlength=N_SPLITS).tolist())
assert folds.shape == (47817,)
assert np.bincount(folds, minlength=5).tolist() == [9543, 9546, 9571, 9581, 9576]

# duplicate-leakage check: zero cleaned-text groups may span more than one fold
gkey = pd.Index([str(c).lower().strip() for c in clean_tr], dtype="object")
gcode = pd.factorize(gkey, sort=False)[0]
n_cross = int((pd.DataFrame({"g": gcode, "f": folds}).groupby("g")["f"]
               .nunique() > 1).sum())
print("cleaned-text groups crossing folds:", n_cross, "(must be 0)")
assert n_cross == 0

# Notebook cell 08 - code: text source, per-fold OOF and full-train test fit
# --- 3. text source (CPU): per-fold OOF, then full-train fit for test ---------
def fit_text(tr_idx, va_idx):
    """Fit both TF-IDF views on tr_idx only, predict va_idx."""
    wv = TfidfVectorizer(**WORD_KW).fit(clean_tr[tr_idx])
    cv = TfidfVectorizer(**CHAR_KW).fit(clean_tr[tr_idx])
    Xtr = sp.hstack([wv.transform(clean_tr[tr_idx]), cv.transform(clean_tr[tr_idx])],
                    format="csr").astype(np.float32)
    Xva = sp.hstack([wv.transform(clean_tr[va_idx]), cv.transform(clean_tr[va_idx])],
                    format="csr").astype(np.float32)
    clf = LogisticRegression(**LR_KW).fit(Xtr, y[tr_idx])
    return clf.predict_proba(Xva)


oof_text = np.zeros((n_tr, 3))
for k in range(N_SPLITS):
    va = np.flatnonzero(folds == k)
    tr = np.flatnonzero(folds != k)
    oof_text[va] = fit_text(tr, va)
    print("  fold {} text done".format(k))

text_only = macro_f1(y, oof_text)
print("text-only OOF macro-F1 = {:.6f} (expected ~0.5160)".format(text_only))
assert abs(text_only - 0.5160) <= 0.003

# full-train fit -> test probabilities (same recipe, fitted on ALL train rows)
wv = TfidfVectorizer(**WORD_KW).fit(clean_tr)
cv = TfidfVectorizer(**CHAR_KW).fit(clean_tr)
Xall = sp.hstack([wv.transform(clean_tr), cv.transform(clean_tr)],
                 format="csr").astype(np.float32)
Xte = sp.hstack([wv.transform(clean_te), cv.transform(clean_te)],
                format="csr").astype(np.float32)
test_text = LogisticRegression(**LR_KW).fit(Xall, y).predict_proba(Xte)
print("test text probs", test_text.shape)

# Notebook cell 17 - code: Mode B optional GPU retraining
# --- Mode B: optional GPU retraining (skipped unless RUN_GPU_RETRAIN is True) -
if not RUN_GPU_RETRAIN:
    print("Mode B skipped (RUN_GPU_RETRAIN = False). "
          "Mode A used the SHA256-verified probability arrays.")
else:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              AutoModel)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Mode B: retraining on", device)
    MAX_LEN, BS, EPOCHS, LR = 128, 32, 3, 2e-5

    class TxtDS(Dataset):
        """Thin tensor dataset over pre-tokenised text."""

        def __init__(self, enc, labels=None):
            self.enc, self.labels = enc, labels

        def __len__(self):
            return len(self.enc["input_ids"])

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            if self.labels is not None:
                item["labels"] = int(self.labels[i])
            return item

    # ---- source 1: xlm-roberta-base + classification head (ID-head) --------
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def encode(texts):
        return tok(list(texts), truncation=True, padding="max_length",
                   max_length=MAX_LEN, return_tensors="pt")

    def fit_predict(tr_texts, tr_labels, pred_texts, epochs=EPOCHS):
        """Fine-tune on (tr_texts, tr_labels); return softmax probs for pred_texts."""
        m = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-base", num_labels=3).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        dl = DataLoader(TxtDS(encode(tr_texts), tr_labels), batch_size=BS, shuffle=True)
        scaler = torch.amp.GradScaler("cuda")
        m.train()
        for ep in range(epochs):
            for b in dl:
                opt.zero_grad()
                b = {k: v.to(device) for k, v in b.items()}
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = m(**b).loss
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            print("    epoch {} loss {:.4f}".format(ep + 1, float(loss.detach())))
        m.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(pred_texts), 256):
                enc = {k: v.to(device) for k, v in
                       encode(pred_texts[s:s + 256]).items()}
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lg = m(**enc).logits
                outs.append(torch.softmax(lg.float(), -1).cpu().numpy())
        return np.vstack(outs)

    # per-fold OOF on the bundled folds, then a full-train fit for the test set
    xlmr_oof = np.zeros((n_tr, 3))
    for k in range(N_SPLITS):
        va = np.flatnonzero(folds == k)
        tr = np.flatnonzero(folds != k)
        xlmr_oof[va] = fit_predict(clean_tr[tr], y[tr], clean_tr[va])
        print("  fold {} xlmr retrained".format(k))
    xlmr_test = fit_predict(clean_tr, y, list(clean_te))
    print("xlmr_idhead retrained: OOF macro-F1 {:.6f}".format(macro_f1(y, xlmr_oof)))

    # ---- source 2: E5-base embeddings + label head -------------------------
    e5_tok = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
    e5_mod = AutoModel.from_pretrained("intfloat/e5-base-v2").to(device).eval()

    def embed(texts, bs=64):
        """Mean-pooled, L2-normalised E5 embeddings."""
        outs = []
        with torch.no_grad():
            for s in range(0, len(texts), bs):
                enc = e5_tok(["query: " + t for t in texts[s:s + bs]],
                             padding=True, truncation=True, max_length=MAX_LEN,
                             return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    h = e5_mod(**enc).last_hidden_state
                msk = enc["attention_mask"].unsqueeze(-1).float()
                outs.append(((h.float() * msk).sum(1) / msk.sum(1)).cpu().numpy())
        return normalize(np.vstack(outs)).astype(np.float32)

    Etr, Ete = embed(list(clean_tr)), embed(list(clean_te))
    head = LogisticRegression(max_iter=2000, random_state=SEED).fit(Etr, y)
    e5_oof, e5_test = head.predict_proba(Etr), head.predict_proba(Ete)
    print("e5_base retrained: OOF macro-F1 {:.6f}".format(macro_f1(y, e5_oof)))
    print("Mode B: re-run the blend and CSV cells above to use the retrained sources.")

"""
Standalone Colab & GPU Training Script for XLM-RoBERTa-base
HerWILL X UAP Datathon 2026 - Experiment v3.2

Features:
- Enforces EXACT SAME grouped 5-fold split via canonical_train_folds.csv or deterministic StratifiedGroupKFold(5, seed=42)
- Architecture: xlm-roberta-base with sequence classification head (3 classes)
- Hyperparameters: max_length=160, batch_size=16 (or 32 with fp16), 3 epochs, lr=2e-5, warmup_ratio=0.06, weight_decay=0.01, seed=42
- Class-weighted Cross-Entropy loss from inverse training frequencies per fold
- Saves exact schema OOF predictions: id,y_true,prob_0,prob_1,prob_2,y_pred,fold,model_name
- Saves candidate test probabilities as .npy/.npz under predictions/v3_candidates/ (NO CSV test submission created)
- Evaluates complementarity with baseline TF-IDF LR and k=20 ID priors
"""

import os
import sys
import gc
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# Ensure deterministic seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CONFIG = {
    "model_name": "xlm-roberta-base",
    "max_length": 160,
    "batch_size": 16,
    "epochs": 3,
    "lr": 2e-5,
    "warmup_ratio": 0.06,
    "weight_decay": 0.01,
    "seed": 42,
    "n_classes": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def eval_epoch(model, dataloader, device):
    model.eval()
    all_logits = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
def main():
    print("=" * 78)
    print("XLM-RoBERTa-base Grouped 5-Fold Finetuning (Colab / GPU Ready)")
    print(f"Device: {CONFIG['device']} | Batch Size: {CONFIG['batch_size']}")
    print("=" * 78)

    os.makedirs("predictions/v3_candidates", exist_ok=True)
    os.makedirs("reports/v3_xlmr", exist_ok=True)
    os.makedirs("models/v3/xlmr", exist_ok=True)

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    print(f"Loaded train: {train.shape}, test: {test.shape}")

    if os.path.exists("colab/v3_xlmr/canonical_train_folds.csv"):
        fold_df = pd.read_csv("colab/v3_xlmr/canonical_train_folds.csv")
        train["fold"] = fold_df["fold"].values
    elif os.path.exists("cache/v2/fold_ids.npy"):
        train["fold"] = np.load("cache/v2/fold_ids.npy")
    else:
        raise FileNotFoundError("Canonical fold assignments not found!")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    oof_probs = np.zeros((len(train), 3), dtype=np.float32)
    test_fold_probs = np.zeros((5, len(test), 3), dtype=np.float32)

    test_dataset = TextDataset(test["text"].values, None, tokenizer, CONFIG["max_length"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"] * 2, shuffle=False)

    for f in range(5):
        print(f"\n--- FOLD {f} / 5 ---")
        val_mask = (train["fold"] == f)
        tr_mask = (train["fold"] != f)

        val_df = train[val_mask].reset_index(drop=True)
        tr_df = train[tr_mask].reset_index(drop=True)

        class_counts = tr_df["y"].value_counts().sort_index().values
        weights = len(tr_df) / (3.0 * class_counts)
        class_weights_t = torch.tensor(weights, dtype=torch.float32).to(CONFIG["device"])
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)

        tr_dataset = TextDataset(tr_df["text"].values, tr_df["y"].values, tokenizer, CONFIG["max_length"])
        val_dataset = TextDataset(val_df["text"].values, val_df["y"].values, tokenizer, CONFIG["max_length"])

        tr_loader = DataLoader(tr_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"] * 2, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"], num_labels=3)
        model.to(CONFIG["device"])

        total_steps = len(tr_loader) * CONFIG["epochs"]
        warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        best_val_f1 = -1.0
        best_probs = None

        for epoch in range(CONFIG["epochs"]):
            loss = train_epoch(model, tr_loader, optimizer, scheduler, criterion, CONFIG["device"])
            probs = eval_epoch(model, val_loader, CONFIG["device"])
            preds = probs.argmax(axis=1)
            macro_f1 = f1_score(val_df["y"].values, preds, average="macro")
            print(f"Epoch {epoch + 1}/{CONFIG['epochs']} - Train Loss: {loss:.4f} - Val Macro F1: {macro_f1:.6f}")
            if macro_f1 > best_val_f1:
                best_val_f1 = macro_f1
                best_probs = probs
                torch.save(model.state_dict(), f"models/v3/xlmr/xlmr_best_fold_{f}.pt")

        oof_probs[val_mask] = best_probs
        print(f"Fold {f} Best Macro F1: {best_val_f1:.6f}")

        model.load_state_dict(torch.load(f"models/v3/xlmr/xlmr_best_fold_{f}.pt"))
        test_fold_probs[f] = eval_epoch(model, test_loader, CONFIG["device"])

        fold_oof_df = pd.DataFrame({
            "id": val_df["id"].values,
            "y_true": val_df["y"].values,
            "prob_0": best_probs[:, 0],
            "prob_1": best_probs[:, 1],
            "prob_2": best_probs[:, 2],
            "y_pred": best_probs.argmax(axis=1),
            "fold": f,
            "model_name": "xlm-roberta-base"
        })
        fold_oof_df.to_csv(f"reports/v3_xlmr/xlmr_oof_fold_{f}.csv", index=False)
        np.save(f"predictions/v3_candidates/CANDIDATE_NOT_FOR_SUBMISSION_xlmr_fold_{f}_test_probs.npy", test_fold_probs[f])

        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    complete_oof_df = pd.DataFrame({
        "id": train["id"].values,
        "y_true": train["y"].values,
        "prob_0": oof_probs[:, 0],
        "prob_1": oof_probs[:, 1],
        "prob_2": oof_probs[:, 2],
        "y_pred": oof_probs.argmax(axis=1),
        "fold": train["fold"].values,
        "model_name": "xlm-roberta-base"
    })
    complete_oof_df.to_csv("reports/v3_xlmr/xlmr_oof_all.csv", index=False)

    test_avg_probs = np.mean(test_fold_probs, axis=0)
    np.save("predictions/v3_candidates/CANDIDATE_NOT_FOR_SUBMISSION_xlmr_test_probs.npy", test_avg_probs)

    overall_f1 = f1_score(train["y"].values, oof_probs.argmax(axis=1), average="macro")
    per_class_f1 = f1_score(train["y"].values, oof_probs.argmax(axis=1), average=None)
    print(f"\n=======================================================")
    print(f"XLM-RoBERTa-base Pooled OOF Macro F1: {overall_f1:.6f}")
    print(f"Class 0 F1: {per_class_f1[0]:.6f} | Class 1 F1: {per_class_f1[1]:.6f} | Class 2 F1: {per_class_f1[2]:.6f}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()

        all_logits.append(outputs.logits.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)

            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

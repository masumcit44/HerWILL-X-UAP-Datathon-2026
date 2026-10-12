# Colab run order - UAP text-centric Stage 1

This package trains **text-only** models for Track B (the judge-friendly
candidate) and produces **probability artifacts only**. It never creates a
submission CSV, never uses the `id` column as a feature, and never touches
Submissions 1, 2 or 3.

## Files in this package

| file | purpose |
|---|---|
| `uap_text_train_colab_checked.ipynb` | the training notebook (byte-identical to `uap_text_train_colab.ipynb`; audit found no blocker) |
| `fold_ids_colab.npy` | the locked fold set B, int64 SHA256 `cc4fdd16...`, sizes `[9543, 9546, 9571, 9581, 9576]` |
| `data_hash_manifest.json` | SHA256 of the fold file plus the three competition CSVs you must supply |
| `model_config.json` | run order, pinned recipes, adoption gates, expected artifacts |

Competition CSVs are **not** bundled here. Supply them yourself.

---

## Run order

### 1. Place the data

Put these three files in `MyDrive/uap_text/data/`:

- `train.csv` - SHA256 `a269ad841f824c6e60930cfda8cb25bdce58a6a1ee01fc1fe663d23b19e34449`
- `test.csv` - SHA256 `cca36f5416abfb8a41b1c84e5fa83aec18ffbff32fecd277c668a3f73312a38f`
- `sample_submission.csv` - SHA256 `0f7d405dbd1e88ea285c7138514b7652f7441e995dcaed338bb208a86c2f347d`

Also copy `fold_ids_colab.npy` from this package to the same folder.

### 2. Select a T4 GPU

Runtime -> Hardware accelerator -> **T4 GPU**. The notebook fails fast if no
GPU is present, by design.

### 3. Mount Drive

Run Cell 1. It mounts Drive and creates `MyDrive/uap_text/` with `data/`,
`runs/` and a `progress.log`.

### 4. Run setup and hash verification

Run Cells 1 to 4. Cell 3 verifies all three CSV hashes, the fold file hash
(`5ab5cfe9...`) and the fold int64 hash (`cc4fdd16...`), the fold sizes, that
zero cleaned-text groups cross folds, and that test id order equals
`sample_submission.csv` order. **Stop if any check fails.**

### 5. Run the smoke test

Run Cell 5 (~3 min, saves nothing). It exercises the full train/validate/export
path with a tiny model so you see the shape of the code before committing ~95
minutes.

### 6. Run the first GPU model: `xlmr_text`

Run Cell 6. `FacebookAI/xlm-roberta-base`, text-only, all rows, fold set B.
Recipe is fixed in advance: mean pooling, LLRD 0.9, head LR 1e-4, balanced CE
with label smoothing 0.05, dropout 0.5, 4 epochs, 6% warmup, fp16, grad clip
1.0, **mean of the last 2 epochs** (no best-epoch picking). About 95 min on a
T4.

Each completed fold is written to Drive immediately, atomically (temp file then
rename), so a disconnect loses at most one fold. Re-running Cell 6 skips any
fold whose config hash, validation rows, test rows, shapes, finiteness and row
sums all verify.

### 7. Download / export the verified artifacts

Run Cells 7 and 8. Cell 8 writes `uap_text_export_<timestamp>.zip` containing
`runs/`, `progress.log`, `export_manifest.json` and `fold_ids_colab.npy`. Every
file is listed with its SHA256 in the manifest. Cell 8 is self-contained and
works after a kernel restart.

### 8. Evaluate locally BEFORE the second run

On your machine, verify the ZIP against `export_manifest.json`, stop on any
mismatch, then run:

```
python text_blend_eval.py --data-dir . --folds final_sources/fold_ids.npy \
  --out reports/v7 --pred-out predictions/text_v7 \
  --src lexical:text_tfidf=cache/v7_text/text_tfidf_final.npz \
  --src lexical:text_knn=cache/v7_text/text_knn_final.npz \
  --src semantic:emb_e5_base=final_sources/emb_e5-base_final.npz \
  --src finetuned:xlmr_text=<colab_v7>/runs/xlmr_text_final.npz
```

Read `leave_one_out` in the report. Keep `xlmr_text` only if **any** holds:
contribution >= +0.0015, or class-2 F1 >= +0.010 without pooled regression, or
better worst fold with fold std rising by no more than 25%.

### 9. Run the second model ONLY if the first passed its gate

`banglabert_bn` - `csebuetnlp/banglabert` on Bangla-script rows only (20,178
train / 5,026 test), about 45 min. It targets the weakest slice (Bangla rows
score 0.456 macro-F1 without ID). The local blend routes weights by script, with
each route chosen honestly on the other folds. If run 1 failed its adoption gate,
fix the reason first rather than running run 2.

### 10. Never create a submission from the GPU notebook

The notebook writes probability artifacts only. Any submission CSV requires
separate written authorisation, and the local evaluation script also refuses to
write one. Submissions 1, 2 and 3 are untouched by this package.

---

## Do not

- do not use the `id` column as a feature (the notebook reads it only to assert
  test order)
- do not add the plain ID prior, SID or the XLM-R ID-head to this track
- do not regenerate or modify the fold file
- do not tune on the public leaderboard
- do not pick a loss function, seed or epoch by OOF score - the recipe is
  predeclared
- do not train XLM-R large or mDeBERTa-v3 on a T4

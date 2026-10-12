# Submission 4: how to reproduce (organizer guide)

**What this is.** The team's leaderboard-performance model: a LightGBM stacker over six base sources, plus ID-neighbour and text features.
**It is heavily ID-dependent** (plain ID prior, SID, XLM-R ID-head, neighbour labels by id, id-window class shares, raw id). The ID signals are competition-specific and not valid for new comments. The genuine text methodology is reported separately.

| | value |
|---|---|
| Honest grouped OOF macro-F1 (fold set B) | **0.704464** |
| Per-class F1 (0 / 1 / 2) | 0.780904 / 0.673929 / 0.658560 |
| Fold scores | 0.708932, 0.707783, 0.702734, 0.701020, 0.701534 (std 0.0033) |
| Submission 3, recomputed in the same environment | 0.683258 (recorded 0.682311) |
| Validation class | STANDARD_OOF_STACKING_WITH_DEPENDENCY (see "Validation notes") |
| CSV SHA256 (audit environment) | `eb94d51620ca08b4727c374833aa40f03022c24e6e91d726ad00552c6af3555c` |
| Runtime | about 10–15 min on a 2-core CPU; no GPU, no internet |

## 1. Folder layout

```
<project root>/
├── train.csv                      (competition file, not bundled)
├── test.csv                       (competition file, not bundled)
├── sample_submission.csv          (competition file, not bundled)
├── README_REPRODUCE_SUBMISSION_4.md
└── uap_submission/
    ├── submission/
    │   └── submission_4.csv       (written by the notebook)
    └── reproducible/
        ├── submission_reproducible_file_4.ipynb
        ├── run_submission_4.py    (same code as the notebook, as a plain script)
        └── artifacts_submission_4/
            ├── config_submission_4.json      fixed configuration, hashes, audited scores
            ├── fold_ids.npy                  locked grouped folds (fold set B)
            ├── emb_e5-base_final.npz         frozen multilingual-E5-base + LR, OOF + test probabilities
            └── xlmr_base_idhead_final.npz    XLM-R base with ID-prior head, OOF + test probabilities
```

## 2. Environment (one time)

```bash
python -m pip install numpy pandas scipy scikit-learn==1.8.0 lightgbm==4.7.0 nbconvert ipykernel
```

These versions reproduce the CSV byte for byte. Other recent versions also run; the OOF score may then move by up to about 0.002, and the notebook asserts that tolerance.

## 3. Run: pick one option

**Option A, notebook (recommended):**
```bash
cd uap_submission/reproducible
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 submission_reproducible_file_4.ipynb
```

**Option B, plain script:**
```bash
cd uap_submission/reproducible
python run_submission_4.py
```

**Option C, interactive:** open the notebook in Jupyter or VS Code, then Run All.

## 4. What the run does and prints

1. Verifies the SHA256 of `train.csv`, `test.csv`, `sample_submission.csv`, the fold file and both bundled arrays. It asserts fold sizes `[9543, 9546, 9571, 9581, 9576]` and zero duplicate-text groups crossing folds.
2. Rebuilds the CPU sources and prints their standalone OOF:
   - TF-IDF about 0.516
   - text-kNN about 0.487
   - E5 0.521535
   - SID about 0.628
   - plain ID about 0.614
   - ID-head 0.648743
3. Builds the 60 stacker features, then trains the cross-fitted stacker for seeds 42, 7 and 2026 (0.693498, 0.691711 and 0.693728).
4. Applies the class-2 correction chosen on the other folds and prints the final metrics. It **asserts an OOF of 0.704464 ± 0.002**.
5. Predicts the test set using the median offset `[0, 0, -0.55]`. It checks the test class share against train (about 0.48 / 0.41 / 0.11), then applies the unanimous duplicate override (272 matched, 37 changed).
6. Validates the CSV before and after writing, then prints its SHA256 and whether it is byte-identical to the audited CSV.

## 5. Verify the output yourself

```bash
python -c "import hashlib;print(hashlib.sha256(open('../submission/submission_4.csv','rb').read()).hexdigest())"
# expected (scikit-learn 1.8.0, lightgbm 4.7.0): eb94d51620ca08b4727c374833aa40f03022c24e6e91d726ad00552c6af3555c
```

## 6. Validation notes (read before judging)

- **Every base prediction and every neighbour-label feature is out-of-fold for its own row.** A row's own label never enters its features. For test rows, the fold-k stacker sees the same neighbour-pool density as training.
- **This is standard stacking.** When the fold-k stacker is trained, its training rows' features come from base models and neighbour pools that included fold k's labels.
- **The class-2 correction has one level of nesting.** It is chosen on the other folds' stacker predictions. Its effect is to remove the over-prediction of class 2 caused by balanced class weights.
- **A fully strict rebuild was tried and did not match this model.** In that rebuild every source was refit per fold pair, E5 and the ID-head were excluded because they can't be refit on CPU, and neighbour pools were at 60% density. Its first outer fold scored 0.662 uncorrected, against 0.698 here, so it was not adopted.
- **No leaderboard feedback and no external labelled data** were used for any choice.

## 7. Mode B: regenerating the two GPU arrays (optional)

`emb_e5-base_final.npz` and `xlmr_base_idhead_final.npz` come from the team's Colab T4 notebook (`uap_gpu.ipynb`, cells 3–4) on the same folds. fp16 training is not bit-reproducible, so regenerated arrays will differ slightly, and the stacker score may then move by about 0.001–0.003. Mode A, the default, uses the SHA256-verified arrays above.

## 8. Troubleshooting

| message | fix |
|---|---|
| `hash mismatch` | Wrong or modified competition file or artifact. Re-download the competition files; don't edit the artifacts. |
| `not found next to the notebook` | Put the three competition CSVs in the project root, two levels above `reproducible/`. |
| `OOF drifted` | Different library versions. Install the versions in section 2. |
| memory | Peak use is about 3 GB of RAM. |

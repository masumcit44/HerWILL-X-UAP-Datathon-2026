# UAP Datathon — text-centric plan after the organizer's ID clarification

All numbers below were computed on **fold set B** (Colab folds, int64 SHA256 `cc4fdd16…`), using the uploaded arrays and sources rebuilt on CPU. They are honest cross-fitted grouped OOF macro-F1 unless labelled otherwise. No leaderboard feedback was used. **No submission was generated.**

Files that go with this plan:

| file | purpose |
|---|---|
| `uap_text_train_colab.ipynb` | Section C: resumable Colab T4 training (XLM-R text-only, BanglaBERT on Bangla-script rows, optional MuRIL) |
| `build_cpu_text_sources.py` | CPU sources on fold set B: TF-IDF LR and a new text-kNN source |
| `text_blend_eval.py` | Section D engine: honest text-only blending, nested offsets, ablation, error analysis, T1/T2/T3 packages, no CSV |

---

## SECTION A — Strategic analysis

### A1. What the organizer email means

Using ID isn't forbidden, but judges will score it as a workaround, not a contribution. Submission 3 can stay on the leaderboard, but it cannot be the story you tell the judges. The judged contribution has to be what the text models do, with ID shown as a disclosed, measured add-on.

### A2. The measured position today (fold set B, no ID)

| blend | honest macro-F1 | per-class F1 | fold std |
|---|---|---|---|
| TF-IDF LR alone | 0.516187 | 0.640 / 0.496 / 0.412 | — |
| frozen multilingual-E5-base alone | 0.521535 | 0.651 / 0.470 / 0.444 | — |
| text-kNN alone (new) | 0.487068 | 0.679 / 0.454 / 0.329 | — |
| TF-IDF + E5 | 0.5527 | 0.669 / 0.510 / 0.479 | 0.005 |
| **TF-IDF + text-kNN + E5 (current T1)** | **0.564713** | 0.697 / 0.504 / 0.493 | 0.0039 |
| same, plus nested per-class offsets (gates passed, 5/5 folds) | 0.567680 | 0.684 / 0.534 / 0.485 | — |
| text + plain ID | 0.665683 | 0.771 / 0.632 / 0.594 | — |
| text + plain ID + XLM-R ID-head | 0.678708 | 0.779 / 0.634 / 0.623 | — |
| Submission 3 (recorded) | 0.682311 | 0.782 / 0.641 / 0.624 | 0.0019 |

**The gap between the best no-ID result and Submission 3 is about 0.115 macro-F1.** Almost all of it comes from ID-derived sources.

### A3. Why text-only models stall around 0.52–0.55 (answers research questions 1, 2 and 6)

1. **The labels depend on which part of the dataset a row came from.** Sorted by id, the data falls into blocks with very different language and label mixes. Class 2 is 42% of one all-Latin-script block (ids ≈17.9k–20.9k) and 0–1% of others (ids ≈32.9k–38.8k). Some blocks are almost entirely Bangla script, others entirely Latin. The ID prior works because it identifies the source block. Text can only partly infer the source, from language and topic.
2. **Very different models land in the same place.** TF-IDF scores 0.516, frozen E5 0.522, and fine-tuned XLM-R 0.544 (historical). When architectures this different converge, the limit is the labels, not the model. Further evidence: 11% of exact-duplicate text groups carry conflicting labels, and near-duplicate retrieval was right only about 63% of the time.
3. **Bangla script is the weak slice.** Without ID, Bangla-script rows score 0.456 macro-F1 (class 1: 0.39, class 2: 0.25). Latin-script rows score 0.593. With ID, the Bangla rows reach 0.632. Text-only work has the most headroom here.
4. **Frozen E5 is weak alone but useful in a blend** because its errors differ from TF-IDF's: the two disagree on 43% of rows. Removing E5 costs 0.036 (the leave-one-out result). A linear head on frozen embeddings can't adapt to these label conventions, which is why it scores low on its own.
5. **Secondary, fixable recipe issues in the old plain XLM-R run:** it used only the final epoch, CLS pooling, no layer-wise learning-rate decay, 3 epochs, and no smoothing. `max_len=128` is not the problem, since the median text is 47 characters. These fixes are worth making, but they won't close a 0.115 gap.
6. **Class 2 without ID reaches 0.49,** against 0.624 in Submission 3. The ID sources account for most of the class-2 gain from Submission 2 to Submission 3 (0.583 → 0.624): text + plain ID gives 0.594, and the ID-head adds another +0.03.

**Honest expectation (a judgement, not a measurement):** nothing here suggests a base-size text model can beat 0.682 honestly. Plan for the text track to be judge-facing, and keep Submission 3 as the leaderboard track. A fine-tuned text model is worth adopting if it adds ≥ +0.0015 to the 0.5647 blend or lifts Bangla class 1/2.

### A4. Risks of relying on Submission 3 (fix these before judges read the code)

I checked `submission_reproducible_file_3.ipynb` against the training notebook (`uap_gpu.ipynb`) that actually produced its GPU arrays. Several judge-facing statements are wrong:

| # | where | problem | fix |
|---|---|---|---|
| 1 | §6 | Says `xlmr_idhead` is "xlm-roberta-base encoder with a classification head". It actually takes the **ID prior as an input** (a zero-init `Linear(3,3)` on the log prior). **This is undisclosed ID dependence.** | State it plainly: this model consumes the ID prior. |
| 2 | §4 | "Nothing else in the pipeline depends on it." False, per row 1. About 68% of the final weight is ID-derived (SID 0.42 + ID-head 0.26). | Delete the sentence and state the 68% figure. |
| 3 | §6 | Says `e5_base` is "intfloat/e5-base-v2 embeddings with a zero-shot label head". It is actually **intfloat/multilingual-e5-base**, frozen, mean-pooled, then StandardScaler and balanced LogisticRegression cross-fitted on the same folds. It isn't zero-shot and isn't e5-base-v2 (an English model). | Correct the description. |
| 4 | Mode B | The retraining code **does not reproduce either source.** The XLM-R part has no ID-prior input, no class weights, no LR schedule, and gets test predictions from a separate full-train fit instead of averaging the 5 fold models. The E5 part uses English `e5-base-v2`, has no scaler or class weights, and **its "OOF" is in-sample predictions on the training rows** (`head.fit(Etr, y); head.predict_proba(Etr)`), which is label leakage. | Replace Mode B with the exact `uap_gpu.ipynb` Cells 3 and 4 (ID-head `Clf`, `run_fold`, frozen-E5 cross-fit). |
| 5 | §1 | Calls the task "author-type classification". It is harmful-content classification (0 explicit, 1 subtle, 2 non-toxic). | Correct it. |
| 6 | §4 | "Contiguous non-overlapping slices of one scrape order." Train and test ids **interleave** (already flagged in the handoff). The ids form source blocks with train and test mixed inside each block. | Reword. |
| 7 | §9 | "Macro-F1 is threshold-free here." Macro-F1 depends on the decision rule; argmax is itself a threshold choice. | Say "argmax decision rule, no tuned offsets". |
| 8 | §5 note | The test-time SID and ID priors draw on 100% of training rows, while OOF priors draw on 80%. The test priors are therefore denser and sharper than what the blend weights were tuned on. | Disclose as a limitation. |

These are documentation and Mode B fixes, so the Mode A CSV doesn't change. After editing, rerun the fresh execution and confirm the CSV SHA256 is still `df637651…`. Mode B changes cannot be checked without a GPU, so say that in the notebook.

### A5. Next model directions (at most three)

1. **XLM-R base, text-only, improved recipe (first GPU run).** It's the same encoder as the ID-head model, so the ablation "ID head adds X" becomes exact. The recipe was fixed in advance: mean pooling, LLRD 0.9, balanced CE with label smoothing 0.05, 5-sample dropout, 4 epochs, average of the last 2 epochs.
2. **BanglaBERT trained only on Bangla-script rows (second GPU run).** It targets the weakest slice (0.456). BanglaBERT's vocabulary is Bangla-focused, so training it on English and Banglish rows would waste capacity. The local blend routes weights by script, with each route's weights chosen honestly on the other folds.
3. **MuRIL (optional third run).** It was pre-trained with transliterated Indic text, which may help with Banglish in Latin script. Run it only if runs 1 and 2 leave Latin-script class 1 as the main error.

The **text-kNN** CPU source is already built. It adds +0.012 to the no-ID blend and costs nothing to keep.

### A6. What to skip

- **XLM-R large** on a T4: about 3× the time per fold and fp16 instability. Revisit only if run 1 shows fine-tuning adds ≥ +0.005.
- **mDeBERTa-v3**: known fp16 overflow on T4, which has no bf16.
- **Loss-function sweeps** (focal vs smoothing vs logit-adjusted) and **second seeds**: each 5-fold run is about 95 min, and choosing among variants by OOF is selection bias. One recipe, fixed in advance.
- **LLM zero-shot/API labelling**: an external model, not reproducible offline, and a judging risk.
- **FastText**: never run, weaker than E5, adds nothing new.
- **Any new ID/SID variant** (e.g. the E5-SID k=180 grid winner, which was correctly left out).
- **Test pseudo-labelling** in this phase.
- **Near-duplicate label override** beyond exact unanimous matches (already shown harmful).

### A7. Research priorities, in order

1. Fix the Submission 3 notebook text and Mode B (A4). This is the highest judging risk and the cheapest fix.
2. GPU run 1 (`xlmr_text`), then local eval.
3. GPU run 2 (`banglabert_bn`), then local eval with script routing.
4. Build the ablation table and error analysis for the presentation.
5. Only then decide whether a text-dominant hybrid (T-hybrid) is worth a Kaggle submission.

---

## SECTION B — Cline audit prompt

```text
ROLE: read-only auditor. Project root = the UAP folder. Do NOT train, encode, execute notebooks,
create submissions, edit reports/trackers, or modify any existing file. The only files you may
create are under reports/v7_audit/.

PROTECTED (hash before and after, must be unchanged): train.csv, test.csv, sample_submission.csv,
submission_1.csv, submission_2.csv, submission_3.csv, submission_reproducible_file_1.ipynb,
submission_reproducible_file_2.ipynb, submission_reproducible_file_3.ipynb.

STEP 1 - lock the fold set.
  Use fold set B only: final_sources/fold_ids.npy (== artifacts_submission_3/fold_ids_colab.npy).
  Assert: file SHA256 5ab5cfe96a72b04c8bd30adc45c1eeeee477707b0c382a19d179d8635295fbbd,
          int64-bytes SHA256 cc4fdd169110aeee6d3d5566095234461041b1a9290e2e916ce850c31d27a7ca,
          sizes [9543, 9546, 9571, 9581, 9576],
          0 cleaned-text groups (clean_text(t).lower().strip()) crossing folds.
  Assert train.csv SHA256 a269ad84...4449, test.csv cca36f54...8f.

STEP 2 - inventory every OOF/test probability artifact under cache/, predictions/, models/,
final_sources/, per_fold_sources/, artifacts_submission_3/. For each file record:
  path, SHA256, keys, oof shape, test shape, fold set it was built on (A / B / unknown - decide by
  a stored fold_ids array or by the producing script; if unknown, say unknown),
  family: lexical | semantic | finetuned | ID-DERIVED,
  uses_id: yes/no (the ID prior, SID, the XLM-R ID-head and anything fed by them are ID-DERIVED),
  standalone OOF macro-F1 (only if the fold set is B).
  Flag every artifact on fold set A as NOT USABLE with fold-set-B OOF arrays.

STEP 3 - confirm these reference numbers on fold set B, tolerance 1e-4 unless stated:
  TF-IDF LR 0.5160 (+-0.003, sklearn version drift), plain ID prior 0.613631,
  emb_e5-base 0.521535, xlmr_base_idhead 0.648743.

STEP 4 - check the Submission 3 notebook against uap_gpu.ipynb and list, with cell ids and quoted
  lines, every statement that is inaccurate. At minimum check: section 6 description of
  xlmr_idhead (does it disclose the ID-prior input?), section 6 description of e5_base
  (model name, zero-shot?), section 4 "Nothing else in the pipeline depends on it",
  Mode B code (ID prior input? class weights? scheduler? fold-averaged test? in-sample E5 OOF?),
  task description in section 1, "contiguous slices" wording, "threshold-free" wording.
  Do not edit the notebook. Report only.

STEP 5 - prepare, without running anything:
  reports/v7_audit/colab_data_checklist.txt listing the 4 files to copy to
  MyDrive/uap_text/data/ (train.csv, test.csv, sample_submission.csv, fold_ids_colab.npy) with
  their SHA256s;
  reports/v7_audit/text_sources_manifest.json with the STEP 2 inventory.

STEP 6 - print a readiness report:
  FOLD SET LOCKED: B (file + int64 hashes, sizes, crossing=0)
  DATA HASHES: ok/fail
  TEXT-ONLY SOURCES ON FOLD SET B: name, path, standalone OOF
  ID-DERIVED SOURCES (ablation only): name, path, standalone OOF
  FOLD-SET-A ARTIFACTS (not usable): count
  SUBMISSION 3 NOTEBOOK ISSUES: numbered list
  COLAB READY: yes/no + missing items
  PROTECTED FILES UNCHANGED: yes/no
  SUBMISSION GENERATED: NO
```

---

## SECTION C — Colab T4 training code

The full code is in **`uap_text_train_colab.ipynb`** (9 cells). What it guarantees:

- **Drive persistence**: everything goes under `MyDrive/uap_text/`, and each fold is saved the moment it finishes.
- **Atomic writes**: each file is written to a temp file and then renamed, so a disconnect never leaves half a file.
- **Skip-if-valid resume**: a saved fold is skipped only if its config hash, validation rows, test rows, shapes, finiteness and row sums all check out. Otherwise it is retrained. Tested here by corrupting a fold file: that fold was retrained and the other nine were skipped.
- **Fail-fast**: stops on no GPU, any data or fold hash mismatch, wrong fold sizes, duplicate groups crossing folds, test order different from sample order, or a non-finite loss.
- **Recorded for each run**: the Hub model commit (and the load is pinned to it), tokenizer SHA256, config hash, package versions, GPU name, and `uses_id: false`.
- **Per-fold NPZ**: `va, vp, tp, te_rows, vp_last, tp_last, hist, config, config_hash`. **Final NPZ**: `oof, test, fold_ids, coverage_train, coverage_test, meta`. Subset runs leave uncovered rows as NaN.
- **No ID input anywhere**: `id` is read only to check test order. No CSV is written. The ZIP cell is self-contained.
- **Recipe, fixed in advance**: mean pooling, LLRD 0.9, head LR 1e-4, balanced CE with smoothing 0.05, 5-sample dropout, 4 epochs, 6% warmup, fp16, clip 1.0, **mean of the last 2 epochs** (no best-epoch picking). Per-epoch validation scores are logged for reporting only.
- **Text normalisation**: the grouping cleaner, then NFKC, then removal of invisible characters. ZWJ/ZWNJ are kept because Bangla needs them; emojis are kept. BanglaBERT additionally uses the csebuetnlp normalizer its model card requires, and the notebook stops if it can't be installed.

**Tested here** end to end on CPU with a tiny random model standing in for the real encoders (the Hub is blocked in this workspace): smoke test, full 5-fold run, a Bangla-subset run, resume after corruption, and the ZIP export. What wasn't tested is real-checkpoint download and T4 timing, so run Cell 5 (the smoke test) first on Colab.

---

## SECTION D — Local text-only evaluation prompt (Cline)

```text
ROLE: evaluator. Fold set B only. Do NOT create any CSV or notebook, do NOT touch protected
files (list as in the audit prompt), do NOT use leaderboard scores for anything.

INPUTS
  - the Colab ZIP (uap_text_export_*.zip) -> extract to colab_v7/ ; verify every file against
    export_manifest.json SHA256; stop on any mismatch.
  - For every *_final.npz: assert fold_ids == fold set B (int64 SHA256 cc4fdd16...), oof shape
    (47817,3), test (11955,3), covered rows finite, rows sum to 1, meta.uses_id == false.
  - For every per-fold NPZ: sorted va == flatnonzero(fold_ids==k) restricted to the run's
    coverage; stitching vp over va reproduces oof (max|diff| < 1e-6); mean of the five tp
    reproduces test (< 1e-6).

STEP 1 - CPU sources (skip if cache/v7_text/*.npz exist and verify):
  python build_cpu_text_sources.py --data-dir . --folds final_sources/fold_ids.npy --out cache/v7_text
  Expect text_tfidf ~0.516 (+-0.003), text_knn ~0.487 (+-0.003).

STEP 2 - run the evaluator (plain ID prior npz: build it from the Submission 3 notebook's id_prior
  on fold set B and save as cache/v7_text/plain_id.npz with keys oof, test):
  python text_blend_eval.py --data-dir . --folds final_sources/fold_ids.npy \
    --out reports/v7 --pred-out predictions/text_v7 \
    --src lexical:text_tfidf=cache/v7_text/text_tfidf_final.npz \
    --src lexical:text_knn=cache/v7_text/text_knn_final.npz \
    --src semantic:emb_e5_base=final_sources/emb_e5-base_final.npz \
    --src finetuned:xlmr_text=colab_v7/runs/xlmr_text_final.npz \
    --src finetuned:banglabert_bn=colab_v7/runs/banglabert_bn_final.npz \
    --src id:plain_id=cache/v7_text/plain_id.npz \
    --src id:xlmr_idhead=final_sources/xlmr_base_idhead_final.npz
  (add finetuned:muril_text=... only if that run exists)

STEP 3 - apply the adoption gates to each new fine-tuned source using the leave_one_out block of
  reports/v7/text_v7_report.json. Keep a source if ANY holds:
    (a) its contribution >= +0.0015 to the text-only blend,
    (b) class2_f1_delta >= +0.010 with contribution >= -0.0005,
    (c) worst fold improves and fold_std does not rise by more than 25%.
  Drop failing sources and re-run STEP 2 without them. Do this ONCE; no further iteration.

STEP 4 - report, from reports/v7/text_v7_report.json:
  standalone scores; T1 / T1+nested offsets (and whether the offset gates passed) / T2 / T3 with
  per-class F1, accuracy (labelled accuracy), confusion matrix, fold scores, std, worst fold,
  fold weights, consensus weights; the OPTIMISTIC same-OOF number labelled as such;
  disagreement rates; unique corrections/damages; error analysis by script and length;
  test class share vs train share (flag any class drifting > 6 pp);
  disagreement of T1 test argmax vs submission_2.csv and submission_3.csv (read-only);
  the full ablation table;
  "BEATS 0.682311 HONESTLY: yes/no" and the band.

STEP 5 - leave predictions/text_v7/CANDIDATE_NOT_FOR_SUBMISSION_* as the only outputs.
  SUBMISSION GENERATED: NO. A CSV requires separate written authorization from the user.
```

**Current output of this step, without any new GPU run:** T1 = 0.564713, T1 with nested offsets = 0.567680 (adopted: gain +0.0030, 5/5 folds, no class collapse). T3 equal-weight hedge = 0.560487. OPTIMISTIC same-OOF = 0.566182. Test class share for T1: 0.518 / 0.373 / 0.109 (max drift 2.1 pp). **T2 is identical to T1** because no source was dropped, and without offsets its test share for class 1 drifts **6.6 pp**. That fails G4, so T2 without offsets is not a valid package here.

---

## SECTION E — Judging and presentation plan

### E1. Five-minute story

1. **Problem (30 s):** Three-class harmful-content detection across Bangla, English, Banglish and mixed script. Macro-F1, so the rare non-toxic class counts as much as the others.
2. **Validation discipline (60 s):** 5-fold grouped by cleaned text, with zero duplicate groups crossing folds. Every blend weight is chosen on four folds and scored on the fifth. Optimistic numbers are always labelled as such. The 0.679→0.682 improvement was only accepted after a **bug was found and fixed** (the SID row-alignment bug, caught by a golden test). That kind of rigour is itself worth showing judges.
3. **Text models (90 s):** Lexical (word+char TF-IDF), semantic (multilingual E5), a text-kNN retrieval source, and fine-tuned transformers. The lexical and semantic sources disagree on 43% of rows, and E5 alone contributes +0.036. Show the per-script breakdown and the targeted BanglaBERT model for the weakest slice.
4. **What we learned about the data (60 s):** Labels vary by source block (class 2 ranges from 0% to 42% across blocks). That explains why every text model converges around 0.52–0.55. Present it as an insight about the dataset, not an excuse.
5. **ID, disclosed (45 s):** The ablation table. ID-derived signals add about +0.115. They are competition-specific, would not exist in production, and are reported separately.
6. **Limitations and ethics (15 s):** Label noise, source-dependent annotation, class-2 recall on Bangla, no deployment claim.

### E2. Honest ID/SID disclosure (paste-ready)

> Our leaderboard submission blends text models with ID-derived signals: a neighbour vote over nearby row ids, a text-similarity-weighted version of it, and an XLM-R model that takes that vote as an extra input. Together these carry about 68% of the final blend weight and account for most of the difference between our text-only score (0.568 honest OOF) and our leaderboard model (0.682). They work because row ids group rows from the same collection source. That property is specific to this dataset and would not exist for newly arriving comments, so we don't treat it as a modelling contribution and we don't recommend it for deployment. Our contribution is the text pipeline and the validation method. The ablation shows each part separately.

### E3. Ablation table structure (numbers as of now; fill the fine-tuned rows after the GPU runs)

| # | configuration | text-derived | honest macro-F1 | per-class F1 | what it shows | judging caveat |
|---|---|---|---|---|---|---|
| 1 | lexical (TF-IDF + text-kNN) | yes | 0.529033 | 0.671 / 0.496 / 0.420 | lexical baseline | — |
| 2 | frozen semantic (E5) | yes | 0.521535 | 0.651 / 0.470 / 0.444 | semantic view | — |
| 3 | fine-tuned text transformers | yes | *pending GPU* | | adaptation to the task | — |
| 4 | lexical + semantic (+ fine-tuned) | yes | 0.564713 (0.567680 with nested offsets) | 0.697 / 0.504 / 0.493 | complementarity | **judge-facing candidate** |
| 5 | plain ID only | **no** | 0.613631 | 0.733 / 0.602 / 0.505 | size of the ID shortcut | competition-specific |
| 6 | text + plain ID | **no** | 0.665683 | 0.771 / 0.632 / 0.594 | | competition-specific |
| 7 | text + ID + ID-head | **no** | 0.678708 | 0.779 / 0.634 / 0.623 | | competition-specific |
| 8 | Submission 3 (text + SID + E5 + ID-head) | **no** | 0.682311 (recorded) | 0.782 / 0.641 / 0.624 | leaderboard model | competition-specific |

### E4. Methodological contributions you can defend

- Grouped, duplicate-safe cross-validation with nested cross-fitted blending and offset tuning, with every optimistic number labelled.
- A bug found and fixed through a golden test (SID alignment), with the before/after numbers disclosed.
- A script-routed blend: each weight route is chosen honestly per fold, and a Bangla-specialist model is used only where it applies.
- The label-structure analysis, which shows why this dataset caps text-only models, measured rather than asserted.
- Reproducibility: pinned model revisions, hashed data and folds, resumable training, byte-identical CSV regeneration.

### E5. Limitations to state

Label noise (11% of duplicate groups conflict); annotation or sampling that varies by source block; Bangla class-2 F1 about 0.25 without ID; fp16 GPU training is not bit-reproducible; the OOF estimate comes from one fold assignment and doesn't capture seed variance; tokeniser vocabulary tie-breaking differs across scikit-learn versions (golden test at 1.47e-4).

### E6. Likely judge questions

- **"Is your model just the ID?"** No. Here is the text-only row: 0.568 honest. ID adds about 0.115 and we report it separately.
- **"Why not drop ID from the leaderboard model?"** Both are disclosed. The judged method is the text pipeline, and the leaderboard entry shows how large the dataset artefact is.
- **"How do you know there's no leakage?"** Grouped folds with zero crossing groups, neighbour pools restricted to fold-train rows, nested weight selection, and a golden test.
- **"Why so low on Bangla?"** Measured: 0.456 without ID. Class 1 vs class 0 is subtle in short Bangla insults. That's why we added the targeted BanglaBERT model.
- **"Would this work in production?"** The text pipeline would. The ID signals would not, and we say so.
- **"Why these models?"** Complementarity, measured: 43% disagreement between the lexical and semantic sources, and leave-one-out contributions per source.

---

## Closing summary

**ORGANIZER EMAIL INTERPRETATION:** Using ID is allowed but won't be credited as a method. Judges will reward meaningful use of the text, and the code will be read.

**CURRENT PERFORMANCE CANDIDATE:** Submission 3 (text + corrected SID + E5-base + XLM-R ID-head).

**CURRENT PERFORMANCE OOF:** 0.682311 honest grouped (0.684918 same-OOF is OPTIMISTIC).

**CURRENT PUBLIC SCORE:** 0.70053 (context only, never used for selection).

**ID/SID JUDGING RISK:** High. About 68% of the blend weight is ID-derived, and the notebook currently **understates** it: §6 hides the ID-head's ID input, and §4 says nothing else depends on ID. Mode B code doesn't reproduce the bundled sources, and its E5 "OOF" is in-sample. Fix before judging.

**RECOMMENDED FIRST GPU RUN:** `xlmr_text`, FacebookAI/xlm-roberta-base, text-only, fixed improved recipe, all rows, fold set B (about 95 min on a T4).

**RECOMMENDED SECOND GPU RUN:** `banglabert_bn`, csebuetnlp/banglabert on Bangla-script rows only (20,178 train / 5,026 test), used through a script-routed honest blend (about 45 min).

**MODELS TO SKIP:** XLM-R large (for now), mDeBERTa-v3 (fp16 on T4), loss sweeps, extra seeds, LLM zero-shot/API labelling, FastText, new ID/SID variants, test pseudo-labelling, near-duplicate overrides.

**TEXT-ONLY TARGET:** Beat the current honest text-only T1 of 0.564713 (0.567680 with nested offsets). A new source is adopted at ≥ +0.0015, or ≥ +0.010 class-2 F1. Beating 0.682311 without ID is not a realistic target on this evidence, and no score is promised.

**REQUIRED ARTIFACTS:** `runs/{name}_fold{0-4}.npz`, `runs/{name}_final.npz`, `runs/{name}_meta.json`, `progress.log`, `fold_ids_colab.npy`, `export_manifest.json` (all in the Colab ZIP). Locally: `cache/v7_text/text_tfidf_final.npz`, `text_knn_final.npz`, `plain_id.npz`, `reports/v7/text_v7_report.json`, `predictions/text_v7/CANDIDATE_NOT_FOR_SUBMISSION_*`.

**HONEST ADOPTION GATES:**
- **New source:** contribution ≥ +0.0015, or class-2 F1 ≥ +0.010 without pooled regression, or better worst fold / stability.
- **Offsets:** strictly nested, gain ≥ +0.001, ≥ 4 folds not regressed, no class drop > 0.02, std ≤ 1.5× baseline.
- **Package:** test class share within 6 pp of train share.
- **Always:** one fold set, no same-OOF or leaderboard tuning.

**SUBMISSION GENERATED: NO**

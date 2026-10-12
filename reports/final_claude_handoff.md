# HerWILL X UAP Datathon 2026 - Final Project Handoff

Generated 2026-09-28 by direct inspection of `D:\Desktop\UAP`. Read-only task: no
experiment was run, no model trained or encoded, no notebook executed, no
submission created or modified, no report altered, no tracker updated, no
background process started, nothing submitted to Kaggle. The only file written
is this document.

**Precedence rule used throughout:** where chat history and on-disk evidence
disagree, the disk wins. Material discrepancies found on disk are flagged in
*Known Caveats* rather than smoothed over.

# Project

Three-class author-type classification over social-media comment text. Given a
`text` and a numeric `id`, predict `y`. Metric is **macro-averaged F1**, so all
three classes carry equal weight; class 2 is the rare class and dominates the
remaining headroom.

Class semantics (from `reports/small_tfidf_summary.md`):
`0 = explicitly_toxic` (23,756 rows), `1 = subtly_toxic` (18,163),
`2 = non_toxic` (5,898).

Three submissions exist, each a blend of probability sources, each packaged with
a reproducible notebook. Score progression was driven by honest grouped OOF
evaluation, never by public leaderboard feedback.

# Organizer Clarification

No organizer clarification document exists on disk (searched `*.md`, `*.txt`,
`*.json` for "organizer", "workaround", "meaningful use", "comment text" - no
match). The following is user-supplied and is recorded as such:

* IDs may be considered a workaround.
* ID usage is not viewed as a genuine contribution to the intended solution.
* Judges will emphasize meaningful use of comment text.
* Leaderboard rank alone does not guarantee winning.
* Code and methodology will be reviewed.

Strategic consequence:

* Submission 3 remains the strongest performance candidate.
* ID/SID dependence must be fully disclosed in the write-up. Disclosure already
  exists in the Submission 3 notebook (sections 4, 9, 10) and must be preserved.
* The next main research direction must be text-centric.
* At least one no-ID or text-dominant candidate should be prepared for judging.

# Dataset

| Property | Value |
|---|---|
| `train.csv` | 47,817 rows, columns `[id, text, y]`, 5,357,729 bytes |
| `test.csv` | 11,955 rows, columns `[id, text]`, 1,290,735 bytes |
| `sample_submission.csv` | 11,955 rows, columns `[id, y_pred]` |
| Class counts | 0: 23,756 / 1: 18,163 / 2: 5,898 |
| Class share | 0.4968 / 0.3798 / 0.1233 |
| Missing values | none in any file or column |
| `id` uniqueness | unique in both train and test |
| train/test `id` overlap | **0 rows** |
| train `id` range | 0 - 59,771 |
| test `id` range | 4 - 59,670 |
| Duplicate groups | 47,157 groups; 480 duplicate groups; 1,140 rows in duplicates; max group 18 |

Duplicate group key: `clean_text(text).lower().strip()` where `clean_text` maps
URLs to ` URL `, `@mentions` to ` USER `, collapses whitespace, strips.

**Correction to a claim repeated in earlier project docs.** `reports/text_id_blend_summary.md`
(line 160) and the Submission 3 notebook both describe train/test ids as
"contiguous non-overlapping slices of one scrape order". Direct inspection
shows this is **not** accurate: the two id sets **interleave**. Union of ids is
59,772 covering 0-59,771; the sorted test ids begin 4, 15, 17, 20, 24, 32, 33,
49, ...; there are 19,135 contiguous runs with mean run length 3.12. The ids are
an interleaved partition of one scrape order. The ID-locality prior still works
because near neighbours in id space are still near neighbours in scrape time,
but the "contiguous slices" description should be corrected in any judge-facing
write-up.


# Validation and Fold Sets

Official metric: **macro F1**, pooled OOF, with grouped folds.

Grouping key for all grouped work: `clean_text(text).lower().strip()`.

## Fold set A - local canonical (v2/v3/v4)

* Path: `cache/v2/fold_ids.npy`
* dtype/shape: int64, (47817,)
* Sizes: `[9564, 9564, 9563, 9563, 9563]`
* **File** SHA256: `2bd8cfdedb32342931d17cc27ca0a1be99d8a567bb8128e32955ec18cd91880d`
* **int64-array-bytes** SHA256: `925065f4d4a0c6a533fad774f4875b039ea526eb915bee8c8c8b819c88760f56`
* Used by: v2 experiments, v3 experiments, v4 fast-path, Submissions 1 and 2,
  all `predictions/v2`, `v3_candidates`, `v4_candidates`.

**Naming trap:** the value `925065f4...` is the **int64-array-bytes** hash, not
the file hash. Several reports (`reports/v4/source_verification.json`,
`reports/v3_configs/v3_frozen_emb_metrics.json`) label it "fold_sha256" and call
it "canonical". It is correct as an array-content hash. Do not compare it to a
file hash.

## Fold set B - recovered Colab (v5/v6, Submission 3)

* Path (zip): `uap_complete_export_20260928_064140.zip` -> `final_sources/fold_ids.npy`
* Path (bundled): `uap_submission/reproducible/artifacts_submission_3/fold_ids_colab.npy`
* Path (extracted): repo root `fold_ids.npy` (byte-identical to the bundled copy)
* Sizes: `[9543, 9546, 9571, 9581, 9576]`
* **File** SHA256: `5ab5cfe96a72b04c8bd30adc45c1eeeee477707b0c382a19d179d8635295fbbd`
* **int64-array-bytes** SHA256: `cc4fdd169110aeee6d3d5566095234461041b1a9290e2e916ce850c31d27a7ca`
* Used by: v5, v6, Submission 3 only. Produced by the original Colab run.

## Hard rule

**No OOF probabilities from different fold sets may be combined.** Fold set A
and fold set B are different partitions; an OOF vector from one is not
comparable to, or blendable with, an OOF vector from the other. This is exactly
why v4 could not consume the GPU arrays (`reports/v4/fast_path_final_report.txt`
line 13 lists all four GPU sources as MISSING), and why v6 rebuilds text and ID
sources on the Colab folds rather than reusing v5/v2/v4 caches.

## Group-crossing verification

* Fold set A: `reports/revalidation_grouped_metrics.json` -> `groups_spanning_folds: 0`
  (n_groups 47,157, duplicate groups 480, rows in duplicates 1,140).
* Fold set B: `reports/v6/step123.json` plus a live check during the v6 run ->
  **0** cleaned-text groups cross folds.

Both verified as zero.


# Submission History

Note on file locations, verified on disk: Submission 1 is at
`uap_submission/reproducible/submission/submission_1.csv`; Submissions 2 and 3
are at `uap_submission/submission/`. Sub 1 sits in a different directory than
the other two.

## Submission 1

| Field | Value |
|---|---|
| Honest local OOF | **0.639579** (grouped, `reports/revalidation_grouped_metrics.json`); 0.642185 on plain StratifiedKFold (`reports/text_id_blend_summary.md`) |
| Public leaderboard (user-supplied) | 0.65662 |
| Sources | TF-IDF word(1,2)+char_wb(3,5) LR (0.512943 standalone) + k=20 ID prior (0.614135 standalone) |
| Blend weights | alpha_text 0.40 / alpha_id 0.60, single scalar, chosen on pooled OOF |
| Per-class F1 | 0.75727 / 0.620188 / 0.541279 |
| CSV | `uap_submission/reproducible/submission/submission_1.csv` |
| SHA256 | `1b173503d8fdd0e4a3f4dad4e38f743dbca9ae10e0ed460216399740fee40a50` |
| Notebook | `uap_submission/reproducible/submission_reproducible_file_1.ipynb` |
| SHA256 | `6366a78d5a4e08e05a496bd563a374f6d168e28c2b5310fd4cb57c6baf780d56` |
| Reproducibility | Executed in place: 12 cells (7 code, 5 md), 7/7 code cells have execution_count, 0 error outputs |
| Prediction distribution | 6,554 / 4,361 / 1,040 = 0.5482 / 0.3648 / 0.0870 |

## Submission 2

| Field | Value |
|---|---|
| Honest local OOF | **0.657612** (`logs/v3_harden_candidate.log`, refined 3-way) |
| Public leaderboard (user-supplied) | 0.67256 |
| Sources | dual-text TF-IDF ensemble (base 0.512943 + all3-cleaned 0.518947) + k=20 ID prior + frozen intfloat/multilingual-e5-small (0.496241 standalone) |
| Blend weights | v3 refined 3-way kept text 0.2 / id 0.4 / e5 0.4 on 4 of 5 folds; the v4 fast-path median-consensus (0.45 text / 0.1111 id / 0.1 e5_small / 0.35 e5_base) superseded it but was never submitted |
| Per-class F1 | 0.765439 / 0.624523 / 0.582876 |
| Fold scores | 0.663779, 0.645736, 0.659119, 0.660625, 0.658759 |
| Fold std | 0.006192 |
| CSV | `uap_submission/submission/submission_2.csv` |
| SHA256 | `894ce6c210ccc0f2e956c1e6d32fcb82189f1798a6993ea4646555f93bfae40e` |
| Notebook | `uap_submission/reproducible/submission_reproducible_file_2.ipynb` |
| SHA256 | `f3b8f60883dfa0e659b86e07d4be7ffda2ce5b1048904891e21ad1bb1efe9597` |
| Notebook state | 26 cells (15 code, 11 md), **0 executed cells**, 0 error outputs - shipped unexecuted |
| Bundled artifacts | `uap_submission/reproducible/models/multilingual-e5-small/` - 10 files, 470.4 MB, with `multilingual-e5-small_SHA256.json` |
| Fresh execution | PASS. `_fresh_exec_test/exec_log.txt` shows `nbconvert --execute --inplace` wrote 98,563 bytes; regenerated CSV **byte-identical** to packaged. `_preflight/pf_log.txt` corroborates. |
| Disagreement vs Sub 1 | **1,212 / 11,955 = 10.14%** |

## Submission 3

| Field | Value |
|---|---|
| Honest local OOF | **0.682311** (`reports/v6/blend.json`) |
| Public leaderboard (user-supplied) | 0.70053 |
| Sources | text, corrected SID, E5 Base, XLM-R ID-head |
| Final weights | 0.1053 / 0.4211 / 0.2105 / 0.2632 (median of 5 cross-fitted fold vectors, renormalised) |
| Per-class F1 | 0.782202 / 0.640594 / 0.624137 |
| Accuracy | 0.710730 |
| Fold scores | 0.682243, 0.685245, 0.680023, 0.680668, 0.683172 |
| Fold std | 0.001859 (worst fold 0.680023) |
| Plain-ID blend comparison | 0.677401 |
| Corrected SID contribution | **+0.004910** over the plain-ID blend |
| CSV | `uap_submission/submission/submission_3.csv` |
| SHA256 | `df637651e087e3d667c8552a37bef9c37b36e2ec9af41a4558b641495196edaa` |
| Notebook | `uap_submission/reproducible/submission_reproducible_file_3.ipynb` |
| SHA256 | `b7bcc940053c2af623530978e84947a56e1747d98758baf046dea5af505fc65c` |
| Bundled artifacts | `uap_submission/reproducible/artifacts_submission_3/` - 9 files + manifest |
| Fresh execution | PASS. Clean temp dir, `nbconvert --execute --inplace`, **0 cell errors** |
| Byte-identical CSV | **YES** - `df637651...` both packaged and regenerated |
| Duplicate override | 272 rows matched, 42 labels changed, 53 conflicting groups skipped |
| Prediction distribution | 6,274 / 4,232 / 1,449 = 0.5248 / 0.3540 / 0.1212 |
| Disagreement vs Sub 2 | **1,427 / 11,955 = 11.94%** (vs Sub 1: 1,780 = 14.89%) |
| **PACKAGE READY** | **YES** |

# Public Leaderboard Results

User-supplied leaderboard scores, confirmed consistent with local files. These
are **not** model-selection criteria and were not used for tuning.

| Submission | Public score (user-supplied) |
|---|---|
| Submission 1 | 0.65662 |
| Submission 2 | 0.67256 |
| Submission 3 | 0.70053 |

Final leaderboard selections currently shown: **0 / 2 selected**.

Visible public rank 1 was approximately **0.77157** in the latest screenshot.

The public leaderboard uses only a subset of test data. Public scores must not
be used for model tuning. Do not recommend final selection on public score
alone.


# Verified Model Sources

## Submission 3 (fold set B) - all independently re-verified on disk

| Source | OOF shape | Test shape | Standalone OOF | Status |
|---|---|---|---|---|
| `xlmr_base_idhead` | (47817, 3) | (11955, 3) | **0.648743** (diff 2.36e-07) | VERIFIED |
| `emb_e5-base` | (47817, 3) | (11955, 3) | **0.521535** (diff 1.55e-07) | VERIFIED |
| text (TF-IDF LR) | (47817, 3) | - | 0.516093 | VERIFIED |
| plain k=20 ID prior | (47817, 3) | - | 0.613631 | VERIFIED |
| SID config A (corrected) | (47817, 3) | - | 0.627895 | VERIFIED |

GPU sources: Colab T4, fp16. `metadata/export_meta.json` records
sklearn 1.6.1, torch 2.11.0+cu128, transformers 5.16.1, seed 42.

**`xlmr_base` (plain, no ID-head) standalone OOF = 0.544479** is recorded in
`metadata/complete_export_manifest.json` as `expected_source_scores`, but its
artifact `xlmr_base_final.npz` is listed in `preferred_files_missing` and is
**absent from disk**. This is a HISTORICAL number only - the probabilities are
gone and it cannot be rebuilt without a GPU. Both `uap_export.zip` (4 entries)
and `uap_complete_export_20260928_064140.zip` (13 entries) were read without
modification; `testzip()` clean.

Per-fold stitch check passed exactly: sorted `va` of each
`xlmr_base_idhead_fold{k}.npz` equals `flatnonzero(fold_ids==k)`, and stitching
`vp` over `va` reproduces the combined `oof` with max|diff| **0.00e+00**; mean of
the five `tp` reproduces `test` with max|diff| **0.00e+00**.

## Submission 2 (fold set A) - CPU, fully reproducible

| Source | Standalone OOF | Status |
|---|---|---|
| base TF-IDF + LR | 0.512943 | VERIFIED |
| all3-cleaned TF-IDF + LR | 0.518947 | VERIFIED |
| k=20 ID prior | 0.614135 | VERIFIED |
| frozen multilingual-e5-small + LR | 0.496241 | VERIFIED |
| fast-path e5_base (frozen) | 0.515829 | VERIFIED |

The e5-small checkpoint is bundled locally (470.4 MB) with a SHA256 sidecar, so
Submission 2 needs no internet.


# Experiment Timeline

## A. Classical text models

| Experiment | OOF macro F1 | Delta vs 0.639579 | Classification |
|---|---|---|---|
| Small-TFIDF LR, 3-fold grouped (`small_tfidf_summary.md`) | 0.520652 | - | MARGINAL (superseded protocol) |
| `ref_text_lr_only` 5-fold grouped | 0.512943 | -0.126636 | USEFUL (component) |
| Text ensemble LR + CalibratedLinearSVC + ComplementNB (`v2_1_ens_equal_alpha0.40`) | 0.632463 | -0.007116 | REJECTED |
| `v2_1_ens_equal_alpha_best0.45` | 0.634323 | -0.005256 | REJECTED |
| `v2_1_ens_simplex_tuned_alpha0.45` | 0.639421 | -0.000158 | REJECTED |
| 9 hand-crafted dense features, scale 0.1, alpha 0.35 (`v2_2`) | 0.638968 | -0.000611 | REJECTED |
| all3-cleaned text (`v3_6_text_cleaning_all3`) | 0.640569 | +0.000990 | MARGINAL |
| all3 text-only | 0.518947 | +0.006004 vs text anchor | USEFUL (component) |

The text ensemble (LR + `CalibratedClassifierCV(LinearSVC, sigmoid, cv=5)` +
`ComplementNB` on word TF-IDF) was implemented and run; it never beat the plain
LR text model. **NB-SVM and character-only variants: no implementation or result
exists on disk** - they are not part of the completed history.

Per-fold text-model numbers and the liblinear/`OneVsRestClassifier` compatibility
note are in `reports/small_tfidf_summary.md`.

## B. ID-locality work

| Experiment | OOF | Delta | Classification |
|---|---|---|---|
| k=20 inverse-distance ID prior alone | 0.614135 | -0.025444 | USEFUL (component) |
| text + ID, alpha 0.40 (LOCKED BASELINE) | 0.639579 | 0.0 | USEFUL, superseded |
| `v3_1a_multiscale_id_k30` | 0.639179 | -0.000400 | REJECTED |
| learned ID, 49 features + scaled LR | 0.635429 | -0.004150 | REJECTED |
| learned ID, 49 features + LightGBM | 0.634206 | -0.005373 | REJECTED |
| per-class alpha, tuned on scored OOF (`v2_3_perclass_lr_tuned_on_oof`) | 0.640810 | +0.001231 | NOT ADOPTED (tuned on scored rows) |
| per-class alpha, cross-fitted (`v2_3_perclass_lr_crossfitted`) | 0.639812 | +0.000233 | MARGINAL (within fold noise 0.0038) |
| meta-stacker LR (`v2_4_meta_textidhand`) | 0.625762 | -0.013817 | REJECTED |
| meta, text+hand, no ID | 0.503458 | -0.136121 | REJECTED |
| meta, ID+hand, no text | 0.597552 | -0.042027 | REJECTED |
| `v2_5_combined_best` | 0.639812 | +0.000233 | REJECTED (gate: no survivor beat baseline on both tests) |

`reports/v3_id_structure/id_leakage_proof.json`: ID-prior leak-free verified for
k=1, 5, 20, 80 - all folds disjoint, max|diff vs brute force 0.0 to 2.22e-16.

## C. Retrieval and cleaning

| Experiment | OOF | Delta | Classification |
|---|---|---|---|
| near-duplicate char n-gram retrieval, thr 0.9 | 0.638330 | -0.001249 | REJECTED |
| near-duplicate thr 0.95 / 0.99 | 0.638960 / 0.639454 | -0.000619 / -0.000125 | REJECTED |
| exact duplicate override (unanimous, conflicting skipped) | applied in all 3 submissions | - | USEFUL (kept) |

Near-duplicate retrieval fails because paraphrases carry opposing annotations:
retrieval accuracy ~0.63 versus baseline accuracy ~0.77 on the same subset, so
the false-override rate is ~72% (thr 0.8: 362 false vs 128 correct).

Cleaning A/B, 8 variants, text-only column (from `reports/t0_cleaning_metrics.json`):
base 0.512943; nfkc 0.514276; repeat 0.513814; emoji 0.513634;
nfkc_repeat 0.516345; nfkc_emoji 0.516332; repeat_emoji 0.517492;
**all3 (nfkc+repeat+emoji) 0.518947, +0.006004 - best text-only**.
Blend-0.40 column: base 0.639579; all3 0.640569 (+0.000990).
Gates: text TRUE, blend FALSE. Status MARGINAL - text lifts meaningfully, blend
does not exceed the 0.005 noise band.

## D. Embedding models

| Model | Standalone | Blended | Classification |
|---|---|---|---|
| multilingual-e5-small, frozen, CPU | 0.496241 | 3-way honest 0.655701; refined honest **0.657612** | USEFUL, superseded |
| e5-base, frozen, CPU (v4 fast path) | 0.515829 | Candidate A 0.663412, B 0.664275, C 0.662660 | USEFUL, superseded |
| e5-base, GPU-trained (Submission 3) | 0.521535 | part of 0.682311 | VERIFIED (in winner) |
| FastText, from scratch per fold | - | - | **NOT RUN** |

E5-small contributed +0.016122 honestly despite being the weakest standalone
signal; 46.55% disagreement with the locked blend, +19,273 corrected / -9,232
damaged. Its value is diversity.

**FastText: only `src/experiment_v3_4_fasttext_embeddings.py` exists. No
results file, no metrics JSON, no OOF/test arrays, no model directory, and the
`fasttext` package is not installed.** The claims in `prompt.txt`
(FastText-only OOF 0.428033, text+ID+FastText 0.641974, weights 0.30/0.50/0.20)
were never verified and remain unverified claims. Treat as NOT RUN.


## E. Transformer work

| Model | Result | Status |
|---|---|---|
| plain XLM-R Base | 0.544479 (manifest) | **HISTORICAL / ARTIFACT LOST** - `xlmr_base_final.npz` absent |
| XLM-R Base + ID-head | 0.648743 recovered | VERIFIED |
| E5-base GPU zero-shot head | 0.521535 | VERIFIED |
| lost historical GPU blend figure ~0.678567 | referenced in v5 comparison | **HISTORICAL / ARTIFACT LOST** |

Recovery trail: four archives recovered from the Recycle Bin into `recovered/`
with hashes in `recovered/RECOVERED_HASHES.json` (`UAP.zip` 214,943,384 bytes,
`HerWILL-Safe-Social-Media-Datathon-2026.zip`, two Drive downloads). Scripts
`scripts/phase1*.py`, `scripts/phase2_verify_sources.py`,
`scripts/phase3_e5base_feasibility.py`. GPU source training code:
`uap_gpu.ipynb`, `colab/train_xlmr_colab.py`, `colab/canonical_train_folds.csv`.

## F. SID work

| Stage | Result | Status |
|---|---|---|
| First SID (v5), config A | 0.612153 | **REJECTED - alignment bug** |
| v5 SID B / C | 0.606312 / 0.612612 | rejected with A |
| v5 blend (buggy SID) | 0.678057, worst fold 0.672756 | gates FAILED, no CSV written |
| Corrected golden test | max abs diff 1.47e-04 vs 1e-4 requested | diagnosed, see *Known Caveats* |
| Corrected SID config A (v6) | **0.627895** | VERIFIED |
| Corrected SID B / C | 0.626534 / 0.627606 | sensitivity, reporting only |
| Corrected 4-source blend | **0.682311** | **VERIFIED WINNER** |
| Plain-ID comparison blend | 0.677401 | reference |
| Optional sid_e5 beta=12 k=90 | 0.683437, +0.001126, 3/5 folds | NOT ADOPTED (fails both criteria) |
| Optional sid_e5 beta=12 k=180 | 0.684811, +0.002500, 4/5 folds | **NOT ADOPTED - see below** |

Why sid_e5 beta=12/k=180 was not adopted despite nominally passing the
predeclared rule (>= +0.002 and >= 4 folds): it was selected as the best of a
6-config grid scored on the same OOF metric, so the +0.0025 is optimistically
biased; the best ID-only config (k=90) failed the rule, which is the signature of
noise-driven selection. Adoption would also have required bundling 175 MB of E5
embeddings and redoing full verification. `reports/v6/sid_e5_adoption.json`
records `"adopted": true` for the k=180 variant, but the shipped Submission 3
uses **config A**. The JSON flag and the shipped artifact disagree; the shipped
artifact is authoritative. Flagged again in *Known Caveats*.


# Rejected and Superseded Work

**REJECTED** (honest OOF did not beat the locked baseline):
`v2_1` text ensemble (3 variants), `v2_2` hand features (5 variants),
`v2_4` meta-stacker (5 variants), `v2_5` combined, `v3_1a` multiscale ID k=30,
learned ID LR, learned ID LightGBM, near-duplicate retrieval (8 thresholds).

**MARGINAL** (gain inside the ~0.0038-0.005 fold-noise band):
all3 cleaning, per-class cross-fitted alpha, v3 text-ens per-class alpha
(+0.002136), v3 frozen e5-small 3-way (+0.016122, but superseded by refined),
v4 fast-path candidates A/B/C.

**USEFUL BUT SUPERSEDED**: the 0.639579 baseline (superseded by Sub 2 then Sub 3);
frozen e5-small and e5-base CPU sources; the v4 logit-offset candidate.

**NOT ADOPTED DUE TO OPTIMISM OR REPRODUCIBILITY RISK**:
`v2_3_perclass_lr_tuned_on_oof` (+0.001231 but tuned on the scored rows);
`sid_e5` beta=12/k=180 (+0.0025 selected as grid max); v5 buggy-SID blend (0.678057,
correctly stopped before writing a CSV).

**HISTORICAL / ARTIFACT LOST**: plain `xlmr_base` 0.544479; the ~0.678567
historical GPU blend figure; FastText (script only, never run).

# SID Alignment Bug and Correction

## The bug

In the v5 implementation the OOF call passed the **full** similarity matrix as
`Xq`, then the function sliced it by chunk position:

```
sid(ids[va], Xtr_n, tr, ids, y, **cfg)      # Xq = Xtr_n  <- WRONG
    ...
    xq = Xq[s:s + chunk]                    # positional slice of the whole matrix
```

So the similarity rows used for the query at local position `s` belonged to
training row `s`, **not** to `va[s]`. Query ids and similarity rows were
misaligned. Test was unaffected because the test call passed a genuinely
query-ordered matrix (`Xte_n`). The incorrect SID ID-only OOF was **0.612153**,
below even the plain ID prior 0.613631, which was the visible symptom.

## The correction

The v6 implementation follows the reference contract exactly:

```
q_ids = ids[va]
Xq    = Xn[va]        # query rows and IDs in identical order
```

* Corrected SID ID-only OOF config A: **0.627895** (predeclared band 0.6280 +/- 0.001)
* Per-fold: 0.622745, 0.637608, 0.627997, 0.625435, 0.62559 - matches the
  expected pattern [0.623, 0.637, 0.629, 0.626, 0.626]
* Sampled neighbour-fold leakage: **0** (300-query sample)
* Golden test: 5 fixed fold-0 rows (global indices 1, 4, 8, 9, 13; ids 32893,
  25368, 31917, 27178, 2429) compared against 5 fixed reference vectors

## Golden-test caveat (not hidden)

* Observed max abs difference: **1.47e-04**
* Requested tolerance: **1e-04** - **exceeded**
* Per-row max diff: 4.29e-05, 3.39e-05, 8.32e-05, 9.50e-05, 1.47e-04

Diagnosis, from `logs/v6_diag.log`:
* float32 and float64 give **bitwise identical** results, so this is not
  floating-point precision noise.
* The char vectoriser is `char_wb (2,4)` with `min_df=2`, `max_features=80000`;
  the uncapped vocabulary is **90,920** terms, so the cap binds and **21,308
  terms tie at document-frequency 2** at the cut. Which of the tied terms are
  retained depends on scikit-learn's internal tie-breaking, which differs
  between the export environment (sklearn 1.6.1) and the local environment
  (sklearn 1.9.0). That is the most likely cause of the ~1e-4 per-row residual.
* Alignment is confirmed correct independently by the full SID OOF landing in
  the predeclared band with the expected per-fold pattern, and by the corrected
  four-source blend hitting 0.682311 with all gates passing.

This caveat is disclosed in the Submission 3 notebook (the golden-test cell
prints a NOTE explaining the cause and states alignment is correct). If a strict
1e-4 golden bound must hold, the fix is to pin scikit-learn to 1.6.1 (the export
version) rather than to change the SID code. **Loose this bound deliberately or
pin the version; do not silently re-tune.**

# Current Best Submission

**Submission 3.** Best packaged and fully verified result.

All values below independently verified from `reports/v6/blend.json` and
`reports/v6/final.json`.

| Metric | Value |
|---|---|
| Expected honest grouped OOF | **0.682311** |
| Expected per-class F1 | 0.782202 / 0.640594 / 0.624137 |
| Expected accuracy | 0.710730 |
| Expected fold scores | 0.682243, 0.685245, 0.680023, 0.680668, 0.683172 |
| Expected fold std | 0.001859 |
| Final source order | text, corrected SID, E5 Base, XLM-R ID-head |
| Final median-normalised weights | 0.1053, 0.4211, 0.2105, 0.2632 |
| Plain-ID comparison | 0.677401 |
| SID gain | +0.004910 |
| Same-OOF optimistic | 0.684918 |

**The optimistic score 0.684918 is NOT the honest result.** It is the simplex
optimum selected and scored on the same OOF rows. The honest number is
**0.682311**. Never quote 0.684918 as performance.

Note the fold std 0.001859 landed **below** the anticipated 0.002-0.004 band,
which makes the +0.0049 margin over the plain-ID blend more comfortable than a
first reading suggests, but also means fold-level variance is lower than
expected.


# Reproducibility Audit

## Submission 3

| Aspect | Status |
|---|---|
| Notebook cells | **19** (12 code, 7 markdown) |
| Executed in place | No - 0 execution counts (shipped clean) |
| Mode A CPU default | Yes - `RUN_GPU_RETRAIN = False`, `mode: "A"` in bundled config |
| Mode B opt-in | Yes - full XLM-R ID-head + E5-base training code, documented as ~70 min on a T4, fp16 not bit-reproducible |
| `USE_DUP_OVERRIDE` flag | Present and `True` |
| Bundled artifact count | **9** + `manifest_submission_3.json` |
| Artifact hashes | config 52907a4d7e9b809c..., emb_e5-base_final bca3e9ec8fa9b288..., fold_ids_colab 5ab5cfe96a72b04c..., xlmr_base_idhead_final 8cc027296059e148..., fold0 c6c1953e89db8e6d..., fold1 d27f69100c41fc0a..., fold2 df3a279d6e7c5a8d..., fold3 6f13875e067d76ad..., fold4 8ca5c57c85d0eda8... |
| Internet dependency | **None in Mode A** - all probabilities bundled |
| GPU dependency | **None in Mode A**; optional in Mode B |
| Raw-data-to-output scope | Yes - rebuilds text and SID on bundled folds, recomputes honest OOF, blends, overrides, writes CSV |
| Probability-artifact replay scope | GPU sources supplied as verified probabilities, not retrained |
| Package versions | Export: sklearn 1.6.1, torch 2.11.0+cu128, transformers 5.16.1, seed 42. Local run: Python 3.14.3, sklearn 1.9.0, numpy 2.4.4, scipy 1.18.1 |
| Notebook SHA256 | `b7bcc940053c2af623530978e84947a56e1747d98758baf046dea5af505fc65c` |
| CSV SHA256 | `df637651e087e3d667c8552a37bef9c37b36e2ec9af41a4558b641495196edaa` |
| Clean nbconvert execution | PASS |
| Zero code-cell errors | YES |
| Fresh CSV byte-identical | **YES** |
| Protected files unchanged | YES |
| **PACKAGE READY** | **YES** |

## Submission 2

| Aspect | Status |
|---|---|
| Cells | 26 (15 code, 11 markdown), shipped **unexecuted** (0 execution counts) |
| Fresh execution | PASS - `_fresh_exec_test/exec_log.txt`, regenerated CSV byte-identical |
| Bundled artifacts | multilingual-e5-small checkpoint, 470.4 MB, 10 files, SHA256 sidecar |
| Internet dependency | None - checkpoint bundled |
| GPU dependency | **None - CPU only** |
| Scope | Raw-data-to-output; recomputes text, ID prior, E5-small embeddings |
| Notebook SHA256 | `f3b8f60883dfa0e659b86e07d4be7ffda2ce5b1048904891e21ad1bb1efe9597` |
| CSV SHA256 | `894ce6c210ccc0f2e956c1e6d32fcb82189f1798a6993ea4646555f93bfae40e` |
| Package versions | Python 3.14, sklearn 1.9.0 |

## Submission 1

| Aspect | Status |
|---|---|
| Cells | 12 (7 code, 5 markdown), **executed** - 7/7 execution counts, 0 errors |
| Fresh execution | Not evidenced in a dedicated log directory |
| Bundled artifacts | None beyond the CSVs |
| Internet / GPU dependency | None |
| Notebook SHA256 | `6366a78d5a4e08e05a496bd563a374f6d168e28c2b5310fd4cb57c6baf780d56` |
| CSV SHA256 | `1b173503d8fdd0e4a3f4dad4e38f743dbca9ae10e0ed460216399740fee40a50` |

## Cross-cutting

* Protected-file verification: `reports/v4_protected_manifest.json` (7 files) and
  `reports/v3_protected_manifest_recovery.json` (7 files + SHA1). All unchanged.
* `submission_3.csv` and `submission_reproducible_file_3.ipynb` have **no entry in
  any pre-existing manifest** - they were created after those manifests were
  frozen. Their hashes are recorded here and in `reports/v6/final.json` /
  `manifest_submission_3.json` context instead.
* Discipline throughout: candidate probability arrays were saved only as
  `.npy`/`.npz` under `predictions/v*/CANDIDATE_NOT_FOR_SUBMISSION_*` during
  frozen periods; the v4 report states explicitly "NO SUBMISSION CSV OR NOTEBOOK
  WAS CREATED. SUBMISSION FREEZE ACTIVE."

# Current Repository Artifacts

| Path | Contents |
|---|---|
| `train.csv`, `test.csv`, `sample_submission.csv` | raw competition data (read-only) |
| `uap_submission/submission/` | submission_2.csv, submission_3.csv |
| `uap_submission/reproducible/` | file_1/2/3 notebooks, models/, submission/submission_1.csv |
| `uap_submission/reproducible/artifacts_submission_3/` | 9 bundled artifacts + manifest + config |
| `uap_submission/reproducible/models/multilingual-e5-small/` | 10 files, 470.4 MB, SHA256 sidecar |
| `uap_complete_export_20260928_064140.zip` | 13 entries, Colab GPU export (read-only) |
| `uap_export.zip` | 4 entries, earlier partial export (read-only) |
| `final_sources/`, `per_fold_sources/`, `metadata/`, root `fold_ids.npy`, `xlmr_base_idhead_final.npz`, `emb_e5-base_final.npz` | extracted GPU export |
| `recovered/` | 4 recovered archives + RECOVERED_HASHES.json |
| `colab/` | `train_xlmr_colab.py`, `canonical_train_folds.csv` |
| `uap_gpu.ipynb` | original Colab GPU notebook (287,106 bytes) |
| `src/` (18 files) | v2/v3 harnesses, experiments, `make_repro_notebook.py`, `revalidate_grouped_cv.py` |
| `scripts/` (34 files incl. `nb3_cells/`) | v3/v4/v5/v6 pipeline, phase1 recovery, `v6_make_nb3.py` |
| `reports/` | v2_*, v3_*, v4/, v5/, v6/, trackers, manifests, summaries |
| `cache/` | t0/, v2/ (local folds + TF-IDF), v3/, v4/, v5/, v6/ |
| `predictions/` | oof/, test/, v2/, v3_candidates/, v4_candidates/ |
| `models/` | small_tfidf, text_id_blend, v3, v4, v5, v6 |
| `logs/` | 50 files; v2_*, v3_*, v4_*, v5_*, v6_* |
| `_preflight/`, `_fresh_exec_test/` | Submission 2 execution evidence |
| `data_checks/`, `configs/`, `final_local_package/`, `submissions/`, `dist/` | supporting dirs |
| `full_project_code.txt` (620 KB), `prompt.txt` | concatenated project source; FastText task prompt |

Submission 3 build scripts (for reference, do not re-run blindly):
`v6_step123.py` (artifacts + text/plain-ID + SID), `v6_step4_blend.py`,
`v6_step5_sid_e5.py`, `v6_step5_adopt.py`, `v6_step7_final.py`,
`v6_step8_bundle.py`, `v6_make_nb3.py` + `scripts/nb3_cells/`,
`v6_diag_golden.py`.


# Known Caveats

1. **Golden test 1.47e-04 vs requested 1e-04** - exceeded. Root cause is
   scikit-learn version tie-breaking in `max_features=80000` (21,308 tied terms
   at the cut; export used sklearn 1.6.1, local is 1.9.0). Not precision (f32 and
   f64 are bitwise identical). Alignment independently confirmed correct. Fix by
   pinning sklearn 1.6.1, not by changing SID code. Disclosed in the notebook.

2. **`sid_e5_adoption.json` says `"adopted": true` but Submission 3 ships config
   A.** The k=180 variant passed the numeric rule; it was rejected as a
   selection-bias risk. The JSON flag is stale relative to the shipped artifact.
   The shipped artifact is authoritative.

3. **"Contiguous non-overlapping train/test ids" is inaccurate.** Ids interleave
   (19,135 runs, mean length 3.12). This wording appears in
   `reports/text_id_blend_summary.md` and the Submission 3 notebook section 4 and
   should be corrected before any judge-facing submission.

4. **Fold-hash naming trap.** `925065f4...` is the int64-array-bytes hash of the
   local canonical folds, not a file hash. The file hash is `2bd8cfde...`.
   Several reports call the former "fold_sha256".

5. **Plain `xlmr_base` (0.544479) is unrecoverable.** Artifact listed in
   `preferred_files_missing`. Historical only.

6. **FastText was never run.** Claims in `prompt.txt` are unverified; the
   `fasttext` package is not installed and no artifacts exist.

7. **Submission 1 has no dedicated fresh-execution log** (unlike 2 and 3).

8. **Submissions 2 and 3 ship unexecuted** (0 execution counts). Both have
   external byte-identical fresh-execution evidence; neither has in-notebook
   outputs.

9. **Fold std 0.001859 for Submission 3 is below** the anticipated 0.002-0.004.

10. **Two different fold sets coexist** in `cache/`. Any new work must state
    which fold set it uses before comparing OOF numbers.

11. **Organizer clarification is user-supplied only** - no on-disk record.

# Judging Risk

Submission 3's weight vector is 0.4211 on SID, and the ID-head transformer was
trained with an ID head. Together the ID-derived sources carry the majority of
the blend weight. The organizer has stated ID usage is not viewed as a genuine
contribution and that methodology will be reviewed. A judge reading only the
leaderboard would see 0.70053; a judge reading the methodology sees a solution
whose strongest signal is dataset-specific scrape-order locality.

Mitigations already in place: the notebook discloses the ID competition-specific
nature explicitly in sections 4, 9 and 10, and labels the prior as inappropriate
for production. Disclosure must be preserved and extended in any write-up.

# Next Text-Centric Research Priority

**A. Preserve Submission 3 as the performance candidate.** Do not modify it. Its
CSV, notebook, bundle and hashes are the verified best.

**B. Build a text-centric candidate using only comment-derived sources.**

Candidate sources (all already partly validated on disk):
* word/character TF-IDF LR - 0.512943 (base), 0.518947 (all3-cleaned)
* all3-cleaned TF-IDF (NFKC + repeat collapse + emoji demojisation) - best
  text-only result, +0.006004
* frozen multilingual-e5-small - 0.496241
* frozen e5-base - 0.515829
* e5-base GPU zero-shot head - 0.521535
* plain XLM-R Base - 0.544479, **artifact lost, would need a GPU rebuild**
* BanglaBERT, MuRIL, or another justified multilingual encoder - untested
* XLM-R Base fine-tuned **without** any ID head - the most promising untried
  judge-friendly source, since `xlmr_base_idhead` at 0.648743 shows the
  architecture has headroom and plain `xlmr_base` at 0.544479 shows the ID head
  contributes roughly +0.10

Explicitly exclude from the main judge-friendly candidate: plain ID prior, SID,
XLM-R ID-head, row-order features.

**C. Build an ablation table** so the text-only contribution is legible to
judges: lexical text only / semantic text only / lexical + semantic / ID only /
text + ID / final performance candidate. Note the existing decomposition already
contains most cells: ID-only 0.613631, text-only 0.516093 (0.518947 all3),
semantic-only 0.521535, text+ID blend 0.677401 (plain), final 0.682311. The gap
to close is a competitive **semantic-dominant, zero-ID** blend.


# Protected Files

All verified unchanged against their manifests
(`reports/v4_protected_manifest.json`, `reports/v3_protected_manifest_recovery.json`).
Nothing was altered by this task.

| File | SHA256 | Status |
|---|---|---|
| `train.csv` | `a269ad841f824c6e60930cfda8cb25bdce58a6a1ee01fc1fe663d23b19e34449` | UNCHANGED |
| `test.csv` | `cca36f5416abfb8a41b1c84e5fa83aec18ffbff32fecd277c668a3f73312a38f` | UNCHANGED |
| `sample_submission.csv` | `0f7d405dbd1e88ea285c7138514b7652f7441e995dcaed338bb208a86c2f347d` | UNCHANGED |
| `submission_1.csv` | `1b173503d8fdd0e4a3f4dad4e38f743dbca9ae10e0ed460216399740fee40a50` | UNCHANGED |
| `submission_2.csv` | `894ce6c210ccc0f2e956c1e6d32fcb82189f1798a6993ea4646555f93bfae40e` | UNCHANGED |
| `submission_3.csv` | `df637651e087e3d667c8552a37bef9c37b36e2ec9af41a4558b641495196edaa` | no pre-existing manifest entry |
| `submission_reproducible_file_1.ipynb` | `6366a78d5a4e08e05a496bd563a374f6d168e28c2b5310fd4cb57c6baf780d56` | UNCHANGED |
| `submission_reproducible_file_2.ipynb` | `f3b8f60883dfa0e659b86e07d4be7ffda2ce5b1048904891e21ad1bb1efe9597` | UNCHANGED |
| `submission_reproducible_file_3.ipynb` | `b7bcc940053c2af623530978e84947a56e1747d98758baf046dea5af505fc65c` | no pre-existing manifest entry |

Export zips intact: `uap_complete_export_20260928_064140.zip` SHA256
`c4ea87bdc0a87ab10405c39f3dfe31886e08bbf19dc5349149a866c168aeb060`, 13 entries,
`testzip()` clean.

# Exact Next Action for Claude

1. Read this file, then `reports/v6/blend.json`, `reports/v6/final.json`,
   `reports/v6/step123.json` and the Submission 3 notebook. Treat Submission 3
   as frozen and authoritative.
2. State explicitly which fold set any new work uses. Do not mix fold set A
   (`925065f4...` int64) with fold set B (`cc4fdd16...` int64).
3. Build the text-only ablation (section *Next Text-Centric Research Priority*
   item C) reusing the already-validated component OOF arrays in `cache/v6/`
   before training anything new.
4. Fine-tune XLM-R Base **without** an ID head on fold set B and add it as a
   text-only source. This is the single highest-value untried step: it is the
   same architecture that reached 0.648743 with an ID head, and the plain
   `xlmr_base` score of 0.544479 bounds its value.
5. Keep every OOF evaluation grouped, leak-free, and honestly cross-fitted. Label
   same-OOF optima OPTIMISTIC. Never tune on public leaderboard feedback.
6. Correct the "contiguous non-overlapping ids" wording in any judge-facing text,
   and decide deliberately whether to pin scikit-learn 1.6.1 to tighten the
   golden-test bound.
7. Do not create Submission 4 until its honest grouped OOF and fold stability
   justify it, and keep full ID/SID disclosure regardless of score.

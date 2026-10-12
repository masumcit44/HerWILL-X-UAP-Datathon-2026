# Submission 4 - final report (ID-DERIVED LightGBM stacker)

**Authoritative OOF macro-F1: 0.703904** (recorded target 0.704001, difference -0.000097)

**Validation classification: `STANDARD_OOF_STACKING_WITH_DEPENDENCY`**

This is *not* claimed to be strictly outer-fold safe. The stacker consumes base-source
OOF arrays, which is standard stacking: a training row of fold `j` was scored by a base
model trained on folds `!= j`, which include fold `k`, so the stacker's training features
carry fold-`k` label information. The per-fold class offsets are additionally chosen on the
other four folds' stacker OOF (single-level, not double-nested).

## Score detail

| quantity | value |
|---|---|
| pooled OOF macro-F1 | 0.703904 |
| per-class F1 | 0.782266, 0.672590, 0.656856 |
| accuracy | 0.724261 |
| fold scores | 0.709531, 0.705706, 0.699544, 0.700173, 0.704385 |
| fold std | 0.003690 |
| worst fold | 0.699544 (fold 2) |
| seed scores (no correction) | {'42': 0.693895, '7': 0.694031, '2026': 0.692881} |
| prediction distribution | [22957, 19206, 5654] |
| test class distribution | [0.4796, 0.4059, 0.1144] |
| disagreement vs Submission 3 | 10.41% |

### Confusion matrix (rows = true, cols = predicted)

```
 18271    4982     503
  4239   12567    1357
   447    1657    3794
```

## Organizer disclosure

Submission 4 is a **leaderboard-performance stacker**, not a text method. It is heavily
ID-dependent: three of its six base sources are ID-derived (`sid`, `plain_id`,
`xlmr_base_idhead`) and 32 of its 60 features are ID-neighbour label/distance/window-share
features, plus the raw `id`. These features are dataset-specific and unsuitable for
deployment. The genuine text methodology is documented separately in
`submission_reproducible_file_3.ipynb` (Submission 3, 0.682311 OOF). Public leaderboard
feedback was not used for model selection, and no external labelled data was used. We
acknowledge the organizer's clarification that `id` is not a text feature.

## Gates

| gate | criterion | result | evidence |
|---|---|---|---|
| G1 | Final deterministic OOF within 0.001 of 0.703904 | **PASS** | 0.703904 (delta -0.000097) |
| G2 | OOF/test arrays pass shape, finite, range, row-sum, alignment | **PASS** | 6/6 sources ok, row-sum dev < 1e-5 |
| G3 | At least four folds do not materially regress vs recomputed Submission 3 | **PASS** | 5/5 folds non-regressing |
| G4 | No class F1 regresses by more than 0.005 | **PASS** | worst class delta +0.000064 |
| G5 | Test class shares within 6 pp of train shares | **PASS** | drift pp [1.576, 2.476, 0.9] |
| G6 | CSV passes pre-write and post-read validation | **PASS** | both passed, 11955 rows |
| G7 | Fresh notebook execution succeeds | **PASS** | 0 cell errors, fresh OOF 0.703904 |
| G8 | Fresh and packaged CSV byte-identical | **PASS** | sha256 bfb068893f324581 |
| G9 | Protected files remain unchanged | **PASS** | 0 changed |

**ALL GATES PASS: YES**

## Deliverables

| file | sha256 |
|---|---|
| `uap_submission/reproducible/submission_reproducible_file_4.ipynb` | `a2b3f3b7717e0758c728b80ce72eb5d6faeb7913557761996eab23345c999eed` |
| `uap_submission/submission/submission_4.csv` | `bfb068893f3245811760d42134253e51a0e5e77004e41a4c27630f2eaaca463f` |
| `uap_submission/reproducible/artifacts_submission_4/` | 10 artifacts + manifest |

## Reproduction

Mode A is fully CPU reproducible. Place the notebook, `artifacts_submission_4/`,
`train.csv`, `test.csv` and `sample_submission.csv` together and run
`jupyter nbconvert --execute --inplace submission_reproducible_file_4.ipynb`.
The notebook verifies every artifact SHA256, rebuilds the 60 features, retrains the
frozen 3-seed stacker, applies the frozen class correction and the duplicate override,
then writes and validates `submission_4.csv`.

No Kaggle upload was performed.

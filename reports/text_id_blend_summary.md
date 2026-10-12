# Text + ID-locality-prior blend -- local summary

**Model:** `blend_text_id_a0.40` -- word(1-2) + char_wb(3-5) TF-IDF (60k each, min_df=2,
sublinear, URL/USER masking only) with `LogisticRegression(solver='lbfgs', C=4.0,
class_weight='balanced', max_iter=2000)`, blended with a k=20 nearest-id label prior
(weight `1/(distance+1e-6)`) built from **training ids only**.

`p = alpha * p_text + (1 - alpha) * p_id`, **alpha = 0.40** (text weight) chosen on pooled
OOF only. Validation: StratifiedKFold(5, shuffle=True, random_state=42).
**Generated:** 2026-09-27 20:36:47. **GPU used:** False.

## 1. Per-fold Macro F1

| fold | n_val | macro_f1_text | macro_f1_id | macro_f1_blend_0.40 | accuracy_blend_0.40 | val_rows_with_text_also_in_train |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 9564 | 0.518704 | 0.618244 | 0.642747 | 0.688729 | 174 |
| 1 | 9564 | 0.515202 | 0.608781 | 0.634119 | 0.679005 | 170 |
| 2 | 9563 | 0.522421 | 0.619377 | 0.644688 | 0.687232 | 188 |
| 3 | 9563 | 0.524024 | 0.614708 | 0.643815 | 0.684409 | 185 |
| 4 | 9563 | 0.524123 | 0.614113 | 0.645667 | 0.689533 | 175 |

Weakest fold: text-only 0.515202, id-only
0.608781, blend@0.40 0.634119.

## 2. Pooled OOF comparison (47,817 rows)

| Model | Macro F1 | Accuracy | F1 c0 | F1 c1 | F1 c2 |
| --- | --- | --- | --- | --- | --- |
| text-only (lbfgs LR) | 0.520908 | 0.560324 | 0.645499 | 0.500657 | 0.416567 |
| id-prior only (k=20) | 0.615040 | 0.660916 | 0.734201 | 0.600826 | 0.510091 |
| blend @ alpha=0.40 | 0.642185 | 0.685781 | 0.759759 | 0.621358 | 0.545436 |
| blend @ alpha=0.40 (chosen) | 0.642185 | 0.685781 | 0.759759 | 0.621358 | 0.545436 |
| blend @ alpha=0.40 + text override (leak-free OOF) | 0.643056 | - | - | - | - |

Confusion matrix (OOF, blend @ chosen alpha):

| true/pred | pred_0 | pred_1 | pred_2 |
| --- | --- | --- | --- |
| true_0 | 18986 | 4339 | 431 |
| true_1 | 6159 | 11090 | 914 |
| true_2 | 1078 | 2104 | 2716 |

## 3. Alpha grid (pooled OOF; alpha = text weight)

| alpha_text_weight | alpha_id_weight | macro_f1 | accuracy | f1_class0 | f1_class1 | f1_class2 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 1.0 | 0.615040 | 0.660916 | 0.734201 | 0.600826 | 0.510091 |
| 0.05 | 0.95 | 0.618784 | 0.664826 | 0.737601 | 0.605154 | 0.513598 |
| 0.1 | 0.9 | 0.622679 | 0.668674 | 0.741237 | 0.608664 | 0.518135 |
| 0.15 | 0.85 | 0.627198 | 0.673568 | 0.746177 | 0.613268 | 0.522148 |
| 0.2 | 0.8 | 0.631799 | 0.677897 | 0.750354 | 0.617141 | 0.527903 |
| 0.25 | 0.75 | 0.634220 | 0.680448 | 0.753609 | 0.618703 | 0.530348 |
| 0.3 | 0.7 | 0.637013 | 0.682728 | 0.755924 | 0.620209 | 0.534908 |
| 0.35 | 0.65 | 0.640412 | 0.685070 | 0.758484 | 0.621738 | 0.541014 |
| 0.4 | 0.6 | 0.642185 | 0.685781 | 0.759759 | 0.621358 | 0.545436 |
| 0.45 | 0.55 | 0.640962 | 0.683502 | 0.758333 | 0.618024 | 0.546529 |
| 0.5 | 0.5 | 0.635707 | 0.678524 | 0.754882 | 0.612597 | 0.539643 |
| 0.55 | 0.45 | 0.624063 | 0.667629 | 0.746528 | 0.600667 | 0.524993 |
| 0.6 | 0.4 | 0.609488 | 0.654182 | 0.735784 | 0.587127 | 0.505553 |
| 0.65 | 0.35 | 0.593651 | 0.639124 | 0.722805 | 0.573051 | 0.485096 |
| 0.7 | 0.3 | 0.579770 | 0.624987 | 0.709812 | 0.559951 | 0.469547 |
| 0.75 | 0.25 | 0.567941 | 0.611916 | 0.696934 | 0.547881 | 0.459008 |
| 0.8 | 0.2 | 0.557097 | 0.600393 | 0.685936 | 0.537396 | 0.447959 |
| 0.85 | 0.15 | 0.546231 | 0.588598 | 0.674548 | 0.526034 | 0.438111 |
| 0.9 | 0.1 | 0.537054 | 0.578811 | 0.664785 | 0.517522 | 0.428853 |
| 0.95 | 0.05 | 0.529094 | 0.569567 | 0.655068 | 0.508940 | 0.423274 |
| 1.0 | 0.0 | 0.520908 | 0.560324 | 0.645499 | 0.500657 | 0.416567 |

## 4. Comparison with the externally verified targets

| Quantity | External reference | This run | Delta |
| --- | --- | --- | --- |
| text-only macro F1 | 0.52 | 0.520908 | +0.000908 |
| id-prior-only macro F1 | 0.61-0.62 | 0.615040 | +0.000040 |
| blend @ 0.40 macro F1 | 0.64 | 0.642185 | +0.002185 |
| best alpha found | 0.35-0.45 | 0.40 | - |
| best blend macro F1 | ~0.64 | 0.642185 | +0.002185 |

**Stop condition #1 (per-fold scores inside the verified ranges): PASSED**

## 5. Duplicate-text override

* Unanimous cleaned-text groups only; conflicting groups skipped:
  47,142 usable texts,
  53 conflicting groups skipped
  (132 rows).
* Test rows overridden: **272** of 11,955
  (2.275%).
* Leak-free OOF effect (per-fold lookups built from that fold's training split only):
  858 OOF rows overridden, macro F1
  0.642185 -> 0.643056
  (+0.000871).

## 6. Bias sanity check (threshold 10pp)

| class | train share % | predicted share % | drift (pp) |
| --- | --- | --- | --- |
| 0 | 49.68 | 54.82 | +5.14 |
| 1 | 37.98 | 36.48 | -1.51 |
| 2 | 12.33 | 8.7 | -3.64 |

No class drifts by more than the threshold.

## 7. Invariants

| check | status | detail |
| --- | --- | --- |
| oof_columns_exact | PASS | got=['id', 'y_true', 'prob_0', 'prob_1', 'prob_2', 'y_pred', 'fold', 'model_name'] |
| oof_rows_equal_train_rows | PASS | oof=47817 train=47817 |
| oof_ids_unique | PASS | nunique=47817 |
| oof_ids_match_train | PASS | symmetric set equality |
| oof_labels_subset_of_012 | PASS | unique=[np.int64(0), np.int64(1), np.int64(2)] |
| oof_y_true_matches_train | PASS | mismatch=0 |
| oof_probs_finite | PASS | non_finite=0 |
| oof_probs_in_0_1 | PASS | min=9.36979e-07 max=0.999936 |
| oof_probs_rows_sum_to_1 | PASS | max_abs_dev=3.33e-16 tol=1e-06 |
| oof_pred_in_012 | PASS | unique=[np.int64(0), np.int64(1), np.int64(2)] |
| oof_pred_equals_argmax | PASS | mismatch=0 |
| oof_no_missing | PASS | n_missing=0 |
| fold_sizes_sum_to_train_rows | PASS | sum(n_val)=47817 fold sizes=[9564, 9564, 9563, 9563, 9563] |
| fold_ids_cover_all_rows | PASS | folds=[np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)] |
| train_val_id_overlap_zero_in_every_fold | PASS | fold0=0; fold1=0; fold2=0; fold3=0; fold4=0 |
| id_prior_labels_train_only | PASS | prior built exclusively from each fold's training ids/labels |
| oof_pred_class_distribution_logged | PASS | {0: 26223, 1: 17533, 2: 4061} |
| test_prob_columns_exact | PASS | got=['id', 'prob_0', 'prob_1', 'prob_2', 'y_pred', 'model_name'] |
| test_prob_rows_equal_test_rows | PASS | probs=11955 test=11955 |
| test_prob_id_order_equals_test_csv | PASS | row order comparison |
| test_prob_probs_finite | PASS | non_finite=0 |
| test_prob_probs_in_0_1 | PASS | min=3.4045e-07 max=0.999815 |
| test_prob_probs_rows_sum_to_1 | PASS | max_abs_dev=2.22e-16 tol=1e-06 |
| test_prob_pred_in_012 | PASS | unique=[np.int64(0), np.int64(1), np.int64(2)] |
| test_prob_pred_equals_argmax | PASS | mismatch=0 |
| test_prob_no_missing | PASS | n_missing=0 |
| submission_columns_exact | PASS | got=['id', 'y_pred'] |
| submission_rows_equal_sample | PASS | sub=11955 sample=11955 |
| submission_id_order_equals_sample | PASS | row order comparison |
| submission_id_set_equals_sample | PASS | symmetric set equality |
| submission_ids_unique | PASS | nunique=11955 |
| submission_no_nulls | PASS | n_missing=0 |
| submission_values_in_012 | PASS | unique=[np.int64(0), np.int64(1), np.int64(2)] |
| submission_values_integer | PASS | dtype=int64 |
| submission_pred_class_distribution_logged | PASS | {0: 6554, 1: 4361, 2: 1040} |
| submission_re_read_from_disk_is_valid | PASS | rows=11955 cols=['id', 'y_pred'] |

All invariants passed: **True**
(submission validated BEFORE it was written; see scope notes.)

## 8. Outputs

* Submission: `submissions\submission_text_id_blend.csv` (11,955 rows, written: True)
* OOF blend: `predictions\oof\oof_text_id_blend.csv` | OOF text: `predictions\oof\oof_text_only_lbfgs.csv` | OOF id-prior: `predictions\oof\oof_id_prior_only.csv`
* Test probs: `predictions\test\test_text_id_blend.csv` (+ `predictions\test\test_text_only_lbfgs.csv`, `predictions\test\test_id_prior_only.csv`)
* Fold scores: `reports\text_id_blend_fold_scores.csv` | Metrics: `reports\text_id_blend_metrics.json`
* Fold assignments: `data_checks\text_id_blend_fold_assignments.csv` | Models: `models\text_id_blend`
* Run log: `logs/experiment_text_id_blend.log`
* Runtime: 51.81s (folds 5.55s, final fit 43.97s)

## 9. Scope notes

- The id-locality signal is dataset-specific (contiguous non-overlapping train/test ids from one scrape order); do not reuse it as a generic feature elsewhere.
- The prior never sees validation/test labels: the neighbour pool is each fold's training split only and the function signature excludes any query labels.
- Plain StratifiedKFold is used as specified, so duplicate texts may straddle folds; val_rows_with_text_also_in_train per fold quantifies that optimism.
- test_text_id_blend.csv keeps y_pred = argmax(blended probs) (pure model); the duplicate-text override is applied only in the submission file.
- Submission validated before writing and re-read from disk afterwards; alpha was selected on pooled OOF only, never on leaderboard information.

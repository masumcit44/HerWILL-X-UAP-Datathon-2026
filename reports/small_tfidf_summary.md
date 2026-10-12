# Small-TFIDF experiment (Logistic Regression) -- local summary

**Model:** `lr_small_tfidf` -- word(1-2) + char_wb(3-5) TF-IDF hstacked, one
LogisticRegression per fold (C=4.0, liblinear, class_weight=balanced, max_iter=2000),
wrapped in `OneVsRestClassifier` because sklearn 1.9 removed native multiclass support
from `solver='liblinear'`; the requested OvR scheme, solver, C, class_weight and
max_iter are unchanged and probabilities still sum to 1.
**Validation:** StratifiedGroupKFold(n_splits=3, shuffle=True,
random_state=42) grouped by normalized text -> duplicate-safe folds.
**Primary metric:** Macro F1. **Generated:** 2026-09-27 19:59:05. **GPU used:** False.

## 1. Feature matrices

| Block | Shape | nnz | Config |
| --- | --- | --- | --- |
| word TF-IDF (train) | 60000 features, n=47817 | 5,712,066 (word+char) | 1-2 grams, min_df=2, max_df=0.995, sublinear, max_features=60000 |
| char_wb TF-IDF (train) | 60000 features | see above | 3-5 grams, min_df=2, sublinear, max_features=60000 |
| combined train | [47817, 120000] | 5,712,066 | scipy.sparse.hstack, float32 |
| combined test | [11955, 120000] | 1,411,745 | scipy.sparse.hstack, float32 |

Vocabulary fitted unsupervised on 59772 concatenated train+test documents (no labels, no external data).

## 2. Per-fold Macro F1

| fold | n_val | macro_f1 | accuracy | f1_class0 | f1_class1 | f1_class2 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 15939 | 0.523044 | 0.562896 | 0.643900 | 0.508378 | 0.416853 |
| 1 | 15939 | 0.521321 | 0.560198 | 0.643712 | 0.501635 | 0.418616 |
| 2 | 15939 | 0.517509 | 0.559634 | 0.644747 | 0.498527 | 0.409253 |

* fold mean = **0.520625**, fold std (ddof=0) = **0.002313**
* weakest fold = 0.517509

## 3. Overall OOF metrics

* **Overall OOF Macro F1 = 0.520652**
* OOF accuracy = 0.560909

| class | label | f1 | support |
| --- | --- | --- | --- |
| class0 | explicitly_toxic | 0.644122 | 23756 |
| class1 | subtly_toxic | 0.502857 | 18163 |
| class2 | non_toxic | 0.414976 | 5898 |

## 4. Confusion matrix (OOF, rows = true, cols = predicted)

| true\\pred | pred_0_explicitly_toxic | pred_1_subtly_toxic | pred_2_non_toxic |
| --- | --- | --- | --- |
| true_0_explicitly_toxic | 15076 | 7152 | 1528 |
| true_1_subtly_toxic | 6781 | 9240 | 2142 |
| true_2_non_toxic | 1198 | 2195 | 2505 |

## 5. Prediction distribution

| class | train_truth | oof_pred | test_pred |
| --- | --- | --- | --- |
| 0 | 23756 | 23055 | 5955 |
| 1 | 18163 | 18587 | 4589 |
| 2 | 5898 | 6175 | 1411 |

OOF mean top-class probability = 0.6145;
test mean top-class probability = 0.5922.

## 6. Benchmark comparison

| Benchmark | Macro F1 | Delta (this - benchmark) |
| --- | --- | --- |
| Robust grouped-OOF best (LR + ComplementNB) | 0.525448 | -0.004796 |
| Research best (cross-fitted calibration) | 0.525574 | -0.004922 |

**Verdict: USEFUL DIVERSITY ONLY** -- OOF Macro F1 is 0.004796 below the robust benchmark; the model remains interesting only if its errors are sufficiently different (blend test not performed in this run).

## 7. Integrity checks

| check | status | detail |
| --- | --- | --- |
| oof_columns_exact | PASS | got=['id', 'y_true', 'prob_0', 'prob_1', 'prob_2', 'y_pred', 'fold', 'model_name'] |
| test_probs_columns_exact | PASS | got=['id', 'prob_0', 'prob_1', 'prob_2', 'y_pred', 'model_name'] |
| oof_rows_equals_train | PASS | oof=47817 train=47817 expected=47817 |
| test_prob_rows_equals_test | PASS | test_probs=11955 test=11955 expected=11955 |
| oof_ids_unique | PASS | nunique=47817 |
| oof_ids_match_train | PASS | symmetric set equality |
| test_ids_preserve_test_csv_order | PASS | row order compared against test.csv |
| test_ids_also_preserve_sample_order | PASS | row order compared against sample_submission.csv |
| oof_no_missing | PASS | n_missing=0 |
| test_probs_no_missing | PASS | n_missing=0 |
| oof_probs_finite | PASS | non_finite=0 |
| oof_probs_in_0_1 | PASS | min=6.36868e-05 max=0.997612 |
| oof_probs_sum_to_1 | PASS | max_abs_deviation=2.22e-16 |
| oof_labels_only_012 | PASS | unique=[np.int64(0), np.int64(1), np.int64(2)] |
| oof_labels_are_argmax | PASS | mismatch=0 |
| test_probs_probs_finite | PASS | non_finite=0 |
| test_probs_probs_in_0_1 | PASS | min=1.86933e-05 max=0.997069 |
| test_probs_probs_sum_to_1 | PASS | max_abs_deviation=2.22e-16 |
| test_probs_labels_only_012 | PASS | unique=[np.int64(0), np.int64(1), np.int64(2)] |
| test_probs_labels_are_argmax | PASS | mismatch=0 |
| oof_y_true_matches_train_labels | PASS | mismatch=0 |
| duplicate_groups_never_cross_folds | PASS | groups_spanning_multiple_folds=0 (n_groups=47147, n_dup_groups=486) |
| fold_ids_cover_all_rows | PASS | folds=[np.int64(0), np.int64(1), np.int64(2)] |
| protected_inputs_unmodified | PASS | train.csv/test.csv/sample_submission.csv opened read-only |

All checks passed: **True**
(row/order checks compare directly against train.csv/test.csv/sample_submission.csv.)

## 8. Outputs

* OOF probabilities: `predictions\oof\oof_small_tfidf_lr.csv`
* Test probabilities: `predictions\test\test_small_tfidf_lr.csv`
* Fold scores: `reports\small_tfidf_fold_scores.csv`
* Fold assignments: `data_checks\fold_assignments_3fold.csv`
* Models: `models\small_tfidf` (word_vectorizer.joblib, char_vectorizer.joblib, lr_fold_0..2.joblib)
* Run log: `logs/experiment_small_tfidf.log`
* Runtime: 43.64s total
  (features 9.93s, folds 20.73s)

## 9. Scope notes

- No submission file was created (out of scope for this experiment).
- No ensemble/blend weights were fitted; the benchmark comparison is indicative.
- 3-fold OOF is not strictly comparable to benchmarks computed with a different fold count; fold count changes the amount of training data per fold.
- TF-IDF vocabularies were fitted unsupervised on concatenated train+test text; no labels, no external data, no pretrained weights were used.
- Duplicate-safe folds were verified: zero normalized-text groups span >1 fold.
- Compatibility deviation (documented): the requested solver='liblinear' is kept, but wrapped in OneVsRestClassifier because scikit-learn 1.9 no longer allows native multiclass liblinear models.

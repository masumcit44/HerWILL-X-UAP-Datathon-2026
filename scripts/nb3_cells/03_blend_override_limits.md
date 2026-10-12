# Notebook cell 03 - markdown: blend, override, limitations, compliance
## 7. Honest blend versus final weights

Four sources are blended: `[text, sid, e5_base, xlmr_idhead]`. Candidate weight
vectors are all points on the 4-simplex with step 0.05 (1771 vectors,
lexicographic order). Blending is a weighted sum of probability vectors,
renormalised per row.

**Honest (cross-fitted) evaluation.** For each outer fold the weights that
maximise macro-F1 on the *other four* folds are applied to the held-out fold.
Because weights for a fold are never chosen using that fold, the pooled result
is unbiased:

* honest OOF macro-F1 **0.682311**
* per-class F1 `[0.782202, 0.640594, 0.624137]`
* fold scores `[0.682243, 0.685245, 0.680023, 0.680668, 0.683172]`, std 0.001859
* the same-OOF optimum is 0.684918, labelled **OPTIMISTIC** because it is
  selected and scored on the same rows - it is *not* a performance claim

For reference, swapping `sid` for the plain ID prior gives 0.677401, so the
similarity weighting is worth **+0.0049** in the blend.

**Final weights.** The element-wise median of the five cross-fitted fold weight
vectors, renormalised to sum to 1, gives
`[0.1053, 0.4211, 0.2105, 0.2632]` in the source order above. These are used
for the test predictions. Taking the median of cross-fitted weights (rather
than one globally tuned set) keeps the test-time weights honest.

## 8. Duplicate-text override

After `argmax`, any test row whose cleaned text exactly matches a **training**
text whose label is unanimous across all its occurrences takes that label.
Groups with conflicting training labels (53 of them) are skipped entirely, and
the lookup is built from training rows only. This mirrors Submissions 1 and 2:
272 test rows matched and 42 labels changed. `USE_DUP_OVERRIDE` is kept so the
behaviour is explicit and can be switched off.

## 9. Limitations

* The ID priors, and therefore a large share of the blend weight (`sid` at
  0.42), exploit dataset-specific id locality and would not transfer to live
  data (section 4).
* The honest-OOF margin over the plain-ID blend is +0.0049. Real but modest,
  and part of the apparent headroom is weight-selection noise: the same-OOF
  optimum is only ~0.0026 above the honest figure.
* The blended OOF estimate is 5 folds of one 47,817-row dataset; fold std 0.0019
  does not capture seed or fold-assignment variance.
* Character TF-IDF vocabulary truncation at `max_features=80000` is tie-break
  sensitive across scikit-learn versions, moving individual probabilities at
  the 1e-4 level. Pooled macro-F1 is stable to ~1e-5.
* `e5_base` at 0.5215 is barely above the text model; its blend contribution is
  small and it could be dropped with little loss.
* Macro-F1 is threshold-free here, so no decision threshold was tuned.

## 10. Rule compliance

* **No external data.** Only `train.csv`, `test.csv` and
  `sample_submission.csv` from the competition. Nothing was scraped, downloaded
  or joined in.
* **Public pretrained weights only.** `e5-base-v2` and `xlm-roberta-base` are
  public checkpoints; in Mode A they are used only through the supplied
  probability arrays.
* **No leaderboard feedback.** Every number reported here comes from OOF
  predictions on the bundled folds.
* **Competition files untouched.** `train.csv`, `test.csv`,
  `sample_submission.csv` and Submissions 1 and 2 are read-only inputs.
* **CPU-only by default.** Mode A runs on CPU. Mode B is opt-in and needs a GPU
  only to retrain the two transformer sources.
* The CSV is produced by this notebook itself, top to bottom.

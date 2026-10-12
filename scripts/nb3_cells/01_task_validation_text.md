# Notebook cell 01 - markdown: task, metric, validation, text source
# Submission 3 - UAP author-type classification

## 1. Task and metric

Given a tweet/post `text` and a numeric `id`, predict the author type `y` in
`{0, 1, 2}`. The metric is **macro-averaged F1** over the three classes, so
every class carries equal weight: a model that is excellent on the majority
class and poor on the rare class scores badly. All model selection here is
driven by macro-F1 on out-of-fold (OOF) predictions only. No public leaderboard
feedback was used at any point.

Train class balance: `0.4968 / 0.3798 / 0.1233`. Class 2 is the rare one, and
it is also the class that gains most from the ID-locality signal (section 4).

## 2. Grouped validation and duplicate-leakage prevention

The folds are **not** regenerated. They are loaded from
`artifacts_submission_3/fold_ids_colab.npy`, produced by the original Colab run
and verified byte-for-byte against a SHA256 manifest. This matters: the GPU
probability arrays bundled beside it were produced with exactly these folds, so
mixing them with any other fold set would silently invalidate every OOF number
in this notebook.

The folds are grouped by **cleaned text**: every row sharing
`clean_text(text).lower().strip()` sits in the same fold. The verification cell
below asserts that **zero** groups span more than one fold. That prevents
scoring a model on a near-copy of a row it trained on, the dominant leakage
risk in a dataset full of retweets and boilerplate.

Every OOF estimate uses those five folds, and blend weights are chosen by
nested cross-fitting (section 7) so no weight is ever selected on the fold it
is scored on.

## 3. Text source (CPU)

Two TF-IDF views are fitted **on fold-training rows only** and stacked:

* word `(1,2)`-grams, `min_df=2`, `max_features=60000`, `sublinear_tf=True`
* `char_wb` `(3,5)`-grams, `min_df=2`, `max_features=60000`, `sublinear_tf=True`

A `LogisticRegression(solver="lbfgs", C=4.0, class_weight="balanced",
max_iter=2000, random_state=42)` is fitted on the hstacked matrix. Text is
cleaned first: URLs and `@mentions` become the tokens `URL` / `USER`,
whitespace is collapsed, ends stripped. `class_weight="balanced"` matters for
macro-F1 because it counteracts the 4:1 imbalance.

Standalone OOF macro-F1: **0.516093**.

# Notebook cell 02 - markdown: plain ID prior, sid, GPU sources
## 4. Plain ID prior - and why it is competition-specific, not a production feature

Rows are sorted by `id`. For a query row, the `k=20` nearest ids on each side
supply neighbour labels weighted by inverse distance `w = 1/(d + 1e-6)`; the
weighted vote is normalised into a distribution. The neighbour pool is always
the fold-training rows only, so the prior never sees a validation or test label.

This works because train and test ids are **contiguous, non-overlapping slices
of one scrape order**, so id proximity proxies topical and temporal adjacency.
Standalone OOF macro-F1: **0.613631**.

**This is not a general modelling feature.** It exploits an artefact of how this
dataset was collected. On live deployment with freshly arrived rows no such id
locality exists, and a model leaning on it would degrade sharply. It is kept
here because it is a legitimate, reproducible signal in *this* competition and
is reported as a dataset-specific component rather than dressed up as portable.
Nothing else in the pipeline depends on it.

## 5. Similarity-weighted ID prior (`sid`)

`sid` keeps the ID-locality idea but re-weights each neighbour by how
textually similar it is to the query, so a near neighbour that is topically
unrelated votes less:

    w_j      =  1 / (d_j + 1e-6) ** gamma   *   exp(beta * sim(q, j))
    p(y | q) =  sum_j  w_j * 1[y_j = y]  /  sum_j  w_j

* `d_j` - absolute id difference to the neighbour, after sorting the pool by id
* `sim(q, j)` - cosine similarity of **L2-normalised character TF-IDF**
  `(2,4)`-grams (`char_wb`, `min_df=2`, `max_features=80000`,
  `sublinear_tf=True`) fitted on **all training text with no labels**
* the `k` nearest ids inside a `+/- KW` positional window are candidates
* the neighbour pool excludes the query's own fold, so no validation row can
  ever vote on itself

Config A (**predeclared**, not tuned by score): `KW=120, k=180, beta=10,
gamma=1.0`. Sensitivity variants (B: `k=90, beta=8, gamma=0.75`; C: `k=120`)
are reported for information only and were not used for selection.

**Similarity uses text only - never labels.** The vectoriser is unsupervised
and label-free, so `sim` cannot encode the target.

Standalone ID-only OOF macro-F1: **0.627895** (A), 0.626534 (B), 0.627606 (C).

## 6. GPU-trained sources

Two transformer sources were trained on a Colab T4 and are supplied here as
**verified probability arrays** (OOF + test), each checked against a SHA256
manifest and re-verified in the notebook:

* **`e5_base`** - `intfloat/e5-base-v2` embeddings with a zero-shot label head.
  Standalone OOF macro-F1 0.521535.
* **`xlmr_idhead`** - `xlm-roberta-base` encoder with a classification head.
  Standalone OOF macro-F1 0.648743, the strongest single source.

Both were also re-exported per fold; the notebook stitches the five `vp` blocks
over their `va` indices, checks this reproduces the combined `oof`, and checks
the mean of the five `tp` blocks reproduces `test`.

**The only GPU dependency in this submission is these two probability arrays.**
Mode A (the default) needs no GPU at all. Mode B, disabled by default, contains
the full training code so an organiser with a T4 can retrain them; fp16 GPU
training is *not* bit-reproducible, so Mode B output may differ slightly from
the bundled arrays.

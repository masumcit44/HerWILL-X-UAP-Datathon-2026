# Notebook cell 10 - markdown: sid in code
## 12. The similarity-weighted ID prior (`sid`) in code

`q_ids` are the ids of the query rows and `Xq` holds the **L2-normalised TF-IDF
rows of those same queries in the same order** - the row order of `Xq` must
match the order of `q_ids` exactly, which is what the `Xn[va]` indexing below
guarantees. `pool_rows` are global train-row indices allowed to vote.
Similarity comes from an unsupervised vectoriser fitted on training text only,
so labels never enter it.

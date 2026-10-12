# Notebook cell 13 - markdown: honest blend
## 13. Honest cross-fitted blend

Weights live on the 4-simplex with step 0.05. For each outer fold the best
weights are chosen on the other four folds and applied to the held-out fold, so
the pooled score never sees its own weight selection.

## 11. Reproducing the submission

Run all cells top to bottom. Two modes:

* **Mode A (default, CPU).** Verifies the bundled artifacts against the SHA256
  manifest, rebuilds the text and `sid` sources on the bundled Colab folds,
  recomputes the honest OOF and checks it against the recorded value, blends
  with the final weights, applies the override and writes
  `uap_submission/submission/submission_3.csv`. No GPU required.
* **Mode B (`RUN_GPU_RETRAIN = True`).** Additionally retrains the XLM-R
  ID-head and E5-base sources on a T4 (~70 min). fp16 is not bit-reproducible,
  so results may differ slightly from the bundled arrays.

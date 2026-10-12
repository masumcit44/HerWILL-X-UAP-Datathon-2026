# Submission 5: reproduce in about 4 minutes

**Leaderboard-performance model. Heavily ID-derived. Not proposed for deployment.**
Validation class: STANDARD_OOF_STACKING_WITH_DEPENDENCY. No leaderboard feedback and no external labelled data were used.

| | value |
|---|---|
| Honest grouped OOF macro-F1 (fold set B) | **0.706268** (Submission 4 in the same environment: 0.704464, delta +0.0018) |
| Per-class F1 | 0.786217 / 0.672289 / 0.660299 |
| Fold scores | 0.704399, 0.709964, 0.705406, 0.707403, 0.703769 (std 0.0023, worst 0.7038) |
| Seeds (uncorrected) | 42: 0.696349 · 7: 0.697013 · 2026: 0.695975 |
| Class correction | per fold on the other folds; test uses the median [0, −0.10, −0.70] |
| Test class share | 0.4996 / 0.3901 / 0.1102 (train 0.4968 / 0.3798 / 0.1233) |
| Duplicate override | 272 matched, 39 changed |
| CSV SHA256 | `278ced47f4bf46d9490624bed32a128cc3a85184427280e3bb50ca022e347aab` |

**Steps**

1. Put `train.csv`, `test.csv` and `sample_submission.csv` in this folder (the project root).
2. Install the libraries: `pip install numpy pandas scikit-learn==1.8.0 lightgbm==4.7.0`
3. Go to the notebook folder: `cd uap_submission/reproducible`
4. Run it with `python run_submission_5.py`. Alternatively, run all cells of `submission_reproducible_file_5.ipynb`; it contains the same code.
5. The run verifies every SHA256 and asserts the OOF within 0.001. It then writes `uap_submission/submission/submission_5.csv` and prints its SHA256.

**Bundled artifacts** (`artifacts_submission_5/`, all SHA256-checked):
- fold set B
- six fold-set-B base probability arrays: TF-IDF, text-kNN, E5-base, SID, plain ID prior, XLM-R ID-head
- the config

All six arrays are identical to the inputs Submission 4 builds or bundles. The TF-IDF, kNN, SID and ID arrays are the outputs of `build_cpu_text_sources.py` and `build_sid_source.py`; E5 and the ID-head are the Colab arrays.

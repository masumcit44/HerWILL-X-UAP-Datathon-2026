Submission 4 candidate - LightGBM stacker (performance track, ID-DERIVED, must be disclosed)

Honest grouped OOF macro-F1 (fold set B): 0.704001  (Submission 3 recomputed in the same environment: 0.683258)
Per-class F1 0.7818 / 0.6725 / 0.6577 | folds 0.7066 0.7079 0.7033 0.6995 0.7025 | std 0.0030

Pipeline
 1. build_cpu_text_sources.py -> text_tfidf, text_knn (text only)
 2. build_sid_source.py      -> SID config A (ID-derived)
 3. existing Colab arrays    -> emb_e5-base_final.npz, xlmr_base_idhead_final.npz
 4. feats.py  -> 60 features: 18 source probabilities, ID-neighbour labels/distances/class shares
                 (neighbour pool = train rows of OTHER folds only), text meta (length, script, emoji, caps...), raw id
 5. stack.py all 400 {42,7,2026} -> cross-fitted LightGBM (fold k model trained on folds != k),
                 class-balanced weights, 400 rounds, test features rebuilt per fold with the same pool density
 6. average of 3 seeds; class-2 logit offset -0.55 = median of offsets each chosen on the other four folds
 7. duplicate override identical to Submission 3 (unanimous cleaned-text train groups): 272 matched, 38 changed

Caveats: stacking uses base-source OOF arrays (standard stacking; base models of one fold saw other folds' labels);
offsets are tuned on other folds' stacker OOF (single-level, not double-nested); scripts use absolute paths from the
build environment - edit W and d at the top of feats.py before rerunning.

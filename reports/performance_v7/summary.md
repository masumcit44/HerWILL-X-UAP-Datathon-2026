# Performance sprint - fixed E5-SID beta12/k180

Track A only. Fold set B, honest cross-fitted grouped OOF macro-F1.

## Result: REJECTED at gates G4 and G5

- Submission 3 reference reproduced: **0.682311** (diff 1.78e-07)
- Fixed E5-SID beta12/k180 standalone OOF: **0.628273** (exact match to recorded)
- Challenger honest OOF: **0.684811** (delta +0.002500) - reproduced exactly
- Per-class F1: [0.783668, 0.642736, 0.628029] | accuracy 0.713554
- Fold scores: [0.687264, 0.690604, 0.683387, **0.675750**, 0.686805]
- fold std **0.005053** vs Submission 3 **0.001859** (2.7x worse)
- worst fold **0.675750** vs Submission 3 **0.680023**
- Same-OOF OPTIMISTIC (not a result): 0.685434

## Gates

| gate | measured | result |
|---|---|---|
| G1 delta >= +0.0015 | +0.002500 | PASS |
| G2 >= 4 folds not worse | 4/5 | PASS |
| G3 no class drop > 0.005 | +0.001466 / +0.002142 / +0.003892 | PASS |
| G4 std <= 1.5x reference | 0.005053 vs 0.002789 | **FAIL** |
| G5 worst fold not below ref by > 0.001 | 0.675750 vs 0.679023 | **FAIL** |
| G6 test integrity | shape/finite/range/rowsum all pass | PASS |
| G7 test share within 6pp | drift 3.90 / 2.93 / 0.98 pp | PASS |

## Interpretation

The challenger is genuinely better on pooled macro-F1 by +0.0025 and improves
all three classes, but the gain is unevenly distributed: folds 0, 1, 2 and 4 gain
between +0.0013 and +0.0054, while fold 3 loses 0.0049. Fold variance nearly
triples (0.001859 -> 0.005053).

The predeclared gates exist precisely to block a pooled gain that is bought with
fold instability, so the candidate is **rejected** and **no P1/P2 probability
files were created**.

Note also that per-fold weights are near-identical (0.10/0.45/0.25/0.20 on four
folds, 0.05/0.50/0.25/0.20 on fold 3), so this is not weight-search noise - the
E5-SID source itself generalises less uniformly across folds than the TF-IDF-SID
it replaces.

## Provenance

- The configuration was fixed in advance: beta=12, k=180, KW=120, gamma=1.
- The six-configuration E5-SID grid was **not** rerun and **no other
  configuration was selected**.
- The stored OOF array `cache/v6/sidE5_beta12.0_k180_oof.npy` was audited
  (integrity, shape, fold alignment) and reproduced its recorded standalone score
  exactly. Only the **missing test array** was reconstructed, from cached E5
  Base embeddings, via an atomic temp-file-then-rename write.
- Subset safety: every OOF fold drew its neighbour pool from the other four
  folds only; the test array drew from all train rows.
- `submission_generated: false`. No submission CSV or notebook was created.
- Protected files unchanged: verified before and after.

# Text-only fast sprint

Text-derived sources ONLY. No ID, SID, row order, or XLM-R ID-head.

## Headline

- Honest cross-fitted OOF macro F1: **0.559094** (WEAK)
- Without class-logit offsets: 0.552649
- Per-class F1: [0.673872, 0.519367, 0.484043]
- Accuracy: 0.590313
- Fold scores: [0.554776, 0.555329, 0.565105, 0.562936, 0.557502] (std 0.004153)
- OPTIMISTIC_DIAGNOSTIC_ONLY: 0.553788 (same-OOF, not a performance claim)
- Best single text source: text 0.524057 (delta +0.035037)
- Final status: **WEAK**

## Why this is low, stated plainly

The plain ID prior alone scores 0.614135 and SID alone 0.627895, both of
which are excluded here by the task rules. Every text-only source on disk
sits in a narrow 0.496-0.524 band and their blend reaches only ~0.56. The
gap to Submission 3 (0.682311) is almost entirely the ID-family signal.
This candidate is methodologically honest for a text-only judging
requirement, but it is NOT competitive on score.

## Blend

- Sources: e5_base, e5_small, text, text_all3
- Test-capable 3-source variant: e5_base, e5_small, text (honest 0.553638), used for the test artifact
- Simplex step 0.10, 286 candidates, first max in lexicographic order
- Median final weights: {'e5_base': 0.4, 'e5_small': 0.2, 'text': 0.3, 'text_all3': 0.1}
- Offsets adopted: [0.0, -0.1]

## Fold set

- Set A (local canonical), sizes [9564, 9564, 9563, 9563, 9563]
- file sha256 `2bd8cfdedb32342931d17cc27ca0a1be99d8a567bb8128e32955ec18cd91880d`
- int64-bytes sha256 `925065f4d4a0c6a533fad774f4875b039ea526eb915bee8c8c8b819c88760f56`
- Fold set B (Colab) sources were NOT mixed in.

## Provenance

- submission_generated: false
- stopped_due_to_time_limit: false

"""Render reports/submission_4_final_report.md from the audit JSON."""
import json

A = json.load(open("reports/submission_4_audit.json"))
g = A["gates"]
GATE_TEXT = [
    ("G1", "Final deterministic OOF within 0.001 of 0.703904", g["G1_deterministic_oof_within_0.001"],
     "%.6f (delta %+.6f)" % (A["authoritative_oof"], A["score_difference"])),
    ("G2", "OOF/test arrays pass shape, finite, range, row-sum, alignment", g["G2_array_integrity"],
     "6/6 sources ok, row-sum dev < 1e-5"),
    ("G3", "At least four folds do not materially regress vs recomputed Submission 3",
     g["G3_at_least_4_folds_no_material_regression"],
     "%d/5 folds non-regressing" % A["folds_not_regressing"]),
    ("G4", "No class F1 regresses by more than 0.005", g["G4_no_class_f1_regress_over_0.005"],
     "worst class delta %+.6f" % A["worst_class_delta"]),
    ("G5", "Test class shares within 6 pp of train shares", g["G5_test_shares_within_6pp"],
     "drift pp %s" % A["drift_pp"]),
    ("G6", "CSV passes pre-write and post-read validation", g["G6_csv_pre_and_post_validation"],
     "both passed, 11955 rows"),
    ("G7", "Fresh notebook execution succeeds", g["G7_fresh_notebook_execution"],
     "%d cell errors, fresh OOF %.6f" % (A["fresh_cell_errors"], A["fresh_oof"])),
    ("G8", "Fresh and packaged CSV byte-identical", g["G8_fresh_and_packaged_csv_identical"],
     "sha256 %s" % (A["csv_sha256"] or "n/a")[:16]),
    ("G9", "Protected files remain unchanged", g["G9_protected_files_unchanged"],
     "%d changed" % len(A["protected_files_changed"])),
]

L = []
w = L.append
w("# Submission 4 - final report (ID-DERIVED LightGBM stacker)")
w("")
w("**Authoritative OOF macro-F1: %.6f** (recorded target %.6f, difference %+.6f)"
  % (A["authoritative_oof"], A["recorded_target"], A["score_difference"]))
w("")
w("**Validation classification: `%s`**" % A["validation_classification"])
w("")
w("This is *not* claimed to be strictly outer-fold safe. The stacker consumes base-source")
w("OOF arrays, which is standard stacking: a training row of fold `j` was scored by a base")
w("model trained on folds `!= j`, which include fold `k`, so the stacker's training features")
w("carry fold-`k` label information. The per-fold class offsets are additionally chosen on the")
w("other four folds' stacker OOF (single-level, not double-nested).")
w("")
w("## Score detail")
w("")
w("| quantity | value |")
w("|---|---|")
w("| pooled OOF macro-F1 | %.6f |" % A["authoritative_oof"])
w("| per-class F1 | %s |" % ", ".join("%.6f" % v for v in A["per_class_f1"]))
w("| accuracy | %.6f |" % A["accuracy"])
w("| fold scores | %s |" % ", ".join("%.6f" % v for v in A["folds"]))
w("| fold std | %.6f |" % A["fold_std"])
w("| worst fold | %s |" % A["worst_fold"])
w("| seed scores (no correction) | %s |" % A["seed_scores"])
w("| prediction distribution | %s |" % A["prediction_distribution"])
w("| test class distribution | %s |" % A["test_class_distribution"])
w("| disagreement vs Submission 3 | %.2f%% |" % A["disagreement_vs_submission_3_pct"])
w("")
w("### Confusion matrix (rows = true, cols = predicted)")
w("")
w("```")
for row in A["confusion"]:
    w("  ".join("%6d" % v for v in row))
w("```")
w("")
w("## Organizer disclosure")
w("")
w("Submission 4 is a **leaderboard-performance stacker**, not a text method. It is heavily")
w("ID-dependent: three of its six base sources are ID-derived (`sid`, `plain_id`,")
w("`xlmr_base_idhead`) and 32 of its 60 features are ID-neighbour label/distance/window-share")
w("features, plus the raw `id`. These features are dataset-specific and unsuitable for")
w("deployment. The genuine text methodology is documented separately in")
w("`submission_reproducible_file_3.ipynb` (Submission 3, 0.682311 OOF). Public leaderboard")
w("feedback was not used for model selection, and no external labelled data was used. We")
w("acknowledge the organizer's clarification that `id` is not a text feature.")
w("")
w("## Gates")
w("")
w("| gate | criterion | result | evidence |")
w("|---|---|---|---|")
for gid, crit, ok, ev in GATE_TEXT:
    w("| %s | %s | **%s** | %s |" % (gid, crit, "PASS" if ok else "FAIL", ev))
w("")
w("**ALL GATES PASS: %s**" % ("YES" if A["all_gates_pass"] else "NO"))
w("")
w("## Deliverables")
w("")
w("| file | sha256 |")
w("|---|---|")
w("| `%s` | `%s` |" % (A["notebook_path"], A["notebook_sha256"]))
w("| `%s` | `%s` |" % (A["csv_path"], A["csv_sha256"]))
w("| `uap_submission/reproducible/artifacts_submission_4/` | 10 artifacts + manifest |")
w("")
w("## Reproduction")
w("")
w("Mode A is fully CPU reproducible. Place the notebook, `artifacts_submission_4/`,")
w("`train.csv`, `test.csv` and `sample_submission.csv` together and run")
w("`jupyter nbconvert --execute --inplace submission_reproducible_file_4.ipynb`.")
w("The notebook verifies every artifact SHA256, rebuilds the 60 features, retrains the")
w("frozen 3-seed stacker, applies the frozen class correction and the duplicate override,")
w("then writes and validates `submission_4.csv`.")
w("")
w("No Kaggle upload was performed.")

open("reports/submission_4_final_report.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote reports/submission_4_final_report.md")
print("ALL GATES PASS:", A["all_gates_pass"])

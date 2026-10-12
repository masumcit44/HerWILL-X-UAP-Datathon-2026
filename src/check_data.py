"""
src/check_data.py -- Stage 1: dataset audit for the HerWILL X UAP Datathon 2026.

Produces (all inside this workspace; the three inputs are opened read-only):
  data_checks/*.csv          every individual audit table
  reports/data_quality.json  machine-readable summary consumed by train_local.py
  reports/eda_report.md      human-readable EDA report

Run:  python src/check_data.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils as U  # noqa: E402


# --------------------------------------------------------------------------- #
# Small markdown helper (avoids the optional `tabulate` dependency)
# --------------------------------------------------------------------------- #
def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    body = [[("" if pd.isna(v) else str(v)) for v in row]
            for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def quantiles(series: pd.Series) -> dict:
    qs = [0, 1, 5, 25, 50, 75, 90, 95, 99, 100]
    vals = np.percentile(series.to_numpy(dtype=float), qs)
    out = {"mean": round(float(series.mean()), 2), "std": round(float(series.std()), 2)}
    out.update({f"p{p}": int(v) for p, v in zip(qs, vals)})
    return out


def encoding_fingerprint(path: Path) -> dict:
    raw = path.read_bytes()
    info = {"bytes": len(raw), "has_utf8_bom": raw[:3] == b"\xef\xbb\xbf"}
    for enc in ("utf-8", "utf-8-sig", "ascii"):
        try:
            raw.decode(enc)
            ok = True
        except Exception:
            ok = False
        info[f"decodes_as_{enc.replace('-', '_')}"] = ok
    info["replacement_char_count"] = raw.decode("utf-8", errors="replace").count("\ufffd")
    info["non_ascii_bytes"] = int(sum(1 for b in raw if b > 127))
    info["lone_cr_bytes"] = int(raw.count(b"\r") - raw.count(b"\r\n"))
    return info


# --------------------------------------------------------------------------- #
def main() -> int:
    U.ensure_dirs()
    res = U.detect_resources()
    U.log("=== STAGE 1: dataset audit ===")
    train, test, sample = U.load_inputs()
    checks: dict = {}

    # ---- 1. file fingerprints / encoding ------------------------------------
    fp = {p.name: {"sha1": U.sha1_file(p), **encoding_fingerprint(p)}
          for p in U.PROTECTED_INPUTS}
    checks["file_fingerprints"] = fp
    U.atomic_write_csv(pd.DataFrame([{"file": k, **v} for k, v in fp.items()]),
                       U.DATA_CHECKS_DIR / "file_fingerprints.csv")

    # ---- 2. schema ----------------------------------------------------------
    rows = []
    for name, df in (("train.csv", train), ("test.csv", test),
                     ("sample_submission.csv", sample)):
        rows.append({
            "file": name, "n_rows": len(df), "columns": ",".join(df.columns),
            "dtypes": ",".join(df.dtypes.astype(str).tolist()),
            "n_missing": int(df.isna().to_numpy().sum()),
            "id_unique": bool(df["id"].is_unique),
            "id_min": int(df["id"].min()), "id_max": int(df["id"].max()),
        })
    schema = pd.DataFrame(rows)
    U.atomic_write_csv(schema, U.DATA_CHECKS_DIR / "schema_check.csv")
    checks["schema"] = rows

    # ---- 3. ids -------------------------------------------------------------
    checks["ids"] = {
        "duplicate_train_ids": int(len(train) - train["id"].nunique()),
        "duplicate_test_ids": int(len(test) - test["id"].nunique()),
        "train_test_id_overlap": int(len(set(train["id"]) & set(test["id"]))),
        "sample_matches_test_id_order": bool((sample["id"].to_numpy() == test["id"].to_numpy()).all()),
        "sample_matches_test_id_set": bool(set(sample["id"]) == set(test["id"])),
        "train_id_is_contiguous_range": bool(int(train["id"].max() - train["id"].min()) + 1 == len(train)),
        "target_values": sorted(int(v) for v in train["y"].unique()),
        "target_is_subset_of_012": bool(set(map(int, train["y"].unique())) <= {0, 1, 2}),
        "sample_y_pred_values": sorted(int(v) for v in sample["y_pred"].unique()),
    }

    # ---- 4. blank text ------------------------------------------------------
    checks["blank_text"] = {
        "train": int((train["text"].astype("string").str.strip() == "").sum()),
        "test": int((test["text"].astype("string").str.strip() == "").sum()),
    }

    # ---- 5. class balance ---------------------------------------------------
    y = train["y"].to_numpy(dtype=int)
    cnt = Counter(y.tolist())
    bal = pd.DataFrame([
        {"class": c, "label": U.CLASS_LABELS[c], "count": int(cnt[c]),
         "share_pct": round(100.0 * cnt[c] / len(train), 2),
         "ratio_vs_minority": round(cnt[c] / min(cnt.values()), 3)}
        for c in U.CLASSES])
    U.atomic_write_csv(bal, U.DATA_CHECKS_DIR / "class_balance.csv")
    checks["class_balance"] = bal.to_dict(orient="records")
    maj = int(bal.loc[bal["count"].idxmax(), "class"])
    checks["trivial_predictor_macro_f1"] = {
        f"always_{maj}": round(U.macro_f1(y, np.full_like(y, maj)), 4),
        "always_1": round(U.macro_f1(y, np.full_like(y, 1)), 4),
    }
    return _stage2(train, test, sample, checks, res, bal, schema, y)


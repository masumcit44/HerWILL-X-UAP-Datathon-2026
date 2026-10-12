"""Phase 1: exhaustive read-only search for lost GPU/Colab artifacts."""
import os
import re
import sys
import json
import time
import hashlib
import zipfile

EXTS = {".npy", ".npz", ".joblib", ".pkl", ".json", ".csv", ".zip",
        ".ipynb", ".log", ".txt", ".pt", ".bin", ".safetensors", ".pth"}

PATTERNS = [r"^uap_results", r"^base_sources", r"^results_summary", r"^CANDIDATE_",
            r"xlmr", r"xlm-roberta", r"idhead", r"e5", r"^emb_", r"fasttext",
            r"all3", r"oof", r"test_probs", r"^fold_ids", r"fold_assignments",
            r"uap_gpu", r"multilingual", r"xlm", r"roberta", r"bengali", r"muril"]

ROOT_DIRS = [
    r"d:\Desktop\UAP",
    os.path.expanduser(r"~\Downloads"),
    os.path.expanduser(r"~\Documents"),
    os.path.expanduser(r"~\OneDrive"),
    r"d:\Desktop",
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Temp"),
    r"C:\$Recycle.Bin",
    r"D:\$Recycle.Bin",
    r"C:\$Recycle.Bin",
]
SKIP_DIR_NAMES = {"site-packages", "node_modules", ".git", "Windows",
                  "System32", "venv", ".venv", "WinSxS", "DriverStore"}


def interesting(name):
    low = name.lower()
    for pat in PATTERNS:
        if re.search(pat, low):
            return True
    return False


def sha256(path, cap=None):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if cap and n > cap:
                break
    return h.hexdigest()


def main():
    hits = []
    seen_roots = set()
    for root in ROOT_DIRS:
        if not os.path.isdir(root):
            continue
        rl = os.path.abspath(root).lower()
        if rl in seen_roots:
            continue
        seen_roots.add(rl)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in EXTS:
                    continue
                if not interesting(fn):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                hits.append({
                    "path": p,
                    "size": st.st_size,
                    "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                          time.localtime(st.st_mtime)),
                })

    print("scanned roots:", len(seen_roots))
    print("candidate hits:", len(hits))
    for h in sorted(hits, key=lambda x: x["path"]):
        print("  %-105s %12s  %s" % (h["path"][:105], f"{h['size']:,}", h["mtime"]))
    json.dump(hits, open("reports/phase1_inventory_raw.json", "w"), indent=2)
    print("\n-> reports/phase1_inventory_raw.json")


if __name__ == "__main__":
    main()

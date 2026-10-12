"""Phase 1b: targeted hunt for the lost Colab artifacts across ALL drives."""
import os
import sys
import time

TARGET_SUBSTRINGS = [
    "uap_results", "base_sources", "results_summary", "xlmr", "xlm-roberta",
    "idhead", "e5-base", "e5_base", "emb_e5-base", "emb_e5_base",
    "multilingual-e5-base", "fasttext", "csebuet", "muril", "bengalibert",
    "uap_work", "run_transformer",
]
TARGET_EXTS = {".npy", ".npz", ".zip", ".pt", ".bin", ".safetensors", ".pth",
               ".joblib", ".pkl", ".json", ".csv", ".txt", ".log", ".ipynb"}

SKIP = {"site-packages", "node_modules", ".git", "Windows", "System32",
        "WinSxS", "DriverStore", "$Recycle.Bin", "venv", ".venv", "AppData"}


def drives():
    out = []
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        root = letter + ":\\"
        if os.path.isdir(root):
            out.append(root)
    return out


def main():
    print("drives present:", drives())
    hits = []
    for root in drives():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP]
            for fn in filenames:
                low = fn.lower()
                ext = os.path.splitext(low)[1]
                if ext not in TARGET_EXTS:
                    continue
                if not any(t in low for t in TARGET_SUBSTRINGS):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                hits.append((p, st.st_size,
                             time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime(st.st_mtime))))
    print("targeted hits:", len(hits))
    for p, s, m in sorted(hits):
        print("  %-100s %12s  %s" % (p[:100], f"{s:,}", m))


if __name__ == "__main__":
    main()

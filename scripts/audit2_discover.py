"""READ-ONLY discovery of every Submission 2 artifact. Modifies nothing."""
import os
import time
import hashlib


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    print("=== Submission 2 related files ===")
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in sorted(files):
            low = fn.lower()
            if "submission_2" in low or "file_2" in low:
                p = os.path.join(root, fn)
                st = os.stat(p)
                mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
                print(p)
                print("    size={:,}  mtime={}".format(st.st_size, mt))
                print("    sha256={}".format(sha(p)))

    print()
    print("=== candidate probability arrays ===")
    for d in ("predictions/v3_candidates", "models/v3/v3_3_frozen_e5_small"):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            st = os.stat(p)
            mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
            print("  {:<58} {:>12,}  {}".format(p, st.st_size, mt))

    print()
    print("=== v3 configs / reports / manifests / logs ===")
    for d in ("reports/v3_configs", "reports"):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if "v3" in fn.lower() or "manifest" in fn.lower():
                p = os.path.join(d, fn)
                if os.path.isfile(p):
                    st = os.stat(p)
                    mt = time.strftime("%Y-%m-%d %H:%M:%S",
                                       time.localtime(st.st_mtime))
                    print("  {:<58} {:>12,}  {}".format(p, st.st_size, mt))
    if os.path.isdir("logs"):
        for fn in sorted(os.listdir("logs")):
            p = os.path.join("logs", fn)
            st = os.stat(p)
            mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
            print("  {:<58} {:>12,}  {}".format(p, st.st_size, mt))


if __name__ == "__main__":
    main()

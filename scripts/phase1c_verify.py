"""Phase 1c: verify traversal actually happened + explicitly probe recycle bins."""
import os
import time

SKIP = {"site-packages", "node_modules", ".git", "Windows", "System32",
        "WinSxS", "DriverStore", "$Recycle.Bin", "venv", ".venv"}


def main():
    total_files = 0
    total_dirs = 0
    per_drive = {}
    for letter in "CDEF":
        root = letter + ":\\"
        if not os.path.isdir(root):
            continue
        f = d = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames if x not in SKIP]
            d += 1
            f += len(filenames)
        per_drive[letter] = (d, f)
        total_dirs += d
        total_files += f
    print("traversal proof (excluding SKIP dirs):")
    for k, (d, f) in per_drive.items():
        print(f"  {k}:\\  dirs={d:,}  files={f:,}")
    print(f"  TOTAL dirs={total_dirs:,} files={total_files:,}")

    print("\nrecycle bin probe:")
    for p in (r"C:\$Recycle.Bin", r"D:\$Recycle.Bin", r"E:\$Recycle.Bin",
              r"F:\$Recycle.Bin"):
        exists = os.path.isdir(p)
        n = 0
        sample = []
        if exists:
            try:
                for r, dd, ff in os.walk(p):
                    dd[:] = [x for x in dd if x not in ("System",)]
                    n += len(ff)
                    if len(sample) < 8:
                        sample += [os.path.join(r, x) for x in ff[:8]]
            except Exception as e:
                sample = [f"<error {e}>"]
        print(f"  {p:<22} exists={exists}  recovered-items={n}")
        for s in sample[:8]:
            print("      ", s)

    print("\nany .zip anywhere on C:/D: (name-agnostic, first 40):")
    n = 0
    for letter in "CD":
        for r, dd, ff in os.walk(letter + ":\\"):
            dd[:] = [x for x in dd if x not in SKIP]
            for x in ff:
                if x.lower().endswith(".zip"):
                    n += 1
                    if n <= 40:
                        p = os.path.join(r, x)
                        try:
                            sz = f"{os.path.getsize(p):,}"
                        except OSError:
                            sz = "?"
                        print(f"   {p[:95]}  {sz}")
    print("   total zips:", n)


if __name__ == "__main__":
    main()

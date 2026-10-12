"""Copy recovered Recycle Bin items into recovered/ (READ-ONLY on the source)."""
import os
import json
import shutil
import hashlib

BIN = r"D:\$Recycle.Bin\S-1-5-21-103008578-1184166770-3893742488-1001"
ITEMS = [
    ("$RFY47UM.zip", "UAP.zip"),
    ("$R2PQX7Q.zip", "HerWILL-Safe-Social-Media-Datathon-2026.zip"),
    ("$R4K7GIG.zip", "drive-download-20260927T065936Z-1-001.zip"),
    ("$RTGL78Y.zip", "drive-download-20260926T033416Z-1-001.zip"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    os.makedirs("recovered", exist_ok=True)
    man = {}
    for src_name, out_name in ITEMS:
        src = os.path.join(BIN, src_name)
        if not os.path.exists(src):
            print(f"  [skip] {src_name} not present")
            continue
        dst = os.path.join("recovered", out_name)
        shutil.copy2(src, dst)
        h = sha256(dst)
        man[out_name] = {"recovered_from": src, "size": os.path.getsize(dst),
                         "sha256": h}
        print(f"  {out_name:<48} {os.path.getsize(dst):>13,}B  sha256={h}")
    with open("recovered/RECOVERED_HASHES.json", "w") as f:
        json.dump(man, f, indent=2)
    print("\n-> recovered/RECOVERED_HASHES.json")


if __name__ == "__main__":
    main()
